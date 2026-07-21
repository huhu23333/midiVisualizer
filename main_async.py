import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import time
import cv2
import numpy as np
from collections import deque
from pyLib.MidiParser import MidiParser, StateProcessor
from pyLib.RenderBackend.Render_cpp import AsyncMidiPianoRender

# ====== Configurable parameters ======
# Maximum number of buffered ticks shared across all layers.
# Higher values increase pipeline parallelism at the cost of memory.
MAX_BUFFER_TICKS = 10

# Whether to show the real-time rendering window.
SHOW_WINDOW = True
# =====================================

midi_dir_path = os.path.join(base_path, "midi")

def midi_path(midi_name):
    return os.path.join(midi_dir_path, midi_name)


def shift_frames(frames):
    result = []
    tx, ty = 2, -7
    tm = np.float32([[1, 0, tx], [0, 1, ty]])
    for frame in frames:
        result.append(np.zeros_like(frame))
        cv2.warpAffine(
            src=frame,
            M=tm,
            dsize=(frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
            dst=result[-1],
        )
    return result


class TickSource:
    """
    Wraps a MIDI iterator and StateProcessor to lazily yield pre-processed ticks.
    Serves as the input source for the bottom-most layer.
    """

    def __init__(self, mp, sp):
        self._iter = mp.iter_ticks()
        self._sp = sp
        self._next_tick_id = 0
        self._exhausted = False
        self._pending = None  # pre-fetched next entry, or None

    def has_next(self):
        """Return True if another tick is available (may trigger pre-fetch)."""
        if self._exhausted:
            return False
        if self._pending is not None:
            return True
        try:
            raw_state = next(self._iter)
        except StopIteration:
            self._exhausted = True
            return False
        shifted_state, split_state = self._sp.shift_and_split_state(raw_state)
        self._pending = {
            "tick_id": self._next_tick_id,
            "shifted_state": shifted_state,
            "split_state": split_state,
        }
        self._next_tick_id += 1
        return True

    def pop(self):
        """Return the oldest pre-fetched tick, or None if exhausted."""
        if not self.has_next():
            return None
        entry = self._pending
        self._pending = None
        return entry

    @property
    def exhausted(self):
        """True when the iterator is fully consumed AND the last prefetch was popped."""
        return self._exhausted and self._pending is None


class AsyncLayer:
    """
    A single layer in the async rendering pipeline.

    Maintains:
      - idle / busy flag
      - current async job (if busy)
      - output buffer (deque) shared with the next layer
    """

    def __init__(self, layer_idx, n_layers, render, input_source,
                 track_idx_list, output_buffer):
        """
        Args:
            layer_idx: 0-based index (0 = bottom, n_layers-1 = top).
            n_layers: total number of layers.
            render: AsyncMidiPianoRender instance for this layer.
            input_source: TickSource (bottom layer) or deque (higher layers).
            track_idx_list: list of all track indices (1-based).
            output_buffer: deque for this layer's output (None for top layer).
        """
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.is_bottom = (layer_idx == 0)
        self.is_top = (layer_idx == n_layers - 1)
        self.render = render
        self.input_source = input_source
        self.track_idx_list = track_idx_list
        self.output_buffer = output_buffer

        self.idle = True
        self.current_job_id = None
        self.current_entry = None  # the input dict being processed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_input(self):
        """Check whether this layer's input source has data available."""
        if self.is_bottom:
            return self.input_source.has_next()
        else:
            return len(self.input_source) > 0

    def _pop_input(self):
        """Pop and return the oldest entry from the input source."""
        if self.is_bottom:
            return self.input_source.pop()
        else:
            return self.input_source.popleft()

    def _output_is_full(self):
        """Check whether this layer's output buffer has reached the limit."""
        if self.is_top:
            return False  # top layer writes directly, no buffer limit
        return len(self.output_buffer) >= MAX_BUFFER_TICKS

    def _get_layer_state(self, entry):
        """
        Extract the state dict appropriate for this layer from a tick entry.

        Bottom layer:  split_state[track_idx_list[-1]]
        Middle layers: split_state[track_idx_list[-(layer_idx + 1)]]
        Top layer:     shifted_state (full combined state)
        """
        if self.is_bottom:
            return entry["split_state"][self.track_idx_list[-1]]
        elif self.is_top:
            return entry["shifted_state"]
        else:
            track_pos = -(self.layer_idx + 1)
            return entry["split_state"][self.track_idx_list[track_pos]]

    # ------------------------------------------------------------------
    # Polling entry point
    # ------------------------------------------------------------------

    def poll(self):
        """
        Poll this layer once.

        Returns:
            (bool, list | None)
            bool: True if the top layer just wrote new frames (for progress printing).
            list: The rendered frames from the top layer, or None.
                  Always (False, None) for non-top layers.
        """
        # ---- Step 1: If busy, check whether the job finished ----
        if not self.idle:
            if self.render.is_ready(self.current_job_id):
                # Job completed → retrieve result
                frames = self.render.get_result(self.current_job_id)
                entry = self.current_entry
                self.current_job_id = None
                self.current_entry = None
                self.idle = True

                if self.is_top:
                    # Top layer: frames go directly to video output
                    return True, frames
                else:
                    # Middle / bottom layer: shift and push to output buffer
                    shifted = shift_frames(frames)
                    entry["frames"] = shifted
                    self.output_buffer.append(entry)
                    # Fall through to try starting new work
            else:
                # Still working — skip this layer
                return False, None

        # ---- Step 2: If idle (or just became idle), try to start new work ----
        if self.idle:
            if not self._output_is_full() and self._has_input():
                entry = self._pop_input()
                if entry is not None:
                    layer_state = self._get_layer_state(entry)
                    low_layers = None if self.is_bottom else entry.get("frames")

                    job_id = self.render.render_frames_async(layer_state, low_layers)
                    self.current_job_id = job_id
                    self.current_entry = entry
                    self.idle = False

        return False, None


# ======================================================================
# Main
# ======================================================================

def main():
    mp = MidiParser(midi_path("Asterlore.mid"), int(10 / 0.02))
    tracks_info = mp.get_tracks_info()
    print(tracks_info)
    track_idx_list = list(range(1, max(tracks_info.keys()) + 1))
    sp = StateProcessor(track_idx_list, {1: 12, 2: 12, 4: 12, 6: -12, 7: 12, 8: -12})

    # ---- Build the tick source (bottom layer's input) ----
    tick_source = TickSource(mp, sp)

    # ---- Create render instances ----
    # Track layers (bottom-to-top in pipeline order)
    # low_layers_renders[0] corresponds to track_idx_list[0] (top track)
    # low_layers_renders[-1] corresponds to track_idx_list[-1] (bottom track)
    low_layers_renders = [
        AsyncMidiPianoRender(track_layer_idx=track_idx)
        for track_idx in track_idx_list
    ]
    mpr = AsyncMidiPianoRender(render_note=False)

    n_track_layers = len(track_idx_list)
    n_layers = n_track_layers + 1  # +1 for the top mpr layer

    # ---- Create output buffers (one per adjacent pair except top) ----
    output_buffers = [deque() for _ in range(n_layers - 1)]

    # ---- Assemble layers (bottom → top) ----
    layers = []
    for i in range(n_track_layers):
        # Layer i (0 = bottom track): render = low_layers_renders[-(i+1)]
        render = low_layers_renders[-(i + 1)]
        input_src = tick_source if i == 0 else output_buffers[i - 1]
        output_buf = output_buffers[i]

        layers.append(AsyncLayer(
            layer_idx=i,
            n_layers=n_layers,
            render=render,
            input_source=input_src,
            track_idx_list=track_idx_list,
            output_buffer=output_buf,
        ))

    # Top layer (mpr)
    layers.append(AsyncLayer(
        layer_idx=n_layers - 1,
        n_layers=n_layers,
        render=mpr,
        input_source=output_buffers[-1],
        track_idx_list=track_idx_list,
        output_buffer=None,
    ))

    # ---- Video output setup ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(base_path, "output.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 60, (1920, 1080))

    t_start = time.time()
    total_frame_count = 0
    last_printed_frame_count = -1
    estimate_total_frame = int((4 * 60 + 50) * 60)

    print("开始异步渲染...")
    try:
        while True:
            # ---- Poll all layers from bottom to top ----
            for layer in layers:
                wrote_frames, frames = layer.poll()
                if wrote_frames and frames is not None:
                    for f in frames:
                        total_frame_count += 1
                        out.write(f)
                        if SHOW_WINDOW:
                            cv2.imshow("1", cv2.resize(f, (1280, 720)))
                            cv2.waitKey(1)

            # ---- After each full round, print progress if changed ----
            if total_frame_count != last_printed_frame_count:
                total_time = time.time() - t_start
                average_spf = total_time / total_frame_count if total_frame_count > 0 else 0.0
                if average_spf > 0:
                    remaining = estimate_total_frame - total_frame_count
                    print(
                        f"已渲染[{total_frame_count}]帧 平均[{average_spf:.2f}]s每帧 "
                        f"([{1 / average_spf:.3f}fps) "
                        f"估计剩余[{remaining}]帧 ([{remaining * average_spf:.2f}]s)"
                    )
                last_printed_frame_count = total_frame_count

            # ---- Check termination ----
            if tick_source.exhausted:
                all_idle = all(layer.idle for layer in layers)
                all_buffers_empty = all(len(buf) == 0 for buf in output_buffers)
                if all_idle and all_buffers_empty:
                    break

    except KeyboardInterrupt:
        print("提前终止渲染")
    finally:
        out.release()
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print(
        f"渲染完成! 共 {total_frame_count} 帧, "
        f"耗时 {total_time:.2f}s, "
        f"平均 {total_time / total_frame_count:.3f}s/帧 "
        f"({total_frame_count / total_time:.2f} fps)"
    )


if __name__ == "__main__":
    main()