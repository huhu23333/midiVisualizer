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

midi_dir_path = os.path.join(base_path, "midi")
def midi_path(midi_name):
    return os.path.join(midi_dir_path, midi_name)


# ====== Configurable parameters ======
# Number of renders each layer can buffer ahead.
# Higher values allow more pipeline parallelism but use more memory.
# Recommended range: 1-4.
CACHE_DEPTH = 2
# =====================================


def test():
    mp = MidiParser(midi_path("Asterlore.mid"))
    print(mp.get_tracks_info())
    t = time.time()
    for state in mp.iter_ticks():
        notes = state["notes"]
        to_print = "".join(["■" if len(notes[note]["playing"]) else "□" for note in range(128)]) + f" bpm{state['bpm']:.2f} dt{60 / (state['bpm'] * 48):.5f}"
        print(to_print)
        slp_t = time.time() - t + 60 / (state['bpm'] * 48)
        if slp_t > 0:
            time.sleep(slp_t)
        t = time.time()


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
            dst=result[-1]
        )
    return result


class PipelineStage:
    """
    Wraps an AsyncMidiPianoRender instance to manage pending jobs per tick.
    """
    def __init__(self, render):
        self.render = render
        self.pending = {}  # tick_id -> job_id

    def submit(self, tick_id, state, low_layers=None):
        """Submit an async render job. Returns immediately."""
        job_id = self.render.render_frames_async(state, low_layers)
        self.pending[tick_id] = job_id
        return job_id

    def is_ready(self, tick_id):
        """Check if the render for the given tick is done."""
        if tick_id not in self.pending:
            return False
        return self.render.is_ready(self.pending[tick_id])

    def get_result(self, tick_id):
        """Block until the render for the given tick is done, then return frames."""
        job_id = self.pending.pop(tick_id)
        return self.render.get_result(job_id)


def main():
    mp = MidiParser(midi_path("Asterlore.mid"), int(10 / 0.02))
    tracks_info = mp.get_tracks_info()
    print(tracks_info)
    track_idx_list = list(range(1, max(tracks_info.keys())+1))
    sp = StateProcessor(track_idx_list, {1:12, 2:12, 4:12, 6:-12, 7:12, 8:-12})

    # Create async render instances.
    # Each has its own worker thread in C++, so they can render concurrently.
    mpr = AsyncMidiPianoRender(render_note=False)
    # low_layers_mpr from bottom to top (index -1 is bottom layer)
    low_layers_mpr = [AsyncMidiPianoRender(track_layer_idx=track_idx) for track_idx in track_idx_list]
    
    # Build pipeline stages: bottom layer first, top layer last
    # layer order: low_layers_mpr[-1] (bottom) -> low_layers_mpr[-2] -> ... -> low_layers_mpr[0] -> mpr (top)
    stages = []
    for i in range(len(low_layers_mpr)-1, -1, -1):
        stages.append(PipelineStage(low_layers_mpr[i]))
    stages.append(PipelineStage(mpr))  # top stage
    n_layers = len(stages)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(base_path, "output.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 60, (1920, 1080))
    show_window = True

    t_start = time.time()
    total_frame_count = 0
    estimate_total_frame = int((4*60+50)*60)
    
    # Pre-fetch states and process them into shifted/split states
    # We buffer CACHE_DEPTH ticks for pipeline parallelism
    state_iter = mp.iter_ticks()
    
    # Buffered data: each entry is (tick_id, shifted_state, split_state)
    buffered = deque()
    
    tick_id = 0
    
    def fill_buffer():
        """Refill the buffer with up to CACHE_DEPTH pre-processed states."""
        nonlocal tick_id
        while len(buffered) < CACHE_DEPTH:
            try:
                raw_state = next(state_iter)
            except StopIteration:
                return False
            shifted_state, split_state = sp.shift_and_split_state(raw_state)
            buffered.append((tick_id, shifted_state, split_state))
            tick_id += 1
        return True
    
    # Initial fill
    has_more = fill_buffer()
    
    try:
        while buffered:
            # ====== PIPELINE ROUND: process buffered ticks through all layers ======
            # For each buffered tick, submit to the bottom layer (layer 0)
            # Then for each subsequent layer, collect the previous layer's result
            # and submit this layer.
            #
            # The key insight: by submitting all buffered ticks to layer 0 first,
            # layer 0's worker thread can process them all while the main thread
            # works on collecting results and submitting higher layers.
            
            # Step 1: Submit bottom layer for all buffered ticks
            for (tid, shifted_state, split_state) in buffered:
                bottom_state = split_state[track_idx_list[-1]]
                stages[0].submit(tid, bottom_state, None)
            
            # Step 2: For each higher layer, for each buffered tick,
            # collect the previous layer's result and submit this layer
            for layer_idx in range(1, n_layers):
                for (tid, shifted_state, split_state) in buffered:
                    # Get the previous layer's result for this tick
                    prev_frames = stages[layer_idx - 1].get_result(tid)
                    shifted = shift_frames(prev_frames)
                    
                    # Determine the state for this layer
                    if layer_idx < n_layers - 1:
                        # Middle layer: use the corresponding track's split state
                        # stages[layer_idx] corresponds to low_layers_mpr[-(layer_idx+1)]
                        track_idx = track_idx_list[-(layer_idx + 1)]
                        layer_state = split_state[track_idx]
                    else:
                        # Top layer: use shifted_state
                        layer_state = shifted_state
                    
                    # Submit this layer
                    stages[layer_idx].submit(tid, layer_state, shifted)
            
            # Step 3: Collect top layer results and write frames
            # Results must be written in tick_id order
            sorted_ticks = sorted([tid for tid, _, _ in buffered])
            for tid in sorted_ticks:
                frames = stages[-1].get_result(tid)
                for f in frames:
                    total_frame_count += 1
                    total_time = time.time() - t_start
                    out.write(f)
                    if show_window:
                        cv2.imshow("1", cv2.resize(f, (1280, 720)))
                        cv2.waitKey(1)
                average_spf = total_time / total_frame_count if total_frame_count > 0 else 0
                if average_spf > 0:
                    print(f"已渲染[{total_frame_count}]帧 平均[{average_spf:.2f}]s每帧 ([{1/average_spf:.3f}fps) "
                          f"估计剩余[{estimate_total_frame-total_frame_count}]帧 "
                          f"([{(estimate_total_frame-total_frame_count)*average_spf:.2f}]s)")
            
            # Clear the processed buffer
            buffered.clear()
            
            # Refill for next round
            has_more = fill_buffer()
            if not has_more:
                break
        
        print("渲染完成")
    except KeyboardInterrupt:
        print("提前终止渲染")
    finally:
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()