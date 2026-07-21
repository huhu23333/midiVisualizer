import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import time
import cv2
import numpy as np
from pyLib.MidiParser import MidiParser, StateProcessor
from pyLib.RenderBackend.Render import MidiPianoRender

midi_dir_path = os.path.join(base_path, "midi")
def midi_path(midi_name):
    return os.path.join(midi_dir_path, midi_name)

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

def main():
    mp = MidiParser(midi_path("Asterlore.mid"), int(10 / 0.02))
    tracks_info = mp.get_tracks_info()
    print(tracks_info)
    track_idx_list = list(range(1, max(tracks_info.keys())+1 ))
    sp = StateProcessor(track_idx_list, {1:12, 2:12, 4:12, 6:-12, 7:12, 8:-12})

    mpr = MidiPianoRender(render_note = False)
    low_layers_mpr = [MidiPianoRender(track_layer_idx = track_idx) for track_idx in track_idx_list]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式常用编解码器
    output_path = os.path.join(base_path, "output.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 60, (1920, 1080))
    show_window = True

    t_start = time.time()
    total_frame_count = 0
    estimate_total_frame = int((4*60+50)*60)
    try:
        for state in mp.iter_ticks():
            shifted_state, split_state = sp.shift_and_split_state(state)
            low_layers = low_layers_mpr[-1].render_frames(split_state[track_idx_list[-1]])
            low_layers = shift_frames(low_layers)
            for track_idx in reversed(range(len(track_idx_list)-1)):
                low_layers = low_layers_mpr[track_idx].render_frames(split_state[track_idx_list[track_idx]], low_layers)
                low_layers = shift_frames(low_layers)
            frames = mpr.render_frames(shifted_state, low_layers)
            for f in frames:
                total_frame_count += 1
                total_time = time.time() - t_start
                out.write(f)
                if show_window:
                    cv2.imshow("1", cv2.resize(f, (1280, 720)))
                    cv2.waitKey(1)
            if total_time > 0 and total_frame_count > 0:
                average_spf = total_time/total_frame_count
                print(f"已渲染[{total_frame_count}]帧 平均[{average_spf:.2f}]s每帧 ([{1/average_spf:.3f}fps) "
                      f"估计剩余[{estimate_total_frame-total_frame_count}]帧 ([{(estimate_total_frame-total_frame_count)*average_spf:.2f}]s)")
        
        print("渲染完成")
    except KeyboardInterrupt:
        print("提前终止渲染")
    out.release()



if __name__ == "__main__":
    main()
