import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import time
import cv2
from pyLib.MidiParser import MidiParser
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

def main():
    mp = MidiParser(midi_path("Asterlore.mid"))
    mpr = MidiPianoRender()
    for state in mp.iter_ticks():
        frames = mpr.render_frames(state)
        for f in frames:
            cv2.imshow("1", cv2.resize(f, (1280, 720)))
            cv2.waitKey(1)
            time.sleep(0.02)



if __name__ == "__main__":
    main()
