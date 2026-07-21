import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import numpy as np
import cv2
import PianoRender, NotesRender
from CvFuncs import shift_and_fill_inplace

max_note = 108
note_shift = 12
note_in_tmp_height = 200
note_out_tmp_height = 2000
note_speed = 500.0

screen_size = (1920, 1080)
frame_dt = 1.0 / 60.0

color_map = [None,
             (120, 100, 255), (255, 100, 120), (120, 255, 100),
             (120, 255, 255), (255, 120, 255), (255, 255, 120), 
             (120, 100, 255), (255, 100, 120), (120, 255, 100),
             (120, 255, 255), (255, 120, 255), (255, 255, 120), ]



class MidiPianoRender:
    def __init__(self):

        self.pr = PianoRender.PianoRender(screen_size, max_note)
        self.nr  = NotesRender.NotesRender(self.pr.get_side_x)

        self.note_tmp_frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
        self.tick_time_count = 0.0
        self.frame_count = 0

        self.note_in_tmp = np.zeros((note_in_tmp_height, screen_size[0], 3), np.uint8)
        self.note_out_tmp = np.zeros((note_out_tmp_height, screen_size[0], 3), np.uint8)

        self.last_move_note_tmp_pix_residual = 0.0

    def render_frames(self, state):
        piano_frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
        glow_illuminant = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
        notes = state["notes"]
        bpm = state["bpm"]
        tick_dt = 60 / (bpm * 48)

        shifted_notes = {i: notes[i+note_shift]
            for i in range(max_note)
        }

        nsl_for_pr = [(len(shifted_notes[i]['playing']) > 0)
            for i in range(max_note)
        ]
        on_color_list = [(120, 100, 255) if nsl_for_pr[i] else None
            for i in range(max_note)
        ]
        self.pr.draw_piano(piano_frame, glow_illuminant, nsl_for_pr, on_color_list)

        target_pix = tick_dt * note_speed + self.last_move_note_tmp_pix_residual
        step_pix = round(target_pix)
        self.last_move_note_tmp_pix_residual = target_pix - step_pix
        self.nr.update_in_temp(self.note_in_tmp, step_pix, notes, color_map)
        shift_and_fill_inplace(self.note_out_tmp, self.note_in_tmp, step_pix)

        result = []
        self.tick_time_count += tick_dt
        while self.frame_count * frame_dt <= self.tick_time_count:
            over_dt = self.tick_time_count - self.frame_count * frame_dt
            over_pix = round(note_speed * over_dt)
            note_frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
            note_frame[:screen_size[1]-over_pix] = self.note_out_tmp[self.note_out_tmp.shape[0]-(screen_size[1]-over_pix):]

            result.append(note_frame)

            self.frame_count += 1

        # cv2.imshow("1", piano_frame)
        # cv2.imshow("2", glow_illuminant)
        # cv2.waitKey(1)

        cv2.imshow("2", self.note_in_tmp)

        return result
