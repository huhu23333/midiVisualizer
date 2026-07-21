import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import numpy as np
import cv2
import PianoRender, NotesRender
from CvFuncs import shift_and_fill_inplace, mix_color, extract_colors_by_indices

max_note = 128
note_shift = 0
note_in_tmp_height = 200
note_out_tmp_height = 2000
note_speed = 500.0

screen_size = (1920, 1080)
frame_dt = 1.0 / 60.0

color_map = [None,
             (80, 30, 255), (102, 176, 255), (120, 255, 120), (240, 120, 255), 
             (255, 240, 60), (160, 160, 160), (82, 255, 252), (146, 190, 214),
             (255, 30, 180), (180, 30, 255), (180, 180, 255), (255, 180, 180),
]
            #  (60, 28, 255), (154, 148, 255), (189, 153, 255), (255, 80, 147),
            #  (250, 254, 255), (255, 157, 161), (204, 244, 255), (255, 225, 169), 
            #  (255, 218, 204), (175, 253, 255), (126, 150, 255), (255, 232, 178)]



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

        self.note_distance_dt = (self.nr.keep_for_out_pix + note_out_tmp_height - self.pr.white_key_height) / note_speed
        self.piano_delay_list = [(0.0, [False for _ in range(max_note)], [{'on':[],'playing':[],'off':[]} for _ in range(max_note)])]

    def render_frames(self, state):
        notes = state["notes"]
        bpm = state["bpm"]
        tick_dt = 60 / (bpm * 48)

        shifted_notes = {i: notes[i+note_shift]
            for i in range(max_note)
        }

        nsl_for_delay = [(len(shifted_notes[i]['playing']) > 0)
            for i in range(max_note)
        ]
        self.piano_delay_list.append((self.tick_time_count + self.note_distance_dt, nsl_for_delay, shifted_notes))

        target_pix = tick_dt * note_speed + self.last_move_note_tmp_pix_residual
        step_pix = round(target_pix)
        self.last_move_note_tmp_pix_residual = target_pix - step_pix
        self.nr.update_in_temp(self.note_in_tmp, step_pix, shifted_notes, color_map)
        shift_and_fill_inplace(self.note_out_tmp, self.note_in_tmp, step_pix)

        result = []
        self.tick_time_count += tick_dt
        while self.frame_count * frame_dt <= self.tick_time_count:
            over_dt = self.tick_time_count - self.frame_count * frame_dt
            over_pix = round(note_speed * over_dt)
            frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
            frame[:screen_size[1]-over_pix] = self.note_out_tmp[self.note_out_tmp.shape[0]-(screen_size[1]-over_pix):]



            glow_illuminant = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
            while len(self.piano_delay_list) >= 2 and (self.frame_count * frame_dt >= self.piano_delay_list[1][0]):
                del self.piano_delay_list[0]

            nsl_for_pr = self.piano_delay_list[0][1]
            delayed_notes = self.piano_delay_list[0][2]
            on_color_list = [mix_color(extract_colors_by_indices(color_map, delayed_notes[i]['playing'])) if nsl_for_pr[i] else None
                for i in range(max_note)
            ]
            self.pr.draw_piano(frame, glow_illuminant, nsl_for_pr, on_color_list)

            result.append(frame)

            self.frame_count += 1

        # cv2.imshow("1", piano_frame)
        # cv2.imshow("2", glow_illuminant)
        # cv2.waitKey(1)

        cv2.imshow("2", self.note_in_tmp)

        return result
