import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import numpy as np
import cv2
import PianoRender, NotesRender, ParticleRender
from CvFuncs import shift_and_fill_inplace, mix_color, extract_colors_by_indices, add_glow_effect

ParticleRender.set_seed(42)

max_note = 128
note_shift = 0
note_in_tmp_height = 200
note_out_tmp_height = 2000
note_speed = 400.0

screen_size = (1920, 1080)
fps = 60.0
frame_dt = 1.0 / fps

color_map = [None,
             (120, 0, 255), (0, 180, 255), (255, 240, 60), (255, 30, 180), 
             (255, 255, 255), (180, 180, 180), (0, 255, 255), (180, 180, 255),
             (180, 30, 255), (0, 255, 0), (255, 120, 120), (240, 120, 255),
]
            #  (60, 28, 255), (154, 148, 255), (189, 153, 255), (255, 80, 147),
            #  (250, 254, 255), (255, 157, 161), (204, 244, 255), (255, 225, 169), 
            #  (255, 218, 204), (175, 253, 255), (126, 150, 255), (255, 232, 178)]

class MidiPianoRender:
    def __init__(self, track_layer_idx = None, render_note = True):
        self.pir = PianoRender.PianoRender(screen_size, max_note, color_map[track_layer_idx] if track_layer_idx else None)
        self.nr  = NotesRender.NotesRender(self.pir.get_side_x)
        if render_note:

            self.note_in_tmp = np.zeros((note_in_tmp_height, screen_size[0], 3), np.uint8)
            self.note_out_tmp = np.zeros((note_out_tmp_height, screen_size[0], 3), np.uint8)
            self.note_in_tmp_mask = np.zeros((note_in_tmp_height, screen_size[0]), np.uint8)
            self.note_out_tmp_mask = np.zeros((note_out_tmp_height, screen_size[0]), np.uint8)

            self.last_move_note_tmp_pix_residual = 0.0

            self.par = ParticleRender.ParticleRender(screen_size, max_note, self.pir.get_side_x, frame_dt, self.pir.white_key_height, color_map)

        self.note_distance_dt = (self.nr.keep_for_out_pix + note_out_tmp_height - self.pir.white_key_height) / note_speed

        self.render_note = render_note
        self.track_layer_idx = track_layer_idx

        self.tick_time_count = 0.0
        self.frame_count = 0

        self.piano_delay_list = [(0.0, [False for _ in range(max_note)], {i: {'on':[],'playing':[],'off':[]} for i in range(max_note)})]

    def render_frames(self, state, low_layers = None):
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

        if self.render_note:
            target_pix = tick_dt * note_speed + self.last_move_note_tmp_pix_residual
            step_pix = round(target_pix)
            self.last_move_note_tmp_pix_residual = target_pix - step_pix
            self.nr.update_in_temp(self.note_in_tmp, self.note_in_tmp_mask, step_pix, shifted_notes, color_map)
            shift_and_fill_inplace(self.note_out_tmp, self.note_in_tmp, step_pix)
            shift_and_fill_inplace(self.note_out_tmp_mask, self.note_in_tmp_mask, step_pix)

        result = []
        self.tick_time_count += tick_dt
        frame_idx = 0
        while self.frame_count * frame_dt <= self.tick_time_count:
            frame = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
            glow_illuminant = np.zeros((screen_size[1], screen_size[0], 3), np.uint8)

            if low_layers is not None:
                assert len(low_layers) > frame_idx
                assert frame.shape == low_layers[frame_idx].shape
                frame = low_layers[frame_idx]

            if self.render_note:
                over_dt = self.tick_time_count - self.frame_count * frame_dt
                over_pix = round(note_speed * over_dt)
                
                tmp_note_frame = np.zeros_like(frame)
                tmp_note_frame[:screen_size[1]-over_pix] = self.note_out_tmp[self.note_out_tmp.shape[0]-(screen_size[1]-over_pix):]
                note_mask = np.zeros(frame.shape[:2], np.uint8)
                note_mask[:screen_size[1]-over_pix] = self.note_out_tmp_mask[self.note_out_tmp.shape[0]-(screen_size[1]-over_pix):]
                cv2.copyTo(tmp_note_frame, note_mask, frame)
                cv2.copyTo(tmp_note_frame, note_mask, glow_illuminant)
                cv2.rectangle(glow_illuminant, 
                              (0, screen_size[1] - self.pir.white_key_height), 
                              (screen_size[0], screen_size[1]), 
                              (0, 0, 0), -1)

            while len(self.piano_delay_list) >= 2 and (self.frame_count * frame_dt >= self.piano_delay_list[1][0]):
                del self.piano_delay_list[0]

            nsl_for_pir = self.piano_delay_list[0][1]
            delayed_notes = self.piano_delay_list[0][2]
            on_color_list = [mix_color(extract_colors_by_indices(color_map, delayed_notes[i]['playing'])) if nsl_for_pir[i] else None
                for i in range(max_note)
            ]
            self.pir.draw_piano(frame, glow_illuminant, nsl_for_pir, on_color_list)
            if self.render_note:
                self.par.draw_particle(frame, glow_illuminant, delayed_notes)

            add_glow_effect(frame, glow_illuminant)

            result.append(frame)

            self.frame_count += 1
            frame_idx += 1

        return result
    
