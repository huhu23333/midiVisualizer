import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

import numpy as np
import cv2
from CvFuncs import draw_rounded_rect, extract_colors_by_indices, mix_color, border_color



class NotesRender:
    def __init__(self, get_side_x_func):
        self.radius = 7
        self.border_thickness = 3
        self.gap = 7
        self.shrink = 1

        self.keep_for_out_pix = 50
        self.min_update_pix = 100

        self.get_side_x_func = get_side_x_func
        self.last_update_pix = [-(self.min_update_pix+1) for _ in range(128)]
        self.last_on_border_pix = [0 for _ in range(128)]
        self.off_region_pix = max(self.radius, self.border_thickness) + self.gap

    def update_in_temp(self, in_tmp_img : np.ndarray, step_pix : int, notes : dict[int, object], color_map):
        end_draw_row = in_tmp_img.shape[0] - self.keep_for_out_pix
        start_draw_row = end_draw_row - self.min_update_pix

        for note in notes.keys():
            note_state = notes[note]
            x_left, x_right = self.get_side_x_func(note)

            self.last_update_pix[note] += step_pix
            self.last_on_border_pix[note] += step_pix

            on_set = set(note_state['on'])
            playing_set = set(note_state['playing'])
            off_set = set(note_state['off'])
            continue_set = playing_set - on_set
            last_set = continue_set | off_set

            if off_set:
                off_tmp = np.zeros((in_tmp_img.shape[0], x_right - x_left, 3), np.uint8)
                this_off_region_pix = self.off_region_pix
                this_gap = self.gap
                if self.last_on_border_pix[note] - max(self.radius, self.border_thickness) >= self.off_region_pix:
                    pass
                elif self.last_on_border_pix[note] - max(self.radius, self.border_thickness) >= self.off_region_pix - self.gap:
                    this_off_region_pix = self.last_on_border_pix[note] - max(self.radius, self.border_thickness)
                    this_gap = self.gap - (self.off_region_pix - this_off_region_pix)
                else:
                    this_off_region_pix = max(self.radius, self.border_thickness) // 2
                    this_gap = 0

                if continue_set:
                    color_list = extract_colors_by_indices(color_map, continue_set)
                    mixed_color = border_color(mix_color(color_list))
                    draw_rounded_rect(off_tmp[ : end_draw_row+this_off_region_pix-1], (self.shrink, None), (x_right-x_left-self.shrink, None), 
                                            self.radius, self.border_thickness, mixed_color, color_list)
                else:
                    pass
                    # cv2.rectangle(off_tmp, (0, 0), (x_right-x_left, end_draw_row+this_off_region_pix-1), (0, 0, 0), -1)

                color_list = extract_colors_by_indices(color_map, last_set)
                mixed_color = border_color(mix_color(color_list))
                draw_rounded_rect(off_tmp, (self.shrink, end_draw_row+this_gap), (x_right-x_left-self.shrink, None), 
                                          self.radius, self.border_thickness, mixed_color, color_list)
                
                in_tmp_img[ : end_draw_row+this_off_region_pix, x_left : x_right] = off_tmp[ : end_draw_row+this_off_region_pix]
                
                self.last_update_pix[note] = 0
            if on_set:
                color_list = extract_colors_by_indices(color_map, playing_set)
                mixed_color = border_color(mix_color(color_list))
                draw_rounded_rect(in_tmp_img, (x_left+self.shrink, None), (x_right-self.shrink, end_draw_row), 
                                          self.radius, self.border_thickness, mixed_color, color_list)
                self.last_on_border_pix[note] = 0
                self.last_update_pix[note] = 0
                
            if (self.last_update_pix[note] >= start_draw_row):
                if continue_set:
                    color_list = extract_colors_by_indices(color_map, continue_set)
                    mixed_color = border_color(mix_color(color_list))
                    draw_rounded_rect(in_tmp_img[ : end_draw_row], (x_left+self.shrink, None), (x_right-self.shrink, None), 
                                            self.radius, self.border_thickness, mixed_color, color_list)
                self.last_update_pix[note] = 0
