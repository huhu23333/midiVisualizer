import cv2
import numpy as np

class PianoRender:
    def __init__(self, screen_size, max_note, light_bar_color = None):
        self.screen_size = screen_size
        self.max_note = max_note

        self.white_key_height = int(screen_size[1] * 80 / 720)
        self.black_key_height = int(screen_size[1] * 48 / 720)
        self.line_width_piano = 1
        self.line_color_piano = (0,0,0)
        self.piano_white_note_color = (240, 240, 240)
        self.piano_black_note_color = (20,20,20)
        self.light_bar_color = light_bar_color if light_bar_color else (200, 240, 255)


        self.note_to_x_ratio_total = self.screen_size[0]/self.max_note
        self.note_to_x_ratio_white = self.note_to_x_ratio_total*12/7
        self.note_type_in_a_octave_list = [0,1,0,1,0,0,1,0,1,0,1,0]
        self.note_to_white = [0,0,1,1,2,3,3,4,4,5,5,6]

    def get_side_x(self, note):
        x_left = int(note * self.note_to_x_ratio_total)
        x_right = int((note + 1) * self.note_to_x_ratio_total)
        return x_left, x_right

    def draw_piano(self, image, glow_illuminant, note_state_list, on_color_list):
        for note in range(self.max_note):
            if self.note_type_in_a_octave_list[note%12] == 0:
                x_left = int(self.note_to_white[note%12] * self.note_to_x_ratio_white + (note//12)*12*self.note_to_x_ratio_total)
                x_right = int((self.note_to_white[note%12] + 1) * self.note_to_x_ratio_white + (note//12)*12*self.note_to_x_ratio_total)
                key_color = on_color_list[note] if note_state_list[note] else self.piano_white_note_color
                pt1 = (x_left, self.screen_size[1]-self.white_key_height-1)
                pt2 = (x_right, self.screen_size[1]-1)
                cv2.rectangle(image, pt1=pt1, pt2=pt2, color=key_color, thickness=-1)
                cv2.rectangle(image, pt1=pt1, pt2=pt2, color=self.line_color_piano, thickness=self.line_width_piano)
        for note in range(self.max_note):
            if self.note_type_in_a_octave_list[note%12] == 1:
                x_left = int(note * self.note_to_x_ratio_total)
                x_right = int((note + 1) * self.note_to_x_ratio_total)
                key_color = on_color_list[note] if note_state_list[note] else self.piano_black_note_color
                pt1 = (x_left, self.screen_size[1]-self.white_key_height-1)
                pt2 = (x_right, self.screen_size[1]-self.white_key_height+self.black_key_height-1)
                cv2.rectangle(image, pt1=pt1, pt2=pt2, color=key_color, thickness=-1)
                cv2.rectangle(image, pt1=pt1, pt2=pt2, color=self.line_color_piano, thickness=self.line_width_piano)
        for note in range(self.max_note):
            if note_state_list[note]:
                mask_temp = np.zeros(image.shape, dtype=np.uint8)
                if self.note_type_in_a_octave_list[note%12] == 0:
                    x_left = int(self.note_to_white[note%12] * self.note_to_x_ratio_white + (note//12)*12*self.note_to_x_ratio_total)
                    x_right = int((self.note_to_white[note%12] + 1) * self.note_to_x_ratio_white + (note//12)*12*self.note_to_x_ratio_total)
                    pt1 = (x_left, self.screen_size[1]-self.white_key_height-1)
                    pt2 = (x_right, self.screen_size[1]-1)
                    cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=on_color_list[note], thickness=-1)
                    note -= 1
                    x_left = int(note * self.note_to_x_ratio_total)
                    x_right = int((note + 1) * self.note_to_x_ratio_total)
                    pt1 = (x_left, self.screen_size[1]-self.white_key_height-1)
                    pt2 = (x_right, self.screen_size[1]-self.white_key_height+self.black_key_height-1)
                    cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=(0, 0, 0), thickness=-1)
                    note += 2
                    x_left = int(note * self.note_to_x_ratio_total)
                    x_right = int((note + 1) * self.note_to_x_ratio_total)
                    pt1 = (x_left, self.screen_size[1]-self.white_key_height-1)
                    pt2 = (x_right, self.screen_size[1]-self.white_key_height+self.black_key_height-1)
                    cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=(0, 0, 0), thickness=-1)
                    note -= 1
                else:
                    x_left = int(note * self.note_to_x_ratio_total)
                    x_right = int((note + 1) * self.note_to_x_ratio_total)
                    pt1 = (x_left, self.screen_size[1]-self.white_key_height-1)
                    pt2 = (x_right, self.screen_size[1]-self.white_key_height+self.black_key_height-1)
                    cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=on_color_list[note], thickness=-1)
                np.maximum(glow_illuminant, mask_temp, out=glow_illuminant)
        pt1 = (0, self.screen_size[1]-self.white_key_height-3)
        pt2 = (self.screen_size[0], self.screen_size[1]-self.white_key_height-1)
        cv2.rectangle(image, pt1=pt1, pt2=pt2, color=self.light_bar_color, thickness=-1)
        pt1 = (0, self.screen_size[1]-self.white_key_height-1)
        pt2 = (self.screen_size[0], self.screen_size[1]-self.white_key_height+3)
        cv2.rectangle(glow_illuminant, pt1=pt1, pt2=pt2, color=self.light_bar_color, thickness=-1)