import numpy as np
import cv2
from CvFuncs import extract_colors_by_indices
import random

def set_seed(seed):
    random.seed(seed)

particle_speed = 12
particle_spread = 0.4
particle_decay = 0.95
particle_size = 2
particle_amount = 7
particle_masks_number = 3

class ParticleRender:
    def __init__(self, screen_size, max_note, get_side_x_func, frame_dt, white_key_height, color_map):
        self.screen_size = screen_size
        self.max_note = max_note
        self.get_side_x_func = get_side_x_func
        self.frame_dt = frame_dt
        self.white_key_height = white_key_height
        self.color_map = color_map

        self.particle_masks = [[np.zeros((screen_size[1], screen_size[0], 3), np.float32),{
            "particle_speed": particle_speed,
            "particle_spread": particle_spread * float(i),
            "particle_decay": particle_decay,
            "particle_size": particle_size,
            "particle_amount": particle_amount
        }] for i in list(np.linspace(-1, 1, particle_masks_number))]
        
    def draw_particle(self, frame, glow_illuminant, notes):
        for note in notes.keys():
            note_state = notes[note]
            playing_track_idx_list = note_state['playing']
            if playing_track_idx_list:
                x_left, x_right = self.get_side_x_func(note)
                color_list = extract_colors_by_indices(self.color_map, playing_track_idx_list)
                random_range_x = [x_left, x_right]
                for particle_mask_index in range(len(self.particle_masks)):
                    particle_mask_params = self.particle_masks[particle_mask_index][1]
                    random_range_y = [self.screen_size[1]-self.white_key_height-particle_mask_params["particle_speed"], self.screen_size[1]-self.white_key_height]
                    for _ in range(particle_mask_params["particle_amount"]):
                        pt = (random.randint(*random_range_x), random.randint(*random_range_y))
                        rand_color = random.choice(color_list)
                        cv2.circle(self.particle_masks[particle_mask_index][0], pt, particle_mask_params["particle_size"], rand_color, -1)

        for particle_mask_index in range(len(self.particle_masks)):
            uint8_mask = self.particle_masks[particle_mask_index][0].astype(np.uint8)
            np.maximum(frame, uint8_mask, out=frame)
            np.maximum(glow_illuminant, uint8_mask, out=glow_illuminant)

            particle_mask_params = self.particle_masks[particle_mask_index][1]
            translation_matrix = np.float32([[1, 0, particle_mask_params["particle_spread"]], [0, 1, -particle_mask_params["particle_speed"]]])
            self.particle_masks[particle_mask_index][0] = cv2.warpAffine(
                src=self.particle_masks[particle_mask_index][0], 
                M=translation_matrix,
                dsize=(frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            ) * particle_mask_params["particle_decay"]