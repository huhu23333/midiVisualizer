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
             (120, 0, 255), (0, 180, 255), (255, 240, 60), (255, 30, 180), 
             (255, 255, 255), (180, 180, 180), (0, 255, 255), (180, 180, 255),
             (180, 30, 255), (0, 255, 0), (255, 120, 120), (240, 120, 255),
]
            #  (60, 28, 255), (154, 148, 255), (189, 153, 255), (255, 80, 147),
            #  (250, 254, 255), (255, 157, 161), (204, 244, 255), (255, 225, 169), 
            #  (255, 218, 204), (175, 253, 255), (126, 150, 255), (255, 232, 178)]

def tone_map_hdr(img: np.ndarray) -> np.ndarray:
    """
    对 HDR 浮点图像进行简单色调映射（色度归一化 + 亮度截断）
    
    Args:
        img: np.float32 类型，形状为 (H, W, 3)，BGR 或 RGB 顺序均可，值域任意非负
    
    Returns:
        np.float32 类型，形状相同，处理后最大像素值 ≤ 255
    """
    # 1. 计算每个像素三个通道的最大值 a，保持维度以便广播 (H, W, 1)
    a = np.max(img, axis=2, keepdims=True)
    
    # 2. 处理除零：将 a=0 的位置临时置为 1，方便计算缩放因子
    #    因为当 a=0 时，img 必为 0，最终结果也必为 0，所以这里置 1 不影响最终乘法结果
    safe_a = np.where(a == 0, 1.0, a)
    
    # 3. 计算缩放因子 scale = min(a, 255) / a
    #    - 当 a <= 255 时，scale = 1.0，图片保持原样
    #    - 当 a > 255 时，scale = 255.0 / a，将该像素的最大通道压到 255，其余通道等比缩放
    scale = np.minimum(a, 255.0) / safe_a
    
    # 4. 原图乘以缩放因子
    result = img * scale
    
    return result.astype(np.float32)

def add_glow_effect(image, glow_illuminant, gaussian_size=63, alpha=1.5, sigma=0):
    blurred_glow = cv2.GaussianBlur(glow_illuminant, (gaussian_size, gaussian_size), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
    image_float = image.astype(np.float32) + blurred_glow * alpha
    image_float = tone_map_hdr(image_float)
    np.clip(image_float, 0, 255, out=image_float)
    image[:] = image_float.astype(np.uint8)


class MidiPianoRender:
    def __init__(self, track_layer_idx = None, render_note = True):
        self.pr = PianoRender.PianoRender(screen_size, max_note, color_map[track_layer_idx] if track_layer_idx else None)
        self.nr  = NotesRender.NotesRender(self.pr.get_side_x)
        if render_note:

            self.note_in_tmp = np.zeros((note_in_tmp_height, screen_size[0], 3), np.uint8)
            self.note_out_tmp = np.zeros((note_out_tmp_height, screen_size[0], 3), np.uint8)
            self.note_in_tmp_mask = np.zeros((note_in_tmp_height, screen_size[0]), np.uint8)
            self.note_out_tmp_mask = np.zeros((note_out_tmp_height, screen_size[0]), np.uint8)

            self.last_move_note_tmp_pix_residual = 0.0

        self.note_distance_dt = (self.nr.keep_for_out_pix + note_out_tmp_height - self.pr.white_key_height) / note_speed

        self.render_note = render_note
        self.track_layer_idx = track_layer_idx

        self.tick_time_count = 0.0
        self.frame_count = 0

        self.piano_delay_list = [(0.0, [False for _ in range(max_note)], [{'on':[],'playing':[],'off':[]} for _ in range(max_note)])]

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

            while len(self.piano_delay_list) >= 2 and (self.frame_count * frame_dt >= self.piano_delay_list[1][0]):
                del self.piano_delay_list[0]

            nsl_for_pr = self.piano_delay_list[0][1]
            delayed_notes = self.piano_delay_list[0][2]
            on_color_list = [mix_color(extract_colors_by_indices(color_map, delayed_notes[i]['playing'])) if nsl_for_pr[i] else None
                for i in range(max_note)
            ]
            self.pr.draw_piano(frame, glow_illuminant, nsl_for_pr, on_color_list)

            add_glow_effect(frame, glow_illuminant)

            result.append(frame)

            self.frame_count += 1
            frame_idx += 1

        return result
