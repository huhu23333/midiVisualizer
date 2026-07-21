import cv2
import numpy as np
from typing import List, Tuple, Set, Iterable
import math

def generate_round_rect_contour(x1, y1, x2, y2, radius,
                                top_border, bottom_border, left_border=False, right_border=False):
    """
    生成圆角矩形的轮廓点（顺时针），
    根据四边是否为图像边界决定是否绘制圆角。
    返回点列表，用于 fillPoly 或 polylines。
    """
    pts = []

    def add_arc(cx, cy, r, start_ang, end_ang):
        if r <= 0:
            return []
        # print((cx, cy), (r, r), 0, start_ang, end_ang, 1)
        arc = cv2.ellipse2Poly((round(cx), round(cy)), (round(r), round(r)), 0, start_ang, end_ang, 1)
        return arc[1:] if len(arc) > 1 else []

    # 判断四个角是否应该画圆角
    tl_round = (not top_border) and (not left_border) and radius > 0
    tr_round = (not top_border) and (not right_border) and radius > 0
    br_round = (not bottom_border) and (not right_border) and radius > 0
    bl_round = (not bottom_border) and (not left_border) and radius > 0

    # 从左上角开始顺时针构建
    if tl_round:
        pts.append((x1, y1 + radius))
    else:
        pts.append((x1, y1))
    if tl_round:
        pts.extend(add_arc(x1 + radius, y1 + radius, radius, 180, 270))

    if tr_round:
        pts.append((x2 - radius, y1))
    else:
        pts.append((x2, y1))
    if tr_round:
        pts.extend(add_arc(x2 - radius, y1 + radius, radius, 270, 360))

    if br_round:
        pts.append((x2, y2 - radius))
    else:
        pts.append((x2, y2))
    if br_round:
        pts.extend(add_arc(x2 - radius, y2 - radius, radius, 0, 90))

    if bl_round:
        pts.append((x1 + radius, y2))
    else:
        pts.append((x1, y2))
    if bl_round:
        pts.extend(add_arc(x1 + radius, y2 - radius, radius, 90, 180))

    return pts


def draw_rounded_rect(img, pt1, pt2, radius, border_thickness, border_color, color_list):
    """
    在图像上绘制圆角矩形（整个图形完全位于 pt1, pt2 定义的矩形内）。

    参数：
        img              : np.ndarray，输入图像（原地修改）
        pt1, pt2         : 左上角和右下角坐标 (x, y)，其中 y 可为 None（表示直达图像边界，该侧不画圆角）
        radius           : 外轮廓的圆角半径（像素）
        border_thickness : 边框线宽（0 表示不画边框）
        border_color     : 边框颜色 (B, G, R)
        color_list       : 内部填充颜色列表，每个元素为 (B, G, R)，从左到右均分填充竖向条
    """
    h, w = img.shape[:2]
    x1, y1 = pt1
    x2, y2 = pt2

    # 处理 y = None
    top_border = (y1 is None)
    bottom_border = (y2 is None)
    if top_border:
        y1 = 0
    if bottom_border:
        y2 = h

    # 确保坐标顺序
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # ---------- 1. 外部轮廓（边框外边缘线）坐标 ----------
    out_x1 = x1
    out_y1 = y1
    out_x2 = x2
    out_y2 = y2
    if out_x1 >= out_x2 or out_y1 >= out_y2:
        return img  # 边框太粗，无法绘制

    # 限制外半径不超过矩形尺寸的一半
    max_r = min((out_x2 - out_x1) // 2, (out_y2 - out_y1) // 2)
    out_radius = max(0, min(radius, max_r))

    # 生成外部轮廓点（用于绘制边框）
    out_pts = generate_round_rect_contour(
        out_x1, out_y1, out_x2, out_y2, out_radius,
        top_border, bottom_border
    )

    # ---------- 2. 内部填充区域坐标（边框内边缘，再向内缩进 border_thickness） ----------
    in_x1 = x1 + border_thickness
    in_y1 = y1 + (border_thickness if not top_border else 0)
    in_x2 = x2 - border_thickness
    in_y2 = y2 - (border_thickness if not bottom_border else 0)
    in_radius = max(0, out_radius - border_thickness)

    # ---------- 4. 绘制边框（在外轮廓上） ----------
    if border_thickness > 0 and len(out_pts) >= 3:
        pts_np = np.array(out_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [pts_np], border_color)

    # ---------- 3. 绘制竖向均分色条（仅限内部区域） ----------
    if in_x1 < in_x2 and in_y1 < in_y2 and color_list:
        in_pts = generate_round_rect_contour(
            in_x1, in_y1, in_x2, in_y2, in_radius,
            top_border, bottom_border
        )
        mask_in = np.zeros((h, w), dtype=np.uint8)
        if len(in_pts) >= 3:
            cv2.fillPoly(mask_in, [np.array(in_pts, dtype=np.int32)], 255)

        temp = np.zeros_like(img)
        bar_width = (in_x2 - in_x1) / len(color_list)
        for i, color in enumerate(color_list):
            x_start = int(round(in_x1 + i * bar_width))
            x_end = int(round(in_x1 + (i + 1) * bar_width))
            if i == len(color_list) - 1:
                x_end = int(round(in_x2))  # 确保最后一条到达右边界
            cv2.rectangle(temp,
                          (x_start, int(round(in_y1))),
                          (x_end, int(round(in_y2))),
                          color, -1)

        cv2.copyTo(temp, mask_in, img)

    return img



def shift_and_fill_inplace(img1, img2, step_pix):
    """
    原地修改 img1 和 img2：
    - 将 img1 向下平移 step_pix，并用 img2 底部 step_pix 行填充其顶部。
    - 将 img2 向下平移 step_pix。
    注意：img1 和 img2 的宽度必须相同，step_pix 不能超过两者的最小高度。
    """
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    assert W1 == W2, "宽度必须相同"
    assert 0 < step_pix <= min(H1, H2), f"step_pix 必须在 (0, {min(H1, H2)}] 之间"

    # ----- 关键操作顺序（不可颠倒） -----
    
    # 步骤1：将 img1 下移（上半部复制到下半部）
    img1[step_pix:H1, :] = img1[0:H1 - step_pix, :]
    
    # 步骤2：将 img2 底部 step_pix 行复制到 img1 顶部（此时 img2 尚未修改）
    img1[0:step_pix, :] = img2[H2 - step_pix:H2, :]
    
    # 步骤3：将 img2 下移（同样操作）
    img2[step_pix:H2, :] = img2[0:H2 - step_pix, :]



def extract_colors_by_indices(
    colors: List[Tuple[int, int, int]], 
    indices: Iterable[int]
) -> List[Tuple[int, int, int]]:
    """
    从颜色列表中提取指定索引对应的颜色，并按索引从小到大排序。

    参数:
        colors: 颜色列表，每个元素为 (b, g, r) 三元组，各分量在 0~255 之间。
        indices: 可迭代的索引集合（如 set、list 等）。

    返回:
        按索引升序排列的颜色子列表。
        如果索引无效（超出范围），会引发 IndexError。
    """
    # 去重并排序
    sorted_indices = sorted(set(indices))
    # 按索引提取颜色
    return [colors[i] for i in sorted_indices]


def mix_color(colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if not colors:
        return (0, 0, 0)

    total_b = total_g = total_r = 0
    for b, g, r in colors:
        total_b += b
        total_g += g
        total_r += r

    max_part = max(total_b, total_g, total_r)
    if max_part == 0:
        return (0, 0, 0)
    out_v = min(max_part, 255)

    r = total_r / max_part
    g = total_g / max_part
    b = total_b / max_part

    # 排序 (值, 索引)  索引 0:B, 1:G, 2:R
    pairs = [(b, 0), (g, 1), (r, 2)]
    pairs.sort(key=lambda x: x[0])          # 升序

    min_val, min_idx = pairs[0]
    mid_val, mid_idx = pairs[1]
    max_val, max_idx = pairs[2]

    v = max_val
    min_c = min_val
    mid_c = mid_val

    S = (v - min_c) / v if v != 0 else 0
    if S == 0:
        return (out_v, out_v, out_v)

    a = 2.0 * (len(colors) - 1) # 2.34 * (math.sqrt(len(colors)) - 1)
    S_new = (1 + a) * S / (1 + a * S)

    min_new = v * (1 - S_new)
    mid_new = v * (1 - S_new) + (mid_c - min_c) * (S_new / S)

    # 直接按索引填回，不做任何相等判断
    new_vals = [0, 0, 0]
    new_vals[min_idx] = min_new
    new_vals[mid_idx] = mid_new
    new_vals[max_idx] = v

    return (
        min(int(new_vals[0] * out_v), out_v),
        min(int(new_vals[1] * out_v), out_v),
        min(int(new_vals[2] * out_v), out_v)
    )


def border_color(mixed_color):
    return tuple(map(int, (mixed_color[0] * 0.5, mixed_color[1] * 0.5, mixed_color[2] * 0.5)))


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