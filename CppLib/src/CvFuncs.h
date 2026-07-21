#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <set>
#include <random>

// Generate rounded rectangle contour points
std::vector<cv::Point> generate_round_rect_contour(
    int x1, int y1, int x2, int y2, int radius,
    bool top_border, bool bottom_border,
    bool left_border = false, bool right_border = false);

// Draw rounded rectangle on image
void draw_rounded_rect(
    cv::Mat& img,
    std::pair<int, int> pt1,
    std::pair<int, int> pt2,
    int radius,
    int border_thickness,
    const cv::Scalar& border_color,
    const std::vector<cv::Scalar>& color_list);

// Shift and fill in-place (like numpy slice operations)
void shift_and_fill_inplace(cv::Mat& img1, cv::Mat& img2, int step_pix);

// Extract colors by indices from color list
std::vector<cv::Scalar> extract_colors_by_indices(
    const std::vector<cv::Scalar>& colors,
    const std::vector<int>& indices);

// Mix multiple colors (HSV-based saturation boost)
cv::Scalar mix_color(const std::vector<cv::Scalar>& colors);

// Compute border color (0.5x of mixed color)
cv::Scalar border_color(const cv::Scalar& mixed_color);

// Simple HDR tone mapping
cv::Mat tone_map_hdr(const cv::Mat& img);

// Add glow effect
void add_glow_effect(cv::Mat& image, cv::Mat& glow_illuminant,
                     int gaussian_size = 63, float alpha = 1.5f, float sigma = 0.0f);

// Set global random seed
void set_seed(unsigned int seed);