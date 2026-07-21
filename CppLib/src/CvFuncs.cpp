#include "CvFuncs.h"
#include <cmath>
#include <algorithm>
#include <random>

static std::mt19937 g_rng(42);

void set_seed(unsigned int seed) {
    g_rng.seed(seed);
}

std::vector<cv::Point> generate_round_rect_contour(
    int x1, int y1, int x2, int y2, int radius,
    bool top_border, bool bottom_border,
    bool left_border, bool right_border)
{
    std::vector<cv::Point> pts;

    auto add_arc = [](int cx, int cy, int r, int start_ang, int end_ang) -> std::vector<cv::Point> {
        if (r <= 0) return {};
        std::vector<cv::Point> arc;
        cv::ellipse2Poly(cv::Point(cx, cy), cv::Size(r, r), 0, start_ang, end_ang, 1, arc);
        if (arc.size() > 1)
            return std::vector<cv::Point>(arc.begin() + 1, arc.end());
        return {};
    };

    bool tl_round = (!top_border) && (!left_border) && radius > 0;
    bool tr_round = (!top_border) && (!right_border) && radius > 0;
    bool br_round = (!bottom_border) && (!right_border) && radius > 0;
    bool bl_round = (!bottom_border) && (!left_border) && radius > 0;

    if (tl_round)
        pts.emplace_back(x1, y1 + radius);
    else
        pts.emplace_back(x1, y1);

    if (tl_round) {
        auto arc = add_arc(x1 + radius, y1 + radius, radius, 180, 270);
        pts.insert(pts.end(), arc.begin(), arc.end());
    }

    if (tr_round)
        pts.emplace_back(x2 - radius, y1);
    else
        pts.emplace_back(x2, y1);

    if (tr_round) {
        auto arc = add_arc(x2 - radius, y1 + radius, radius, 270, 360);
        pts.insert(pts.end(), arc.begin(), arc.end());
    }

    if (br_round)
        pts.emplace_back(x2, y2 - radius);
    else
        pts.emplace_back(x2, y2);

    if (br_round) {
        auto arc = add_arc(x2 - radius, y2 - radius, radius, 0, 90);
        pts.insert(pts.end(), arc.begin(), arc.end());
    }

    if (bl_round)
        pts.emplace_back(x1 + radius, y2);
    else
        pts.emplace_back(x1, y2);

    if (bl_round) {
        auto arc = add_arc(x1 + radius, y2 - radius, radius, 90, 180);
        pts.insert(pts.end(), arc.begin(), arc.end());
    }

    return pts;
}

void draw_rounded_rect(
    cv::Mat& img,
    std::pair<int, int> pt1,
    std::pair<int, int> pt2,
    int radius,
    int border_thickness,
    const cv::Scalar& border_color,
    const std::vector<cv::Scalar>& color_list)
{
    int h = img.rows;
    int w = img.cols;
    int x1 = pt1.first, y1_tmp = pt1.second;
    int x2 = pt2.first, y2_tmp = pt2.second;

    bool top_border = (pt1.second == -1);
    bool bottom_border = (pt2.second == -1);
    int y1 = top_border ? 0 : y1_tmp;
    int y2 = bottom_border ? h : y2_tmp;

    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);

    int out_x1 = x1, out_y1 = y1;
    int out_x2 = x2, out_y2 = y2;
    if (out_x1 >= out_x2 || out_y1 >= out_y2)
        return;

    int max_r = std::min((out_x2 - out_x1) / 2, (out_y2 - out_y1) / 2);
    int out_radius = std::max(0, std::min(radius, max_r));

    std::vector<cv::Point> out_pts = generate_round_rect_contour(
        out_x1, out_y1, out_x2, out_y2, out_radius,
        top_border, bottom_border);

    int in_x1 = x1 + border_thickness;
    int in_y1 = y1 + (top_border ? 0 : border_thickness);
    int in_x2 = x2 - border_thickness;
    int in_y2 = y2 - (bottom_border ? 0 : border_thickness);
    int in_radius = std::max(0, out_radius - border_thickness);

    if (border_thickness > 0 && out_pts.size() >= 3) {
        std::vector<std::vector<cv::Point>> pts_v = {out_pts};
        cv::fillPoly(img, pts_v, border_color);
    }

    if (in_x1 < in_x2 && in_y1 < in_y2 && !color_list.empty()) {
        std::vector<cv::Point> in_pts = generate_round_rect_contour(
            in_x1, in_y1, in_x2, in_y2, in_radius,
            top_border, bottom_border);

        cv::Mat mask_in = cv::Mat::zeros(h, w, CV_8UC1);
        if (in_pts.size() >= 3) {
            std::vector<std::vector<cv::Point>> pts_v = {in_pts};
            cv::fillPoly(mask_in, pts_v, cv::Scalar(255));
        }

        cv::Mat temp = cv::Mat::zeros(img.size(), img.type());
        float bar_width = (float)(in_x2 - in_x1) / color_list.size();
        for (size_t i = 0; i < color_list.size(); i++) {
            int x_start = (int)std::round(in_x1 + i * bar_width);
            int x_end;
            if (i == color_list.size() - 1)
                x_end = (int)std::round((float)in_x2);
            else
                x_end = (int)std::round(in_x1 + (i + 1) * bar_width);
            cv::rectangle(temp,
                          cv::Point(x_start, (int)std::round((float)in_y1)),
                          cv::Point(x_end, (int)std::round((float)in_y2)),
                          color_list[i], -1);
        }

        cv::copyTo(temp, img, mask_in);
    }
}

void shift_and_fill_inplace(cv::Mat& img1, cv::Mat& img2, int step_pix) {
    int H1 = img1.rows, W1 = img1.cols;
    int H2 = img2.rows, W2 = img2.cols;
    CV_Assert(W1 == W2);
    CV_Assert(step_pix > 0 && step_pix <= std::min(H1, H2));

    // Python logic:
    // img1[step_pix:H1, :] = img1[0:H1-step_pix, :]   -- copy top part to bottom
    // img1[0:step_pix, :] = img2[H2-step_pix:H2, :]    -- copy img2 bottom to img1 top
    // img2[step_pix:H2, :] = img2[0:H2-step_pix, :]    -- shift img2 down

    // Re-implement correctly:
    // Step 1: remember top part of img1
    cv::Mat img1_top = img1(cv::Rect(0, 0, W1, H1 - step_pix)).clone();
    // Step 2: remember bottom part of img2
    cv::Mat img2_bottom = img2(cv::Rect(0, H2 - step_pix, W2, step_pix)).clone();
    // Step 3: remember top part of img2
    cv::Mat img2_top = img2(cv::Rect(0, 0, W2, H2 - step_pix)).clone();

    // img1[step_pix:H1, :] = img1[0:H1-step_pix, :]
    img1_top.copyTo(img1(cv::Rect(0, step_pix, W1, H1 - step_pix)));

    // img1[0:step_pix, :] = img2[H2-step_pix:H2, :]
    img2_bottom.copyTo(img1(cv::Rect(0, 0, W1, step_pix)));

    // img2[step_pix:H2, :] = img2[0:H2-step_pix, :]
    img2_top.copyTo(img2(cv::Rect(0, step_pix, W2, H2 - step_pix)));
}

std::vector<cv::Scalar> extract_colors_by_indices(
    const std::vector<cv::Scalar>& colors,
    const std::vector<int>& indices)
{
    std::set<int> sorted_idx(indices.begin(), indices.end());
    std::vector<cv::Scalar> result;
    for (int i : sorted_idx) {
        if (i >= 0 && i < (int)colors.size())
            result.push_back(colors[i]);
    }
    return result;
}

cv::Scalar mix_color(const std::vector<cv::Scalar>& colors) {
    if (colors.empty())
        return cv::Scalar(0, 0, 0);

    double total_b = 0, total_g = 0, total_r = 0;
    for (const auto& c : colors) {
        total_b += c[0];
        total_g += c[1];
        total_r += c[2];
    }

    double max_part = std::max({total_b, total_g, total_r});
    if (max_part == 0)
        return cv::Scalar(0, 0, 0);
    int out_v = (int)std::min(max_part, 255.0);

    double r = total_r / max_part;
    double g = total_g / max_part;
    double b = total_b / max_part;

    // pairs: (value, index)  index 0:B, 1:G, 2:R
    std::vector<std::pair<double, int>> pairs = {{b, 0}, {g, 1}, {r, 2}};
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    double min_val = pairs[0].first, min_idx = pairs[0].second;
    double mid_val = pairs[1].first, mid_idx = pairs[1].second;
    double max_val = pairs[2].first, max_idx = pairs[2].second;

    double v = max_val;
    double min_c = min_val;
    double mid_c = mid_val;

    double S = (v != 0) ? (v - min_c) / v : 0;
    if (S == 0)
        return cv::Scalar(out_v, out_v, out_v);

    double a = 2.0 * (colors.size() - 1);
    double S_new = (1 + a) * S / (1 + a * S);

    double min_new = v * (1 - S_new);
    double mid_new = v * (1 - S_new) + (mid_c - min_c) * (S_new / S);

    double new_vals[3] = {0, 0, 0};
    new_vals[(int)min_idx] = min_new;
    new_vals[(int)mid_idx] = mid_new;
    new_vals[(int)max_idx] = v;

    return cv::Scalar(
        std::min((int)(new_vals[0] * out_v), out_v),
        std::min((int)(new_vals[1] * out_v), out_v),
        std::min((int)(new_vals[2] * out_v), out_v)
    );
}

cv::Scalar border_color(const cv::Scalar& mixed_color) {
    return cv::Scalar(
        (int)(mixed_color[0] * 0.5),
        (int)(mixed_color[1] * 0.5),
        (int)(mixed_color[2] * 0.5)
    );
}

cv::Mat tone_map_hdr(const cv::Mat& img) {
    CV_Assert(img.type() == CV_32FC3);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    // max per pixel across channels
    cv::Mat a = cv::Mat::zeros(img.size(), CV_32FC1);
    for (int i = 0; i < 3; i++)
        a = cv::max(a, channels[i]);

    cv::Mat safe_a = a.clone();
    cv::Mat zero_mask = (a == 0);
    safe_a.setTo(1.0f, zero_mask);

    cv::Mat scale = cv::min(a, 255.0f) / safe_a;

    cv::Mat result;
    cv::Mat scale_3ch;
    cv::merge(std::vector<cv::Mat>(3, scale), scale_3ch);
    cv::multiply(img, scale_3ch, result, 1.0, CV_32FC3);

    return result;
}

void add_glow_effect(cv::Mat& image, cv::Mat& glow_illuminant,
                     int gaussian_size, float alpha, float sigma)
{
    cv::Mat blurred_glow;
    cv::GaussianBlur(glow_illuminant, blurred_glow,
                     cv::Size(gaussian_size, gaussian_size), sigma, sigma);

    cv::Mat image_float, blurred_float;
    image.convertTo(image_float, CV_32FC3);
    blurred_glow.convertTo(blurred_float, CV_32FC3);

    cv::Mat result_float = image_float + blurred_float * alpha;
    result_float = tone_map_hdr(result_float);

    // clip to [0, 255]
    cv::Mat clipped;
    result_float.copyTo(clipped);
    // clamp manually
    for (int y = 0; y < clipped.rows; y++) {
        for (int x = 0; x < clipped.cols; x++) {
            cv::Vec3f& v = clipped.at<cv::Vec3f>(y, x);
            v[0] = std::max(0.0f, std::min(255.0f, v[0]));
            v[1] = std::max(0.0f, std::min(255.0f, v[1]));
            v[2] = std::max(0.0f, std::min(255.0f, v[2]));
        }
    }

    clipped.convertTo(image, CV_8UC3);
}