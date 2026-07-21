#include "NotesRender.h"
#include "CvFuncs.h"

NotesRender::NotesRender(std::function<std::pair<int,int>(int)> get_side_x_func)
    : get_side_x_func(get_side_x_func)
{
    radius = 7;
    border_thickness = 3;
    gap = 7;
    shrink = 1;
    keep_for_out_pix = 50;
    min_update_pix = 100;
    off_region_pix = std::max(radius, border_thickness) + gap;

    last_update_pix = std::vector<int>(128, -(min_update_pix + 1));
    last_on_border_pix = std::vector<int>(128, 0);
}

void NotesRender::update_in_temp(cv::Mat& in_tmp_img, cv::Mat& in_tmp_img_mask, int step_pix,
                                 const std::map<int, std::map<std::string, std::vector<int>>>& notes,
                                 const std::vector<cv::Scalar>& color_map)
{
    int end_draw_row = in_tmp_img.rows - keep_for_out_pix;
    int start_draw_row = end_draw_row - min_update_pix;

    for (const auto& kv : notes) {
        int note = kv.first;
        const auto& note_state = kv.second;

        auto side_x = get_side_x_func(note);
        int x_left = side_x.first;
        int x_right = side_x.second;

        last_update_pix[note] += step_pix;
        last_on_border_pix[note] += step_pix;

        // Get sets from note_state
        std::set<int> on_set;
        std::set<int> playing_set;
        std::set<int> off_set;

        auto it_on = note_state.find("on");
        if (it_on != note_state.end()) {
            for (int v : it_on->second) on_set.insert(v);
        }
        auto it_playing = note_state.find("playing");
        if (it_playing != note_state.end()) {
            for (int v : it_playing->second) playing_set.insert(v);
        }
        auto it_off = note_state.find("off");
        if (it_off != note_state.end()) {
            for (int v : it_off->second) off_set.insert(v);
        }

        // continue_set = playing_set - on_set
        std::set<int> continue_set;
        for (int p : playing_set) {
            if (on_set.find(p) == on_set.end()) continue_set.insert(p);
        }

        // last_set = continue_set | off_set
        std::set<int> last_set = continue_set;
        for (int o : off_set) last_set.insert(o);

        if (!off_set.empty()) {
            cv::Mat off_tmp = cv::Mat::zeros(in_tmp_img.rows, x_right - x_left, CV_8UC3);
            cv::Mat off_tmp_mask = cv::Mat::zeros(in_tmp_img_mask.rows, x_right - x_left, CV_8UC1);

            int this_off_region_pix = off_region_pix;
            int this_gap = gap;

            if (last_on_border_pix[note] - std::max(radius, border_thickness) >= off_region_pix) {
                // pass
            } else if (last_on_border_pix[note] - std::max(radius, border_thickness) >= off_region_pix - gap) {
                this_off_region_pix = last_on_border_pix[note] - std::max(radius, border_thickness);
                this_gap = gap - (off_region_pix - this_off_region_pix);
            } else {
                this_off_region_pix = std::max(radius, border_thickness) / 2;
                this_gap = 0;
            }

            if (!continue_set.empty()) {
                std::vector<int> continue_vec(continue_set.begin(), continue_set.end());
                std::vector<cv::Scalar> color_list = extract_colors_by_indices(color_map, continue_vec);
                cv::Scalar mixed = border_color(mix_color(color_list));

                std::pair<int, int> pt1(shrink, -1);
                std::pair<int, int> pt2(x_right - x_left - shrink, -1);
                // For continue_set: draw only in [0, end_draw_row + this_off_region_pix - 1)
                // We'll use a temporary ROI
                cv::Mat off_tmp_roi = off_tmp(cv::Rect(0, 0, off_tmp.cols, end_draw_row + this_off_region_pix - 1));
                std::pair<int, int> pt1_r(shrink, -1);
                std::pair<int, int> pt2_r(x_right - x_left - shrink, end_draw_row + this_off_region_pix - 1);
                draw_rounded_rect(off_tmp_roi, pt1_r, pt2_r, radius, border_thickness, mixed, color_list);

                cv::Mat off_tmp_mask_roi = off_tmp_mask(cv::Rect(0, 0, off_tmp_mask.cols, end_draw_row + this_off_region_pix - 1));
                std::vector<cv::Scalar> mask_fill = {cv::Scalar(255, 255, 255)};
                draw_rounded_rect(off_tmp_mask_roi, pt1_r, pt2_r, radius, border_thickness, cv::Scalar(255, 255, 255), mask_fill);
            }

            // Draw last_set (off+continue) at the bottom
            {
                std::vector<int> last_vec(last_set.begin(), last_set.end());
                std::vector<cv::Scalar> color_list = extract_colors_by_indices(color_map, last_vec);
                cv::Scalar mixed = border_color(mix_color(color_list));

                std::pair<int, int> pt1_b(shrink, end_draw_row + this_gap);
                std::pair<int, int> pt2_b(x_right - x_left - shrink, -1);
                draw_rounded_rect(off_tmp, pt1_b, pt2_b, radius, border_thickness, mixed, color_list);

                std::vector<cv::Scalar> mask_fill = {cv::Scalar(255, 255, 255)};
                draw_rounded_rect(off_tmp_mask, pt1_b, pt2_b, radius, border_thickness, cv::Scalar(255, 255, 255), mask_fill);
            }

            // Copy off_tmp to in_tmp
            cv::Mat in_tmp_roi = in_tmp_img(cv::Rect(x_left, 0, x_right - x_left, end_draw_row + this_off_region_pix));
            off_tmp(cv::Rect(0, 0, x_right - x_left, end_draw_row + this_off_region_pix)).copyTo(in_tmp_roi);

            cv::Mat in_tmp_mask_roi = in_tmp_img_mask(cv::Rect(x_left, 0, x_right - x_left, end_draw_row + this_off_region_pix));
            off_tmp_mask(cv::Rect(0, 0, x_right - x_left, end_draw_row + this_off_region_pix)).copyTo(in_tmp_mask_roi);

            last_update_pix[note] = 0;
        }

        if (!on_set.empty()) {
            std::vector<int> playing_vec(playing_set.begin(), playing_set.end());
            std::vector<cv::Scalar> color_list = extract_colors_by_indices(color_map, playing_vec);
            cv::Scalar mixed = border_color(mix_color(color_list));

            std::pair<int, int> pt1(x_left + shrink, -1);
            std::pair<int, int> pt2(x_right - shrink, end_draw_row);
            draw_rounded_rect(in_tmp_img, pt1, pt2, radius, border_thickness, mixed, color_list);

            std::vector<cv::Scalar> mask_fill = {cv::Scalar(255, 255, 255)};
            draw_rounded_rect(in_tmp_img_mask, pt1, pt2, radius, border_thickness, cv::Scalar(255, 255, 255), mask_fill);

            last_on_border_pix[note] = 0;
            last_update_pix[note] = 0;
        }

        if (last_update_pix[note] >= start_draw_row) {
            if (!continue_set.empty()) {
                std::vector<int> continue_vec(continue_set.begin(), continue_set.end());
                std::vector<cv::Scalar> color_list = extract_colors_by_indices(color_map, continue_vec);
                cv::Scalar mixed = border_color(mix_color(color_list));

                cv::Mat in_tmp_roi = in_tmp_img(cv::Rect(0, 0, in_tmp_img.cols, end_draw_row));
                std::pair<int, int> pt1(x_left + shrink, -1);
                std::pair<int, int> pt2(x_right - shrink, -1);
                draw_rounded_rect(in_tmp_roi, pt1, pt2, radius, border_thickness, mixed, color_list);

                cv::Mat in_tmp_mask_roi = in_tmp_img_mask(cv::Rect(0, 0, in_tmp_img_mask.cols, end_draw_row));
                std::vector<cv::Scalar> mask_fill = {cv::Scalar(255, 255, 255)};
                draw_rounded_rect(in_tmp_mask_roi, pt1, pt2, radius, border_thickness, cv::Scalar(255, 255, 255), mask_fill);
            }
            last_update_pix[note] = 0;
        }
    }
}