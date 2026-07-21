#include "PianoRender.h"

PianoRender::PianoRender(const cv::Size& sz, int mn, const cv::Scalar& lbc)
    : screen_size(sz), max_note(mn), light_bar_color(lbc)
{
    white_key_height = (int)(sz.height * 80.0 / 720.0);
    black_key_height = (int)(sz.height * 48.0 / 720.0);
    line_width_piano = 1;
    line_color_piano = cv::Scalar(0, 0, 0);
    piano_white_note_color = cv::Scalar(240, 240, 240);
    piano_black_note_color = cv::Scalar(20, 20, 20);

    note_to_x_ratio_total = (float)sz.width / max_note;
    note_to_x_ratio_white = note_to_x_ratio_total * 12.0f / 7.0f;
    note_type_in_a_octave_list = {0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0};
    note_to_white = {0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6};
}

PianoRender::PianoRender(const cv::Size& sz, int mn)
    : PianoRender(sz, mn, cv::Scalar(200, 240, 255))
{
}

std::pair<int, int> PianoRender::get_side_x(int note) const {
    int x_left = (int)(note * note_to_x_ratio_total);
    int x_right = (int)((note + 1) * note_to_x_ratio_total);
    return {x_left, x_right};
}

void PianoRender::draw_piano(cv::Mat& image, cv::Mat& glow_illuminant,
                             const std::vector<bool>& note_state_list,
                             const std::vector<cv::Scalar>& on_color_list)
{
    // Draw white keys
    for (int note = 0; note < max_note; note++) {
        if (note_type_in_a_octave_list[note % 12] == 0) {
            int x_left = (int)(note_to_white[note % 12] * note_to_x_ratio_white
                               + (note / 12) * 12 * note_to_x_ratio_total);
            int x_right = (int)((note_to_white[note % 12] + 1) * note_to_x_ratio_white
                                + (note / 12) * 12 * note_to_x_ratio_total);

            cv::Scalar key_color = note_state_list[note] ? on_color_list[note] : piano_white_note_color;
            cv::Point pt1(x_left, screen_size.height - white_key_height - 1);
            cv::Point pt2(x_right, screen_size.height - 1);
            cv::rectangle(image, pt1, pt2, key_color, -1);
            cv::rectangle(image, pt1, pt2, line_color_piano, line_width_piano);
        }
    }

    // Draw black keys
    for (int note = 0; note < max_note; note++) {
        if (note_type_in_a_octave_list[note % 12] == 1) {
            int x_left = (int)(note * note_to_x_ratio_total);
            int x_right = (int)((note + 1) * note_to_x_ratio_total);

            cv::Scalar key_color = note_state_list[note] ? on_color_list[note] : piano_black_note_color;
            cv::Point pt1(x_left, screen_size.height - white_key_height - 1);
            cv::Point pt2(x_right, screen_size.height - white_key_height + black_key_height - 1);
            cv::rectangle(image, pt1, pt2, key_color, -1);
            cv::rectangle(image, pt1, pt2, line_color_piano, line_width_piano);
        }
    }

    // Draw glow illuminant for active notes
    for (int note = 0; note < max_note; note++) {
        if (note_state_list[note]) {
            cv::Mat mask_temp = cv::Mat::zeros(image.size(), CV_8UC3);

            if (note_type_in_a_octave_list[note % 12] == 0) {
                int x_left = (int)(note_to_white[note % 12] * note_to_x_ratio_white
                                   + (note / 12) * 12 * note_to_x_ratio_total);
                int x_right = (int)((note_to_white[note % 12] + 1) * note_to_x_ratio_white
                                    + (note / 12) * 12 * note_to_x_ratio_total);
                cv::Point pt1(x_left, screen_size.height - white_key_height - 1);
                cv::Point pt2(x_right, screen_size.height - 1);
                cv::rectangle(mask_temp, pt1, pt2, on_color_list[note], -1);

                int note_l = note - 1;
                x_left = (int)(note_l * note_to_x_ratio_total);
                x_right = (int)((note_l + 1) * note_to_x_ratio_total);
                pt1 = cv::Point(x_left, screen_size.height - white_key_height - 1);
                pt2 = cv::Point(x_right, screen_size.height - white_key_height + black_key_height - 1);
                cv::rectangle(mask_temp, pt1, pt2, cv::Scalar(0, 0, 0), -1);

                int note_r = note + 1;
                x_left = (int)(note_r * note_to_x_ratio_total);
                x_right = (int)((note_r + 1) * note_to_x_ratio_total);
                pt1 = cv::Point(x_left, screen_size.height - white_key_height - 1);
                pt2 = cv::Point(x_right, screen_size.height - white_key_height + black_key_height - 1);
                cv::rectangle(mask_temp, pt1, pt2, cv::Scalar(0, 0, 0), -1);
            } else {
                int x_left = (int)(note * note_to_x_ratio_total);
                int x_right = (int)((note + 1) * note_to_x_ratio_total);
                cv::Point pt1(x_left, screen_size.height - white_key_height - 1);
                cv::Point pt2(x_right, screen_size.height - white_key_height + black_key_height - 1);
                cv::rectangle(mask_temp, pt1, pt2, on_color_list[note], -1);
            }

            cv::max(glow_illuminant, mask_temp, glow_illuminant);
        }
    }

    // Draw light bar
    cv::Point pt1(0, screen_size.height - white_key_height - 3);
    cv::Point pt2(screen_size.width, screen_size.height - white_key_height - 1);
    cv::rectangle(image, pt1, pt2, light_bar_color, -1);

    pt1 = cv::Point(0, screen_size.height - white_key_height - 1);
    pt2 = cv::Point(screen_size.width, screen_size.height - white_key_height + 3);
    cv::rectangle(glow_illuminant, pt1, pt2, light_bar_color, -1);
}