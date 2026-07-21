#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

class PianoRender {
public:
    PianoRender(const cv::Size& screen_size, int max_note, const cv::Scalar& light_bar_color);
    PianoRender(const cv::Size& screen_size, int max_note);

    std::pair<int, int> get_side_x(int note) const;
    void draw_piano(cv::Mat& image, cv::Mat& glow_illuminant,
                    const std::vector<bool>& note_state_list,
                    const std::vector<cv::Scalar>& on_color_list);

    int white_key_height;
    int black_key_height;

private:
    cv::Size screen_size;
    int max_note;
    int line_width_piano;
    cv::Scalar line_color_piano;
    cv::Scalar piano_white_note_color;
    cv::Scalar piano_black_note_color;
    cv::Scalar light_bar_color;

    float note_to_x_ratio_total;
    float note_to_x_ratio_white;
    std::vector<int> note_type_in_a_octave_list;
    std::vector<int> note_to_white;
};