#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <functional>

struct ParticleParams {
    float particle_speed;
    float particle_spread;
    float particle_decay;
    int particle_size;
    int particle_amount;
};

class ParticleRender {
public:
    ParticleRender(const cv::Size& screen_size, int max_note,
                   std::function<std::pair<int,int>(int)> get_side_x_func,
                   float frame_dt, int white_key_height,
                   const std::vector<cv::Scalar>& color_map);

    void draw_particle(cv::Mat& frame, cv::Mat& glow_illuminant,
                       const std::map<int, std::map<std::string, std::vector<int>>>& notes);

    cv::Size screen_size;
    int max_note;
    std::function<std::pair<int,int>(int)> get_side_x_func;
    float frame_dt;
    int white_key_height;
    std::vector<cv::Scalar> color_map;

    std::vector<std::pair<cv::Mat, ParticleParams>> particle_masks;

private:
    static float particle_speed_val;
    static float particle_spread_val;
    static float particle_decay_val;
    static int particle_size_val;
    static int particle_amount_val;
    static int particle_masks_number_val;
};