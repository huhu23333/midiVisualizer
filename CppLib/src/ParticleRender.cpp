#include "ParticleRender.h"
#include "CvFuncs.h"
#include <random>

// Global random engine (shared with CvFuncs)
extern std::mt19937& get_rng();

float ParticleRender::particle_speed_val = 10.0f;
float ParticleRender::particle_spread_val = 0.2f;
float ParticleRender::particle_decay_val = 0.95f;
int ParticleRender::particle_size_val = 2;
int ParticleRender::particle_amount_val = 7;
int ParticleRender::particle_masks_number_val = 3;

ParticleRender::ParticleRender(const cv::Size& screen_size, int max_note,
                               std::function<std::pair<int,int>(int)> get_side_x_func,
                               float frame_dt, int white_key_height,
                               const std::vector<cv::Scalar>& color_map)
    : screen_size(screen_size), max_note(max_note),
      get_side_x_func(get_side_x_func), frame_dt(frame_dt),
      white_key_height(white_key_height), color_map(color_map)
{
    // linspace(-1, 1, particle_masks_number)
    for (int i = 0; i < particle_masks_number_val; i++) {
        float t = -1.0f + (2.0f * i) / (particle_masks_number_val - 1);  // linspace(-1, 1, 3)
        cv::Mat mask = cv::Mat::zeros(screen_size.height, screen_size.width, CV_32FC3);
        ParticleParams params;
        params.particle_speed = particle_speed_val;
        params.particle_spread = particle_spread_val * t;
        params.particle_decay = particle_decay_val;
        params.particle_size = particle_size_val;
        params.particle_amount = particle_amount_val;
        particle_masks.push_back({mask, params});
    }
}

void ParticleRender::draw_particle(cv::Mat& frame, cv::Mat& glow_illuminant,
                                   const std::map<int, std::map<std::string, std::vector<int>>>& notes)
{
    auto& rng = get_rng();

    for (const auto& kv : notes) {
        int note = kv.first;
        const auto& note_state = kv.second;

        auto it_playing = note_state.find("playing");
        if (it_playing == note_state.end()) continue;
        const std::vector<int>& playing_track_idx_list = it_playing->second;

        if (!playing_track_idx_list.empty()) {
            auto side_x = get_side_x_func(note);
            int x_left = side_x.first;
            int x_right = side_x.second;

            std::vector<cv::Scalar> color_list = extract_colors_by_indices(color_map, playing_track_idx_list);

            std::uniform_int_distribution<int> dist_x(x_left, x_right - 1);
            if (x_right - 1 < x_left) dist_x = std::uniform_int_distribution<int>(x_left, x_left);

            for (size_t particle_mask_index = 0; particle_mask_index < particle_masks.size(); particle_mask_index++) {
                const ParticleParams& params = particle_masks[particle_mask_index].second;
                int y_range_start = screen_size.height - white_key_height - (int)params.particle_speed;
                int y_range_end = screen_size.height - white_key_height - 1;
                if (y_range_end < y_range_start) y_range_end = y_range_start;

                std::uniform_int_distribution<int> dist_y(y_range_start, y_range_end);

                for (int _ = 0; _ < params.particle_amount; _++) {
                    int px = dist_x(rng);
                    int py = dist_y(rng);

                    if (!color_list.empty()) {
                        std::uniform_int_distribution<int> dist_c(0, (int)color_list.size() - 1);
                        cv::Scalar rand_color = color_list[dist_c(rng)];
                        cv::circle(particle_masks[particle_mask_index].first,
                                   cv::Point(px, py), params.particle_size, rand_color, -1);
                    }
                }
            }
        }
    }

    for (size_t particle_mask_index = 0; particle_mask_index < particle_masks.size(); particle_mask_index++) {
        cv::Mat& mask_float = particle_masks[particle_mask_index].first;
        const ParticleParams& params = particle_masks[particle_mask_index].second;

        cv::Mat uint8_mask;
        mask_float.convertTo(uint8_mask, CV_8UC3);

        cv::max(frame, uint8_mask, frame);
        cv::max(glow_illuminant, uint8_mask, glow_illuminant);

        // Translation matrix: [[1, 0, particle_spread], [0, 1, -particle_speed]]
        cv::Mat translation_matrix = (cv::Mat_<float>(2, 3) <<
            1.0f, 0.0f, params.particle_spread,
            0.0f, 1.0f, -params.particle_speed);

        cv::Mat warped;
        cv::warpAffine(mask_float, warped, translation_matrix,
                       cv::Size(frame.cols, frame.rows),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        mask_float = warped * params.particle_decay;
    }
}