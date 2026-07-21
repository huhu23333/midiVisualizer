#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <functional>
#include "PianoRender.h"
#include "NotesRender.h"
#include "ParticleRender.h"

class MidiPianoRender {
public:
    MidiPianoRender(int track_layer_idx, bool render_note);

    std::vector<cv::Mat> render_frames(
        const std::map<std::string, std::map<int, std::map<std::string, std::vector<int>>>>& state,
        const std::vector<cv::Mat>* low_layers);

    // For accessing white_key_height from outside (used by bindings)
    int get_white_key_height() const { return pir.white_key_height; }

    cv::Size screen_size;
    int max_note;
    int note_shift;
    int note_in_tmp_height;
    int note_out_tmp_height;
    float note_speed;
    float fps;
    float frame_dt;

    PianoRender pir;
    NotesRender nr;

    // Note temp buffers (only when render_note)
    cv::Mat note_in_tmp;
    cv::Mat note_out_tmp;
    cv::Mat note_in_tmp_mask;
    cv::Mat note_out_tmp_mask;
    float last_move_note_tmp_pix_residual;

    // Particle render (only when render_note)
    ParticleRender* par;  // pointer since only exists when render_note

    float note_distance_dt;
    bool render_note;
    int track_layer_idx;

    float tick_time_count;
    int frame_count;

    // piano_delay_list: vector of (time, note_state_list, delayed_notes)
    std::vector<std::tuple<float, std::vector<bool>, std::map<int, std::map<std::string, std::vector<int>>>>> piano_delay_list;

    static void set_seed(unsigned int seed);
};