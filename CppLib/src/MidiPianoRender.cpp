#include "MidiPianoRender.h"
#include "CvFuncs.h"

// color_map: indexed by track index, [0] is placeholder None
std::vector<cv::Scalar> color_map = {
    cv::Scalar(0,0,0),           // [0]  None
    cv::Scalar(120,0,255),       // [1]
    cv::Scalar(0,180,255),       // [2]
    cv::Scalar(255,240,60),      // [3]
    cv::Scalar(255,30,180),      // [4]
    cv::Scalar(255,255,255),     // [5]
    cv::Scalar(180,180,180),     // [6]
    cv::Scalar(0,255,255),       // [7]
    cv::Scalar(180,180,255),     // [8]
    cv::Scalar(180,30,255),      // [9]
    cv::Scalar(0,255,0),         // [10]
    cv::Scalar(255,120,120),     // [11]
    cv::Scalar(240,120,255),     // [12]
};

MidiPianoRender::MidiPianoRender(int track_layer_idx, bool render_note)
    : screen_size(1920, 1080), max_note(128), note_shift(0),
      note_in_tmp_height(200), note_out_tmp_height(2000), note_speed(400.0f),
      fps(60.0f), frame_dt(1.0f / 60.0f),
      pir(screen_size, max_note,
          track_layer_idx >= 0 && track_layer_idx < 13 ?
            color_map[track_layer_idx] :
            cv::Scalar(200,240,255)),
      nr([this](int note) { return pir.get_side_x(note); }),
      last_move_note_tmp_pix_residual(0.0f),
      par(nullptr),
      note_distance_dt(0.0f),
      render_note(render_note),
      track_layer_idx(track_layer_idx),
      tick_time_count(0.0f),
      frame_count(0)
{
    if (render_note) {
        note_in_tmp = cv::Mat::zeros(note_in_tmp_height, screen_size.width, CV_8UC3);
        note_out_tmp = cv::Mat::zeros(note_out_tmp_height, screen_size.width, CV_8UC3);
        note_in_tmp_mask = cv::Mat::zeros(note_in_tmp_height, screen_size.width, CV_8UC1);
        note_out_tmp_mask = cv::Mat::zeros(note_out_tmp_height, screen_size.width, CV_8UC1);

        last_move_note_tmp_pix_residual = 0.0f;

        par = new ParticleRender(screen_size, max_note,
                                 [this](int note) { return pir.get_side_x(note); },
                                 frame_dt, pir.white_key_height, color_map);
    }

    note_distance_dt = (nr.keep_for_out_pix + (float)note_out_tmp_height - pir.white_key_height) / note_speed;

    // Initialize piano_delay_list
    std::vector<bool> nsl_init(max_note, false);
    std::map<int, std::map<std::string, std::vector<int>>> delayed_notes_init;
    for (int i = 0; i < max_note; i++) {
        delayed_notes_init[i] = {
            {"on", {}},
            {"playing", {}},
            {"off", {}}
        };
    }
    piano_delay_list.push_back({0.0f, nsl_init, delayed_notes_init});
}

void MidiPianoRender::set_seed(unsigned int seed) {
    ::set_seed(seed);
}

// Helper: shift notes by note_shift
static std::map<int, std::map<std::string, std::vector<int>>> shift_notes(
    const std::map<int, std::map<std::string, std::vector<int>>>& notes, int note_shift, int max_note)
{
    std::map<int, std::map<std::string, std::vector<int>>> shifted;
    for (int i = 0; i < max_note; i++) {
        auto it = notes.find(i + note_shift);
        if (it != notes.end()) {
            shifted[i] = it->second;
        } else {
            shifted[i] = {{"on", {}}, {"playing", {}}, {"off", {}}};
        }
    }
    return shifted;
}

std::vector<cv::Mat> MidiPianoRender::render_frames(
    const std::map<std::string, std::map<int, std::map<std::string, std::vector<int>>>>& state,
    const std::vector<cv::Mat>* low_layers)
{
    const auto& notes = state.at("notes");

    // bpm is passed inside state as: state["bpm"][0]["bpm"][0] containing bpm*1000 as int
    // We'll extract it. The bindings layer encodes bpm as:
    // state["bpm"][0] -> map with key "bpm" -> vector<int> with one element = (int)(bpm * 1000)
    float bpm = 120.0f;
    auto bpm_it = state.find("bpm");
    if (bpm_it != state.end()) {
        auto note0_it = bpm_it->second.find(0);
        if (note0_it != bpm_it->second.end()) {
            auto vec_it = note0_it->second.find("bpm");
            if (vec_it != note0_it->second.end() && !vec_it->second.empty()) {
                bpm = (float)vec_it->second[0] / 1000.0f;
            }
        }
    }

    float tick_dt = 60.0f / (bpm * 48.0f);

    std::map<int, std::map<std::string, std::vector<int>>> shifted_notes =
        shift_notes(notes, note_shift, max_note);

    // Build nsl_for_delay
    std::vector<bool> nsl_for_delay(max_note);
    for (int i = 0; i < max_note; i++) {
        auto it = shifted_notes[i].find("playing");
        nsl_for_delay[i] = (it != shifted_notes[i].end() && !it->second.empty());
    }

    piano_delay_list.push_back({tick_time_count + note_distance_dt, nsl_for_delay, shifted_notes});

    if (render_note) {
        float target_pix = tick_dt * note_speed + last_move_note_tmp_pix_residual;
        int step_pix = (int)std::round(target_pix);
        last_move_note_tmp_pix_residual = target_pix - (float)step_pix;

        nr.update_in_temp(note_in_tmp, note_in_tmp_mask, step_pix, shifted_notes, color_map);
        shift_and_fill_inplace(note_out_tmp, note_in_tmp, step_pix);
        shift_and_fill_inplace(note_out_tmp_mask, note_in_tmp_mask, step_pix);
    }

    std::vector<cv::Mat> result;
    tick_time_count += tick_dt;
    int frame_idx = 0;
    while ((float)frame_count * frame_dt <= tick_time_count) {
        cv::Mat frame = cv::Mat::zeros(screen_size.height, screen_size.width, CV_8UC3);
        cv::Mat glow_illuminant = cv::Mat::zeros(screen_size.height, screen_size.width, CV_8UC3);

        if (low_layers != nullptr) {
            assert((size_t)frame_idx < low_layers->size());
            assert(frame.size() == (*low_layers)[frame_idx].size());
            (*low_layers)[frame_idx].copyTo(frame);
        }

        if (render_note) {
            float over_dt = tick_time_count - (float)frame_count * frame_dt;
            int over_pix = (int)std::round(note_speed * over_dt);

            cv::Mat tmp_note_frame = cv::Mat::zeros(frame.size(), frame.type());
            int copy_rows = screen_size.height - over_pix;
            if (copy_rows > 0) {
                note_out_tmp(cv::Rect(0, note_out_tmp.rows - copy_rows, 
                                      note_out_tmp.cols, copy_rows))
                    .copyTo(tmp_note_frame(cv::Rect(0, 0, note_out_tmp.cols, copy_rows)));
            }

            cv::Mat note_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
            if (copy_rows > 0) {
                note_out_tmp_mask(cv::Rect(0, note_out_tmp_mask.rows - copy_rows,
                                           note_out_tmp_mask.cols, copy_rows))
                    .copyTo(note_mask(cv::Rect(0, 0, note_out_tmp_mask.cols, copy_rows)));
            }

            cv::copyTo(tmp_note_frame, frame, note_mask);
            cv::copyTo(tmp_note_frame, glow_illuminant, note_mask);

            cv::rectangle(glow_illuminant,
                         cv::Point(0, screen_size.height - pir.white_key_height),
                         cv::Point(screen_size.width, screen_size.height),
                         cv::Scalar(0, 0, 0), -1);
        }

        // Pop piano delay list
        while (piano_delay_list.size() >= 2 && 
               ((float)frame_count * frame_dt >= std::get<0>(piano_delay_list[1]))) {
            piano_delay_list.erase(piano_delay_list.begin());
        }

        const std::vector<bool>& nsl_for_pir = std::get<1>(piano_delay_list[0]);
        const auto& delayed_notes = std::get<2>(piano_delay_list[0]);

        // Build on_color_list
        std::vector<cv::Scalar> on_color_list(max_note);
        for (int i = 0; i < max_note; i++) {
            if (nsl_for_pir[i]) {
                auto it = delayed_notes.find(i);
                if (it != delayed_notes.end()) {
                    auto play_it = it->second.find("playing");
                    if (play_it != it->second.end()) {
                        std::vector<cv::Scalar> extracted = extract_colors_by_indices(color_map, play_it->second);
                        on_color_list[i] = mix_color(extracted);
                    } else {
                        on_color_list[i] = cv::Scalar(0, 0, 0);
                    }
                } else {
                    on_color_list[i] = cv::Scalar(0, 0, 0);
                }
            } else {
                on_color_list[i] = cv::Scalar(0, 0, 0);
            }
        }

        pir.draw_piano(frame, glow_illuminant, nsl_for_pir, on_color_list);

        if (render_note) {
            par->draw_particle(frame, glow_illuminant, delayed_notes);
        }

        add_glow_effect(frame, glow_illuminant);

        result.push_back(frame.clone());

        frame_count++;
        frame_idx++;
    }

    return result;
}