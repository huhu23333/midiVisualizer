#include "render_backend.h"
#include "MidiPianoRender.h"
#include "AsyncRender.h"
#include <cstring>
#include <cstdlib>
#include <random>

// Expose RNG for ParticleRender
std::mt19937& get_rng() {
    static std::mt19937 rng(42);
    return rng;
}

extern "C" {

MidiRenderHandle midi_render_create(int track_layer_idx, int render_note) {
    // track_layer_idx: -1 means None (no specific track color)
    // In Python: color_map[track_layer_idx] if track_layer_idx else None
    // track_layer_idx is used as index into color_map, None means use default
    // We'll pass track_layer_idx directly, the constructor handles -1 -> default
    return new MidiPianoRender(track_layer_idx, render_note != 0);
}

void midi_render_destroy(MidiRenderHandle handle) {
    delete static_cast<MidiPianoRender*>(handle);
}

void midi_render_set_seed(unsigned int seed) {
    MidiPianoRender::set_seed(seed);
}

int midi_render_get_white_key_height(MidiRenderHandle handle) {
    return static_cast<MidiPianoRender*>(handle)->get_white_key_height();
}

uint8_t* midi_render_render_frames(
    MidiRenderHandle handle,
    float bpm,
    const int32_t* notes_data,
    int notes_count,
    const uint8_t* low_layers_data,
    int low_layers_count,
    int low_layer_cols,
    int low_layer_rows,
    int* out_frame_count,
    int* out_frame_cols,
    int* out_frame_rows)
{
    MidiPianoRender* render = static_cast<MidiPianoRender*>(handle);

    // Decode notes_data into state structure
    // Format: [total_notes_count][note0_idx][on0_count][on_values...][playing_count][playing_values...][off_count][off_values...]
    //         [note1_idx][on1_count]... etc.
    std::map<int, std::map<std::string, std::vector<int>>> notes;

    int pos = 0;
    int total_notes = notes_data[pos++];
    for (int n = 0; n < total_notes; n++) {
        int note_idx = notes_data[pos++];

        // on
        int on_count = notes_data[pos++];
        std::vector<int> on_vec;
        for (int i = 0; i < on_count; i++) {
            on_vec.push_back(notes_data[pos++]);
        }
        notes[note_idx]["on"] = on_vec;

        // playing
        int playing_count = notes_data[pos++];
        std::vector<int> playing_vec;
        for (int i = 0; i < playing_count; i++) {
            playing_vec.push_back(notes_data[pos++]);
        }
        notes[note_idx]["playing"] = playing_vec;

        // off
        int off_count = notes_data[pos++];
        std::vector<int> off_vec;
        for (int i = 0; i < off_count; i++) {
            off_vec.push_back(notes_data[pos++]);
        }
        notes[note_idx]["off"] = off_vec;
    }

    // Build state: state["notes"] = notes, state["bpm"] encoded
    std::map<std::string, std::map<int, std::map<std::string, std::vector<int>>>> state;
    state["notes"] = notes;

    // Encode bpm inside state["bpm"][0]["bpm"] as vector with one element = bpm*1000
    state["bpm"] = {
        {0, {{"bpm", {(int)(bpm * 1000.0f)}}}}
    };

    // Decode low_layers if provided
    std::vector<cv::Mat> low_layers;
    if (low_layers_data != nullptr && low_layers_count > 0) {
        int frame_size = low_layer_cols * low_layer_rows * 3;
        for (int i = 0; i < low_layers_count; i++) {
            cv::Mat frame(low_layer_rows, low_layer_cols, CV_8UC3,
                         (void*)(low_layers_data + i * frame_size));
            low_layers.push_back(frame.clone());
        }
    }

    // Call render
    std::vector<cv::Mat> result = render->render_frames(
        state, low_layers.empty() ? nullptr : &low_layers);

    // Pack output
    *out_frame_count = (int)result.size();
    if (!result.empty()) {
        *out_frame_cols = result[0].cols;
        *out_frame_rows = result[0].rows;
    } else {
        *out_frame_cols = 0;
        *out_frame_rows = 0;
        return nullptr;
    }

    int frame_size = result[0].cols * result[0].rows * 3;
    int total_size = frame_size * (int)result.size();
    uint8_t* output = (uint8_t*)malloc(total_size);

    for (size_t i = 0; i < result.size(); i++) {
        // Ensure continuous
        cv::Mat cont;
        if (!result[i].isContinuous()) {
            cont = result[i].clone();
        } else {
            cont = result[i];
        }
        memcpy(output + i * frame_size, cont.data, frame_size);
    }

    return output;
}

// ========== Async Render API implementation ==========

AsyncRenderHandle async_render_create() {
    return new AsyncRender();
}

void async_render_destroy(AsyncRenderHandle handle) {
    delete static_cast<AsyncRender*>(handle);
}

int async_render_submit(
    AsyncRenderHandle handle,
    MidiRenderHandle render,
    float bpm,
    const int32_t* notes_data,
    int notes_count,
    const uint8_t* low_layers_data,
    int low_layers_count,
    int low_layer_cols,
    int low_layer_rows)
{
    AsyncRender* ar = static_cast<AsyncRender*>(handle);
    MidiPianoRender* mpr = static_cast<MidiPianoRender*>(render);
    return ar->submit_render(mpr, bpm, notes_data, notes_count,
                             low_layers_data, low_layers_count,
                             low_layer_cols, low_layer_rows);
}

int async_render_is_ready(AsyncRenderHandle handle, int job_id) {
    AsyncRender* ar = static_cast<AsyncRender*>(handle);
    return ar->is_ready(job_id) ? 1 : 0;
}

uint8_t* async_render_get_result(
    AsyncRenderHandle handle,
    int job_id,
    int* out_frame_count,
    int* out_frame_cols,
    int* out_frame_rows)
{
    AsyncRender* ar = static_cast<AsyncRender*>(handle);
    return ar->get_result(job_id, out_frame_count, out_frame_cols, out_frame_rows);
}

} // extern "C"
