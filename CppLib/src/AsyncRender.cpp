#include "AsyncRender.h"
#include "MidiPianoRender.h"
#include <cstring>
#include <cstdlib>

static std::map<int, std::map<std::string, std::vector<int>>> decode_notes(
    const int32_t* notes_data, int notes_count)
{
    std::map<int, std::map<std::string, std::vector<int>>> notes;
    int pos = 0;
    int total_notes = notes_data[pos++];
    for (int n = 0; n < total_notes; n++) {
        int note_idx = notes_data[pos++];

        int on_count = notes_data[pos++];
        std::vector<int> on_vec;
        for (int i = 0; i < on_count; i++) {
            on_vec.push_back(notes_data[pos++]);
        }
        notes[note_idx]["on"] = on_vec;

        int playing_count = notes_data[pos++];
        std::vector<int> playing_vec;
        for (int i = 0; i < playing_count; i++) {
            playing_vec.push_back(notes_data[pos++]);
        }
        notes[note_idx]["playing"] = playing_vec;

        int off_count = notes_data[pos++];
        std::vector<int> off_vec;
        for (int i = 0; i < off_count; i++) {
            off_vec.push_back(notes_data[pos++]);
        }
        notes[note_idx]["off"] = off_vec;
    }
    return notes;
}

AsyncRender::AsyncRender()
{
    worker_thread_ = std::thread(&AsyncRender::worker_loop, this);
}

AsyncRender::~AsyncRender()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_one();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

int AsyncRender::submit_render(
    MidiPianoRender* render,
    float bpm,
    const int32_t* notes_data,
    int notes_count,
    const uint8_t* low_layers_data,
    int low_layers_count,
    int low_layer_cols,
    int low_layer_rows)
{
    int job_id = next_job_id_++;

    AsyncRenderJob job;
    job.job_id = job_id;
    job.render = render;
    job.bpm = bpm;
    job.notes_data.assign(notes_data, notes_data + notes_count);
    job.notes_count = notes_count;
    job.low_layers_count = low_layers_count;
    job.low_layer_cols = low_layer_cols;
    job.low_layer_rows = low_layer_rows;
    job.completed = false;

    if (low_layers_data != nullptr && low_layers_count > 0) {
        int frame_size = low_layer_cols * low_layer_rows * 3;
        job.low_layers_data.assign(
            low_layers_data,
            low_layers_data + low_layers_count * frame_size);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_[job_id] = std::move(job);
        job_queue_.push(job_id);
    }
    cv_.notify_one();

    return job_id;
}

bool AsyncRender::is_ready(int job_id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) return false;
    return it->second.completed;
}

uint8_t* AsyncRender::get_result(int job_id,
    int* out_frame_count, int* out_frame_cols, int* out_frame_rows)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end() || !it->second.completed) {
        return nullptr;
    }

    auto& job = it->second;
    *out_frame_count = job.result_frame_count;
    *out_frame_cols = job.result_frame_cols;
    *out_frame_rows = job.result_frame_rows;

    if (job.result.empty()) {
        return nullptr;
    }

    int frame_size = job.result_frame_cols * job.result_frame_rows * 3;
    int total_size = frame_size * job.result_frame_count;
    uint8_t* output = (uint8_t*)malloc(total_size);

    for (int i = 0; i < job.result_frame_count; i++) {
        cv::Mat cont;
        if (!job.result[i].isContinuous()) {
            cont = job.result[i].clone();
        } else {
            cont = job.result[i];
        }
        memcpy(output + i * frame_size, cont.data, frame_size);
    }

    // Remove the job from the map to free memory
    jobs_.erase(it);

    return output;
}

void AsyncRender::worker_loop()
{
    while (running_) {
        int job_id = -1;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return !job_queue_.empty() || !running_;
            });

            if (!running_) break;

            if (!job_queue_.empty()) {
                job_id = job_queue_.front();
                job_queue_.pop();
            }
        }

        if (job_id < 0) continue;

        // Find the job
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = jobs_.find(job_id);
        if (it == jobs_.end()) continue;
        AsyncRenderJob& job = it->second;
        lock.unlock();

        // Perform rendering
        // Decode notes
        auto notes = decode_notes(job.notes_data.data(), job.notes_count);

        // Build state
        std::map<std::string, std::map<int, std::map<std::string, std::vector<int>>>> state;
        state["notes"] = notes;
        state["bpm"] = {
            {0, {{"bpm", {(int)(job.bpm * 1000.0f)}}}}
        };

        // Decode low_layers
        std::vector<cv::Mat> low_layers;
        if (!job.low_layers_data.empty()) {
            int frame_size = job.low_layer_cols * job.low_layer_rows * 3;
            for (int i = 0; i < job.low_layers_count; i++) {
                cv::Mat frame(job.low_layer_rows, job.low_layer_cols, CV_8UC3,
                             (void*)(job.low_layers_data.data() + i * frame_size));
                low_layers.push_back(frame.clone());
            }
        }

        // Call the render_frames method on the MidiPianoRender instance
        std::vector<cv::Mat> result = job.render->render_frames(
            state, low_layers.empty() ? nullptr : &low_layers);

        // Store result
        lock.lock();
        job.result = std::move(result);
        job.result_frame_count = (int)job.result.size();
        if (!job.result.empty()) {
            job.result_frame_cols = job.result[0].cols;
            job.result_frame_rows = job.result[0].rows;
        } else {
            job.result_frame_cols = 0;
            job.result_frame_rows = 0;
        }
        job.completed = true;
        lock.unlock();
    }
}