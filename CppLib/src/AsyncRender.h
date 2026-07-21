#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <atomic>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>

class MidiPianoRender;

struct AsyncRenderJob {
    int job_id;
    MidiPianoRender* render;
    float bpm;
    std::vector<int32_t> notes_data;
    int notes_count;  // original length of notes_data
    std::vector<uint8_t> low_layers_data;
    int low_layers_count;
    int low_layer_cols;
    int low_layer_rows;
    bool completed;
    std::vector<cv::Mat> result;
    int result_frame_count;
    int result_frame_cols;
    int result_frame_rows;
};

class AsyncRender {
public:
    AsyncRender();
    ~AsyncRender();

    // Submit a render job. Returns immediately with a job_id.
    // The actual rendering happens on the worker thread.
    // notes_data is copied into the job, so the caller can free it.
    int submit_render(
        MidiPianoRender* render,
        float bpm,
        const int32_t* notes_data,
        int notes_count,
        const uint8_t* low_layers_data,
        int low_layers_count,
        int low_layer_cols,
        int low_layer_rows
    );

    // Check if a job has completed rendering.
    bool is_ready(int job_id);

    // Retrieve the result of a completed job.
    // Returns raw BGR data (malloc'd, caller must free with libc's free()).
    // Sets out_frame_count, out_frame_cols, out_frame_rows.
    // Returns nullptr if job not ready or not found.
    uint8_t* get_result(int job_id, int* out_frame_count, int* out_frame_cols, int* out_frame_rows);

private:
    void worker_loop();

    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<int> job_queue_;
    std::map<int, AsyncRenderJob> jobs_;
    std::atomic<int> next_job_id_{1};
    std::atomic<bool> running_{true};
};