#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Opaque handle to MidiPianoRender
typedef void* MidiRenderHandle;

// Create a MidiPianoRender instance
// track_layer_idx: -1 for None (default light bar color), 0..12 for track color
// render_note: 1 to enable note rendering, 0 to disable
MidiRenderHandle midi_render_create(int track_layer_idx, int render_note);

// Destroy a MidiPianoRender instance
void midi_render_destroy(MidiRenderHandle handle);

// Set random seed
void midi_render_set_seed(unsigned int seed);

// Get white_key_height (needed by Python to compute note_distance_dt)
int midi_render_get_white_key_height(MidiRenderHandle handle);

// Render frames
// Parameters:
//   handle: MidiPianoRender instance
//   bpm: beats per minute (float)
//   notes_data: packed note data (see Python wrapper for encoding)
//   notes_count: number of notes with data
//   low_layers_data: optional pre-rendered bottom layers (may be NULL)
//   low_layers_count: number of low layer frames
//   low_layer_rows, low_layer_cols: dimensions of each low layer frame
//   out_frame_data: output buffer (pre-allocated, will be filled)
//   out_frame_count: number of frames rendered (output)
// Returns: pointer to array of frame data blocks (caller must free each with free())
// Frame data: rows * cols * 3 bytes (BGR), total bytes = out_frame_count * rows * cols * 3
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
    int* out_frame_rows
);

// ========== Async Render API ==========

// Opaque handle to AsyncRender
typedef void* AsyncRenderHandle;

// Create an AsyncRender instance (allocates a persistent worker thread)
AsyncRenderHandle async_render_create();

// Destroy an AsyncRender instance (joins worker thread)
void async_render_destroy(AsyncRenderHandle handle);

// Submit a render job. Returns immediately with a job_id.
// The worker thread will process the job asynchronously.
// notes_data and low_layers_data are copied internally, caller can free them.
int async_render_submit(
    AsyncRenderHandle handle,
    MidiRenderHandle render,
    float bpm,
    const int32_t* notes_data,
    int notes_count,
    const uint8_t* low_layers_data,
    int low_layers_count,
    int low_layer_cols,
    int low_layer_rows
);

// Check if a submitted job has finished rendering.
// Returns 1 if ready, 0 if not ready or not found.
int async_render_is_ready(AsyncRenderHandle handle, int job_id);

// Retrieve the result of a completed job.
// Returns pointer to raw BGR data (malloc'd, caller must free with libc's free()).
// Sets out_frame_count, out_frame_cols, out_frame_rows.
// Returns NULL if job not ready or not found.
// After calling this, the job is removed from memory.
uint8_t* async_render_get_result(
    AsyncRenderHandle handle,
    int job_id,
    int* out_frame_count,
    int* out_frame_cols,
    int* out_frame_rows
);

#ifdef __cplusplus
}
#endif
