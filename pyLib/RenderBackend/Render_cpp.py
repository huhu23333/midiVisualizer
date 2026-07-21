import os
import numpy as np
import cv2
import ctypes
from ctypes import c_void_p, c_int, c_float, c_uint32, c_uint8, POINTER, byref

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), '../../CppLib/build/libmidi_render_backend.so')
_lib = ctypes.cdll.LoadLibrary(_lib_path)

# Function prototypes
_lib.midi_render_create.argtypes = [c_int, c_int]
_lib.midi_render_create.restype = c_void_p

_lib.midi_render_destroy.argtypes = [c_void_p]
_lib.midi_render_destroy.restype = None

_lib.midi_render_set_seed.argtypes = [c_uint32]
_lib.midi_render_set_seed.restype = None

_lib.midi_render_get_white_key_height.argtypes = [c_void_p]
_lib.midi_render_get_white_key_height.restype = c_int

_lib.midi_render_render_frames.argtypes = [
    c_void_p,                              # handle
    c_float,                               # bpm
    POINTER(c_int),                        # notes_data
    c_int,                                 # notes_count
    POINTER(c_uint8),                      # low_layers_data
    c_int,                                 # low_layers_count
    c_int,                                 # low_layer_cols
    c_int,                                 # low_layer_rows
    POINTER(c_int),                        # out_frame_count
    POINTER(c_int),                        # out_frame_cols
    POINTER(c_int),                        # out_frame_rows
]
_lib.midi_render_render_frames.restype = POINTER(c_uint8)

# Re-export CvFuncs for compatibility (Python versions are still used for some utilities)
# from CvFuncs import shift_and_fill_inplace, mix_color, extract_colors_by_indices, add_glow_effect

# Constants (same as Render.py)
max_note = 128
note_shift = 0
note_in_tmp_height = 200
note_out_tmp_height = 2000
note_speed = 400.0

screen_size = (1920, 1080)
fps = 60.0
frame_dt = 1.0 / fps

color_map = [None,
             (120, 0, 255), (0, 180, 255), (255, 240, 60), (255, 30, 180),
             (255, 255, 255), (180, 180, 180), (0, 255, 255), (180, 180, 255),
             (180, 30, 255), (0, 255, 0), (255, 120, 120), (240, 120, 255),
]


def set_seed(seed):
    _lib.midi_render_set_seed(seed)


def _pack_notes(notes):
    """
    Pack notes dict into flat int32 array for C API.
    Format: [total_notes_count][note_idx][on_count][on_values...][playing_count][playing_values...][off_count][off_values...]
    """
    data = []
    # Count notes that have any data
    active_notes = []
    for note_idx in sorted(notes.keys()):
        note_state = notes[note_idx]
        has_data = (len(note_state.get('on', [])) > 0 or 
                   len(note_state.get('playing', [])) > 0 or 
                   len(note_state.get('off', [])) > 0)
        if has_data:
            active_notes.append(note_idx)
    
    data.append(len(active_notes))
    for note_idx in active_notes:
        note_state = notes[note_idx]
        data.append(note_idx)
        
        for key in ['on', 'playing', 'off']:
            vals = note_state.get(key, [])
            data.append(len(vals))
            data.extend(vals)
    
    return np.array(data, dtype=np.int32)


def _unpack_frames(data_ptr, frame_count, cols, rows):
    """
    Unpack raw BGR data from C into list of numpy arrays.
    """
    frame_size = cols * rows * 3
    result = []
    for i in range(frame_count):
        offset = i * frame_size
        frame_data = np.ctypeslib.as_array(
            (ctypes.c_uint8 * frame_size).from_address(
                ctypes.addressof(data_ptr.contents) + offset
            )
        ).copy()
        frame = frame_data.reshape((rows, cols, 3))
        result.append(frame)
    return result


class MidiPianoRender:
    def __init__(self, track_layer_idx=None, render_note=True):
        if track_layer_idx is not None:
            c_track = track_layer_idx
        else:
            c_track = -1  # None maps to -1
        
        self._handle = _lib.midi_render_create(c_track, 1 if render_note else 0)
        if not self._handle:
            raise RuntimeError("Failed to create MidiPianoRender C++ instance")
        
        self.render_note = render_note
        self.track_layer_idx = track_layer_idx

        # Access white_key_height from C++ instance
        self.white_key_height = _lib.midi_render_get_white_key_height(self._handle)

        # These attributes are needed for Python-level compatibility
        # (e.g., for note_distance_dt calculation in the original code)
        # but are not used by the C++ implementation itself
        
        # The following are provided for API compatibility with the Python version
        # but most rendering is done in C++
        
        # For compatibility with PianoRender access patterns
        class _PianoRenderCompat:
            def __init__(self, outer):
                self.white_key_height = outer.white_key_height
                self._outer = outer
            
            def get_side_x(self, note):
                # Use the C++ implementation via helper
                return (int(note * screen_size[0] / max_note), 
                        int((note + 1) * screen_size[0] / max_note))
        
        self.pir = _PianoRenderCompat(self)
        
        class _NotesRenderCompat:
            def __init__(self):
                self.keep_for_out_pix = 50
                self.min_update_pix = 100
        
        self.nr = _NotesRenderCompat()
        
        # These are stored for compatibility but rendering happens in C++
        self.tick_time_count = 0.0
        self.frame_count = 0
        
        # piano_delay_list for compatibility (managed by C++ internally)
        self.piano_delay_list = [(0.0, [False for _ in range(max_note)], 
                                  {i: {'on': [], 'playing': [], 'off': []} for i in range(max_note)})]
        
        # note_distance_dt (same calculation as Python version)
        self.note_distance_dt = (self.nr.keep_for_out_pix + note_out_tmp_height - self.white_key_height) / note_speed

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.midi_render_destroy(self._handle)
            self._handle = None

    def render_frames(self, state, low_layers=None):
        notes = state["notes"]
        bpm = state["bpm"]
        
        # Pack notes
        notes_data = _pack_notes(notes)
        
        # Prepare low_layers
        low_layers_ptr = None
        low_layers_count = 0
        low_layer_cols = 0
        low_layer_rows = 0
        if low_layers is not None:
            low_layers_count = len(low_layers)
            if low_layers_count > 0:
                low_layer_rows = low_layers[0].shape[0]
                low_layer_cols = low_layers[0].shape[1]
                # Ensure contiguous
                low_layers_concat = np.concatenate(
                    [np.ascontiguousarray(f).ravel() for f in low_layers]
                )
                low_layers_ptr = low_layers_concat.ctypes.data_as(POINTER(c_uint8))
        
        # Call C function
        out_frame_count = c_int()
        out_frame_cols = c_int()
        out_frame_rows = c_int()
        
        notes_data_ptr = notes_data.ctypes.data_as(POINTER(c_int))
        
        raw_ptr = _lib.midi_render_render_frames(
            self._handle,
            c_float(bpm),
            notes_data_ptr,
            len(notes_data),
            low_layers_ptr,
            low_layers_count,
            low_layer_cols,
            low_layer_rows,
            byref(out_frame_count),
            byref(out_frame_cols),
            byref(out_frame_rows)
        )
        
        # Unpack
        result = _unpack_frames(raw_ptr, out_frame_count.value, 
                                out_frame_cols.value, out_frame_rows.value)
        
        # Free C memory (using libc's free)
        _libc = ctypes.CDLL("libc.so.6")
        _libc.free.argtypes = [ctypes.c_void_p]
        _libc.free.restype = None
        _libc.free(raw_ptr)
        
        # Update tick_time_count and frame_count for compatibility
        tick_dt = 60 / (bpm * 48)
        self.tick_time_count += tick_dt
        self.frame_count += out_frame_count.value
        
        return result