import mido
from typing import Dict, List, Set, Iterator, Tuple

class MidiParser:
    def __init__(self, midi_path: str, extra_ticks = 0):
        """
        解析 MIDI 文件，提取音符事件和 tempo 事件。
        """
        self.mid = mido.MidiFile(midi_path)
        self.track_events = []   # 每个元素为该轨道的事件列表，事件为 (abs_tick, message)
        self.max_tick = 0

        # 收集所有 tempo 事件（绝对 tick, BPM）
        self.tempo_changes = []  # (abs_tick, bpm)
        
        for track_idx, track in enumerate(self.mid.tracks):
            abs_time = 0
            events = []
            for msg in track:
                abs_time += msg.time
                # 记录 tempo 事件
                if msg.type == 'set_tempo':
                    tempo = msg.tempo  # 微秒/四分音符
                    bpm = 60000000.0 / tempo
                    self.tempo_changes.append((abs_time, bpm))
                # 记录音符事件
                if msg.type in ('note_on', 'note_off'):
                    events.append((abs_time, msg))
                    if abs_time > self.max_tick:
                        self.max_tick = abs_time
            self.track_events.append(events)

        # 确保初始 BPM 存在（默认 120 BPM）
        if not self.tempo_changes or self.tempo_changes[0][0] != 0:
            self.tempo_changes.insert(0, (0, 120.0))
        else:
            # 如果 0 tick 有多个 tempo，保留最后一个（通常按顺序）
            # 简单去重：按 tick 分组，取最后一个
            tempo_dict = {}
            for tick, bpm in self.tempo_changes:
                tempo_dict[tick] = bpm
            self.tempo_changes = sorted(tempo_dict.items())  # (tick, bpm)

    def iter_ticks(self) -> Iterator[Dict[str, object]]:
        """
        返回一个生成器，每次迭代返回一个字典，包含：
            'notes': 当前 tick 下所有音符的状态（同原返回值）
            'bpm':   当前 tick 对应的 BPM 值

        音符状态字典格式：
            {
                note_pitch: {
                    'on': [track_indices],      # 本 tick 开启该音符的轨道索引（升序）
                    'playing': [track_indices], # 本 tick 正在播放该音符的轨道索引（升序）
                    'off': [track_indices]      # 本 tick 关闭该音符的轨道索引（升序）
                }
            }
        """
        track_events = self.track_events
        num_tracks = len(track_events)
        track_states = [set() for _ in range(num_tracks)]   # 当前各轨道正在播放的音符集合
        track_pointers = [0] * num_tracks                   # 各轨道已处理到的事件索引
        current_tick = 0

        # BPM 相关
        tempo_idx = 0
        current_bpm = self.tempo_changes[0][1] if self.tempo_changes else 120.0

        while current_tick <= self.max_tick:
            # 更新 BPM（处理所有 tick ≤ current_tick 的 tempo 事件）
            while tempo_idx < len(self.tempo_changes) and self.tempo_changes[tempo_idx][0] <= current_tick:
                current_bpm = self.tempo_changes[tempo_idx][1]
                tempo_idx += 1

            open_dict = {}      # note -> list of track indices (开启)
            close_dict = {}     # note -> list of track indices (关闭)
            playing_dict = {}   # note -> list of track indices (正在播放)

            for track_idx in range(num_tracks):
                events = track_events[track_idx]
                ptr = track_pointers[track_idx]

                # 收集当前 tick 下该轨道的所有事件
                tick_msgs = []
                while ptr < len(events) and events[ptr][0] == current_tick:
                    tick_msgs.append(events[ptr][1])
                    ptr += 1
                track_pointers[track_idx] = ptr

                if tick_msgs:
                    # 记录处理前的状态，用于计算 "playing"
                    state_before = track_states[track_idx].copy()
                    open_set = set()
                    close_set = set()

                    # 按原始顺序处理事件，更新轨道状态
                    for msg in tick_msgs:
                        if msg.type == 'note_on' and msg.velocity > 0:
                            note = msg.note
                            open_set.add(note)
                            track_states[track_idx].add(note)
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            note = msg.note
                            close_set.add(note)
                            track_states[track_idx].discard(note)

                    # 计算本 tick 该轨道上正在播放的音符集合
                    playing_set = (state_before - close_set) | open_set

                    # 记录到全局字典
                    for note in open_set:
                        open_dict.setdefault(note, []).append(track_idx)
                    for note in close_set:
                        close_dict.setdefault(note, []).append(track_idx)
                    for note in playing_set:
                        playing_dict.setdefault(note, []).append(track_idx)
                else:
                    # 无事件，当前轨道播放状态不变，全部纳入 "playing"
                    for note in track_states[track_idx]:
                        playing_dict.setdefault(note, []).append(track_idx)

            # 组装音符状态字典
            notes_result = {}
            all_notes = range(128) # set(open_dict.keys()) | set(close_dict.keys()) | set(playing_dict.keys())
            for note in all_notes:
                notes_result[note] = {
                    'on': sorted(open_dict.get(note, [])),
                    'playing': sorted(playing_dict.get(note, [])),
                    'off': sorted(close_dict.get(note, []))
                }

            # 返回包含 notes 和 bpm 的字典
            yield {
                'notes': notes_result,
                'bpm': current_bpm
            }

            current_tick += 1

    def get_tracks_info(self) -> Dict[int, str]:
        """
        返回所有轨道的索引及对应的轨道名称。
        """
        return {i: self.mid.tracks[i].name for i in range(len(self.mid.tracks))}


class StateProcessor:
    def __init__(self, track_idx_list, shift_rule = {}):
        self.track_idx_list = track_idx_list
        self.shift_rule = shift_rule
        self.history_min_n = None
        self.history_max_n = None

    def shift_and_split_state(self, state):
        notes = state['notes']
        bpm = state['bpm']

        new_min_n, new_max_n = None, None

        new_notes = {n : {'on': [], 'playing': [], 'off': []} for n in notes.keys()}
        for n in notes.keys():
            for event_type in ['on', 'playing', 'off']:
                tracks = notes[n][event_type]
                for track_idx in tracks:
                    now_n = n
                    if track_idx in self.shift_rule:
                        shifted_n = n + self.shift_rule[track_idx]
                        if shifted_n in new_notes:
                            new_notes[shifted_n][event_type].append(track_idx)
                        now_n = shifted_n
                    else:
                        new_notes[n][event_type].append(track_idx)

                    if new_min_n is None:
                        new_min_n = now_n
                    else:
                        new_min_n = min(new_min_n, now_n)
                    if new_max_n is None:
                        new_max_n = now_n
                    else:
                        new_max_n = max(new_max_n, now_n)
                    if self.history_min_n is None:
                        self.history_min_n = now_n
                    else:
                        self.history_min_n = min(self.history_min_n, now_n)
                    if self.history_max_n is None:
                        self.history_max_n = now_n
                    else:
                        self.history_max_n = max(self.history_max_n, now_n)

        
        # print(new_min_n, new_max_n, self.history_min_n, self.history_max_n)
            
        for n in new_notes.keys():
            for event_type in ['on', 'playing', 'off']:
                new_notes[n][event_type].sort()

        split_notes = {}

        for track_idx in self.track_idx_list:
            split_notes[track_idx] = {n : {'on': [], 'playing': [], 'off': []} for n in new_notes.keys()}
            for n in new_notes.keys():
                for event_type in ['on', 'playing', 'off']:
                    if track_idx in new_notes[n][event_type]:
                        split_notes[track_idx][n][event_type].append(track_idx)
        
        return {
                'notes': new_notes,
                'bpm': bpm
            }, {
                track_idx: {
                    'notes': split_notes[track_idx],
                    'bpm': bpm
                } for track_idx in self.track_idx_list
            }


