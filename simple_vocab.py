from dataclasses import dataclass
import numpy as np

from torch import LongTensor
import mido

# The number of subdivisions of a whole note.
SUBDIVISIONS = 64

# The number of possible pitches.
PITCH_COUNT = 88
# The number of different key velocities.
VELOCITY_COUNT = 16
# The number of possible steps since last key press.
ADVANCE_COUNT = SUBDIVISIONS
# The number of possible steps a key press can last.
DURATION_COUNT = SUBDIVISIONS

def midi_note_to_pitch(note: int) -> int:
    pitch = note - 21
    assert pitch >= 0 and pitch < PITCH_COUNT
    return pitch

def pitch_to_midi_note(note: int) -> int:
    return note + 21

def midi_velocity_to_velocity(velocity: int) -> int:
    return min(round(velocity / (128 // VELOCITY_COUNT)), VELOCITY_COUNT - 1)

def velocity_to_midi_velocity(velocity: int) -> int:
    return velocity * (128 // VELOCITY_COUNT)

@dataclass
class Token:
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int

@dataclass
class ActiveKey:
    start_tick: int
    velocity: int

def parse_track(track) -> list[Token]:
    active_keys: dict[int, ActiveKey] = {}
    tick, tokens = 0, []
    for event in track:
        tick += event.time
        if event.is_meta: continue
        match event.type:
            case 'note_on' if event.velocity != 0:
                pitch = midi_note_to_pitch(event.note)
                velocity = midi_velocity_to_velocity(event.velocity)
                active_keys[pitch] = ActiveKey(start_tick=tick, velocity=velocity)
            case 'note_off' | 'note_on':
                pitch = midi_note_to_pitch(event.note)
                try: active_key = active_keys.pop(pitch)
                except KeyError: continue
                tokens.append(Token(
                    start_tick=active_key.start_tick,
                    velocity=active_key.velocity,
                    end_tick=tick,
                    pitch=pitch,
                ))
            case _: continue
    return tokens

def tokens_to_data(tokens: list[Token], ticks_per_step: int) -> np.ndarray:
    data = np.zeros((len(tokens), 4), dtype=np.int64)
    step = 0

    for index, token in enumerate(tokens):
        duration = (token.end_tick - token.start_tick) // ticks_per_step

        start_step = token.start_tick / ticks_per_step
        quantized_start_step = round(start_step)

        duration_bias = quantized_start_step - start_step
        duration = min(round(duration + duration_bias), DURATION_COUNT - 1)

        advance = min(start_step - step, ADVANCE_COUNT - 1)
        step = max(step, start_step)

        assert token.pitch < PITCH_COUNT
        assert token.velocity < VELOCITY_COUNT
        assert advance < ADVANCE_COUNT
        assert duration < DURATION_COUNT

        data[index, :] = np.array([token.pitch, token.velocity, advance, duration])

    return data

def midi_to_tokens(file_path: str, transpose: int = 0) -> np.ndarray:
    midi = mido.MidiFile(file_path)

    # Assume that the song is in 4/4.
    steps_per_beat = SUBDIVISIONS // 4
    ticks_per_step=midi.ticks_per_beat // steps_per_beat

    tracks = []

    for track in midi.tracks:
        tokens = parse_track(track)
        tokens.sort(key=lambda token: (token.start_tick, token.pitch))
        tracks.append(tokens_to_data(tokens, ticks_per_step))

    data = np.vstack(tracks)

    min_pitch, max_pitch = np.min(data[:, 0]), np.max(data[:, 0])
    transpose = min(max(transpose, -min_pitch), PITCH_COUNT - max_pitch - 1)

    data[:, 0] -= transpose

    return data

def tokens_to_midi(file_path: str, tokens: np.ndarray):
    midi = mido.MidiFile()
    midi.ticks_per_beat = 960

    ticks_per_step = midi.ticks_per_beat // (SUBDIVISIONS // 4)

    meta_track = mido.MidiTrack()
    midi.tracks.append(meta_track)

    meta_track.append(mido.MetaMessage(
        type='set_tempo',
        tempo=mido.bpm2tempo(120),
        time=0,
    ))
    meta_track.append(mido.MetaMessage(
        type='time_signature',
        numerator=4,
        denominator=4,
        time=0,
    ))

    track = mido.MidiTrack()
    midi.tracks.append(track)

    tick = 0

    note_lift = {}

    for row in tokens:
        start = tick + row[2] * ticks_per_step

        note_off_events = sorted(
            iter((note, off_tick) for note, off_tick in note_lift.items() if off_tick < start),
            key=lambda x: x[1],
        )

        for note, off_tick in note_off_events:
            note_lift.pop(note)
            track.append(mido.Message(
                type='note_off', note=note, velocity=64, time=off_tick - tick,
            ))
            tick = off_tick

        note = pitch_to_midi_note(row[0])
        velocity = velocity_to_midi_velocity(row[1])
        duration = row[3]

        note_lift[note] = start + duration * ticks_per_step

        track.append(mido.Message(
            type='note_on', note=note, velocity=velocity, time=start - tick,
        ))

        tick = start

    midi.save(file_path)

if __name__ == "__main__":
    path = 'maestro-v3.0.0/2008/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--1.midi'
    tokens = midi_to_tokens(path, -2)
    tokens_to_midi('out.midi', tokens)