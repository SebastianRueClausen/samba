from dataclasses import dataclass
import math

from torch import LongTensor
import mido

Token = int

SUBDIVISIONS = 64

META_TOKEN_COUNT = 2
ADVANCE_TOKEN_COUNT = int(math.log2(SUBDIVISIONS)) + 1
VELOCITY_TOKEN_COUNT = 16
KEY_ON_TOKEN_COUNT = 88
KEY_OFF_TOKEN_COUNT = 88
PEDAL_TOKEN_COUNT = 2

ADVANCE_TOKEN_START = META_TOKEN_COUNT
VELOCITY_TOKEN_START = ADVANCE_TOKEN_START + ADVANCE_TOKEN_COUNT
KEY_ON_TOKEN_START = VELOCITY_TOKEN_START + VELOCITY_TOKEN_COUNT
KEY_OFF_TOKEN_START = KEY_ON_TOKEN_START + KEY_ON_TOKEN_COUNT
PEDAL_TOKEN_START = KEY_OFF_TOKEN_START + KEY_OFF_TOKEN_COUNT

VOCAB_SIZE = PEDAL_TOKEN_START + PEDAL_TOKEN_COUNT

TOKEN_NAMES = ['<pad>', '<next_track>'] \
    + [f'advance_{pow(2, i)}' for i in range(ADVANCE_TOKEN_COUNT)] \
    + [f'velocity_{i}' for i in range(VELOCITY_TOKEN_COUNT)] \
    + [f'key_on_{i}' for i in range(KEY_ON_TOKEN_COUNT)] \
    + [f'key_off_{i}' for i in range(KEY_OFF_TOKEN_COUNT)] \
    + ['pedal_off', 'pedal_on']

def midi_note_to_key(note: int) -> int:
    return note - 21

def key_to_midi_note(key: int) -> int:
    return key + 21

def midi_velocity_to_velocity(velocity: int) -> int:
    assert velocity < 128
    return round(velocity / (128 // VELOCITY_TOKEN_COUNT))

def velocity_to_midi_velocity(velocity: int) -> int:
    return velocity * (128 // VELOCITY_TOKEN_COUNT)


@dataclass
class ParseState:
    ticks_per_step: int
    # The current step into the track, i.e. the number 32nd notes
    # (or whatever the minimum subdivision is) into the track. May be slightly
    # behind or ahead of tick.
    step: int = 0
    # The current tick into the track
    tick: int = 0
    # The velocity of the last key pressed.
    velocity: int = 0
    # True if the damper pedal is pressed, False otherwise.
    pedal: bool = False
    tokens = [1]
    # Holds the fraction of a 32nd note each plaing key was offset by when
    # pressed. This is an attempt to retain the duration of each note as much
    # as possible.
    quantize_bias = {}

    def advance_ticks(self, ticks: int, key: int | None = None, set_bias: bool = False):
        self.tick += ticks

        bias = 0.0
        if key is not None:
            if key in self.quantize_bias and not set_bias:
                bias = self.quantize_bias.pop(key)

        step = self.tick / self.ticks_per_step
        quantized_step = round(step + bias)

        if set_bias:
            self.quantize_bias[key] = quantized_step - step

        for subdivision in range(ADVANCE_TOKEN_COUNT - 1, -1, -1):
            steps = pow(2, subdivision)

            while quantized_step - self.step >= steps:
                self.tokens.append(ADVANCE_TOKEN_START + subdivision)
                self.step += steps

    def key_on(self, key: int, velocity: int):
        velocity = midi_velocity_to_velocity(velocity)

        if self.velocity != velocity:
            self.tokens.append(VELOCITY_TOKEN_START + velocity)

        self.tokens.append(KEY_ON_TOKEN_START + key)

    def key_off(self, key: int):
        self.tokens.append(KEY_OFF_TOKEN_START + key)

    def set_pedal(self, pedal: bool):
        if self.pedal != pedal:
            self.tokens.append(PEDAL_TOKEN_START + pedal)
            self.pedal = pedal


def midi_to_tokens(file_path: str) -> LongTensor:
    """ Load a midi file and convert it to a tensor of tokens. """

    midi = mido.MidiFile(file_path)
    state = ParseState(ticks_per_step=midi.ticks_per_beat // (SUBDIVISIONS // 4))

    for track in midi.tracks:
        time_delta = 0

        for event in track:
            time_delta += event.time

            if event.is_meta:
                continue

            match event.type:
                case 'note_on':
                    key = midi_note_to_key(event.note)
                    if event.velocity == 0:
                        state.advance_ticks(time_delta, key); time_delta = 0
                        state.key_off(key)
                    else:
                        state.advance_ticks(time_delta, key, set_bias=True); time_delta = 0
                        state.key_on(key, event.velocity)
                case 'note_off':
                    key = midi_note_to_key(event.note)
                    state.advance_ticks(time_delta, key); time_delta = 0
                    state.key_off(key)
                case 'control_change' if event.control == 64:
                    state.advance_ticks(time_delta); time_delta = 0
                    state.set_pedal(True if event.value >= 64 else False)
                case _:
                    continue

    return LongTensor(state.tokens)

def tokens_to_midi(file_path: str, tokens: LongTensor):
    """ Converts a tensor of tokens into midi and saves to to a file """

    midi = mido.MidiFile()

    midi.ticks_per_beat = 960

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

    time = 0
    last_event_time = 0
    velocity = 0

    for token in tokens.tolist():
        if token < VELOCITY_TOKEN_START and token >= ADVANCE_TOKEN_START:
            time += pow(2, token - ADVANCE_TOKEN_START) \
                * (midi.ticks_per_beat // (SUBDIVISIONS // 4))
        elif token < KEY_ON_TOKEN_START:
            velocity = velocity_to_midi_velocity(token - VELOCITY_TOKEN_START)
        elif token < KEY_OFF_TOKEN_START:
            track.append(mido.Message(
                type='note_on',
                note=key_to_midi_note(token - KEY_ON_TOKEN_START),
                velocity=velocity,
                time=time - last_event_time,
            ))
            last_event_time = time
        elif token < PEDAL_TOKEN_START:
            track.append(mido.Message(
                type='note_off',
                note=key_to_midi_note(token - KEY_OFF_TOKEN_START),
                velocity=64,
                time=time - last_event_time,
            ))
            last_event_time = time
        else:
            track.append(mido.Message(
                type='control_change',
                control=64,
                value=0 if token - PEDAL_TOKEN_START == 0 else 127,
            ))

    midi.save(file_path)

if __name__ == "__main__":
    tokens = midi_to_tokens('maestro-v3.0.0/2008/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--1.midi')
    tokens_to_midi('out.midi', tokens)
