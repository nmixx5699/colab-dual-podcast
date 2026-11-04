import math
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import write_json


def build_timeline(utterances, sample_rate, pause_sec=0.3):
    """Schedule on integer sample indices; inactive speaker is exact digital silence."""
    if sample_rate <= 0 or not math.isfinite(pause_sec) or pause_sec < 0:
        raise ValueError('Invalid sample rate or pause')
    if not utterances:
        raise ValueError('No utterances')
    pause = round(pause_sec * sample_rate)
    cursor, events, checked = 0, [], []
    for i, (row, wav) in enumerate(utterances):
        wav = np.asarray(wav, dtype=np.float32)
        if row['speaker'] not in ('A', 'B') or wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise ValueError('Invalid speaker or audio waveform')
        if np.max(np.abs(wav)) == 0:
            raise ValueError('TTS produced silent audio')
        if i:
            cursor += pause
        events.append({**row, 'start_sample': cursor, 'end_sample': cursor + len(wav),
                       'start': cursor / sample_rate, 'end': (cursor + len(wav)) / sample_rate})
        checked.append(wav)
        cursor += len(wav)
    tracks = {s: np.zeros(cursor, dtype=np.float32) for s in ('A', 'B')}
    for event, wav in zip(events, checked):
        tracks[event['speaker']][event['start_sample']:event['end_sample']] = wav
    peak = max(float(np.max(np.abs(x))) for x in tracks.values())
    gain = min(1.0, 0.95 / peak)
    tracks = {s: x * gain for s, x in tracks.items()}
    return tracks, tracks['A'] + tracks['B'], events


def save_timeline(directory, tracks, mix, events, sample_rate):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for speaker, wav in {**tracks, 'mix': mix}.items():
        sf.write(directory / f'{speaker}.wav', wav, sample_rate, subtype='PCM_16')
    write_json(directory / 'timeline.json', {'sample_rate': sample_rate, 'samples': len(mix), 'events': events})


def audio_window(wav, sample_rate, start_frame, length=81, fps=25):
    """Frame-aligned window with zero padding, including motion overlap."""
    if start_frame < 0 or length <= 0 or fps <= 0:
        raise ValueError('Invalid audio window')
    start = round(start_frame * sample_rate / fps)
    size = round(length * sample_rate / fps)
    out = np.zeros(size, dtype=np.float32)
    piece = wav[start:start + size]
    out[:len(piece)] = piece
    return out


def segment_plan(samples, sample_rate, fps=25, chunk=81, motion=9):
    if samples <= 0 or sample_rate <= 0 or fps != 25 or chunk % 4 != 1 or not 0 < motion < chunk:
        raise ValueError('Invalid segment geometry (InfiniteTalk requires 25 fps and 4n+1 frames)')
    target = math.ceil(samples * fps / sample_rate)
    result, written = [], 0
    while written < target:
        trim = motion if result else 0
        take = min(chunk - trim, target - written)
        result.append({'index': len(result), 'audio_start_frame': max(0, written - trim),
                       'trim': trim, 'take': take, 'output_start': written, 'length': chunk})
        written += take
    return result
