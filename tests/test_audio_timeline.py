import numpy as np
import pytest
from src.audio_timeline import build_timeline, audio_window, segment_plan


def test_turn_order_silence_and_sample_alignment():
    utterances = [({'speaker': 'A', 'text': '甲'}, np.ones(103)),
                  ({'speaker': 'B', 'text': '乙'}, np.ones(97) * 0.5),
                  ({'speaker': 'A', 'text': '丙'}, np.ones(31) * 0.2)]
    tracks, mix, events = build_timeline(utterances, 1000, 0.013)
    assert [e['start_sample'] for e in events] == [0, 116, 226]
    assert len(mix) == 257
    assert not np.any(tracks['A'] * tracks['B'])
    assert np.max(mix) <= 0.951
    assert np.all(tracks['A'][116:213] == 0)
    assert np.all(tracks['B'][:116] == 0)
    np.testing.assert_array_equal(mix, tracks['A'] + tracks['B'])


@pytest.mark.parametrize('seconds', [5, 10, 15, 30, 60])
def test_segment_coverage_no_drift(seconds):
    plan = segment_plan(seconds * 48000, 48000)
    frames = []
    for seg in plan:
        first = seg['audio_start_frame'] + seg['trim']
        frames.extend(range(first, first + seg['take']))
    assert frames == list(range(seconds * 25))
    assert plan[0]['trim'] == 0
    assert all(s['trim'] == 9 for s in plan[1:])


def test_fractional_end_and_late_audio_window():
    plan = segment_plan(48001, 48000)
    assert sum(s['take'] for s in plan) == 26
    wav = np.arange(16000 * 10, dtype=np.float32)
    window = audio_window(wav, 16000, 144)
    assert window[0] == 92160
    assert len(window) == 51840
    window = audio_window(wav, 16000, 249)
    assert np.count_nonzero(window) == 640


@pytest.mark.parametrize('wav', [[], [float('nan')], [0, 0], [[1, 2]]])
def test_bad_tts_rejected(wav):
    with pytest.raises(ValueError):
        build_timeline([({'speaker': 'A'}, wav)], 48000)
