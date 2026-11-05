import os
from pathlib import Path
import shutil
import numpy as np
import pytest
import soundfile as sf
from src.common import ffmpeg, run
from src.video_merge import trim_segment, merge_segments


@pytest.mark.parametrize('width,height', [(64,48), (832,480), (480,832)])
def test_ffmpeg_trim_concat_and_audio_duration(tmp_path, monkeypatch, width, height):
    if not shutil.which('ffmpeg'):
        try:
            import imageio_ffmpeg
            monkeypatch.setenv('FFMPEG_BINARY', imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pytest.skip('Install FFmpeg or imageio-ffmpeg for integration test')
    clips = []
    for i, color in enumerate(('red', 'blue')):
        raw, clip = tmp_path / f'raw{i}.mp4', tmp_path / f'clip{i}.mp4'
        run([ffmpeg(), '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=c={color}:s={width}x{height}:r=25',
             '-frames:v', '81', '-c:v', 'libx264', raw])
        trim_segment(raw, clip, 0 if i == 0 else 9, 81 if i == 0 else 44)
        clips.append(clip)
    t = np.arange(80000) / 16000
    mix = tmp_path / 'mix.wav'
    sf.write(mix, np.sin(2 * np.pi * 440 * t) * 0.2, 16000)
    result = merge_segments(clips, mix, tmp_path / 'podcast.mp4', 5)
    # Decode actual output and count bytes: 125 RGB frames, not merely successful exit.
    import subprocess
    frames = subprocess.run([ffmpeg(), '-v', 'error', '-i', str(result), '-map', '0:v:0',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    assert len(frames) == 125 * width * height * 3
    pixels = np.frombuffer(frames, np.uint8).reshape(125, height, width, 3)
    assert pixels[80, :, :, 0].mean() > 200
    assert pixels[81, :, :, 2].mean() > 200
    audio = tmp_path / 'decoded.wav'
    run([ffmpeg(), '-v', 'error', '-y', '-i', result, '-vn', '-ac', '1', '-ar', '16000', audio])
    samples, sr = sf.read(audio)
    assert abs(len(samples) / sr - 5) < 0.07  # AAC packet padding
    assert np.max(np.abs(samples)) > 0.1

