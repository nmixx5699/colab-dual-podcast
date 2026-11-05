"""Orchestrator integration with synthetic backend; NOT an AI inference test."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import soundfile as sf
from src.common import ffmpeg, run
from src.comfy_workflow import ROOT
from src.video_generator import generate


@pytest.mark.parametrize('cleanup_bad', [False, True])
def test_failed_second_segment_resumes_and_stages_inputs(tmp_path, monkeypatch, cleanup_bad):
    if not shutil.which('ffmpeg'):
        try:
            import imageio_ffmpeg
            monkeypatch.setenv('FFMPEG_BINARY', imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pytest.skip('FFmpeg unavailable')
    inputs, audio = tmp_path / 'inputs', tmp_path / 'audio'
    inputs.mkdir()
    audio.mkdir()
    for name in ('hosts.png', 'mask_A.png', 'mask_B.png'):
        Image.new('RGB', (64, 48), 'white').save(inputs / name)
    t = np.arange(80000) / 16000
    wave = np.sin(2 * np.pi * 220 * t) * 0.2
    for name in ('A', 'B', 'mix'):
        sf.write(audio / f'{name}.wav', wave, 16000)

    class Backend:
        def __init__(self):
            self.root = tmp_path / 'comfy'
            (self.root / 'output').mkdir(parents=True)
            self.calls = 0
            self.fail_second = True

        def get(self, route):
            assert route == '/object_info'
            staged = list((self.root / 'input').iterdir())
            assert any(p.suffix == '.wav' for p in staged), 'Schema must be queried AFTER staging'
            return {}

        def post(self, *args):
            from src.comfy_client import ComfyClient
            import requests
            response = requests.Response()
            response.status_code = 200
            response._content = b'non-json' if cleanup_bad else b''
            return ComfyClient.decode(response, '/free')

        def output_file(self, item):
            return self.root / 'output' / item['filename']

        def generate(self, graph, checkpoint, timeout):
            self.calls += 1
            for node in graph.values():
                if node['class_type'] in ('LoadImage', 'LoadImageMask', 'LoadAudio'):
                    name = node['inputs'].get('image', node['inputs'].get('audio'))
                    assert '/' not in name
                    assert (self.root / 'input' / name).exists()
            if self.fail_second and self.calls == 2:
                raise RuntimeError('simulated interruption')
            output = self.root / 'output'
            raw = output / f'{self.calls}.mp4'
            run([ffmpeg(), '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=green:s=64x48:r=25',
                 '-frames:v', '81', '-c:v', 'libx264', raw])
            tail = []
            for j in range(9):
                p = output / f'{self.calls}_{j}.png'
                Image.new('RGB', (64, 48), 'green').save(p)
                tail.append({'filename': p.name})
            video_id = next(k for k, n in graph.items() if n['class_type'] == 'SaveVideo')
            tail_id = next(k for k, n in graph.items() if n['class_type'] == 'SaveImage')
            return {video_id: {'images': [{'filename': raw.name}]}, tail_id: {'images': tail}}

    # Runtime schema/inference are outside this fake-backend test; separately audited and validated live.
    monkeypatch.setattr('src.video_generator.validate_graph', lambda *args: True)
    cfg = json.loads((ROOT / 'configs/default_config.json').read_text())
    client = Backend()
    output = tmp_path / 'result'
    with pytest.raises(RuntimeError, match='simulated interruption'):
        generate(client, cfg, inputs, audio, output)
    assert (output / 'segment_000/complete.json').exists()
    assert not (output / 'segment_001/complete.json').exists()
    client.fail_second = False
    assert generate(client, cfg, inputs, audio, output).is_file()
    assert client.calls == 3  # first completed segment was not regenerated
    generate(client, cfg, inputs, audio, output)
    assert client.calls == 3  # both checkpoints reused
    (output / 'segment_000/clip.mp4').write_bytes(b'corrupt')
    generate(client, cfg, inputs, audio, output)
    assert client.calls == 5  # corruption invalidated the dependent motion chain

