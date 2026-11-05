import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import requests
from PIL import Image
from scripts import setup_colab as setup
from src.comfy_client import ComfyClient
from src.common import digest
from src.gpu_utils import detect_gpu
from src.masks import aspect_dimensions, prepare_image_and_masks
from src.subtitles import write_srt, burn_subtitles


@pytest.mark.parametrize('body,value', [('', {}), ('  \n', {}), ('{}', {}), ('{"ok":true}', {'ok': True})])
@pytest.mark.parametrize('method', ['get', 'post'])
def test_api_success(monkeypatch, tmp_path, body, value, method):
    response = requests.Response()
    response.status_code, response._content = 200, body.encode()
    monkeypatch.setattr('requests.' + method, lambda *a, **k: response)
    c = ComfyClient(tmp_path, 'python', tmp_path / 'log')
    assert (c.get('/free') if method == 'get' else c.post('/free', {})) == value


@pytest.mark.parametrize('status,body,match', [(400, 'bad input', '400.*bad input'), (500, 'failed', '500.*failed'), (200, 'broken', 'protocol error')])
@pytest.mark.parametrize('method', ['get', 'post'])
def test_api_errors(monkeypatch, tmp_path, status, body, match, method):
    r = requests.Response()
    r.status_code, r._content = status, body.encode()
    monkeypatch.setattr('requests.' + method, lambda *a, **k: r)
    c = ComfyClient(tmp_path, 'python', tmp_path / 'log')
    with pytest.raises(RuntimeError, match=match):
        c.get('/free') if method == 'get' else c.post('/free', {})


@pytest.mark.parametrize('name,memory,override,expected', [('NVIDIA L4', 23034, None, False), ('NVIDIA A100', 40960, None, False), ('L4', 23034, True, True), ('L4', 23034, False, False)])
def test_gpu(monkeypatch, name, memory, override, expected):
    monkeypatch.setattr('src.gpu_utils.shutil.which', lambda _: '/bin/nvidia-smi')
    monkeypatch.setattr('src.gpu_utils.run', lambda _: SimpleNamespace(stdout=f'{name}, {memory}'))
    assert detect_gpu(override)['low_vram'] is expected


def test_python313_package(monkeypatch):
    monkeypatch.setattr(setup.sys, 'version_info', SimpleNamespace(major=3, minor=13))
    assert setup.venv_package() == 'python3.13-venv'
    monkeypatch.setattr(setup.sys, 'version_info', SimpleNamespace(major=3, minor=14))
    with pytest.raises(RuntimeError, match='Unsupported Python'):
        setup.venv_package()


def test_real_venv_recovery(tmp_path, monkeypatch):
    path = tmp_path / 'venv'
    cache = tmp_path / 'models'
    cache.mkdir()
    (cache / 'keep').write_text('valid data')
    real = setup.venv.EnvBuilder.create
    def failed(self, path):
        (path / 'bin').mkdir(exist_ok=True)
        (path / 'bin/python').symlink_to(setup.sys.executable)
        raise RuntimeError('first creation failure')
    monkeypatch.setattr(setup.venv.EnvBuilder, 'create', failed)
    with pytest.raises(RuntimeError, match='first creation'):
        setup.install_environment(path)
    assert not setup.valid_environment(path)
    monkeypatch.setattr(setup.venv.EnvBuilder, 'create', real)
    python = setup.install_environment(path)
    assert setup.valid_environment(path)
    assert setup.install_environment(path) == python
    assert list(tmp_path.glob('venv.failed-*'))
    assert (cache / 'keep').read_text() == 'valid data'


def test_wrong_version_venv(tmp_path, monkeypatch):
    (tmp_path / 'bin').mkdir()
    python = tmp_path / 'bin/python'
    python.write_text('#!/bin/sh\necho 2.7\n')
    python.chmod(0o755)
    assert not setup.valid_environment(tmp_path)


def model_fixture(tmp_path):
    good = tmp_path / 'good'
    good.write_bytes(b'good model')
    return {'repo': 'example/model', 'file': 'model.bin', 'revision': '123', 'folder': 'checkpoints', 'size': good.stat().st_size, 'sha256': digest(good)}, good


def test_cached_model_reuse(tmp_path, monkeypatch):
    model, good = model_fixture(tmp_path)
    monkeypatch.setattr('huggingface_hub.try_to_load_from_cache', lambda *a, **k: str(good))
    assert setup.plan_models([model], tmp_path / 'comfy', tmp_path) == []
    download = Mock(side_effect=AssertionError('must not download'))
    assert setup.obtain_model(model, tmp_path, download) == good
    download.assert_not_called()


def test_bad_blob_redownload(tmp_path, monkeypatch):
    model, good = model_fixture(tmp_path)
    blob = tmp_path / 'hf/blobs/abc'
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b'bad')
    monkeypatch.setattr('huggingface_hub.try_to_load_from_cache', lambda *a, **k: str(blob))
    calls = []
    def download(*a, **kw):
        calls.append(kw)
        if kw['force_download']:
            blob.write_bytes(good.read_bytes())
        return str(blob)
    assert setup.obtain_model(model, tmp_path, download) == blob
    assert len(calls) == 2
    assert list(blob.parent.glob('abc.corrupt-*'))
    assert setup.valid_model(blob, model)


def test_bad_blob_limited_retry(tmp_path, monkeypatch):
    model, _ = model_fixture(tmp_path)
    monkeypatch.setattr('huggingface_hub.try_to_load_from_cache', lambda *a, **k: None)
    calls = []
    def bad(*a, **k):
        calls.append(1)
        blob = tmp_path / 'hf/bad'
        blob.parent.mkdir(exist_ok=True)
        blob.write_bytes(b'bad')
        return blob
    with pytest.raises(RuntimeError, match='after 2'):
        setup.obtain_model(model, tmp_path, bad)
    assert len(calls) == 2


@pytest.mark.parametrize('aspect', ['16:9', '9:16'])
def test_aspect_masks(tmp_path, aspect):
    width, height = aspect_dimensions(aspect)
    assert width % 16 == height % 16 == 0
    image = tmp_path / 'source.png'
    Image.new('RGB', (width, height), 'white').save(image)
    prepare_image_and_masks(image, tmp_path / 'out', width, height)
    for name in ['hosts.png', 'mask_A.png', 'mask_B.png', 'mask_preview.png']:
        assert Image.open(tmp_path / 'out' / name).size == (width, height)


def test_subtitles_actual_timing(tmp_path):
    timeline = tmp_path / 'timeline.json'
    timeline.write_text(json.dumps({'events': [{'start': .125, 'end': 1.5, 'speaker': 'A', 'text': '你好世界'}, {'start': 1.8, 'end': 3, 'speaker': 'B', 'text': '今天好嗎'}]}))
    result = write_srt(timeline, tmp_path / 'out.srt', line_chars=2).read_text()
    assert '00:00:00,125 --> 00:00:01,500' in result
    assert '你好\n世界' in result
    assert '00:00:01,800 --> 00:00:03,000' in result


def test_burn_subtitle_ffmpeg(tmp_path):
    from src.common import run, ffmpeg
    video = tmp_path / 'original.mp4'
    run([ffmpeg(), '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=480x832:r=25', '-t', '0.5', '-c:v', 'libx264', video])
    srt = tmp_path / 'caption.srt'
    srt.write_text('1\n00:00:00,000 --> 00:00:00,500\nTest caption\n')
    output = burn_subtitles(video, srt, tmp_path / 'captioned.mp4', font='DejaVu Sans')
    assert output.is_file() and video.is_file()


def test_cache_flags(tmp_path):
    assert ComfyClient(tmp_path, 'python', tmp_path / 'log', cache_strategy='default').cache_args() == []
    assert ComfyClient(tmp_path, 'python', tmp_path / 'log').cache_args() == ['--cache-none']
    assert ComfyClient(tmp_path, 'python', tmp_path / 'log', cache_strategy='low-memory').cache_args() == ['--cache-lru', '1']


def test_generation_failure_closes_server(tmp_path, monkeypatch):
    c = ComfyClient(tmp_path, 'python', tmp_path / 'log')
    monkeypatch.setattr(c, '_generate', Mock(side_effect=KeyboardInterrupt))
    close = Mock()
    monkeypatch.setattr(c, 'close', close)
    with pytest.raises(KeyboardInterrupt):
        c.generate({}, tmp_path / 'checkpoint')
    close.assert_called_once()


def test_disk_budget_cached_vs_missing(tmp_path, monkeypatch):
    model, _ = model_fixture(tmp_path)
    cache, comfy = tmp_path / 'cache', tmp_path / 'comfy'
    cache.mkdir()
    comfy.mkdir()
    budgets = []
    monkeypatch.setattr(setup.shutil, 'disk_usage', lambda p: SimpleNamespace(free=21 * 1024**3))
    setup.check_space(cache, comfy, [])  # 20 GiB reserve, no duplicate model bytes
    model['size'] = 10 * 1024**3
    with pytest.raises(RuntimeError, match='1 models missing'):
        setup.check_space(cache, comfy, [model])


def test_idle_timeout_stops_owned_process(tmp_path, monkeypatch):
    c = ComfyClient(tmp_path, 'python', tmp_path / 'log', idle_timeout=1)
    c.log_path.write_text('initial')
    monkeypatch.setattr(c, 'post', lambda *a: {'prompt_id': 'p'})
    monkeypatch.setattr(c, 'get', lambda *a: {})
    tick = iter([0, 0, 2])
    monkeypatch.setattr('src.comfy_client.time.monotonic', lambda: next(tick))
    close = Mock()
    monkeypatch.setattr(c, 'close', close)
    with pytest.raises(TimeoutError, match='No ComfyUI log progress'):
        c.generate({}, tmp_path / 'checkpoint.json')
    close.assert_called_once()


@pytest.mark.parametrize('aspect', ['16:9', '9:16'])
def test_workflow_dimensions(aspect):
    from src.comfy_workflow import build_graph, ROOT
    cfg = json.loads((ROOT / 'configs/default_config.json').read_text())
    cfg['width'], cfg['height'] = aspect_dimensions(aspect)
    graph, _, _ = build_graph(cfg, 'h.png', 'a.wav', 'b.wav', 'ma.png', 'mb.png', 'test', [f'{i}.png' for i in range(9)])
    talk = next(n['inputs'] for n in graph.values() if n['class_type'] == 'PodcastInfiniteTalkToVideo')
    assert (talk['width'], talk['height']) == aspect_dimensions(aspect)
    assert talk['motion_frame_count'] == 9
    assert 'previous_frames' in talk
