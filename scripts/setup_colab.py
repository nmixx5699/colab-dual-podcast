"""Run from repository root: python -m scripts.setup_colab --cache /content/cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import venv
import os
import uuid
from src.common import digest, run, write_json
from src.tts import VOX_REV

ROOT = Path(__file__).resolve().parents[1]


def venv_package():
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor not in (12, 13):
        raise RuntimeError(f'Unsupported Python {major}.{minor}; supported installer versions: 3.12, 3.13')
    return f'python{major}.{minor}-venv'


def valid_environment(path):
    python = Path(path) / 'bin/python'
    if not (Path(path) / 'pyvenv.cfg').is_file() or not python.is_file() or not os.access(python, os.X_OK) or (Path(path) / '.creating').exists():
        return False
    try:
        version = run([python, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")']).stdout.strip()
        if version != f'{sys.version_info.major}.{sys.version_info.minor}':
            return False
        run([python, '-m', 'ensurepip', '--version'])
        run([python, '-m', 'pip', '--version'])
        return True
    except (OSError, RuntimeError):
        return False


def install_environment(path):
    path = Path(path)
    venv_package()
    if valid_environment(path):
        return path / 'bin/python'
    # Quarantine this venv only, never its parent or any model cache.
    if path.is_symlink():
        raise RuntimeError(f'Refusing symlink venv: {path}')
    if path.exists():
        path.rename(path.with_name(path.name + '.failed-' + uuid.uuid4().hex))
    path.mkdir(parents=True)
    marker = path / '.creating'
    marker.touch()
    try:
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(path)
        marker.unlink()
        if not valid_environment(path):
            raise RuntimeError('venv Python/ensurepip/pip validation failed')
    except BaseException:
        marker.touch()
        raise
    return path / 'bin/python'


def valid_model(path, model):
    path = Path(path)
    return path.is_file() and path.stat().st_size == model['size'] and digest(path) == model['sha256']


def cached_model(model, cache):
    from huggingface_hub import try_to_load_from_cache
    value = try_to_load_from_cache(model['repo'], model['file'], revision=model['revision'], cache_dir=str(cache / 'hf'))
    return Path(value) if isinstance(value, str) and valid_model(value, model) else None


def quarantine(path):
    path = Path(path)
    if path.is_symlink():
        path.unlink()  # Only detach destination, never remove an arbitrary symlink target.
    elif path.exists():
        path.rename(path.with_name(path.name + '.corrupt-' + uuid.uuid4().hex))


def obtain_model(model, cache, download, attempts=2):
    cached = cached_model(model, cache)
    if cached:
        return cached
    for attempt in range(attempts):
        value = Path(download(model['repo'], model['file'], revision=model['revision'],
                              cache_dir=str(cache / 'hf'), force_download=attempt > 0))
        if valid_model(value, model):
            return value
        # Only quarantine the identified HF blob within our configured cache.
        blob = value.resolve()
        if not blob.is_relative_to((cache / 'hf').resolve()):
            raise RuntimeError(f'Corrupt model outside configured cache: {value}')
        quarantine(blob)
    raise RuntimeError(f'Model checksum/size mismatch after {attempts} downloads: {model["file"]}')


def plan_models(models, comfy, cache):
    missing = []
    for model in models:
        destination = comfy / 'models' / model['folder'] / Path(model['file']).name
        if not valid_model(destination, model) and not cached_model(model, cache):
            missing.append(model)
    return missing


def check_space(cache, comfy, missing):
    # HF writes blobs and download temporaries on cache FS; models are symlinked.
    cache_need = sum(m['size'] for m in missing) + max([m['size'] for m in missing] or [0]) + 8 * 1024**3
    install_need = 12 * 1024**3  # packages and audio/video intermediate reserve
    install_fs = comfy if comfy.exists() else comfy.parent
    install_fs.mkdir(parents=True, exist_ok=True)
    same_fs = cache.stat().st_dev == install_fs.stat().st_dev
    requirements = [(cache, cache_need + (install_need if same_fs else 0))]
    if not same_fs:
        requirements.append((install_fs, install_need))
    for path, needed in requirements:
        if shutil.disk_usage(path).free < needed:
            raise RuntimeError(f'Insufficient disk at {path}: need {needed / 1024**3:.1f} GiB; {len(missing)} models missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='/content/podcast-cache')
    parser.add_argument('--comfy', default='/content/ComfyUI-dual')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('GPU installer targets Google Colab Linux. CPU tests are cross-platform.')
    venv_package()
    cache, comfy = Path(args.cache).resolve(), Path(args.comfy).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    check_space(cache, comfy, plan_models(json.loads((ROOT / 'configs/models.lock.json').read_text())['models'], comfy, cache))
    lock = json.loads((ROOT / 'configs/models.lock.json').read_text())
    venv_package()
    run(['apt-get', 'update', '-qq'])
    run(['apt-get', 'install', '-y', '-qq', 'ffmpeg', venv_package()])
    if not comfy.exists():
        run(['git', 'clone', '--depth', '1', '--branch', lock['comfy_tag'], 'https://github.com/Comfy-Org/ComfyUI.git', comfy])
    head = run(['git', '-C', comfy, 'rev-parse', 'HEAD']).stdout.strip()
    if head != lock['comfy_commit']:
        raise RuntimeError('Existing ComfyUI checkout has wrong revision. Choose another --comfy directory.')
    source = (comfy / 'comfy_extras/nodes_wan.py').read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest() != lock['upstream_node_sha256_lf']:
        raise RuntimeError('ComfyUI node source differs from audited version')
    # Add independent GPL derivative; never edit the original nodes_wan.py.
    shutil.copytree(ROOT / 'custom_nodes/dual_podcast', comfy / 'custom_nodes/dual_podcast', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    vox_python = install_environment(Path('/content/venv-podcast-vox'))
    comfy_python = install_environment(Path('/content/venv-podcast-comfy'))
    # Separate dependency resolvers, while reusing Colab's CUDA PyTorch installation.
    for python in (vox_python, comfy_python):
        run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt'])
    run([vox_python, '-m', 'pip', 'install', f'git+https://github.com/OpenBMB/VoxCPM.git@{VOX_REV}'])
    run([comfy_python, '-m', 'pip', 'install', '-r', comfy / 'requirements.txt'])
    for name, python in [('vox', vox_python), ('comfy', comfy_python)]:
        # Record resolved versions: upstream does not fully lock transitive dependencies.
        (cache / f'{name}-pip-freeze.txt').write_text(run([python, '-m', 'pip', 'freeze']).stdout)
        run([python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable after install"; print(torch.__version__)'])
    from huggingface_hub import hf_hub_download
    check_space(cache, comfy, plan_models(lock['models'], comfy, cache))
    for model in lock['models']:
        destination = comfy / 'models' / model['folder'] / Path(model['file']).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if valid_model(destination, model):
            print('Verified final model:', destination.name, flush=True)
            continue
        cached = obtain_model(model, cache, hf_hub_download)
        quarantine(destination)
        destination.symlink_to(cached.resolve())
    write_json(ROOT / '.runtime.json', {'comfy': str(comfy), 'vox_python': str(vox_python),
                                      'comfy_python': str(comfy_python), 'cache': str(cache)})
    print('Setup complete. GPU generation still requires the staged acceptance tests.')


if __name__ == '__main__':
    main()

