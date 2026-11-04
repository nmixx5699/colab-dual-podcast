"""Run from repository root: python -m scripts.setup_colab --cache /content/cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import venv
from src.common import digest, run, write_json
from src.tts import VOX_REV

ROOT = Path(__file__).resolve().parents[1]


def install_environment(path):
    if not (path / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(path)
    return path / 'bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='/content/podcast-cache')
    parser.add_argument('--comfy', default='/content/ComfyUI-dual')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('GPU installer targets Google Colab Linux. CPU tests are cross-platform.')
    cache, comfy = Path(args.cache).resolve(), Path(args.comfy).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / 'configs/models.lock.json').read_text())
    run(['apt-get', 'update', '-qq'])
    run(['apt-get', 'install', '-y', '-qq', 'ffmpeg', 'python3-venv'])
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
    missing_size = sum(m['size'] for m in lock['models'] if not (comfy / 'models' / m['folder'] / Path(m['file']).name).exists())
    if shutil.disk_usage(cache).free < missing_size + 8 * 1024**3:
        raise RuntimeError('Insufficient cache disk: reserve model bytes plus 8 GiB for TTS and temporary files')
    for model in lock['models']:
        destination = comfy / 'models' / model['folder'] / Path(model['file']).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == model['size'] and digest(destination) == model['sha256']:
            print('Verified cache:', destination.name, flush=True)
            continue
        cached = Path(hf_hub_download(model['repo'], model['file'], revision=model['revision'], cache_dir=str(cache / 'hf')))
        if cached.stat().st_size != model['size'] or digest(cached) != model['sha256']:
            raise RuntimeError(f'Model checksum mismatch: {cached}; remove corrupt cache entry and retry')
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(f'Unexpected existing model at {destination}; move it aside before retrying')
        destination.symlink_to(cached.resolve())
    write_json(ROOT / '.runtime.json', {'comfy': str(comfy), 'vox_python': str(vox_python),
                                      'comfy_python': str(comfy_python), 'cache': str(cache)})
    print('Setup complete. GPU generation still requires the staged acceptance tests.')


if __name__ == '__main__':
    main()
