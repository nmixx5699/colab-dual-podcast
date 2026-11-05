import gc
import shutil
from .common import run


def detect_gpu(low_vram=None):
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('No NVIDIA GPU. Select L4 or A100 in Colab.')
    line = run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits']).stdout.splitlines()[0]
    name, memory = line.rsplit(',', 1)
    memory = int(memory.strip())
    if memory < 20000:
        raise RuntimeError(f'{name}: {memory} MiB; this project requires an L4 24GB or larger GPU for testing')
    return {'name': name.strip(), 'memory_mib': memory,
            'low_vram': (memory < 35000 and 'L4' not in name.upper()) if low_vram is None else bool(low_vram)}


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

