"""Read safetensors headers through HTTP Range; never download entire model weights."""
import json
from pathlib import Path
import struct
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from src.common import write_json
from src.comfy_workflow import ROOT


def read_range(url, start, end):
    # Distinct query avoids intermediary caches returning a prior byte range.
    req = urllib.request.Request(url + f'?download=true&header_range={start}-{end}',
                                 headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f'Host did not honor Range ({response.status}); refusing full download')
        payload = response.read(end - start + 2)
    if len(payload) != end - start + 1:
        raise RuntimeError('Unexpected Range response length')
    return payload


def inspect_model(model):
    length = struct.unpack('<Q', read_range(model['url'], 0, 7))[0]
    if not 0 < length < 4 * 1024**2:
        raise ValueError('Unexpected safetensors header size')
    header = json.loads(read_range(model['url'], 8, 7 + length))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    result = {'file': model['file'], 'revision': model['revision'], 'header_bytes': length,
              'tensor_count': len(tensors), 'dtypes': dict(Counter(v['dtype'] for v in tensors.values()))}
    if model['folder'] == 'model_patches':
        # Pinned ModelPatchLoader uses these exact keys to select and size MultiTalkModelPatch.
        required = ['audio_proj.proj1.weight', 'audio_proj.norm.weight', 'blocks.0.audio_cross_attn.proj.weight']
        for key in required:
            if key not in tensors:
                raise ValueError(f'InfiniteTalk weight missing loader discriminator {key}')
        result['loader_discriminators'] = {k: tensors[k] for k in required}
        if not any(v['dtype'] == 'F16' for v in tensors.values()):
            raise ValueError('Expected FP16 tensors in Multi patch')
        # The official *fp16 filename contains F32/BF16 tensors too; preserve its mixed precision.
        result['precision_note'] = 'Official FP16-labelled file contains mixed F32/F16/BF16 tensors; no forced recast.'
    return result


if __name__ == '__main__':
    models = json.loads((ROOT / 'configs/models.lock.json').read_text())['models']
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(inspect_model, models))
    write_json(ROOT / 'docs/model_header_audit.json', {'results': results,
        'scope': 'HTTP safetensors headers only; runtime verifies full SHA-256, inference NOT tested'})
    print(json.dumps(results, indent=2))
