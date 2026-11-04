"""Source-level audit only; does not claim GPU/model execution validation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from src.comfy_workflow import build_graph, ROOT
from src.common import write_json


def audit(comfy, vox):
    comfy, vox = Path(comfy), Path(vox)
    lock = json.loads((ROOT / 'configs/models.lock.json').read_text())
    source = (comfy / 'comfy_extras/nodes_wan.py').read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest() == lock['upstream_node_sha256_lf']
    upstream = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'WanInfiniteTalkToVideo')
    derivative = (ROOT / 'custom_nodes/dual_podcast/__init__.py').read_text(encoding='utf-8')
    custom = next(n for n in ast.parse(derivative).body if isinstance(n, ast.ClassDef) and n.name == 'PodcastInfiniteTalkToVideo')
    original = ast.get_source_segment(source, upstream)
    adapted = ast.get_source_segment(derivative, custom).replace('PodcastInfiniteTalkToVideo', 'WanInfiniteTalkToVideo')
    adapted = adapted.replace('multi_audio_type = "para"', 'multi_audio_type = "add"')
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(adapted)), 'Unexpected derivative changes'
    all_sources = list(comfy.glob('nodes.py')) + list((comfy / 'comfy_extras').glob('*.py'))
    classes = {}
    for path in all_sources:
        text = path.read_text(encoding='utf-8-sig')
        for item in ast.parse(text).body:
            if isinstance(item, ast.ClassDef):
                classes[item.name] = (path, item)
    config = json.loads((ROOT / 'configs/default_config.json').read_text())
    graph, _, _ = build_graph(config, 'hosts.png', 'A.wav', 'B.wav', 'mask_A.png', 'mask_B.png', 'test', [f'{i}.png' for i in range(9)])
    verified = []
    for item in graph.values():
        name = item['class_type']
        if name == 'PodcastInfiniteTalkToVideo':
            node = custom
            path = ROOT / 'custom_nodes/dual_podcast/__init__.py'
        else:
            path, node = classes[name]
        strings = {x.value for x in ast.walk(node) if isinstance(x, ast.Constant) and isinstance(x.value, str)}
        for key in item['inputs']:
            # Dynamic keys: child declaration can live in a helper (SaveVideo codec).
            terminal = key.split('.')[-1]
            assert terminal in strings or (name == 'SaveVideo' and key == 'format.codec'), (name, key)
        verified.append(name)
    io_source = (comfy / 'comfy_api/latest/_io.py').read_text(encoding='utf-8')
    assert 'return ".".join(prefix_list)' in io_source
    assert 'COMFY_DYNAMICCOMBO_V3' in io_source
    vox_tree = ast.parse((vox / 'src/voxcpm/core.py').read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(vox_tree) if isinstance(n, ast.FunctionDef) and n.name == '_generate')
    params = [a.arg for a in method.args.args]
    assert {'text', 'seed', 'reference_wav_path', 'cfg_value', 'inference_timesteps'} <= set(params)
    return {'source_audit': 'passed', 'comfy_commit': lock['comfy_commit'],
            'nodes_checked': sorted(set(verified)), 'tts_parameters': params,
            'dynamic_inputs': ['mode.audio_encoder_output_2', 'mode.mask_1', 'mode.mask_2'],
            'derivative_changes': ['independent class/node id', 'parallel audio instead of concatenation'],
            'runtime_object_info_validation': 'performed in Colab before each prompt; NOT executed by this audit',
            'gpu_inference': 'NOT TESTED'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy', required=True)
    parser.add_argument('--vox', required=True)
    parser.add_argument('--output', default='docs/source_audit.json')
    args = parser.parse_args()
    result = audit(args.comfy, args.vox)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
