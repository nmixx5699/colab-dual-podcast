import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.comfy_workflow import ROOT, build_graph, expand_inputs, validate_graph
from src.common import digest, write_json
from src.video_generator import valid_checkpoint


def config():
    return json.loads((ROOT / 'configs/default_config.json').read_text())


def test_two_tracks_dynamic_keys_and_previous_frames():
    graph, video, tail = build_graph(config(), 'h.png', 'a.wav', 'b.wav', 'a.png', 'b.png', 'test', [f'{i}.png' for i in range(9)])
    talk = next(n['inputs'] for n in graph.values() if n['class_type'] == 'PodcastInfiniteTalkToVideo')
    assert talk['mode'] == 'two_speakers'
    assert talk['mode.audio_encoder_output_2'] != talk['audio_encoder_output_1']
    assert talk['mode.mask_1'] != talk['mode.mask_2']
    assert 'previous_frames' in talk
    assert graph[video]['class_type'] == 'SaveVideo'
    assert graph[tail]['class_type'] == 'SaveImage'
    assert sum(n['class_type'] == 'ImageBatch' for n in graph.values()) == 8
    assert all('single_fp16' not in str(n) for n in graph.values())


def test_parallel_audio_branch_executes_from_actual_node_source():
    # Execute the actual derivative's audio-combination AST on arrays, without loading GPU models.
    tree = ast.parse((ROOT / 'custom_nodes/dual_podcast/__init__.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'multi_audio_type' for t in n.targets))
    branch = fn.body[fn.body.index(assignment) + 1]
    a, b = np.array([[1.], [0.], [3.], [0.]]), np.array([[0.], [2.], [0.], [4.]])
    scope = {'encoded_audio_list': [a, b], 'seq_lengths': [4, 4],
             'torch': SimpleNamespace(zeros=lambda *shape, dtype=None: np.zeros(shape, dtype=dtype),
                                     cat=lambda pieces, dim=0: np.concatenate(pieces, axis=dim))}
    exec(compile(ast.Module(body=[assignment, branch], type_ignores=[]), '<actual audio alignment>', 'exec'), scope)
    np.testing.assert_array_equal(scope['encoded_audio_list'][0], a)
    np.testing.assert_array_equal(scope['encoded_audio_list'][1], b)
    assert len(scope['encoded_audio_list'][1]) == 4  # upstream add mode would give 8 and shift B by 4


def test_dynamic_schema_validation_and_missing_node():
    schema = {'required': {'mode': ['COMFY_DYNAMICCOMBO_V3', {'options': [
        {'key': 'two_speakers', 'inputs': {'optional': {'mask_1': ['MASK', {}]}}}]}]}}
    expanded = expand_inputs(schema, {'mode': 'two_speakers'})
    assert 'mode.mask_1' in expanded
    with pytest.raises(ValueError, match='Missing node'):
        validate_graph({'1': {'class_type': 'bad', 'inputs': {}}}, {})
    with pytest.raises(ValueError, match='unknown input'):
        validate_graph({'1': {'class_type': 'talk', 'inputs': {'mode': 'two_speakers', 'mask_1': 'wrong'}}},
                       {'talk': {'input': schema, 'output': []}})


def test_resume_rejects_corruption_and_changed_config(tmp_path):
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'example checkpoint fixture')
    checkpoint = tmp_path / 'complete.json'
    write_json(checkpoint, {'signature': 'good', 'files': {'clip.mp4': digest(clip)}})
    assert valid_checkpoint(checkpoint, 'good')
    assert not valid_checkpoint(checkpoint, 'changed')
    clip.write_bytes(b'broken')
    assert not valid_checkpoint(checkpoint, 'good')


def test_all_model_sources_are_pinned_and_hashed():
    lock = json.loads((ROOT / 'configs/models.lock.json').read_text())
    assert len(lock['models']) == 6
    for model in lock['models']:
        assert len(model['revision']) == 40 and len(model['sha256']) == 64 and model['size'] > 0
