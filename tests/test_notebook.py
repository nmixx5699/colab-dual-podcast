import json
from src.comfy_workflow import ROOT


def test_notebook_all_cells_parse_without_outputs_or_secrets():
    notebook = json.loads((ROOT / 'podcast_dual.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            compile(source, f'notebook cell {i}', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None
            assert 'ghp_' not in source
