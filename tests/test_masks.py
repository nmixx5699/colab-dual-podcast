import numpy as np
import pytest
from PIL import Image
from src.masks import prepare_image_and_masks, validate_masks


def test_masks_image_letterbox(tmp_path):
    original = tmp_path / 'source.png'
    Image.new('RGB', (300, 200), 'white').save(original)
    prepare_image_and_masks(original, tmp_path / 'inputs', 832, 480)
    a, b = [np.array(Image.open(tmp_path / 'inputs' / f'mask_{s}.png')) for s in 'AB']
    assert np.all(a[:, :416] == 255) and np.all(b[:, 416:] == 255)
    assert not np.any((a > 0) & (b > 0))
    assert Image.open(tmp_path / 'inputs/hosts.png').size == (832, 480)


def test_overlap_empty_and_swapped_rejected():
    a = np.array([[255, 0], [255, 0]])
    b = 255 - a
    for pair in ((a, a), (a * 0, b), (b, a)):
        with pytest.raises(ValueError):
            validate_masks(*pair, (2, 2))
