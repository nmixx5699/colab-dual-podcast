from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def validate_masks(a, b, size):
    a, b = np.asarray(a), np.asarray(b)
    expected = (size[1], size[0])
    if a.shape != expected or b.shape != expected:
        raise ValueError(f'Masks must have shape {expected}')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Masks must be finite')
    a, b = a > 0, b > 0
    if not a.any() or not b.any() or np.any(a & b):
        raise ValueError('Masks must be nonempty and nonoverlapping')
    if np.where(a)[1].mean() >= np.where(b)[1].mean():
        raise ValueError('A mask must be left of B mask')
    return (a.astype(np.uint8) * 255, b.astype(np.uint8) * 255)


def prepare_image_and_masks(image_path, output, width=832, height=480, split=0.5, custom=None):
    if width % 16 or height % 16 or min(width, height) < 16 or not 0 < split < 1:
        raise ValueError('Dimensions must be multiples of 16; split must be within (0,1)')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Letterbox preserves both people instead of cropping either face off-screen.
    image = ImageOps.pad(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'),
                         (width, height), method=Image.Resampling.LANCZOS, color=(32, 32, 32))
    image.save(output / 'hosts.png')
    if custom:
        a, b = [np.asarray(Image.open(p).convert('L')) for p in custom]
    else:
        a = np.zeros((height, width), dtype=np.uint8)
        a[:, :round(width * split)] = 255
        b = 255 - a
    a, b = validate_masks(a, b, (width, height))
    for name, mask in [('mask_A.png', a), ('mask_B.png', b)]:
        Image.fromarray(mask).save(output / name)
    overlay = np.asarray(image).copy().astype(float)
    overlay[a > 0] = overlay[a > 0] * 0.65 + np.array([255, 60, 60]) * 0.35
    overlay[b > 0] = overlay[b > 0] * 0.65 + np.array([60, 120, 255]) * 0.35
    Image.fromarray(overlay.astype('uint8')).save(output / 'mask_preview.png')
    return output / 'hosts.png'
