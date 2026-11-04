from pathlib import Path
import os
from .common import ffmpeg, run


def trim_segment(raw, output, trim, take):
    output = Path(output)
    temporary = output.with_suffix('.tmp.mp4')
    run([ffmpeg(), '-v', 'error', '-y', '-i', raw, '-vf',
         f'trim=start_frame={trim}:end_frame={trim + take},setpts=PTS-STARTPTS',
         '-an', '-r', '25', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', temporary])
    os.replace(temporary, output)


def merge_segments(segments, mix, output, duration):
    if not segments or duration <= 0:
        raise ValueError('No segments or invalid duration')
    output = Path(output)
    # Generated filenames live in one directory; reject path characters unsafe in ffconcat.
    manifest = output.parent / 'concat.txt'
    paths = [Path(p).resolve().as_posix() for p in segments]
    if any("'" in p or '\n' in p or '\r' in p for p in paths):
        raise ValueError('Output path cannot contain quotes or newlines')
    manifest.write_text(''.join(f"file '{p}'\n" for p in paths), encoding='utf-8')
    temporary = output.with_suffix('.tmp.mp4')
    run([ffmpeg(), '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', manifest,
         '-i', mix, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
         '-b:a', '192k', '-t', f'{duration:.9f}', '-movflags', '+faststart', temporary])
    os.replace(temporary, output)
    return output
