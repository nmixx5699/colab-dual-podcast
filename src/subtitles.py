"""Optional subtitles use actual TTS utterance boundaries, never estimated speech speed."""
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from .common import ffmpeg, run


def timestamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid subtitle time')
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{sec:02},{ms:03}'


def write_srt(timeline, output, line_chars=20, speaker_labels=False):
    if line_chars < 1:
        raise ValueError('line_chars must be positive')
    data = json.loads(Path(timeline).read_text(encoding='utf-8'))
    cues, previous = [], 0
    for event in data['events']:
        start, end = event['start'], event['end']
        if start < previous or end <= start:
            raise ValueError('Subtitle events must be ordered and nonoverlapping')
        previous = end
        # Strip TTS expression prefixes and disallow SRT markup/control injection.
        text = re.sub(r'^\s*\[[^\]]*\]\s*', '', event['text'])
        text = re.sub(r'<[^>]*>', '', text).replace('-->', '→')
        text = ' '.join(text.split())
        if speaker_labels:
            text = event['speaker'] + '：' + text
        lines = '\n'.join(text[i:i + line_chars] for i in range(0, len(text), line_chars))
        cues.append(f'{len(cues)+1}\n{timestamp(start)} --> {timestamp(end)}\n{lines}\n')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(cues), encoding='utf-8')
    return output


def burn_subtitles(video, srt, output, font='Noto Sans CJK TC', font_size=24, fonts_dir=None):
    # Fixed temp names avoid FFmpeg filter escaping hazards from user file paths.
    if not re.fullmatch(r'[\w -]+', font) or not 8 <= int(font_size) <= 96:
        raise ValueError('Invalid subtitle font or size')
    output = Path(output).resolve()
    if output == Path(video).resolve():
        raise ValueError('Subtitle output must not overwrite the original MP4')
    with tempfile.TemporaryDirectory(prefix='podcast-subs-') as directory:
        directory = Path(directory)
        shutil.copy2(srt, directory / 'captions.srt')
        if fonts_dir:
            shutil.copytree(fonts_dir, directory / 'fonts')
        style = f'FontName={font},FontSize={int(font_size)},Outline=2,MarginV=24,Alignment=2'
        vf = f"subtitles=filename=captions.srt:force_style='{style}'"
        if fonts_dir:
            vf += ':fontsdir=fonts'
        run([ffmpeg(), '-v', 'error', '-y', '-i', Path(video).resolve(), '-vf', vf,
             '-c:v', 'libx264', '-crf', '18', '-c:a', 'copy', output], cwd=directory)
    return output
