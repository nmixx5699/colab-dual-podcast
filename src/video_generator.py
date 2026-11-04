import json
import logging
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
from .audio_timeline import segment_plan, audio_window
from .common import digest, fingerprint, write_json, ffmpeg, run
from .comfy_workflow import build_graph, validate_graph, ROOT
from .video_merge import trim_segment, merge_segments


def valid_checkpoint(path, signature):
    try:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        return record['signature'] == signature and all(
            (Path(path).parent / name).is_file() and digest(Path(path).parent / name) == sha
            for name, sha in record['files'].items())
    except (OSError, ValueError, KeyError):
        return False


def generate(client, config, inputs_dir, audio_dir, output_dir):
    inputs_dir, audio_dir, output_dir = map(Path, (inputs_dir, audio_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    mix_info = sf.info(audio_dir / 'mix.wav')
    duration = mix_info.frames / mix_info.samplerate
    if not 0 < duration <= 60:
        raise ValueError(f'Dialogue is {duration:.2f}s; supported experimental range is >0 to 60s')
    if duration > config['max_duration_sec']:
        raise ValueError(f'{duration:.2f}s exceeds current stage limit {config["max_duration_sec"]}s; shorten dialogue or advance stage explicitly')
    sources = [inputs_dir / n for n in ('hosts.png', 'mask_A.png', 'mask_B.png')]
    sources += [audio_dir / n for n in ('A.wav', 'B.wav', 'mix.wav')]
    # Include implementation, model lock, and all generation settings in resume identity.
    code = {str(p.relative_to(ROOT)): digest(p) for folder in ('src', 'custom_nodes') for p in (ROOT / folder).rglob('*.py')}
    signature = fingerprint({'config': config, 'inputs': [digest(p) for p in sources],
                             'models': digest(ROOT / 'configs/models.lock.json'), 'code': code})
    manifest = output_dir / 'run.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Inputs/config/code changed. Use a new RUN_NAME to avoid mixing incompatible segments.')
    write_json(manifest, {'signature': signature, 'duration': duration, 'config': config})
    namespace = 'podcast_' + signature[:16]
    target = client.root / 'input'
    target.mkdir(parents=True, exist_ok=True)
    for source in sources[:3]:
        shutil.copy2(source, target / f'{namespace}_{source.name}')
    waves = {}
    for speaker in ('A', 'B'):
        target_audio = output_dir / f'{speaker}_16k.wav'
        run([ffmpeg(), '-v', 'error', '-y', '-i', audio_dir / f'{speaker}.wav', '-ac', '1', '-ar', '16000', target_audio])
        waves[speaker], rate = sf.read(target_audio, dtype='float32')
        if rate != 16000 or waves[speaker].ndim != 1:
            raise ValueError('Expected 16kHz mono audio')
    plan = segment_plan(mix_info.frames, mix_info.samplerate, chunk=config['chunk_frames'], motion=config['motion_frames'])
    write_json(output_dir / 'segments.json', plan)
    previous, videos = [], []
    chain_valid = True
    for segment in plan:
        i = segment['index']
        folder = output_dir / f'segment_{i:03d}'
        folder.mkdir(exist_ok=True)
        checkpoint = folder / 'complete.json'
        stage_sig = fingerprint({'run': signature, 'segment': segment})
        chain_valid = chain_valid and valid_checkpoint(checkpoint, stage_sig)
        if chain_valid:
            print(f'Resume {i + 1}/{len(plan)}: verified saved segment', flush=True)
        else:
            print(f'Generate {i + 1}/{len(plan)}; output frame {segment["output_start"]}', flush=True)
            audio_names = []
            for speaker in ('A', 'B'):
                name = f'{namespace}_{i:03d}_{speaker}.wav'
                window = audio_window(waves[speaker], 16000, segment['audio_start_frame'], config['chunk_frames'])
                sf.write(target / name, window, 16000, subtype='PCM_16')
                audio_names.append(name)
            graph, save_id, tail_id = build_graph(config, f'{namespace}_hosts.png', *audio_names,
                         f'{namespace}_mask_A.png', f'{namespace}_mask_B.png',
                         f'{namespace}/segment_{i:03d}', previous, config['seed'] + i)
            # LoadImage/LoadAudio enumerate the root input directory; refresh after staging.
            info = client.get('/object_info')
            write_json(output_dir / 'object_info.json', info)
            validate_graph(graph, info)
            write_json(folder / 'workflow.json', graph)
            try:
                outputs = client.generate(graph, folder / 'history.json', timeout=config['segment_timeout_sec'])
                raw_items = outputs[save_id].get('images', [])
                tail_items = outputs[tail_id].get('images', [])
                if len(raw_items) != 1 or len(tail_items) != config['motion_frames']:
                    raise RuntimeError('Unexpected SaveVideo/SaveImage output count')
                shutil.copy2(client.output_file(raw_items[0]), folder / 'raw.mp4')
                for j, item in enumerate(tail_items):
                    shutil.copy2(client.output_file(item), folder / f'motion_{j:02d}.png')
                trim_segment(folder / 'raw.mp4', folder / 'clip.mp4', segment['trim'], segment['take'])
                names = ['raw.mp4', 'clip.mp4'] + [f'motion_{j:02d}.png' for j in range(config['motion_frames'])]
                write_json(checkpoint, {'signature': stage_sig, 'files': {n: digest(folder / n) for n in names}})
            except Exception:
                logging.exception('Segment %s failed; prior checkpoints retained', i)
                raise
        previous = []
        for j in range(config['motion_frames']):
            name = f'{namespace}_previous_{i:03d}_{j:02d}.png'
            shutil.copy2(folder / f'motion_{j:02d}.png', target / name)
            previous.append(name)
        videos.append(folder / 'clip.mp4')
        # cache-none prevents old decoded frames accumulating across segments.
        client.post('/free', {'unload_models': False, 'free_memory': True})
    return merge_segments(videos, audio_dir / 'mix.wav', output_dir / 'podcast.mp4', duration)
