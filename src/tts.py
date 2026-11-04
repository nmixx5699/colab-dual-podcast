from pathlib import Path
import os
import numpy as np
import soundfile as sf
from .common import digest, fingerprint
from .gpu_utils import release_gpu

VOX_REV = 'cce58a5b59303c9bd63d12f23afe6a49a4a80c59'
MODEL_REV = '32279effe8c19989596f05d353d1447f51d9e915'


class VoiceEngine:
    def __init__(self, cache_dir, model_cache=None):
        from huggingface_hub import snapshot_download
        from voxcpm import VoxCPM
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        path = snapshot_download('openbmb/VoxCPM2', revision=MODEL_REV, cache_dir=model_cache)
        self.model = VoxCPM.from_pretrained(path, load_denoiser=False, optimize=False, device='cuda')
        self.sample_rate = self.model.tts_model.sample_rate

    def _cached(self, text, seed, reference=None):
        key = fingerprint({'text': text, 'seed': seed, 'reference': digest(reference) if reference else None,
                           'code': VOX_REV, 'model': MODEL_REV, 'cfg': 2.0, 'steps': 10})
        path = self.cache / f'{key}.wav'
        if path.exists():
            wav, sr = sf.read(path, dtype='float32')
            if sr == self.sample_rate and wav.ndim == 1 and wav.size and np.isfinite(wav).all() and np.any(wav):
                return wav, path
            raise RuntimeError(f'Corrupt cached voice: {path}; remove this file and retry')
        kwargs = {'reference_wav_path': str(reference)} if reference else {}
        wav = np.asarray(self.model.generate(text=text, seed=seed, cfg_value=2.0,
                                            inference_timesteps=10, **kwargs), dtype='float32').reshape(-1)
        if not wav.size or not np.isfinite(wav).all() or not np.any(wav):
            raise RuntimeError('VoxCPM2 returned empty, silent, or nonfinite audio')
        temporary = path.with_suffix('.tmp.wav')
        sf.write(temporary, wav, self.sample_rate, subtype='FLOAT')
        os.replace(temporary, path)
        return wav, path

    def synthesize(self, rows, voices, seed=42):
        if set(voices) != {'A', 'B'}:
            raise ValueError('Configure exactly voices A and B')
        references = {}
        for i, speaker in enumerate(('A', 'B')):
            voice = voices[speaker]
            if voice.get('reference'):
                reference = Path(voice['reference'])
                if not reference.is_file():
                    raise FileNotFoundError(reference)
            else:
                description = voice.get('description', '').strip()
                if not description:
                    raise ValueError(f'{speaker} needs a description or reference recording')
                _, reference = self._cached(f'({description})大家好，欢迎收看今天的节目，我们马上开始。', seed + i * 1000)
            references[speaker] = reference
        if digest(references['A']) == digest(references['B']):
            raise ValueError('A and B must use distinct voices')
        result = []
        for i, row in enumerate(rows):
            wav, _ = self._cached(row['synthesis_text'], seed + i, references[row['speaker']])
            result.append((row, wav))
            print(f'TTS {i + 1}/{len(rows)}: {row["speaker"]} {len(wav) / self.sample_rate:.2f}s', flush=True)
        return result

    def close(self):
        self.model = None
        release_gpu()
