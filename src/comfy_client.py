import json
from pathlib import Path
import subprocess
import time
import requests


class ComfyClient:
    def __init__(self, root, python, log_path, low_vram=False):
        self.root = Path(root)
        self.python, self.log_path, self.low_vram = str(python), Path(log_path), low_vram
        self.url = 'http://127.0.0.1:8188'
        self.process = self.log = None

    def get(self, route):
        response = requests.get(self.url + route, timeout=30)
        response.raise_for_status()
        return response.json()

    def post(self, route, data):
        response = requests.post(self.url + route, json=data, timeout=60)
        if not response.ok:
            raise RuntimeError(f'ComfyUI {route}: {response.status_code} {response.text[:4000]}')
        return response.json()

    def start(self):
        try:
            self.get('/system_stats')
        except requests.RequestException:
            pass
        else:
            raise RuntimeError('Port 8188 already occupied. Stop the previous notebook server or restart runtime.')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open('a', encoding='utf-8')
        args = [self.python, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--cache-none']
        if self.low_vram:
            args.append('--novram')
        self.process = subprocess.Popen(args, cwd=self.root, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(300):
            if self.process.poll() is not None:
                raise RuntimeError(f'ComfyUI exited. See {self.log_path}')
            try:
                self.get('/system_stats')
                return self
            except requests.RequestException:
                time.sleep(1)
        self.close()
        raise TimeoutError(f'ComfyUI startup timeout: {self.log_path}')

    def generate(self, graph, checkpoint, timeout=7200):
        from .common import write_json
        checkpoint = Path(checkpoint)
        prompt_id = self.post('/prompt', {'prompt': graph})['prompt_id']
        write_json(checkpoint, {'prompt_id': prompt_id, 'state': 'submitted'})
        start, last = time.monotonic(), 0
        while time.monotonic() - start < timeout:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f'ComfyUI terminated (possible OOM). See {self.log_path}')
            history = self.get(f'/history/{prompt_id}')
            if prompt_id in history:
                record = history[prompt_id]
                write_json(checkpoint, record)
                if record.get('status', {}).get('status_str') == 'error':
                    raise RuntimeError(json.dumps(record['status'].get('messages'), ensure_ascii=False))
                return record['outputs']
            if time.monotonic() - last > 30:
                last = time.monotonic()
                with self.log_path.open('rb') as f:
                    f.seek(max(0, self.log_path.stat().st_size - 500))
                    tail = f.read().decode(errors='replace').replace('\r', '\n').strip().splitlines()
                print(f'  {time.monotonic() - start:.0f}s | {tail[-1] if tail else "waiting"}', flush=True)
            time.sleep(2)
        self.post('/interrupt', {})
        raise TimeoutError(f'Segment timeout after {timeout}s; completed earlier segments are retained')

    def output_file(self, item):
        path = (self.root / 'output' / item.get('subfolder', '') / item['filename']).resolve()
        if not path.is_relative_to((self.root / 'output').resolve()) or not path.is_file():
            raise RuntimeError('Invalid or missing ComfyUI output')
        return path

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log:
            self.log.close()
