"""Regenerate podcast_dual.ipynb from reviewable Python cell sources."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, text):
    value = {'cell_type': kind, 'metadata': {}, 'source': dedent(text).strip().splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell('markdown', '''
# 雙人 AI 播客：Colab 實驗版
兩人同框圖片 → VoxCPM2 雙聲音 → 同步雙音軌 → InfiniteTalk Multi → MP4。

**尚未完成 GPU 實測。先做 5 秒，再 10–15 秒，最後才嘗試 30–60 秒。**
建議 A100 40GB；L4 24GB 自動開 LOW_VRAM，容量與速度仍待驗證。
兩人同框、眨眼、未發言者閉嘴與跨段一致性是生成目標，不能保證。

私人倉庫請先登入 GitHub，在 Code → Download ZIP 下載完整專案。
即使 Colab 能開啟此 Notebook，也不代表 runtime 能匿名 clone 私人倉庫；以下使用 ZIP，不需貼上 token。
''')
cell('code', '''
# ① 匯入完整私人專案 ZIP（已匯入時跳過）
from pathlib import Path
import sys, os, zipfile, subprocess
PROJECT = Path('/content/colab-dual-podcast')
if not (PROJECT / 'src/comfy_workflow.py').exists():
    from google.colab import files
    print('請上傳 colab-dual-podcast 的完整原始碼 ZIP')
    uploaded = files.upload()
    archives = [Path(name) for name in uploaded if name.lower().endswith('.zip')]
    if len(archives) != 1:
        raise ValueError('請只上傳一個專案 ZIP')
    destination = Path('/content/podcast-import')
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('ZIP 包含不安全的路徑')
        archive.extractall(destination)
    roots = [p.parent for p in destination.rglob('podcast_dual.ipynb') if (p.parent / 'src/comfy_workflow.py').is_file()]
    if len(roots) != 1:
        raise ValueError('ZIP 中找不到唯一的完整專案')
    import shutil
    shutil.copytree(roots[0], PROJECT, dirs_exist_ok=True)
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
print('專案就緒：', PROJECT)
''')
cell('code', '''
# ② 設定：先以 5 秒上限測試，配音的實際長度會在生成後顯示
import json
from src.common import write_json
from src.gpu_utils import detect_gpu
LOW_VRAM = None  # None=依 GPU 自動；True=強制 --novram
GPU = detect_gpu(LOW_VRAM)
print(GPU)

USE_DRIVE = False  # True 可讓模型快取與片段跨 runtime 保存；首次會要求掛載 Drive
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/dual-podcast') if USE_DRIVE else Path('/content/dual-podcast')
CACHE = BASE / 'cache'
RUN_NAME = 'test-5s-01'  # 修改圖片、對話、聲音、設定後換新名稱
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME:
    raise ValueError('RUN_NAME 必須是單一資料夾名稱')
RUN_DIR = BASE / 'runs' / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_PATH = '/content/hosts.png'  # 上傳一張兩人同框圖，A 左、B 右
CUSTOM_MASKS = None  # 或 ['/content/mask_A.png', '/content/mask_B.png']，必須是 832×480 灰階 PNG
MASK_SPLIT = 0.5  # 自動遮罩的左右分界；白色=對應主持人區域
STAGE_LIMIT_SEC = 5  # 按人工驗收逐步改成 10、15、30、60；不會自動升級
DIALOGUE = [
    {'speaker': 'A', 'text': '大家好，我是甲。'},
    {'speaker': 'B', 'text': '你好，我是乙。'},
]
VOICES = {
    'A': {'description': '三十歲女性，標準普通話，音色清亮自然，語速中等，親切平穩'},
    'B': {'description': '四十歲男性，標準普通話，音色低沉溫和，語速中等，沉著自然'},
}
# 自有或已獲授權的參考音也可使用，例如 VOICES['A'] = {'reference': '/content/voice_A.wav'}
CONFIG = json.loads((PROJECT / 'configs/default_config.json').read_text())
if STAGE_LIMIT_SEC not in (5, 10, 15, 30, 60):
    raise ValueError('請選 5、10、15、30 或 60 秒階段')
CONFIG['max_duration_sec'] = STAGE_LIMIT_SEC
SETTINGS = {'dialogue': DIALOGUE, 'voices': VOICES, 'config': CONFIG,
            'cache': str(CACHE), 'run_dir': str(RUN_DIR)}
write_json(RUN_DIR / 'settings.json', SETTINGS)
write_json(RUN_DIR / 'gpu.json', GPU)
print('設定已儲存。僅改階段上限不會拉長聲音，必須增加 DIALOGUE。')
''')
cell('code', '''
# ③ 準備圖片與遮罩；紅色 A、藍色 B
from src.masks import prepare_image_and_masks
from IPython.display import display, Image as DisplayImage
if not Path(IMAGE_PATH).is_file():
    from google.colab import files
    print('請上傳圖片，檔名需與 IMAGE_PATH 相符（預設 hosts.png）')
    os.chdir('/content')
    files.upload()
    os.chdir(PROJECT)
prepare_image_and_masks(IMAGE_PATH, RUN_DIR / 'inputs', CONFIG['width'], CONFIG['height'], MASK_SPLIT, CUSTOM_MASKS)
display(DisplayImage(filename=str(RUN_DIR / 'inputs/mask_preview.png'), width=832))
print('檢查兩張臉是否各自在正確色區，嘴部沒有被麥克風遮住；需要調整時回到②。')
''')
cell('code', '''
# ④ 安裝獨立 TTS / ComfyUI 環境，下載及核對固定模型（約 29.7 GB，另需 TTS 與暫存空間）
# 安裝紀錄在 setup.log；未完成的 HF 下載可重跑續傳
with (RUN_DIR / 'setup.log').open('a') as setup_log:
    result = subprocess.run([sys.executable, '-m', 'scripts.setup_colab', '--cache', str(CACHE)],
                            stdout=setup_log, stderr=subprocess.STDOUT)
if result.returncode:
    print((RUN_DIR / 'setup.log').read_text(errors='replace')[-6000:])
    raise RuntimeError('安裝失敗，請查看 setup.log')
RUNTIME = json.loads((PROJECT / '.runtime.json').read_text())
print('安裝完成，模型 checksum 已核對。')
''')
cell('code', '''
# ⑤ 雙聲音配音與雙軌同步（獨立程序退出後釋放 TTS 顯存）
from IPython.display import Audio
command = [RUNTIME['vox_python'], '-u', '-m', 'scripts.run_tts', str(RUN_DIR / 'settings.json')]
with (RUN_DIR / 'tts.log').open('a') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
        log.write(line)
    code = process.wait()
if code:
    raise RuntimeError('配音失敗，請查看 tts.log')
for speaker in ('A', 'B', 'mix'):
    print(speaker)
    display(Audio(filename=str(RUN_DIR / 'audio' / f'{speaker}.wav')))
''')
cell('code', '''
# ⑥ 逐段生成、保存與銜接；重跑會校驗並跳過已完成片段
import logging
from src.comfy_client import ComfyClient
from src.video_generator import generate
logging.basicConfig(filename=str(RUN_DIR / 'errors.log'), level=logging.INFO, force=True)
if globals().get('client'):
    client.close()
client = ComfyClient(RUNTIME['comfy'], RUNTIME['comfy_python'], RUN_DIR / 'comfyui.log', GPU['low_vram'])
try:
    client.start()
    MP4 = generate(client, CONFIG, RUN_DIR / 'inputs', RUN_DIR / 'audio', RUN_DIR / 'video')
finally:
    client.close()
print('影片已輸出：', MP4)
''')
cell('code', '''
# ⑦ 預覽、下載與保存驗收紀錄（檢視結果後手動填寫；預設未驗收）
from IPython.display import Video
from google.colab import files
MP4 = RUN_DIR / 'video/podcast.mp4'
display(Video(str(MP4), embed=True, width=832))
files.download(str(MP4))
''')
cell('code', '''
# ⑧ 人工驗收：看完影片後再設定；此格不會自動宣稱通過
ACCEPTED = False
OBSERVATIONS = ''  # 記錄 A/B 嘴型、未發言者張嘴、身分漂移、段落跳動等
write_json(RUN_DIR / 'acceptance.json', {'accepted_by_user': ACCEPTED, 'observations': OBSERVATIONS,
    'stage_limit_sec': STAGE_LIMIT_SEC, 'gpu': GPU,
    'duration_sec': json.loads((RUN_DIR / 'video/run.json').read_text())['duration']})
print('驗收紀錄已存；請確認下載完成，最後在 Colab 中斷連線並刪除 runtime。')
''')
cell('markdown', '''
## 下一階段與恢復
- 5 秒結果人工驗收後，換 RUN_NAME，增加對話並改 `STAGE_LIMIT_SEC=10`／`15`。
- 最後才用 `30`／`60`；長度由實際 TTS 音軌決定，超過上限會停下，不會截斷語句。
- 同一份設定失敗：重跑⑥。若 runtime 已刪除，只有開啟 Drive 或自行保存的 cache/runs 能恢復。
- 新 runtime：重新匯入專案，掛載同一 Drive、同一 RUN_NAME，依序執行各格。已快取音訊不需重新推論。
- 換素材或設定要換 RUN_NAME，避免混用舊片段。
- 硬體容量、自然互動與 30–60 秒穩定性仍待 GPU 實測；詳見 docs/test_results.md。
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
           'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'name': 'podcast_dual.ipynb'}},
           'nbformat': 4, 'nbformat_minor': 5}
for i, c in enumerate(cells):
    c['id'] = f'podcast-{i:02d}'
(ROOT / 'podcast_dual.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {len(cells)} cells')
