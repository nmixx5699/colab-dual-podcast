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
建議 A100 40GB；L4 24GB 預設正常顯存模式，可手動開 LOW_VRAM，容量與速度仍待驗證。
兩人同框、眨眼、未發言者閉嘴與跨段一致性是生成目標，不能保證。

公開倉庫可直接下載，不需 Token。先用 Colab 左側「檔案」上傳 hosts.png 及參考音，再全部執行。
''')
cell('code', '''
# ① 從公開 GitHub 固定版本取得專案；main 為可選最新版本
from pathlib import Path
import sys, os, zipfile, subprocess, urllib.request, tempfile, shutil
PROJECT = Path('/content/colab-dual-podcast')
SOURCE = 'github'  # github 或 zip；不會彈出互動上傳
PROJECT_REF = '16c04c8e11dd3c6c0bfe1eddc7b15afb0375fc27'  # 或 main；固定 SHA 可重現
ZIP_PATH = '/content/project.zip'  # zip 備用：先從左側檔案上傳
if not (PROJECT / 'src/comfy_workflow.py').exists():
    archive_path = Path(ZIP_PATH)
    if SOURCE == 'github':
        archive_path = Path('/content/podcast-source.zip')
        urllib.request.urlretrieve(f'https://github.com/nmixx5699/colab-dual-podcast/archive/{PROJECT_REF}.zip', archive_path)
    elif SOURCE != 'zip':
        raise ValueError('SOURCE 必須是 github 或 zip')
    if not archive_path.is_file():
        raise FileNotFoundError('請先從 Colab 左側檔案上傳 project.zip')
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                    raise ValueError('ZIP 包含不安全路徑')
            archive.extractall(destination)
        roots = [p.parent for p in destination.rglob('podcast_dual.ipynb') if (p.parent / 'src/comfy_workflow.py').is_file()]
        if len(roots) != 1:
            raise ValueError('找不到唯一完整專案')
        shutil.copytree(roots[0], PROJECT)
else:
    print('沿用已匯入專案；切換版本請先更名或移除 PROJECT 原始碼資料夾，保留 BASE 快取與片段。')
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
print('專案就緒：', PROJECT)
# IPython exception handler does not disconnect or change the user's account/runtime.
def billing_exception(shell, etype, evalue, tb, tb_offset=None):
    print('⚠ GPU 仍可能計費！請至 Colab 執行階段 → 中斷連線並刪除執行階段。')
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
get_ipython().set_custom_exc((Exception, KeyboardInterrupt), billing_exception)
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
MAX_ALLOWED_DURATION_SEC = 5  # 按人工驗收逐步改成 10、15、30、60；不會自動升級
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
if MAX_ALLOWED_DURATION_SEC not in (5, 10, 15, 30, 60):
    raise ValueError('請選 5、10、15、30 或 60 秒階段')
CONFIG['max_duration_sec'] = MAX_ALLOWED_DURATION_SEC
VIDEO_ASPECT = '16:9'  # 或 9:16（尚未 GPU 實測）
from src.masks import aspect_dimensions
CONFIG['width'], CONFIG['height'] = aspect_dimensions(VIDEO_ASPECT)
CACHE_STRATEGY = 'disabled'  # default / disabled / low-memory；保守預設可回退
CONFIG['cache_strategy'] = CACHE_STRATEGY
CONFIG['segment_timeout_sec'] = 7200  # 總超時；未經 L4 實測，保留原有慢速容忍值
CONFIG['idle_timeout_sec'] = 1800  # 無日誌變化超時，可依實測調高；不是 GPU 利用率偵測
CONFIG['free_memory_between_segments'] = True
CONFIG['unload_models_between_segments'] = False
REFERENCE_AUDIO_AUTHORIZED = False  # 自有或已取得錄音使用者授權才改 True；此聲明不是法律授權保證
if any(v.get('reference') for v in VOICES.values()) and not REFERENCE_AUDIO_AUTHORIZED:
    raise ValueError('使用參考音前須確認為本人錄音或已取得授權')
SETTINGS = {'dialogue': DIALOGUE, 'voices': VOICES, 'config': CONFIG,
            'cache': str(CACHE), 'run_dir': str(RUN_DIR), 'reference_audio_authorized': REFERENCE_AUDIO_AUTHORIZED}
write_json(RUN_DIR / 'settings.json', SETTINGS)
write_json(RUN_DIR / 'gpu.json', GPU)
print('設定已儲存。僅改階段上限不會拉長聲音，必須增加 DIALOGUE。')
''')
cell('code', '''
# ③ 準備圖片與遮罩；紅色 A、藍色 B
from src.masks import prepare_image_and_masks
from IPython.display import display, Image as DisplayImage
if not Path(IMAGE_PATH).is_file():
    raise FileNotFoundError('請先從 Colab 左側檔案上傳 hosts.png，避免全部執行停在互動輸入')
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
    print('⚠ GPU 仍可能計費！請至 Colab 執行階段 → 中斷連線並刪除執行階段。')
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
    print('⚠ GPU 仍可能計費！請至 Colab 執行階段 → 中斷連線並刪除執行階段。')
    raise RuntimeError('配音失敗，請查看 tts.log')
import soundfile as sf
from src.audio_timeline import segment_plan
audio_info = sf.info(RUN_DIR / 'audio/mix.wav')
plan = segment_plan(audio_info.frames, audio_info.samplerate, chunk=CONFIG['chunk_frames'], motion=CONFIG['motion_frames'])
import math
print(f"TTS 實際 {audio_info.duration:.2f}s / 最大允許 {MAX_ALLOWED_DURATION_SEC}s / 預估總幀數 {math.ceil(audio_info.duration * 25)} / 段數 {len(plan)}")
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
client = ComfyClient(RUNTIME['comfy'], RUNTIME['comfy_python'], RUN_DIR / 'comfyui.log', GPU['low_vram'], CACHE_STRATEGY, CONFIG['idle_timeout_sec'])
try:
    client.start()
    MP4 = generate(client, CONFIG, RUN_DIR / 'inputs', RUN_DIR / 'audio', RUN_DIR / 'video')
except BaseException:
    logging.exception('Generation stopped')
    print('⚠ GPU 仍可能計費！請至 Colab 執行階段 → 中斷連線並刪除執行階段。已完成片段保留。')
    raise
finally:
    client.close()
print('影片已輸出：', MP4)
''')
cell('code', '''
# 可選字幕後製：依實際 TTS 時間軸輸出 SRT，並可燒入另一份 MP4
ENABLE_SUBTITLES = True
BURN_SUBTITLES = False  # 需 FFmpeg libass 及中文字型；失敗不影響原始影片
SUBTITLE_FONT = 'Noto Sans CJK TC'
SUBTITLE_FONTS_DIR = None  # 可設上傳的 .ttf/.otf 資料夾
SUBTITLE_FONT_SIZE = 24
if ENABLE_SUBTITLES:
    try:
        from src.subtitles import write_srt, burn_subtitles
        SRT = write_srt(RUN_DIR / 'audio/timeline.json', RUN_DIR / 'video/podcast.srt')
        print('字幕：', SRT)
        if BURN_SUBTITLES:
            MP4 = burn_subtitles(RUN_DIR / 'video/podcast.mp4', SRT, RUN_DIR / 'video/podcast_subtitled.mp4',
                                 SUBTITLE_FONT, SUBTITLE_FONT_SIZE, SUBTITLE_FONTS_DIR)
    except Exception:
        logging.exception('Optional subtitle postprocessing failed')
        print('字幕後製失敗，原始 podcast.mp4 已保留；請查看 errors.log。')
''')
cell('code', '''
# ⑦ 預覽、下載與保存驗收紀錄（檢視結果後手動填寫；預設未驗收）
from IPython.display import Video
from google.colab import files
MP4 = Path(globals().get('MP4', RUN_DIR / 'video/podcast.mp4'))
display(Video(str(MP4), embed=True, width=832))
files.download(str(MP4))
if (RUN_DIR / 'video/podcast.srt').exists():
    files.download(str(RUN_DIR / 'video/podcast.srt'))
''')
cell('code', '''
# ⑧ 人工驗收：看完影片後再設定；此格不會自動宣稱通過
ACCEPTED = False
OBSERVATIONS = ''  # 記錄 A/B 嘴型、未發言者張嘴、身分漂移、段落跳動等
write_json(RUN_DIR / 'acceptance.json', {'accepted_by_user': ACCEPTED, 'observations': OBSERVATIONS,
    'stage_limit_sec': MAX_ALLOWED_DURATION_SEC, 'gpu': GPU,
    'duration_sec': json.loads((RUN_DIR / 'video/run.json').read_text())['duration']})
print('驗收紀錄已存；請確認下載完成，最後在 Colab 中斷連線並刪除 runtime。')
''')
cell('markdown', '''
## 下一階段與恢復
- 5 秒結果人工驗收後，換 RUN_NAME，增加對話並改 `MAX_ALLOWED_DURATION_SEC=10`／`15`。
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

