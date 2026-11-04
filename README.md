# colab-dual-podcast

AI Two-Person Podcast Video Generator powered by Google Colab, VoxCPM2, Wan2.1 and InfiniteTalk Multi.

[在 Google Colab 開啟](https://colab.research.google.com/github/nmixx5699/colab-dual-podcast/blob/main/podcast_dual.ipynb) · [操作指南](docs/colab_guide.md) · [測試結果](docs/test_results.md)

一張兩人同框圖片，加上中文 A／B 對話，產生兩種聲音、同步雙軌及 MP4。
**目前是完成 CPU 測試與上游原始碼核對的實驗版，尚未通過 Colab GPU 端到端驗證。5 秒、10–15 秒與 30–60 秒都不得視為已成功。**

## 開始使用

1. 在私人 GitHub 倉庫登入後，下載 `Code → Download ZIP`。開啟上方 Colab 連結；若無法讀取私人倉庫，將 `podcast_dual.ipynb` 上傳到 Colab。
2. 選 A100 40GB（主要建議測試環境）或 L4 24GB（低顯存測試）。Notebook 會自動檢測 GPU，L4 預設 `--novram`。兩者容量／速度均尚待本專案實測。
3. 第一格上傳完整專案 ZIP。私人 repo 無法匿名 clone，因此不用將 GitHub token 寫進 Notebook。
4. 上傳 `hosts.png`：兩位虛構主持人 A 左、B 右，上半身入鏡、正面清晰，麥克風低於嘴部。預設輸出 832×480，以留邊縮放保留兩人，不裁臉。
5. 修改②的 `DIALOGUE`、`VOICES`，先保留 `STAGE_LIMIT_SEC=5`。檢查③的紅色 A／藍色 B 遮罩預覽，再往下安裝、配音、生成。
6. 預覽⑦的 MP4 並下載。看過實際嘴型、人物與接縫後，才在⑧填寫人工驗收紀錄。

```python
DIALOGUE = [
    {"speaker": "A", "text": "大家好，歡迎收看今天的節目。"},
    {"speaker": "B", "text": "今天我們要來討論一個新話題。"},
    {"speaker": "A", "text": "那就讓我們開始吧。"},
]
VOICES = {
    "A": {"description": "三十歲女性，音色清亮自然，標準普通話"},
    "B": {"description": "四十歲男性，音色低沉溫和，標準普通話"},
}
# 或：VOICES['A'] = {'reference': '/content/voice_A.wav'}
```

音色描述會先生成並快取一段虛構參考音，再逐句使用同一參考音；兩人分別管理。
自備參考音請使用自有或已獲授權的乾淨單人錄音，建議 5–15 秒。此版本只提供可控參考音模式，不混用 prompt_text 極致克隆。

## 已實作

- VoxCPM2 中文配音、80 字切句、語氣前綴保留、按文字／參考音內容／版本／種子快取。
- 取樣點級時間軸、未發言者數位靜音、同步等長 A/B 音軌及最終混音；影片音訊使用原始取樣率，特徵音軌使用 16kHz。
- 左右遮罩、自訂遮罩、尺寸／重疊／左右順序檢查、可視化預覽。
- Wan2.1 I2V 14B 480P FP8、InfiniteTalk Multi FP16、Chinese wav2vec2、Wan VAE、LightX2V LoRA，另含必要的 UMT5 文字編碼器。
- 對照 ComfyUI v0.43.0 原始碼建構 API 工作流；在 runtime 用 `/object_info` 檢查節點、參數、連線類型、模型選項。
- 81 幀生成窗、9 幀無損 PNG 動作續接、逐段 MP4、SHA-256 checkpoint、失敗恢復、FFmpeg 拼接與 AAC 混音。
- GPU 檢測、LOW_VRAM、固定模型版本與雜湊驗證、Drive 持久快取選項、進度／錯誤紀錄、預覽與下載。

重要修正：上游雙人節點硬編碼 `multi_audio_type="add"`，會將 A、B 特徵依序串接。
本專案的獨立 `PodcastInfiniteTalkToVideo` 衍生節點使用上游既有 `para` 分支，才能接收完整同步雙軌。
原始 ComfyUI 檔案保持原樣；[原理與參數證據](docs/architecture.md)、[授權與來源](THIRD_PARTY_NOTICES.md)。

## 30–60 秒流程

先人工確認 5 秒，再換 `RUN_NAME` 測 10–15 秒，最後加長對話並設定 `STAGE_LIMIT_SEC=30` 或 `60`。
實際長度由 TTS 決定；上限不是強制輸出長度，也不會截斷超長語句。
每個續接片段新增 72 幀；30 秒約 11 段，60 秒約 21 段。
長片段會增加運算與儲存成本；不保證 L4 可完成，也不宣稱自然互動或無漂移。
如需跨 runtime 恢復，開始前設定 `USE_DRIVE=True`；只存在 `/content` 的檔案會隨 runtime 刪除。

## 測試

```bash
pip install -r requirements.txt
# 系統需安裝 FFmpeg；Windows 測試可額外安裝 imageio-ffmpeg
python -m pytest -q
python -m compileall -q src scripts custom_nodes
```

目前 CPU 測試包含實際 FFmpeg 5 秒合成、125 幀計數、音訊長度與片段顏色交界檢查；這不等同 AI 生成測試。
上游相容性來源核對可重現：

```bash
python -m scripts.audit_upstream --comfy /path/to/ComfyUI-v0.43.0 --vox /path/to/VoxCPM-cce58a5b
```

完整目錄見 [docs/file_tree.txt](docs/file_tree.txt)。問題處理見 [troubleshooting](docs/troubleshooting.md)。

## 授權

本專案以 **GPL-3.0-only** 發布，因包含 ComfyUI 衍生節點；完整原始碼隨倉庫提供。
參考專案的 MIT 聲明保留在 `licenses/colab-talking-avatar-MIT.txt`。
模型權重不隨倉庫發佈，各自遵守其來源授權；Kijai VAE／LoRA 轉換檔的托管頁沒有明示授權，不能用本專案 GPL 取代模型權利要求。詳見第三方聲明。
