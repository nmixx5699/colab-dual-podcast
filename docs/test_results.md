# v0.2.0 CPU validation — 2026-11-05

| Environment | Command | Result |
|---|---|---|
| Untouched 688a27c baseline, Python 3.12.14 | python -m pytest -q | 29 passed (2.27s) |
| Updated source, Python 3.12.14 | python -m pytest -q | 66 passed (11.30s) |
| Updated source, CPython 3.13.17 via uv | python -m pytest -q | 66 passed (11.03s) |
| Updated source | python -m compileall -q src scripts custom_nodes | passed |
| Regenerated Notebook | tests/test_notebook.py | JSON, all code-cell syntax, cleared outputs passed |

Python 3.13 tests run real ensurepip/pip venv creation, simulated failed first creation, quarantine, successful retry and reuse. Package-name selection is independently tested for 3.13. This is **not** an Ubuntu 24.04.4 Colab 3.13.16 apt/CUDA dependency installation test.

New tests cover GET/POST empty/JSON/HTTP/non-JSON responses, L4/A100/manual LOW_VRAM, cache flags, no-progress timeout, KeyboardInterrupt cleanup, HF valid-cache reuse/disk budget, corrupted-blob retry/exhaustion, both aspect ratios and masks/workflow continuation geometry, actual 125-frame FFmpeg concat at 832×480 and 480×832, timeline SRT and real libass burn-in. Burn-in fixture uses Latin text/DejaVu Sans; Chinese font availability/glyph rendering must still be checked in Colab.

Recovery integration now exercises successful empty `/free` as well as malformed cleanup responses: valid completed checkpoints remain reusable. Synthetic backend tests do not run AI inference.

GPU: **NOT RUN**. No NVIDIA GPU or Colab session available. 3–5s/6–10s AI generation, default vs LOW_VRAM, measured disk/GPU model reload times, sampler times, GPU memory and portrait identity/mouth/seam quality remain pending. Provisional idle timeout is not L4 calibrated; original 7200s total budget is retained.

See [change inventory and limitations](v0.2.0_changes.md). Voice packs/menu, green-screen/removal, upscale and complete ZIP export remain deferred behind GPU acceptance; optional subtitles are included by explicit request.

---

## Historical baseline report

# 實際驗證紀錄

日期：2026-11-05（Asia/Taipei）。環境：Windows、Python 3.12；本機 GPU 為 NVIDIA GTX 1650 4GB。
沒有可用的 L4／A100 Colab runtime，未載入 TTS 或視訊模型做推論。

## 已執行

| 檢查 | 實際結果 |
|---|---|
| `python -m pytest -q` | **29 passed**；最後一次完整執行 2.46 秒 |
| `python -m compileall -q src scripts custom_nodes` | 通過 |
| Notebook 程式格語法、空輸出與執行狀態檢查 | 通過（包含於 pytest） |
| 5／10／15／30／60 秒分段覆蓋數學 | 通過；沒有重複／遺漏輸出幀 |
| 雙軌靜音、對話順序、取樣點對齊及無效音訊拒絕 | 通過 |
| 遮罩左右順序、互斥、輸出尺寸與留邊 | 通過 |
| 實際 FFmpeg 合成 | 通過：合成紅／藍測試圖，輸出 125 幀／約 5 秒；第 81 幀後顏色切換符合裁接；音訊有內容且長度容差 <70ms |
| 模擬生成後端中斷／恢復 | 通過：第二段中斷後只補第二段；完整 checkpoint 不重生；首段損壞會使後續段失效 |
| 24 種節點及所用參數的上游原始碼核對 | 通過，見 source_audit.json；並非 ComfyUI server 端執行結果 |
| 衍生節點 AST 比對 | 通過：原始類別運算僅將音訊 add 改 para，另改獨立 ID |
| 實際平行音訊排列區塊執行 | 通過：B 未被推遲到 A 全長後；以 NumPy shim 驗證排列，不是張量模型推論 |
| 六個權重來源、固定 revision、大小、LFS SHA-256 | 已向 Hugging Face API 核對；沒有下載全權重，因此本機未計算完整權重內容雜湊 |
| 六個 safetensors 實際 HTTP Range 標頭 | 讀取成功；模型 loader 判別鍵存在，見 model_header_audit.json |
| 參考資料盤點 | 38 個檔案雜湊／大小、10 張圖、WAV metadata；見 reference_inventory.json |

Multi patch 的實際 `audio_proj.proj1.weight` 是 [512,46080]，`audio_proj.norm.weight` 是 [768]，首個 audio cross-attention projection 是 [5120,5120]，對應固定版 ModelPatchLoader 讀取的判別鍵與維度。
官方 `wan2.1_infiniteTalk_multi_fp16.safetensors` 包含 F32:130、BF16:80、F16:120 個張量。
FP8 底模也保留部分 F32／F16 張量；不強制重轉整檔 dtype。

## 尚未驗證／不可宣稱成功

- Colab 安裝全流程、目前 Colab CUDA/PyTorch 與間接依賴的實際相容性。
- `/object_info` live validation 與 ComfyUI prompt 執行；Notebook 在真正提交前會執行，但本機沒有啟動模型環境。
- VoxCPM2 中文發音品質、兩種聲音的可分辨性與音色一致性。
- InfiniteTalk Multi 5 秒、10–15 秒、30 秒、60 秒影片。
- L4 24GB LOW_VRAM 或 A100 40GB 的顯存／RAM 峰值、速度與成本。
- A/B 嘴型正確性、未發言者閉嘴、自然眨眼點頭、跨段人物與服裝穩定性。

## GPU 驗收表（由實際執行者填寫）

| GPU | 實際時長 | 狀態 | 嘴型／人物／接縫 | 峰值 VRAM／RAM | 耗時 |
|---|---|---|---|---|---|
| A100 40GB | 5 秒 | 未測 | — | — | — |
| L4 24GB LOW_VRAM | 5 秒 | 未測 | — | — | — |
| A100 40GB | 10–15 秒 | 未測 | — | — | — |
| A100 40GB | 30 秒 | 未測 | — | — | — |
| A100 40GB | 60 秒 | 未測 | — | — | — |

每次保留 run_dir、pip freeze、gpu.json、workflow.json、comfyui.log、輸出 MP4 與 acceptance.json。
只有通過實際短片人工檢視後，才應增加 STAGE_LIMIT_SEC；CPU 測試不能替代這項驗收。

