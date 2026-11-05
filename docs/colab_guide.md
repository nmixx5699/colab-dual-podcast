# v0.2.0 Colab 操作指南

1. 在 Colab 左側「檔案」先上傳 `/content/hosts.png`、參考音（如使用），再執行 Notebook。第一格 `SOURCE='github'` 自動下載固定 SHA；`PROJECT_REF='main'` 可選最新版本。公開倉庫不需登入或 Token。
2. 備用匯入：先上傳 `/content/project.zip`，設定 `SOURCE='zip'`。不會要求互動上傳。既有 PROJECT 會重用；切換版本請更名原始碼資料夾，保留 BASE 模型快取和 runs。
3. 設定 `LOW_VRAM=None`，L4/A100 預設正常模式。真實 CUDA OOM 後伺服器會停止；改 True 可手動重跑。不能保證 L4 正常模式足夠。
4. `CACHE_STRATEGY='disabled'` 保留舊 cache-none；另可選 default / low-memory。未經 L4 比較前不承諾速度改善。
5. `VIDEO_ASPECT='16:9'` 或 `'9:16'`；提供相同比例的雙人圖片，檢查預覽保留完整雙臉與左右 Mask。直式仍待 GPU 測試。
6. `MAX_ALLOWED_DURATION_SEC` 是上限；配音後顯示實際長度、上限、總幀數和段數，不會自動拉長配音。
7. 可選字幕格：ENABLE_SUBTITLES 產生 SRT；BURN_SUBTITLES 另產生 podcast_subtitled.mp4。指定 SUBTITLE_FONT、SUBTITLE_FONT_SIZE、SUBTITLE_FONTS_DIR；中文字型可先安裝 `fonts-noto-cjk` 或上傳有中文字形的 .ttf/.otf。缺少 libass/字型時原始 MP4 仍保留。
8. Notebook 可直接下載 MP4 和 SRT。字幕以 TTS 每句時間為邊界，不是逐字卡拉 OK。編輯 SRT 可再調整斷行。
9. 參考音必須為本人或已取得授權，設定 REFERENCE_AUDIO_AUTHORIZED=True；這不是法律授權證明。不要拿未授權的他人錄音來克隆。
10. 完成或失敗後，請手動至「執行階段 → 中斷連線並刪除執行階段」。停止 Python 程序不等於停止 GPU 計費。

完整變更和 GPU 待驗收項目見 [v0.2.0_changes.md](v0.2.0_changes.md)。

