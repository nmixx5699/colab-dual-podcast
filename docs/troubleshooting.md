# 問題處理

| 問題 | 處理 |
|---|---|
| GitHub／Colab 404 | 私人倉庫需對應帳號授權；可下載 ZIP 和 Notebook 後上傳 Colab。 |
| 沒有 L4／A100 | 在 Colab 更改 runtime，確認取得的 GPU；本機小顯存卡無法替代 GPU 驗證。 |
| OOM／程序退出 | 停止目前 server、LOW_VRAM=True、先做 5 秒，改用 A100；查看 comfyui.log 與系統 RAM，不能保證 L4 成功。 |
| 安裝相依衝突 | 查 setup.log 與 cache/*-pip-freeze.txt。TTS／Comfy 已分 venv；Colab PyTorch 與間接套件仍會變動，請保留紀錄回報。 |
| 找不到雙人節點 | 確認固定 Comfy commit、custom_nodes/dual_podcast 已安裝並重新啟動；不可直接換成原始單人／串接節點。 |
| unknown input／模型不存在 | runtime object_info 檢查會提前停止。確認版本及模型檔名，不要猜參數名稱或跳過檢查。 |
| 模型 checksum 不符 | 不會靜默使用檔案；移走損壞的單一快取檔後重新下載。不要刪整個 Drive。 |
| 聲音超過階段上限 | 先聽⑤的混音，再縮短對話；通過前階段驗收後才能提高上限。 |
| A/B 嘴型對調 | 檢查 A 在左、B 在右，紅／藍遮罩與音軌是否正確，使用 Multi 權重。 |
| 未發言者仍張嘴 | 檢查靜音軌及遮罩；數位靜音不是閉嘴的保證，調整圖像、提示與種子並記錄失敗。 |
| 續接漂移／跳動 | 保留原始雙人圖及 9 幀參考，避免人物跨遮罩；長片段累積漂移仍屬模型限制。 |
| resume 提示設定改變 | 用新 RUN_NAME；不要手動改 signature 來混用不同生成結果。 |
| 8188 已占用 | 執行 client.close() 或重新啟動 runtime；程式不會任意終止不屬於自己的程序。 |
| MP4 未下載 | 從 RUN_DIR/video/podcast.mp4 手動下載；先確認檔案存在。 |

如 ComfyUI 在片段生成中斷線，該段尚未驗證的輸出不會當成完成；已完成段保留。重試時伺服器會重新生成未完成段。
