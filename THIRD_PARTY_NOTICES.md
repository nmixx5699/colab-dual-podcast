# Third-party notices

This repository is distributed under GPL-3.0-only (see LICENSE), including the ComfyUI-derived node.
Original orchestration and documentation: Copyright (c) 2026 colab-dual-podcast contributors.
No model weights are redistributed in this repository.

## Reference project (MIT)

Reference: [dawang-sakura/colab-talking-avatar](https://github.com/dawang-sakura/colab-talking-avatar/tree/cd18c818a50a9d3c3deb5d1c7eed71e639ae5c59).
Copyright (c) 2026 dawang-sakura. Complete MIT notice retained in licenses/colab-talking-avatar-MIT.txt.
The TTS configuration, fixed model sources, 81/9-frame generation parameters and workflow design informed this implementation.
README, all Notebook cells, docs, examples, voice catalog and LICENSE were reviewed. Binary assets were inventoried; voice recordings were not independently audited for provenance or perceptual quality. They are not copied into this project.

## ComfyUI derivative (GPL-3.0)

`custom_nodes/dual_podcast/__init__.py` incorporates `WanInfiniteTalkToVideo` from
[ComfyUI source at 426f5eea52a7ece1dd599fff4cb6cd1f352ddd24](https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy_extras/nodes_wan.py).
Copyright ComfyUI contributors. The upstream GPL license is included as this repository's LICENSE.
Modified in November 2026: independent node/class identifier, parallel audio (`para`) instead of sequential (`add`), independent extension registration and imports.
Corresponding modified source is shipped here; remaining unmodified dependencies are obtained from the pinned upstream source, retaining its copyright and license files. No upstream file is overwritten.

## Downloaded components

| Component | Source / stated license |
|---|---|
| VoxCPM code | [OpenBMB/VoxCPM LICENSE](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/LICENSE), Apache-2.0 |
| VoxCPM2 weights | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), Apache-2.0 model metadata |
| Wan I2V FP8 | [Kijai/WanVideo_comfy_fp8_scaled](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled), Apache-2.0 metadata |
| UMT5 and InfiniteTalk Multi FP16 | [Comfy-Org/Wan-2.1](https://huggingface.co/Comfy-Org/Wan-2.1), Apache-2.0 metadata; [InfiniteTalk upstream license](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/LICENSE.txt) |
| Chinese wav2vec2 | [Kijai/wav2vec2_safetensors](https://huggingface.co/Kijai/wav2vec2_safetensors), MIT metadata |
| VAE and LightX2V converted files | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy): hosting metadata does NOT specify a license |
| Upstream Wan / LightX2V | [Wan2.1 license](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt), [LightX2V license](https://github.com/ModelTC/LightX2V/blob/main/LICENSE): Apache-2.0; converted-file equivalence and licensing are not independently established |
| FFmpeg | [FFmpeg legal](https://ffmpeg.org/legal.html): license depends on build; external executable, not bundled |

The metadata and file hashes for the six converted Comfy models were queried directly on 2026-11-05 and are recorded in configs/models.lock.json.
File-name correspondence is not proof that converted VAE/LoRA weights are byte-equivalent to upstream files. Resolve those terms before redistributing those weights or asserting commercial permissions. This project's GPL does not relicense models, input recordings, or generated likenesses.
