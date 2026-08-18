#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# L30 MP4 files are already mounted as a read-only Kaggle Input.  Reuse one
# resident AutoShot/FG-CLIP2 process and never download/copy the video archive.
export AIC_VIDEO_ROOT="${AIC_VIDEO_ROOT:-/kaggle/input/datasets/miphu2005/aic2026-videos-l21-a/Videos_L30_a/video}"
export AIC_OFFLINE_MODEL_ROOT="${AIC_OFFLINE_MODEL_ROOT:-/kaggle/input/datasets/quanglongl040305/modelfortrainning/aic_l28_offline_models}"
export AIC_VIDEO_LEVEL="L30"
exec bash "${SCRIPT_DIR}/kaggle_setup_and_run.sh"
