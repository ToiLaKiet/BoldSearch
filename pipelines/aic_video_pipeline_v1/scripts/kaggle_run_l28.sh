#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The L28 MP4 files are already attached as a Kaggle Input.  Override this
# variable if Kaggle mounts the dataset under a different slug in your session.
export AIC_VIDEO_ROOT="${AIC_VIDEO_ROOT:-/kaggle/input/datasets/miphu2005/aic2026-videos-l21-a/Videos_L28_a/video}"
# Model is also attached as an extracted Kaggle Input, so setup must never
# download it from the network.
export AIC_OFFLINE_MODEL_ROOT="${AIC_OFFLINE_MODEL_ROOT:-/kaggle/input/datasets/quanglongl040305/modelfortrainning/aic_l28_offline_models}"
export AIC_VIDEO_LEVEL="L28"
exec bash "${SCRIPT_DIR}/kaggle_setup_and_run.sh"
