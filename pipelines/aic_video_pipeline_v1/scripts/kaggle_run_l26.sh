#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_DATASET="/kaggle/input/datasets/miphu2005/aic2026-videos-l21-a"
export AIC_VIDEO_ROOTS="${AIC_VIDEO_ROOTS:-${VIDEO_DATASET}/Videos_L26_a/video:${VIDEO_DATASET}/Videos_L26_b/video:${VIDEO_DATASET}/Videos_L26_c/video:${VIDEO_DATASET}/Videos_L26_d/video:${VIDEO_DATASET}/Videos_L26_e/video}"
export AIC_OFFLINE_MODEL_ROOT="${AIC_OFFLINE_MODEL_ROOT:-/kaggle/input/datasets/quanglongl040305/modelfortrainning/aic_l28_offline_models}"
export AIC_VIDEO_LEVEL="L26"
unset AIC_VIDEO_ROOT AIC_ARCHIVES
exec bash "${SCRIPT_DIR}/kaggle_setup_and_run.sh"
