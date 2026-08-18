#!/usr/bin/env bash
set -Eeuo pipefail

# Offline Kaggle launcher for exactly L26_V175 through L26_V199.  It reads
# direct MP4 Input files; no ZIP, wget, apt-get, PyPI, or Hugging Face access.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

KAGGLE_INPUT_ROOT="/kaggle/input"
KAGGLE_WORKING_ROOT="/kaggle/working"
RUNTIME_ROOT="${AIC_RUNTIME_ROOT:-/tmp/aic_l26_b_v175_v199_offline_runtime}"
WORK_ROOT="${RUNTIME_ROOT}/work"
OFFLINE_SITE_PACKAGES="${RUNTIME_ROOT}/python_packages"
OFFLINE_WHEEL_ROOT="${PIPELINE_ROOT}/offline_wheels"
RESULT_ROOT="${AIC_RESULT_ROOT:-${KAGGLE_WORKING_ROOT}/aic_l26_b_v175_v199_offline_results}"
START_VIDEO_ID="L26_V175"
END_VIDEO_ID="L26_V199"

CHECKPOINT_SIZE="57243097"
CHECKPOINT_SHA256="3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6"
FGCLIP2_MODEL_SIZE="3586589392"

die() {
    echo "LỖI: $*" >&2
    exit 1
}

cleanup() {
    local status=$?
    trap - EXIT
    if (( status == 0 )) && [[ "$RUNTIME_ROOT" == "/tmp/aic_l26_b_v175_v199_offline_runtime" ]]; then
        find "$RUNTIME_ROOT" -xdev -depth -delete 2>/dev/null || true
    elif (( status != 0 )); then
        echo "Lần chạy dừng lỗi; giữ work tạm để có thể chạy lại: $RUNTIME_ROOT" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

[[ -d "$KAGGLE_INPUT_ROOT" && -d "$KAGGLE_WORKING_ROOT" ]] || die "Script này chỉ chạy trên Kaggle Notebook"
[[ -f "${PIPELINE_ROOT}/configs/default.yaml" ]] || die "Thiếu configs/default.yaml"
[[ -f "${PIPELINE_ROOT}/scripts/kaggle_directory_runner.py" ]] || die "Thiếu direct GPU runner"
[[ -d "${PIPELINE_ROOT}/src/aic_video_pipeline_v1" ]] || die "Thiếu source aic_video_pipeline_v1"
[[ -d "$OFFLINE_WHEEL_ROOT" ]] || die "Thiếu offline_wheels trong source dataset"

mkdir -p "$WORK_ROOT" "$OFFLINE_SITE_PACKAGES" "$RESULT_ROOT"
find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.tar.tmp' -delete 2>/dev/null || true
command -v ffmpeg >/dev/null || die "Kaggle runtime thiếu binary ffmpeg; không thể cài vì Internet tắt"

GPU_WORKERS="$("$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA không khả dụng. Hãy bật GPU trong Kaggle Notebook Settings.")
print("GPU: " + ", ".join(torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())), file=sys.stderr)
print(min(2, torch.cuda.device_count()))
PY
)"

shopt -s nullglob
offline_wheels=("${OFFLINE_WHEEL_ROOT}"/*.whl)
shopt -u nullglob
(( ${#offline_wheels[@]} > 0 )) || die "offline_wheels không chứa file .whl"
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    "$PYTHON_BIN" -m pip install -q --no-index --no-deps --upgrade \
    --target "$OFFLINE_SITE_PACKAGES" "${offline_wheels[@]}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${OFFLINE_SITE_PACKAGES}:${PIPELINE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${WORK_ROOT}/matplotlib"

"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

import cv2
import einops
import ffmpeg
import matplotlib
import numpy
import safetensors
import torch
import yaml
from PIL import Image
from torchvision.ops import roi_align
from transformers import AutoImageProcessor, AutoModelForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import auto_docstring

if not version("transformers").startswith("4.57."):
    raise SystemExit(f"Cần transformers 4.57.x, đang có {version('transformers')}")
print("Offline Python dependencies: OK")
PY

find_model_root() {
    local model_file candidate
    if [[ -n "${AIC_OFFLINE_MODEL_ROOT:-}" ]]; then
        printf '%s\n' "$AIC_OFFLINE_MODEL_ROOT"
        return 0
    fi
    while IFS= read -r model_file; do
        candidate="$(dirname "$(dirname "$model_file")")"
        if [[ -f "${candidate}/autoshot/ckpt_0_200_0.pth" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$KAGGLE_INPUT_ROOT" -type f -path '*/fgclip2/model.safetensors' -print | sort)
    return 1
}

find_l26_b_root() {
    local candidate
    if [[ -n "${AIC_L26_B_VIDEO_ROOT:-}" ]]; then
        printf '%s\n' "$AIC_L26_B_VIDEO_ROOT"
        return 0
    fi
    while IFS= read -r candidate; do
        if [[ -f "${candidate}/${START_VIDEO_ID}.mp4" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$KAGGLE_INPUT_ROOT" -type d -path '*/Videos_L26_b/video' -print | sort)
    return 1
}

MODEL_ROOT="$(find_model_root)" || die "Không tìm thấy model offline aic_l28_offline_models"
MODEL_ROOT="$(realpath "$MODEL_ROOT")"
AUTOSHOT_ROOT="${MODEL_ROOT}/autoshot"
FGCLIP2_ROOT="${MODEL_ROOT}/fgclip2"
CHECKPOINT="${AUTOSHOT_ROOT}/ckpt_0_200_0.pth"
L26_B_VIDEO_ROOT="$(find_l26_b_root)" || die "Không tìm thấy ${START_VIDEO_ID}.mp4 trong Videos_L26_b/video"
L26_B_VIDEO_ROOT="$(realpath "$L26_B_VIDEO_ROOT")"

for required in \
    "${AUTOSHOT_ROOT}/supernet_flattransf_3_8_8_8_13_12_0_16_60.py" \
    "${AUTOSHOT_ROOT}/linear.py" \
    "${AUTOSHOT_ROOT}/utils.py" \
    "$CHECKPOINT" \
    "${FGCLIP2_ROOT}/config.json" \
    "${FGCLIP2_ROOT}/preprocessor_config.json" \
    "${FGCLIP2_ROOT}/modeling_fgclip2.py" \
    "${FGCLIP2_ROOT}/configuration_fgclip2.py" \
    "${FGCLIP2_ROOT}/model.safetensors"; do
    [[ -s "$required" ]] || die "Thiếu model file: $required"
done
[[ "$(stat -c '%s' "$CHECKPOINT")" == "$CHECKPOINT_SIZE" ]] || die "Checkpoint AutoShot sai kích thước"
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA256" ]] || die "Checkpoint AutoShot sai SHA256"
[[ "$(stat -c '%s' "${FGCLIP2_ROOT}/model.safetensors")" == "$FGCLIP2_MODEL_SIZE" ]] || die "FG-CLIP2 model.safetensors sai kích thước"

"$PYTHON_BIN" - "${PIPELINE_ROOT}/configs/default.yaml" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
assert config["indexer"]["sample_every_frames"] == 10
assert config["autoshot"]["threshold"] == 0.296
assert config["autoshot"]["allow_fallback"] is False
assert config["embedding"]["provider"] == "fgclip2"
assert config["embedding"]["batch_size"] == 4
assert config["similarity"]["threshold"] == 0.5
print("Logic cũ: sample=10, AutoShot=0.296, FG-CLIP2 batch=4, cosine=0.5")
PY

echo "============================================================"
echo "L26_b offline: ${START_VIDEO_ID} -> ${END_VIDEO_ID}"
echo "Video root: $L26_B_VIDEO_ROOT"
echo "Model root: $MODEL_ROOT"
echo "Output: $RESULT_ROOT"
echo "GPU workers: $GPU_WORKERS"
echo "Không dùng Internet, ZIP, hoặc kết quả cũ."
echo "============================================================"

"$PYTHON_BIN" "${PIPELINE_ROOT}/scripts/kaggle_directory_runner.py" \
    --pipeline-root "$PIPELINE_ROOT" \
    --config "${PIPELINE_ROOT}/configs/default.yaml" \
    --model-path "$FGCLIP2_ROOT" \
    --autoshot-root "$AUTOSHOT_ROOT" \
    --autoshot-checkpoint "$CHECKPOINT" \
    --video-root "$L26_B_VIDEO_ROOT" \
    --level L26 \
    --start-at-video "$START_VIDEO_ID" \
    --end-at-video "$END_VIDEO_ID" \
    --work-root "$WORK_ROOT" \
    --result-root "$RESULT_ROOT" \
    --gpu-workers "$GPU_WORKERS"

echo "Hoàn tất. Kết quả: $RESULT_ROOT"
find "$RESULT_ROOT" -maxdepth 1 -type f -name 'L26_V???.tar' -printf '%f\n' | sort
