#!/usr/bin/env bash
set -Eeuo pipefail

# Fully offline Kaggle launcher for the original L26 pipeline.  L26_d/e are
# already mounted as Kaggle Inputs, so no ZIP download or MP4 extraction is
# performed.  AutoShot and FG-CLIP2 weights are read directly from the model
# Input; Python wheels bundled with this source provide the pinned FG-CLIP2
# runtime without contacting PyPI or Hugging Face.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

KAGGLE_INPUT_ROOT="/kaggle/input"
KAGGLE_WORKING_ROOT="/kaggle/working"
RUNTIME_ROOT="${AIC_RUNTIME_ROOT:-/tmp/aic_l26_de_offline_runtime}"
WORK_ROOT="${RUNTIME_ROOT}/work"
OFFLINE_SITE_PACKAGES="${RUNTIME_ROOT}/python_packages"
OFFLINE_WHEEL_ROOT="${PIPELINE_ROOT}/offline_wheels"
RESULT_ROOT="${AIC_RESULT_ROOT:-${KAGGLE_WORKING_ROOT}/aic_l26_de_offline_results}"

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
    if (( status == 0 )) && [[ "$RUNTIME_ROOT" == "/tmp/aic_l26_de_offline_runtime" ]]; then
        # Only delete the fixed temporary location. Final TARs are already in
        # RESULT_ROOT and all Kaggle Input data remains untouched.
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

# No network and no dependency resolution: all required wheels are shipped in
# the source dataset.  --target keeps Kaggle's global Python untouched.
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

find_video_root() {
    local part="$1" candidate
    if [[ "$part" == "d" && -n "${AIC_L26_D_VIDEO_ROOT:-}" ]]; then
        printf '%s\n' "$AIC_L26_D_VIDEO_ROOT"
        return 0
    fi
    if [[ "$part" == "e" && -n "${AIC_L26_E_VIDEO_ROOT:-}" ]]; then
        printf '%s\n' "$AIC_L26_E_VIDEO_ROOT"
        return 0
    fi
    while IFS= read -r candidate; do
        if find "$candidate" -maxdepth 1 -type f -iname 'L26_V*.mp4' -print -quit | grep -q .; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$KAGGLE_INPUT_ROOT" -type d -path "*/Videos_L26_${part}/video" -print | sort)
    while IFS= read -r candidate; do
        if find "$candidate" -maxdepth 1 -type f -iname 'L26_V*.mp4' -print -quit | grep -q .; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$KAGGLE_INPUT_ROOT" -type d -path "*/Videos_L26_${part}" -print | sort)
    return 1
}

MODEL_ROOT="$(find_model_root)" || die "Không tìm thấy model offline aic_l28_offline_models"
MODEL_ROOT="$(realpath "$MODEL_ROOT")"
AUTOSHOT_ROOT="${MODEL_ROOT}/autoshot"
FGCLIP2_ROOT="${MODEL_ROOT}/fgclip2"
CHECKPOINT="${AUTOSHOT_ROOT}/ckpt_0_200_0.pth"
L26_D_VIDEO_ROOT="$(find_video_root d)" || die "Không tìm thấy MP4 trực tiếp trong Videos_L26_d/video"
L26_E_VIDEO_ROOT="$(find_video_root e)" || die "Không tìm thấy MP4 trực tiếp trong Videos_L26_e/video"

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
echo "L26_d và L26_e — offline, direct Kaggle Input"
echo "Video d: $L26_D_VIDEO_ROOT"
echo "Video e: $L26_E_VIDEO_ROOT"
echo "Model  : $MODEL_ROOT"
echo "Output : $RESULT_ROOT"
echo "GPU workers: $GPU_WORKERS"
echo "Không wget, apt-get, Hugging Face, hoặc dùng kết quả cũ."
echo "============================================================"

"$PYTHON_BIN" "${PIPELINE_ROOT}/scripts/kaggle_directory_runner.py" \
    --pipeline-root "$PIPELINE_ROOT" \
    --config "${PIPELINE_ROOT}/configs/default.yaml" \
    --model-path "$FGCLIP2_ROOT" \
    --autoshot-root "$AUTOSHOT_ROOT" \
    --autoshot-checkpoint "$CHECKPOINT" \
    --video-root "$L26_D_VIDEO_ROOT" \
    --video-root "$L26_E_VIDEO_ROOT" \
    --level L26 \
    --work-root "$WORK_ROOT" \
    --result-root "$RESULT_ROOT" \
    --gpu-workers "$GPU_WORKERS"

echo "Hoàn tất. Kết quả: $RESULT_ROOT"
find "$RESULT_ROOT" -maxdepth 1 -type f -name 'L26_V???.tar' -printf '%f\n' | sort
