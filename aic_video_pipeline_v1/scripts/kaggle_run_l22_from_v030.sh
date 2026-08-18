#!/usr/bin/env bash
set -Eeuo pipefail

# Kaggle launcher for the original streaming pipeline.  It deliberately does
# not import or inspect any older L22 result: this run starts again at V030.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

KAGGLE_INPUT_ROOT="/kaggle/input"
KAGGLE_WORKING_ROOT="/kaggle/working"
ARCHIVE_NAME="Videos_L22_a.zip"
START_VIDEO_ID="L22_V030"

RUNTIME_ROOT="${AIC_RUNTIME_ROOT:-/tmp/aic_l22_from_v030_runtime}"
WORK_ROOT="${RUNTIME_ROOT}/work"
RESULT_ROOT="${AIC_RESULT_ROOT:-${KAGGLE_WORKING_ROOT}/aic_l22_from_v030_results}"

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
    if (( status == 0 )) && [[ "$RUNTIME_ROOT" == "/tmp/aic_l22_from_v030_runtime" ]]; then
        # Final TAR files have already been written to RESULT_ROOT.  Remove
        # only this known temporary directory, never a caller-supplied root.
        find "$RUNTIME_ROOT" -xdev -depth -delete 2>/dev/null || true
    elif (( status != 0 )); then
        echo "Lần chạy dừng lỗi; dữ liệu tạm được giữ để wget -c có thể tiếp tục: $RUNTIME_ROOT" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

[[ -f "${PIPELINE_ROOT}/configs/default.yaml" ]] || die "Thiếu configs/default.yaml"
[[ -f "${PIPELINE_ROOT}/scripts/kaggle_archive_runner.py" ]] || die "Thiếu archive runner"
[[ -d "${PIPELINE_ROOT}/src/aic_video_pipeline_v1" ]] || die "Thiếu source pipeline"
[[ -d "$KAGGLE_INPUT_ROOT" && -d "$KAGGLE_WORKING_ROOT" ]] || die "Script này chỉ chạy trên Kaggle Notebook"

mkdir -p "$WORK_ROOT" "$RESULT_ROOT"
find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.tar.tmp' -delete 2>/dev/null || true

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

missing_commands=()
command -v wget >/dev/null || missing_commands+=(wget)
command -v ffmpeg >/dev/null || missing_commands+=(ffmpeg)
if (( ${#missing_commands[@]} > 0 )); then
    apt-get -qq update
    apt-get -qq install -y "${missing_commands[@]}"
fi

runtime_ready() {
    "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

import cv2
import einops
import ffmpeg
import matplotlib
import numpy
import torch
import yaml
from PIL import Image
from torchvision.ops import roi_align
from transformers import AutoImageProcessor, AutoModelForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import auto_docstring

major, minor, *_ = (int(value) for value in version("transformers").split(".")[:2])
if (major, minor) != (4, 57):
    raise RuntimeError(f"transformers {version('transformers')} không tương thích; cần 4.57.x")
print("Python dependencies: OK")
PY
}

# Kaggle needs Internet enabled anyway to download the L22 ZIP.  Only repair
# Python packages when its base image is not compatible; do not download model
# weights because those are supplied by the attached offline-model dataset.
if ! runtime_ready; then
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
        "$PYTHON_BIN" -m pip install -q --upgrade \
        "transformers==4.57.6" "huggingface-hub>=0.34,<1" \
        "ffmpeg-python==0.2.0" "matplotlib>=3.8" "einops>=0.7"
    runtime_ready
fi

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

MODEL_ROOT="$(find_model_root)" || die "Không tìm thấy model offline. Hãy attach dataset chứa aic_l28_offline_models (autoshot/ và fgclip2/)."
MODEL_ROOT="$(realpath "$MODEL_ROOT")"
AUTOSHOT_ROOT="${MODEL_ROOT}/autoshot"
FGCLIP2_ROOT="${MODEL_ROOT}/fgclip2"
CHECKPOINT="${AUTOSHOT_ROOT}/ckpt_0_200_0.pth"

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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${WORK_ROOT}/matplotlib"
export PYTHONPATH="${PIPELINE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "============================================================"
echo "L22 chạy lại từ ${START_VIDEO_ID} (bao gồm V030)"
echo "ZIP tải về: https://aic-data.ledo.io.vn/${ARCHIVE_NAME}"
echo "Kết quả mới: ${RESULT_ROOT}"
echo "GPU workers: ${GPU_WORKERS}"
echo "Không dùng kết quả L22 cũ."
echo "============================================================"

"$PYTHON_BIN" "${PIPELINE_ROOT}/scripts/kaggle_archive_runner.py" \
    --pipeline-root "$PIPELINE_ROOT" \
    --config "${PIPELINE_ROOT}/configs/default.yaml" \
    --model-path "$FGCLIP2_ROOT" \
    --autoshot-root "$AUTOSHOT_ROOT" \
    --autoshot-checkpoint "$CHECKPOINT" \
    --work-root "$WORK_ROOT" \
    --result-root "$RESULT_ROOT" \
    --archives "$ARCHIVE_NAME" \
    --start-at-video "$START_VIDEO_ID" \
    --gpu-workers "$GPU_WORKERS"

echo "Hoàn tất. Kết quả: $RESULT_ROOT"
find "$RESULT_ROOT" -maxdepth 1 -type f -name 'L22_V???.tar' -printf '%f\n' | sort
