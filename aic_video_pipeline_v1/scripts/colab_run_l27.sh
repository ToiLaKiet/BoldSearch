#!/usr/bin/env bash
set -Eeuo pipefail

# Google Colab launcher for the original aic_video_pipeline_v1 flow.
# Processing semantics are kept in configs/default.yaml and run_streaming():
# sample every 10 source frames, AutoShot, FG-CLIP2 FP32 batch 4, cosine 0.5,
# and persist only KEPT PNG/NPY artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"

ARCHIVE_URL="https://aic-data.ledo.io.vn/Videos_L27_a.zip"
ARCHIVE_NAME="${ARCHIVE_URL##*/}"
ARCHIVE_BASE_URL="${ARCHIVE_URL%/${ARCHIVE_NAME}}"
EXPECTED_VIDEO_COUNT=16

DEFAULT_RUNTIME_ROOT="/content/aic_l27_runtime"
DEFAULT_RESULT_ROOT="/content/drive/MyDrive/aic_pipeline_results/L27"
RUNTIME_ROOT="${AIC_RUNTIME_ROOT:-$DEFAULT_RUNTIME_ROOT}"
WORK_ROOT="${RUNTIME_ROOT}/work"
MODEL_ROOT="${AIC_MODEL_ROOT:-${RUNTIME_ROOT}/models}"
HF_CACHE_ROOT="${RUNTIME_ROOT}/hf_cache"
RESULT_ROOT="${AIC_RESULT_ROOT:-$DEFAULT_RESULT_ROOT}"

AUTOSHOT_ROOT="${MODEL_ROOT}/autoshot"
FGCLIP2_ROOT="${MODEL_ROOT}/fgclip2"
CHECKPOINT="${AUTOSHOT_ROOT}/ckpt_0_200_0.pth"

CHECKPOINT_SIZE="57243097"
CHECKPOINT_SHA256="3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6"
AUTOSHOT_MODEL_SHA256="8a3fe9cf079a5020950758e67e7ddf079cfac4b3fa916844d11eba6226ff7382"
AUTOSHOT_LINEAR_SHA256="bc5eb81c94c142625435fb8b85b9ecc815c3507be67733dcfb83c12f1c79a885"
AUTOSHOT_UTILS_SHA256="75cb1415b1305b47b8aab7676313a3f08ff0fff8d8ec62ce41ec770116d1187a"
FGCLIP2_MODEL_SIZE="3586589392"
FGCLIP2_REVISION="4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null || {
        echo "Không tìm thấy PYTHON_BIN: $PYTHON_BIN" >&2
        exit 1
    }
elif command -v python >/dev/null; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null; then
    PYTHON_BIN="python3"
elif [[ -x /home/long/miniconda3/bin/python ]]; then
    PYTHON_BIN="/home/long/miniconda3/bin/python"
else
    echo "Không tìm thấy Python để chạy pipeline." >&2
    exit 1
fi

die() {
    echo "LỖI: $*" >&2
    exit 1
}

cleanup_on_exit() {
    local status=$?
    trap - EXIT

    # A partial TAR is not a valid resumable result. Completed TAR files and
    # runner_state.json on Drive are deliberately retained.
    if [[ -d "$RESULT_ROOT" ]]; then
        find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.tar.tmp' -delete \
            2>/dev/null || true
    fi

    if (( status == 0 )) && [[ "${AIC_KEEP_RUNTIME:-0}" != "1" ]]; then
        # Only auto-delete the known Colab runtime path. A custom runtime root
        # belongs to the caller and is never removed implicitly.
        if [[ "$RUNTIME_ROOT" == "$DEFAULT_RUNTIME_ROOT" && -d "$RUNTIME_ROOT" ]]; then
            find "$RUNTIME_ROOT" -xdev -depth -delete 2>/dev/null || true
        fi
    elif (( status != 0 )); then
        echo
        echo "Phiên chạy bị lỗi; giữ dữ liệu tạm để chạy tiếp trong cùng session:"
        echo "  $RUNTIME_ROOT"
        echo "Các TAR hoàn chỉnh vẫn nằm tại: $RESULT_ROOT"
    fi

    exit "$status"
}
trap cleanup_on_exit EXIT

[[ -f "${PIPELINE_ROOT}/configs/default.yaml" ]] || \
    die "Không tìm thấy configs/default.yaml trong $PIPELINE_ROOT"
[[ -f "${PIPELINE_ROOT}/scripts/kaggle_archive_runner.py" ]] || \
    die "Không tìm thấy archive runner của pipeline cũ."
[[ -d "${PIPELINE_ROOT}/src/aic_video_pipeline_v1" ]] || \
    die "Không tìm thấy source package aic_video_pipeline_v1."

case "$(realpath -m "$RESULT_ROOT")/" in
    "$(realpath -m "$RUNTIME_ROOT")/"*)
        die "AIC_RESULT_ROOT phải nằm ngoài runtime tạm để không bị xóa."
        ;;
esac

if [[ "$RESULT_ROOT" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    die "Google Drive chưa được mount. Hãy chạy drive.mount('/content/drive') trước."
fi

mkdir -p "$WORK_ROOT" "$MODEL_ROOT" "$HF_CACHE_ROOT" "$RESULT_ROOT"
find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.tar.tmp' -delete \
    2>/dev/null || true
find "${WORK_ROOT}/video" -maxdepth 1 -type f -name '.*.part' -delete \
    2>/dev/null || true

echo "============================================================"
echo "AIC video pipeline V1 — Google Colab — L27"
echo "Source ZIP : $ARCHIVE_URL"
echo "Work local: $WORK_ROOT"
echo "Output    : $RESULT_ROOT"
echo "============================================================"
df -h /content "$RESULT_ROOT" 2>/dev/null || true

# Do not download models or the 2.54 GB archive again if all 16 final TARs on
# Drive are already structurally complete.
completed_count="$("$PYTHON_BIN" - "$RESULT_ROOT" "$EXPECTED_VIDEO_COUNT" <<'PY'
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
expected_count = int(sys.argv[2])

def valid(path: Path, video_id: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with tarfile.open(path, "r") as archive:
            members = archive.getmembers()
        required = {
            f"metadata/{video_id}/Shot.json",
            f"metadata/{video_id}/Frame.json",
            f"checkpoints/{video_id}.json",
        }
        names = {member.name for member in members
                 if member.isfile() and member.size > 0}
        has_png = any(name.startswith(f"frames/{video_id}/")
                      and name.endswith(".png") for name in names)
        has_npy = any(name.startswith(f"vectors/{video_id}/")
                      and name.endswith(".npy") for name in names)
        return required.issubset(names) and has_png and has_npy
    except (OSError, tarfile.TarError):
        return False

completed = 0
for number in range(1, expected_count + 1):
    video_id = f"L27_V{number:03d}"
    if valid(root / f"{video_id}.tar", video_id):
        completed += 1
print(completed)
PY
)"
echo "Kết quả L27 hoàn chỉnh đã có trên Drive: ${completed_count}/${EXPECTED_VIDEO_COUNT}"
if [[ "$completed_count" == "$EXPECTED_VIDEO_COUNT" ]]; then
    echo "L27 đã hoàn tất; không tải lại model hoặc video."
    exit 0
fi

GPU_WORKERS="$("$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA không khả dụng. Chọn Runtime > Change runtime type > T4 GPU."
    )
print(
    "GPU: " + ", ".join(
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ),
    file=sys.stderr,
)
print(min(2, torch.cuda.device_count()))
PY
)"
echo "Số GPU worker thường trú: $GPU_WORKERS"

missing_commands=()
command -v wget >/dev/null || missing_commands+=(wget)
command -v ffmpeg >/dev/null || missing_commands+=(ffmpeg)
if (( ${#missing_commands[@]} > 0 )); then
    apt-get -qq update
    apt-get -qq install -y "${missing_commands[@]}"
fi

# Preserve the dependency behavior of the old source. In particular its
# AutoShot utils imports ffmpeg-python and matplotlib, while FG-CLIP2 requires
# Transformers 4.57.x APIs.
PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
"$PYTHON_BIN" -m pip install -q -e "${PIPELINE_ROOT}[fgclip2]"

"$PYTHON_BIN" - <<'PY'
errors = []
for label, statement in (
    ("OpenCV", "import cv2"),
    ("einops", "import einops"),
    ("ffmpeg-python", "import ffmpeg"),
    ("matplotlib", "import matplotlib"),
    ("NumPy", "import numpy"),
    ("Pillow", "from PIL import Image"),
    ("PyYAML", "import yaml"),
    ("PyTorch", "import torch"),
    ("torchvision", "from torchvision.ops import roi_align"),
    (
        "Transformers FG-CLIP2 APIs",
        "; ".join((
            "from transformers import AutoImageProcessor, AutoModelForCausalLM",
            "from transformers.modeling_layers import GradientCheckpointingLayer",
            "from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS",
            "from transformers.utils import auto_docstring",
        )),
    ),
):
    try:
        exec(statement, {})
    except Exception as error:
        errors.append(f"{label}: {type(error).__name__}: {error}")
if errors:
    raise SystemExit("Dependency không tương thích:\n- " + "\n- ".join(errors))
print("Dependencies: OK")
PY

verify_sha256() {
    local path="$1"
    local expected="$2"
    [[ -s "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

if [[ -n "${AIC_OFFLINE_MODEL_ROOT:-}" ]]; then
    [[ -d "$AIC_OFFLINE_MODEL_ROOT" ]] || \
        die "AIC_OFFLINE_MODEL_ROOT không tồn tại: $AIC_OFFLINE_MODEL_ROOT"
    MODEL_ROOT="$(cd "$AIC_OFFLINE_MODEL_ROOT" && pwd -P)"
    AUTOSHOT_ROOT="${MODEL_ROOT}/autoshot"
    FGCLIP2_ROOT="${MODEL_ROOT}/fgclip2"
    CHECKPOINT="${AUTOSHOT_ROOT}/ckpt_0_200_0.pth"
    echo "Dùng model đã có: $MODEL_ROOT"
else
    mkdir -p "$AUTOSHOT_ROOT" "$FGCLIP2_ROOT"
    RAW_BASE="https://raw.githubusercontent.com/Longhehehe/AIC/master"
    for source_name in \
        supernet_flattransf_3_8_8_8_13_12_0_16_60.py \
        linear.py \
        utils.py; do
        case "$source_name" in
            supernet_flattransf_3_8_8_8_13_12_0_16_60.py)
                expected_source_sha="$AUTOSHOT_MODEL_SHA256"
                ;;
            linear.py)
                expected_source_sha="$AUTOSHOT_LINEAR_SHA256"
                ;;
            utils.py)
                expected_source_sha="$AUTOSHOT_UTILS_SHA256"
                ;;
        esac
        if ! verify_sha256 "${AUTOSHOT_ROOT}/${source_name}" \
                          "$expected_source_sha"; then
            echo "Tải AutoShot source: $source_name"
            wget -q "${RAW_BASE}/${source_name}" \
                -O "${AUTOSHOT_ROOT}/${source_name}.tmp"
            verify_sha256 "${AUTOSHOT_ROOT}/${source_name}.tmp" \
                          "$expected_source_sha" || \
                die "AutoShot source tải về sai SHA256: $source_name"
            mv "${AUTOSHOT_ROOT}/${source_name}.tmp" \
               "${AUTOSHOT_ROOT}/${source_name}"
        fi
    done

    if ! verify_sha256 "$CHECKPOINT" "$CHECKPOINT_SHA256"; then
        rm -f "${CHECKPOINT}.tmp"
        echo "Tải checkpoint AutoShot..."
        wget --progress=dot:giga \
            "${RAW_BASE}/ckpt_0_200_0.pth" \
            -O "${CHECKPOINT}.tmp"
        [[ "$(stat -c '%s' "${CHECKPOINT}.tmp")" == "$CHECKPOINT_SIZE" ]] || \
            die "Checkpoint tải về sai kích thước."
        verify_sha256 "${CHECKPOINT}.tmp" "$CHECKPOINT_SHA256" || \
            die "Checkpoint tải về sai SHA256."
        mv "${CHECKPOINT}.tmp" "$CHECKPOINT"
    fi

    fgclip2_size=0
    [[ -f "${FGCLIP2_ROOT}/model.safetensors" ]] && \
        fgclip2_size="$(stat -c '%s' "${FGCLIP2_ROOT}/model.safetensors")"
    if [[ "$fgclip2_size" != "$FGCLIP2_MODEL_SIZE" ||
          ! -s "${FGCLIP2_ROOT}/config.json" ||
          ! -s "${FGCLIP2_ROOT}/preprocessor_config.json" ||
          ! -s "${FGCLIP2_ROOT}/modeling_fgclip2.py" ||
          ! -s "${FGCLIP2_ROOT}/configuration_fgclip2.py" ]]; then
        echo "Tải FG-CLIP2 large tại revision đã khóa..."
        HF_HOME="$HF_CACHE_ROOT" \
        FGCLIP2_ROOT="$FGCLIP2_ROOT" \
        FGCLIP2_REVISION="$FGCLIP2_REVISION" \
        "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="qihoo360/fg-clip2-large",
    revision=os.environ["FGCLIP2_REVISION"],
    local_dir=os.environ["FGCLIP2_ROOT"],
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "modeling_fgclip2.py",
        "configuration_fgclip2.py",
    ],
)
PY
    fi
fi

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
    [[ -s "$required" ]] || die "Thiếu model: $required"
done
[[ "$(stat -c '%s' "$CHECKPOINT")" == "$CHECKPOINT_SIZE" ]] || \
    die "Checkpoint AutoShot sai kích thước."
verify_sha256 "$CHECKPOINT" "$CHECKPOINT_SHA256" || \
    die "Checkpoint AutoShot sai SHA256."
verify_sha256 \
    "${AUTOSHOT_ROOT}/supernet_flattransf_3_8_8_8_13_12_0_16_60.py" \
    "$AUTOSHOT_MODEL_SHA256" || die "AutoShot model source sai SHA256."
verify_sha256 "${AUTOSHOT_ROOT}/linear.py" "$AUTOSHOT_LINEAR_SHA256" || \
    die "AutoShot linear.py sai SHA256."
verify_sha256 "${AUTOSHOT_ROOT}/utils.py" "$AUTOSHOT_UTILS_SHA256" || \
    die "AutoShot utils.py sai SHA256."
[[ "$(stat -c '%s' "${FGCLIP2_ROOT}/model.safetensors")" == \
   "$FGCLIP2_MODEL_SIZE" ]] || die "FG-CLIP2 model.safetensors sai kích thước."

PYTHONPATH="$AUTOSHOT_ROOT" "$PYTHON_BIN" - <<'PY'
from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
from utils import get_batches, get_frames, predictions_to_scenes
print("AutoShot source/checkpoint: OK")
PY

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

previous_args=(--previous-result-root "$RESULT_ROOT")
if [[ -n "${AIC_PREVIOUS_RESULT_ROOT:-}" &&
      "$(realpath -m "$AIC_PREVIOUS_RESULT_ROOT")" != \
      "$(realpath -m "$RESULT_ROOT")" ]]; then
    [[ -d "$AIC_PREVIOUS_RESULT_ROOT" ]] || \
        die "AIC_PREVIOUS_RESULT_ROOT không tồn tại: $AIC_PREVIOUS_RESULT_ROOT"
    previous_args+=(--previous-result-root "$AIC_PREVIOUS_RESULT_ROOT")
fi

echo
echo "Bắt đầu tải và xử lý $ARCHIVE_NAME"
echo "Runner chỉ giải nén tối đa một MP4 cho mỗi GPU worker."
"$PYTHON_BIN" "${PIPELINE_ROOT}/scripts/kaggle_archive_runner.py" \
    --pipeline-root "$PIPELINE_ROOT" \
    --config "${PIPELINE_ROOT}/configs/default.yaml" \
    --model-path "$FGCLIP2_ROOT" \
    --autoshot-root "$AUTOSHOT_ROOT" \
    --autoshot-checkpoint "$CHECKPOINT" \
    --work-root "$WORK_ROOT" \
    --result-root "$RESULT_ROOT" \
    --base-url "$ARCHIVE_BASE_URL" \
    --archives "$ARCHIVE_NAME" \
    --gpu-workers "$GPU_WORKERS" \
    "${previous_args[@]}"

final_count="$(find "$RESULT_ROOT" -maxdepth 1 -type f \
    -name 'L27_V???.tar' | wc -l)"
echo
echo "Hoàn tất runner L27. Số TAR hiện có: $final_count"
echo "Kết quả: $RESULT_ROOT"
du -sh "$RESULT_ROOT" 2>/dev/null || true
sync
