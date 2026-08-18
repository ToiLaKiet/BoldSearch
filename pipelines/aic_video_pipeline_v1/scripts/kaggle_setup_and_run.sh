#!/usr/bin/env bash
set -Eeuo pipefail

KAGGLE_INPUT_ROOT="/kaggle/input"
KAGGLE_WORKING_ROOT="/kaggle/working"
RUNTIME_ROOT="$(mktemp -d -p /tmp aic_video_pipeline_runtime.XXXXXX)"
PIPELINE_ROOT="${RUNTIME_ROOT}/aic_video_pipeline_v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
AUTOSHOT_ROOT="${RUNTIME_ROOT}/AutoShot"
FGCLIP2_ROOT="${RUNTIME_ROOT}/models/fgclip2"
HF_CACHE_ROOT="${RUNTIME_ROOT}/hf_cache"
WORK_ROOT="${RUNTIME_ROOT}/work"
RESULT_ROOT="${KAGGLE_WORKING_ROOT}/aic_pipeline_results"

OFFLINE_MODEL_ROOT="${AIC_OFFLINE_MODEL_ROOT:-}"
USE_OFFLINE_MODEL_ASSETS=0
CHECKPOINT_FILE="ckpt_0_200_0.pth"
CHECKPOINT="${AUTOSHOT_ROOT}/${CHECKPOINT_FILE}"
CHECKPOINT_SIZE="57243097"
CHECKPOINT_SHA256="3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6"
FGCLIP2_MODEL_SIZE="3586589392"

cleanup_runtime() {
    local status=$?
    trap - EXIT

    # A .tar.tmp is never a resumable result. Keeping it only makes a failed
    # Kaggle Save Version larger, so retain final TARs and runner_state only.
    if [[ -d "$RESULT_ROOT" ]]; then
        find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.tar.tmp' -delete \
            2>/dev/null || true
    fi

    # Runtime assets are reproducible and must never become Kaggle Output.
    if [[ "$RUNTIME_ROOT" == /tmp/aic_video_pipeline_runtime.* &&
          -d "$RUNTIME_ROOT" ]]; then
        find "$RUNTIME_ROOT" -xdev -depth -delete 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup_runtime EXIT

echo "Dung lượng filesystem trước khi chạy:"
df -h /tmp "$KAGGLE_WORKING_ROOT"

# An offline Kaggle session must receive this directory as an Input:
#   aic_l28_offline_models/{autoshot,fgclip2}
# When no explicit root is supplied, discover the uploaded bundle by its
# required FG-CLIP2 weight and verify that its matching AutoShot files exist.
if [[ -z "$OFFLINE_MODEL_ROOT" ]]; then
    offline_weight="$(
        find "$KAGGLE_INPUT_ROOT" -type f \
            -path '*/fgclip2/model.safetensors' -print -quit
    )"
    if [[ -n "$offline_weight" ]]; then
        candidate_root="$(dirname "$(dirname "$offline_weight")")"
        if [[ -f "${candidate_root}/autoshot/${CHECKPOINT_FILE}" ]]; then
            OFFLINE_MODEL_ROOT="$candidate_root"
        fi
    fi
fi

if [[ -n "$OFFLINE_MODEL_ROOT" ]]; then
    if [[ ! -d "$OFFLINE_MODEL_ROOT" ]]; then
        echo "AIC_OFFLINE_MODEL_ROOT không tồn tại: $OFFLINE_MODEL_ROOT"
        exit 1
    fi
    OFFLINE_MODEL_ROOT="$(cd "$OFFLINE_MODEL_ROOT" && pwd -P)"
    AUTOSHOT_ROOT="${OFFLINE_MODEL_ROOT}/autoshot"
    FGCLIP2_ROOT="${OFFLINE_MODEL_ROOT}/fgclip2"
    CHECKPOINT="${AUTOSHOT_ROOT}/${CHECKPOINT_FILE}"

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
        [[ -s "$required" ]] || {
            echo "Thiếu file model ngoại tuyến: $required"
            exit 1
        }
    done
    actual_size="$(stat -c '%s' "$CHECKPOINT")"
    actual_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    if [[ "$actual_size" != "$CHECKPOINT_SIZE" ||
          "$actual_sha" != "$CHECKPOINT_SHA256" ]]; then
        echo "Checkpoint AutoShot trong model Input không đúng phiên bản."
        exit 1
    fi
    if [[ "$(stat -c '%s' "${FGCLIP2_ROOT}/model.safetensors")" != "$FGCLIP2_MODEL_SIZE" ]]; then
        echo "model.safetensors trong model Input không đúng kích thước."
        exit 1
    fi
    if [[ -f "${OFFLINE_MODEL_ROOT}/MANIFEST.sha256" ]]; then
        (
            cd "$OFFLINE_MODEL_ROOT"
            sha256sum -c MANIFEST.sha256
        ) || {
            echo "Kiểm tra toàn vẹn model Input thất bại."
            exit 1
        }
    fi
    USE_OFFLINE_MODEL_ASSETS=1
    echo "Dùng model ngoại tuyến: $OFFLINE_MODEL_ROOT"
fi

GPU_WORKERS="$(python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA không khả dụng. Hãy bật GPU trong Kaggle Settings.")
print("GPU:", ", ".join(torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())), file=__import__('sys').stderr)
print(min(2, torch.cuda.device_count()))
PY
)"
echo "Số GPU worker thường trú: ${GPU_WORKERS}"

missing_packages=()
command -v ffmpeg >/dev/null || missing_packages+=(ffmpeg)
if (( USE_OFFLINE_MODEL_ASSETS == 0 )); then
    command -v wget >/dev/null || missing_packages+=(wget)
fi
if (( ${#missing_packages[@]} > 0 )); then
    if (( USE_OFFLINE_MODEL_ASSETS == 1 )); then
        echo "Kaggle offline đang thiếu lệnh hệ thống: ${missing_packages[*]}"
        exit 1
    fi
    apt-get -qq update
    apt-get -qq install -y "${missing_packages[@]}"
fi

if [[ -f "${SCRIPT_PIPELINE_ROOT}/configs/default.yaml" ]]; then
    UPLOADED_PIPELINE_ROOT="$SCRIPT_PIPELINE_ROOT"
else
    UPLOADED_CONFIG="$(
        find "$KAGGLE_INPUT_ROOT" \
            -type f \
            -path '*/aic_video_pipeline_v1/configs/default.yaml' \
            -print -quit
    )"
    if [[ -z "$UPLOADED_CONFIG" ]]; then
        echo "Không tìm thấy source aic_video_pipeline_v1 đã giải nén."
        exit 1
    fi
    UPLOADED_PIPELINE_ROOT="$(dirname "$(dirname "$UPLOADED_CONFIG")")"
fi

mkdir -p "$PIPELINE_ROOT" "$HF_CACHE_ROOT"
cp -a "${UPLOADED_PIPELINE_ROOT}/." "${PIPELINE_ROOT}/"

if (( USE_OFFLINE_MODEL_ASSETS == 1 )); then
    # Both Kaggle runners add this source tree to sys.path themselves.  Avoid an
    # editable package build here: even ``--no-deps`` starts PEP 517 isolation
    # and tries to download setuptools, which fails without Internet.
    export PYTHONPATH="${PIPELINE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

    # Import the APIs actually used by AutoShot and FG-CLIP2.  This is more
    # useful than letting model initialization fail several minutes later.
    python - <<'PY'
errors = []

for label, statement in (
    ("opencv-python", "import cv2"),
    ("einops", "import einops"),
    ("numpy", "import numpy"),
    ("Pillow", "from PIL import Image"),
    ("PyYAML", "import yaml"),
    ("safetensors", "import safetensors"),
    ("torch", "import torch"),
    ("torchvision", "from torchvision.ops import roi_align"),
    ("huggingface-hub", "import huggingface_hub"),
    (
        "transformers (FG-CLIP2 APIs)",
        "; ".join((
            "from transformers import AutoImageProcessor, AutoModelForCausalLM",
            "from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask",
            "from transformers.modeling_layers import GradientCheckpointingLayer",
            "from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel",
            "from transformers.processing_utils import Unpack",
            "from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, filter_out_non_signature_kwargs",
            "from transformers.utils.generic import check_model_inputs",
        )),
    ),
):
    try:
        exec(statement, {})
    except Exception as error:
        errors.append(f"{label}: {type(error).__name__}: {error}")

if errors:
    raise SystemExit(
        "Kaggle image thiếu hoặc không tương thích dependency offline:\n- "
        + "\n- ".join(errors)
    )

print("Python dependencies cho AutoShot/FG-CLIP2: OK")
PY
else
    PIP_NO_CACHE_DIR=1 python -m pip install -q \
        -e "${PIPELINE_ROOT}[fgclip2]"
fi

if (( USE_OFFLINE_MODEL_ASSETS == 1 )); then
    mkdir -p "$WORK_ROOT" "$RESULT_ROOT"
else
    mkdir -p "$AUTOSHOT_ROOT" "$FGCLIP2_ROOT" "$WORK_ROOT" "$RESULT_ROOT"
fi
find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.tar.tmp' -delete \
    2>/dev/null || true

if (( USE_OFFLINE_MODEL_ASSETS == 0 )); then
    RAW_BASE="https://raw.githubusercontent.com/Longhehehe/AIC/master"
    for source_name in \
        supernet_flattransf_3_8_8_8_13_12_0_16_60.py \
        linear.py \
        utils.py; do
        if [[ ! -s "${AUTOSHOT_ROOT}/${source_name}" ]]; then
            wget -q "${RAW_BASE}/${source_name}" \
                -O "${AUTOSHOT_ROOT}/${source_name}.tmp"
            mv "${AUTOSHOT_ROOT}/${source_name}.tmp" \
               "${AUTOSHOT_ROOT}/${source_name}"
        fi
    done

    checkpoint_valid=0
    if [[ -f "$CHECKPOINT" ]]; then
        actual_size="$(stat -c '%s' "$CHECKPOINT")"
        actual_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
        if [[ "$actual_size" == "$CHECKPOINT_SIZE" &&
              "$actual_sha" == "$CHECKPOINT_SHA256" ]]; then
            checkpoint_valid=1
        fi
    fi

    if (( checkpoint_valid == 0 )); then
        wget --progress=dot:giga \
            "${RAW_BASE}/ckpt_0_200_0.pth" \
            -O "${CHECKPOINT}.tmp"
        actual_size="$(stat -c '%s' "${CHECKPOINT}.tmp")"
        actual_sha="$(sha256sum "${CHECKPOINT}.tmp" | awk '{print $1}')"
        if [[ "$actual_size" != "$CHECKPOINT_SIZE" ||
              "$actual_sha" != "$CHECKPOINT_SHA256" ]]; then
            echo "Checkpoint AutoShot tải về không hợp lệ."
            exit 1
        fi
        mv "${CHECKPOINT}.tmp" "$CHECKPOINT"
    fi

    fgclip2_size="0"
    if [[ -f "${FGCLIP2_ROOT}/model.safetensors" ]]; then
        fgclip2_size="$(stat -c '%s' "${FGCLIP2_ROOT}/model.safetensors")"
    fi

    if [[ "$fgclip2_size" != "$FGCLIP2_MODEL_SIZE" ||
          ! -s "${FGCLIP2_ROOT}/config.json" ||
          ! -s "${FGCLIP2_ROOT}/preprocessor_config.json" ]]; then
        HF_HOME="$HF_CACHE_ROOT" FGCLIP2_DOWNLOAD_DIR="$FGCLIP2_ROOT" \
        python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="qihoo360/fg-clip2-large",
    revision="4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3",
    local_dir=os.environ["FGCLIP2_DOWNLOAD_DIR"],
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

previous_roots=()
add_previous_root() {
    local candidate="$1"
    for existing in "${previous_roots[@]}"; do
        [[ "$existing" == "$candidate" ]] && return
    done
    previous_roots+=("$candidate")
}

if [[ -n "${AIC_PREVIOUS_RESULT_ROOT:-}" ]]; then
    if [[ ! -d "$AIC_PREVIOUS_RESULT_ROOT" ]]; then
        echo "Không tìm thấy AIC_PREVIOUS_RESULT_ROOT: $AIC_PREVIOUS_RESULT_ROOT"
        exit 1
    fi
    add_previous_root "$AIC_PREVIOUS_RESULT_ROOT"
fi

while IFS= read -r previous_root; do
    add_previous_root "$previous_root"
done < <(
    find "$KAGGLE_INPUT_ROOT" \
        -type f \
        -name 'L??_V???.tar' \
        -printf '%h\n' | sort -u
)

# Kaggle may automatically expose every saved Lxx_Vxxx.tar as a directory
# named Lxx_Vxxx. Detect only complete extracted payloads, not pipeline work
# directories or checkpoints that happened to survive an interrupted session.
while IFS= read -r extracted_root; do
    add_previous_root "$extracted_root"
done < <(
    find "$KAGGLE_INPUT_ROOT" -type d -name 'L??_V???' -print0 |
    while IFS= read -r -d '' candidate; do
        video_id="$(basename "$candidate")"
        checkpoint="${candidate}/checkpoints/${video_id}.json"
        nested_metadata="${candidate}/metadata/${video_id}"
        flat_metadata="${candidate}/metadata"
        if [[ -s "$checkpoint" ]] && {
            [[ -s "${nested_metadata}/Shot.json" &&
               -s "${nested_metadata}/Frame.json" ]] ||
            [[ -s "${flat_metadata}/Shot.json" &&
               -s "${flat_metadata}/Frame.json" ]]
        }; then
            dirname "$candidate"
        fi
    done | sort -u
)

previous_args=()
for previous_root in "${previous_roots[@]}"; do
    previous_args+=(--previous-result-root "$previous_root")
    python - "$previous_root" <<'PY'
import tarfile
import sys
from pathlib import Path

root = Path(sys.argv[1])

def valid(path: Path) -> bool:
    video_id = path.stem
    try:
        with tarfile.open(path, "r") as archive:
            names = set(archive.getnames())
        return {
            f"metadata/{video_id}/Shot.json",
            f"metadata/{video_id}/Frame.json",
            f"checkpoints/{video_id}.json",
        }.issubset(names)
    except (OSError, tarfile.TarError):
        return False

def valid_directory(path: Path) -> bool:
    video_id = path.name
    checkpoint = path / "checkpoints" / f"{video_id}.json"
    layouts = (
        (path / "metadata" / video_id,
         path / "frames" / video_id,
         path / "vectors" / video_id),
        (path / "metadata", path / "frames", path / "vectors"),
    )
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False
    for metadata, frames, vectors in layouts:
        if not ((metadata / "Shot.json").is_file()
                and (metadata / "Frame.json").is_file()
                and frames.is_dir() and vectors.is_dir()):
            continue
        if (any(frames.glob("*.png")) and any(vectors.glob("*.npy"))):
            return True
    return False

final_tars = sorted(root.glob("L??_V???.tar"))
valid_tars = [path.name for path in final_tars if valid(path)]
invalid_tars = [path.name for path in final_tars if not valid(path)]
temporary_tars = sorted(path.name for path in root.glob("L??_V???.tar.tmp"))
extracted = sorted(path.name for path in root.glob("L??_V???")
                   if valid_directory(path))
print(f"[PREVIOUS RESULT] {root}")
print(f"  final TAR hợp lệ: {len(valid_tars)}")
print(f"  thư mục TAR đã extract hợp lệ: {len(extracted)}")
print(f"  final TAR lỗi: {len(invalid_tars)}")
print(f"  TAR tạm sẽ chạy lại: {len(temporary_tars)}")
if invalid_tars:
    print("  TAR lỗi:", ", ".join(invalid_tars[:10]))
if temporary_tars:
    print("  TAR tạm:", ", ".join(temporary_tars[:10]))
PY
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${WORK_ROOT}/matplotlib"

direct_video_roots=()
if [[ -n "${AIC_VIDEO_ROOTS:-}" ]]; then
    IFS=':' read -r -a direct_video_roots <<< "$AIC_VIDEO_ROOTS"
elif [[ -n "${AIC_VIDEO_ROOT:-}" ]]; then
    direct_video_roots=("$AIC_VIDEO_ROOT")
fi

if (( ${#direct_video_roots[@]} > 0 )); then
    video_root_args=()
    for direct_video_root in "${direct_video_roots[@]}"; do
        if [[ -z "$direct_video_root" || ! -d "$direct_video_root" ]]; then
            echo "Không tìm thấy thư mục MP4 Input: $direct_video_root"
            echo "Hãy kiểm tra AIC_VIDEO_ROOT hoặc AIC_VIDEO_ROOTS."
            exit 1
        fi
        video_root_args+=(--video-root "$direct_video_root")
    done
    if [[ ! "${AIC_VIDEO_LEVEL:-}" =~ ^L[0-9]{2}$ ]]; then
        echo "AIC_VIDEO_LEVEL phải có dạng Lxx, ví dụ L28."
        exit 1
    fi
    echo "Chạy tuần tự MP4 từ ${#direct_video_roots[@]} thư mục Kaggle Input:"
    printf '  - %s\n' "${direct_video_roots[@]}"
    echo "Mỗi video xong được đóng gói TAR và xóa artifact tạm trước video kế tiếp."
    python "${PIPELINE_ROOT}/scripts/kaggle_directory_runner.py" \
        --pipeline-root "$PIPELINE_ROOT" \
        --config "${PIPELINE_ROOT}/configs/default.yaml" \
        --model-path "$FGCLIP2_ROOT" \
        --autoshot-root "$AUTOSHOT_ROOT" \
        --autoshot-checkpoint "$CHECKPOINT" \
        "${video_root_args[@]}" \
        --level "$AIC_VIDEO_LEVEL" \
        --work-root "$WORK_ROOT" \
        --result-root "$RESULT_ROOT" \
        "${previous_args[@]}"
    echo "Kết quả: $RESULT_ROOT"
    du -sh "$RESULT_ROOT"
    exit 0
fi

archive_args=()
if [[ -n "${AIC_ARCHIVES:-}" ]]; then
    read -r -a requested_archives <<< "$AIC_ARCHIVES"
    if (( ${#requested_archives[@]} == 0 )); then
        echo "AIC_ARCHIVES không chứa archive nào."
        exit 1
    fi
    for archive_name in "${requested_archives[@]}"; do
        if [[ ! "$archive_name" =~ ^Videos_L[0-9]{2}_[a-z]\.zip$ ]]; then
            echo "Tên archive không hợp lệ: $archive_name"
            exit 1
        fi
    done
    archive_args=(--archives "${requested_archives[@]}")
    echo "Chỉ chạy archive: ${requested_archives[*]}"
fi

python "${PIPELINE_ROOT}/scripts/kaggle_archive_runner.py" \
    --pipeline-root "$PIPELINE_ROOT" \
    --config "${PIPELINE_ROOT}/configs/default.yaml" \
    --model-path "$FGCLIP2_ROOT" \
    --autoshot-root "$AUTOSHOT_ROOT" \
    --autoshot-checkpoint "$CHECKPOINT" \
    --work-root "$WORK_ROOT" \
    --result-root "$RESULT_ROOT" \
    --gpu-workers "$GPU_WORKERS" \
    "${archive_args[@]}" \
    "${previous_args[@]}"

echo "Kết quả: $RESULT_ROOT"
du -sh "$RESULT_ROOT"
echo "Model, cache, source và work tạm sẽ được xóa trước khi cell kết thúc."
