#!/usr/bin/env bash
set -Eeuo pipefail

KAGGLE_INPUT_ROOT="/kaggle/input"
KAGGLE_WORKING_ROOT="/kaggle/working"
PIPELINE_ROOT="${KAGGLE_WORKING_ROOT}/aic_video_pipeline_v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
AUTOSHOT_ROOT="${KAGGLE_WORKING_ROOT}/AutoShot"
FGCLIP2_ROOT="${KAGGLE_WORKING_ROOT}/models/fgclip2"
WORK_ROOT="${KAGGLE_WORKING_ROOT}/aic_pipeline_work"
RESULT_ROOT="${KAGGLE_WORKING_ROOT}/aic_pipeline_results"

CHECKPOINT="${AUTOSHOT_ROOT}/ckpt_0_200_0.pth"
CHECKPOINT_SIZE="57243097"
CHECKPOINT_SHA256="3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6"
FGCLIP2_MODEL_SIZE="3586589392"

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
command -v wget >/dev/null || missing_packages+=(wget)
command -v ffmpeg >/dev/null || missing_packages+=(ffmpeg)
if (( ${#missing_packages[@]} > 0 )); then
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

mkdir -p "$PIPELINE_ROOT"
cp -a "${UPLOADED_PIPELINE_ROOT}/." "${PIPELINE_ROOT}/"

PIP_NO_CACHE_DIR=1 python -m pip install -q \
    -e "${PIPELINE_ROOT}[fgclip2]"

mkdir -p "$AUTOSHOT_ROOT" "$FGCLIP2_ROOT" "$WORK_ROOT" "$RESULT_ROOT"

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
    HF_HOME="${KAGGLE_WORKING_ROOT}/hf_cache" \
    python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="qihoo360/fg-clip2-large",
    revision="4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3",
    local_dir="/kaggle/working/models/fgclip2",
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
du -sh "$RESULT_ROOT" "$WORK_ROOT" "$FGCLIP2_ROOT"
