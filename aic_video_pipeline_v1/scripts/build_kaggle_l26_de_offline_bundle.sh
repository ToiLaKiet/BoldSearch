#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SOURCE="$(dirname "$SCRIPT_DIR")"
OUTPUT_PATH="$(dirname "$PIPELINE_SOURCE")/aic_video_pipeline_v1_l26_de_offline_kaggle.zip"
STAGING_ROOT="$(mktemp -d -t aic-l26-de-offline-bundle.XXXXXX)"
STAGED_PIPELINE="${STAGING_ROOT}/aic_video_pipeline_v1"
TEMP_ARCHIVE="${STAGING_ROOT}/aic_video_pipeline_v1_l26_de_offline_kaggle.zip"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null || {
        echo "Không tìm thấy PYTHON_BIN: $PYTHON_BIN" >&2
        exit 1
    }
elif command -v python >/dev/null; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Không tìm thấy Python để tạo ZIP." >&2
    exit 1
fi

cleanup_staging() {
    local status=$?
    trap - EXIT
    if [[ "$STAGING_ROOT" == /tmp/aic-l26-de-offline-bundle.* && -d "$STAGING_ROOT" ]]; then
        find "$STAGING_ROOT" -xdev -depth -delete 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup_staging EXIT

for required in \
    "${PIPELINE_SOURCE}/src/aic_video_pipeline_v1" \
    "${PIPELINE_SOURCE}/configs/default.yaml" \
    "${PIPELINE_SOURCE}/offline_wheels" \
    "${PIPELINE_SOURCE}/scripts/kaggle_archive_runner.py" \
    "${PIPELINE_SOURCE}/scripts/kaggle_directory_runner.py" \
    "${PIPELINE_SOURCE}/scripts/kaggle_run_l26_de_offline.sh" \
    "${PIPELINE_SOURCE}/scripts/kaggle_cell_run_l26_de_offline.txt"; do
    [[ -e "$required" ]] || {
        echo "Thiếu file bắt buộc: $required" >&2
        exit 1
    }
done

mkdir -p "${STAGED_PIPELINE}/configs" "${STAGED_PIPELINE}/scripts"
cp -a "${PIPELINE_SOURCE}/src" "$STAGED_PIPELINE/"
cp -a "${PIPELINE_SOURCE}/pyproject.toml" "$STAGED_PIPELINE/"
cp -a "${PIPELINE_SOURCE}/configs/default.yaml" "${STAGED_PIPELINE}/configs/"
cp -a "${PIPELINE_SOURCE}/offline_wheels" "$STAGED_PIPELINE/"
cp -a "${PIPELINE_SOURCE}/scripts/kaggle_archive_runner.py" "${STAGED_PIPELINE}/scripts/"
cp -a "${PIPELINE_SOURCE}/scripts/kaggle_directory_runner.py" "${STAGED_PIPELINE}/scripts/"
cp -a "${PIPELINE_SOURCE}/scripts/kaggle_run_l26_de_offline.sh" "${STAGED_PIPELINE}/scripts/"
cp -a "${PIPELINE_SOURCE}/scripts/kaggle_cell_run_l26_de_offline.txt" "${STAGED_PIPELINE}/scripts/"

find "$STAGED_PIPELINE" -type f -name '*.pyc' -delete
find "$STAGED_PIPELINE" -depth -type d -name '__pycache__' -empty -delete

"$PYTHON_BIN" - "$STAGING_ROOT" "$TEMP_ARCHIVE" <<'PY'
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((root / "aic_video_pipeline_v1").rglob("*")):
        if path.is_file() and path != output:
            archive.write(path, path.relative_to(root).as_posix())
PY

"$PYTHON_BIN" - "$TEMP_ARCHIVE" <<'PY'
import sys
import zipfile

path = sys.argv[1]
required = {
    "aic_video_pipeline_v1/configs/default.yaml",
    "aic_video_pipeline_v1/pyproject.toml",
    "aic_video_pipeline_v1/scripts/kaggle_archive_runner.py",
    "aic_video_pipeline_v1/scripts/kaggle_directory_runner.py",
    "aic_video_pipeline_v1/scripts/kaggle_run_l26_de_offline.sh",
    "aic_video_pipeline_v1/scripts/kaggle_cell_run_l26_de_offline.txt",
    "aic_video_pipeline_v1/src/aic_video_pipeline_v1/orchestrator.py",
    "aic_video_pipeline_v1/src/aic_video_pipeline_v1/services.py",
}
required_wheel_prefixes = (
    "transformers-4.57.6-",
    "tokenizers-0.22.2-",
    "safetensors-0.8.0-",
    "huggingface_hub-0.36.2-",
    "ffmpeg_python-0.2.0-",
    "einops-0.8.2-",
)

with zipfile.ZipFile(path) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit(f"ZIP lỗi tại member: {bad}")
    names = set(archive.namelist())
    missing = sorted(required - names)
    if missing:
        raise SystemExit("ZIP thiếu: " + ", ".join(missing))
    wheels = {
        name.rsplit("/", 1)[-1]
        for name in names
        if name.startswith("aic_video_pipeline_v1/offline_wheels/")
    }
    launcher = archive.read(
        "aic_video_pipeline_v1/scripts/kaggle_run_l26_de_offline.sh"
    ).decode()
    config = archive.read(
        "aic_video_pipeline_v1/configs/default.yaml"
    ).decode()

if not all(any(wheel.startswith(prefix) for wheel in wheels)
           for prefix in required_wheel_prefixes):
    raise SystemExit("ZIP thiếu offline wheel bắt buộc")
for marker in (
    "Videos_L26_d",
    "Videos_L26_e",
    "--video-root \"$L26_D_VIDEO_ROOT\"",
    "--video-root \"$L26_E_VIDEO_ROOT\"",
    "--gpu-workers \"$GPU_WORKERS\"",
    "PIP_NO_INDEX=1",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
):
    if marker not in launcher:
        raise SystemExit(f"Launcher offline thiếu: {marker}")
for forbidden in ("wget ", "apt-get ", "snapshot_download", "https://"):
    if forbidden in launcher:
        raise SystemExit(f"Launcher offline còn network command: {forbidden}")
for marker in (
    "sample_every_frames: 10",
    "threshold: 0.296",
    "batch_size: 4",
    "threshold: 0.5",
):
    if marker not in config:
        raise SystemExit(f"Config không còn logic cũ: {marker}")
if any("__pycache__" in name or name.endswith(".pyc") for name in names):
    raise SystemExit("ZIP không được chứa Python cache")
PY

mv "$TEMP_ARCHIVE" "$OUTPUT_PATH"
echo "Đã tạo bundle offline L26_d/e: $OUTPUT_PATH"
sha256sum "$OUTPUT_PATH"
du -h "$OUTPUT_PATH"
