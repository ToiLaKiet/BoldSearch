#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SOURCE="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$(dirname "$PIPELINE_SOURCE")"
STAGING_ROOT="$(mktemp -d -t aic-kaggle-bundles.XXXXXX)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null || {
        echo "Không tìm thấy PYTHON_BIN: $PYTHON_BIN"
        exit 1
    }
elif command -v python >/dev/null; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null; then
    PYTHON_BIN="python3"
elif [[ -x /home/long/miniconda3/bin/python ]]; then
    PYTHON_BIN="/home/long/miniconda3/bin/python"
else
    echo "Không tìm thấy Python để kiểm tra ZIP."
    exit 1
fi

cleanup_staging() {
    local status=$?
    trap - EXIT
    if [[ "$STAGING_ROOT" == /tmp/aic-kaggle-bundles.* &&
          -d "$STAGING_ROOT" ]]; then
        find "$STAGING_ROOT" -xdev -depth -delete 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup_staging EXIT

levels=(25 26 27 28 29 30)
if (( $# > 0 )); then
    if [[ $# -ne 2 || "$1" != "--level" || ! "$2" =~ ^(25|26|27|28|29|30)$ ]]; then
        echo "Cách dùng: $0 [--level 25|26|27|28|29|30]"
        exit 2
    fi
    levels=("$2")
fi

for level in "${levels[@]}"; do
    bundle_root="${STAGING_ROOT}/l${level}"
    staged_pipeline="${bundle_root}/aic_video_pipeline_v1"
    staged_scripts="${staged_pipeline}/scripts"
    staged_configs="${staged_pipeline}/configs"
    archive_tmp="${STAGING_ROOT}/aic_video_pipeline_l${level}.zip"
    archive_final="${OUTPUT_DIR}/aic_video_pipeline_l${level}.zip"

    mkdir -p "$staged_scripts" "$staged_configs"
    cp -a "${PIPELINE_SOURCE}/src" "$staged_pipeline/"
    cp -a "${PIPELINE_SOURCE}/pyproject.toml" "$staged_pipeline/"
    cp -a "${PIPELINE_SOURCE}/configs/default.yaml" "$staged_configs/"
    cp -a "${PIPELINE_SOURCE}/scripts/kaggle_setup_and_run.sh" "$staged_scripts/"
    cp -a "${PIPELINE_SOURCE}/scripts/kaggle_archive_runner.py" "$staged_scripts/"
    cp -a "${PIPELINE_SOURCE}/scripts/kaggle_directory_runner.py" "$staged_scripts/"
    cp -a "${PIPELINE_SOURCE}/scripts/kaggle_run_l${level}.sh" "$staged_scripts/"
    cp -a "${PIPELINE_SOURCE}/scripts/kaggle_cell_run_l${level}.txt" "$staged_scripts/"

    find "$staged_pipeline" -type f -name '*.pyc' -delete
    find "$staged_pipeline" -depth -type d -name '__pycache__' -empty -delete

    (
        cd "$bundle_root"
        zip -q -r "$archive_tmp" aic_video_pipeline_v1
    )

    "$PYTHON_BIN" - "$archive_tmp" "$level" <<'PY'
import sys
import zipfile
from pathlib import PurePosixPath

archive_path, level = sys.argv[1], sys.argv[2]
prefix = PurePosixPath("aic_video_pipeline_v1")
required = {
    str(prefix / "configs/default.yaml"),
    str(prefix / "src/aic_video_pipeline_v1/services.py"),
    str(prefix / "scripts/kaggle_setup_and_run.sh"),
    str(prefix / "scripts/kaggle_archive_runner.py"),
    str(prefix / "scripts/kaggle_directory_runner.py"),
    str(prefix / f"scripts/kaggle_run_l{level}.sh"),
    str(prefix / f"scripts/kaggle_cell_run_l{level}.txt"),
}

with zipfile.ZipFile(archive_path) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit(f"ZIP lỗi tại member: {bad}")
    names = set(archive.namelist())
    missing = sorted(required - names)
    if missing:
        raise SystemExit("ZIP thiếu file: " + ", ".join(missing))
    setup = archive.read(str(prefix / "scripts/kaggle_setup_and_run.sh")).decode()
    services = archive.read(
        str(prefix / "src/aic_video_pipeline_v1/services.py")
    ).decode()
    launcher = archive.read(
        str(prefix / f"scripts/kaggle_run_l{level}.sh")
    ).decode()
    cell = archive.read(
        str(prefix / f"scripts/kaggle_cell_run_l{level}.txt")
    ).decode()

if "def _decode_autoshot_frames" not in services:
    raise SystemExit("ZIP chứa services.py cũ")
if '("ffmpeg-python", "import ffmpeg")' in setup:
    raise SystemExit("ZIP còn dependency ffmpeg-python cũ")
if f'BUNDLE_NAME="aic_video_pipeline_l{level}.zip"' not in cell:
    raise SystemExit("Cell Kaggle không khớp level")
if level in {"25", "26", "28", "29", "30"}:
    if f'export AIC_VIDEO_LEVEL="L{level}"' not in launcher:
        raise SystemExit("Launcher direct-input sai level")
    if "AIC_ARCHIVES=" in launcher:
        raise SystemExit("Launcher direct-input vẫn tải archive")
if level == "26" and "AIC_VIDEO_ROOTS=" not in launcher:
    raise SystemExit("Launcher L26 chưa hỗ trợ nhiều video root")
PY
    mv "$archive_tmp" "$archive_final"
    echo "Đã tạo: $archive_final"
done
