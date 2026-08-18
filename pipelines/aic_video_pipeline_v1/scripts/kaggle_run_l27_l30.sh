#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
SETUP_SCRIPT="${SCRIPT_DIR}/kaggle_setup_and_run.sh"
MANIFEST="${PIPELINE_ROOT}/configs/video_archives_l27_l30.tsv"

if [[ ! -f "$SETUP_SCRIPT" || ! -f "$MANIFEST" ]]; then
    echo "Thiếu kaggle_setup_and_run.sh hoặc manifest video L27-L30."
    exit 1
fi

# Bắt buộc chạy đúng một level trong mỗi Kaggle session để output TAR không
# tích lũy đến mức đầy working disk. Ví dụ: AIC_LEVEL=27.
LEVEL="${AIC_LEVEL:-}"
if [[ ! "$LEVEL" =~ ^(27|28|29|30)$ ]]; then
    echo "Hãy chọn một level: AIC_LEVEL=27, 28, 29, hoặc 30."
    exit 1
fi

if [[ "$LEVEL" =~ ^(28|29|30)$ ]]; then
    echo "L${LEVEL} dùng MP4 đã gắn vào Kaggle Input, không tải ZIP video."
    exec bash "${SCRIPT_DIR}/kaggle_run_l${LEVEL}.sh"
fi

ARCHIVE="Videos_L${LEVEL}_a.zip"
DOWNLOAD_URL="$({
    awk -F '\t' -v archive="$ARCHIVE" \
        '$1 == archive { print $2; exit }' "$MANIFEST"
} || true)"

if [[ -z "$DOWNLOAD_URL" ]]; then
    echo "Không có video archive $ARCHIVE trong manifest: $MANIFEST"
    exit 1
fi

export AIC_ARCHIVES="$ARCHIVE"

echo "Level chạy: L${LEVEL}"
echo "Chỉ tải video: $ARCHIVE"
echo "URL: $DOWNLOAD_URL"
echo "Runner tự quét kết quả cũ trong /kaggle/input để resume."

exec bash "$SETUP_SCRIPT"
