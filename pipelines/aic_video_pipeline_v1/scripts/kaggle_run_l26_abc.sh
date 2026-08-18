#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/kaggle_setup_and_run.sh"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo "Không tìm thấy setup script: $SETUP_SCRIPT"
    exit 1
fi

# Kaggle không đủ working disk để tích lũy output của cả ba archive trong một
# session. Bắt buộc chọn đúng một part; cùng script được dùng lại cho a, b, c.
read -r -a requested_parts <<< "${AIC_L26_PARTS:-}"

if (( ${#requested_parts[@]} == 0 )); then
    echo "Hãy chọn đúng một part: AIC_L26_PARTS=a, b, hoặc c."
    exit 1
fi

if (( ${#requested_parts[@]} != 1 )); then
    echo "Mỗi Kaggle session chỉ được chạy một part để tránh đầy ổ đĩa."
    echo "Giá trị hiện tại: ${requested_parts[*]}"
    exit 1
fi

archives=()
seen=" "
for part in "${requested_parts[@]}"; do
    if [[ ! "$part" =~ ^[abc]$ ]]; then
        echo "Part L26 không hợp lệ: $part (chỉ chấp nhận a, b, c)"
        exit 1
    fi
    if [[ "$seen" == *" $part "* ]]; then
        continue
    fi
    seen+="$part "
    archives+=("Videos_L26_${part}.zip")
done

export AIC_ARCHIVES="${archives[*]}"

echo "Archive L26 sẽ chạy trong session này: $AIC_ARCHIVES"
echo "Runner sẽ tự tìm kết quả cũ trong /kaggle/input và dùng tối đa 2 GPU."

exec bash "$SETUP_SCRIPT"
