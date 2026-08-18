#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/kaggle_setup_and_run.sh"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo "Không tìm thấy setup script: $SETUP_SCRIPT"
    exit 1
fi

# Mặc định chạy lần lượt ba archive. Có thể giới hạn một archive cho mỗi
# Kaggle session bằng AIC_L26_PARTS="a", "b", hoặc "c".
read -r -a requested_parts <<< "${AIC_L26_PARTS:-a b c}"

if (( ${#requested_parts[@]} == 0 )); then
    echo "AIC_L26_PARTS không chứa part nào."
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

echo "Các archive L26 sẽ chạy: $AIC_ARCHIVES"
echo "Runner sẽ tự tìm kết quả cũ trong /kaggle/input và dùng tối đa 2 GPU."

exec bash "$SETUP_SCRIPT"
