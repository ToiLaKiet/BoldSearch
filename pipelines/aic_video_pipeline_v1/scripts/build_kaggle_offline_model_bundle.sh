#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "${PIPELINE_ROOT}/../.." && pwd)"
OUTPUT_ROOT="${1:-${PROJECT_ROOT}/pipelines/aic_l28_offline_models}"
ARCHIVE_PATH="${OUTPUT_ROOT}.tar"
FGCLIP2_SOURCE="${AIC_FGCLIP2_SOURCE:-${PROJECT_ROOT}/models/fgclip2}"
AUTOSHOT_SOURCE="${AIC_AUTOSHOT_SOURCE:-/home/long/Documents/AIC/AutoShot}"

CHECKPOINT="${AUTOSHOT_SOURCE}/ckpt_0_200_0.pth"
CHECKPOINT_SIZE="57243097"
CHECKPOINT_SHA256="3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6"
FGCLIP2_MODEL_SIZE="3586589392"

if [[ -e "$OUTPUT_ROOT" || -e "$ARCHIVE_PATH" ]]; then
    echo "Đích đã tồn tại; không ghi đè để tránh mất bộ model cũ:"
    echo "  $OUTPUT_ROOT"
    echo "  $ARCHIVE_PATH"
    exit 1
fi

for required in \
    "${AUTOSHOT_SOURCE}/supernet_flattransf_3_8_8_8_13_12_0_16_60.py" \
    "${AUTOSHOT_SOURCE}/linear.py" \
    "${AUTOSHOT_SOURCE}/utils.py" \
    "$CHECKPOINT" \
    "${FGCLIP2_SOURCE}/config.json" \
    "${FGCLIP2_SOURCE}/preprocessor_config.json" \
    "${FGCLIP2_SOURCE}/modeling_fgclip2.py" \
    "${FGCLIP2_SOURCE}/configuration_fgclip2.py" \
    "${FGCLIP2_SOURCE}/model.safetensors"; do
    [[ -s "$required" ]] || { echo "Thiếu model nguồn: $required"; exit 1; }
done

if [[ "$(stat -c '%s' "$CHECKPOINT")" != "$CHECKPOINT_SIZE" ||
      "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" != "$CHECKPOINT_SHA256" ]]; then
    echo "Checkpoint AutoShot nguồn không đúng phiên bản đã khóa."
    exit 1
fi
if [[ "$(stat -c '%s' "${FGCLIP2_SOURCE}/model.safetensors")" != "$FGCLIP2_MODEL_SIZE" ]]; then
    echo "model.safetensors nguồn không đúng kích thước đã khóa."
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}/autoshot" "${OUTPUT_ROOT}/fgclip2"
for name in \
    supernet_flattransf_3_8_8_8_13_12_0_16_60.py \
    linear.py \
    utils.py \
    ckpt_0_200_0.pth; do
    cp --reflink=auto --preserve=mode,timestamps \
        "${AUTOSHOT_SOURCE}/${name}" "${OUTPUT_ROOT}/autoshot/${name}"
done
for name in \
    config.json \
    preprocessor_config.json \
    modeling_fgclip2.py \
    configuration_fgclip2.py \
    model.safetensors; do
    cp --reflink=auto --preserve=mode,timestamps \
        "${FGCLIP2_SOURCE}/${name}" "${OUTPUT_ROOT}/fgclip2/${name}"
done

(
    cd "$OUTPUT_ROOT"
    sha256sum \
        autoshot/supernet_flattransf_3_8_8_8_13_12_0_16_60.py \
        autoshot/linear.py \
        autoshot/utils.py \
        autoshot/ckpt_0_200_0.pth \
        fgclip2/config.json \
        fgclip2/preprocessor_config.json \
        fgclip2/modeling_fgclip2.py \
        fgclip2/configuration_fgclip2.py \
        fgclip2/model.safetensors > MANIFEST.sha256
)

# No compression: model.safetensors is already compressed.  The TAR is only a
# convenient single upload file; Kaggle will unpack it locally before running.
tar -C "$(dirname "$OUTPUT_ROOT")" -cf "$ARCHIVE_PATH" "$(basename "$OUTPUT_ROOT")"

echo "Đã tạo thư mục model offline: $OUTPUT_ROOT"
du -sh "$OUTPUT_ROOT"
echo "Đã tạo TAR để upload: $ARCHIVE_PATH"
du -h "$ARCHIVE_PATH"
