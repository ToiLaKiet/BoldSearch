#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AIC_ARCHIVES="Videos_L27_a.zip"
exec bash "${SCRIPT_DIR}/kaggle_setup_and_run.sh"
