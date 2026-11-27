#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python examples/native_multimodal.py "${LAYA_MULTIMODAL_MODEL:-models/jev-multimodal-qwen3.5-4b}" "$@"
