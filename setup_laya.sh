#!/usr/bin/env bash
# Set up local multimodal inference and training on Linux or Apple silicon.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'HELP'
Usage: ./setup_laya.sh [--skip-models] [--skip-verify]
  --skip-models  use a checkpoint that is already available locally
  --skip-verify  skip the runtime import and FFmpeg checks
HELP
}

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) echo "Use start.bat on Windows or run this script in WSL." >&2; exit 2 ;;
esac

SKIP_MODELS=0
SKIP_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown setup option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON:-python3}"
VENV="$DIR/.venv-mm"
export PIP_CACHE_DIR="$DIR/.pip-cache" HF_HOME="$DIR/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1 USE_TF=0 USE_TORCH=1
"$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.version_info < (3, 11):
  sys.exit("Multimodal setup requires Python 3.11 or newer.")
if sys.platform == "darwin" and platform.machine() != "arm64":
  sys.exit("Multimodal setup on macOS requires an arm64 Python on Apple silicon.")
PY
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$VENV/bin/python" -m ensurepip --upgrade
fi
TORCH_INDEX="${TORCH_INDEX_URL:-}"
if [ -z "$TORCH_INDEX" ] && [ "$(uname -s)" = "Linux" ]; then
  if [ "${LAYA_DEVICE:-}" = "cpu" ] || ! nvidia-smi -L >/dev/null 2>&1; then
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
  fi
fi
if [ -n "$TORCH_INDEX" ]; then
  "$VENV/bin/python" -m pip install --index-url "$TORCH_INDEX" 'torch>=2.8' 'torchvision>=0.23'
fi
"$VENV/bin/python" -m pip install -e "$DIR[multimodal,serve]"
if [ "$SKIP_VERIFY" -eq 0 ]; then
  "$VENV/bin/python" -c 'from laya_multimodal import MultimodalAgent; import soundfile, torchvision'
  for tool in ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      printf 'Video input requires %s on PATH. Install FFmpeg and rerun setup.\n' "$tool" >&2
      exit 1
    fi
  done
fi
if [ "$SKIP_MODELS" -eq 0 ]; then
  "$VENV/bin/python" - "$DIR/models/jev-multimodal-qwen3.5-4b" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
  "ZeeoRe/jev-multimodal-qwen3.5-4b",
  revision="5ea706ae049932f34aa78cb26e119939eb145cc3",
  local_dir=sys.argv[1],
  allow_patterns=["multimodal_config.json", "backbone_config/*", "processor/*", "*.safetensors"],
)
PY
fi
printf 'Activate with: source "%s/bin/activate"\n' "$VENV"
