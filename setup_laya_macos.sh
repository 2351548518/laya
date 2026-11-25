#!/usr/bin/env bash
# Local setup for the classic Laya text checkpoints on macOS.
#
# Creates an isolated virtualenv, uses the supported Intel stack on x86_64 or the current
# package dependencies on Apple silicon, then downloads and verifies the text checkpoints.
#
# Usage:   ./setup_laya_macos.sh [--skip-models] [--skip-verify]   (or ./setup_laya.sh)
# Idempotent: re-running only fixes what is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # repository root
VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON:-python3}"

# Keep every byte this setup writes inside the project directory.
export PIP_CACHE_DIR="$ROOT/.pip-cache"
export HF_HOME="$ROOT/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export USE_TF=0          # Laya is torch-only; a stray TensorFlow install can deadlock loading
export USE_TORCH=1

SKIP_MODELS=0
SKIP_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Host"
uname -srm
ENV_PYTHON="$PYTHON_BIN"
if [ -x "$VENV/bin/python" ]; then
  ENV_PYTHON="$VENV/bin/python"
fi
ARCH="$("$ENV_PYTHON" - <<'PY'
import platform
import sys

arch = platform.machine()
if sys.version_info < (3, 10):
    sys.exit("Laya requires Python 3.10 or newer.")
if arch == "x86_64" and sys.version_info >= (3, 13):
    sys.exit("Use Python 3.10–3.12 for the Intel macOS PyTorch 2.2.2 wheel.")
print(arch)
PY
)"
"$ENV_PYTHON" -V

if [ "$(uname -s)" != "Darwin" ]; then
  echo "warning: this script targets macOS; continuing anyway" >&2
fi

say "Virtualenv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip

say "Dependencies"
if [ "$ARCH" = "x86_64" ]; then
  # The Intel wheel uses NumPy 1.x and Transformers 4.x. Resolve the pins together
  # with the editable package so a later install cannot replace the supported stack.
  "$VENV/bin/python" -m pip install -e "$ROOT" \
    "numpy<2" "torch==2.2.2" "transformers==4.57.6" "huggingface_hub>=0.34.0,<1.0"
else
  "$VENV/bin/python" -m pip install -e "$ROOT"
fi

if [ "$SKIP_MODELS" -eq 0 ]; then
  say "Checkpoints -> $ROOT/models"
  if [ -f "$ROOT/models/laya/model.safetensors" ]; then
    echo "already present, skipping download"
  else
    # One repo bundles all three checkpoints; ~2.3 GB of safetensors.
    ( cd "$ROOT" && "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "convaiinnovations/laya",
    local_dir="models/laya",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer.json"],
)
print("downloaded ->", p)
PY
    )
  fi
  # tests/test_local_e2e.py expects <root>/{laya,laya-multilingual,laya-typed-decisions};
  # the bundle ships the two extra checkpoints as subfolders of the first.
  ln -sfn laya/multilingual   "$ROOT/models/laya-multilingual"
  ln -sfn laya/typed-decisions "$ROOT/models/laya-typed-decisions"
  ls -l "$ROOT/models"
fi

# A truncated or silently-updated download turns into quietly wrong answers, so check the weights
# against the hashes recorded in verify/checkpoints.json. Runs whenever the weights are there,
# including on a --skip-models re-run.
if [ -f "$ROOT/models/laya/model.safetensors" ]; then
  say "Checking checkpoint integrity"
  "$VENV/bin/python" "$ROOT/verify/checkpoints.py" --models "$ROOT/models"
elif [ "$SKIP_MODELS" -eq 0 ]; then
  echo "warning: models/laya/model.safetensors is still missing" >&2
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  say "Verifying with real weights"
  "$VENV/bin/python" "$ROOT/verify/laya_smoke_test.py" --models "$ROOT/models"
fi

say "Done"
echo "Activate with:  source \"$VENV/bin/activate\""
echo "Run the suite:  python tests/test_local_e2e.py \"$ROOT/models\""
