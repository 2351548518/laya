#!/usr/bin/env bash
# Run classic examples and report which ones work.
#
#   ./examples/run_all.sh                   # all classic checkpoint examples
#   ./examples/run_all.sh predict_batch     # one example by name
#   ./examples/run_all.sh '*routing*'       # examples matching a filename pattern
#
# Each example is a separate process, so each one loads its own checkpoints; the full sweep takes
# several minutes on CPU and a bit less on MPS. Output is captured and shown only for failures.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the selected interpreter, a checkout virtualenv, or Python 3 on PATH.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in "$ROOT/.venv-mm/bin/python" "$ROOT/.venv/bin/python"; do
    if [ -x "$candidate" ]; then
      PY="$candidate"
      break
    fi
  done
  PY="${PY:-$(command -v python3 || true)}"
fi

if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "no python found -- set PYTHON=/path/to/python" >&2
  exit 2
fi
PATTERN="${1:-[!_]*}"
if [[ "$PATTERN" != *.py ]]; then
  PATTERN="${PATTERN}.py"
fi
shopt -s nullglob
files=()
for f in "$ROOT"/examples/classic/$PATTERN; do
  if [[ -f "$f" && "${f##*/}" != _* ]]; then
    files+=("$f")
  fi
done
if [ ! -e "${files[0]:-}" ]; then
  echo "no examples match '$PATTERN'" >&2
  exit 2
fi
LOG="$(mktemp -t laya_examples.XXXXXX)" || exit 2
trap 'unlink "$LOG"' EXIT

pass=0
fail=0
failed=()

for f in "${files[@]}"; do
  name="$(basename "$f")"
  start=$(date +%s)
  if "$PY" "$f" >"$LOG" 2>&1; then
    printf '  PASS  %-42s %3ds\n' "$name" "$(( $(date +%s) - start ))"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-42s %3ds\n' "$name" "$(( $(date +%s) - start ))"
    fail=$((fail + 1))
    failed+=("$name")
    sed -n '1,40p' "$LOG" | sed 's/^/        /'
  fi
done

echo
echo "  $pass passed, $fail failed"
for f in "${failed[@]:-}"; do
  [ -n "$f" ] && echo "    FAIL $f"
done
[ "$fail" -eq 0 ]
