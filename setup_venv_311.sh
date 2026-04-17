#!/usr/bin/env bash
# Recreate .venv with Python 3.11 (avoids very slow openai package import on Python 3.14).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v python3.11 &>/dev/null; then
  echo "ERROR: python3.11 not found."
  echo "Install:  brew install python@3.11"
  echo "Then add to PATH if needed, e.g.:  echo 'export PATH=\"/opt/homebrew/opt/python@3.11/bin:\$PATH\"' >> ~/.zshrc"
  exit 1
fi

echo "Using $(command -v python3.11) — $(python3.11 --version)"

if [[ -d .venv ]]; then
  echo "Removing existing .venv ..."
  rm -rf .venv
fi

python3.11 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r backend/requirements.txt

echo ""
echo "OK — Python in venv: $(python --version)"
echo ""
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python -c \"from openai import OpenAI; print('openai import ok')\""
echo "  python backend/scripts/generate_dataset.py --dry-run"
