#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "$PROJECT_DIR"
"$PYTHON_BIN" src/kiln_mvp.py --config config.json

