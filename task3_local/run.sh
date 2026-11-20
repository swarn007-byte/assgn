#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -f .cache/base/model.safetensors || ! -x .cache/llama-b12359/llama-quantize ]]; then
    ./setup.sh
fi
.venv/bin/python task3_local/train_commands.py
.venv/bin/python task3_local/convert.py --commands
.cache/llama-b12359/llama-quantize .cache/commands-f16.gguf .cache/commands-q4_0.gguf Q4_0
.venv/bin/python task3_local/benchmark.py
