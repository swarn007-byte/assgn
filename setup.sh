#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for tool in python curl pdftotext sha256sum tar; do
    command -v "$tool" >/dev/null
done
mkdir -p .cache
if [[ ! -x .venv/bin/python ]]; then
    python -m venv .venv
fi
PIP_CACHE_DIR="$PWD/.cache/pip" .venv/bin/python -m pip install -r requirements.txt

fetch() {
    local url=$1 file=$2 checksum=$3
    if ! printf '%s  %s\n' "$checksum" "$file" | sha256sum --check --status; then
        curl --fail --silent --show-error --location --retry 2 "$url" -o "$file.part"
        printf '%s  %s\n' "$checksum" "$file.part" | sha256sum --check
        mv -- "$file.part" "$file"
    fi
}
fetch https://github.com/progit/progit2/releases/download/2.1.450/progit.pdf \
    .cache/progit.pdf 403f4051cdca2c585361d85f6e87d5dd87d8c544fc91dced007711babe81e8ef
fetch https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz \
    .cache/glove.gz 5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24
fetch https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-bin-ubuntu-x64.tar.gz \
    .cache/llama.tar.gz 0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e
fetch https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b12359 \
    .cache/llama-source.tar.gz e2eadc0fd06f81abea3b8012584e67bea3b403d91bcd1b3072b656cbcb2fb4bb
pdftotext -layout .cache/progit.pdf .cache/progit.txt
tar -xzf .cache/llama.tar.gz -C .cache
tar -xzf .cache/llama-source.tar.gz -C .cache
if [[ ! -f .cache/base/model.safetensors ]]; then
    mkdir -p .cache/base
    cp task1_lora/checkpoint/{config.json,generation_config.json,model.safetensors,special_tokens_map.json,tokenizer.json,tokenizer_config.json} .cache/base/
fi
.venv/bin/python task3_local/convert.py > .cache/convert.log 2>&1
rm -f .cache/progit-q4_0.gguf
.cache/llama-b12359/llama-quantize .cache/progit-f16.gguf .cache/progit-q4_0.gguf Q4_0 > .cache/quantize.log 2>&1
.venv/bin/python -m pip check
.cache/llama-b12359/llama-server --version
