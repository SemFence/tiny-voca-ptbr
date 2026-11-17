#!/usr/bin/env bash
# Collects the files the demo serves into public/assets (not stored in git): the ONNX model and its metadata
# (python -m tiny_voca.export ...), and the compiled TypeScript phonemizer with its data (pnpm build in
# @semfence/dizione-ptbr). Nothing is downloaded or sent anywhere.
#
#   bash web/build_assets.sh release/onnx ../dizione-ptbr-js      (from the repository root)
set -euo pipefail
ONNX="$(cd "${1:?folder with model.onnx and model.json}" && pwd)"          # paths are relative to where you run it
DIZIONE_JS="$(cd "${2:?path to the dizione-ptbr-js checkout (built)}" && pwd)"
cd "$(dirname "$0")"
mkdir -p public/assets/dizione/data
cp "$ONNX/model.onnx" "$ONNX/model.json" public/assets/
cp "$DIZIONE_JS"/dist/*.js public/assets/dizione/
cp "$DIZIONE_JS"/data/lexicon.tsv "$DIZIONE_JS"/data/open_table.json "$DIZIONE_JS"/data/LICENSE.mfa-dictionary.md public/assets/dizione/data/
ls -l public/assets public/assets/dizione
