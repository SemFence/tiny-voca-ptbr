#!/usr/bin/env bash
# Collects the files the demo serves into public/assets (not stored in git): the ONNX model and its metadata
# (python -m tiny_voca.export ...), and the compiled TypeScript phonemizer with its data (pnpm build in
# @semfence/dizione-ptbr). Nothing is downloaded or sent anywhere.
#
#   bash build_assets.sh ../release/onnx ../../dizione-ptbr-js
set -euo pipefail
cd "$(dirname "$0")"
ONNX="${1:?folder with model.onnx and model.json}"
DIZIONE_JS="${2:?path to the dizione-ptbr-js checkout (built)}"
mkdir -p public/assets/dizione/data
cp "$ONNX/model.onnx" "$ONNX/model.json" public/assets/
cp "$DIZIONE_JS"/dist/*.js public/assets/dizione/
cp "$DIZIONE_JS"/data/lexicon.tsv "$DIZIONE_JS"/data/open_table.json "$DIZIONE_JS"/data/LICENSE.mfa-dictionary.md public/assets/dizione/data/
ls -l public/assets public/assets/dizione
