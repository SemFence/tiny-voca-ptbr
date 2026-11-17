# Demo (browser)

Runs the model in the browser with ONNX Runtime Web: **WebGPU** when the browser has it, WASM (CPU) otherwise.
The phonemizer is [`@semfence/dizione-ptbr`](https://github.com/SemFence/dizione-ptbr-js) in TypeScript (no Python runtime).
Nothing the reader types leaves the device.

```bash
# 1. the model:   python -m tiny_voca.export --checkpoint runs/ptbr/inference.pt --output release/onnx/model.onnx
# 2. the phonemizer, built:   (cd ../../dizione-ptbr-js && pnpm build)
bash build_assets.sh ../release/onnx ../../dizione-ptbr-js      # copies them into public/assets (not in git)
python3 -m http.server 8765 --directory public                  # then open http://localhost:8765
```

`?provider=wasm` forces the CPU path. `node test_tokens.mjs tokens_golden.json` checks that the demo's tokens equal the
Python front end's on the whole sentence bank (`tokens.js` is the port of `tiny_voca.text.tokenize`).

Measured here (Chromium, Apple M4): 15 s of speech in about 0.23 s with WebGPU (65× real time), 1.8 s with WASM (8.5×).
The audio has the same length as PyTorch's and the same energy envelope; only the noise component differs by seed.
