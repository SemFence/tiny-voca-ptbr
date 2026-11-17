"""Time and compare synthesis on CPU and on the Mac GPU (MPS): python scripts/bench_devices.py --checkpoint runs/ptbr/inference.pt"""
import argparse, time
import torch
from tiny_voca.infer import load_model, speak

p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); a = p.parse_args()
TEXT = "O outono chega pintando as árvores de tons dourados, alaranjados e avermelhados."
LONG = "Depois de uma longa e difícil viagem pelas montanhas, os viajantes finalmente chegaram à pequena vila, onde foram recebidos com comida quente."
waves = {}
for dev in ("cpu", "mps"):
    if dev == "mps" and not torch.backends.mps.is_available():
        continue
    m = load_model(a.checkpoint, dev)
    for text, tag in ((TEXT, "curta"), (LONG, "longa")):
        for _ in range(2):
            speak(m, text, "F_s4", seed=0)
        if dev == "mps": torch.mps.synchronize()
        t = time.perf_counter(); n = 5
        for _ in range(n):
            w, _, _ = speak(m, text, "F_s4", seed=0)
        if dev == "mps": torch.mps.synchronize()
        dt = (time.perf_counter() - t) / n
        secs = w.shape[-1] / m.config.sample_rate
        waves[(dev, tag)] = w.cpu().flatten()
        print(f"{dev:4s} {tag:6s} audio {secs:5.2f}s  tempo {dt*1000:6.1f} ms  {secs/dt:6.1f}x tempo real", flush=True)
if ("mps", "curta") in waves:
    for tag in ("curta", "longa"):
        c, g = waves[("cpu", tag)], waves[("mps", tag)]
        n = min(len(c), len(g))
        print(f"{tag}: comprimento cpu={len(c)} mps={len(g)}; erro máximo {float((c[:n]-g[:n]).abs().max()):.4f}; erro relativo {float((c[:n]-g[:n]).norm()/c[:n].norm()):.4f}")
