"""Portuguese evaluation of a checkpoint, free-running (text in, speech out), scored by Whisper large-v3-turbo.

    python scripts/eval_ptbr.py --checkpoint runs/ptbr/inference.pt --corpus corpus.jsonl --output eval/ptbr

Three groups: sentences of the held-out test split of the dataset (the model never saw these texts), sentences written
for this evaluation that are in no split, and the same Whisper model on the recordings the model was trained to imitate.
Whisper is an external scoring tool, never part of the TTS model. Numbers support a listening test; they do not replace one.
"""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from tiny_voca.infer import load_model, speak
from dizione_ptbr import normalize as expand

NOVEL = (
    "A professora explicou a lição com muita paciência para todos os alunos.",
    "Será que ainda dá tempo de pegar o último ônibus para o centro?",
    "Meu irmão mais novo ganhou uma bicicleta vermelha no aniversário.",
    "Não esqueça de levar o guarda-chuva, porque pode chover à tarde.",
    "O gato dormiu a manhã inteira em cima do sofá da sala.",
    "Quando chegamos à praia, o sol já estava se pondo atrás das montanhas.",
    "Preciso comprar pão, leite e algumas frutas antes do jantar.",
    "Ela abriu a janela e ficou olhando a chuva cair devagar sobre a rua.",
    "Vamos marcar uma reunião na quarta-feira, às quinze horas e trinta minutos.",
    "O livro que você me emprestou é muito mais interessante do que eu imaginava!",
    "Foram apenas três dias de viagem, mas parecia que tinha passado um mês inteiro.",
    "Você acha que o time vai conseguir vencer o campeonato este ano?",
    "A cidade ficou em silêncio depois que o último trem partiu da estação.",
    "Eles compraram uma casa pequena perto do rio, com um quintal cheio de árvores.",
    "Eu pago duzentos e cinquenta reais pelo conserto, se ficar pronto amanhã.",
    "A conversa foi longa, mas no fim todo mundo concordou com a proposta.",
)


def norm(text):
    t = unicodedata.normalize("NFC", expand(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t.replace("-", " "))
    return re.sub(r"\s+", " ", t).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True, help="frases/corpus.jsonl of the dataset (text_id, text, split, category)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tests", type=int, default=48, help="held-out test sentences (spread over categories)")
    p.add_argument("--asr-model", default="large-v3-turbo")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    import jiwer
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    rows = [json.loads(l) for l in a.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    known = {hashlib.sha256(r["text"].strip().encode()).hexdigest() for r in rows}
    for t in NOVEL:
        assert hashlib.sha256(t.strip().encode()).hexdigest() not in known, f"in the dataset: {t}"
    test = [r for r in rows if r["split"] == "test"]
    by_cat = {}
    for r in sorted(test, key=lambda r: r["text_id"]):
        by_cat.setdefault(r["category"], []).append(r)
    chosen, i = [], 0
    while len(chosen) < a.tests and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(chosen) < a.tests:
                chosen.append(by_cat[cat].pop(len(by_cat[cat]) // 2 if i % 2 else 0))
        i += 1
    jobs = [("test", r["text"], r["category"]) for r in chosen] + [("novel", t, "novo") for t in NOVEL]

    model = load_model(a.checkpoint, "cpu")
    voices = model.config.speakers
    asr = WhisperModel(a.asr_model, device="cpu", compute_type="int8", cpu_threads=8)
    audio_dir = a.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def transcribe(wave, rate):
        f = int(np.gcd(rate, 16000))
        x = resample_poly(wave, 16000 // f, rate // f).astype(np.float32)
        segs, _ = asr.transcribe(x, language="pt", beam_size=5, condition_on_previous_text=False, temperature=0.0)
        return " ".join(s.text.strip() for s in segs).strip()

    samples = []
    for n, (group, text, cat) in enumerate(jobs):
        for voice in voices:
            wave, phones, _ = speak(model, text, voice, seed=a.seed)
            wave = wave.numpy()
            peak = float(np.abs(wave).max())
            out = wave / peak if peak > 1 else wave
            name = f"{group}{n:03d}_{voice}.wav"
            sf.write(audio_dir / name, out, model.config.sample_rate, subtype="PCM_16")
            hyp = transcribe(out.astype(np.float32), model.config.sample_rate)
            wer = jiwer.wer(norm(text), norm(hyp) or "<vazio>")
            samples.append({"group": group, "category": cat, "voice": voice, "text": text, "transcript": hyp, "wer": round(float(wer), 4),
                            "seconds": round(len(wave) / model.config.sample_rate, 2), "file": name})
            print(json.dumps({"n": len(samples), "wer": samples[-1]["wer"]}), flush=True)

    def stat(sel):
        w = [s["wer"] for s in sel]
        if not w:
            return None
        ref = " ".join(norm(s["text"]) for s in sel)
        hyp = " ".join(norm(s["transcript"]) for s in sel)
        return {"clips": len(w), "mean_wer": round(float(np.mean(w)), 4), "corpus_wer": round(float(jiwer.wer(ref, hyp)), 4),
                "clips_with_wer_0": round(float(np.mean([x == 0 for x in w])), 3), "clips_wer_over_20pct": int(sum(x > 0.2 for x in w))}
    summary = {}
    for g in ("test", "novel"):
        summary[g] = {"all": stat([s for s in samples if s["group"] == g])}
        for v in voices:
            summary[g][v] = stat([s for s in samples if s["group"] == g and s["voice"] == v])
    # What Whisper scored on the dataset's own recordings of the same test texts (from the dataset's quality check).
    recorded = {}
    for v in voices:
        meta = Path(a.corpus).parent.parent / "audios" / "ptbr" / f"metadata_{v}.jsonl"
        if meta.exists():
            ids = {r["text_id"] for r in chosen}
            w = [json.loads(l)["normalized_wer"] for l in meta.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["text_id"] in ids]
            recorded[v] = {"clips": len(w), "mean_wer": round(float(np.mean(w)), 4)} if w else None
    report = {"checkpoint": str(a.checkpoint), "asr_model": "faster-whisper " + a.asr_model, "note": "free-running synthesis, language=pt",
              "summary": summary, "recordings_same_texts_dataset_qc": recorded, "samples": samples}
    (a.output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "recordings": recorded}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
