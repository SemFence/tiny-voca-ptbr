"""Writes the tokens the Python front end gives for the sentence bank, for web/test_tokens.mjs to match:
   python scripts/make_tokens_golden.py ../ptbr-voz-dataset/frases/corpus.jsonl web/tokens_golden.json"""
import json, sys
from tiny_voca.text import phonemize, NothingToPronounce

EXTRA = ["Olá! Esta voz foi gerada pelo modelo tiny pt-br da SemFence.", "Oi!! Tudo bem??", "...", "😀", "Uau — que legal… (sério)",
         "Paguei R$ 15,50 às 10h30.", "Dr. Silva chegou às 21 horas.", "\"Aspas\" e 'simples'", "a, b; c: d"]
texts = [json.loads(l)["text"] for l in open(sys.argv[1], encoding="utf-8")] + EXTRA
out = []
for t in texts:
    try:
        p = phonemize(t); out.append([t, list(p.ids), list(p.stress)])
    except (NothingToPronounce, ValueError):
        out.append([t, None, None])
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "sentences,", sum(o[1] is None for o in out), "with nothing to pronounce")
