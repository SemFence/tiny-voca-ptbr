# tiny-voca-ptbr

Um modelo de texto para fala em **português do Brasil (pt-BR)** com **4,5 milhões de parâmetros** e duas vozes sintéticas.
Roda em CPU, na GPU do Mac (MPS) e no navegador (WebGPU/WASM). Treinado do zero, com dados sintéticos abertos.

[Pesos](https://huggingface.co/semfence/tiny-voca-ptbr) · [Dataset de fala](https://huggingface.co/datasets/semfence/ptbr-synthetic-speech) · [Dataset de frases](https://huggingface.co/datasets/semfence/ptbr-sentences) · [Fonemizador](https://github.com/SemFence/dizione-ptbr)

Criado pela **SemFence** · pesquisador: **Gustavo Toledo**. É a receita do [tiny-voca](https://github.com/n4ze3m/tiny-voca)
(Muhammed Nazeem, Apache-2.0) aplicada ao português do Brasil.

> **Só pt-BR.** Não serve para português de Portugal nem para outras línguas.

## Características

- **Pequeno**: 4.533.259 parâmetros, gerador de onda incluído.
- **Rápido**: de ~113× (CPU) a ~165× (GPU do Mac M4) o tempo real; ~65× no navegador com WebGPU.
- **Duas vozes**: `F_s4` (feminina) e `M_sm4` (masculina), sintéticas, 24 kHz.
- **Roda em qualquer lugar**: PyTorch, ONNX Runtime e ONNX Runtime Web.
- **Aberto**: código, pesos, receita de treino e dados.

A qualidade do som é a do tiny-voca original: o sintetizador por filtro e pulsos (25 mil parâmetros) soa mais sintético do que
um vocoder neural. Números de avaliação e limitações estão em [`release/README.md`](release/README.md).

## Instalação

```bash
git clone https://github.com/SemFence/tiny-voca-ptbr.git
cd tiny-voca-ptbr
pip install -e ".[own]"          # inclui o fonemizador dizione-ptbr
hf download semfence/tiny-voca-ptbr --local-dir weights
```

## Uso

```bash
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca-ptbr.pt \
  --voice F_s4 \
  --text "Olá! Este é o tiny voca em português." \
  --output ola.wav
```

| Opção | Padrão | O que faz |
|---|---|---|
| `--text-file` | | Lê o texto de um arquivo em vez de `--text` |
| `--voice` | `F_s4` | `F_s4` ou `M_sm4` |
| `--speed` | 1,0 | Velocidade da fala |
| `--pitch-shift` | 0 | Sobe ou desce o tom |
| `--pitch-range` | 1,0 | Mais ou menos variação de tom |
| `--device` | `cpu` | `cpu`, `mps` (Mac) ou `cuda` |

Texto longo é lido frase a frase e o áudio é juntado. Um texto sem pontuação final recebe um ponto, para a última palavra não
ser cortada. Uma frase sem nada para ler (por exemplo, só emoji) é pulada. O áudio nunca passa de 0,99 de pico.
Em Python, `tiny_voca.infer.speak_text` faz o mesmo.

## ONNX

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca-ptbr.pt --output tiny-voca-ptbr.onnx
python scripts/check_onnx.py --help      # compara o ONNX com o PyTorch
```

## Demonstração no navegador

A pasta [`web/`](web/README.md) tem a demo: o fonemizador em TypeScript
([`@semfence/dizione-ptbr`](https://github.com/SemFence/dizione-ptbr-js)) e o ONNX no WebGPU (ou WASM). Nada do que a pessoa
digita sai do aparelho.

```bash
cd web
bash build_assets.sh ../release/onnx /caminho/para/dizione-ptbr-js
python3 -m http.server 8765 --directory public       # abra http://localhost:8765
```

## Treino

O fluxo completo (ambiente, dados, características, anotação e treino) está em [`scripts/cloud_run.sh`](scripts/cloud_run.sh),
pensado para uma máquina alugada com uma GPU CUDA. Passo a passo:

```bash
python -m tiny_voca.prepare --inputs data/train-*.parquet --split train --output feats --workers 8
python -m tiny_voca.annotate --features feats --output ann --dataset-revision <sha-do-dataset> --device cuda
python -m tiny_voca.train --train ann/train.jsonl --validation ann/validation.jsonl --output runs/ptbr \
  --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0
python scripts/eval_ptbr.py --checkpoint runs/ptbr/inference.pt --corpus frases/corpus.jsonl --output eval/ptbr
```

O modelo publicado foi treinado por 104.000 passos em uma RTX 5080 alugada (cerca de 8 horas, ~US$ 3 de GPU). Mais sobre o
modelo e os dados em [`docs/model.md`](docs/model.md).

## Testes

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests
node web/test_tokens.mjs <tokens_golden.json>     # os tokens da demo contra os do Python
```

## Limitações

- Só pt-BR, duas vozes fixas, sem clonagem de voz.
- Nomes e palavras estrangeiras são lidos com as regras do português.
- Sotaques regionais (chiado carioca, "r" caipira) não são modelados.
- Sem fala em fluxo (streaming) por enquanto.
- Não houve teste de audição com pessoas; a avaliação é automática (WER com Whisper).

## O que mudou em relação ao tiny-voca original

`text.py` (fonemizador `dizione-ptbr`, 55 símbolos), `config.py` (vozes e vocabulário), `align.py` (rótulos do alinhador para
os fonemas do pt-BR), `prepare.py` (`--workers`), `infer.py` e `evaluate.py` (frases e avaliação em português), `model.py`
(a fase em float64 roda na CPU quando o dispositivo é MPS), `scripts/` e `web/`.

## Licença

Apache-2.0 para o código e os pesos. CC BY 4.0 para os datasets (crédito obrigatório a **SemFence (Gustavo Toledo)**) e para o
léxico do fonemizador (dicionário MFA).

## Citação

Veja [`CITATION.cff`](CITATION.cff). Se usar este trabalho, cite também o tiny-voca original.
