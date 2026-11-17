#!/usr/bin/env bash
# Pipeline completo numa máquina alugada (1 GPU CUDA): ambiente -> dados -> características -> anotação -> treino.
# Cada etapa é idempotente (pula se já foi feita) e grava um marcador em /workspace/stages.
#
#   STAGES="env data prepare annotate train" bash cloud_run.sh        # tudo (padrão)
#   STAGES="train" STEPS=104000 bash cloud_run.sh                      # só o treino (retoma com --resume)
#
# Variáveis: DATASET_REV (sha do dataset no Hugging Face, obrigatório em annotate), WORKERS (núcleos para a
# extração de características), STEPS (passos de treino), RUN (nome da pasta de saída do treino).
set -euo pipefail
cd /workspace
mkdir -p stages logs
STAGES="${STAGES:-env data prepare annotate train}"
WORKERS="${WORKERS:-$(nproc)}"
STEPS="${STEPS:-104000}"
RUN="${RUN:-ptbr}"
DATASET_REPO="${DATASET_REPO:-semfence/ptbr-synthetic-speech}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
log() { echo "[$(date +%H:%M:%S)] $*"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
done_() { [[ -f "stages/$1.done" ]]; }
mark() { touch "stages/$1.done"; log "etapa $1 concluída"; }

if want env && ! done_ env; then
  log "env: ambiente Python, PyTorch (CUDA 12.8) e dependências"
  command -v git >/dev/null || { apt-get update -q && apt-get install -y -q git; }
  python3 -m venv venv
  # shellcheck disable=SC1091
  . venv/bin/activate
  pip install -q -U pip
  pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchaudio==2.8.0"
  pip install -q -e ./dizione-ptbr -e "./tiny-voca-ptbr[prepare,annotate]" "huggingface_hub[hf_transfer]"
  python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0), "| arch", torch.cuda.get_arch_list())
import tiny_voca.text as t
print("símbolos:", len(t.SYMBOLS), "| frontend:", t.frontend())
PY
  mark env
fi
# shellcheck disable=SC1091
[[ -f venv/bin/activate ]] && . venv/bin/activate

if want data && ! done_ data; then
  log "data: dataset $DATASET_REPO + modelo de alinhamento"
  HF_HUB_ENABLE_HF_TRANSFER=1 python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$DATASET_REPO", repo_type="dataset", allow_patterns=["data/*"], local_dir="/workspace/data")
print("dataset em", p)
p = snapshot_download("facebook/wav2vec2-lv-60-espeak-cv-ft", allow_patterns=["*.json", "*.bin", "*.txt", "*.safetensors"])
print("alinhador em", p)
PY
  mark data
fi

if want prepare && ! done_ prepare; then
  for split in validation test train; do
    log "prepare $split ($WORKERS processos)"
    # --resume is safe on a fresh directory and makes a repeated run continue where it stopped
    python -m tiny_voca.prepare --inputs data/data/$split-*.parquet --output feats --split $split --workers "$WORKERS" \
      --resume 2>&1 | tee -a logs/prepare_$split.log | tail -2
  done
  du -sh feats
  mark prepare
fi

if want annotate && ! done_ annotate; then
  : "${DATASET_REV:?defina DATASET_REV (sha do dataset)}"
  log "annotate (CUDA)"
  python -m tiny_voca.annotate --features feats --output ann --dataset-revision "$DATASET_REV" --device cuda --resume \
    2>&1 | tee -a logs/annotate.log | grep -E "complete|aligning|timing_probe|Error" | cut -c1-600
  mark annotate
fi

if want train; then
  log "train: $STEPS passos -> runs/$RUN"
  RESUME=""
  [[ -f runs/$RUN/last.pt ]] && RESUME="--resume runs/$RUN/last.pt"
  python -m tiny_voca.train --train ann/train.jsonl --validation ann/validation.jsonl --output runs/$RUN --device cuda \
    --steps "$STEPS" --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
    --max-seconds 400000 --save-every 1000 --keep-every 10000 --log-every 50 $RESUME \
    2>&1 | tee -a logs/train.log
  log "treino terminou"
fi
