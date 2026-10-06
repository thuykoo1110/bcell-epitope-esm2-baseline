#!/usr/bin/env bash
set -euo pipefail

# embed.py đọc danh sách model từ MODELS_TO_RUN trong file 
python -m epitope.embed

for MODEL in esm2_8m esm2_35m; do
  for PW in none auto; do
    for SEED in 0 1 2; do
      python -m epitope.train --model "$MODEL" --pos-weight "$PW" --seed "$SEED"
    done
  done
done

python -m epitope.summarize
