#!/usr/bin/env bash
# Full experiment grid: 2 embedding sizes x 2 loss settings x 3 seeds = 12 runs.
# Run from the repo root after `python scripts/build_dataset.py ...`.
set -euo pipefail

for MODEL in esm2_8m esm2_35m; do
  python -m epitope.embed --model "$MODEL"
  for PW in none auto; do
    for SEED in 0 1 2; do
      python -m epitope.train --model "$MODEL" --pos-weight "$PW" --seed "$SEED"
    done
  done
done

python -m epitope.summarize
