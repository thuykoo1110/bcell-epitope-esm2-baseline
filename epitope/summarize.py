"""Aggregate results/*.json into results/results.md (mean ± std over seeds).

    python -m epitope.summarize
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", "F1"), ("mcc", "MCC"),
        ("precision", "Precision"), ("recall", "Recall")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    groups = defaultdict(list)
    prevalence = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        r = json.loads(path.read_text())
        key = (r["config"]["model"], r["config"]["pos_weight"])
        groups[key].append(r["test"])
        prevalence.append(r["baselines"]["prevalence_test"])
    if not groups:
        raise SystemExit("no result files found")

    lines = [
        "| Embedding | pos_weight | seeds | " + " | ".join(n for _, n in COLS) + " |",
        "|---|---|---|" + "|".join("---" for _ in COLS) + "|",
    ]
    for (model, pw), runs in sorted(groups.items()):
        cells = []
        for k, _ in COLS:
            vals = np.array([r[k] for r in runs])
            cells.append(f"{vals.mean():.3f} ± {vals.std():.3f}")
        lines.append(f"| {model} | {pw} | {len(runs)} | " + " | ".join(cells) + " |")
    prev = float(np.mean(prevalence))
    lines += [
        "",
        f"Random scorer (prevalence) on the test set: PR-AUC ≈ {prev:.3f}, ROC-AUC ≈ 0.5, MCC = 0 "
        "(an always-'non-epitope' model has MCC = 0 and F1 = 0).",
        "Threshold-based metrics use the F1-optimal threshold selected on the validation set.",
    ]
    out = Path(args.results_dir) / "results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
