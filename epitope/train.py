"""Train and evaluate the per-residue baseline.

    python -m epitope.train --model esm2_8m --pos-weight none --seed 0
    python -m epitope.train --model esm2_8m --pos-weight auto --seed 0

Protocol
  * train on `train`, early-stop on validation PR-AUC (val = most recent 10% of the training chains),
  * choose the F1-optimal threshold on `val` only, then apply it unchanged to `test`,
  * report PR-AUC / ROC-AUC (threshold-free) and F1 / MCC / precision / recall (at that threshold),
    next to the prevalence ("random scorer") and always-negative baselines.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import data, metrics
from .model import ResidueMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(emb_path: Path, jsonl_path: Path):
    """Flatten all chains of a split into residue-level tensors X [N, D], y [N]."""
    emb = torch.load(emb_path)
    records = data.read_jsonl(jsonl_path)
    X = torch.cat([emb[r["id"]].float() for r in records], dim=0)
    y = torch.tensor([lab for r in records for lab in r["labels"]], dtype=torch.float32)
    assert X.shape[0] == y.shape[0]
    return X, y


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: str, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        out.append(torch.sigmoid(model(X[i:i + batch].to(device))).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_8m", help="embedding name used by epitope.embed")
    ap.add_argument("--pos-weight", default="none", help="'none', 'auto' (= #neg/#pos) or a number")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    dd, ed = Path(args.data_dir), Path(args.emb_dir)
    Xtr, ytr = load_split(ed / f"{args.model}_train.pt", dd / "train.jsonl")
    Xva, yva = load_split(ed / f"{args.model}_val.pt", dd / "val.jsonl")
    Xte, yte = load_split(ed / f"{args.model}_test.pt", dd / "test.jsonl")

    # standardize with training statistics only
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xva, Xte = [(X - mu) / sd for X in (Xtr, Xva, Xte)]

    n_pos, n_neg = float(ytr.sum()), float(len(ytr) - ytr.sum())
    if args.pos_weight == "none":
        pw = None
    elif args.pos_weight == "auto":
        pw = torch.tensor(n_neg / n_pos)
    else:
        pw = torch.tensor(float(args.pos_weight))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=None if pw is None else pw.to(args.device))

    model = ResidueMLP(Xtr.shape[1], args.hidden, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, best_epoch, bad = -1.0, None, 0, 0
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0], generator=g)
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(args.device), ytr[idx].to(args.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_scores = predict(model, Xva, args.device)
        val_prauc = metrics.pr_auc(yva.numpy(), val_scores)
        print(f"epoch {epoch:02d} | train loss {total / len(perm):.4f} | val PR-AUC {val_prauc:.4f}")
        if val_prauc > best_val:
            best_val, best_epoch, bad = val_prauc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    val_scores = predict(model, Xva, args.device)
    test_scores = predict(model, Xte, args.device)
    thr = metrics.best_f1_threshold(yva.numpy(), val_scores)

    y_te = yte.numpy()
    result = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "val": metrics.full_report(yva.numpy(), val_scores, thr),
        "test": metrics.full_report(y_te, test_scores, thr),
        "baselines": {
            "prevalence_test": metrics.prevalence(y_te),
            "random_scorer_pr_auc_test": metrics.random_scorer_pr_auc(y_te, args.seed),
            "all_negative_test": metrics.all_negative_baseline(y_te),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_pw-{args.pos_weight}_seed{args.seed}"
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    t = result["test"]
    print(f"\nTEST  PR-AUC {t['pr_auc']:.4f} (random ≈ {t['prevalence']:.4f}) | ROC-AUC {t['roc_auc']:.4f} "
          f"| F1 {t['f1']:.4f} | MCC {t['mcc']:.4f} | thr {thr:.3f}")
    print(f"saved {out_dir / (name + '.json')}")


if __name__ == "__main__":
    main()
