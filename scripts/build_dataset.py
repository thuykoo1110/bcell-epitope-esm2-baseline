#!/usr/bin/env python
"""Build train/val/test JSONL files from GraphBepi's `total.csv`.

Usage (from the repo root):
    git clone https://github.com/biomed-AI/GraphBepi.git third_party/GraphBepi
    python scripts/build_dataset.py --csv third_party/GraphBepi/data/BCE_633/total.csv

It downloads each PDB entry from RCSB (cached in data/raw/pdb), extracts the chain
sequence, maps epitope labels, filters, and splits by deposition date.
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epitope import data  # noqa: E402


def fetch_pdb(pdb_id: str, cache_dir: Path, retries: int = 5) -> str:
    path = cache_dir / f"{pdb_id}.pdb"
    if path.exists():
        return path.read_text()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            path.write_text(text)
            return text
        except Exception as exc:  # network hiccup: wait and retry
            print(f"  retry {attempt + 1}/{retries} for {pdb_id}: {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {pdb_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to GraphBepi data/BCE_633/total.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--pdb-cache", default="data/raw/pdb")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    cache = Path(args.pdb_cache)
    cache.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, header=0, index_col=0)
    col = "Epitopes (resi_resn)"
    assert col in df.columns, f"column '{col}' not found, got {list(df.columns)}"

    chains, skipped = [], 0
    for i, (chain_name, epitopes) in enumerate(df[col].items(), 1):
        pdb_id, chain_id = chain_name[:4], chain_name[-1]   # same as GraphBepi extract_chain(i[:4], i[-1])
        try:
            text = fetch_pdb(pdb_id.upper(), cache)
        except RuntimeError as exc:
            print(f"[skip] {chain_name}: {exc}")
            skipped += 1
            continue
        date, seq, sites = data.parse_pdb_chain(text, chain_id)
        if not seq:
            print(f"[skip] {chain_name}: no CA atoms for chain '{chain_id}'")
            skipped += 1
            continue
        labels = data.map_epitope_labels(seq, sites, epitopes)
        chains.append(data.Chain(chain_name, date, seq, sites, labels))
        if i % 50 == 0:
            print(f"processed {i}/{len(df)}")

    kept = [c for c in chains if data.keep_chain(c)]
    train_all, test = data.split_by_date(kept)
    train, val = data.split_train_val(train_all, args.val_frac)

    out = Path(args.out_dir)
    data.write_jsonl(train, out / "train.jsonl")
    data.write_jsonl(val, out / "val.jsonl")
    data.write_jsonl(test, out / "test.jsonl")

    print(f"\nrows in csv: {len(df)} | parsed: {len(chains)} | skipped: {skipped} | kept after filter: {len(kept)}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s}", data.dataset_stats(split))
    print("\nSanity check vs. GraphBepi: total.csv has 633 data rows, the paper reports 633 antigens"
          " (577 train / 56 test) with epitope rate ~11.8% (train) and ~9.0% (test)."
          " 'kept after filter' should be close to 633; small differences are expected, large ones"
          " mean the PDB/label parsing needs checking.")


if __name__ == "__main__":
    main()
