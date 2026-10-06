"""Extract frozen per-residue ESM-2 embeddings and cache them to disk.

    python -m epitope.embed_simple

Output: embeddings/<model>_<split>.pt = {chain_id: float16 tensor [L, D]}
"""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from . import data

MODELS = {
    "esm2_8m": "facebook/esm2_t6_8M_UR50D",     # hidden size 320
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",  # hidden size 480
}

MODEL = "esm2_8m"                # "esm2_8m" hoặc "esm2_35m"
DATA_DIR = "data/processed"
EMB_DIR = "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed_sequence(model, tokenizer, sequence: str, device: str) -> torch.Tensor:
    enc = tokenizer(sequence, return_tensors="pt").to(device)
    hidden = model(**enc).last_hidden_state[0]          # [L + 2, D]
    assert hidden.shape[0] == len(sequence) + 2, (hidden.shape, len(sequence))
    return hidden[1:-1].float().cpu()


def embed_split(records, model, tokenizer, device: str) -> dict:
    out = {}
    for r in tqdm(records, desc="embedding"):
        out[r["id"]] = embed_sequence(model, tokenizer, r["sequence"], device).half()
    return out


def main():
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(MODELS[MODEL])
    model = EsmModel.from_pretrained(MODELS[MODEL], add_pooling_layer=False).to(DEVICE).eval()

    emb_dir = Path(EMB_DIR)
    emb_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = emb_dir / f"{MODEL}_{split}.pt"
        if path.exists():
            print(f"{path} exists")
            continue
        records = data.read_jsonl(Path(DATA_DIR) / f"{split}.jsonl")
        torch.save(embed_split(records, model, tokenizer, DEVICE), path)
        print(f"saved {path} ({len(records)} chains)")


if __name__ == "__main__":
    main()
