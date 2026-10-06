"""Extract frozen per-residue ESM-2 embeddings and cache them to disk.

    python -m epitope.embed

Chọn model cần trích embedding ở MODELS_TO_RUN bên dưới (không dùng tham số dòng lệnh).

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

MODELS_TO_RUN = ["esm2_8m", "esm2_35m"]    # bỏ bớt 1 tên nếu chỉ muốn chạy 1 model
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


def embed_model(name: str) -> None:
    """Trích embedding cho cả 3 split bằng model `name` (bỏ qua split đã có file)."""
    from transformers import AutoTokenizer, EsmModel

    emb_dir = Path(EMB_DIR)
    emb_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in ["train", "val", "test"] if not (emb_dir / f"{name}_{s}.pt").exists()]
    for split in ["train", "val", "test"]:
        if split not in todo:
            print(f"{emb_dir / f'{name}_{split}.pt'} exists")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
    model = EsmModel.from_pretrained(MODELS[name], add_pooling_layer=False).to(DEVICE).eval()
    for split in todo:
        path = emb_dir / f"{name}_{split}.pt"
        records = data.read_jsonl(Path(DATA_DIR) / f"{split}.jsonl")
        torch.save(embed_split(records, model, tokenizer, DEVICE), path)
        print(f"saved {path} ({len(records)} chains)")


def main():
    for name in MODELS_TO_RUN:
        print(f"=== {name} ({MODELS[name]}) ===")
        embed_model(name)


if __name__ == "__main__":
    main()
