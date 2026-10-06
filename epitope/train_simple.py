import copy
from pathlib import Path

import torch
import torch.nn as nn

from . import data, metrics
from .model import ResidueMLP

TAG = "esm2_8m"        
POS_WEIGHT = "none"      # "none" hoặc "auto"
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 2048
SEED = 0
DATA_DIR = Path("data/processed")
EMB_DIR = Path("embeddings")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


def load_split(split):
    """Gộp mọi chain của 1 split thành 1 bảng: X [N, D] (mỗi hàng là 1 residue), y [N] (0/1)."""
    emb = torch.load(EMB_DIR / f"{TAG}_{split}.pt")
    records = data.read_jsonl(DATA_DIR / f"{split}.jsonl")
    X = torch.cat([emb[r["id"]].float() for r in records])
    y = torch.tensor([label for r in records for label in r["labels"]], dtype=torch.float32)
    assert len(X) == len(y)
    return X, y


def predict(model, X):
    """Trả về xác suất epitope (0..1) cho từng residue."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()


# 1) Đọc dữ liệu
Xtr, ytr = load_split("train")
Xva, yva = load_split("val")
Xte, yte = load_split("test")

# 2) Chuẩn hóa: trung bình/độ lệch chuẩn chỉ tính từ train, rồi áp cho cả val và test
mean, std = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
Xtr, Xva, Xte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

# 3) Hàm loss (nếu POS_WEIGHT = "auto" thì residue epitope bị phạt nặng hơn khi đoán sai)
n_pos = ytr.sum()
n_neg = len(ytr) - n_pos
pos_weight = (n_neg / n_pos).to(DEVICE) if POS_WEIGHT == "auto" else None
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"train: {int(n_pos)} epitope / {len(ytr)} residue ({n_pos / len(ytr):.2%}), pos_weight = {POS_WEIGHT}")

# 4) Model và optimizer
model = ResidueMLP(Xtr.shape[1]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

# 5) Vòng train
best_score, best_state = -1.0, None
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(Xtr))                    # xáo trộn thứ tự residue mỗi epoch
    for i in range(0, len(order), BATCH_SIZE):
        idx = order[i:i + BATCH_SIZE]
        loss = loss_fn(model(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    score = metrics.pr_auc(yva.numpy(), predict(model, Xva))    # chấm trên val
    print(f"epoch {epoch:02d} | last batch loss {loss.item():.4f} | val PR-AUC {score:.4f}")
    if score > best_score:                              # nhớ lại model tốt nhất theo val
        best_score = score
        best_state = copy.deepcopy(model.state_dict())

# 6) Đánh giá: dùng model tốt nhất, ngưỡng chọn trên val, chấm trên test
model.load_state_dict(best_state)
threshold = metrics.best_f1_threshold(yva.numpy(), predict(model, Xva))
report = metrics.full_report(yte.numpy(), predict(model, Xte), threshold)

print(f"\nTEST  PR-AUC {report['pr_auc']:.4f} (đoán bừa ≈ {report['prevalence']:.4f}) "
      f"| ROC-AUC {report['roc_auc']:.4f} | F1 {report['f1']:.4f} "
      f"| MCC {report['mcc']:.4f} | threshold {threshold:.3f}")
