import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # để import được package epitope
from epitope import data

CSV_CANDIDATES = [
    "external/GraphBepi/data/BCE_633/total.csv",
    "../GraphBepi/data/BCE_633/total.csv",            
]
OUT_DIR = Path("data/processed")
PDB_DIR = Path("data/raw/pdb")
VAL_FRAC = 0.1                                          


def fetch_pdb(pdb_id):
    """trả về nội dung file PDB """
    path = PDB_DIR / f"{pdb_id}.pdb"
    if path.exists():
        return path.read_text()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            path.write_text(text)
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:                           # RCSB không có bản .pdb của cấu trúc này
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:                               # lỗi mạng 
            time.sleep(2 * (attempt + 1))
    return None


# Tìm và đọc total.csv
csv_path = next((Path(p) for p in CSV_CANDIDATES if Path(p).exists()), None)
if csv_path is None:
    sys.exit("Không thấy total.csv")
PDB_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv_path, index_col=0)
epitope_col = df["Epitopes (resi_resn)"]
print(f"đọc {csv_path}: {len(df)} chain")

# mỗi chain tải PDB -> lấy sequence/ngày -> gán nhãn epitope
chains, skipped = [], 0
for i, (chain_name, epitopes) in enumerate(epitope_col.items(), 1):
    pdb_id, chain_id = chain_name[:4].upper(), chain_name[-1]      # "4hf5_A" -> "4HF5", "A"
    text = fetch_pdb(pdb_id)
    if text is None:
        print(f"{chain_name}: không tải được {pdb_id}")
        skipped += 1
        continue
    date, seq, sites = data.parse_pdb_chain(text, chain_id)
    if not seq:
        print(f"{chain_name}: không thấy residue nào của chain '{chain_id}'")
        skipped += 1
        continue
    labels = data.map_epitope_labels(seq, sites, epitopes)
    chains.append(data.Chain(chain_name, date, seq, sites, labels))
    if i % 50 == 0:
        print(f"đã xử lý {i}/{len(df)}")

# độ dài < 1024, có ít nhất 1 epitope, rồi chia: test = công bố từ 2021-04-01, val = 10% mới nhất của train
kept = [c for c in chains if data.keep_chain(c)]
train_all, test = data.split_by_date(kept)
train, val = data.split_train_val(train_all, VAL_FRAC)

# ghi file và báo cáo
for name, split in [("train", train), ("val", val), ("test", test)]:
    data.write_jsonl(split, OUT_DIR / f"{name}.jsonl")

print(f"\nchain trong csv: {len(df)} | đọc được: {len(chains)} | bỏ qua: {skipped} | giữ lại sau khi lọc: {len(kept)}")
for name, split in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5s}", data.dataset_stats(split))
print(f"\nĐã ghi vào {OUT_DIR}/")
print("Mốc tham chiếu (ghi trong bản script cũ, chưa kiểm chứng lại): ~633 chain, tỉ lệ epitope ~11.8% (train), ~9.0% (test).")
print("Số chain giữ lại nên gần 633; lệch nhiều thì phần đọc PDB/gán nhãn cần kiểm tra lại.")
