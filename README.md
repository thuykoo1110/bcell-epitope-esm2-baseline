# B-cell epitope baseline: ESM-2 đóng băng + MLP

Baseline dự đoán **B-cell epitope conformational ở mức residue** từ chuỗi amino acid. Mỗi residue của kháng nguyên được biểu diễn bằng embedding của ESM-2 (đóng băng, không fine-tune), rồi một MLP nhỏ dự đoán xác suất residue đó nằm trong epitope. Dữ liệu là **BCE_633** của [GraphBepi](https://github.com/biomed-AI/GraphBepi) (Zeng et al., *Bioinformatics* 2023), chia train/test theo ngày công bố giống bài báo.

Repo được viết lại từ đầu để hiểu rõ từng bước (đọc PDB → gán nhãn → trích embedding → huấn luyện → đánh giá), và so sánh hai cỡ ESM-2 (8M và 35M) cùng hai cách đặt trọng số loss, mỗi cấu hình chạy 3 seed.

## Mục lục

- [Chạy nhanh](#chạy-nhanh)
- [Kết quả](#kết-quả)
- [Pipeline](#pipeline)
- [Hướng dẫn chạy chi tiết](#hướng-dẫn-chạy-chi-tiết)
- [Dữ liệu và quy trình đánh giá](#dữ-liệu-và-quy-trình-đánh-giá)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Hạn chế](#hạn-chế)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Chạy nhanh

Chạy toàn bộ từ thư mục gốc của repo (cần internet và nên có GPU):

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/biomed-AI/GraphBepi external/GraphBepi   # lấy total.csv của BCE_633
python scripts/build_dataset.py     # tải PDB từ RCSB, ghi data/processed/{train,val,test}.jsonl
bash scripts/run_all.sh             # trích embedding 8M + 35M, chạy 12 run, tạo results/results.md
```

Chỉ muốn thử một run nhanh (sau khi đã có `data/processed/` và `embeddings/`):

```bash
python -m epitope.train --model esm2_35m --pos-weight none --seed 0
```

Chạy trên Google Colab: xem [mục Colab](#chạy-trên-google-colab). Chạy trên Windows: xem [ghi chú Windows](#ghi-chú-cho-windows).

## Kết quả

Đánh giá trên tập test độc lập (các chain công bố từ 2021-04-01). Số của repo này là **trung bình ± độ lệch chuẩn của 3 seed** (độ lệch chuẩn tổng thể, `ddof=0`, đúng như `epitope/summarize.py` tính).

| Mô hình | pos_weight | ROC-AUC | PR-AUC | F1 | MCC |
|---|---|---|---|---|---|
| ESM-2 8M + MLP | none | 0.671 ± 0.002 | 0.164 ± 0.002 | 0.227 ± 0.001 | 0.132 ± 0.003 |
| ESM-2 8M + MLP | auto | 0.677 ± 0.004 | 0.169 ± 0.003 | 0.233 ± 0.002 | 0.143 ± 0.005 |
| ESM-2 35M + MLP | none | 0.706 ± 0.002 | **0.192 ± 0.002** | 0.248 ± 0.002 | **0.162 ± 0.006** |
| ESM-2 35M + MLP | auto | **0.707 ± 0.002** | 0.187 ± 0.001 | 0.248 ± 0.003 | 0.160 ± 0.007 |

Đoán ngẫu nhiên có PR-AUC ≈ 0.090 (bằng tỉ lệ epitope của tập test, 8,96%), ROC-AUC = 0.5, MCC = 0. Bảng đầy đủ (kèm Precision và Recall) nằm ở [`results/results.md`](results/results.md), kết quả từng run ở `results/*.json`.

**So với các phương pháp trong bài báo GraphBepi** (số trích từ Table 3 của bài, *không chạy lại*, cùng tập test độc lập):

| Phương pháp | ROC-AUC | PR-AUC | F1 | MCC |
|---|---|---|---|---|
| Bepipred-2.0 | 0.648 | 0.132 | 0.220 | 0.126 |
| DiscoTope-2.0 | 0.655 | 0.154 | 0.231 | 0.136 |
| ScanNet_T | 0.712 | 0.182 | 0.257 | 0.170 |
| GraphBepi (ESM-2 3B + cấu trúc AlphaFold2 + đồ thị) | 0.751 | 0.261 | 0.310 | 0.232 |

### Nhận xét

- **Encoder lớn hơn cho kết quả tốt hơn.** ESM-2 35M cao hơn 8M ở cả 6 cặp (cùng seed, cùng `pos_weight`), PR-AUC tăng từ +0.016 đến +0.030. Mức tăng này lớn hơn nhiều so với độ lệch giữa các seed (≈ 0.002–0.003).
- **`pos_weight` không có tác động nhất quán.** Với 35M, `none` nhỉnh hơn một chút; với 8M thì `auto` nhỉnh hơn một chút. Chênh lệch giữa hai cấu hình nhỏ hơn rất nhiều so với chênh lệch giữa hai cỡ model. Ngưỡng chọn trên val bù lại phần lớn việc đổi trọng số (ngưỡng ≈ 0.13–0.16 với `none`, ≈ 0.5–0.6 với `auto`), nên F1 và MCC gần như không đổi.
- **MLP overfit rất nhanh.** Epoch tốt nhất theo val PR-AUC chỉ là epoch 2–4, sau đó val giảm dù train loss vẫn giảm; early stopping dừng đúng lúc.
- **Vị trí so với bài báo.** Baseline 35M vượt Bepipred-2.0 và DiscoTope-2.0 ở cả bốn chỉ số, và có PR-AUC nhỉnh hơn ScanNet_T (0.192 so với 0.182), nhưng ROC-AUC, F1 và MCC thấp hơn nhẹ. Còn thấp hơn GraphBepi khá nhiều, điều dễ hiểu vì GraphBepi dùng ESM-2 3B (`esm2_t36_3B_UR50D`), cấu trúc dự đoán bởi AlphaFold2 và mạng đồ thị, còn baseline này chỉ dùng chuỗi, encoder 8M/35M đóng băng và một MLP theo từng residue.

## Pipeline

```
total.csv (BCE_633)            RCSB PDB
  chain + epitope  ───────────►  file .pdb ──► chuỗi (nguyên tử CA) + ngày công bố + nhãn 0/1 từng residue
                                                        │
                       scripts/build_dataset.py         ▼
                                       data/processed/{train,val,test}.jsonl
                                                        │
                       epitope/embed.py                 ▼
                          ESM-2 đóng băng  ──►  embeddings/<model>_<split>.pt   ({chain_id: [L, D] float16})
                                                        │
                       epitope/train.py                 ▼
                          chuẩn hóa (thống kê train) → MLP theo từng residue → early stopping (val PR-AUC)
                          → chọn ngưỡng F1 trên val → chấm test
                                                        │
                                                        ▼
                                  results/<model>_pw-<pw>_seed<k>.json
                                                        │
                       epitope/summarize.py             ▼
                                         results/results.md   (trung bình ± độ lệch chuẩn theo seed)
```

## Hướng dẫn chạy chi tiết

> **Mọi lệnh đều chạy từ thư mục gốc của repo** (đường dẫn `data/`, `embeddings/`, `results/` là đường dẫn tương đối).

### 0. Yêu cầu

- Python 3.9 trở lên (nên dùng môi trường ảo).
- Internet: tải `total.csv` từ GitHub, tải ~600 file PDB từ RCSB, và tải trọng số ESM-2 từ Hugging Face (lần đầu).
- GPU được khuyến nghị cho bước trích embedding. Không có GPU thì code tự chuyển sang CPU, chỉ chậm hơn. Bước train MLP nhẹ, chạy được trên CPU (`--device cpu`).
- Thư viện (xem [`requirements.txt`](requirements.txt)): `torch>=2.0`, `transformers>=4.40`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `pytest`.

### 1. Cài đặt

```bash
git clone https://github.com/thuykoo1110/bcell-epitope-esm2-baseline.git
cd bcell-epitope-esm2-baseline

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Lấy dữ liệu BCE_633

Repo chỉ cần một file là `data/BCE_633/total.csv` trong repo GraphBepi, nên clone nông vào `external/` (thư mục này đã nằm trong `.gitignore`):

```bash
git clone --depth 1 https://github.com/biomed-AI/GraphBepi external/GraphBepi
python scripts/build_dataset.py
```

Script tìm `total.csv` ở `external/GraphBepi/data/BCE_633/total.csv` (hoặc `../GraphBepi/data/BCE_633/total.csv`), tải PDB của từng chain từ RCSB (có cache ở `data/raw/pdb/`, chạy lại sẽ không tải lại), gán nhãn epitope, lọc và chia tập. Kết quả:

```
data/raw/pdb/*.pdb              PDB đã tải (cache)
data/processed/train.jsonl      mỗi dòng: {"id", "date", "sequence", "labels"}   labels là chuỗi '0'/'1' cùng độ dài sequence
data/processed/val.jsonl
data/processed/test.jsonl
```

Cuối lần chạy script in số chain và thống kê từng tập. Số chain giữ lại nên gần 633. Tập test phải ra **15 543 residue, 1 393 epitope (8,96%)** và tập val **14 851 residue, 13,3% epitope**; nếu lệch nhiều thì bước đọc PDB hoặc gán nhãn đang có vấn đề.

### 3. Trích embedding ESM-2

```bash
python -m epitope.embed
```

Chọn model cần trích ở danh sách `MODELS_TO_RUN` đầu file [`epitope/embed.py`](epitope/embed.py) (mặc định cả hai; xóa bớt một tên nếu chỉ cần một model). File nào đã tồn tại thì được bỏ qua.

| Tên trong repo | Checkpoint Hugging Face | Số chiều embedding |
|---|---|---|
| `esm2_8m` | `facebook/esm2_t6_8M_UR50D` | 320 |
| `esm2_35m` | `facebook/esm2_t12_35M_UR50D` | 480 |

Đầu ra: `embeddings/<model>_<split>.pt`, mỗi file là một dict `{chain_id: tensor float16 [L, D]}` (bỏ token đặc biệt đầu và cuối, nên đúng L residue).

### 4. Huấn luyện và đánh giá

**Một run:**

```bash
python -m epitope.train --model esm2_35m --pos-weight none --seed 0
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--model` | `esm2_8m` | tên embedding (`esm2_8m` hoặc `esm2_35m`), trùng tên file trong `embeddings/` |
| `--pos-weight` | `none` | trọng số lớp dương trong BCE loss: `none`, `auto` (= số residue âm / số residue dương của train), hoặc một số |
| `--seed` | `0` | seed cho khởi tạo và xáo trộn dữ liệu |
| `--epochs` | `40` | số epoch tối đa |
| `--patience` | `6` | dừng sớm nếu val PR-AUC không tăng sau ngần ấy epoch |
| `--lr` | `1e-3` | learning rate (AdamW) |
| `--weight-decay` | `1e-2` | weight decay |
| `--batch-size` | `2048` | số residue mỗi batch |
| `--hidden` | `256` | số neuron lớp ẩn thứ nhất (lớp thứ hai cố định 64) |
| `--dropout` | `0.3` | dropout |
| `--data-dir` / `--emb-dir` / `--out-dir` | `data/processed` / `embeddings` / `results` | thư mục dữ liệu, embedding, kết quả |
| `--device` | `cuda` nếu có, không thì `cpu` | thiết bị chạy |

Ví dụ: `python -m epitope.train --model esm2_8m --pos-weight auto --seed 1 --device cpu`.

Mỗi epoch in train loss và val PR-AUC; cuối cùng in một dòng `TEST ...` gồm PR-AUC, ROC-AUC, F1, MCC và ngưỡng đã chọn, rồi ghi `results/<model>_pw-<pos_weight>_seed<seed>.json`. File JSON gồm `config` (toàn bộ tham số), `best_epoch`, `val` và `test` (PR-AUC, ROC-AUC, precision, recall, F1, MCC, ngưỡng, ma trận nhầm lẫn), cùng `baselines` (tỉ lệ epitope, PR-AUC của scorer ngẫu nhiên, mô hình luôn đoán "không epitope").

**Toàn bộ thí nghiệm (2 model × 2 `pos_weight` × 3 seed = 12 run):**

```bash
bash scripts/run_all.sh
```

Script này chạy trích embedding, 12 run train, rồi tổng hợp. Nếu đã có embedding thì bước đầu chỉ in "exists".

### 5. Tổng hợp kết quả

```bash
python -m epitope.summarize
```

Đọc mọi `results/*.json`, gộp theo (model, `pos_weight`), tính trung bình ± độ lệch chuẩn qua các seed và ghi `results/results.md`.

### 6. Chạy kiểm thử

```bash
python -m pytest
```

Kiểm thử gồm xử lý PDB và gán nhãn (`tests/test_data.py`) và các metric (`tests/test_metrics.py`). Không cần GPU và không cần tải dữ liệu.

### 7. Bản rút gọn để đọc hiểu (tùy chọn)

```bash
python -m epitope.train_simple
```

[`epitope/train_simple.py`](epitope/train_simple.py) là một file chạy tuần tự từ trên xuống, cấu hình bằng các hằng số đầu file (`TAG`, `POS_WEIGHT`, `EPOCHS`...), chỉ in kết quả, không early stopping và không lưu JSON. Dùng để đọc hiểu quy trình, còn số liệu trong README này đến từ `epitope/train.py`.

### Chạy trên Google Colab

Chọn Runtime → Change runtime type → GPU, rồi chạy lần lượt:

```python
!git clone https://github.com/thuykoo1110/bcell-epitope-esm2-baseline.git
%cd bcell-epitope-esm2-baseline
!pip install -q -r requirements.txt
!git clone --depth 1 https://github.com/biomed-AI/GraphBepi external/GraphBepi
!python scripts/build_dataset.py
!bash scripts/run_all.sh
!cat results/results.md
```

Colab đã có sẵn `torch`, nên bước `pip install` chủ yếu cài thêm phần còn thiếu. Dữ liệu và embedding nằm trong ổ đĩa tạm của Colab và mất khi hết phiên; muốn giữ thì copy `data/processed/`, `embeddings/` và `results/` sang Google Drive.

### Ghi chú cho Windows

`scripts/run_all.sh` là script bash, nên cần chạy trong Git Bash hoặc WSL. Nếu dùng PowerShell thuần, chạy tương đương:

```powershell
python -m epitope.embed
foreach ($m in "esm2_8m","esm2_35m") {
  foreach ($pw in "none","auto") {
    foreach ($s in 0,1,2) {
      python -m epitope.train --model $m --pos-weight $pw --seed $s
    }
  }
}
python -m epitope.summarize
```

### Cấu trúc thư mục sau khi chạy

```
data/raw/pdb/            PDB đã tải            (không commit)
data/processed/          train/val/test.jsonl  (không commit)
embeddings/              <model>_<split>.pt    (không commit)
external/GraphBepi/      repo GraphBepi        (không commit)
results/                 *.json từng run + results.md   (có commit)
```

`data/`, `embeddings/`, `external/` và `checkpoints/` nằm trong `.gitignore`.

### Tái lập kết quả

Seed được đặt cho `random`, `numpy` và `torch`, và việc xáo trộn dữ liệu mỗi epoch dùng generator riêng theo seed. Tuy vậy, tính toán trên GPU có thể không hoàn toàn tất định giữa các phiên bản thư viện hoặc loại GPU, nên số có thể lệch nhẹ ở chữ số thập phân thứ ba so với bảng trên. Chênh lệch giữa 8M và 35M lớn hơn mức lệch này nhiều.

## Dữ liệu và quy trình đánh giá

- **Nguồn:** `total.csv` của BCE_633 (chain ID và các residue epitope dạng `resi_resn`). Cấu trúc PDB được tải từ RCSB, lấy nguyên tử CA để dựng chuỗi và vị trí residue.
- **Xử lý PDB** theo GraphBepi: dừng đọc ở dòng `TER` của chain, giữ conformation altloc xuất hiện đầu tiên, quy đổi `MSE`/`CCS`/`CS?` về MET/CYS, bỏ các residue không chuẩn; nhãn epitope được gán theo số residue (có dự phòng theo mã chèn như `3A`). Giữ chain dài ≤ 1023 residue và có ít nhất 1 residue epitope.
- **Chia tập:**
  - **test**: chain công bố từ **2021-04-01**, gồm **15 543 residue, trong đó 1 393 là epitope (8,96%)**, khớp với số liệu tập test độc lập trong bài báo (1 393 + 14 150 residue);
  - **val**: 10% chain mới nhất trong phần còn lại, **14 851 residue, 13,3% epitope**;
  - **train**: phần còn lại.
- **Huấn luyện:** MLP `in → 256 → 64 → 1` (ReLU, dropout 0.3), AdamW (lr 1e-3, weight decay 1e-2), batch 2048, tối đa 40 epoch, dừng sớm theo val PR-AUC (patience 6). Đặc trưng được chuẩn hóa bằng trung bình/độ lệch chuẩn của riêng tập train.
- **Chọn ngưỡng:** ngưỡng tối ưu F1 chỉ chọn trên **val**, rồi áp nguyên cho test. Checkpoint dùng để đánh giá là epoch tốt nhất theo val.
- **Seed:** 0, 1, 2 cho mỗi cấu hình (2 cỡ model × 2 cách `pos_weight` × 3 seed = 12 run).
- **Các chỉ số:** PR-AUC và ROC-AUC không phụ thuộc ngưỡng; F1, MCC, Precision, Recall tính tại ngưỡng đã chọn trên val. Với dữ liệu mất cân bằng (epitope chỉ ≈ 9% residue test), PR-AUC và MCC phản ánh chất lượng tốt hơn accuracy: mô hình luôn đoán "không epitope" đã đạt accuracy 91,0% nhưng F1 = 0 và MCC = 0.

## Cấu trúc repo

```
epitope/
  data.py          đọc PDB, gán nhãn epitope, lọc và chia train/val/test
  embed.py         trích embedding ESM-2 đóng băng (chọn model ở MODELS_TO_RUN)
  model.py         MLP theo từng residue
  train.py         huấn luyện và đánh giá một run (nhận tham số dòng lệnh, lưu JSON)
  train_simple.py  bản rút gọn để đọc hiểu (dùng hằng số, chỉ in kết quả)
  metrics.py       PR-AUC, ROC-AUC, F1, MCC, ngưỡng tối ưu F1
  summarize.py     gộp results/*.json thành results/results.md (trung bình ± độ lệch chuẩn)
scripts/
  build_dataset.py tải PDB, dựng train/val/test.jsonl
  run_all.sh       trích embedding + 12 run + tổng hợp
tests/             kiểm thử cho xử lý dữ liệu và metric
results/           kết quả từng run và bảng tổng hợp
requirements.txt   thư viện cần cài
```

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp và cách xử lý |
|---|---|
| `Không thấy total.csv` khi chạy `build_dataset.py` | Chưa clone GraphBepi, hoặc clone sai chỗ. Chạy lại `git clone --depth 1 https://github.com/biomed-AI/GraphBepi external/GraphBepi` từ thư mục gốc repo. |
| `build_dataset.py` in `không tải được <PDB>` cho vài chain | Lỗi mạng hoặc RCSB không có bản `.pdb` của cấu trúc đó. Chạy lại script (PDB đã tải được cache nên chỉ tải phần còn thiếu). Chain nào vẫn thiếu sẽ bị bỏ qua và được đếm ở dòng tổng kết cuối. |
| `FileNotFoundError: embeddings/..._train.pt` khi train | Chưa trích embedding, hoặc `--model` không khớp tên file. Chạy `python -m epitope.embed` trước. |
| `FileNotFoundError: data/processed/train.jsonl` | Chưa chạy `scripts/build_dataset.py`, hoặc đang đứng sai thư mục (phải ở thư mục gốc repo). |
| Bước embedding rất chậm | Đang chạy trên CPU. Dùng GPU (ví dụ Colab T4), hoặc bỏ bớt một model trong `MODELS_TO_RUN`. |
| Muốn ép chạy CPU (hoặc GPU bị hết bộ nhớ) | Với embedding: đặt biến môi trường `CUDA_VISIBLE_DEVICES=""` rồi chạy `python -m epitope.embed`. Với train: thêm `--device cpu`. |
| `bash: scripts/run_all.sh: No such file` hoặc không có `bash` trên Windows | Dùng Git Bash/WSL, hoặc chạy vòng lặp PowerShell ở [ghi chú Windows](#ghi-chú-cho-windows). |
| `python -m epitope.summarize` báo `no result files found` | Chưa có file nào trong `results/*.json`; chạy ít nhất một lệnh `epitope.train` trước. |

## Hạn chế

- Tập test chỉ có vài chục chain. Độ lệch chuẩn ở trên chỉ phản ánh sự khác nhau giữa các seed huấn luyện, **không** phản ánh việc chọn những chain test cụ thể; chưa có khoảng tin cậy theo chain (bootstrap).
- Ngưỡng chọn trên val (13,3% epitope) được áp lên test (8,96% epitope), nên F1, MCC, Precision và Recall chịu ảnh hưởng của sự lệch tỉ lệ này; PR-AUC và ROC-AUC thì không phụ thuộc ngưỡng.
- Số của các phương pháp khác được trích từ bài báo, không chạy lại trong cùng môi trường, nên chi tiết xử lý dữ liệu có thể khác.
- Chỉ dùng chuỗi: không có thông tin cấu trúc hay ngữ cảnh giữa các residue; encoder đóng băng; siêu tham số giữ mặc định, chưa tinh chỉnh.

## Tài liệu tham khảo

- Zeng, Y., Wei, Z., Yuan, Q., Chen, S., Yu, W., Lu, Y., Gao, J., Yang, Y. *Identifying B-cell epitopes using AlphaFold2 predicted structures and pretrained language model.* Bioinformatics 39(4), btad187 (2023). [doi:10.1093/bioinformatics/btad187](https://doi.org/10.1093/bioinformatics/btad187). Mã nguồn và dữ liệu BCE_633: [biomed-AI/GraphBepi](https://github.com/biomed-AI/GraphBepi).
- Lin, Z. et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science 379, 1123–1130 (2023). [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574). Trọng số ESM-2 dùng qua [Hugging Face](https://huggingface.co/facebook/esm2_t12_35M_UR50D).
