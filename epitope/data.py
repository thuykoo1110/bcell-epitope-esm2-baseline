from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

LABEL_AA = {**AA3, "MSE": "M", "CCS": "C"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

MAX_LEN = 1023           # GraphBepi chains len < 1024 
TEST_DATE = 20210401     


@dataclass
class Chain:
    id: str
    date: int
    sequence: str
    sites: list          # residue key (number + insertion code)
    labels: list         # 0/1 per residue


def parse_pdb_date(header_line: str) -> int:
    """header line -> YYYYMMDD. PDB uses DD-MON-YY"""
    token = header_line[50:59].strip()
    day, mon, yy = token.split("-")
    yy = int(yy)
    year = 2000 + yy if yy < 23 else 1900 + yy 
    return year * 10000 + MONTHS[mon.upper()] * 100 + int(day)


def normalize_resname(field: str):
    altloc, amino = "", field
    if len(amino) > 3:
        altloc, amino = amino[0], amino[-3:]
    if amino == "MSE":
        amino = "MET"
    elif amino == "CCS" or amino[:-1] == "CS":     
        amino = "CYS"
    elif amino not in AA3:
        return None
    return altloc, amino


def parse_pdb_chain(pdb_text: str, chain_id: str):
    date = 0
    sequence, sites, first_altloc = [], [], {}
    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec == "HEADER" and date == 0:
            date = parse_pdb_date(line)
            continue
        if rec == "TER" and len(line) > 21 and line[21] == chain_id:
            break                                      # hết chain: bỏ phần sau TER (như GraphBepi)
        if rec not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        if line[21] != chain_id or line[12:16].strip() != "CA":
            continue
        norm = normalize_resname(line[16:21].strip())
        if norm is None:
            continue
        altloc, resn = norm
        key = line[22:28].strip()                      # residue number + insertion code
        if altloc:
            if key not in first_altloc:
                first_altloc[key] = altloc
            if first_altloc[key] != altloc:
                continue
        sequence.append(AA3[resn])
        sites.append(key)
    return date, "".join(sequence), sites


def map_epitope_labels(sequence: str, sites: list, epitope_field: str) -> list:
    labels = [0] * len(sequence)
    site2idx = {str(k): i for i, k in enumerate(sites)}
    for token in str(epitope_field).split(", "):
        if "_" not in token:
            continue
        pos, resn = token.split("_")
        if resn not in LABEL_AA:
            continue
        want = LABEL_AA[resn]
        idx = site2idx.get(pos)
        if idx is not None and sequence[idx] == want:  # khớp đúng số residue + đúng tên
            labels[idx] = 1
            continue
        # Phần sau `pos` phải là chữ cái, nếu không "3" sẽ khớp nhầm
        # "30", "31", "300"... (khớp tiền tố thuần túy gán nhãn sai residue).
        for key, j in site2idx.items():
            if key[:len(pos)] == pos and key[len(pos):].isalpha() and sequence[j] == want:
                labels[j] = 1
                break
    return labels


def keep_chain(chain: Chain) -> bool:
    """GraphBepi length < 1024 and >= one epitope residue."""
    return 0 < len(chain.sequence) <= MAX_LEN and sum(chain.labels) > 0


def split_by_date(chains: list, test_date: int = TEST_DATE):
    train = [c for c in chains if c.date < test_date]
    test = [c for c in chains if c.date >= test_date]
    return train, test


def split_train_val(train: list, val_frac: float = 0.1):
    ordered = sorted(train, key=lambda c: (c.date, c.id))
    n_val = max(1, int(round(len(ordered) * val_frac)))
    return ordered[:-n_val], ordered[-n_val:]


def write_jsonl(chains: list, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in chains:
            f.write(json.dumps({
                "id": c.id, "date": c.date, "sequence": c.sequence,
                "labels": "".join(str(x) for x in c.labels),
            }) + "\n")


def read_jsonl(path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["labels"] = [int(ch) for ch in r["labels"]]
            assert len(r["labels"]) == len(r["sequence"]), r["id"]
            records.append(r)
    return records


def dataset_stats(chains) -> dict:
    n_res = sum(len(c["sequence"]) if isinstance(c, dict) else len(c.sequence) for c in chains)
    n_pos = sum(sum(c["labels"]) if isinstance(c, dict) else sum(c.labels) for c in chains)
    return {
        "n_chains": len(chains),
        "n_residues": n_res,
        "n_epitope": n_pos,
        "epitope_rate": (n_pos / n_res) if n_res else 0.0,
    }
