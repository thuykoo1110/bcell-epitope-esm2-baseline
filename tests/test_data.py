from epitope import data


def line(record, serial, name, resn, chain, resseq, icode=" ", altloc=" "):
    """Build a fixed-column PDB ATOM/HETATM line."""
    return (f"{record:<6s}{serial:5d} {name:<4s}{altloc}{resn:>3s} {chain}{resseq:>4d}{icode}   "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}")


def atom(serial, name, resn, chain, resseq, icode=" ", altloc=" "):
    return line("ATOM", serial, name, resn, chain, resseq, icode, altloc)


def hetatm(serial, name, resn, chain, resseq, icode=" ", altloc=" "):
    return line("HETATM", serial, name, resn, chain, resseq, icode, altloc)


HEADER = "HEADER    IMMUNE SYSTEM                           12-APR-21   7ABC"

PDB = "\n".join([
    HEADER,
    atom(1, "N", "MET", "A", 1),                       # not a CA atom: ignored
    atom(2, "CA", "MET", "A", 1),
    atom(3, "CA", "LYS", "A", 2),
    atom(4, "CA", "VAL", "A", 3),
    atom(5, "CA", "GLY", "A", 3, icode="A"),           # insertion code 3A
    atom(6, "CA", "SER", "A", 4, altloc="A"),          # alternate conformations: first one is kept
    atom(7, "CA", "SER", "A", 4, altloc="B"),
    hetatm(8, "CA", "MSE", "A", 5),                    # selenomethionine -> MET
    hetatm(9, "CA", "CSO", "A", 6),                    # modified cysteine (CS?) -> CYS
    hetatm(10, "CA", "SEP", "A", 7),                   # phosphoserine: not in DICT, dropped
    hetatm(11, "CA", "CA", "A", 8),                    # calcium ion named CA: dropped
    "TER".ljust(21) + "A",                             # end of chain A
    atom(12, "CA", "ALA", "A", 9),                     # after TER: ignored
    atom(13, "CA", "ALA", "B", 1),                     # other chain: ignored
])


def test_parse_date():
    assert data.parse_pdb_date(HEADER) == 20210412
    assert data.parse_pdb_date(HEADER.replace("12-APR-21", "01-JAN-98")) == 19980101


def test_normalize_resname_mirrors_judge():
    assert data.normalize_resname("ARG") == ("", "ARG")
    assert data.normalize_resname("AARG") == ("A", "ARG")      # altLoc + resName
    assert data.normalize_resname("MSE") == ("", "MET")
    assert data.normalize_resname("CCS") == ("", "CYS")
    assert data.normalize_resname("CSO") == ("", "CYS")
    assert data.normalize_resname("SEP") is None
    assert data.normalize_resname("CA") is None


def test_parse_chain_sequence_and_sites():
    date, seq, sites = data.parse_pdb_chain(PDB, "A")
    assert date == 20210412
    assert seq == "MKVGSMC"
    assert sites == ["1", "2", "3", "3A", "4", "5", "6"]


def test_parse_other_chain():
    _, seq, sites = data.parse_pdb_chain(PDB, "B")
    assert seq == "A" and sites == ["1"]


def test_altloc_keeps_first_seen_even_if_not_A():
    pdb = "\n".join([HEADER, atom(1, "CA", "SER", "A", 1, altloc="B"), atom(2, "CA", "SER", "A", 1, altloc="C")])
    _, seq, sites = data.parse_pdb_chain(pdb, "A")
    assert seq == "S" and sites == ["1"]


def test_label_mapping_exact_and_insertion_code():
    _, seq, sites = data.parse_pdb_chain(PDB, "A")
    labels = data.map_epitope_labels(seq, sites, "2_LYS, 3A_GLY, 4_SER")
    assert labels == [0, 1, 0, 1, 1, 0, 0]


def test_label_mapping_modified_residues():
    _, seq, sites = data.parse_pdb_chain(PDB, "A")
    # label tokens may use MSE / CCS directly (both are in GraphBepi's DICT)
    assert data.map_epitope_labels(seq, sites, "5_MSE, 6_CCS") == [0, 0, 0, 0, 0, 1, 1]
    # CSO is not in DICT, so a label token with it is ignored
    assert data.map_epitope_labels(seq, sites, "6_CSO") == [0, 0, 0, 0, 0, 0, 0]


def test_label_mapping_prefix_fallback_and_mismatch():
    _, seq, sites = data.parse_pdb_chain(PDB, "A")
    # "3_GLY": residue 3 is VAL (mismatch) -> falls back to the prefix match "3A" which is GLY
    assert data.map_epitope_labels(seq, sites, "3_GLY") == [0, 0, 0, 1, 0, 0, 0]
    # residue name that disagrees everywhere is ignored
    assert data.map_epitope_labels(seq, sites, "2_ALA") == [0, 0, 0, 0, 0, 0, 0]


def test_filter_and_splits():
    def mk(i, date, n_pos=1, length=10):
        return data.Chain(f"c{i}", date, "A" * length, [str(k) for k in range(length)],
                          [1] * n_pos + [0] * (length - n_pos))
    chains = [mk(0, 20150101), mk(1, 20190101), mk(2, 20200101), mk(3, 20210501), mk(4, 20190601, n_pos=0)]
    kept = [c for c in chains if data.keep_chain(c)]
    assert [c.id for c in kept] == ["c0", "c1", "c2", "c3"]
    train, test = data.split_by_date(kept)
    assert [c.id for c in test] == ["c3"]
    tr, va = data.split_train_val(train, 0.34)
    assert [c.id for c in va] == ["c2"]          # most recent training chain held out
    assert not {c.id for c in tr} & {c.id for c in va}


def test_jsonl_roundtrip(tmp_path):
    c = data.Chain("1abc_A", 20200101, "MKV", ["1", "2", "3"], [0, 1, 0])
    data.write_jsonl([c], tmp_path / "x.jsonl")
    rec = data.read_jsonl(tmp_path / "x.jsonl")[0]
    assert rec["sequence"] == "MKV" and rec["labels"] == [0, 1, 0]
    assert data.dataset_stats([rec])["n_epitope"] == 1
