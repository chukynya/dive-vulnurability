"""
Assemble a regenerated synthetic set: DROP the old templated Bad Randomness / Front Running
synthetic, KEEP everything else, and fold in the freshly-generated DIVERSE BR/FR — writing the
canonical Dataset/synthetic/ schema that tools/build_split.py expects.

Pipeline (run in order):
  # in the sibling repo (sc-vulnerability-detector), needs solc/solcx:
  python pipeline/mutation_inject.py      # diversified (16+16) + train-fold-only
  python pipeline/compile_synthetic.py    # -> Dataset/synthetic/bytecode.csv
  # then here:
  python tools/regen_synthetic_brfr.py
  python tools/build_split.py
  python tools/build_split_bytecode.py

It backs up the existing CSVs to *.bak and moves dropped BR/FR source files to
Dataset/synthetic/_dropped_brfr_src/ (non-destructive). Idempotent-ish: re-running re-reads
the *.bak-free current state, so keep a clean copy if you iterate.

NOTE: untested end-to-end here (no solc in this environment) — smoke-test on a few rows first.
"""
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # rm-sc-test
SYN = ROOT / "Dataset" / "synthetic"

LAB = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
       "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
DROP_PRIMARY = {"Bad Randomness", "Front Running"}
TARGET_NAME = {"BR": "Bad Randomness", "FR": "Front Running"}

# Existing canonical synthetic (rewritten in place; backed up to *.bak)
EX_PROV = SYN / "synthetic_provenance.csv"       # contractID, base_id, pragma_bucket, primary, classes
EX_LAB  = SYN / "synthetic_injected_labels.csv"  # contractID + 8 labels
EX_BYTE = SYN / "Bytecode_Synthetic.csv"         # contractID, bytecode
EX_SRC  = SYN / "Source_Synthetic"               # <contractID>.sol

# Fresh-generation inputs (from the sibling pipeline). ADJUST if you ran it elsewhere.
FRESH_ROOT  = Path(r"C:/Users/henry/Desktop/research-methodology/sc-vulnerability-detector/Dataset")
FRESH_SRC   = FRESH_ROOT / "Synthetic_LLMs"
FRESH_PROV  = FRESH_ROOT / "synthetic" / "provenance.csv"   # synthetic_id, source_contract_id, target_class, pattern_name, pragma_minor
FRESH_BYTE  = FRESH_ROOT / "synthetic" / "bytecode.csv"     # contractID(=synthetic_id), bytecode, solc_version
FRESH_LAB   = FRESH_ROOT / "DIVE_Labels_append.csv"         # contractID(=synthetic_id) + 8 labels


def main():
    for p in (EX_PROV, EX_LAB, EX_BYTE, FRESH_PROV, FRESH_BYTE, FRESH_LAB):
        assert p.exists(), f"missing input: {p}"

    # 1. existing synthetic -> keep non-BR/FR-primary
    prov = pd.read_csv(EX_PROV)
    prov["contractID"] = prov["contractID"].astype(int)
    keep_ids = set(prov.loc[~prov["primary"].isin(DROP_PRIMARY), "contractID"])
    drop_ids = set(prov["contractID"]) - keep_ids
    print(f"existing synthetic: {len(prov)} | keep non-BR/FR {len(keep_ids)} | drop BR/FR {len(drop_ids)}")

    ex_lab = pd.read_csv(EX_LAB);  ex_lab["contractID"] = ex_lab["contractID"].astype(int)
    ex_byte = pd.read_csv(EX_BYTE); ex_byte["contractID"] = ex_byte["contractID"].astype(int)
    prov_keep = prov[prov["contractID"].isin(keep_ids)].copy()
    lab_keep = ex_lab[ex_lab["contractID"].isin(keep_ids)][["contractID"] + LAB].copy()
    byte_keep = ex_byte[ex_byte["contractID"].isin(keep_ids)][["contractID", "bytecode"]].copy()

    # 2. fresh BR/FR -> only those that compiled
    fprov = pd.read_csv(FRESH_PROV)
    fbyte = pd.read_csv(FRESH_BYTE).rename(columns={"contractID": "synthetic_id"})
    flab = pd.read_csv(FRESH_LAB)
    flab = flab.rename(columns={flab.columns[0]: "synthetic_id"})
    compiled = set(fbyte["synthetic_id"])
    fprov = fprov[fprov["synthetic_id"].isin(compiled)].reset_index(drop=True)
    print(f"fresh BR/FR: provenance {len(pd.read_csv(FRESH_PROV))} | compiled {len(compiled)} | usable {len(fprov)}")

    # 3. assign new int IDs >= max(kept, 999999)+1 (no collision with kept)
    start_id = max([999_999] + list(keep_ids)) + 1
    fprov["new_id"] = range(start_id, start_id + len(fprov))
    flab_i = flab.set_index("synthetic_id")
    fbyte_i = fbyte.set_index("synthetic_id")

    # 4. move dropped BR/FR source out (non-destructive), then build fresh rows
    dropped_dir = SYN / "_dropped_brfr_src"; dropped_dir.mkdir(exist_ok=True)
    for cid in drop_ids:
        p = EX_SRC / f"{cid}.sol"
        if p.exists():
            shutil.move(str(p), str(dropped_dir / f"{cid}.sol"))

    new_prov, new_lab, new_byte = [], [], []
    for _, r in fprov.iterrows():
        sid, nid = r["synthetic_id"], int(r["new_id"])
        labels = [int(flab_i.loc[sid, c]) for c in LAB]
        classes = "|".join(c for c, v in zip(LAB, labels) if v == 1)
        primary = TARGET_NAME.get(str(r["target_class"]), str(r["target_class"]))
        pm = str(r.get("pragma_minor", "")).strip()
        pragma = f"0{int(float(pm))}" if pm not in ("", "-1", "nan") else ""
        new_prov.append({"contractID": nid, "base_id": int(r["source_contract_id"]),
                         "pragma_bucket": pragma, "primary": primary, "classes": classes})
        new_lab.append({"contractID": nid, **dict(zip(LAB, labels))})
        new_byte.append({"contractID": nid, "bytecode": fbyte_i.loc[sid, "bytecode"]})
        shutil.copyfile(FRESH_SRC / f"{sid}.sol", EX_SRC / f"{nid}.sol")

    # 5. back up + write combined
    for f in (EX_PROV, EX_LAB, EX_BYTE):
        shutil.copyfile(f, f.with_suffix(f.suffix + ".bak"))
    pd.concat([prov_keep, pd.DataFrame(new_prov)], ignore_index=True).to_csv(EX_PROV, index=False)
    pd.concat([lab_keep, pd.DataFrame(new_lab)], ignore_index=True).to_csv(EX_LAB, index=False)
    pd.concat([byte_keep, pd.DataFrame(new_byte)], ignore_index=True).to_csv(EX_BYTE, index=False)

    out = pd.read_csv(EX_LAB)
    print(f"\nWROTE synthetic: {len(out)} rows = {len(keep_ids)} kept + {len(new_prov)} fresh BR/FR")
    print("synthetic per-class totals:")
    for c in LAB:
        print(f"  {c:>26s}  {int(out[c].sum())}")
    print(f"\nbackups: *.bak | dropped BR/FR source moved to {dropped_dir}")
    print("next: python tools/build_split.py  &&  python tools/build_split_bytecode.py")


if __name__ == "__main__":
    main()
