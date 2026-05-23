"""
Emit SELF-CONTAINED bytecode split CSVs from the manifests in Dataset/splits/.

Each output row = contractID, bytecode, + 8 label columns, so the files are
directly trainable with no cross-dataset join. Real bytecode comes from
Bytecode_filled.csv, synthetic (contractID >= 1_000_000) from
Bytecode_Synthetic.csv. All split contracts have usable bytecode (verified
100%), so the folds stay exactly 80/10/10: 22,774 / 2,847 / 2,847.

Inputs:  Dataset/splits/{train,val,test}.csv (manifests),
         Dataset/backup/Bytecode_filled.csv, Dataset/synthetic/Bytecode_Synthetic.csv
Output:  Dataset/used/{train,val,test}.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "Dataset"
SRC = D / "splits"
OUT = D / "used"
OUT.mkdir(parents=True, exist_ok=True)

LAB = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
       "DoS", "Bad Randomness", "Front Running", "Time manipulation"]

bc = pd.concat([
    pd.read_csv(D / "backup" / "Bytecode_filled.csv", usecols=["contractID", "bytecode"]),
    pd.read_csv(D / "synthetic" / "Bytecode_Synthetic.csv", usecols=["contractID", "bytecode"]),
], ignore_index=True)
bc_map = bc.drop_duplicates("contractID").set_index("contractID")["bytecode"]

for f in ["train", "val", "test"]:
    m = pd.read_csv(SRC / f"{f}.csv")
    m["bytecode"] = m["contractID"].map(bc_map)
    assert m["bytecode"].notna().all(), f"{f}: some contractIDs have no bytecode"
    out = m[["contractID", "bytecode"] + LAB]
    out.to_csv(OUT / f"{f}.csv", index=False)
    print(f"{f:5s} rows={len(out):6d}  synthetic={int((out['contractID']>=1_000_000).sum()):5d}")
print("done ->", OUT)
