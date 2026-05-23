"""
Build the canonical split as CSVs so the FINAL (real + synthetic) dataset is
80/10/10, with every synthetic row living in train only.

Inputs:
  Dataset/backup/DIVE_Labels.csv                  real labels   (R = 22,330)
  Dataset/synthetic/synthetic_injected_labels.csv synthetic     (S = 6,138, contractID >= 1_000_000)
  Dataset/synthetic/synthetic_provenance.csv      base_id -> real parent (all parents clean)

Strategy: pin every synthetic parent (all clean/all-zero) into train, then
multilabel-stratify the remaining real contracts into rest-of-train / val / test
with val = test = round(0.10 * (R + S)). This keeps all S synthetic, guarantees
no parent lands in val/test (zero leakage), and makes train + synthetic = 80%.

Outputs (Dataset/splits/): train.csv, val.csv, test.csv  (contractID + 8 labels).
Reproducible: seed 42.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
OUT = DATA / "splits"
OUT.mkdir(parents=True, exist_ok=True)

LAB = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
       "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
COLS = ["contractID"] + LAB

real = pd.read_csv(DATA / "backup" / "DIVE_Labels.csv")[COLS]
syn = pd.read_csv(DATA / "synthetic" / "synthetic_injected_labels.csv")[COLS]
prov = pd.read_csv(DATA / "synthetic" / "synthetic_provenance.csv")

R, S = len(real), len(syn)
T = R + S
val_n = test_n = round(0.10 * T)
real_train_n = R - val_n - test_n
assert syn["contractID"].min() >= 1_000_000, "synthetic IDs must be >= 1_000_000"

parents = set(prov["base_id"].unique())
is_parent = real["contractID"].isin(parents)
pinned = real[is_parent]                         # all go to train
pool = real[~is_parent].reset_index(drop=True)   # everything else gets stratified

pool_test_n = test_n
pool_val_n = val_n
Yp = pool[LAB].values

mss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=pool_test_n, random_state=SEED)
rest_pos, test_pos = next(mss1.split(np.zeros(len(pool)), Yp))
mss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=pool_val_n, random_state=SEED)
trainrest_local, val_local = next(mss2.split(np.zeros(len(rest_pos)), Yp[rest_pos]))
val_pos = rest_pos[val_local]
trainrest_pos = rest_pos[trainrest_local]

real_train = pd.concat([pinned, pool.iloc[trainrest_pos]], ignore_index=True)
val_df = pool.iloc[val_pos].reset_index(drop=True)
test_df = pool.iloc[test_pos].reset_index(drop=True)

train_out = pd.concat([real_train, syn], ignore_index=True)
val_out, test_out = val_df, test_df

train_out.to_csv(OUT / "train.csv", index=False)
val_out.to_csv(OUT / "val.csv", index=False)
test_out.to_csv(OUT / "test.csv", index=False)

# ---- verification ----
def stats(name, df):
    clean = int((df[LAB].sum(axis=1) == 0).sum())
    pos = df[LAB].sum().astype(int)
    print(f"{name:5s} n={len(df):6d}  clean={clean:5d}  "
          f"FR={pos['Front Running']:4d} BR={pos['Bad Randomness']:4d} DoS={pos['DoS']:4d}")

ft = len(train_out)
print(f"R={R} S={S} T={T} | target val=test={val_n} real_train={real_train_n}")
print(f"FINAL ratio: train={ft} val={len(val_out)} test={len(test_out)} -> "
      f"{ft/T:.4f}/{len(val_out)/T:.4f}/{len(test_out)/T:.4f}")
stats("train", train_out); stats("val", val_out); stats("test", test_out)

val_ids = set(val_out["contractID"]); test_ids = set(test_out["contractID"])
train_ids = set(train_out["contractID"])
real_train_ids = set(real_train["contractID"])
syn_max_in_eval = max([0] + [c for c in (val_ids | test_ids) if c >= 1_000_000])
print("\nchecks:")
print(f"  synthetic in val/test (must be 0): {syn_max_in_eval and 'FOUND' or 0}")
print(f"  parents in val/test (must be 0):   {len(parents & (val_ids | test_ids))}")
print(f"  all parents in train:              {parents.issubset(real_train_ids)}")
print(f"  real folds disjoint:               {len(real_train_ids & val_ids)==0 and len(real_train_ids & test_ids)==0 and len(val_ids & test_ids)==0}")
print(f"  all {S} synthetic in train:         {syn['contractID'].isin(train_ids).all()}")
print(f"  real total preserved ({R}):       {len(real_train)+len(val_out)+len(test_out)==R}")
