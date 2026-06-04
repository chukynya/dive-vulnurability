#!/usr/bin/env python3
"""Build aug_gcb_1.ipynb.

aug-gcb-1 = the dive-dedup-4 engine (frozen GraphCodeBERT unit embeddings on
bytecode-derived readable EVM assembly basic blocks + set-Transformer aggregator)
run on the **pre-split, CBOR-stripped AUGMENTED no-dedup benchmark**:

    train 27,130 (15,630 real + 11,500 SolidiFI synthetic) | val 2,233 | test 4,467

The only difference from dedup-4 is the data layer: this dataset ships pre-split,
bytecode-modality, ALREADY CBOR-stripped (lowercase hex, no 0x), as six flat CSVs
under augmented/ ({train,val,test}_{bytecode,labels}.csv). There is no arm switch -
a single augmented run; val/test are 100% real and frozen.

This script loads build_dive_dedup_4.ipynb and surgically replaces the title, config
(single-dataset wiring + naming), and fold-load (per-split bytecode/label CSV loader
+ new gate numbers) cells, plus narrative. The whole bytecode disassembler, frozen
GraphCodeBERT stage-1, set-Transformer, ASL+aux loss, calibration, sampler, EMA,
training, eval, graphs, inference and table cells are kept verbatim. Run:
  python tools/build_aug_gcb_1.py
"""
import io
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "build_dive_dedup_4.ipynb"
OUT_NB = ROOT / "aug_gcb_1.ipynb"


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


# ---------------------------------------------------------------------------
TITLE_MD = r"""
# aug-gcb-1 — bytecode-as-assembly on the augmented (no-dedup) benchmark

**What this is.** The **dive-dedup-4 engine** (frozen GraphCodeBERT unit embeddings →
set-Transformer aggregator → per-class heads + aux BCE), where each contract's **bytecode**
is turned into **readable EVM assembly basic blocks** and those blocks are the units
GraphCodeBERT encodes — run on the **pre-split, CBOR-stripped AUGMENTED no-dedup benchmark**:

| Fold | Rows | Composition |
|---|---|---|
| train | 27,130 | 15,630 real + 11,500 SolidiFI synthetic (`buggy_<id>_<type>`) |
| val   | 2,233  | 100% real, frozen |
| test  | 4,467  | 100% real, frozen |

This is a **single augmented run — no arm switch**. The whole model/loss/calibration/sampler/
EMA stack is dedup-4 verbatim; only the **data layer** changes.

**The data layer (the new bit).** Unlike the dedup line (one global `Bytecode_filled.csv` +
labels you re-fold), this dataset ships **pre-split** as six flat CSVs under `augmented/`:
`{train,val,test}_bytecode.csv` (`contractID, contractAddress, bytecode`) and
`{train,val,test}_labels.csv` (`contractID` + the 8 class columns). The bytecode is **already
CBOR-stripped** (lowercase hex, no `0x`) — so the splitter's `strip_metadata` is now a
defensive no-op rather than the load-bearing first pass. We stream the three bytecode CSVs into
one `contractID → hex` map and join each fold's labels on `contractID`.

**Preprocessing (cell 7, unchanged from dedup-4).** Per contract: disassemble the runtime
bytecode → decode PUSH operands (hex) + 4-byte function selectors → segment into
JUMPDEST-delimited basic blocks → emit each block as readable labelled assembly → risk-rank and
cap to `MAX_UNITS` (dispatcher/entry blocks first, then risky-opcode density + length). The
dispatcher/entry blocks take unit-type 0; ordinary basic blocks take type 1.

**Caveat to read with.** GraphCodeBERT was pretrained on high-level source, **not** EVM
assembly, so assembly text is out-of-distribution for it. The selector/operand decoding nudges
the text closer to source-like. `MAX_UNITS` stays at dedup-4's 32 for **engine parity**; on this
corpus most contracts have far more basic blocks than 32, so this is an inherent heavy-truncation
regime and the **risk-ranking** is what carries the signal. Raising `MAX_UNITS` (watch the 4h
`STAGE1_BUDGET_SECS` gate) is the first knob if test F1 disappoints.

**Reading the result.**
1. **Gates:** leakage (train∩val, train∩test, val∩test = 0; no synthetic in val/test) + recorded
   fold sizes/support (val 2,233, test 4,467) + bytecode coverage (100%) + units/contract.
2. **`test_calibrated_tuned.f1_macro`** on the 4,467 test — the headline.
3. **Per-class F1 / overfit gap on BR / FR / DoS** — the rare classes the synthetic rows target.
"""

CONFIG_TOP = r'''# === Augmented (no-dedup) bytecode-modality run — single dataset, no arms =====
# The dive-dedup-4 GraphCodeBERT engine (frozen unit embeddings on bytecode-derived
# readable EVM assembly basic blocks + set-Transformer) on the PRE-SPLIT, ALREADY
# CBOR-stripped augmented benchmark:
#   train 27,130 (15,630 real + 11,500 SolidiFI synthetic) | val 2,233 | test 4,467
# The dataset ships as six flat CSVs under augmented/:
#   {train,val,test}_bytecode.csv  -> contractID, contractAddress, bytecode (stripped)
#   {train,val,test}_labels.csv    -> contractID + 8 class columns
# Single configuration: no arm switch. Val/test are 100% real and frozen.
RUN_TAG    = "aug"
AUG_SOURCE = "aug"     # alias kept so downstream metrics/graph/table cells key cleanly
_ARM       = RUN_TAG   # output-naming tag (downstream cells reference variables, not literals)

# -- Locate the augmented/ dataset dir (Kaggle mounts or local) ---------------
def _has(d, *parts):
    p = Path(d)
    for x in parts:
        p = p / x
    return p.exists()

def _find_aug_root():
    """Find the augmented/ dir holding the 6 pre-split CSVs (+ manifest.json)."""
    for c in ("/kaggle/input/datasets/henrychristian7555/dive-augment-no-dedup/augmented",
              "/kaggle/input/dive-augment-no-dedup/augmented",
              "/kaggle/input/datasets/henrychristian7555/dive-augment-no-dedup",
              "/kaggle/input/dive-augment-no-dedup",
              r"C:\Users\henry\Desktop\research-methodology\Dataset\augmented",
              "../Dataset/augmented",
              "augmented"):
        if _has(c, "train_labels.csv") and _has(c, "train_bytecode.csv"):
            return Path(c)
    base = Path("/kaggle/input")
    if base.exists():
        for mp in sorted(base.rglob("train_labels.csv")):
            r = mp.parent
            if (r / "train_bytecode.csv").exists() and (r / "test_labels.csv").exists():
                return r
    return None

DATA_ROOT = _find_aug_root()
assert DATA_ROOT is not None, (
    "augmented dataset not attached. Add Input -> 'dive-augment-no-dedup' "
    "(henrychristian7555/dive-augment-no-dedup). Expected <mount>/augmented/"
    "{train,val,test}_{bytecode,labels}.csv.")

# -- Per-fold paths: split label CSVs + split bytecode CSVs -------------------
TRAIN_CSV = DATA_ROOT / "train_labels.csv"
VAL_CSV   = DATA_ROOT / "val_labels.csv"
TEST_CSV  = DATA_ROOT / "test_labels.csv"
TRAIN_BC_CSV = DATA_ROOT / "train_bytecode.csv"
VAL_BC_CSV   = DATA_ROOT / "val_bytecode.csv"
TEST_BC_CSV  = DATA_ROOT / "test_bytecode.csv"

_FOLD_CSV = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}
_FOLD_BC  = {"train": TRAIN_BC_CSV, "val": VAL_BC_CSV, "test": TEST_BC_CSV}

def _fold_csv(name):
    """Return path to the label split CSV for fold 'train', 'val', or 'test'."""
    p = _FOLD_CSV[name]
    if not p.exists():
        raise FileNotFoundError(f"No label CSV for fold '{name}' - tried: {p}")
    return p

# Real contracts have numeric IDs; synthetic (TRAIN-only) injections are buggy_<id>_<type>.
def _is_synthetic(cid):
    return not str(cid).isdigit()

def _aug_kind(cid):
    return "real" if str(cid).isdigit() else "buggy"

# === OUTPUT NAMING ===========================================================
OUT_DIR     = Path("/kaggle/working");   OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR   = OUT_DIR / "cache";         CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH    = OUT_DIR / f"aug_gcb1_{_ARM}_train.log"
HIST_CSV    = OUT_DIR / f"history_{_ARM}.csv"
HIST_JSON   = OUT_DIR / f"history_{_ARM}.json"
STATE_PATH  = OUT_DIR / f"last_state_{_ARM}.pt"
BEST_PATH   = OUT_DIR / f"best_model_{_ARM}.pt"
EMB_NPZ     = CACHE_DIR / f"aug_gcb1_units_{_ARM}.npz"
METRICS_ARM_JSON = OUT_DIR / f"metrics_arm_{_ARM}.json"

# -- Resume control -----------------------------------------------------------
RESUME_FROM = None    # set to str(STATE_PATH) to resume

'''

CONFIG_TAIL = r'''
# -- Verification gates -------------------------------------------------------
MIN_BYTECODE_COVERAGE = 0.99      # >=99% of label rows must have a bytecode entry
MIN_UNIT_COVERAGE     = 0.99      # >=99% of contracts must yield >=1 unit (basic block)
STAGE1_BUDGET_SECS    = 4 * 3600  # abort if estimated stage-1 encode exceeds this

# -- Sanity check on data files -----------------------------------------------
assert TRAIN_CSV.exists(), f"Missing train label CSV {TRAIN_CSV}"
assert VAL_CSV.exists() and TEST_CSV.exists(), f"Missing val/test label CSV under {DATA_ROOT}"
for _nm, _p in _FOLD_BC.items():
    assert _p.exists(), f"Missing {_nm} bytecode CSV {_p}"
print(f"RUN_TAG={RUN_TAG}  DATA_ROOT={DATA_ROOT}", flush=True)
print(f"  train labels  ={TRAIN_CSV}", flush=True)
print(f"  train bytecode={TRAIN_BC_CSV}", flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
print(f"Stage1: MAX_UNITS={MAX_UNITS} UNIT_TOKENS={UNIT_TOKENS} model={GCB_MODEL}", flush=True)
print(f"Stage2: d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} d_ff={D_FF}", flush=True)
'''

# fold-load cell — fully rewritten for the pre-split layout
FOLDLOAD_CODE = r'''# === Load per-split bytecode CSVs into one in-memory map =====================
# This dataset ships PRE-SPLIT, bytecode-modality, ALREADY CBOR-stripped (lowercase
# hex, no 0x). Each fold has its own {train,val,test}_bytecode.csv + _labels.csv in a
# flat augmented/ dir. We stream the three bytecode CSVs into one contractID -> hex
# map (real IDs are numeric strings; synthetic TRAIN-only IDs are buggy_<id>_<type>),
# then join each fold's labels on contractID. Bytecode coverage is gated below.
def _load_bc_csv(path, label):
    d = pd.read_csv(path, usecols=["contractID", "bytecode"], dtype={"contractID": str})
    n0 = len(d)
    d = d.dropna(subset=["bytecode"])
    d = d[d["bytecode"].astype(str).str.len() > 4]
    log(f"  {label:>6s}: {len(d):6d} / {n0:6d} rows with bytecode  ({path.name})")
    return dict(zip(d["contractID"].astype(str), d["bytecode"].astype(str)))


log("Loading per-split bytecode CSVs...")
bytecode_map = {}
for _nm in ("train", "val", "test"):
    bytecode_map.update(_load_bc_csv(_FOLD_BC[_nm], _nm))
log(f"Combined bytecode map: {len(bytecode_map):,} unique IDs")


def _load_fold_raw(name):
    d = pd.read_csv(_fold_csv(name))
    d["contractID"] = d["contractID"].astype(str)
    have = d["contractID"].apply(lambda c: c in bytecode_map)
    cov = have.mean()
    log(f"  {name:>5s} (raw): {len(d):6d} rows | bytecode coverage {cov:.2%}")
    if cov < 1.0:
        log(f"    dropping {int((~have).sum())} rows with no bytecode")
    return d[have].reset_index(drop=True)

# val / test: frozen, 100% real.
val_df  = _load_fold_raw("val")
test_df = _load_fold_raw("test")

# train: the full augmented fold (real + SolidiFI synthetic). No arm filtering -
# the synthetic rows stay in; the quality-rebalance sampler (cell 11) is what damps
# the off-distribution synthetic rare positives via _aug_kind.
train_df = _load_fold_raw("train")
train_df["_kind"] = train_df["contractID"].apply(_aug_kind)
_kc = train_df["_kind"].value_counts().to_dict()
log(f"train composition: real={_kc.get('real',0)} buggy(synthetic)={_kc.get('buggy',0)}")
train_df = train_df.drop(columns=["_kind"]).reset_index(drop=True)

# ===== VERIFICATION GATES (must pass before any training) ====================
def _ids(d):
    return set(d["contractID"].astype(str))

tr_ids, v_ids, te_ids = _ids(train_df), _ids(val_df), _ids(test_df)
tr_real = {c for c in tr_ids if _aug_kind(c) == "real"}

g_tr_te, g_tr_v, g_v_te = tr_ids & te_ids, tr_ids & v_ids, v_ids & te_ids
log(f"[gate] leakage  train.test={len(g_tr_te)}  train.val={len(g_tr_v)}  val.test={len(g_v_te)}")
assert not g_tr_te, f"LEAKAGE: {len(g_tr_te)} train IDs in test (e.g. {list(g_tr_te)[:5]})"
assert not g_tr_v,  f"LEAKAGE: {len(g_tr_v)} train IDs in val"
assert not g_v_te,  f"LEAKAGE: {len(g_v_te)} val IDs in test"
n_syn_eval = int(pd.concat([val_df, test_df])["contractID"].apply(_is_synthetic).sum())
assert n_syn_eval == 0, f"LEAKAGE: {n_syn_eval} synthetic rows in val/test"

# Recorded fold sizes / support for THIS augmented benchmark (manifest + CSV scan).
EXPECTED_VAL_ROWS  = 2233
EXPECTED_TEST_ROWS = 4467
EXPECTED_TEST_SUPPORT = {
    "Reentrancy": 2281, "Access Control": 3346, "Arithmetic": 1909,
    "Unchecked Return Values": 1183, "DoS": 756, "Bad Randomness": 127,
    "Front Running": 121, "Time manipulation": 1265}
_sup = {c: int(test_df[c].sum()) for c in LABEL_COLS}
log(f"[gate] val rows={len(val_df)} (expect {EXPECTED_VAL_ROWS})  "
    f"test rows={len(test_df)} (expect {EXPECTED_TEST_ROWS})")
log(f"[gate] test support={_sup}")
if (len(val_df) != EXPECTED_VAL_ROWS or len(test_df) != EXPECTED_TEST_ROWS
        or _sup != EXPECTED_TEST_SUPPORT):
    log("[gate] WARNING: this val/test set does NOT match the recorded augmented "
        "benchmark - check the dataset version.")
else:
    log("[gate] OK: val/test == recorded augmented benchmark (2,233 / 4,467).")

# ----------------------------------------------------------------------------
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
log(f"working set {df.shape} | "
    f"train={len(train_df)} (real={len(tr_real)}, synthetic={len(train_df)-len(tr_real)}) "
    f"val={len(val_df)} test={len(test_df)}")
log("Positives per class (train fold):")
for c in LABEL_COLS:
    log(f"  {c:>26s}  {int(train_df[c].sum()):>5d}")
'''

SUMMARY_MD = r"""
## Summary

**aug-gcb-1 = the dive-dedup-4 GraphCodeBERT engine on the augmented (no-dedup) benchmark.**
Same bytecode-as-assembly units, set-Transformer, ASL + aux-BCE loss, isotonic calibration,
threshold tuning, quality-rebalance sampler, EMA, gates and graphics as dedup-4 — the only change
is the **data layer**: a single pre-split, already-CBOR-stripped augmented dataset (no arm switch).

**Pipeline.**
1. Stream the three per-split `*_bytecode.csv` into one `contractID → hex` map; join each fold's
   `*_labels.csv` on `contractID` (bytecode already stripped — `strip_metadata` is a defensive no-op).
2. Disassemble → decode PUSH operands + 4-byte selectors → segment into `JUMPDEST`-delimited basic
   blocks → risk-rank and cap to `MAX_UNITS`.
3. Frozen GraphCodeBERT mean-pools each block (≤`UNIT_TOKENS` tokens) into a 768-d vector — cached
   once to `cache/aug_gcb1_units_aug.npz`.
4. The set-Transformer aggregates the block set (+ dispatcher/body type embedding) and feeds the
   8 per-class heads + aux BCE, trained with the same loss / sampler / EMA / calibration / thresholds.

**Folds.** train 27,130 (15,630 real + 11,500 SolidiFI synthetic) · val 2,233 · test 4,467; val/test
100% real and frozen. Runs on Kaggle T4×2 (DataParallel) within the 8h budget.

**Outputs (`/kaggle/working/`, suffixed `aug`).**
- `cache/aug_gcb1_units_aug.npz` — frozen block embeddings (reused on resume).
- `best_model_aug.pt`, `last_state_aug.pt`, `metrics_arm_aug.json`, `history_aug.csv`.
- `loss_curves_aug.png`, `f1_macro_curves_aug.png`,
  `per_class_f1_train_val_test_aug.png`, `per_class_overfit_gap_aug.png`.

**What to read first.**
1. Gates: leakage, recorded fold sizes (val 2,233 / test 4,467) + support, bytecode coverage,
   units/contract, stage-1 timing.
2. **`test_calibrated_tuned.f1_macro`** on the 4,467 test — the headline.
3. Per-class F1 / overfit gap on Bad Randomness / Front Running / DoS — the rare classes the
   synthetic rows target.
"""


def main():
    nb = json.loads(io.open(SRC_NB, encoding="utf-8").read())
    cells = nb["cells"]

    # --- 1) splice config: keep dedup-4 hparams (Labels..Checkpointing) verbatim
    cfg = "".join(cells[5]["source"])
    a = cfg.index("# -- Labels")
    b = cfg.index("# -- Verification gates")
    hparam_middle = cfg[a:b].rstrip("\n")
    new_config = CONFIG_TOP + hparam_middle + "\n" + CONFIG_TAIL
    cells[5]["source"] = new_config.splitlines(keepends=True)

    # --- 2) title + fold-load + summary -------------------------------------
    cells[0] = md(TITLE_MD)
    cells[11]["source"] = FOLDLOAD_CODE.splitlines(keepends=True)
    cells[48] = md(SUMMARY_MD)

    # --- 3) global cosmetic renames across all cells -------------------------
    for c in cells:
        s = "".join(c["source"])
        s2 = (s.replace("dive-dedup-4", "aug-gcb-1")
                .replace("dive_dedup4", "aug_gcb1"))
        if s2 != s:
            c["source"] = s2.splitlines(keepends=True)

    # --- 4) clear outputs / execution counts ---------------------------------
    for c in cells:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # --- 5) normalize (cell ids) + write ------------------------------------
    try:
        import nbformat
        nbobj = nbformat.from_dict(nb)
        _, nbobj = nbformat.normalize(nbobj)
        nbformat.write(nbobj, str(OUT_NB))
    except Exception:
        io.open(OUT_NB, "w", encoding="utf-8").write(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"wrote {OUT_NB}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
