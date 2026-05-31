# Transform build_dive_dedup_1.ipynb -> build_dive_dedup_2.ipynb
# Surgical: only cells 0 (intro md), 5 (config), 11 (loader), 43 (graph titles),
# 47 (arm table) change. All model / loss / training / eval cells stay byte-identical.
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nb = json.load(open(ROOT / "build_dive_dedup_1.ipynb", encoding="utf-8"))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None


# ---------------------------------------------------------------- cell 0 (md)
CELL0 = r'''# dive-dedup-2 — augmentation ablation: real-only vs SolidiFI-augmented

**What this is.** The exact dive-dedup-1 pipeline (frozen GraphCodeBERT +
set-Transformer, identical config), run as a clean **A/B on training data only**.
Two arms share the *same* frozen real val/test (byte-identical across both
datasets) and the *same* model — the **only** variable is what goes into train:

| `ARM` | Train corpus | Rows | Augmentation |
|---|---|---|---|
| `"dedup"` | `dive-dedup` real dedup train | 13,143 | none (baseline) |
| `"synth-dedup"` | `dive-synthesized-dedup/final` train | 24,640 | + 11,497 SolidiFI-injected synthetic |

Val = 1,878 · Test = 3,756, identical to dive-dedup-1 (the locked
`MultilabelStratifiedShuffleSplit` 70/10/20 dedup benchmark). The injected rows
are `buggy_<baseID>_<type>` contracts whose `.sol` ships in `final/train/`.

**How to run the comparison (Kaggle T4 × 2).** This notebook trains **one arm per
execution**, exactly like dive-dedup-1's arm switch. In a single session:

1. set `ARM = "dedup"`, **Run All** → writes `metrics_arm_dedup.json`;
2. set `ARM = "synth-dedup"`, **Run All** → writes `metrics_arm_synth-dedup.json`;
3. the final cell (**§22**) reads every `metrics_arm_*.json` in `/kaggle/working`
   and tabulates macro-F1 + per-class F1 with the augmentation delta — all on the
   one fixed real test set.

Each arm caches its Stage-1 unit embeddings under a distinct name
(`dive_dedup2_units_<arm>.npz`), so switching arms re-encodes (the working sets
differ) but never collides.

**Datasets (Kaggle).**
- `henrychristian7555/dive-dedup` → `…/dive-dedup/Data-before-aug/{splits/, Source/}`
- `henrychristian7555/dive-synthesized-dedup` → `…/dive-synthesized-dedup/Dataset/{final/, Source/, …}`

Both are auto-located by scanning `/kaggle/input` (falls back to the matching
local folder when run off-Kaggle).

## What stays identical to dive-dedup-1

- Frozen GraphCodeBERT + function-level unit splitting (MAX_UNITS=32, UNIT_TOKENS=256).
- Set-Transformer aggregator: d_model=384, 6 heads, 3 layers, d_ff=1024.
- DROPOUT=0.30, DROP_PATH=0.20, WEIGHT_DECAY=0.10.
- AsymmetricLossWeighted (g-=4, g+=1, clip=0.05) + 0.3*BCE aux head on (Bad Randomness, Front Running).
- EMA decay 0.999, AdamW with selective weight decay, cosine warmup, grad clip 1.0.
- WeightedRandomSampler (sqrt-inv-freq per-sample weights, clipped [1,5]).
- Per-class isotonic calibration on val -> 41-pt threshold tune (min-precision 0.25 for support < 200).
- Per-class F1 + PR-AUC every epoch on val; best-epoch val + test per-class blocks; the same graphs.
- Atomic full-state checkpointing, tee logger, history.csv.
'''
set_src(0, CELL0)


# ---------------------------------------------------------------- cell 5 head
s5 = src(5)
i_labels = s5.index("# -- Labels --")
i_sanity = s5.index("# -- Sanity check on data files --")
constants = s5[i_labels:i_sanity]   # KEEP VERBATIM (labels + all hyperparams)

NEW_HEAD = r'''# === ARM SWITCH ==============================================================
# Augmentation ablation. Identical model/config; the ONLY variable is the train
# corpus. Val/test are byte-identical across both datasets (the locked dedup
# 70/10/20 benchmark), so the two arms are directly comparable.
#   "dedup"        baseline  — real-only dedup train (13,143)            [no aug]
#   "synth-dedup"  augmented — real + SolidiFI synthetic (24,640 total)  [final/]
# Run once per arm in the same Kaggle session (set ARM -> Run All; switch -> Run
# All); the final cell tabulates both on the shared real test set.
ARM = "dedup"          # <- flip to "synth-dedup" for the augmented arm
assert ARM in ("dedup", "synth-dedup"), f"bad ARM {ARM!r}"

# -- Locate both datasets (Kaggle mounts or local) ----------------------------
def _has(d, *parts):
    p = Path(d)
    for x in parts:
        p = p / x
    return p.exists()

def _find_dedup_root():
    """dive-dedup root: <root>/{splits/Train_Labels.csv, Source/} and NO final/."""
    for c in ("/kaggle/input/datasets/henrychristian7555/dive-dedup/Data-before-aug",
              "/kaggle/input/dive-dedup/Data-before-aug",
              "/kaggle/input/datasets/henrychristian7555/dive-dedup",
              "/kaggle/input/dive-dedup",
              "dive-dedup"):
        if _has(c, "splits", "Train_Labels.csv") and _has(c, "Source") and not _has(c, "final"):
            return Path(c)
    base = Path("/kaggle/input")
    if base.exists():
        for sp in sorted(base.rglob("splits")):
            r = sp.parent
            if (sp / "Train_Labels.csv").exists() and (r / "Source").is_dir() and not (r / "final").exists():
                return r
    return None

def _find_synth_root():
    """dive-synthesized-dedup root: <root>/{final/Train_Labels.csv, final/train/}."""
    for c in ("/kaggle/input/datasets/henrychristian7555/dive-synthesized-dedup/Dataset",
              "/kaggle/input/dive-synthesized-dedup/Dataset",
              "/kaggle/input/datasets/henrychristian7555/dive-synthesized-dedup",
              "/kaggle/input/dive-synthesized-dedup",
              "dive-synthesized-dedup"):
        if _has(c, "final", "Train_Labels.csv") and _has(c, "final", "train"):
            return Path(c)
    base = Path("/kaggle/input")
    if base.exists():
        for fp in sorted(base.rglob("final")):
            if (fp / "Train_Labels.csv").exists() and (fp / "train").is_dir():
                return fp.parent
    return None

# -- Per-arm paths: train CSV + ordered source dirs --------------------------
# val/test load from each arm's own root; they are byte-identical, so the gates
# below pin them to the one recorded 3,756-row dedup benchmark either way.
if ARM == "dedup":
    DATA_ROOT = _find_dedup_root()
    assert DATA_ROOT is not None, (
        "dive-dedup not attached. Add Input -> 'dive-dedup' "
        "(henrychristian7555/dive-dedup). Expected <mount>/Data-before-aug/"
        "{splits/Train_Labels.csv, Source/}.")
    TRAIN_CSV = DATA_ROOT / "splits" / "Train_Labels.csv"
    VAL_CSV   = DATA_ROOT / "splits" / "Val_Labels.csv"
    TEST_CSV  = DATA_ROOT / "splits" / "Test_Labels.csv"
    SOURCE_DIRS = [DATA_ROOT / "Source"]
else:  # "synth-dedup"
    DATA_ROOT = _find_synth_root()
    assert DATA_ROOT is not None, (
        "dive-synthesized-dedup not attached. Add Input -> 'dive-synthesized-dedup' "
        "(henrychristian7555/dive-synthesized-dedup). Expected <mount>/Dataset/"
        "final/{Train_Labels.csv, train/}.")
    TRAIN_CSV = DATA_ROOT / "final" / "Train_Labels.csv"
    VAL_CSV   = DATA_ROOT / "final" / "Val_Labels.csv"
    TEST_CSV  = DATA_ROOT / "final" / "Test_Labels.csv"
    # train sols (real {cid}.sol + synthetic buggy_*.sol) live in final/train;
    # holdout sols in final/val|test; real corpus Source/ is a backstop.
    SOURCE_DIRS = [DATA_ROOT / "final" / "train",
                   DATA_ROOT / "final" / "val",
                   DATA_ROOT / "final" / "test",
                   DATA_ROOT / "Source"]

# sol_path: resolve contractID -> first existing {cid}.sol across SOURCE_DIRS.
_sol_cache = {}

def sol_path(cid):
    cid_s = str(cid)
    p = _sol_cache.get(cid_s)
    if p is None:
        p = next((d / f"{cid_s}.sol" for d in SOURCE_DIRS if (d / f"{cid_s}.sol").exists()),
                 SOURCE_DIRS[0] / f"{cid_s}.sol")   # miss -> non-existent; coverage gate catches it
        _sol_cache[cid_s] = p
    return p

_FOLD_CSV = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}

def _fold_csv(name):
    """Return path to the split CSV for fold 'train', 'val', or 'test'."""
    p = _FOLD_CSV[name]
    if not p.exists():
        raise FileNotFoundError(f"No split CSV for fold '{name}' - tried: {p}")
    return p

# Real contracts have numeric IDs; synthetic injections are buggy_<id>_<type>.
def _is_synthetic(cid):
    try:
        return int(str(cid)) >= 1_000_000
    except (ValueError, TypeError):
        return True

def _aug_kind(cid):
    s = str(cid)
    if s.isdigit():
        return "llm" if int(s) >= 1_000_000 else "real"
    return "buggy"

# AUG_SOURCE mirrors ARM so every downstream cell (metrics, graphs, arm table)
# keys cleanly on the arm name without further edits.
AUG_SOURCE = ARM

OUT_DIR     = Path("/kaggle/working");   OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR   = OUT_DIR / "cache";         CACHE_DIR.mkdir(parents=True, exist_ok=True)
_ARM        = ARM   # short alias for path suffixes
LOG_PATH    = OUT_DIR / f"dive_dedup2_{_ARM}_train.log"
HIST_CSV    = OUT_DIR / f"history_{_ARM}.csv"
HIST_JSON   = OUT_DIR / f"history_{_ARM}.json"
STATE_PATH  = OUT_DIR / f"last_state_{_ARM}.pt"
BEST_PATH   = OUT_DIR / f"best_model_{_ARM}.pt"
EMB_NPZ     = CACHE_DIR / f"dive_dedup2_units_{_ARM}.npz"
METRICS_ARM_JSON = OUT_DIR / f"metrics_arm_{_ARM}.json"

# -- Resume control -----------------------------------------------------------
RESUME_FROM = None    # set to str(STATE_PATH) to resume

'''

NEW_SANITY = r'''# -- Sanity check on data files -----------------------------------------------
assert TRAIN_CSV.exists(), f"Missing train CSV {TRAIN_CSV}"
assert VAL_CSV.exists() and TEST_CSV.exists(), f"Missing val/test CSV under {DATA_ROOT}"
assert SOURCE_DIRS[0].exists(), f"Missing source dir {SOURCE_DIRS[0]}"
print(f"ARM={ARM}  DATA_ROOT={DATA_ROOT}", flush=True)
print(f"  train={TRAIN_CSV}", flush=True)
print(f"  source dirs: {[str(d) for d in SOURCE_DIRS]}", flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
print(f"Stage1: MAX_UNITS={MAX_UNITS} UNIT_TOKENS={UNIT_TOKENS} model={GCB_MODEL}", flush=True)
print(f"Stage2: d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} d_ff={D_FF}", flush=True)
'''

set_src(5, NEW_HEAD + constants + NEW_SANITY)


# ---------------------------------------------------------------- cell 11
CELL11 = r'''def _load_fold_raw(name):
    d = pd.read_csv(_fold_csv(name))
    have = d["contractID"].apply(lambda c: sol_path(c).exists())
    cov = have.mean()
    log(f"  {name:>5s} (raw): {len(d):6d} rows | source coverage {cov:.2%}")
    if cov < 1.0:
        log(f"    dropping {int((~have).sum())} rows with no .sol")
    return d[have].reset_index(drop=True)

# val / test: FIXED across both arms (the real-only benchmark, byte-identical).
val_df  = _load_fold_raw("val")
test_df = _load_fold_raw("test")

# train: the arm's CSV IS the training set (no per-fold filtering needed) ------
#   dedup        -> 13,143 real rows
#   synth-dedup  -> 24,640 rows (13,143 real + 11,497 SolidiFI-injected synthetic)
train_df = _load_fold_raw("train")
train_df["_kind"] = train_df["contractID"].apply(_aug_kind)
_kc = train_df["_kind"].value_counts().to_dict()
log(f"ARM={ARM} | train composition: real={_kc.get('real',0)} "
    f"buggy={_kc.get('buggy',0)} llm={_kc.get('llm',0)}")
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

EXPECTED_TEST_ROWS = 3756
EXPECTED_TEST_SUPPORT = {
    "Reentrancy": 2005, "Access Control": 2882, "Arithmetic": 1685,
    "Unchecked Return Values": 1025, "DoS": 682, "Bad Randomness": 123,
    "Front Running": 115, "Time manipulation": 1186}
_sup = {c: int(test_df[c].sum()) for c in LABEL_COLS}
log(f"[gate] test rows={len(test_df)} (expect {EXPECTED_TEST_ROWS})")
log(f"[gate] test support={_sup}")
if len(test_df) != EXPECTED_TEST_ROWS or _sup != EXPECTED_TEST_SUPPORT:
    log("[gate] WARNING: this test set does NOT match the recorded dedup 3,756-row "
        "benchmark - check the dataset version.")
else:
    log("[gate] OK: test set == recorded 3,756-row dedup benchmark.")

# ----------------------------------------------------------------------------
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
log(f"ARM={ARM} | working set {df.shape} | "
    f"train={len(train_df)} (real={len(tr_real)}, aug={len(train_df)-len(tr_real)}) "
    f"val={len(val_df)} test={len(test_df)}")
log("Positives per class (train fold):")
for c in LABEL_COLS:
    log(f"  {c:>26s}  {int(train_df[c].sum()):>5d}")
'''
set_src(11, CELL11)


# ---------------------------------------------------------------- cell 43 titles
s43 = src(43).replace("dive-dedup-1 [{AUG_SOURCE}]", "dive-dedup-2 [{ARM}]")
set_src(43, s43)


# ---------------------------------------------------------------- cell 47 arm table
CELL47 = r'''import json as _json

rows = []
for _f in sorted(OUT_DIR.glob("metrics_arm_*.json")):
    m = _json.load(open(_f))
    arm = m.get("aug_source", _f.stem.replace("metrics_arm_", ""))
    rec = {"arm": arm,
           "n_train": m.get("n_train"), "n_aug": m.get("n_train_aug"),
           "macro_f1": round(m["test_calibrated_tuned"]["f1_macro"], 4),
           "micro_f1": round(m["test_calibrated_tuned"]["f1_micro"], 4)}
    for c in LABEL_COLS:
        rec[c] = round(m["per_class"][c]["f1"], 4)
    rows.append(rec)

if rows:
    _order = {"dedup": 0, "synth-dedup": 1}
    rows.sort(key=lambda r: _order.get(r["arm"], 9))
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_DIR / "comparison.csv", index=False)
    with open(OUT_DIR / "comparison.json", "w") as f:
        _json.dump(rows, f, indent=2)
    log("=== ARM COMPARISON (both arms on the SAME fixed real test set) ===")
    for line in cmp_df.to_string(index=False).splitlines():
        log("  " + line)
    if any(r["arm"] == "dedup" for r in rows):
        base = next(r for r in rows if r["arm"] == "dedup")
        for r in rows:
            if r["arm"] != "dedup":
                log(f"  delta macro-F1 ({r['arm']} - dedup) = {r['macro_f1'] - base['macro_f1']:+.4f}")
                for c in LABEL_COLS:
                    log(f"      d {c:>26s} = {r[c] - base[c]:+.4f}")
    log("NOTE: 'dedup' is the no-augmentation baseline; 'synth-dedup' adds the "
        "SolidiFI-injected synthetic train rows. Both score the one frozen real "
        "3,756-row test set, so the delta isolates the augmentation effect.")
else:
    log(f"Only the current arm ({ARM}) is present so far. Re-run with the other ARM "
        f"value in the same Kaggle session to populate the comparison table.")
'''
set_src(47, CELL47)


out = ROOT / "build_dive_dedup_2.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out)
print("cells:", len(cells))
