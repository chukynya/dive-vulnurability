"""Build build_dive_dedup_1.ipynb from dive-synthesized-11.ipynb.

Same model/pipeline; only the dataset changes (bytecode-deduplicated, 70/10/20
splits under Data-before-aug) plus added per-class F1 + PR-AUC recording/graphs
on val and test. Edits are applied to specific cells of the reference notebook.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "dive-synthesized-11.ipynb"
OUT_NB = ROOT / "build_dive_dedup_1.ipynb"

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def set_src(idx, text):
    cells[idx]["source"] = text.splitlines(keepends=True)
    if cells[idx]["cell_type"] == "code":
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None


def replace_in(idx, old, new):
    s = "".join(cells[idx]["source"])
    assert old in s, f"cell {idx}: pattern not found:\n{old}"
    set_src(idx, s.replace(old, new))


# ── Cell 0 — header ──────────────────────────────────────────────────────────
set_src(0, """# dive-dedup-1 — bytecode-deduplicated dataset, 70/10/20 splits

**What this is.** The same frozen-GraphCodeBERT + set-Transformer pipeline as
dive-synthesized-11, retrained on a **bytecode-deduplicated** corpus. Contracts
whose canonical runtime bytecode (Solidity CBOR metadata stripped) collided were
collapsed to one representative (labels OR-merged across the group), taking the
22,330 raw contracts down to **18,777** unique representatives. Those are split
**70 / 10 / 20** (train 13,143 · val 1,878 · test 3,756) with
`MultilabelStratifiedShuffleSplit`. The splits ship pre-built in the dataset, so
this notebook loads them as-is — no re-splitting, no augmentation.

**No augmentation.** This dataset is the pre-augmentation (`Data-before-aug`)
corpus, so `AUG_SOURCE` is fixed to `"none"` — train is real-only. The ablation
machinery from dive-synthesized-11 is kept dormant so the code stays a drop-in.

**Dataset (Kaggle).** `henrychristian7555/dive-dedup`, mounted at
`/kaggle/input/datasets/henrychristian7555/dive-dedup/Data-before-aug`:

| Path | Contents |
|---|---|
| `Data-before-aug/Source/` | `{contractID}.sol` for every representative |
| `Data-before-aug/splits/` | `Train_Labels.csv`, `Val_Labels.csv`, `Test_Labels.csv` |
| `Data-before-aug/Bytecode_filled.csv` | runtime bytecode (not used by this source-modality run) |

Runs on Kaggle **T4 × 2** (DataParallel over both GPUs).

## What stays identical to dive-synthesized-11

- Frozen GraphCodeBERT + function-level unit splitting (MAX_UNITS=32, UNIT_TOKENS=256).
- Set-Transformer aggregator: d_model=384, 6 heads, 3 layers, d_ff=1024.
- DROPOUT=0.30, DROP_PATH=0.20, WEIGHT_DECAY=0.10.
- AsymmetricLossWeighted (g-=4, g+=1, clip=0.05) + 0.3*BCE aux head on (Bad Randomness, Front Running).
- EMA decay 0.999, AdamW with selective weight decay, cosine warmup, grad clip 1.0.
- WeightedRandomSampler (sqrt-inv-freq per-sample weights, clipped [1,5]).
- Per-class isotonic calibration on val -> 41-pt threshold tune (min-precision 0.25 for support < 200).
- Atomic full-state checkpointing, tee logger, history.csv.

## Added metrics & graphs

- **Per-class F1 and PR-AUC (average precision)** recorded **every epoch on val**
  (flattened into `history.csv`/`history.json` as `val_f1_<class>` / `val_ap_<class>`).
- Best-epoch **val per-class** block + **test per-class** block (both with F1 + PR-AUC)
  written to `metrics.json` and `per_class_val_test.csv`.
- New graph **per-class PR-AUC: train vs val vs test** alongside the existing
  per-class F1 / overfit-gap / loss / macro-F1 curves.
""")

# ── Cell 4 — config markdown: cache filename ────────────────────────────────
replace_in(4, "dive_synth11_units.npz", "dive_dedup1_units.npz")

# ── Cell 5 — paths/config: single-Source dedup layout, AUG_SOURCE='none' ────
set_src(5, '''# -- Paths --------------------------------------------------------------------
# Bytecode-deduplicated dataset, pre-built 70/10/20 splits. Single Source/ dir
# (no synthetic, no augmentation), so resolution is a flat contractID -> .sol map.
def _find_root(base):
    """Find dataset root: the dir that contains a 'splits' subdir with CSVs."""
    base = Path(base)
    if not base.exists():
        return None
    if (base / "splits").is_dir() and any((base / "splits").glob("*.csv")):
        return base
    for d in sorted(base.rglob("*")):          # sorted = deterministic
        if d.is_dir() and d.name == "splits" and any(d.glob("*.csv")):
            return d.parent
    return None

_DS_BASE = Path("/kaggle/input/datasets/henrychristian7555/dive-dedup/Data-before-aug")
DATA_ROOT = (_find_root(_DS_BASE)
             or _find_root("/kaggle/input/dive-dedup/Data-before-aug")
             or _find_root("/kaggle/input/datasets/henrychristian7555/dive-dedup")
             or _find_root("/kaggle/input/dive-dedup"))
if DATA_ROOT is None:
    if _DS_BASE.exists():
        print("Dataset base found. Contents:", flush=True)
        for _p in sorted(_DS_BASE.rglob("*")):
            if _p.is_dir() or _p.suffix == ".csv":
                print(f"  {_p.relative_to(_DS_BASE)}", flush=True)
    import kagglehub
    DATA_ROOT = _find_root(Path(kagglehub.dataset_download("henrychristian7555/dive-dedup")) / "Data-before-aug") \\
        or _find_root(kagglehub.dataset_download("henrychristian7555/dive-dedup"))
assert DATA_ROOT is not None, (
    f"dive-dedup dataset not found — could not locate a 'splits/' directory "
    f"containing CSV files under {_DS_BASE}")

SPLIT_DIR = DATA_ROOT / "splits"
SOURCE_DIR = DATA_ROOT / "Source"

# sol_path: resolve contractID -> .sol file path (flat Source/ dir, per-run cache)
_sol_cache = {}

def sol_path(cid):
    cid_s = str(cid)
    p = _sol_cache.get(cid_s)
    if p is None:
        p = SOURCE_DIR / f"{cid_s}.sol"
        _sol_cache[cid_s] = p
    return p

def _fold_csv(name):
    """Return path to the split CSV for fold 'train', 'val', or 'test'."""
    p = SPLIT_DIR / f"{name.capitalize()}_Labels.csv"
    if not p.exists():
        raise FileNotFoundError(f"No split CSV for fold '{name}' — tried: {p}")
    return p

# Everything in this dataset is a real Etherscan contract (no synthetic IDs); the
# helpers below keep the dive-synthesized-11 verification gates working unchanged.
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

# === ABLATION KNOB ===========================================================
# Fixed to "none": this is the pre-augmentation corpus, so train is real-only.
AUG_SOURCE = "none"
assert AUG_SOURCE in ("none", "buggy", "llm"), f"bad AUG_SOURCE {AUG_SOURCE!r}"
LLM_SRC_DIRS = []            # dormant (no synthetic in this dataset)
LLM_LABELS_CANDIDATES = []   # dormant

OUT_DIR     = Path("/kaggle/working");   OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR   = OUT_DIR / "cache";         CACHE_DIR.mkdir(parents=True, exist_ok=True)
_ARM        = AUG_SOURCE   # short alias for path suffixes
LOG_PATH    = OUT_DIR / f"dive_dedup1_{_ARM}_train.log"
HIST_CSV    = OUT_DIR / f"history_{_ARM}.csv"
HIST_JSON   = OUT_DIR / f"history_{_ARM}.json"
STATE_PATH  = OUT_DIR / f"last_state_{_ARM}.pt"
BEST_PATH   = OUT_DIR / f"best_model_{_ARM}.pt"
EMB_NPZ     = CACHE_DIR / f"dive_dedup1_units_{_ARM}.npz"
METRICS_ARM_JSON = OUT_DIR / f"metrics_arm_{_ARM}.json"

# -- Resume control -----------------------------------------------------------
RESUME_FROM = None    # set to str(STATE_PATH) to resume

# -- Labels -------------------------------------------------------------------
LABEL_COLS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
              "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
N_LABELS = len(LABEL_COLS)
BR_IDX = LABEL_COLS.index("Bad Randomness")     # 5
FR_IDX = LABEL_COLS.index("Front Running")      # 6

# -- Stage 1: function-level units + frozen GraphCodeBERT ---------------------
GCB_MODEL    = "microsoft/graphcodebert-base"
EMB_DIM      = 768          # graphcodebert-base hidden size
MAX_UNITS    = 32           # function/header units per contract (risk-ranked cap)
UNIT_TOKENS  = 256          # GraphCodeBERT token cap per unit (<=512)
UNIT_CHAR_CAP = 2000        # pre-truncate unit text before tokenisation
HEADER_BASE_SCORE = 1000.0  # contract-header units outrank function units in the cap
GCB_BATCH    = 64           # units per GraphCodeBERT forward
GCB_FLUSH    = 2048         # units buffered before an encode flush (bounds RAM)

# -- Stage 2: set-Transformer aggregator --------------------------------------
D_MODEL   = 384
N_HEADS   = 6
N_LAYERS  = 3
D_FF      = 1024
DROPOUT   = 0.30
DROP_PATH = 0.20

# -- Training -----------------------------------------------------------------
BATCH_SIZE   = 128
EPOCHS       = 40
LR           = 3e-4
WEIGHT_DECAY = 0.10
GRAD_CLIP    = 1.0
WARMUP_RATIO = 0.10
PATIENCE     = 12
NUM_WORKERS  = 2
PIN_MEMORY   = True

# -- Asymmetric Loss + aux BCE ------------------------------------------------
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP      = 0.05
PER_CLASS_W_MIN = 1.0
PER_CLASS_W_MAX = 5.0
AUX_BCE_INDICES = [BR_IDX, FR_IDX]
AUX_BCE_WEIGHT  = 0.3

# -- EMA ----------------------------------------------------------------------
EMA_DECAY = 0.999

# -- Augmentation (unit-level) ------------------------------------------------
AUG_UNIT_DROP_PROB = 0.10   # per real unit, drop (mask out) at train time
AUG_CUTMIX_PROB    = 0.30   # batch-level unit-slot CutMix
AUG_CUTMIX_BETA    = 0.5

# -- Sampler ------------------------------------------------------------------
SAMPLER_W_MIN = 1.0
SAMPLER_W_MAX = 5.0

# -- Checkpointing ------------------------------------------------------------
CKPT_WALLCLOCK_SECS = 30 * 60

# -- Verification gates -------------------------------------------------------
MIN_SOURCE_COVERAGE = 0.99        # >=99% of label rows must have a .sol file
MIN_UNIT_COVERAGE   = 0.99        # >=99% of contracts must yield >=1 unit
STAGE1_BUDGET_SECS  = 4 * 3600    # abort if estimated stage-1 encode exceeds this

# -- Sanity check on data files -----------------------------------------------
assert SPLIT_DIR.exists(), f"Missing splits dir {SPLIT_DIR}"
assert SOURCE_DIR.exists(), f"Missing source dir {SOURCE_DIR}"
print(f"Layout: dedup (flat Source/)  DATA_ROOT={DATA_ROOT}", flush=True)
print("Inputs OK:", DATA_ROOT, flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
print(f"Stage1: MAX_UNITS={MAX_UNITS} UNIT_TOKENS={UNIT_TOKENS} model={GCB_MODEL}", flush=True)
print(f"Stage2: d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} d_ff={D_FF}", flush=True)
''')

# ── Cell 6 — tee logger markdown: log filename ──────────────────────────────
replace_in(6, "dive_synth11_train.log", "dive_dedup1_train.log")

# ── Cell 11 — verification gate expectations for the dedup test split ───────
replace_in(
    11,
    """EXPECTED_TEST_ROWS = 4466
EXPECTED_TEST_SUPPORT = {
    "Reentrancy": 2280, "Access Control": 3345, "Arithmetic": 1909,
    "Unchecked Return Values": 1182, "DoS": 756, "Bad Randomness": 127,
    "Front Running": 121, "Time manipulation": 1265}""",
    """EXPECTED_TEST_ROWS = 3756
EXPECTED_TEST_SUPPORT = {
    "Reentrancy": 2005, "Access Control": 2882, "Arithmetic": 1685,
    "Unchecked Return Values": 1025, "DoS": 682, "Bad Randomness": 123,
    "Front Running": 115, "Time manipulation": 1186}""",
)
replace_in(
    11,
    """    log("[gate] WARNING: this test set does NOT match the recorded dataset-generation "
        "4,466-row benchmark. Arms remain comparable to EACH OTHER (same test this session) "
        "but NOT to the recorded dive-9/dive-10 numbers.")
else:
    log("[gate] OK: test set == recorded 4,466-row real benchmark.")""",
    """    log("[gate] WARNING: this test set does NOT match the recorded dedup 3,756-row "
        "benchmark — check the dataset version.")
else:
    log("[gate] OK: test set == recorded 3,756-row dedup benchmark.")""",
)

# ── Cell 39 — training loop: record per-class val F1 + PR-AUC each epoch ────
replace_in(
    39,
    "best_calibrators = None",
    "best_calibrators = None\nbest_val_per_class = None",
)
replace_in(
    39,
    """        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_loss_eval": train_loss_eval, "lr": cur_lr,
            "train_dt": train_dt, **val_metrics,
            "val_f1_macro_raw_05": val_metrics_raw_05["f1_macro"],
            "train_f1_macro_raw_05": train_metrics_raw_05["f1_macro"],
            "overfit_gap_f1_macro_raw_05": overfit_gap,
        })""",
    """        # per-class val F1 + PR-AUC (average precision), flattened for history.csv
        val_pc_flat = {}
        for _lab in LABEL_COLS:
            val_pc_flat[f"val_f1_{_lab}"] = val_per_class_metrics[_lab]["f1"]
            val_pc_flat[f"val_ap_{_lab}"] = val_per_class_metrics[_lab]["ap"]
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_loss_eval": train_loss_eval, "lr": cur_lr,
            "train_dt": train_dt, **val_metrics,
            "val_f1_macro_raw_05": val_metrics_raw_05["f1_macro"],
            "train_f1_macro_raw_05": train_metrics_raw_05["f1_macro"],
            "overfit_gap_f1_macro_raw_05": overfit_gap,
            **val_pc_flat,
        })""",
)
replace_in(
    39,
    """            best_thresholds = val_thr.copy()
            best_calibrators = calibrators""",
    """            best_thresholds = val_thr.copy()
            best_calibrators = calibrators
            best_val_per_class = val_per_class_metrics""",
)

# ── Cell 41 — test eval: add val per-class block + per_class_val_test.csv ───
replace_in(
    41,
    """pc_df = pd.DataFrame(per_class_tuned).T.round(4)
log("=== PER-CLASS (calibrated + tuned thresholds) ===")
for line in pc_df.to_string().splitlines():
    log("  " + line)""",
    """pc_df = pd.DataFrame(per_class_tuned).T.round(4)
log("=== PER-CLASS TEST (calibrated + tuned thresholds) ===")
for line in pc_df.to_string().splitlines():
    log("  " + line)

# Per-class F1 + PR-AUC on val (best epoch) and test, side by side
_val_pc = best_val_per_class if best_val_per_class is not None else {}
pc_val_test = pd.DataFrame({
    "class":   LABEL_COLS,
    "support_val":  [_val_pc.get(c, {}).get("support", float("nan")) for c in LABEL_COLS],
    "f1_val":       [_val_pc.get(c, {}).get("f1", float("nan"))      for c in LABEL_COLS],
    "ap_val":       [_val_pc.get(c, {}).get("ap", float("nan"))      for c in LABEL_COLS],
    "support_test": [per_class_tuned[c]["support"] for c in LABEL_COLS],
    "f1_test":      [per_class_tuned[c]["f1"]       for c in LABEL_COLS],
    "ap_test":      [per_class_tuned[c]["ap"]       for c in LABEL_COLS],
})
pc_val_test.to_csv(OUT_DIR / "per_class_val_test.csv", index=False)
log("=== PER-CLASS F1 + PR-AUC (val @ best epoch | test) ===")
for line in pc_val_test.round(4).to_string(index=False).splitlines():
    log("  " + line)""",
)
replace_in(
    41,
    """        "per_class":              per_class_tuned,
        "best_val_f1_macro":      best_macro_f1,""",
    """        "per_class":              per_class_tuned,
        "per_class_test":         per_class_tuned,
        "per_class_val":          (best_val_per_class or {}),
        "best_val_f1_macro":      best_macro_f1,""",
)

# ── Cell 42 — graphics markdown: describe the added PR-AUC figure ───────────
set_src(42, """## 20b — Metrics graphics: loss curves, F1 curves, per-class F1, per-class PR-AUC, overfit gap

Five figures saved to `/kaggle/working/` and shown inline:
1. **Loss curves** — `train_loss` + `val_loss` per epoch, `test_loss` as a dashed line.
2. **Macro-F1 curves** — train/val (raw@0.5) + val (calib+thr), `test` as a dashed line.
3. **Per-class F1** — train vs val vs test (calibrated + frozen thresholds).
4. **Per-class PR-AUC** (average precision) — train vs val vs test.
5. **Per-class overfit gap** — `train_f1 − test_f1` with OK / borderline / overfit bands.
""")

# ── Cell 43 — graphics: rename titles, add per-class PR-AUC graph + CSV ─────
replace_in(43, "dive-synthesized-11 [{AUG_SOURCE}]", "dive-dedup-1 [{AUG_SOURCE}]")
# Replace the per-class F1 helper to also capture PR-AUC, then add Graph 5.
replace_in(
    43,
    """    # ── Graph 3: Per-class F1 bars (train / val / test) ───────────────────────
    def _fold_f1(loader):
        lg, lb, _ = predict_with_loss(eval_model, loader)
        pr = apply_calibration(1 / (1 + np.exp(-lg)), frozen_calibrators)
        _, pc = multilabel_metrics(lb, pr, frozen_thresholds)
        return [pc[c]["f1"] for c in LABEL_COLS]

    f1_train = _fold_f1(train_eval_loader)
    f1_val   = _fold_f1(val_loader)
    f1_test  = [per_class_tuned[c]["f1"] for c in LABEL_COLS]""",
    """    # ── Graph 3: Per-class F1 bars (train / val / test) ───────────────────────
    def _fold_pc(loader):
        lg, lb, _ = predict_with_loss(eval_model, loader)
        pr = apply_calibration(1 / (1 + np.exp(-lg)), frozen_calibrators)
        _, pc = multilabel_metrics(lb, pr, frozen_thresholds)
        return pc

    pc_train = _fold_pc(train_eval_loader)
    pc_val   = _fold_pc(val_loader)
    f1_train = [pc_train[c]["f1"] for c in LABEL_COLS]
    f1_val   = [pc_val[c]["f1"]   for c in LABEL_COLS]
    f1_test  = [per_class_tuned[c]["f1"] for c in LABEL_COLS]
    ap_train = [pc_train[c]["ap"] for c in LABEL_COLS]
    ap_val   = [pc_val[c]["ap"]   for c in LABEL_COLS]
    ap_test  = [per_class_tuned[c]["ap"] for c in LABEL_COLS]""",
)
# Insert the PR-AUC graph (Graph 5) right after Graph 4, before the text summary.
replace_in(
    43,
    """    # ── Text summary ─────────────────────────────────────────────────────────
    pd.DataFrame({"class": LABEL_COLS, "f1_train": f1_train, "f1_val": f1_val,
                  "f1_test": f1_test, "gap_train_test": gaps}).to_csv(
        OUT_DIR / "f1_train_val_test.csv", index=False)""",
    """    # ── Graph 5: Per-class PR-AUC (average precision) bars (train/val/test) ────
    x = np.arange(N_LABELS); bw = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    bars_tr = ax.bar(x - bw, ap_train, bw, label=f"train (macro {np.mean(ap_train):.3f})")
    bars_v  = ax.bar(x,      ap_val,   bw, label=f"val   (macro {np.mean(ap_val):.3f})")
    bars_te = ax.bar(x + bw, ap_test,  bw, label=f"test  (macro {np.mean(ap_test):.3f})")
    for bars in (bars_tr, bars_v, bars_te):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(LABEL_COLS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC (average precision)"); ax.set_ylim(0, 1.12)
    ax.set_title(f"dive-dedup-1 [{AUG_SOURCE}] —per-class PR-AUC: train vs val vs test")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_prauc_train_val_test.png", dpi=120); plt.show(); plt.close(fig)

    # ── Text summary ─────────────────────────────────────────────────────────
    pd.DataFrame({"class": LABEL_COLS,
                  "f1_train": f1_train, "f1_val": f1_val, "f1_test": f1_test,
                  "ap_train": ap_train, "ap_val": ap_val, "ap_test": ap_test,
                  "gap_train_test": gaps}).to_csv(
        OUT_DIR / "f1_prauc_train_val_test.csv", index=False)""",
)
replace_in(
    43,
    '''    log("Saved: loss_curves.png, f1_macro_curves.png, per_class_f1_train_val_test.png, per_class_overfit_gap.png")
    log(f"macro-F1 (calib+thr)  train={np.mean(f1_train):.4f}  val={np.mean(f1_val):.4f}  "
        f"test={np.mean(f1_test):.4f}  (train-test gap={np.mean(f1_train)-np.mean(f1_test):+.4f})")''',
    '''    log("Saved: loss_curves.png, f1_macro_curves.png, per_class_f1_train_val_test.png, "
        "per_class_prauc_train_val_test.png, per_class_overfit_gap.png")
    log(f"macro-F1 (calib+thr)  train={np.mean(f1_train):.4f}  val={np.mean(f1_val):.4f}  "
        f"test={np.mean(f1_test):.4f}  (train-test gap={np.mean(f1_train)-np.mean(f1_test):+.4f})")
    log(f"macro-PR-AUC          train={np.mean(ap_train):.4f}  val={np.mean(ap_val):.4f}  "
        f"test={np.mean(ap_test):.4f}")''',
)

# ── Cell 48 — summary markdown: filenames ───────────────────────────────────
replace_in(48, "cache/dive_synth11_units.npz", "cache/dive_dedup1_units.npz")
replace_in(48, "history.csv`, `dive_synth11_train.log`", "history.csv`, `dive_dedup1_train.log`")
replace_in(48, "cached once to `dive_synth11_units.npz`", "cached once to `dive_dedup1_units.npz`")

# Update Kaggle dataSources metadata note (cleared so Kaggle re-attaches on upload)
nb.setdefault("metadata", {}).setdefault("kaggle", {})["dataSources"] = []

OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT_NB}  ({len(cells)} cells)")
