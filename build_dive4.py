"""Build dive-4.ipynb: dive-3 + hierarchical chunked encoding + class-balanced
sampling + CutMix + span masking + isotonic calibration.

Stays bytecode-only. The encoder architecture is identical to dive-3; the deltas
are:

1. **Hierarchical encoding** — each contract is split into N_CHUNKS=4 windows of
   1024 opcodes (MAX_TOTAL_OPS=4096). The same encoder runs on each chunk; chunk
   embeddings are attention-pooled into a single contract vector. Attacks the
   94 % truncation rate observed in dive-3 directly.
2. **Class-balanced WeightedRandomSampler** — rare-class positives (Bad
   Randomness, Front Running, DoS) are oversampled.
3. **Chunk-level CutMix** — sequence-aware MixUp: combine chunks from two
   contracts with a soft label, so the model sees realistic interpolations of
   positives.
4. **Span masking** + **chunk dropout** replace dive-3's per-token mask/swap.
5. **Per-class loss weights** on top of AsymmetricLoss (sqrt-inverse-freq,
   clipped to [1, 5]).
6. **Isotonic calibration per class** on val probabilities, then threshold
   tuning on the calibrated probs. Fixes the AUC ≫ F1 gap on BR / FR / DoS.

What stays identical to dive-3:
- Encoder (CNN-Transformer, d=256, 6 layers, ~5.5 M params).
- AsymmetricLoss, EMA, stochastic depth, AdamW + cosine warmup.
- Multilabel-stratified 80/10/10 split, fixed seed 42.
- Full-state checkpointing + tee logger + atomic writes.
"""
import json
from pathlib import Path

cells = []


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    })


# ---------------------------------------------------------------------------
md("""# DIVE-4 — Chunked bytecode + better data pipeline

**Why this notebook.** dive-3 plateaued at test macro-F1 = 0.6636. The training
log shows the bottleneck plainly:

```
Truncated at MAX_OPS=1024: 94.39 %
```

94 % of contracts were chopped. No amount of loss / regularisation tuning will
help past that. The per-class report says the same thing from the other side:
rare-class AUCs are high (BR 0.88, FR 0.91, DoS 0.87) but F1s are low (0.43,
0.39, 0.58) — the encoder can *rank* but can't *calibrate* on tiny supports.

dive-4 keeps the encoder architecture from dive-3 byte-for-byte and only
changes the **input pipeline**, **sampling**, and **calibration**.

## What's new vs dive-3

| Change | Why |
|---|---|
| **Hierarchical chunked encoding** (`N_CHUNKS=4 × CHUNK_SIZE=1024`, no overlap, `MAX_TOTAL_OPS=4096`) | Each contract is split into up to 4 windows of 1024 opcodes. The same encoder runs on each chunk; chunk embeddings are attention-pooled into one contract vector. Increases context 4× without quadratic attention cost. |
| **Class-balanced WeightedRandomSampler** | Per-sample weight = max over its positive labels of `sqrt(N_total / class_count)`, clipped to `[1, 5]`. Rare-class positives get sampled more often without breaking multilabel structure. |
| **Chunk-level CutMix** | Sample two contracts (i, j) with `λ ~ Beta(0.5, 0.5)`; take `⌈λ·N_CHUNKS⌉` chunks from i, the rest from j; mix labels by `λ`. Native MixUp for sequences. |
| **Span masking** (span 1-3, 8 % of non-special positions) | Replaces dive-3's per-token mask. Better signal than independent drops. |
| **Chunk dropout** (5 %) | Zero out one whole chunk per sample during training. Robust to partial-view inputs. |
| **Per-class loss weights** | sqrt-inverse-frequency, clipped to `[1, 5]`. Applied as an outer multiplier on ASL. More gradient signal where it's needed. |
| **Isotonic calibration per class** | Fit `IsotonicRegression` on val probs per class, apply to test probs before threshold tuning. Directly addresses the AUC ≫ F1 gap on BR / FR / DoS. |
| **Finer threshold grid + min-precision floor** | 41 points in [0.05, 0.95]. For rare classes (val support < 200), require `P ≥ 0.25` to reject degenerate "predict all positive" thresholds. |

## What stays identical to dive-3

- Encoder: CNN-Transformer, d=256, 6 layers, 8 heads, FFN 1024, ~5.5 M params.
- AsymmetricLoss (`γ⁻=4, γ⁺=1, clip=0.05`).
- EMA (decay 0.999), DropPath (linear 0 → 0.1), AdamW + cosine warmup.
- Multilabel-stratified 80/10/10 split, seed 42.
- Full-state checkpointing (atomic), tee logger, history flush per epoch.
- Test eval on EMA + frozen tuned thresholds (now applied to **calibrated** probs).

## Reading the result

- Compare `test_metrics["f1_macro"]` to dive-3's **0.6636**. If chunking carries
  signal that dive-3 was throwing away, expect **+0.02 to +0.05** on macro-F1.
- Compare per-class F1 for **Bad Randomness, Front Running, DoS**. These should
  move most: chunking exposes more context, sampler pushes more positives
  through training, calibration narrows the AUC↔F1 gap.
- If F1 doesn't move and AUC barely changes either, the dataset is signal-bound
  on bytecode alone — the next move is adding source code (`dive-5`).
""")

# ---------------------------------------------------------------------------
md("""## 1 — Environment""")

code("""import os, gc, json, math, time, random, warnings, signal, sys, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Torch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {n_gpus}", flush=True)
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB", flush=True)
""")

code("""try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "iterative-stratification"])
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
print("iterstrat ready", flush=True)
""")

# ---------------------------------------------------------------------------
md("""## 2 — Config

Edit **only** this cell to switch between fresh run and resume.

- `RESUME_FROM = None` → fresh training, fresh checkpoint at `last_state.pt`.
- `RESUME_FROM = "/kaggle/working/last_state.pt"` → continue from the last
  saved state. Optimizer / scheduler / scaler / RNG / EMA / epoch / history /
  best-metric are all restored.
""")

code("""# ── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/kaggle/input/datasets/henrychristian7555/dive-smart-contract-multi-class-vulnerability")
BYTECODE_CSV = DATA_ROOT / "Bytecode_filled.csv"
LABEL_CSV    = DATA_ROOT / "DIVE_Labels.csv"

OUT_DIR   = Path("/kaggle/working");      OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache";            CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH  = OUT_DIR / "dive4_train.log"
HIST_CSV  = OUT_DIR / "history.csv"
HIST_JSON = OUT_DIR / "history.json"
STATE_PATH = OUT_DIR / "last_state.pt"      # full resumable state
BEST_PATH  = OUT_DIR / "best_model.pt"      # EMA weights + thresholds + calibrators

# ── Resume control ──────────────────────────────────────────────────────────
RESUME_FROM = None    # set to str(STATE_PATH) to resume

# ── Labels ──────────────────────────────────────────────────────────────────
LABEL_COLS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
              "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
N_LABELS = len(LABEL_COLS)

# ── Tokenisation ────────────────────────────────────────────────────────────
PAD_ID, CLS_ID, UNK_ID = 0, 1, 2
SPECIAL = 3
VOCAB_SIZE = 256 + SPECIAL     # 259

# Hierarchical context: N_CHUNKS windows of CHUNK_SIZE opcodes, no overlap.
CHUNK_SIZE      = 1024
N_CHUNKS        = 4
MAX_TOTAL_OPS   = CHUNK_SIZE * N_CHUNKS    # 4096

# ── Model (same as dive-3) ──────────────────────────────────────────────────
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 6
D_FF     = 1024
DROPOUT  = 0.15
DROP_PATH = 0.10

# ── Training ────────────────────────────────────────────────────────────────
BATCH_PER_GPU = 16          # ↓ from 32 — each sample now runs 4 chunks through the encoder
BATCH_SIZE    = BATCH_PER_GPU * max(1, n_gpus)
EPOCHS        = 30
LR            = 2e-4
WEIGHT_DECAY  = 0.05
GRAD_CLIP     = 1.0
WARMUP_RATIO  = 0.10
PATIENCE      = 10
NUM_WORKERS   = 2
PIN_MEMORY    = True

# ── Asymmetric Loss hyper-params (Ridnik et al. 2021) ──────────────────────
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP      = 0.05

# Per-class loss weights are computed from train-set frequencies at runtime
# (sqrt-inverse-freq, clipped to [PER_CLASS_W_MIN, PER_CLASS_W_MAX]).
PER_CLASS_W_MIN = 1.0
PER_CLASS_W_MAX = 5.0

# ── EMA ─────────────────────────────────────────────────────────────────────
EMA_DECAY = 0.999

# ── Augmentation ────────────────────────────────────────────────────────────
AUG_SPAN_MASK_PROB = 0.08    # per non-special position, prob of starting a masked span
AUG_SPAN_MAX_LEN   = 3       # spans are uniform in [1, AUG_SPAN_MAX_LEN]
AUG_CHUNK_DROP_PROB = 0.05   # per sample, prob of zeroing one random non-empty chunk
AUG_CUTMIX_PROB     = 0.30   # per batch, prob of applying chunk-level CutMix
AUG_CUTMIX_BETA     = 0.5    # Beta(alpha, alpha) for the mixing fraction

# ── Sampler ─────────────────────────────────────────────────────────────────
SAMPLER_W_MIN = 1.0
SAMPLER_W_MAX = 5.0

# ── Checkpointing ───────────────────────────────────────────────────────────
CKPT_WALLCLOCK_SECS = 30 * 60

# ── Sanity check on data files ──────────────────────────────────────────────
assert BYTECODE_CSV.exists() and LABEL_CSV.exists(), \\
    f"Missing data files under {DATA_ROOT}"
print("Inputs OK:", BYTECODE_CSV.name, "|", LABEL_CSV.name, flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
print(f"Context: {N_CHUNKS} chunks × {CHUNK_SIZE} = {MAX_TOTAL_OPS} opcodes (4× dive-3)", flush=True)
""")

# ---------------------------------------------------------------------------
md("""## 3 — `tee` logger

Identical to dive-3. Mirrors stdout to `dive4_train.log`. Survives Kaggle "Save
Version" and is reopened in append mode so it persists across resumes.
""")

code("""class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", buffering=1, encoding="utf-8")
        self.fh.write(f"\\n===== run started {datetime.utcnow().isoformat()}Z =====\\n")
        self.fh.flush()

    def __call__(self, msg: str = ""):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\\n"); self.fh.flush()

    def close(self):
        try: self.fh.close()
        except Exception: pass

log = TeeLogger(LOG_PATH)
log(f"Logger writing to {LOG_PATH}")
log(f"GPUs available: {n_gpus}")
""")

# ---------------------------------------------------------------------------
md("""## 4 — Load labels and bytecode""")

code("""labels_df = pd.read_csv(LABEL_CSV)
log(f"Labels shape: {labels_df.shape}")
log("Positives per class:")
for c in LABEL_COLS:
    log(f"  {c:>26s}  {int(labels_df[c].sum()):>5d}")

bc_df = pd.read_csv(BYTECODE_CSV, usecols=["contractID", "bytecode"])
log(f"Bytecode rows: {bc_df.shape}")
bc_df = bc_df.dropna(subset=["bytecode"])
bc_df = bc_df[bc_df["bytecode"].str.len() > 4]
log(f"After dropping empty bytecode: {bc_df.shape}")

df = labels_df.merge(bc_df, on="contractID", how="inner").reset_index(drop=True)
log(f"Joined: {df.shape}")
""")

# ---------------------------------------------------------------------------
md("""## 5 — EVM opcode disassembly @ MAX_TOTAL_OPS=4096 (cached)

Same disassembly rule as dive-3 (skip `PUSH1..PUSH32` immediates so we tokenise
control flow only), but the cap is raised from 1024 to **4096** so chunked
encoding has 4× more to work with. New cache file: `dive4_tokens.npz`.

The log line at the end is what tells you whether the 94 % truncation problem
is actually solved.
""")

code("""def disassemble(bc_hex: str, max_ops: int = MAX_TOTAL_OPS):
    if not isinstance(bc_hex, str):
        return [CLS_ID]
    s = bc_hex.strip().lower()
    if s.startswith("0x"): s = s[2:]
    if len(s) < 2 or (len(s) & 1):
        return [CLS_ID]
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return [CLS_ID]
    toks = [CLS_ID]
    i, n = 0, len(b)
    while i < n and len(toks) < max_ops:
        op = b[i]
        toks.append(op + SPECIAL)
        if 0x60 <= op <= 0x7f:          # PUSH1..PUSH32 → skip immediate
            i += 1 + (op - 0x5f)
        else:
            i += 1
    return toks

cache_path = CACHE_DIR / "dive4_tokens.npz"
if cache_path.exists():
    z = np.load(cache_path, allow_pickle=True)
    token_lists = list(z["tokens"])
    log(f"Loaded cached tokens for {len(token_lists)} contracts (max_ops={MAX_TOTAL_OPS})")
else:
    t0 = time.time()
    token_lists = [disassemble(bc, MAX_TOTAL_OPS) for bc in df["bytecode"].values]
    log(f"Disassembled {len(token_lists)} contracts in {time.time()-t0:.1f}s")
    np.savez_compressed(cache_path, tokens=np.array(token_lists, dtype=object))

lens = np.array([len(t) for t in token_lists])
log(f"Opcode-seq length: median={int(np.median(lens))}, p50={int(np.percentile(lens,50))}, "
    f"p75={int(np.percentile(lens,75))}, p90={int(np.percentile(lens,90))}, "
    f"p99={int(np.percentile(lens,99))}, max={int(lens.max())}")
log(f"Truncated at MAX_TOTAL_OPS={MAX_TOTAL_OPS}: {(lens>=MAX_TOTAL_OPS).mean():.2%}")
log(f"  (dive-3 had 94.39 % truncated at 1024 — that's the bar to clear)")
# Also report what the dive-3 ceiling looked like, for comparison.
log(f"Coverage @ 1024 cutoff: {(lens<=1024).mean():.2%} would have fit in dive-3")
log(f"Coverage @ 2048 cutoff: {(lens<=2048).mean():.2%}")
log(f"Coverage @ 4096 cutoff: {(lens<=4096).mean():.2%}")
""")

# ---------------------------------------------------------------------------
md("""## 6 — Stratified 80/10/10 split

Identical to dive-3 — same seed, same split sizes.
""")

code("""Y = df[LABEL_COLS].values.astype(np.float32)

mss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
idx_trv, idx_te = next(mss1.split(np.zeros(len(df)), Y))
mss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
idx_tr_local, idx_v_local = next(mss2.split(np.zeros(len(idx_trv)), Y[idx_trv]))
idx_tr, idx_v = idx_trv[idx_tr_local], idx_trv[idx_v_local]

log(f"Train {len(idx_tr)} | Val {len(idx_v)} | Test {len(idx_te)}")
for i, lab in enumerate(LABEL_COLS):
    log(f"  {lab:>26s}  tr={int(Y[idx_tr,i].sum()):>5d}  "
        f"v={int(Y[idx_v,i].sum()):>4d}  te={int(Y[idx_te,i].sum()):>4d}")
""")

# ---------------------------------------------------------------------------
md("""## 7 — Padded chunked tensors + augmentation

Each contract is stored as a `(N_CHUNKS, CHUNK_SIZE)` int32 tensor plus a token
mask and a per-chunk validity mask (`chunk_mask[c] = True` if chunk `c` has any
real tokens).

**Augmentation applied on-the-fly in `__getitem__`** (train only):

- **Span masking** (`AUG_SPAN_MASK_PROB` per non-special position): start a span
  of 1–`AUG_SPAN_MAX_LEN` tokens at this position, replace them with `PAD_ID`.
  CLS at position 0 is never masked. Breaks exact n-gram memorisation while
  preserving longer-range basic-block structure.
- **Chunk dropout** (`AUG_CHUNK_DROP_PROB` per sample): zero out one random
  non-empty chunk. Forces the chunk-aggregator to be robust to missing context.

CutMix runs at the **batch** level (mixing two samples), so it lives in the
collate function below, not here.
""")

code("""def to_chunked(tok_list, n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE):
    \"\"\"Return (X, tok_mask, chunk_mask) for a list of token lists.

    X:           (N, n_chunks, chunk_size)        int32
    tok_mask:    (N, n_chunks, chunk_size)        bool   — True where token is real
    chunk_mask:  (N, n_chunks)                    bool   — True if chunk has ≥1 token
    \"\"\"
    n = len(tok_list)
    X  = np.zeros((n, n_chunks, chunk_size), dtype=np.int32)
    TM = np.zeros((n, n_chunks, chunk_size), dtype=np.bool_)
    CM = np.zeros((n, n_chunks), dtype=np.bool_)
    total = n_chunks * chunk_size
    for i, t in enumerate(tok_list):
        L = min(len(t), total)
        for c in range(n_chunks):
            s, e = c * chunk_size, min((c + 1) * chunk_size, L)
            if s >= L:
                break
            X[i, c, : e - s] = t[s:e]
            TM[i, c, : e - s] = True
            CM[i, c] = True
    return X, TM, CM


t0 = time.time()
X_tr, TM_tr, CM_tr = to_chunked([token_lists[i] for i in idx_tr])
X_v,  TM_v,  CM_v  = to_chunked([token_lists[i] for i in idx_v])
X_te, TM_te, CM_te = to_chunked([token_lists[i] for i in idx_te])
Y_tr, Y_v, Y_te = Y[idx_tr], Y[idx_v], Y[idx_te]
log(f"Chunked padding in {time.time()-t0:.1f}s | X_tr {X_tr.shape} {X_tr.dtype}")
log(f"  chunks/sample (train): mean={CM_tr.sum(1).mean():.2f}, "
    f"4-chunk={(CM_tr.sum(1)==4).mean():.2%}, 1-chunk={(CM_tr.sum(1)==1).mean():.2%}")


def span_mask_tokens(x: np.ndarray, tm: np.ndarray):
    \"\"\"In-place span masking on (n_chunks, chunk_size) arrays.

    Walk along real positions; with probability AUG_SPAN_MASK_PROB at any non-
    special position, start a span of length 1..AUG_SPAN_MAX_LEN that masks
    those tokens to PAD_ID. CLS (position [0, 0]) is never touched.
    \"\"\"
    x  = x.copy()
    tm = tm.copy()
    n_chunks, chunk_size = x.shape
    for c in range(n_chunks):
        if not tm[c].any():
            continue
        j = 1 if c == 0 else 0    # protect CLS in chunk 0
        while j < chunk_size:
            if not tm[c, j]:
                break
            if np.random.rand() < AUG_SPAN_MASK_PROB:
                span = np.random.randint(1, AUG_SPAN_MAX_LEN + 1)
                end = min(j + span, chunk_size)
                x[c, j:end]  = PAD_ID
                tm[c, j:end] = False
                j = end
            else:
                j += 1
    return x, tm


def chunk_dropout(x: np.ndarray, tm: np.ndarray, cm: np.ndarray):
    \"\"\"With probability AUG_CHUNK_DROP_PROB drop one random non-empty chunk.\"\"\"
    if np.random.rand() >= AUG_CHUNK_DROP_PROB:
        return x, tm, cm
    valid = np.where(cm)[0]
    if len(valid) <= 1:
        return x, tm, cm
    # Don't drop chunk 0 (carries CLS and the function-dispatch prologue)
    valid = valid[valid != 0]
    if len(valid) == 0:
        return x, tm, cm
    c = int(np.random.choice(valid))
    x = x.copy(); tm = tm.copy(); cm = cm.copy()
    x[c]  = PAD_ID
    tm[c] = False
    cm[c] = False
    return x, tm, cm


class OpcodeDS(Dataset):
    def __init__(self, X, TM, CM, Y, augment: bool = False):
        self.X, self.TM, self.CM, self.Y = X, TM, CM, Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, tm, cm = self.X[i], self.TM[i], self.CM[i]
        if self.augment:
            x, tm = span_mask_tokens(x, tm)
            x, tm, cm = chunk_dropout(x, tm, cm)
        return (torch.from_numpy(x).long(),
                torch.from_numpy(tm),
                torch.from_numpy(cm),
                torch.from_numpy(self.Y[i]))


ds_tr = OpcodeDS(X_tr, TM_tr, CM_tr, Y_tr, augment=True)
ds_v  = OpcodeDS(X_v,  TM_v,  CM_v,  Y_v,  augment=False)
ds_te = OpcodeDS(X_te, TM_te, CM_te, Y_te, augment=False)
log(f"Datasets ready: train={len(ds_tr)} val={len(ds_v)} test={len(ds_te)}")
log(f"  aug: span_mask_prob={AUG_SPAN_MASK_PROB} (max_len={AUG_SPAN_MAX_LEN}), "
    f"chunk_drop={AUG_CHUNK_DROP_PROB}, cutmix={AUG_CUTMIX_PROB} "
    f"(β={AUG_CUTMIX_BETA})")
""")

# ---------------------------------------------------------------------------
md("""## 8 — Class-balanced sampler + CutMix collate

**Sampler.** Per-sample weight = `max over positive labels of sqrt(N_total /
class_count)`, clipped to `[SAMPLER_W_MIN, SAMPLER_W_MAX]`. Samples with no
positive labels get weight 1. The clip prevents Bad Randomness / Front Running
(which have ~450 train positives → raw weight ≈ √35) from completely
dominating the batch.

**CutMix collate.** After the default-collated batch is assembled, with
probability `AUG_CUTMIX_PROB` pair each sample `i` with a random partner `j`
in the batch; sample `λ ~ Beta(0.5, 0.5)`; take `⌈λ·N_CHUNKS⌉` chunks from `i`,
the rest from `j`; mix labels by `λ`. AsymmetricLoss handles the soft targets
without modification.
""")

code("""def build_sampler_weights(Y_train: np.ndarray):
    counts = Y_train.sum(0).clip(min=1)
    inv = np.sqrt(Y_train.shape[0] / counts).astype(np.float32)
    # Per-sample weight = max over positive labels of inv[label]
    has_pos = Y_train.sum(1) > 0
    w = np.where(has_pos, (Y_train * inv[None, :]).max(1), 1.0).astype(np.float32)
    w = np.clip(w, SAMPLER_W_MIN, SAMPLER_W_MAX)
    return w, inv, counts


sampler_w, inv_freq, class_counts = build_sampler_weights(Y_tr)
log("Per-class train counts and sqrt-inverse-freq weights:")
for i, lab in enumerate(LABEL_COLS):
    log(f"  {lab:>26s}  n={int(class_counts[i]):>5d}  inv_w={inv_freq[i]:.2f}")
log(f"Sampler weights: min={sampler_w.min():.2f}, "
    f"median={np.median(sampler_w):.2f}, max={sampler_w.max():.2f}")

train_sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sampler_w).double(),
    num_samples=len(sampler_w),
    replacement=True,
)


def cutmix_collate(batch):
    \"\"\"Default-collate first, then chunk-level CutMix on the assembled batch.\"\"\"
    X  = torch.stack([b[0] for b in batch], dim=0)   # (B, C, T)
    TM = torch.stack([b[1] for b in batch], dim=0)
    CM = torch.stack([b[2] for b in batch], dim=0)
    Y  = torch.stack([b[3] for b in batch], dim=0)   # (B, n_labels)

    if AUG_CUTMIX_PROB <= 0 or np.random.rand() >= AUG_CUTMIX_PROB:
        return X, TM, CM, Y

    B, C, _ = X.shape
    perm = torch.randperm(B)
    lam = float(np.random.beta(AUG_CUTMIX_BETA, AUG_CUTMIX_BETA))
    n_from_i = int(np.ceil(lam * C))
    n_from_i = max(1, min(C - 1, n_from_i))    # always mix something
    keep = torch.zeros(C, dtype=torch.bool)
    keep_idx = torch.randperm(C)[:n_from_i]
    keep[keep_idx] = True

    X_mix  = torch.where(keep[None, :, None], X,  X[perm])
    TM_mix = torch.where(keep[None, :, None], TM, TM[perm])
    CM_mix = torch.where(keep[None, :],       CM, CM[perm])

    # Effective lambda = fraction of *real* tokens that came from sample i
    real_tok = TM.float().sum(dim=(1, 2)).clamp(min=1.0)
    real_tok_perm = TM[perm].float().sum(dim=(1, 2)).clamp(min=1.0)
    real_i = (TM.float() * keep[None, :, None].float()).sum(dim=(1, 2))
    real_j = (TM[perm].float() * (~keep)[None, :, None].float()).sum(dim=(1, 2))
    lam_eff = (real_i / (real_i + real_j).clamp(min=1.0)).unsqueeze(1)
    Y_mix = lam_eff * Y + (1.0 - lam_eff) * Y[perm]

    return X_mix, TM_mix, CM_mix, Y_mix
""")

# ---------------------------------------------------------------------------
md("""## 9 — Hierarchical model: chunk encoder + chunk aggregator

The encoder is **bit-identical** to dive-3 (same CNN front-end, same 6
pre-norm Transformer layers with DropPath, same dual mean⊕attn pool). What's
new is the wrapper: input shape is `(B, N_CHUNKS, CHUNK_SIZE)`; the encoder
runs once per chunk (folded into a `(B·N_CHUNKS, CHUNK_SIZE)` batch), produces
a `2·d_model` vector per chunk; then we add a small learned chunk-position
embedding and attention-pool over chunks into the final `2·d_model` contract
vector. Same head as dive-3.

Param-count delta vs dive-3 is negligible (`+chunk_pos_emb (N_CHUNKS × d)` and
`+chunk_attn_score (2d → 1)`).
""")

code("""class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class StochDepthEncoderLayer(nn.Module):
    \"\"\"Pre-norm Transformer encoder layer with DropPath on each residual.\"\"\"
    def __init__(self, d_model, n_heads, d_ff, dropout, drop_path):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dp1(self.drop1(a))
        h = self.ln2(x)
        x = x + self.dp2(self.drop2(self.ff(h)))
        return x


class ChunkEncoder(nn.Module):
    \"\"\"Same encoder as dive-3, returns (B, 2*d_model) — mean⊕attn pool.\"\"\"
    def __init__(self, vocab_size=VOCAB_SIZE,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                 max_len=CHUNK_SIZE, dropout=DROPOUT, drop_path=DROP_PATH):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.emb_norm  = nn.LayerNorm(d_model)
        self.emb_drop  = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(d_model)

        dp_rates = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            StochDepthEncoderLayer(d_model, n_heads, d_ff, dropout, dp_rates[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.attn_score = nn.Linear(d_model, 1)

    def forward(self, input_ids, attn_mask):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_drop(self.emb_norm(x))

        c = x.transpose(1, 2)
        c = torch.cat([F.gelu(self.conv3(c)), F.gelu(self.conv5(c))], dim=1)
        x = self.conv_norm(x + c.transpose(1, 2))

        kp = ~attn_mask
        for layer in self.layers:
            x = layer(x, key_padding_mask=kp)
        x = self.final_norm(x)

        mf = attn_mask.float().unsqueeze(-1)
        denom = mf.sum(1).clamp(min=1.0)
        mean_pool = (x * mf).sum(1) / denom

        s = self.attn_score(x).squeeze(-1)
        s = s.masked_fill(~attn_mask, -1e4)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        attn_pool = (x * w).sum(1)

        return torch.cat([mean_pool, attn_pool], dim=-1)   # (B, 2*d_model)


class HierarchicalOpcodeModel(nn.Module):
    def __init__(self, num_classes=N_LABELS, d_model=D_MODEL, n_chunks=N_CHUNKS,
                 dropout=DROPOUT):
        super().__init__()
        self.encoder = ChunkEncoder()
        self.chunk_pos = nn.Embedding(n_chunks, 2 * d_model)
        nn.init.trunc_normal_(self.chunk_pos.weight, std=0.02)

        self.chunk_attn_score = nn.Linear(2 * d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model * 2),
            nn.Linear(2 * d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        # Sensible init for the head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, tok_mask, chunk_mask):
        \"\"\"
        input_ids:  (B, C, T)
        tok_mask:   (B, C, T)  bool
        chunk_mask: (B, C)     bool
        \"\"\"
        B, C, T = input_ids.shape
        flat_ids  = input_ids.view(B * C, T)
        flat_mask = tok_mask.view(B * C, T)

        # Empty chunks would produce NaNs in attention softmax over key_padding_mask.
        # Force them to have at least 1 valid token (we'll re-mask their embeddings).
        empty = ~flat_mask.any(dim=1)
        if empty.any():
            flat_mask[empty, 0] = True
            # flat_ids[empty, 0] is already PAD_ID — encoder embeds it harmlessly.

        chunk_emb = self.encoder(flat_ids, flat_mask)     # (B*C, 2d)
        chunk_emb = chunk_emb.view(B, C, -1)
        # Zero out embeddings of fully-empty chunks so they contribute nothing
        chunk_emb = chunk_emb * chunk_mask.unsqueeze(-1).float()

        # Add learned chunk-position embedding to non-empty chunks
        pos_ids = torch.arange(C, device=input_ids.device)
        chunk_emb = chunk_emb + self.chunk_pos(pos_ids).unsqueeze(0) * chunk_mask.unsqueeze(-1).float()

        # mean-pool over valid chunks
        cmf = chunk_mask.float().unsqueeze(-1)
        mean_chunks = (chunk_emb * cmf).sum(1) / cmf.sum(1).clamp(min=1.0)

        # attn-pool over valid chunks
        s = self.chunk_attn_score(chunk_emb).squeeze(-1)
        s = s.masked_fill(~chunk_mask, -1e4)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        attn_chunks = (chunk_emb * w).sum(1)

        return self.head(torch.cat([mean_chunks, attn_chunks], dim=-1))


_m = HierarchicalOpcodeModel()
n_params = sum(p.numel() for p in _m.parameters())
log(f"Parameters: {n_params/1e6:.2f} M  ({n_params:,})")
del _m; gc.collect()
""")

# ---------------------------------------------------------------------------
md("""## 10 — AsymmetricLoss with per-class weights

Same as dive-3 but multiplied per-class by a learnable-free `class_weights`
vector (sqrt-inverse-frequency, clipped to `[1, 5]`). The mean is taken over
B × C (so the per-class weighting is *relative*, not an absolute gradient
scaling) — total magnitude stays comparable to dive-3.
""")

code("""class AsymmetricLossWeighted(nn.Module):
    def __init__(self, class_weights: np.ndarray,
                 gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS, clip=ASL_CLIP, eps=1e-8):
        super().__init__()
        self.register_buffer("class_weights", torch.from_numpy(class_weights.astype(np.float32)))
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # fp32 for AMP stability
        logits = logits.float()
        x_sig = torch.sigmoid(logits)
        xs_pos = x_sig
        xs_neg = 1.0 - x_sig
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg                              # (B, C)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w
        # Per-class weighting (broadcast over batch)
        cw = self.class_weights.to(loss.device)
        loss = loss * cw.unsqueeze(0)
        return -loss.mean()


# Compute per-class weights from train frequencies
_cw = np.sqrt(Y_tr.shape[0] / Y_tr.sum(0).clip(min=1)).astype(np.float32)
class_weights = np.clip(_cw, PER_CLASS_W_MIN, PER_CLASS_W_MAX)
# Normalise so the mean weight is 1 — keeps overall loss magnitude comparable to dive-3
class_weights = class_weights * (len(class_weights) / class_weights.sum())
log("Per-class loss weights (normalised, mean=1):")
for i, lab in enumerate(LABEL_COLS):
    log(f"  {lab:>26s}  w={class_weights[i]:.3f}")

criterion = AsymmetricLossWeighted(class_weights).to(device)
log(f"Loss: AsymmetricLossWeighted(γ⁻={ASL_GAMMA_NEG}, γ⁺={ASL_GAMMA_POS}, "
    f"clip={ASL_CLIP}, per-class w clipped to [{PER_CLASS_W_MIN}, {PER_CLASS_W_MAX}])")
""")

# ---------------------------------------------------------------------------
md("""## 11 — EMA""")

code("""class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        src = model.module if isinstance(model, nn.DataParallel) else model
        self.ema = HierarchicalOpcodeModel().to(next(src.parameters()).device)
        self.ema.load_state_dict(src.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        src = model.module if isinstance(model, nn.DataParallel) else model
        msd = src.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)
""")

# ---------------------------------------------------------------------------
md("""## 12 — Metrics, isotonic calibration, threshold tuning

**Calibration.** For each class we fit an `IsotonicRegression(out_of_bounds=
'clip')` on validation probabilities versus binary labels. At test time, val-
fit isotonic regressors are applied to test probabilities **before** threshold
search. This converts the model's raw scores (good rank, bad probability
shape) into well-calibrated probabilities, which is what F1 actually needs.

**Threshold tuning.** 41 points in `[0.05, 0.95]`. For rare classes (val
support < 200) we add a precision floor of `0.25` — without it the optimiser
sometimes picks `t=0.05` (predict-everything) for tiny supports because that
maximises F1 on the val set but generalises poorly.
""")

code("""from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             hamming_loss, accuracy_score, confusion_matrix)
from sklearn.isotonic import IsotonicRegression


def multilabel_metrics(y_true, y_prob, thresholds):
    if np.isscalar(thresholds):
        thresholds = np.full(y_prob.shape[1], thresholds, dtype=np.float32)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    out = {
        "f1_micro":     f1_score(y_true, y_pred, average="micro",   zero_division=0),
        "f1_macro":     f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_samples":   f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match":  accuracy_score(y_true, y_pred),
    }
    per_class = {}
    for i, lab in enumerate(LABEL_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        try:
            auc = roc_auc_score(yt, y_prob[:, i])
        except ValueError:
            auc = float("nan")
        per_class[lab] = {
            "support":    int(yt.sum()),
            "threshold":  float(thresholds[i]),
            "precision":  precision_score(yt, yp, zero_division=0),
            "recall":     recall_score(yt, yp, zero_division=0),
            "f1":         f1_score(yt, yp, zero_division=0),
            "roc_auc":    auc,
            "ap":         average_precision_score(yt, y_prob[:, i]),
        }
    return out, per_class


def fit_isotonic_per_class(y_true, y_prob):
    \"\"\"Fit one isotonic regressor per class. Returns a list of fitted models.\"\"\"
    calibrators = []
    for i in range(y_prob.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # Add eps-stability if a class is constant in val (shouldn't happen at our
        # supports but cheap insurance).
        if len(np.unique(y_true[:, i])) < 2:
            iso.fit([0.0, 1.0], [0.0, 1.0])
        else:
            iso.fit(y_prob[:, i], y_true[:, i])
        calibrators.append(iso)
    return calibrators


def apply_calibration(y_prob, calibrators):
    out = np.zeros_like(y_prob)
    for i, iso in enumerate(calibrators):
        out[:, i] = iso.transform(y_prob[:, i])
    return out


def tune_thresholds(y_true, y_prob, min_precision_rare=0.25, rare_support=200):
    grid = np.linspace(0.05, 0.95, 41)
    thresholds = np.full(y_prob.shape[1], 0.5, dtype=np.float32)
    for i in range(y_prob.shape[1]):
        support = int(y_true[:, i].sum())
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            # Min-precision floor for rare classes
            if support < rare_support:
                prec = precision_score(y_true[:, i], pred, zero_division=0)
                if prec < min_precision_rare:
                    continue
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds
""")

# ---------------------------------------------------------------------------
md("""## 13 — Optimizer, schedule, predict""")

code("""def build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower() or "emb" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8)

def cosine_warmup(optimizer, total_steps, warmup_ratio=WARMUP_RATIO):
    warm = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    logits_all, labels_all = [], []
    for X, TM, CM, Yb in loader:
        X  = X.to(device, non_blocking=True)
        TM = TM.to(device, non_blocking=True)
        CM = CM.to(device, non_blocking=True)
        with autocast():
            logits = model(X, TM, CM)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(Yb.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)
""")

# ---------------------------------------------------------------------------
md("""## 14 — Dataloaders""")

code("""train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          drop_last=True, persistent_workers=NUM_WORKERS>0,
                          collate_fn=cutmix_collate)
val_loader   = DataLoader(ds_v,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS>0)
test_loader  = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS>0)
log(f"Batch size: {BATCH_SIZE} ({BATCH_PER_GPU}/GPU × {max(1,n_gpus)} GPUs)  "
    f"[encoder sees {BATCH_SIZE * N_CHUNKS} chunks/step]")
log(f"Steps/epoch: {len(train_loader)}, total: {len(train_loader)*EPOCHS}")
""")

# ---------------------------------------------------------------------------
md("""## 15 — Checkpoint save / load (atomic, full state)

Same atomic-rename pattern as dive-3. Additional payload: the list of isotonic
calibrators (pickled via torch.save) under key `calibrators`. Resuming reloads
them too, so a mid-run crash → resume → eval works end-to-end.
""")

code("""def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(path, *, model, ema, optimizer, scheduler, scaler,
                    epoch, best_macro_f1, best_thresholds, history,
                    calibrators=None, extra=None):
    src = model.module if isinstance(model, nn.DataParallel) else model
    payload = {
        "model":     src.state_dict(),
        "ema":       ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "best_macro_f1":    float(best_macro_f1),
        "best_thresholds":  np.asarray(best_thresholds, dtype=np.float32).tolist(),
        "calibrators": calibrators,    # None until end of training
        "history":   history,
        "rng": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": {
            "CHUNK_SIZE": CHUNK_SIZE, "N_CHUNKS": N_CHUNKS, "MAX_TOTAL_OPS": MAX_TOTAL_OPS,
            "VOCAB_SIZE": VOCAB_SIZE,
            "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS, "D_FF": D_FF,
            "DROPOUT": DROPOUT, "DROP_PATH": DROP_PATH, "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "ASL_GAMMA_NEG": ASL_GAMMA_NEG, "ASL_GAMMA_POS": ASL_GAMMA_POS,
            "ASL_CLIP": ASL_CLIP, "EMA_DECAY": EMA_DECAY,
            "AUG_SPAN_MASK_PROB": AUG_SPAN_MASK_PROB, "AUG_SPAN_MAX_LEN": AUG_SPAN_MAX_LEN,
            "AUG_CHUNK_DROP_PROB": AUG_CHUNK_DROP_PROB,
            "AUG_CUTMIX_PROB": AUG_CUTMIX_PROB, "AUG_CUTMIX_BETA": AUG_CUTMIX_BETA,
            "SEED": SEED,
            "class_weights": class_weights.tolist(),
        },
        "extra": extra or {},
    }
    _atomic_save(payload, Path(path))


def load_checkpoint(path, *, model, ema, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=device)
    src = model.module if isinstance(model, nn.DataParallel) else model
    src.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return ckpt
""")

# ---------------------------------------------------------------------------
md("""## 16 — Build model, optimiser, scheduler, scaler, EMA""")

code("""model = HierarchicalOpcodeModel().to(device)
if n_gpus > 1:
    model = nn.DataParallel(model)
optimizer  = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler  = cosine_warmup(optimizer, total_steps, WARMUP_RATIO)
scaler     = GradScaler()
ema        = ModelEMA(model, decay=EMA_DECAY)

start_epoch = 1
best_macro_f1 = -1.0
best_thresholds = np.full(N_LABELS, 0.5, dtype=np.float32)
history = []
patience_left = PATIENCE

if RESUME_FROM is not None and Path(RESUME_FROM).exists():
    ckpt = load_checkpoint(RESUME_FROM,
                           model=model, ema=ema,
                           optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    start_epoch     = int(ckpt["epoch"]) + 1
    best_macro_f1   = float(ckpt["best_macro_f1"])
    best_thresholds = np.asarray(ckpt["best_thresholds"], dtype=np.float32)
    history         = list(ckpt["history"])
    log(f"RESUMED from {RESUME_FROM}: epoch {ckpt['epoch']} → continuing at {start_epoch}")
    log(f"  best so far: macro_f1={best_macro_f1:.4f}")
else:
    log("Starting fresh (RESUME_FROM is None or missing)")

log(f"Optimizer ready | total_steps={total_steps} | warmup={int(total_steps*WARMUP_RATIO)}")
""")

# ---------------------------------------------------------------------------
md("""## 17 — Training loop

Per epoch:

1. **Train** — sampler delivers class-balanced batches; CutMix collate optionally
   mixes pairs; AMP + grad-clip + cosine schedule + EMA update.
2. **Validate on EMA** — predict, **fit isotonic per class on val**, apply it,
   then tune thresholds on the calibrated probs. Reported `f1_macro` is on
   calibrated+thresholded predictions, so the early-stop and best-model logic
   track the *deployed* pipeline, not a raw-logit illusion.
3. **Save full state** atomically. Save `best_model.pt` (EMA weights, isotonic
   calibrators, frozen thresholds) on each new best.
4. **Wall-clock snapshot** every `CKPT_WALLCLOCK_SECS`.
5. **Early stop** after `PATIENCE` consecutive non-improvements.

A `try/except/finally` flushes state on any exception before re-raising.
""")

code("""HEARTBEAT_STEPS = 50


def _run_epoch(epoch):
    model.train()
    t0 = time.time()
    n_steps = len(train_loader)
    running_loss = 0.0
    running_n = 0

    for step, (X, TM, CM, Yb) in enumerate(train_loader, 1):
        X  = X.to(device,  non_blocking=True)
        TM = TM.to(device, non_blocking=True)
        CM = CM.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(X, TM, CM)
            loss   = criterion(logits, Yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        running_loss += loss.item(); running_n += 1
        if step % HEARTBEAT_STEPS == 0 or step == n_steps:
            cur_lr = optimizer.param_groups[0]["lr"]
            avg = running_loss / running_n
            elapsed = time.time() - t0
            eta = elapsed * (n_steps / step - 1)
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"  epoch {epoch:2d}  step {step:>4d}/{n_steps} | "
                f"loss={avg:.4f} | lr={cur_lr:.2e} | "
                f"elapsed={elapsed:5.1f}s eta={eta:5.1f}s | gpu_mem_peak={mem:.2f} GB")
    return running_loss / max(1, running_n), time.time() - t0


def _flush_history():
    pd.DataFrame(history).to_csv(HIST_CSV, index=False)
    with open(HIST_JSON, "w") as f:
        json.dump(history, f, indent=2)


last_ckpt_ts = time.time()
epoch_save_done = False
best_calibrators = None

try:
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_save_done = False
        # 1) Train
        train_loss, train_dt = _run_epoch(epoch)

        # 2) Validate on EMA — calibrate, then threshold
        val_logits, val_labels = predict(ema.ema, val_loader)
        val_probs_raw = 1 / (1 + np.exp(-val_logits))
        calibrators = fit_isotonic_per_class(val_labels, val_probs_raw)
        val_probs   = apply_calibration(val_probs_raw, calibrators)
        val_thr     = tune_thresholds(val_labels, val_probs)
        val_metrics, _ = multilabel_metrics(val_labels, val_probs, val_thr)
        # Also report uncalibrated @ 0.5 for sanity
        val_metrics_raw_05, _ = multilabel_metrics(val_labels, val_probs_raw, 0.5)
        cur_lr = optimizer.param_groups[0]["lr"]

        log(f"Epoch {epoch:2d}/{EPOCHS} | train_dt={train_dt:5.1f}s | "
            f"lr={cur_lr:.2e} | train_loss={train_loss:.4f} | "
            f"EMA val (calib+thr): f1_macro={val_metrics['f1_macro']:.4f} "
            f"f1_micro={val_metrics['f1_micro']:.4f} "
            f"ham={val_metrics['hamming_loss']:.4f} | "
            f"raw@0.5: f1_macro={val_metrics_raw_05['f1_macro']:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "lr": cur_lr,
            "train_dt": train_dt, **val_metrics,
            "val_f1_macro_raw_05": val_metrics_raw_05["f1_macro"],
        })
        _flush_history()

        # 3) Always-save full state (atomic)
        save_checkpoint(STATE_PATH,
                        model=model, ema=ema,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch,
                        best_macro_f1=best_macro_f1,
                        best_thresholds=best_thresholds,
                        history=history,
                        calibrators=calibrators)
        epoch_save_done = True

        # 4) Best-model save (EMA weights + calibrators + thresholds)
        if val_metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = val_metrics["f1_macro"]
            best_thresholds = val_thr.copy()
            best_calibrators = calibrators
            _atomic_save({"model": ema.state_dict(),
                          "thresholds": best_thresholds,
                          "calibrators": calibrators,
                          "epoch": epoch,
                          "val_metrics": val_metrics}, BEST_PATH)
            log(f"  ✓ NEW BEST EMA macro-F1={best_macro_f1:.4f} → {BEST_PATH.name}")
            patience_left = PATIENCE
        else:
            patience_left -= 1
            log(f"  no improvement (best={best_macro_f1:.4f}, patience={patience_left}/{PATIENCE})")

        # 5) 30-min wall-clock checkpoint snapshot
        if time.time() - last_ckpt_ts >= CKPT_WALLCLOCK_SECS:
            snap = OUT_DIR / f"state_epoch{epoch:02d}.pt"
            save_checkpoint(snap,
                            model=model, ema=ema,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch,
                            best_macro_f1=best_macro_f1,
                            best_thresholds=best_thresholds,
                            history=history,
                            calibrators=calibrators)
            log(f"  wall-clock snapshot → {snap.name}")
            last_ckpt_ts = time.time()

        # 6) Early stopping
        if patience_left <= 0:
            log(f"Early stopping at epoch {epoch} (no EMA macro-F1 improvement for {PATIENCE} epochs)")
            break

    log(f"Training done. Best EMA val macro-F1 = {best_macro_f1:.4f}")

except Exception as e:
    log("!! Exception during training — flushing state before re-raising")
    log(traceback.format_exc())
    try:
        if 'epoch' in locals() and not epoch_save_done:
            crash_epoch = max(0, epoch - 1)
            save_checkpoint(STATE_PATH,
                            model=model, ema=ema,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=crash_epoch,
                            best_macro_f1=best_macro_f1,
                            best_thresholds=best_thresholds,
                            history=history,
                            calibrators=best_calibrators,
                            extra={"error": repr(e)})
            log(f"State flushed at epoch={crash_epoch} → {STATE_PATH}. "
                f"Set RESUME_FROM='{STATE_PATH}' and re-run; epoch {epoch} will be redone.")
        else:
            log(f"Most recent successful save in {STATE_PATH} preserved.")
    except Exception as e2:
        log(f"Also failed to flush state: {e2!r}")
    raise
finally:
    _flush_history()
""")

# ---------------------------------------------------------------------------
md("""## 18 — Test evaluation (calibrated + thresholded)

Load `best_model.pt` (EMA weights, isotonic calibrators, frozen thresholds).
Apply calibration to test probabilities, then frozen thresholds. We also
report:

- Test metrics on **calibrated + tuned-threshold** preds (the deployed setting).
- Test metrics on **calibrated @ 0.5** (calibration-only).
- Test metrics on **raw @ 0.5** (no calibration, no tuning — reference).

This makes the threshold-tuning effect and the calibration effect both
explicit and honest.
""")

code("""log("=== TEST EVAL ===")
if not BEST_PATH.exists():
    log("No best_model.pt found — falling back to current EMA weights and last calibrators.")
    eval_model = ema.ema
    frozen_thresholds = best_thresholds
    frozen_calibrators = best_calibrators
else:
    ckpt_b = torch.load(BEST_PATH, map_location=device)
    eval_model = HierarchicalOpcodeModel().to(device)
    eval_model.load_state_dict(ckpt_b["model"])
    frozen_thresholds = np.asarray(ckpt_b["thresholds"], dtype=np.float32)
    frozen_calibrators = ckpt_b["calibrators"]
    log(f"Loaded best model from epoch {ckpt_b.get('epoch','?')}: "
        f"val_f1_macro={ckpt_b.get('val_metrics',{}).get('f1_macro','?')}")

if n_gpus > 1 and not isinstance(eval_model, nn.DataParallel):
    eval_model = nn.DataParallel(eval_model)

test_logits, test_labels_arr = predict(eval_model, test_loader)
test_probs_raw = 1 / (1 + np.exp(-test_logits))
test_probs     = apply_calibration(test_probs_raw, frozen_calibrators)

test_metrics_tuned, per_class_tuned = multilabel_metrics(test_labels_arr, test_probs, frozen_thresholds)
test_metrics_calib_05, _            = multilabel_metrics(test_labels_arr, test_probs, 0.5)
test_metrics_raw_05, _              = multilabel_metrics(test_labels_arr, test_probs_raw, 0.5)

log("=== TEST METRICS (calibrated + frozen per-class tuned thresholds) ===")
for k, v in test_metrics_tuned.items():
    log(f"  {k:>16s}: {v:.4f}")
log("=== TEST METRICS (calibrated @ 0.5) ===")
for k, v in test_metrics_calib_05.items():
    log(f"  {k:>16s}: {v:.4f}")
log("=== TEST METRICS (raw @ 0.5 — no calibration, no tuning) ===")
for k, v in test_metrics_raw_05.items():
    log(f"  {k:>16s}: {v:.4f}")

pc_df = pd.DataFrame(per_class_tuned).T.round(4)
log("=== PER-CLASS (calibrated + tuned thresholds) ===")
for line in pc_df.to_string().splitlines():
    log("  " + line)
""")

code("""# ── Artefact dump ───────────────────────────────────────────────────────────
np.save(OUT_DIR / "test_probs_calibrated.npy", test_probs)
np.save(OUT_DIR / "test_probs_raw.npy",        test_probs_raw)
np.save(OUT_DIR / "test_labels.npy",           test_labels_arr)
np.save(OUT_DIR / "thresholds.npy",            frozen_thresholds)
pc_df.to_csv(OUT_DIR / "per_class.csv")

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "test_calibrated_tuned":  test_metrics_tuned,
        "test_calibrated_at_0.5": test_metrics_calib_05,
        "test_raw_at_0.5":        test_metrics_raw_05,
        "per_class":              per_class_tuned,
        "best_val_f1_macro":      best_macro_f1,
        "thresholds":             frozen_thresholds.tolist(),
        "hparams": {
            "chunk_size": CHUNK_SIZE, "n_chunks": N_CHUNKS, "max_total_ops": MAX_TOTAL_OPS,
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF,
            "dropout": DROPOUT, "drop_path": DROP_PATH,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "asl_gamma_neg": ASL_GAMMA_NEG, "asl_gamma_pos": ASL_GAMMA_POS, "asl_clip": ASL_CLIP,
            "ema_decay": EMA_DECAY,
            "aug_span_mask_prob": AUG_SPAN_MASK_PROB, "aug_span_max_len": AUG_SPAN_MAX_LEN,
            "aug_chunk_drop_prob": AUG_CHUNK_DROP_PROB,
            "aug_cutmix_prob": AUG_CUTMIX_PROB, "aug_cutmix_beta": AUG_CUTMIX_BETA,
            "sampler_w_min": SAMPLER_W_MIN, "sampler_w_max": SAMPLER_W_MAX,
            "per_class_w_min": PER_CLASS_W_MIN, "per_class_w_max": PER_CLASS_W_MAX,
            "class_weights": class_weights.tolist(),
            "patience": PATIENCE, "seed": SEED,
        }}, f, indent=2)

cms = {}
y_pred_final = (test_probs >= frozen_thresholds[None, :]).astype(np.int32)
for i, lab in enumerate(LABEL_COLS):
    cm = confusion_matrix(test_labels_arr[:, i], y_pred_final[:, i], labels=[0, 1])
    cms[lab] = cm.tolist()
with open(OUT_DIR / "confusion_per_class.json", "w") as f:
    json.dump(cms, f, indent=2)

log("Artefacts written:")
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        log(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")

log.close()
""")

# ---------------------------------------------------------------------------
md("""## Summary

**What dive-4 changes vs dive-3 (encoder frozen, input pipeline rebuilt).**

- **Hierarchical chunked encoding** (4 × 1024 = 4096 opcodes / contract). Same
  encoder runs once per chunk, chunk embeddings attention-pooled. Directly
  attacks the 94 % truncation rate in dive-3.
- **Class-balanced WeightedRandomSampler** — rare-class positives oversampled,
  weights clipped to `[1, 5]`.
- **Chunk-level CutMix** — sequence-aware MixUp, soft multilabel targets handled
  natively by ASL.
- **Span masking** (1-3, 8 %) + **chunk dropout** (5 %) replace dive-3's
  per-token augmentation.
- **Per-class loss weights** on top of ASL (sqrt-inverse-freq, clipped, mean-
  normalised to 1 so total loss magnitude is preserved).
- **Isotonic calibration per class** on val probs, then threshold tuning on
  calibrated probs — fixes the AUC ≫ F1 gap for BR / FR / DoS.
- **Finer 41-point threshold grid** with a precision floor (0.25) for rare
  classes — rejects degenerate "predict everything" thresholds.

**Robustness (unchanged from dive-3).**

- Atomic full-state checkpoints every epoch + every 30 min of wall-clock to
  `last_state.pt`. Now also persists `calibrators`.
- `try/except/finally` flushes state on any error before re-raising.
- Tee logger to `dive4_train.log`, per-step heartbeats every 50 steps.

**Outputs (`/kaggle/working/`).**

- `best_model.pt` — EMA weights + isotonic calibrators + frozen thresholds.
- `last_state.pt` — full resumable state (includes current-epoch calibrators).
- `state_epochNN.pt` — wall-clock snapshots every 30 min.
- `dive4_train.log`, `history.csv`, `history.json`.
- `metrics.json`, `per_class.csv`, `confusion_per_class.json`.
- `test_probs_calibrated.npy`, `test_probs_raw.npy`, `test_labels.npy`,
  `thresholds.npy`.
- `cache/dive4_tokens.npz` — disassembly @ MAX_TOTAL_OPS=4096.

**What to look at after the run.**

1. **Truncation log line** — `Truncated at MAX_TOTAL_OPS=4096: X.XX%`. This
   tells you what fraction of contracts is still being chopped at the new
   ceiling. If it's still > 30 % we need to chunk further or go to a long-
   context attention model.
2. **`test_calibrated_tuned.f1_macro` vs dive-3's 0.6636.** Expect +0.02 to
   +0.05 if chunking + calibration are doing real work.
3. **`per_class.csv`** — focus on BR / FR / DoS rows. The hypothesis is that
   calibration narrows the AUC↔F1 gap on those specifically, and the sampler
   improves recall. If they jump and the common classes hold their numbers,
   the diagnosis was right.
4. Compare `val_f1_macro_raw_05` (from `history.csv`) vs `val_f1_macro` — the
   delta tells you exactly how much calibration is buying you each epoch.
""")

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "dive-4.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
