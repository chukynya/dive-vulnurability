"""Build dive-3.ipynb: dive-2 architecture + asymmetric loss, EMA, stochastic
depth, token augmentation, resumable training and log-to-file mirror.

Stays bytecode-only at 1024 ctx for a direct apples-to-apples comparison with
dive-2 (whose best val macro-F1 was 0.6592). The deltas are training-side, not
architectural — the goal is to extract more signal from the same input.
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
md("""# DIVE-3 — Bytecode-only, harder training

**Why this notebook.** `dive-2` (CNN-Transformer on opcode sequences) plateaued
at val macro-F1 ≈ 0.6592 with `train_loss 0.81 → 0.24` — classic overfit. The
model is at the ceiling of what BCE + `pos_weight` lets it extract from this
input. This notebook keeps the architecture and the 1024-opcode context
**identical** so the comparison is clean, and changes only the training side.

## What's new vs dive-2

| Change | Why |
|---|---|
| **Asymmetric Loss** (Ridnik et al. 2021, ICCV) | The standard upgrade over BCE+`pos_weight` for imbalanced multi-label. Decouples positive/negative focusing (`γ⁻=4, γ⁺=1`) and probability-shifts hard negatives (`clip=0.05`) so rare-class positives don't get drowned out. Reported +2–5 % macro-F1 on COCO/PASCAL/OpenImages over BCE. |
| **EMA of weights** (decay 0.999) | Smooths the late-epoch oscillation you see in the dive-2 log (epochs 8–20 bounce between 0.636–0.659). Evaluate on the EMA copy each epoch and checkpoint that. |
| **Stochastic depth** (DropPath, linear 0 → 0.1) | Per-layer regularisation in the Transformer stack. Cheap and consistently helps when train-loss collapses while val stalls. |
| **Token augmentation** (random PAD, opcode swap) | Light, sequence-aware augmentation. Drops 5 % of opcode positions to PAD and swaps adjacent pairs with 5 % probability. Both preserve the basic-block-level signal that matters. |
| **Stronger weight decay** (0.05) | dive-2 used 0.01 — too weak for a 6 M model overfitting this hard. |
| **30 epochs + patience-10 early stop** | Cosine schedule re-tuned for the longer run. Stops if EMA val macro-F1 hasn't improved in 10 epochs. |
| **Full-state checkpointing** | model + EMA + optimizer + scheduler + scaler + RNG + epoch + history dumped every epoch *and* every 30 min wall-clock to `/kaggle/working/last_state.pt`. Resume by setting `RESUME_FROM = "/kaggle/working/last_state.pt"`. Atomic writes (`.tmp → rename`). |
| **`tee`-style file logger** | Everything printed is also written to `/kaggle/working/dive3_train.log` and `history.csv` after every epoch. Kaggle "Save Version" preserves these so you see the full run from the saved-version page. |

## What's intentionally unchanged

- Tokenisation, vocab size 259, `MAX_OPS=1024`, PUSH-skipping rule.
- Model architecture, sizes, and the dual mean⊕attn pooling.
- Multilabel-stratified 80/10/10 split, fixed seed 42.
- Per-class threshold tuning on val, frozen for test.
- `torch.cuda.amp` + `nn.DataParallel` over T4 ×2.

If macro-F1 jumps notably above 0.66 with **only** the training-side changes
above, the dive-2 bottleneck was loss/regularisation, not the bytecode signal.
If it doesn't, the ceiling is signal — the next move would be to add a modality
(source code, transaction features), which is `dive-4` territory.

## How to use

1. **Fresh run** — leave `RESUME_FROM = None` (default) and Run All.
2. **Resume after a crash / timeout** — set `RESUME_FROM = "/kaggle/working/last_state.pt"` in the config cell, Run All. Training continues from the last saved epoch with optimizer/scheduler/scaler/RNG fully restored.
3. **Watching the run from Save Version** — open `dive3_train.log` and `history.csv` in the Output tab; both are flushed after every epoch.
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
from torch.utils.data import Dataset, DataLoader
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
  best-metric are all restored. Training resumes at the next epoch.
""")

code("""# ── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/kaggle/input/datasets/henrychristian7555/dive-smart-contract-multi-class-vulnerability")
BYTECODE_CSV = DATA_ROOT / "Bytecode_filled.csv"
LABEL_CSV    = DATA_ROOT / "DIVE_Labels.csv"

OUT_DIR   = Path("/kaggle/working");      OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache";            CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH  = OUT_DIR / "dive3_train.log"
HIST_CSV  = OUT_DIR / "history.csv"
HIST_JSON = OUT_DIR / "history.json"
STATE_PATH = OUT_DIR / "last_state.pt"      # full resumable state
BEST_PATH  = OUT_DIR / "best_model.pt"      # EMA weights + thresholds only

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
MAX_OPS = 1024                  # same as dive-2 for direct comparison

# ── Model ───────────────────────────────────────────────────────────────────
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 6
D_FF     = 1024
DROPOUT  = 0.15            # ↑ from 0.1
DROP_PATH = 0.10           # stochastic depth max rate (linearly scaled 0→DROP_PATH)

# ── Training ────────────────────────────────────────────────────────────────
BATCH_PER_GPU = 32
BATCH_SIZE    = BATCH_PER_GPU * max(1, n_gpus)
EPOCHS        = 30
LR            = 2e-4
WEIGHT_DECAY  = 0.05       # ↑ from 0.01
GRAD_CLIP     = 1.0
WARMUP_RATIO  = 0.10
PATIENCE      = 10         # early-stop patience on EMA val macro-F1
NUM_WORKERS   = 2
PIN_MEMORY    = True

# ── Asymmetric Loss hyper-params (Ridnik et al. 2021) ──────────────────────
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP      = 0.05

# ── EMA ─────────────────────────────────────────────────────────────────────
EMA_DECAY = 0.999

# ── Augmentation ────────────────────────────────────────────────────────────
AUG_TOKEN_MASK_PROB = 0.05   # randomly PAD this fraction of non-special tokens
AUG_TOKEN_SWAP_PROB = 0.05   # swap each adjacent (i, i+1) pair with this probability

# ── Checkpointing ───────────────────────────────────────────────────────────
CKPT_WALLCLOCK_SECS = 30 * 60   # also flush a checkpoint every 30 minutes

# ── Sanity check on data files ──────────────────────────────────────────────
assert BYTECODE_CSV.exists() and LABEL_CSV.exists(), \\
    f"Missing data files under {DATA_ROOT}"
print("Inputs OK:", BYTECODE_CSV.name, "|", LABEL_CSV.name, flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
""")

# ---------------------------------------------------------------------------
md("""## 3 — `tee` logger

Every `log(...)` line goes to **both** stdout and `dive3_train.log`. The file
is opened in append mode so it survives resumes — you get one continuous
transcript across sessions. Kaggle's "Save Version" persists the file in the
output tab.
""")

code("""class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", buffering=1, encoding="utf-8")   # line-buffered
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
md("""## 5 — EVM opcode disassembly (cached)

Same rule as dive-2: skip `PUSH1..PUSH32` immediates so we tokenise control
flow only. Cache to `.npz` so resuming is fast.
""")

code("""def disassemble(bc_hex: str, max_ops: int = MAX_OPS):
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

cache_path = CACHE_DIR / "dive3_tokens.npz"
if cache_path.exists():
    z = np.load(cache_path, allow_pickle=True)
    token_lists = list(z["tokens"])
    log(f"Loaded cached tokens for {len(token_lists)} contracts")
else:
    t0 = time.time()
    token_lists = [disassemble(bc, MAX_OPS) for bc in df["bytecode"].values]
    log(f"Disassembled {len(token_lists)} contracts in {time.time()-t0:.1f}s")
    np.savez_compressed(cache_path, tokens=np.array(token_lists, dtype=object))

lens = np.array([len(t) for t in token_lists])
log(f"Opcode-seq length: median={int(np.median(lens))}, p90={int(np.percentile(lens,90))}, "
    f"p99={int(np.percentile(lens,99))}, max={int(lens.max())}")
log(f"Truncated at MAX_OPS={MAX_OPS}: {(lens>=MAX_OPS).mean():.2%}")
""")

# ---------------------------------------------------------------------------
md("""## 6 — Stratified 80/10/10 split""")

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
md("""## 7 — Dataset + augmentation

`augment_tokens` is applied **on-the-fly inside `__getitem__`** so each epoch
sees a different perturbation:

- With probability `AUG_TOKEN_MASK_PROB`, replace a non-special token with `PAD`.
- With probability `AUG_TOKEN_SWAP_PROB`, swap each adjacent (i, i+1) pair —
  preserves local n-gram statistics but breaks exact phrase memorisation.

Augmentation is off at val/test time.
""")

code("""def to_padded(tok_list, max_len=MAX_OPS):
    n = len(tok_list)
    X = np.zeros((n, max_len), dtype=np.int32)
    M = np.zeros((n, max_len), dtype=np.bool_)
    for i, t in enumerate(tok_list):
        L = min(len(t), max_len)
        X[i, :L] = t[:L]
        M[i, :L] = True
    return X, M

t0 = time.time()
X_tr, M_tr = to_padded([token_lists[i] for i in idx_tr])
X_v,  M_v  = to_padded([token_lists[i] for i in idx_v])
X_te, M_te = to_padded([token_lists[i] for i in idx_te])
Y_tr, Y_v, Y_te = Y[idx_tr], Y[idx_v], Y[idx_te]
log(f"Padded in {time.time()-t0:.1f}s | X_tr {X_tr.shape} {X_tr.dtype}")


def augment_tokens(x: np.ndarray, m: np.ndarray):
    \"\"\"x, m are 1-D arrays of length MAX_OPS. Returns perturbed copies.\"\"\"
    x = x.copy(); m = m.copy()
    L = int(m.sum())
    if L <= 2:
        return x, m
    # Random PAD mask (keep CLS at position 0 intact)
    if AUG_TOKEN_MASK_PROB > 0:
        keep = np.random.rand(L - 1) > AUG_TOKEN_MASK_PROB
        for j in range(1, L):
            if not keep[j - 1]:
                x[j] = PAD_ID
                m[j] = False
    # Adjacent swap
    if AUG_TOKEN_SWAP_PROB > 0:
        j = 1
        while j < L - 1:
            if np.random.rand() < AUG_TOKEN_SWAP_PROB:
                x[j], x[j + 1] = x[j + 1], x[j]
                j += 2
            else:
                j += 1
    return x, m


class OpcodeDS(Dataset):
    def __init__(self, X, M, Y, augment: bool = False):
        self.X, self.M, self.Y, self.augment = X, M, Y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, m = self.X[i], self.M[i]
        if self.augment:
            x, m = augment_tokens(x, m)
        return (torch.from_numpy(x).long(),
                torch.from_numpy(m),
                torch.from_numpy(self.Y[i]))

ds_tr = OpcodeDS(X_tr, M_tr, Y_tr, augment=True)
ds_v  = OpcodeDS(X_v,  M_v,  Y_v, augment=False)
ds_te = OpcodeDS(X_te, M_te, Y_te, augment=False)
log(f"Datasets ready: train={len(ds_tr)} val={len(ds_v)} test={len(ds_te)} "
    f"(aug_mask={AUG_TOKEN_MASK_PROB}, aug_swap={AUG_TOKEN_SWAP_PROB})")
""")

# ---------------------------------------------------------------------------
md("""## 8 — Model with stochastic depth (DropPath)

Same architecture as dive-2 except every Transformer sub-layer's output is
multiplied by a Bernoulli mask at training time. The drop probability scales
linearly from `0` (layer 0) to `DROP_PATH=0.1` (last layer) following the
Stochastic Depth recipe from Huang et al. 2016.
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


class OpcodeCNNTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_classes=N_LABELS,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                 max_len=MAX_OPS, dropout=DROPOUT, drop_path=DROP_PATH):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.emb_norm  = nn.LayerNorm(d_model)
        self.emb_drop  = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(d_model)

        # Linearly scaled DropPath rates
        dp_rates = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            StochDepthEncoderLayer(d_model, n_heads, d_ff, dropout, dp_rates[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.attn_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

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
        mean_pool = (x * mf).sum(1) / mf.sum(1).clamp(min=1.0)

        s = self.attn_score(x).squeeze(-1)
        s = s.masked_fill(~attn_mask, -1e4)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        attn_pool = (x * w).sum(1)

        return self.head(torch.cat([mean_pool, attn_pool], dim=-1))


_m = OpcodeCNNTransformer()
n_params = sum(p.numel() for p in _m.parameters())
log(f"Parameters: {n_params/1e6:.2f} M  ({n_params:,})")
del _m; gc.collect()
""")

# ---------------------------------------------------------------------------
md("""## 9 — Asymmetric Loss

`L = -[(1-p_clip)^γ⁺ * log(p)]  on y=1`
`L = -[p_clip^γ⁻      * log(1-p_clip)] on y=0`,
where `p_clip = max(p − clip, 0)` is the probability-shifted prediction for the
negative class (Ridnik et al., *Asymmetric Loss For Multi-Label Classification*,
ICCV 2021). With `γ⁻ > γ⁺`, easy negatives are down-weighted and hard positives
keep their gradient — the standard fix for severe class imbalance.

Self-contained implementation; no dependency on `asymmetric-loss` package.
""")

code("""class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS, clip=ASL_CLIP, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # Compute in fp32 for numerical stability under AMP
        logits = logits.float()
        x_sig = torch.sigmoid(logits)
        xs_pos = x_sig
        xs_neg = 1.0 - x_sig
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w
        return -loss.mean()

criterion = AsymmetricLoss()
log(f"Loss: AsymmetricLoss(γ⁻={ASL_GAMMA_NEG}, γ⁺={ASL_GAMMA_POS}, clip={ASL_CLIP})")
""")

# ---------------------------------------------------------------------------
md("""## 10 — Exponential Moving Average (EMA)

Maintain a shadow copy of model weights, updated each step:
`ema_w ← decay * ema_w + (1 − decay) * w`. Evaluate and checkpoint on the EMA
copy — it averages out the late-epoch noise that produced the 0.636 / 0.659
zig-zag in the dive-2 log.

`decay=0.999` reaches effective stationarity in ~2000 steps (~ a few epochs).
""")

code("""class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        # Build a non-trained twin on the same device with weights copied
        self.decay = decay
        if isinstance(model, nn.DataParallel):
            src = model.module
        else:
            src = model
        self.ema = OpcodeCNNTransformer().to(next(src.parameters()).device)
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
md("""## 11 — Metrics + per-class threshold tuning""")

code("""from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             hamming_loss, accuracy_score, confusion_matrix)

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

def tune_thresholds(y_true, y_prob, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    thresholds = np.full(y_prob.shape[1], 0.5, dtype=np.float32)
    for i in range(y_prob.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds
""")

# ---------------------------------------------------------------------------
md("""## 12 — Optimizer, schedule, predict""")

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
    for X, M, Yb in loader:
        X = X.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)
        with autocast():
            logits = model(X, M)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(Yb.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)
""")

# ---------------------------------------------------------------------------
md("""## 13 — Dataloaders""")

code("""train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          drop_last=True, persistent_workers=NUM_WORKERS>0)
val_loader   = DataLoader(ds_v,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS>0)
test_loader  = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS>0)
log(f"Batch size: {BATCH_SIZE} ({BATCH_PER_GPU}/GPU × {max(1,n_gpus)} GPUs)")
log(f"Steps/epoch: {len(train_loader)}, total: {len(train_loader)*EPOCHS}")
""")

# ---------------------------------------------------------------------------
md("""## 14 — Checkpoint save / load (atomic, full state)

`save_checkpoint` dumps everything needed to resume bit-exactly:
`model`, `ema`, `optimizer`, `scheduler`, `scaler`, `epoch`,
`best_macro_f1`, `best_thresholds`, `history`, and the RNG states (python /
numpy / torch / cuda). The write goes to `.tmp` first and is renamed — a
process kill mid-write leaves the previous good checkpoint intact.

`load_checkpoint` restores all of the above. Re-call it **after** `model`,
`optimizer`, `scheduler`, `scaler`, and `ema` already exist, so we only need
their `state_dict`s here.
""")

code("""def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(path, *, model, ema, optimizer, scheduler, scaler,
                    epoch, best_macro_f1, best_thresholds, history, extra=None):
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
        "history":   history,
        "rng": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": {
            "MAX_OPS": MAX_OPS, "VOCAB_SIZE": VOCAB_SIZE,
            "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS, "D_FF": D_FF,
            "DROPOUT": DROPOUT, "DROP_PATH": DROP_PATH, "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "ASL_GAMMA_NEG": ASL_GAMMA_NEG, "ASL_GAMMA_POS": ASL_GAMMA_POS,
            "ASL_CLIP": ASL_CLIP, "EMA_DECAY": EMA_DECAY,
            "AUG_TOKEN_MASK_PROB": AUG_TOKEN_MASK_PROB,
            "AUG_TOKEN_SWAP_PROB": AUG_TOKEN_SWAP_PROB,
            "SEED": SEED,
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
    # Restore RNG so augmentation / dropout are reproducible across resumes
    rng = ckpt["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return ckpt
""")

# ---------------------------------------------------------------------------
md("""## 15 — Build model, optimiser, scheduler, scaler, EMA

Built first so they can be loaded into whether we're resuming or starting fresh.
""")

code("""model = OpcodeCNNTransformer().to(device)
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
md("""## 16 — Training loop

For each epoch:
1. **Train** — AMP, grad-clip, scheduler step *per batch*, EMA update *per
   batch*, periodic heartbeat to the log.
2. **Validate on the EMA copy** — predict, tune per-class thresholds on val,
   score at those thresholds.
3. **Append to `history`, flush `history.csv` and `history.json`**.
4. **Always** dump full state to `last_state.pt` (atomic). Also flush extra
   checkpoint if `CKPT_WALLCLOCK_SECS` has elapsed.
5. **If best**, also dump EMA weights + thresholds to `best_model.pt`.
6. **Early stop** if no improvement for `PATIENCE` consecutive epochs.

A `try/except/finally` wraps the whole loop. On any exception we still flush
`last_state.pt` before re-raising — so an OOM, CUDA error, or session timeout
leaves a clean resumable state.
""")

code("""HEARTBEAT_STEPS = 50

def _run_epoch(epoch):
    model.train()
    t0 = time.time()
    n_steps = len(train_loader)
    running_loss = 0.0
    running_n = 0

    for step, (X, M, Yb) in enumerate(train_loader, 1):
        X  = X.to(device, non_blocking=True)
        M  = M.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(X, M)
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
epoch_save_done = False    # tracks whether THIS iter has done its end-of-epoch save

try:
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_save_done = False
        # 1) Train
        train_loss, train_dt = _run_epoch(epoch)

        # 2) Validate on EMA
        val_logits, val_labels = predict(ema.ema, val_loader)
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_thr   = tune_thresholds(val_labels, val_probs)
        val_metrics, _ = multilabel_metrics(val_labels, val_probs, val_thr)
        cur_lr = optimizer.param_groups[0]["lr"]

        log(f"Epoch {epoch:2d}/{EPOCHS} | train_dt={train_dt:5.1f}s | "
            f"lr={cur_lr:.2e} | train_loss={train_loss:.4f} | "
            f"EMA val: f1_macro={val_metrics['f1_macro']:.4f} "
            f"f1_micro={val_metrics['f1_micro']:.4f} "
            f"ham={val_metrics['hamming_loss']:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "lr": cur_lr,
            "train_dt": train_dt, **val_metrics,
        })
        _flush_history()

        # 3) Always-save full state (atomic)
        save_checkpoint(STATE_PATH,
                        model=model, ema=ema,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch,
                        best_macro_f1=best_macro_f1,
                        best_thresholds=best_thresholds,
                        history=history)
        epoch_save_done = True

        # 4) Best-model save (EMA weights + tuned thresholds)
        if val_metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = val_metrics["f1_macro"]
            best_thresholds = val_thr.copy()
            _atomic_save({"model": ema.state_dict(),
                          "thresholds": best_thresholds,
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
                            history=history)
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
        # If this iter already saved successfully, last_state.pt is fine — don't downgrade it.
        # Otherwise, save with epoch-1 so resume redoes the failed epoch from scratch.
        if 'epoch' in locals() and not epoch_save_done:
            crash_epoch = max(0, epoch - 1)
            save_checkpoint(STATE_PATH,
                            model=model, ema=ema,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=crash_epoch,
                            best_macro_f1=best_macro_f1,
                            best_thresholds=best_thresholds,
                            history=history,
                            extra={"error": repr(e)})
            log(f"State flushed at epoch={crash_epoch} → {STATE_PATH}. "
                f"Set RESUME_FROM='{STATE_PATH}' and re-run; epoch {epoch} will be redone.")
        else:
            log(f"Most recent successful save in {STATE_PATH} preserved. "
                f"Set RESUME_FROM='{STATE_PATH}' and re-run to continue.")
    except Exception as e2:
        log(f"Also failed to flush state: {e2!r}")
    raise
finally:
    _flush_history()
""")

# ---------------------------------------------------------------------------
md("""## 17 — Test evaluation

Reload `best_model.pt` (EMA weights of the highest-val-macro-F1 epoch) and
its **frozen** per-class thresholds, evaluate once on test, dump everything.
We also report metrics at `threshold=0.5` to make the threshold-tuning effect
explicit and honest.
""")

code("""log("=== TEST EVAL ===")
if not BEST_PATH.exists():
    log("No best_model.pt found — falling back to current EMA weights.")
    eval_model = ema.ema
    frozen_thresholds = best_thresholds
else:
    ckpt_b = torch.load(BEST_PATH, map_location=device)
    eval_model = OpcodeCNNTransformer().to(device)
    eval_model.load_state_dict(ckpt_b["model"])
    frozen_thresholds = np.asarray(ckpt_b["thresholds"], dtype=np.float32)
    log(f"Loaded best model from epoch {ckpt_b.get('epoch','?')}: "
        f"val_f1_macro={ckpt_b.get('val_metrics',{}).get('f1_macro','?')}")

if n_gpus > 1 and not isinstance(eval_model, nn.DataParallel):
    eval_model = nn.DataParallel(eval_model)

test_logits, test_labels_arr = predict(eval_model, test_loader)
test_probs = 1 / (1 + np.exp(-test_logits))

test_metrics, per_class = multilabel_metrics(test_labels_arr, test_probs, frozen_thresholds)
metrics_at_05, _ = multilabel_metrics(test_labels_arr, test_probs, 0.5)

log("=== TEST METRICS (frozen per-class tuned thresholds) ===")
for k, v in test_metrics.items():
    log(f"  {k:>16s}: {v:.4f}")
log("=== TEST METRICS @ threshold=0.5 (reference) ===")
for k, v in metrics_at_05.items():
    log(f"  {k:>16s}: {v:.4f}")

pc_df = pd.DataFrame(per_class).T.round(4)
log("=== PER-CLASS (tuned thresholds) ===")
for line in pc_df.to_string().splitlines():
    log("  " + line)
""")

code("""# ── Artefact dump ───────────────────────────────────────────────────────────
np.save(OUT_DIR / "test_probs.npy",  test_probs)
np.save(OUT_DIR / "test_labels.npy", test_labels_arr)
np.save(OUT_DIR / "thresholds.npy",  frozen_thresholds)
pc_df.to_csv(OUT_DIR / "per_class.csv")

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "test_tuned":  test_metrics,
        "test_at_0.5": metrics_at_05,
        "per_class":   per_class,
        "best_val_f1_macro": best_macro_f1,
        "thresholds":  frozen_thresholds.tolist(),
        "hparams": {
            "max_ops": MAX_OPS, "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF,
            "dropout": DROPOUT, "drop_path": DROP_PATH,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "asl_gamma_neg": ASL_GAMMA_NEG, "asl_gamma_pos": ASL_GAMMA_POS, "asl_clip": ASL_CLIP,
            "ema_decay": EMA_DECAY,
            "aug_token_mask_prob": AUG_TOKEN_MASK_PROB,
            "aug_token_swap_prob": AUG_TOKEN_SWAP_PROB,
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

**What dive-3 changes vs dive-2 (architecture frozen, training upgraded).**

- Asymmetric Loss (`γ⁻=4, γ⁺=1, clip=0.05`) replaces BCE + clipped `pos_weight`.
- EMA of weights (decay 0.999) — validation and best-model selection use the EMA copy.
- Stochastic depth (DropPath, 0 → 0.1 linearly across the 6 Transformer layers).
- On-the-fly token augmentation (5 % PAD mask, 5 % adjacent swap) during training.
- `dropout 0.1 → 0.15`, `weight_decay 0.01 → 0.05`, `epochs 20 → 30` with patience-10 early-stop on EMA val macro-F1.

**Robustness.**

- Full-state checkpoint written **atomically** every epoch + every 30 min of wall-clock to `last_state.pt`. Restores model, EMA, optimizer, scheduler, scaler, RNG, epoch, history, best-metric.
- `try/except/finally` around the loop flushes state on any error before re-raising — set `RESUME_FROM = '/kaggle/working/last_state.pt'` and re-run.
- `tee` logger to `dive3_train.log` with per-step heartbeats every 50 steps, per-epoch summaries, and a full per-class table at test time. Survives "Save Version".

**Outputs (`/kaggle/working/`).**

- `best_model.pt` — EMA weights + frozen tuned thresholds of the best epoch.
- `last_state.pt` — full resumable state (always points at the most recent epoch).
- `state_epochNN.pt` — wall-clock snapshots every 30 min.
- `dive3_train.log` — full per-step / per-epoch / per-class transcript.
- `history.csv`, `history.json` — per-epoch metrics, flushed each epoch.
- `metrics.json`, `per_class.csv`, `confusion_per_class.json` — final test report.
- `test_probs.npy`, `test_labels.npy`, `thresholds.npy` — for downstream analysis / ensembling.
- `cache/dive3_tokens.npz` — cached disassembly (survives reruns).

**What to look at after the run.**

1. Open `history.csv` and plot `f1_macro` vs epoch. If it's monotonically higher than the dive-2 curve at every epoch, the training-side fixes worked.
2. Compare `metrics.json["test_tuned"]["f1_macro"]` to dive-2's `0.6592`. A gain of **+0.03 to +0.07** is the realistic expected range from these changes; anything noticeably above that and you've likely uncovered headroom we can push further (e.g. ASL hyper-param sweep). Anything below that on a clean run is signal-bound, not training-bound — next step is to add a modality (`dive-4`).
3. Read `per_class.csv` — if rare classes (`Front Running`, `Bad Randomness`) jumped most, that's the asymmetric-loss effect. If common classes also gained, EMA + DropPath are doing real work.
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

out = Path(__file__).parent / "dive-3.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
