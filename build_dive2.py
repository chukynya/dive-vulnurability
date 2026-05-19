"""Build dive-2.ipynb: CNN-Transformer encoder for bytecode multi-label vuln detection."""
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
md("""# DIVE-2 — CNN-Transformer Encoder for Bytecode Vulnerability Detection

**Task.** Multi-label classification of EVM smart-contract bytecode into 8
vulnerability classes from the DIVE dataset.

**Inputs (bytecode-only).**
- `Bytecode_filled.csv` — `(contractID, contractAddress, bytecode)`
- `DIVE_Labels.csv` — `contractID` + 8 binary label columns

**Output target.** Per-class sigmoid probabilities for
`Reentrancy, Access Control, Arithmetic, Unchecked Return Values, DoS, Bad Randomness, Front Running, Time manipulation`.

**Hardware budget.** Kaggle T4 ×2 (2× 16 GB). Designed to leave headroom.

---

## Methodology — why this exact model

The dataset has these properties:

- **Sequence modality, small vocab.** ~150 EVM opcodes after disassembly. Order matters: `CALL → SSTORE` (reentrancy) and `BLOCKHASH → arith` (bad randomness) are local patterns; access-control / DoS need *global* reasoning.
- **Moderate scale.** 22,330 contracts × 8 labels. Big enough for a real encoder, too small for a giant Transformer trained from scratch with no inductive bias.
- **Severe class imbalance.** Some labels appear in thousands of contracts, others in a few hundred. Labels are non-mutually-exclusive.
- **Long inputs.** p90 opcode length is in the few thousands; we cap at 1024 (covers the majority and bounds attention cost).

### Alternatives considered and rejected

| Family | Why not chosen |
|---|---|
| TF-IDF + LR/XGBoost | Loses opcode order; cannot detect ordered patterns like `CALL → SSTORE`. Underestimates how much signal is in the data. |
| Pure TextCNN | Local n-grams only, no global context. Misses access-control reasoning across the whole runtime. |
| BiLSTM/BiGRU + attention | Sequential bottleneck, weakens past ~1 k tokens, hard to parallelise on 2 GPUs. Already covered by baseline B3. |
| Pure Transformer from scratch | No locality inductive bias. At 22 k samples it must *learn* "adjacent opcodes matter" — exactly the kind of bias-starvation that masks dataset quality. |
| Pretrained CodeBERT / GraphCodeBERT | Pretrained on *source code*, wrong modality for raw bytecode. No public EVM-bytecode pretraining. |
| GNN on CFG (GAT, MANDO, DR-GCN) | Adds a confound — CFG-builder quality — that contaminates the dataset-quality probe. Already covered by baseline B4. |
| Focal / asymmetric loss | 1–2 % gain at the cost of 2–3 hyper-parameters; harder to defend as a "standard" baseline across datasets. |

### What we chose — CNN-Transformer encoder

```
input_ids ─► token_emb + pos_emb ─► LN ─► dropout
                    │
                    ├─► Conv1d(k=3) ─┐
                    ├─► Conv1d(k=5) ─┤── concat ── residual → LN
                    │
                    ▼
        TransformerEncoder × 6 (d=256, h=8, d_ff=1024, pre-norm, GELU)
                    │
                    ├── mean_pool   (mask-weighted average)
                    └── attn_pool   (learned scalar score)
                    │ concat
                    ▼
            Linear(2d → d) → GELU → Dropout → Linear(d → 8)
```

| Choice | Reason |
|---|---|
| **Byte-level opcode tokens, PUSH-immediates skipped** | PUSH operand bytes are *data*, not control. Skipping them is the standard EVM disassembly trick and halves sequence length without losing control-flow signal. |
| **CNN frontend (k=3 and k=5)** | Established for opcode-sequence classification (ESCORT, CBGRU). Provides the locality bias that pure self-attention has to learn from limited data. |
| **Transformer encoder (6 × d=256 × h=8)** | Captures the global dependencies needed for access-control / DoS detection. At ~6 M total params this is ~3.7 k samples / M-param — well inside the safe zone for 22 k samples. |
| **Dual pooling (mean ⊕ attn)** | Mean is the stable baseline; attentive picks out rare critical opcodes. Concatenating both is robust across long and short contracts. |
| **BCEWithLogitsLoss + per-class `pos_weight` (clipped at 10)** | Standard, transparent multi-label loss. `pos_weight = neg/pos` rebalances rare classes without rescaling. Clip prevents pos_weight from blowing up on very rare classes. |
| **Per-class threshold tuning on val** | Multi-label F1 is highly threshold-sensitive. Tuning per class on val and **freezing for test** is honest and is the single biggest free win on this metric family. |
| **AMP fp16 + `nn.DataParallel`** | T4 has fast fp16. DataParallel works in-notebook with zero `torchrun` setup. With 6 M params × batch 32/GPU we sit at ~3 GB/T4 — generous margin. |
| **`MultilabelStratifiedShuffleSplit`** | Preserves joint label distribution across train/val/test. Random split distorts rare classes at this size. Already in `requirements.txt`. |

### Why this is a fair dataset probe

The model is **standard** (every block is PyTorch built-in), **strong** (locality + global context), **right-sized** (capacity matches sample budget), and **portable** (swap CSV → retrain on any bytecode-multi-label dataset). If a clean run produces weak metrics here but strong metrics on a comparable dataset, the difference is the dataset, not the model.
""")

code("""# ── 1. Environment ──────────────────────────────────────────────────────────
import os, gc, json, math, time, random, warnings
from pathlib import Path

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
print(f"Torch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB")
""")

code("""# ── 2. Install iterstrat for multi-label stratified split ───────────────────
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "iterative-stratification"])
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
print("iterstrat ready")
""")

code("""# ── 3. Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/datasets/henrychristian7555/dive-smart-contract-multi-class-vulnerability")
BYTECODE_CSV = DATA_ROOT / "Bytecode_filled.csv"
LABEL_CSV    = DATA_ROOT / "DIVE_Labels.csv"

OUT_DIR   = Path("/kaggle/working"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache";       CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
              "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
N_LABELS = len(LABEL_COLS)

assert BYTECODE_CSV.exists() and LABEL_CSV.exists(), \\
    f"Missing data files under {DATA_ROOT}"
print("Inputs OK:", BYTECODE_CSV.name, "|", LABEL_CSV.name)
""")

# ---------------------------------------------------------------------------
md("""## Step 1 — Load labels and bytecode

`Bytecode_filled.csv` is ~330 MB; loading only the two columns we need
(`contractID`, `bytecode`) keeps RAM around 1 GB.
""")

code("""labels_df = pd.read_csv(LABEL_CSV)
print("Labels:", labels_df.shape)
print(labels_df[LABEL_COLS].sum().to_string())
""")

code("""bc_df = pd.read_csv(BYTECODE_CSV, usecols=["contractID", "bytecode"])
print("Bytecode rows:", bc_df.shape)
bc_df = bc_df.dropna(subset=["bytecode"])
bc_df = bc_df[bc_df["bytecode"].str.len() > 4]  # need at least one opcode after 0x
print("After dropping empty bytecode:", bc_df.shape)

df = labels_df.merge(bc_df, on="contractID", how="inner").reset_index(drop=True)
print("Joined:", df.shape)
""")

# ---------------------------------------------------------------------------
md("""## Step 2 — EVM opcode disassembly

Walk the byte stream. For each opcode in `PUSH1..PUSH32` (0x60–0x7f), consume
the following 1–32 bytes as the immediate operand — **do not** emit them as
tokens. Token IDs are `opcode_byte + 3`, reserving 0/1/2 for `PAD`/`CLS`/`UNK`.
""")

code("""PAD_ID, CLS_ID, UNK_ID = 0, 1, 2
SPECIAL = 3
VOCAB_SIZE = 256 + SPECIAL  # 259

MAX_OPS = 1024  # sequence length cap (see length analysis below)

def disassemble(bc_hex: str, max_ops: int = MAX_OPS):
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
        if 0x60 <= op <= 0x7f:        # PUSH1..PUSH32 → skip immediate
            i += 1 + (op - 0x5f)
        else:
            i += 1
    return toks

# Sanity check on one contract.
demo = df["bytecode"].iloc[0]
print("Sample opcode IDs (first 30):", disassemble(demo, 30))
""")

code("""# Disassemble every contract once and cache to disk.
cache_path = CACHE_DIR / "dive2_tokens.npz"
if cache_path.exists():
    z = np.load(cache_path, allow_pickle=True)
    token_lists = list(z["tokens"])
    print(f"Loaded cached tokens for {len(token_lists)} contracts")
else:
    t0 = time.time()
    token_lists = [disassemble(bc, MAX_OPS) for bc in df["bytecode"].values]
    print(f"Disassembled {len(token_lists)} contracts in {time.time()-t0:.1f}s")
    np.savez_compressed(cache_path, tokens=np.array(token_lists, dtype=object))

lens = np.array([len(t) for t in token_lists])
print(f"Opcode-sequence length: median={int(np.median(lens))}, "
      f"p90={int(np.percentile(lens,90))}, p99={int(np.percentile(lens,99))}, "
      f"max={int(lens.max())}")
print(f"Fraction == MAX_OPS ({MAX_OPS}) (i.e. truncated): {(lens>=MAX_OPS).mean():.2%}")
""")

# ---------------------------------------------------------------------------
md("""## Step 3 — Multi-label stratified split

`MultilabelStratifiedShuffleSplit` preserves the joint label distribution
across train/val/test — critical because rare classes like `Front Running`
would otherwise concentrate in one fold under a random split.

Target: **80 / 10 / 10**.
""")

code("""Y = df[LABEL_COLS].values.astype(np.float32)

mss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
idx_trv, idx_te = next(mss1.split(np.zeros(len(df)), Y))

mss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
idx_tr_local, idx_v_local = next(mss2.split(np.zeros(len(idx_trv)), Y[idx_trv]))
idx_tr, idx_v = idx_trv[idx_tr_local], idx_trv[idx_v_local]

print(f"Train {len(idx_tr)} | Val {len(idx_v)} | Test {len(idx_te)}")
for i, lab in enumerate(LABEL_COLS):
    print(f"  {lab:>26s}: tr={Y[idx_tr,i].sum():>5.0f}  "
          f"v={Y[idx_v,i].sum():>4.0f}  te={Y[idx_te,i].sum():>4.0f}")
""")

# ---------------------------------------------------------------------------
md("""## Step 4 — `Dataset` and `DataLoader`

Pre-pad to `MAX_OPS` once into a contiguous `int32` array. ~90 MB of RAM, but
the worker loop becomes allocation-free and `pin_memory` is very fast.
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
print(f"Padded in {time.time()-t0:.1f}s | X_tr {X_tr.shape} {X_tr.dtype}")

class OpcodeDS(Dataset):
    def __init__(self, X, M, Y):
        self.X, self.M, self.Y = X, M, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]).long(),
                torch.from_numpy(self.M[i]),
                torch.from_numpy(self.Y[i]))

ds_tr = OpcodeDS(X_tr, M_tr, Y_tr)
ds_v  = OpcodeDS(X_v,  M_v,  Y_v)
ds_te = OpcodeDS(X_te, M_te, Y_te)
""")

# ---------------------------------------------------------------------------
md("""## Step 5 — Model

Configuration: `d_model=256`, `n_heads=8`, `n_layers=6`, `d_ff=1024`,
`dropout=0.1`. ~6 M parameters total.
""")

code("""class OpcodeCNNTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_classes=N_LABELS,
                 d_model=256, n_heads=8, n_layers=6, d_ff=1024,
                 max_len=MAX_OPS, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.emb_norm  = nn.LayerNorm(d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # CNN frontend: parallel k=3 and k=5 1-D convolutions, concatenated,
        # then added residually back to the embedding stream.
        self.conv3 = nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Dual pooling + classifier head
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

        # CNN frontend (residual)
        c = x.transpose(1, 2)
        c = torch.cat([F.gelu(self.conv3(c)), F.gelu(self.conv5(c))], dim=1)
        x = self.conv_norm(x + c.transpose(1, 2))

        # Transformer (True in src_key_padding_mask = padding)
        x = self.encoder(x, src_key_padding_mask=~attn_mask)
        x = self.final_norm(x)

        # Dual pooling
        mf = attn_mask.float().unsqueeze(-1)
        mean_pool = (x * mf).sum(1) / mf.sum(1).clamp(min=1.0)

        s = self.attn_score(x).squeeze(-1)
        s = s.masked_fill(~attn_mask, -1e4)        # fp16-safe
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        attn_pool = (x * w).sum(1)

        return self.head(torch.cat([mean_pool, attn_pool], dim=-1))

# Parameter count
_m = OpcodeCNNTransformer()
n_params = sum(p.numel() for p in _m.parameters())
print(f"Parameters: {n_params/1e6:.2f} M  ({n_params:,})")
del _m
""")

# ---------------------------------------------------------------------------
md("""## Step 6 — Metrics

Per-class F1 / precision / recall / ROC-AUC / AP, plus macro / micro / sample
F1, hamming loss, and exact-match accuracy.
""")

code("""from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             hamming_loss, accuracy_score, confusion_matrix)

def multilabel_metrics(y_true, y_prob, thresholds):
    \"\"\"thresholds: array of length n_labels, or a scalar.\"\"\"
    if np.isscalar(thresholds):
        thresholds = np.full(y_prob.shape[1], thresholds, dtype=np.float32)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    out = {
        "f1_micro":    f1_score(y_true, y_pred, average="micro",   zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_samples":  f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match":  accuracy_score(y_true, y_pred),
    }
    per_class = {}
    for i, lab in enumerate(LABEL_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        try:    auc = roc_auc_score(yt, y_prob[:, i])
        except ValueError: auc = float("nan")
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
    \"\"\"Per-class threshold that maximises F1 on the supplied (val) set.\"\"\"
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
md("""## Step 7 — Training loop

- **Optimizer.** AdamW, weight decay 0.01 on weight matrices, no decay on biases / LayerNorm.
- **Schedule.** Linear warmup over the first 10 % of steps, cosine decay to 0.
- **Loss.** `BCEWithLogitsLoss(pos_weight=neg/pos)` computed on the train fold, clipped to 10.
- **AMP.** `torch.cuda.amp` with `GradScaler`. Gradient clip at 1.0.
- **Multi-GPU.** `nn.DataParallel` if `torch.cuda.device_count() > 1`.
- **Selection.** Best macro-F1 on val (using **tuned per-class thresholds**) checkpointed to `best_model.pt`.
""")

code("""def build_optimizer(model, lr=2e-4, weight_decay=0.01):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8)

def cosine_warmup(optimizer, total_steps, warmup_ratio=0.1):
    warm = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_pos_weight(Y, clip=10.0):
    pos = Y.sum(0)
    neg = len(Y) - pos
    w = neg / np.clip(pos, 1.0, None)
    return torch.tensor(np.clip(w, 1.0, clip), dtype=torch.float32)

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

code("""# ── Hyper-parameters ────────────────────────────────────────────────────────
BATCH_PER_GPU = 32
BATCH_SIZE    = BATCH_PER_GPU * max(1, n_gpus)
EPOCHS        = 20
LR            = 2e-4
WEIGHT_DECAY  = 0.01
GRAD_CLIP     = 1.0
WARMUP_RATIO  = 0.10
NUM_WORKERS   = 2
PIN_MEMORY    = True

train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          drop_last=True, persistent_workers=NUM_WORKERS>0)
val_loader   = DataLoader(ds_v,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS>0)
test_loader  = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS>0)

print(f"Batch size: {BATCH_SIZE} ({BATCH_PER_GPU}/GPU × {max(1,n_gpus)} GPUs)")
print(f"Steps/epoch: {len(train_loader)}, total: {len(train_loader)*EPOCHS}")
""")

code("""# ── Build model, loss, optimizer, schedule ──────────────────────────────────
model = OpcodeCNNTransformer().to(device)
if n_gpus > 1:
    model = nn.DataParallel(model)

pos_weight = compute_pos_weight(Y_tr).to(device)
print("pos_weight:", {k: round(float(v), 2)
                      for k, v in zip(LABEL_COLS, pos_weight.cpu().numpy())})
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler   = cosine_warmup(optimizer, total_steps, WARMUP_RATIO)
scaler      = GradScaler()
""")

code("""# ── Train ───────────────────────────────────────────────────────────────────
history = []
best_macro_f1 = -1.0
best_path = OUT_DIR / "best_model.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time(); train_loss = 0.0; n_steps = 0
    for X, M, Yb in train_loader:
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

        train_loss += loss.item(); n_steps += 1

    train_loss /= max(1, n_steps)

    # Validation: predict, tune per-class thresholds on val, score at those thresholds.
    val_logits, val_labels = predict(model, val_loader)
    val_probs = 1 / (1 + np.exp(-val_logits))
    val_thr   = tune_thresholds(val_labels, val_probs)
    val_metrics, _ = multilabel_metrics(val_labels, val_probs, val_thr)

    dt = time.time() - t0
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:2d}/{EPOCHS} | {dt:5.1f}s | lr={cur_lr:.2e} | "
          f"train_loss={train_loss:.4f} | "
          f"val f1_macro={val_metrics['f1_macro']:.4f} "
          f"f1_micro={val_metrics['f1_micro']:.4f} "
          f"ham={val_metrics['hamming_loss']:.4f}")

    history.append({"epoch": epoch, "train_loss": train_loss,
                    "lr": cur_lr, **val_metrics})

    if val_metrics["f1_macro"] > best_macro_f1:
        best_macro_f1 = val_metrics["f1_macro"]
        state = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
        torch.save({"model": state, "thresholds": val_thr}, best_path)
        print(f"  ✓ saved best (val f1_macro={best_macro_f1:.4f}) → {best_path.name}")

print(f"\\nBest val macro-F1: {best_macro_f1:.4f}")
""")

# ---------------------------------------------------------------------------
md("""## Step 8 — Test evaluation and artefact dump

Reload best checkpoint **and frozen per-class thresholds**, score the test
fold once, write everything to `/kaggle/working/`.
""")

code("""ckpt = torch.load(best_path, map_location=device)
eval_model = OpcodeCNNTransformer().to(device)
eval_model.load_state_dict(ckpt["model"])
if n_gpus > 1:
    eval_model = nn.DataParallel(eval_model)
frozen_thresholds = np.asarray(ckpt["thresholds"], dtype=np.float32)

test_logits, test_labels_arr = predict(eval_model, test_loader)
test_probs = 1 / (1 + np.exp(-test_logits))

# Main report: frozen-threshold metrics
test_metrics, per_class = multilabel_metrics(test_labels_arr, test_probs, frozen_thresholds)

# Sanity: also compute @ threshold 0.5 to show the threshold-tuning effect
metrics_at_05, _ = multilabel_metrics(test_labels_arr, test_probs, 0.5)

print("=== TEST METRICS (per-class tuned thresholds) ===")
for k, v in test_metrics.items():
    print(f"  {k:>16s}: {v:.4f}")

print("\\n=== TEST METRICS @ threshold=0.5 (for reference) ===")
for k, v in metrics_at_05.items():
    print(f"  {k:>16s}: {v:.4f}")

print("\\n=== PER-CLASS (tuned thresholds) ===")
pc_df = pd.DataFrame(per_class).T.round(4)
print(pc_df.to_string())
""")

code("""# Save artefacts
np.save(OUT_DIR / "test_probs.npy",     test_probs)
np.save(OUT_DIR / "test_labels.npy",    test_labels_arr)
np.save(OUT_DIR / "thresholds.npy",     frozen_thresholds)
pc_df.to_csv(OUT_DIR / "per_class.csv")
with open(OUT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "test_tuned":  test_metrics,
        "test_at_0.5": metrics_at_05,
        "per_class":   per_class,
        "best_val_f1_macro": best_macro_f1,
        "thresholds":  frozen_thresholds.tolist(),
        "hparams": {
            "max_ops": MAX_OPS, "vocab_size": VOCAB_SIZE,
            "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024,
            "dropout": 0.1, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "grad_clip": GRAD_CLIP,
            "seed": SEED,
        }}, f, indent=2)

# Per-class confusion matrices (handy for error analysis)
cms = {}
y_pred_final = (test_probs >= frozen_thresholds[None, :]).astype(np.int32)
for i, lab in enumerate(LABEL_COLS):
    cm = confusion_matrix(test_labels_arr[:, i], y_pred_final[:, i], labels=[0, 1])
    cms[lab] = cm.tolist()
with open(OUT_DIR / "confusion_per_class.json", "w") as f:
    json.dump(cms, f, indent=2)

print("\\nArtefacts written to", OUT_DIR)
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")
""")

# ---------------------------------------------------------------------------
md("""## Summary

**Architecture.** Byte-level EVM opcode tokens → token + positional embedding →
multi-scale `Conv1d` frontend (residual) → 6-layer pre-norm Transformer
encoder (d=256, h=8, d_ff=1024) → mean ⊕ attentive pool → 2-layer head → 8 sigmoid logits.
~6 M parameters.

**Training.** AdamW (lr=2e-4, wd=0.01), cosine schedule with 10 % warmup,
BCE with per-class `pos_weight`, AMP fp16, `nn.DataParallel` over T4 ×2,
20 epochs.

**Selection.** Best val macro-F1 (under per-class threshold tuning),
thresholds frozen for test.

**Outputs (`/kaggle/working/`).**
- `best_model.pt` — weights + tuned thresholds
- `metrics.json` — test metrics (tuned and @0.5), per-class, hyper-parameters, thresholds
- `per_class.csv` — pretty per-class table
- `confusion_per_class.json` — per-class 2×2 confusion matrices
- `test_probs.npy`, `test_labels.npy`, `thresholds.npy` — for downstream analysis / ensembling
- `history.json` — per-epoch loss + val metrics
- `cache/dive2_tokens.npz` — cached disassembly

**Interpreting the result.** If macro-F1 plateaus below the rates you see on a
comparable bytecode-multi-label dataset under the *same* model, the bottleneck
is the dataset (label noise, weak class boundaries, contracts insufficiently
discriminative under the chosen labels). If it matches, the dataset is fine.
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

out = Path(__file__).parent / "dive-2.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
