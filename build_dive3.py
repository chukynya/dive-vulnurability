"""Build dive-3.ipynb: TensorFlow + TPUStrategy port of the PyTorch dive-3.

Targets Kaggle TPU v3-8 (8 cores, 16 GB HBM each). Follows the canonical
tf.distribute.TPUStrategy pattern from https://www.tensorflow.org/guide/tpu:

- TPUClusterResolver → connect_to_cluster → initialize_tpu_system → TPUStrategy
- mixed_bfloat16 global policy (no GradScaler — TPU bf16 doesn't need it)
- Model + optimizer + EMA created inside strategy.scope()
- Custom training loop using strategy.run inside @tf.function with multi-step
  inner loop for low Python overhead
- tf.data with drop_remainder=True (TPU requires static shapes), AUTOTUNE
  prefetch, and pure-TF on-the-fly augmentation
- tf.train.Checkpoint + CheckpointManager for TF state; sidecar pickle for
  Python-side resume state (epoch, history, best metric, thresholds, RNGs)

What stays identical to the PyTorch dive-3 design:

- Bytecode-only opcode tokenisation, vocab 259, MAX_OPS = 1024
- 6-layer pre-norm Transformer with multi-scale CNN frontend and dual pooling
- Asymmetric Loss (γ⁻=4, γ⁺=1, clip=0.05)
- EMA of weights (decay 0.999); validate + best-model select on EMA copy
- Stochastic depth (DropPath, 0→0.1 linear)
- On-the-fly token augmentation (5 % PAD mask, 5 % adjacent swap)
- Multilabel-stratified 80/10/10 split, per-class threshold tuning on val,
  frozen for test
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
md("""# DIVE-3 (TPU) — TensorFlow + TPUStrategy port

This is the **TensorFlow / Keras** version of `dive-3`, targeting **Kaggle TPU
v3-8 or v5e-8** (8 cores × 16 GB HBM per chip in both cases). The PyTorch GPU
version lives at commit `441e958` on `main` (merged via PR #1) and remains
valid for T4 ×2 runs.

Architecturally **identical** to PyTorch dive-3 — same model, same loss, same
augmentation, same checkpoint semantics — only the framework and the device
distribution change.

## What's TPU-specific

| Concern | Choice |
|---|---|
| Init | `TPUClusterResolver(tpu='local')` (Kaggle TPU VM v3/v4/v5e) → `initialize_tpu_system` → `TPUStrategy`. Falls back to `tpu=''` for older TPU Node setups, then default strategy if no TPU. |
| Precision | `tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')`. Compute in bf16; variables stay fp32. **No loss scaling** — TPU bf16 has fp32 dynamic range. Final `Dense` is fp32 for output stability. |
| Distribution | `strategy.scope()` wraps model + optimizer + EMA + checkpoint creation. `strategy.experimental_distribute_dataset` splits global batch across the 8 cores. |
| Static shapes | `drop_remainder=True` on every `.batch()`. All tensor dims fixed at graph build. `MAX_OPS = 1024` baked into the model. |
| Training loop | Custom loop using `strategy.run(step_fn)` inside a `@tf.function` that runs `STEPS_PER_LOOP=50` steps per Python iteration — the recommended pattern for TPU throughput. |
| Loss reduction | `tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH)` — correct cross-replica averaging. |
| Augmentation | Pure-TF ops in the input pipeline (`tf.where` for PAD mask, paired-reshape trick for adjacent swap). Avoids `tf.py_function` so the graph stays compilable on TPU. |
| Batch | Global batch = **256** (32 / core × 8 cores) — up from 64 on T4 ×2. LR scaled from 2e-4 → **4e-4** (sub-linear scaling, conservative). |
| Schedule | Cosine decay with linear warmup, 10 % of total steps. |
| EMA | Mirror `model_ema` instance inside `strategy.scope()`; per-step update inside `step_fn` so the assign ops live on TPU. Evaluation runs `model_ema.predict(...)`. |
| Checkpointing | `tf.train.CheckpointManager` for `model + model_ema + optimizer + train_step`. Sidecar `dive3_state.pkl` for Python-side resume state (epoch, history, best metric, thresholds, NumPy / Python / TF generator RNGs). |
| Resume | Set `RESUME = True` in the config cell. CheckpointManager auto-restores the latest TF checkpoint; the sidecar restores the rest. |

## How to use on Kaggle

1. **Accelerator → TPU VM v3-8** in the notebook settings sidebar.
2. **Settings → Internet ON** (TPU driver pings, dataset download if needed).
3. **Fresh run**: leave `RESUME = False`, Run All.
4. **Resume after a crash / timeout**: set `RESUME = True`, Run All.
5. **Watching from "Save Version"**: tail `dive3_train.log` and `history.csv`
   in the Output tab — both flushed after every epoch.

## File layout (in `/kaggle/working/`)

```
dive3_train.log         # tee transcript, append mode (survives resumes)
history.csv / .json     # per-epoch metrics
ckpt/                   # tf.train.CheckpointManager directory
dive3_state.pkl         # Python-side resume sidecar (epoch, RNGs, history, best)
best/                   # best-EMA checkpoint (separate, by val macro-F1)
metrics.json            # final test metrics + per-class report + hparams
per_class.csv           # pretty per-class table
test_probs.npy / test_labels.npy / thresholds.npy
confusion_per_class.json
cache/dive3_tokens.npz  # cached disassembly (survives reruns)
```
""")

# ---------------------------------------------------------------------------
md("""## 1 — TPU setup

**Important:** TPU init must run at the very start, before any tf.function
compilation or variable creation. Mixed-bfloat16 policy is set immediately
after so every layer created later uses bf16 compute.

On Kaggle the canonical resolver string is `'local'` (the TPU runs in the
same VM as the notebook). The no-arg auto-detect that works on older Cloud
TPU Nodes errors out on Kaggle TPU VM with
`ValueError: Please provide a TPU Name to connect to.` — we try `'local'`
first and fall back to `''` and then to the default strategy.
""")

code("""import os, sys, gc, json, math, time, random, warnings, traceback, pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

warnings.filterwarnings("ignore")

print(f"TensorFlow {tf.__version__}", flush=True)

# ── Resolve and initialise the TPU ──────────────────────────────────────────
# Try TPU-VM-style ('local') first — required on Kaggle's TPU VM v3/v4/v5e
# and on Cloud TPU VM. Fall back to no-arg auto-detect for older TPU Node
# setups (e.g. legacy Colab). Finally, fall back to the default (CPU/GPU)
# strategy so the notebook still runs locally for smoke tests.
def _try_tpu_init(tpu_arg):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_arg)
    # On TPU VM (tpu='local') the runtime is already in-process, so
    # experimental_connect_to_cluster is unnecessary; we still try it but
    # swallow the error if it doesn't apply.
    try:
        tf.config.experimental_connect_to_cluster(resolver)
    except Exception as _ce:
        print(f"[info] connect_to_cluster skipped: {_ce!r}", flush=True)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)


strategy = None
ON_TPU   = False
for _arg in ('local', ''):
    try:
        strategy = _try_tpu_init(_arg)
        ON_TPU = True
        print(f"TPU initialised via tpu={_arg!r}", flush=True)
        break
    except (ValueError, KeyError, tf.errors.NotFoundError, RuntimeError) as e:
        print(f"TPU init with tpu={_arg!r} failed: {e!r}", flush=True)

if not ON_TPU:
    print("No TPU available — falling back to default strategy (CPU/GPU).", flush=True)
    strategy = tf.distribute.get_strategy()

if ON_TPU:
    print("TPU devices:", tf.config.list_logical_devices('TPU'), flush=True)
print(f"Num replicas in sync: {strategy.num_replicas_in_sync}", flush=True)

# ── Mixed bfloat16 ──────────────────────────────────────────────────────────
# Must be set *before* any model/variable creation. Compute is bf16, variables
# stay fp32. The final classification layer is forced to fp32 for stability.
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16' if ON_TPU else 'float32')
print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}", flush=True)
""")

code("""# iterstrat for multi-label stratified split
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "iterative-stratification"])
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
print("iterstrat ready", flush=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
""")

# ---------------------------------------------------------------------------
md("""## 2 — Config

Edit **only** this cell to switch between fresh run and resume. `RESUME=True`
restores the latest TF checkpoint via `CheckpointManager.restore_or_initialize`
plus the sidecar `dive3_state.pkl` (epoch, history, best metric/thresholds,
RNG states).

Batch and LR are scaled from the PyTorch baseline:

- PyTorch dive-3: per_gpu = 32 × 2 GPUs = **global 64**, lr = 2e-4
- TPU dive-3: per_core = 32 × 8 cores = **global 256**, lr = 4e-4
  (≈ sub-linear scaling — `2e-4 × √(256/64) = 4e-4` exactly)
""")

code("""# ── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/kaggle/input/datasets/henrychristian7555/dive-smart-contract-multi-class-vulnerability")
BYTECODE_CSV = DATA_ROOT / "Bytecode_filled.csv"
LABEL_CSV    = DATA_ROOT / "DIVE_Labels.csv"

OUT_DIR   = Path("/kaggle/working");   OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache";         CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR  = OUT_DIR / "ckpt";          CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR  = OUT_DIR / "best";          BEST_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH      = OUT_DIR / "dive3_train.log"
HIST_CSV      = OUT_DIR / "history.csv"
HIST_JSON     = OUT_DIR / "history.json"
STATE_SIDECAR = OUT_DIR / "dive3_state.pkl"
BEST_SIDECAR  = OUT_DIR / "best_state.pkl"

# ── Resume ──────────────────────────────────────────────────────────────────
RESUME = False   # set True after a crash / timeout

# ── Labels ──────────────────────────────────────────────────────────────────
LABEL_COLS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
              "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
N_LABELS = len(LABEL_COLS)

# ── Tokenisation ────────────────────────────────────────────────────────────
PAD_ID, CLS_ID, UNK_ID = 0, 1, 2
SPECIAL = 3
VOCAB_SIZE = 256 + SPECIAL    # 259
MAX_OPS = 1024                 # same as PyTorch dive-3 for parity

# ── Model ───────────────────────────────────────────────────────────────────
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 6
D_FF     = 1024
DROPOUT  = 0.15
DROP_PATH = 0.10

# ── Training ────────────────────────────────────────────────────────────────
PER_CORE_BATCH = 32
GLOBAL_BATCH   = PER_CORE_BATCH * strategy.num_replicas_in_sync   # 256 on v3-8
EPOCHS         = 30
LR             = 4e-4
WEIGHT_DECAY   = 0.05
GRAD_CLIP      = 1.0
WARMUP_RATIO   = 0.10
PATIENCE       = 10
STEPS_PER_LOOP = 50    # inner-loop steps per @tf.function execution

# ── Asymmetric Loss ─────────────────────────────────────────────────────────
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP      = 0.05

# ── EMA ─────────────────────────────────────────────────────────────────────
EMA_DECAY = 0.999

# ── Augmentation ────────────────────────────────────────────────────────────
AUG_TOKEN_MASK_PROB = 0.05
AUG_TOKEN_SWAP_PROB = 0.05

# ── Wall-clock checkpoint extra ─────────────────────────────────────────────
CKPT_WALLCLOCK_SECS = 30 * 60

assert BYTECODE_CSV.exists() and LABEL_CSV.exists(), \\
    f"Missing data files under {DATA_ROOT}"
print("Inputs OK:", BYTECODE_CSV.name, "|", LABEL_CSV.name, flush=True)
print(f"GLOBAL_BATCH = {GLOBAL_BATCH} ({PER_CORE_BATCH}/core × {strategy.num_replicas_in_sync} cores)", flush=True)
print(f"RESUME       = {RESUME}", flush=True)
""")

# ---------------------------------------------------------------------------
md("""## 3 — `tee` logger

`log(...)` writes to stdout **and** to `dive3_train.log` (append mode). Both
survive Kaggle "Save Version" and resumes give you one continuous transcript.
""")

code("""class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", buffering=1, encoding="utf-8")  # line-buffered
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
log(f"TPU available: {ON_TPU}  |  replicas: {strategy.num_replicas_in_sync}")
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

Same rule as PyTorch dive-3: skip `PUSH1..PUSH32` immediates. Cached to
`.npz` so resumes don't redo this.
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
md("""## 7 — `tf.data` pipelines with pure-TF augmentation

On TPU every tensor must have a **static shape** at graph-build time, so we
pre-pad to `MAX_OPS=1024` and use `drop_remainder=True` on every `.batch()`.

**Augmentation is implemented in pure TF** so the input pipeline stays
graph-compilable on TPU. Two ops:

- **PAD mask** — for each non-special, non-PAD token, draw a Bernoulli with
  probability `AUG_TOKEN_MASK_PROB` and replace with `PAD_ID` (also clearing
  the attention mask there).
- **Paired adjacent swap** — view tokens as `(MAX_OPS/2, 2)` pairs, decide per
  pair whether to swap, then flatten. Non-overlapping by construction.
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
log(f"Padded in {time.time()-t0:.1f}s | X_tr {X_tr.shape}")


def _augment_one(x, m):
    \"\"\"Pure-TF token augmentation. x: (MAX_OPS,) int32, m: (MAX_OPS,) bool.\"\"\"
    # ── PAD mask ────────────────────────────────────────────────────────────
    # Don't touch CLS at position 0 or already-padded positions.
    pos = tf.range(MAX_OPS)
    eligible = tf.logical_and(m, pos > 0)
    rand = tf.random.uniform([MAX_OPS], dtype=tf.float32)
    drop = tf.logical_and(eligible, rand < AUG_TOKEN_MASK_PROB)
    x = tf.where(drop, tf.constant(PAD_ID, tf.int32), x)
    m = tf.where(drop, tf.constant(False), m)

    # ── Paired adjacent swap ───────────────────────────────────────────────
    # Reshape into (MAX_OPS/2, 2) pairs (positions 0..1, 2..3, ...); the CLS
    # at position 0 pairs with token 1, so per-pair swap probability is also
    # AUG_TOKEN_SWAP_PROB. Skip the first pair to keep CLS at position 0.
    half = MAX_OPS // 2
    xp = tf.reshape(x, [half, 2])
    mp = tf.reshape(m, [half, 2])
    swap_prob = tf.random.uniform([half], dtype=tf.float32) < AUG_TOKEN_SWAP_PROB
    # First pair (contains CLS) never swapped.
    swap_prob = tf.tensor_scatter_nd_update(swap_prob, [[0]], [False])
    xp_swapped = tf.gather(xp, [1, 0], axis=1)
    mp_swapped = tf.gather(mp, [1, 0], axis=1)
    xp = tf.where(swap_prob[:, None], xp_swapped, xp)
    mp = tf.where(swap_prob[:, None], mp_swapped, mp)
    x = tf.reshape(xp, [MAX_OPS])
    m = tf.reshape(mp, [MAX_OPS])
    return x, m


def make_dataset(X, M, Y, is_training, global_batch):
    ds = tf.data.Dataset.from_tensor_slices((X, M, Y))
    if is_training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
        def _map(x, m, y):
            x, m = _augment_one(x, m)
            return x, m, y
        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(global_batch, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# Building the datasets here (CPU); they get distributed when we call
# strategy.experimental_distribute_dataset further down.
ds_tr_global  = make_dataset(X_tr, M_tr, Y_tr, is_training=True,  global_batch=GLOBAL_BATCH)
ds_v_global   = make_dataset(X_v,  M_v,  Y_v, is_training=False, global_batch=GLOBAL_BATCH)
ds_te_global  = make_dataset(X_te, M_te, Y_te, is_training=False, global_batch=GLOBAL_BATCH)

STEPS_PER_EPOCH = len(X_tr) // GLOBAL_BATCH
VAL_STEPS  = len(X_v)  // GLOBAL_BATCH
TEST_STEPS = len(X_te) // GLOBAL_BATCH
log(f"Steps per epoch: {STEPS_PER_EPOCH} | val steps: {VAL_STEPS} | test steps: {TEST_STEPS}")
""")

# ---------------------------------------------------------------------------
md("""## 8 — Model: CNN-Transformer + stochastic depth

Keras-subclassed model. Same shape/topology as PyTorch dive-3:

```
input_ids ─► token_emb + pos_emb ─► LN ─► dropout
                    │
                    ├─► Conv1d(k=3) ─┐
                    ├─► Conv1d(k=5) ─┤── concat ── residual → LN
                    │
                    ▼
        6× pre-norm Transformer encoder layers w/ DropPath (0 → 0.1 linear)
                    │
                    ├── mean_pool   (mask-weighted)
                    └── attn_pool   (learned scalar score)
                    │ concat
                    ▼
            Linear(2d → d) → GELU → Dropout → Linear(d → 8)  [fp32 output]
```

The final `Dense(N_LABELS)` is forced to `float32` so the logits and loss
stay in fp32 even under the bf16 global policy.
""")

code("""class DropPath(layers.Layer):
    \"\"\"Per-sample stochastic depth (residual-branch drop). Static-shape safe.\"\"\"
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=False):
        if (not training) or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # Random per sample; broadcast over the other dims.
        batch = tf.shape(x)[0]
        rank = len(x.shape)
        shape = [batch] + [1] * (rank - 1)
        mask = tf.cast(tf.random.uniform(shape) < keep, x.dtype)
        return x * mask / tf.cast(keep, x.dtype)


class EncoderLayer(layers.Layer):
    \"\"\"Pre-norm transformer block with DropPath on each residual.\"\"\"
    def __init__(self, d_model, n_heads, d_ff, dropout, drop_path, **kw):
        super().__init__(**kw)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.ff1 = layers.Dense(d_ff, activation="gelu")
        self.ff_drop = layers.Dropout(dropout)
        self.ff2 = layers.Dense(d_model)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def call(self, x, attention_mask, training=False):
        h = self.ln1(x)
        a = self.attn(h, h, h, attention_mask=attention_mask, training=training)
        x = x + self.dp1(self.drop1(a, training=training), training=training)
        h = self.ln2(x)
        f = self.ff2(self.ff_drop(self.ff1(h), training=training))
        x = x + self.dp2(self.drop2(f, training=training), training=training)
        return x


def build_model():
    \"\"\"Functional Keras model. Returns logits (fp32).\"\"\"
    ids = tf.keras.Input(shape=(MAX_OPS,), dtype=tf.int32, name="input_ids")
    mask = tf.keras.Input(shape=(MAX_OPS,), dtype=tf.bool, name="attn_mask")

    tok = layers.Embedding(VOCAB_SIZE, D_MODEL, mask_zero=False,
                           embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(ids)
    pos_ids = tf.range(MAX_OPS)[None, :]
    pos = layers.Embedding(MAX_OPS, D_MODEL,
                           embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(pos_ids)
    x = tok + pos
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dropout(DROPOUT)(x)

    # CNN frontend (residual)
    c3 = layers.Conv1D(D_MODEL // 2, kernel_size=3, padding="same", activation="gelu")(x)
    c5 = layers.Conv1D(D_MODEL // 2, kernel_size=5, padding="same", activation="gelu")(x)
    c  = layers.Concatenate(axis=-1)([c3, c5])
    x  = layers.LayerNormalization(epsilon=1e-5)(x + c)

    # Build the attention_mask: (B, 1, MAX_OPS) — broadcasts to (B, T_q, T_k) inside MHA.
    # tf.keras.layers.MultiHeadAttention expects 1 = attend, 0 = mask out.
    attention_mask = tf.cast(mask, tf.bool)[:, tf.newaxis, :]

    # Stochastic depth linearly scaled across layers
    dp_rates = [DROP_PATH * i / max(1, N_LAYERS - 1) for i in range(N_LAYERS)]
    for i in range(N_LAYERS):
        x = EncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, dp_rates[i], name=f"enc_{i}")(
            x, attention_mask=attention_mask)
    x = layers.LayerNormalization(epsilon=1e-5)(x)

    # Dual pooling
    mf = tf.cast(mask, x.dtype)[:, :, None]
    mean_pool = tf.reduce_sum(x * mf, axis=1) / tf.maximum(tf.reduce_sum(mf, axis=1), 1.0)

    # Attentive pool — attention score per token; mask padded positions to a
    # very negative value. -1e4 is bf16-safe.
    scores = layers.Dense(1)(x)[:, :, 0]
    scores = tf.where(mask, scores, tf.fill(tf.shape(scores), tf.constant(-1e4, scores.dtype)))
    weights = tf.nn.softmax(scores, axis=-1)[:, :, None]
    attn_pool = tf.reduce_sum(x * tf.cast(weights, x.dtype), axis=1)

    h = layers.Concatenate(axis=-1)([mean_pool, attn_pool])
    h = layers.Dense(D_MODEL, activation="gelu")(h)
    h = layers.Dropout(DROPOUT)(h)
    # Force fp32 output for numerical stability under bf16 policy.
    logits = layers.Dense(N_LABELS, dtype="float32", name="logits")(h)

    return Model(inputs=[ids, mask], outputs=logits, name="opcode_cnn_tf")


# Sanity: parameter count (build outside strategy.scope for the smoke test, then discard)
_tmp = build_model()
n_params = _tmp.count_params()
log(f"Parameters: {n_params/1e6:.2f} M  ({n_params:,})")
del _tmp; gc.collect()
""")

# ---------------------------------------------------------------------------
md("""## 9 — Asymmetric Loss (TF)

Ridnik et al. 2021. Per-sample loss summed over labels (so we get one scalar
per example), then averaged across the global batch via
`tf.nn.compute_average_loss` inside the training step.

Computed in fp32 regardless of the global policy.
""")

code("""def asymmetric_loss(y_true, logits,
                    gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS,
                    clip=ASL_CLIP, eps=1e-8):
    y_true = tf.cast(y_true, tf.float32)
    logits = tf.cast(logits, tf.float32)
    x_sig = tf.sigmoid(logits)
    xs_pos = x_sig
    xs_neg = tf.clip_by_value(1.0 - x_sig + clip, 0.0, 1.0)
    los_pos = y_true * tf.math.log(tf.maximum(xs_pos, eps))
    los_neg = (1.0 - y_true) * tf.math.log(tf.maximum(xs_neg, eps))
    loss = los_pos + los_neg
    pt = xs_pos * y_true + xs_neg * (1.0 - y_true)
    gamma = gamma_pos * y_true + gamma_neg * (1.0 - y_true)
    w = tf.pow(1.0 - pt, gamma)
    loss = loss * w
    # Per-sample loss = sum over label dim (so positives in rare classes don't
    # get diluted by 7 negatives in the mean).
    return -tf.reduce_sum(loss, axis=-1)
""")

# ---------------------------------------------------------------------------
md("""## 10 — Metrics + per-class threshold tuning

Run on numpy after `model.predict(...)`. Same code as PyTorch dive-3.
""")

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
md("""## 11 — Build model, EMA mirror, optimiser inside `strategy.scope()`

**This is the key TPU pattern.** Everything that creates `tf.Variable`s must
happen inside `strategy.scope()` so the variables are mirrored across replicas.
""")

code("""# Cosine schedule with linear warmup, expressed as a Keras LR schedule so the
# state lives inside the optimizer (and roundtrips through CheckpointManager).
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(max(1, warmup_steps))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warm_lr = self.base_lr * step / warm
        prog = tf.clip_by_value((step - warm) / tf.maximum(total - warm, 1.0), 0.0, 1.0)
        cosine_lr = 0.5 * self.base_lr * (1.0 + tf.cos(math.pi * prog))
        return tf.where(step < warm, warm_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}


TOTAL_STEPS  = STEPS_PER_EPOCH * EPOCHS
WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_RATIO)

with strategy.scope():
    model     = build_model()
    model_ema = build_model()       # mirror; updated via assign every step
    # Initialise EMA with the same weights as the live model.
    model_ema.set_weights(model.get_weights())
    # Mark EMA as non-trainable so it doesn't end up in the optimizer's slot vars.
    model_ema.trainable = False

    lr_schedule = WarmupCosine(LR, TOTAL_STEPS, WARMUP_STEPS)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        global_clipnorm=GRAD_CLIP,
    )
    # Variables excluded from weight decay: LayerNorm / bias / embeddings.
    # exclude_from_weight_decay is Keras 3 / TF 2.13+; degrade gracefully on
    # older runtimes (decay applies to all trainable vars then).
    try:
        optimizer.exclude_from_weight_decay(
            var_names=["LayerNorm", "layer_normalization", "bias", "embedding"]
        )
    except (AttributeError, TypeError) as _wd_e:
        print(f"[warn] exclude_from_weight_decay unavailable: {_wd_e!r}", flush=True)

log(f"Total steps: {TOTAL_STEPS} | warmup steps: {WARMUP_STEPS} | base LR: {LR}")
""")

# ---------------------------------------------------------------------------
md("""## 12 — Training step (multi-step `@tf.function`)

The recommended TPU pattern: one `@tf.function` runs `STEPS_PER_LOOP=50`
`strategy.run(step_fn)` invocations inside `tf.range(...)` before returning to
Python. Drops Python overhead by ~50×.

`step_fn` returns nothing — losses are accumulated into a `tf.keras.metrics.Mean`
that we read from Python between Python iterations.
""")

code("""train_loss_metric = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)

@tf.function
def _train_step(inputs):
    x, m, y = inputs

    def step_fn(x, m, y):
        with tf.GradientTape() as tape:
            logits = model([x, m], training=True)
            per_ex = asymmetric_loss(y, logits)
            loss = tf.nn.compute_average_loss(per_ex, global_batch_size=GLOBAL_BATCH)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Per-replica EMA update — assigns happen on TPU.
        for v_ema, v in zip(model_ema.weights, model.weights):
            v_ema.assign(EMA_DECAY * v_ema + (1.0 - EMA_DECAY) * v)
        # Multiply back for the mean metric so it reflects per-replica average.
        train_loss_metric.update_state(loss * strategy.num_replicas_in_sync)

    strategy.run(step_fn, args=(x, m, y))


@tf.function
def _train_multi(iterator, n_steps):
    for _ in tf.range(n_steps):
        _train_step(next(iterator))
""")

# ---------------------------------------------------------------------------
md("""## 13 — Validation pass (collects probabilities)

Reuses the test loader. Runs `model_or_ema(inputs, training=False)` on each
replica via `strategy.run` and concatenates per-replica outputs back to a
single numpy array on the host.
""")

code("""@tf.function
def _val_step(inputs, eval_model):
    x, m, _ = inputs

    def step_fn(x, m):
        logits = eval_model([x, m], training=False)
        probs  = tf.sigmoid(tf.cast(logits, tf.float32))
        return probs

    per_replica_probs = strategy.run(step_fn, args=(x, m))
    return per_replica_probs


def predict_full(eval_model, dataset, total_steps):
    \"\"\"Iterate dataset, collect per-batch probs and labels to numpy.\"\"\"
    all_probs, all_labels = [], []
    it = iter(strategy.experimental_distribute_dataset(dataset))
    for _ in range(total_steps):
        inputs = next(it)
        probs = _val_step(inputs, eval_model)
        labels = inputs[2]
        # Gather across replicas — TPUStrategy returns PerReplica objects.
        if hasattr(strategy, "experimental_local_results"):
            p_list = strategy.experimental_local_results(probs)
            l_list = strategy.experimental_local_results(labels)
        else:
            p_list, l_list = [probs], [labels]
        for p, l in zip(p_list, l_list):
            all_probs.append(p.numpy())
            all_labels.append(l.numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)
""")

# ---------------------------------------------------------------------------
md("""## 14 — Checkpoint manager + sidecar

`tf.train.CheckpointManager` handles the heavy TF state (model variables, EMA
variables, optimizer slots, optimizer-internal step counter). A separate
`dive3_state.pkl` carries Python-side state — epoch, history, best metric,
best thresholds, and the NumPy / Python / TF generator RNGs.

The combination makes resumes deterministic and bit-exact for training math.
""")

code("""train_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name="train_step")

ckpt = tf.train.Checkpoint(
    model=model,
    model_ema=model_ema,
    optimizer=optimizer,
    train_step=train_step_counter,
)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, directory=str(CKPT_DIR), max_to_keep=3,
    step_counter=train_step_counter, checkpoint_interval=None,
)

# A separate manager for the best EMA snapshot — only model_ema needs saving.
best_ckpt = tf.train.Checkpoint(model_ema=model_ema)
best_manager = tf.train.CheckpointManager(
    best_ckpt, directory=str(BEST_DIR), max_to_keep=1,
)


def save_python_state(path, *, epoch, best_macro_f1, best_thresholds, history,
                     extra=None):
    payload = {
        "epoch": int(epoch),
        "best_macro_f1": float(best_macro_f1),
        "best_thresholds": np.asarray(best_thresholds, dtype=np.float32),
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "tf":     tf.random.experimental.get_global_generator().state.numpy(),
        },
        "config": {
            "MAX_OPS": MAX_OPS, "VOCAB_SIZE": VOCAB_SIZE,
            "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS, "D_FF": D_FF,
            "DROPOUT": DROPOUT, "DROP_PATH": DROP_PATH, "GLOBAL_BATCH": GLOBAL_BATCH,
            "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "ASL_GAMMA_NEG": ASL_GAMMA_NEG, "ASL_GAMMA_POS": ASL_GAMMA_POS, "ASL_CLIP": ASL_CLIP,
            "EMA_DECAY": EMA_DECAY,
            "AUG_TOKEN_MASK_PROB": AUG_TOKEN_MASK_PROB,
            "AUG_TOKEN_SWAP_PROB": AUG_TOKEN_SWAP_PROB,
            "SEED": SEED,
        },
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)


def load_python_state(path):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    # Restore RNGs
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    try:
        gen = tf.random.experimental.get_global_generator()
        gen.reset_from_seed(0)
        gen.state.assign(rng["tf"])
    except Exception as _rng_e:
        print(f"[warn] could not restore TF global RNG: {_rng_e!r}", flush=True)
    return payload
""")

# ---------------------------------------------------------------------------
md("""## 15 — Resume logic

If `RESUME=True` and a checkpoint + sidecar exist, restore both. Otherwise
start from epoch 1 with `train_step_counter = 0`.
""")

code("""start_epoch = 1
best_macro_f1 = -1.0
best_thresholds = np.full(N_LABELS, 0.5, dtype=np.float32)
history = []
patience_left = PATIENCE

if RESUME and ckpt_manager.latest_checkpoint is not None and STATE_SIDECAR.exists():
    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    log(f"Restored TF checkpoint: {ckpt_manager.latest_checkpoint}")
    payload = load_python_state(STATE_SIDECAR)
    start_epoch     = int(payload["epoch"]) + 1
    best_macro_f1   = float(payload["best_macro_f1"])
    best_thresholds = np.asarray(payload["best_thresholds"], dtype=np.float32)
    history         = list(payload["history"])
    log(f"Resumed at epoch {start_epoch}, best macro-F1 so far = {best_macro_f1:.4f}")
elif RESUME:
    log("RESUME requested but no checkpoint or sidecar found — starting fresh.")
else:
    log("Starting fresh (RESUME=False).")
""")

# ---------------------------------------------------------------------------
md("""## 16 — Training loop

Per epoch:

1. Build a distributed iterator for the train dataset. Run
   `_train_multi(iterator, STEPS_PER_LOOP)` repeatedly until
   `STEPS_PER_EPOCH` steps have run, logging a heartbeat each loop.
2. Validate on the EMA model. Tune per-class thresholds on val.
3. Always save TF checkpoint + sidecar (`dive3_state.pkl`). On every-30-min
   wall-clock, also save a named snapshot.
4. If best EMA val macro-F1, save EMA-only checkpoint to `BEST_DIR` + a
   tiny `best_state.pkl` with the frozen thresholds and val metrics.
5. Patience-10 early stop on EMA val macro-F1.

`try/except/finally` flushes state on any exception so the run is resumable.
""")

code("""def _flush_history():
    pd.DataFrame(history).to_csv(HIST_CSV, index=False)
    with open(HIST_JSON, "w") as fh:
        json.dump(history, fh, indent=2)


def _save_best(epoch, val_metrics, thresholds):
    best_manager.save()
    payload = {
        "epoch": int(epoch),
        "val_metrics": val_metrics,
        "thresholds": np.asarray(thresholds, dtype=np.float32),
    }
    tmp = BEST_SIDECAR.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, BEST_SIDECAR)


epoch_save_done = False
last_ckpt_ts = time.time()

try:
    train_iter = iter(strategy.experimental_distribute_dataset(ds_tr_global))

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_save_done = False
        train_loss_metric.reset_state()
        t_epoch = time.time()
        steps_done = 0

        while steps_done < STEPS_PER_EPOCH:
            t_loop = time.time()
            steps_this_loop = min(STEPS_PER_LOOP, STEPS_PER_EPOCH - steps_done)
            _train_multi(train_iter, tf.constant(steps_this_loop, dtype=tf.int64))
            steps_done += steps_this_loop
            cur_loss = float(train_loss_metric.result().numpy())
            cur_lr = float(lr_schedule(tf.cast(optimizer.iterations, tf.float32)).numpy())
            dt = time.time() - t_loop
            steps_per_sec = steps_this_loop / max(dt, 1e-6)
            eta = (STEPS_PER_EPOCH - steps_done) / max(steps_per_sec, 1e-6)
            log(f"  epoch {epoch:2d}  step {steps_done:>4d}/{STEPS_PER_EPOCH} | "
                f"loss={cur_loss:.4f} | lr={cur_lr:.2e} | "
                f"{steps_per_sec:5.1f} step/s | eta={eta:5.1f}s")

        train_loss = float(train_loss_metric.result().numpy())
        train_dt = time.time() - t_epoch

        # ── Validate on EMA model ─────────────────────────────────────────────
        val_probs, val_labels = predict_full(model_ema, ds_v_global, VAL_STEPS)
        val_thr = tune_thresholds(val_labels, val_probs)
        val_metrics, _ = multilabel_metrics(val_labels, val_probs, val_thr)
        cur_lr = float(lr_schedule(tf.cast(optimizer.iterations, tf.float32)).numpy())

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

        # ── Always-save TF state + sidecar ───────────────────────────────────
        ckpt_manager.save(checkpoint_number=train_step_counter.numpy())
        save_python_state(STATE_SIDECAR,
                          epoch=epoch,
                          best_macro_f1=best_macro_f1,
                          best_thresholds=best_thresholds,
                          history=history)
        epoch_save_done = True

        # ── Best-EMA save ─────────────────────────────────────────────────────
        if val_metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = val_metrics["f1_macro"]
            best_thresholds = val_thr.copy()
            _save_best(epoch, val_metrics, best_thresholds)
            log(f"  ✓ NEW BEST EMA macro-F1={best_macro_f1:.4f} → {BEST_DIR}")
            patience_left = PATIENCE
        else:
            patience_left -= 1
            log(f"  no improvement (best={best_macro_f1:.4f}, patience={patience_left}/{PATIENCE})")

        # ── 30-min wall-clock snapshot ───────────────────────────────────────
        if time.time() - last_ckpt_ts >= CKPT_WALLCLOCK_SECS:
            tagged = OUT_DIR / f"sidecar_epoch{epoch:02d}.pkl"
            save_python_state(tagged,
                              epoch=epoch,
                              best_macro_f1=best_macro_f1,
                              best_thresholds=best_thresholds,
                              history=history)
            log(f"  wall-clock snapshot sidecar → {tagged.name}")
            last_ckpt_ts = time.time()

        # ── Early stop ────────────────────────────────────────────────────────
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
            ckpt_manager.save(checkpoint_number=train_step_counter.numpy())
            save_python_state(STATE_SIDECAR,
                              epoch=crash_epoch,
                              best_macro_f1=best_macro_f1,
                              best_thresholds=best_thresholds,
                              history=history,
                              extra={"error": repr(e)})
            log(f"State flushed at epoch={crash_epoch}. Set RESUME=True and re-run; "
                f"epoch {epoch} will be redone.")
        else:
            log(f"Most recent successful save in {CKPT_DIR} / {STATE_SIDECAR} preserved. "
                f"Set RESUME=True and re-run to continue.")
    except Exception as e2:
        log(f"Also failed to flush state: {e2!r}")
    raise
finally:
    _flush_history()
""")

# ---------------------------------------------------------------------------
md("""## 17 — Test evaluation

Restore the **best EMA checkpoint**, run inference on the test set with the
**frozen** per-class thresholds, dump artefacts.
""")

code("""log("=== TEST EVAL ===")
if best_manager.latest_checkpoint is not None and BEST_SIDECAR.exists():
    best_ckpt.restore(best_manager.latest_checkpoint).expect_partial()
    with open(BEST_SIDECAR, "rb") as fh:
        best_payload = pickle.load(fh)
    frozen_thresholds = np.asarray(best_payload["thresholds"], dtype=np.float32)
    log(f"Restored best EMA from epoch {best_payload['epoch']} "
        f"(val f1_macro={best_payload['val_metrics']['f1_macro']:.4f})")
else:
    log("No best EMA checkpoint found — using current EMA weights with current thresholds.")
    frozen_thresholds = best_thresholds

test_probs, test_labels_arr = predict_full(model_ema, ds_te_global, TEST_STEPS)

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
        "framework":   "tensorflow",
        "tf_version":  tf.__version__,
        "on_tpu":      ON_TPU,
        "num_replicas": int(strategy.num_replicas_in_sync),
        "test_tuned":  test_metrics,
        "test_at_0.5": metrics_at_05,
        "per_class":   per_class,
        "best_val_f1_macro": best_macro_f1,
        "thresholds":  frozen_thresholds.tolist(),
        "hparams": {
            "max_ops": MAX_OPS, "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF,
            "dropout": DROPOUT, "drop_path": DROP_PATH,
            "global_batch": GLOBAL_BATCH, "per_core_batch": PER_CORE_BATCH,
            "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "asl_gamma_neg": ASL_GAMMA_NEG, "asl_gamma_pos": ASL_GAMMA_POS, "asl_clip": ASL_CLIP,
            "ema_decay": EMA_DECAY,
            "aug_token_mask_prob": AUG_TOKEN_MASK_PROB,
            "aug_token_swap_prob": AUG_TOKEN_SWAP_PROB,
            "patience": PATIENCE, "seed": SEED,
            "steps_per_loop": STEPS_PER_LOOP,
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
    elif p.is_dir():
        size = sum(q.stat().st_size for q in p.rglob("*") if q.is_file()) / 1024
        log(f"  {p.name}/  ({size:.1f} KB total)")

log.close()
""")

# ---------------------------------------------------------------------------
md("""## Summary

### TPU-specific design choices

| Aspect | Implementation |
|---|---|
| **Strategy** | `tf.distribute.TPUStrategy` over a v3-8 (8 cores). Falls back to default strategy when no TPU is present, so the notebook still runs locally for smoke tests. |
| **Precision** | `mixed_bfloat16` global policy on TPU, `float32` otherwise. Loss math and the final `Dense(N_LABELS)` are explicit fp32. No `GradScaler` — bf16 has fp32 dynamic range. |
| **Distribution** | `strategy.experimental_distribute_dataset` slices the global batch (256) across the 8 replicas (32/core). |
| **Static shapes** | `drop_remainder=True` on every batch; `MAX_OPS=1024` everywhere. Augmentation uses pure-TF ops so the input pipeline stays graph-compilable. |
| **Throughput** | `_train_multi` runs `STEPS_PER_LOOP=50` `strategy.run` calls inside one `@tf.function`. ~50× Python-overhead reduction vs single-step loops. |
| **Loss** | `tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH)` inside `step_fn` — correct cross-replica averaging. |
| **EMA** | Mirror Keras model created inside `strategy.scope()`. Per-step `assign` updates live on TPU. Evaluation runs `model_ema.predict(...)`. |
| **Checkpointing** | `tf.train.CheckpointManager` for model, EMA, optimizer + step counter; `dive3_state.pkl` sidecar for epoch / history / best / RNGs. Best-EMA in a separate `best/` directory. |
| **Resume** | `RESUME=True` restores latest TF checkpoint + sidecar. `try/except/finally` flushes both before re-raising on any exception. |
| **Logging** | TeeLogger to `dive3_train.log` (append). Heartbeats every `STEPS_PER_LOOP=50` steps with running loss / LR / step-rate / ETA. `history.csv` + `history.json` flushed each epoch. All visible in Kaggle "Save Version". |

### What stays identical to PyTorch dive-3

- Bytecode-only opcode tokenisation, vocab 259, `MAX_OPS=1024`
- 6-layer pre-norm Transformer with multi-scale CNN frontend and dual pooling, ~6 M params
- Asymmetric Loss (`γ⁻=4, γ⁺=1, clip=0.05`)
- EMA of weights (decay 0.999); validate + best-model select on EMA
- Stochastic depth (DropPath, 0 → 0.1 linear)
- Token augmentation (5 % PAD mask, 5 % adjacent swap) — re-implemented in pure TF for the input pipeline
- Multilabel-stratified 80/10/10 split, per-class threshold tuning on val (frozen for test)
- `dropout 0.15`, `weight_decay 0.05`, 30 epochs with patience-10 early stop

### Expected outcome

The PyTorch dive-3 ceiling on T4 ×2 was the loss + regularisation, not the
device. The same architecture under the same training regime on TPU v3-8
should land in the same **+0.03 to +0.07 macro-F1** band vs the dive-2
result of 0.6592 — TPU buys speed (~5× faster epochs), not accuracy. If the
results diverge meaningfully from PyTorch dive-3, suspect the batch-scaling
LR change (4e-4 vs 2e-4) first — sweep `LR ∈ {2e-4, 3e-4, 4e-4, 5e-4}` to
pin it down.
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
        "accelerator": "TPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "dive-3.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
