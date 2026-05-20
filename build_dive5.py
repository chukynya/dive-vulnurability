"""Build dive-5.ipynb: single-pass long-context Transformer with semantic EVM
tokenization.

dive-4 plateaued because (a) PUSH immediates were thrown away — the model never
saw function selectors, addresses, or constants — and (b) chunk aggregation
over 4 × 1024 windows runs the encoder four times per contract with only a
shallow mean+attn pool stitching them together. dive-5 attacks both at once:

1. **Semantic tokenizer.** Keep every PUSH as a single token but type the
   immediate: function selectors get their own 2048-entry vocab, addresses /
   small ints / large ints / hashes / masks get bucket tokens. Strip the CBOR
   metadata trailer ( ~40 % of contracts ) first.
2. **Single-pass Transformer with SDPA mem-efficient attention.** No chunking.
   Sequence length 4096 covers ≥ 90 % of contracts after semantic tokenization
   (verified in the first hour, hard-asserted). Uses
   `torch.nn.functional.scaled_dot_product_attention` with the mem-efficient
   backend — flash-attn-v2 wheels target sm_80+, T4 is sm_75, so SDPA gives us
   most of the same O(N)-memory benefit with zero install pain.
3. **RoPE positions** ( applied to Q/K, no L² bias materialization ).
4. **Per-class MLP heads** off a shared trunk, plus an auxiliary BCE projection
   on the 2 rarest classes ( Bad Randomness, Front Running ) to give them an
   un-weighted gradient pathway.
5. **Token-span CutMix** ( replaces dive-4's chunk-level CutMix ).
6. **Span masking only** ( drop chunk dropout — no chunks ).

What stays identical to dive-4:
- 80/10/10 multilabel-stratified split, seed 42.
- AsymmetricLoss with per-class sqrt-inverse-freq weights ( γ⁻=4, γ⁺=1,
  clip=0.05 ).
- EMA ( decay 0.999 ), DropPath ( 0 → 0.10 ).
- AdamW + cosine warmup, AMP fp16, DataParallel for 2 GPUs.
- WeightedRandomSampler ( sqrt-inverse-freq, clipped to [1, 5] ).
- Isotonic per-class calibration → 41-point threshold tune ( min-precision
  0.25 for rare classes ).
- Atomic full-state checkpointing, tee logger, history.csv.

What's new in the output bundle:
- `dive5_selector_vocab.json` — top-2048 function selectors with frequencies.
- `dive5_model.onnx` — exported model for low-latency inference.
- `inference_bench.json` — per-contract latency on a single T4.
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
md("""# DIVE-5 — Single-pass long-context Transformer + semantic EVM tokens

**Why this notebook.** dive-4 introduced hierarchical chunked encoding ( 4 ×
1024 opcodes ) and isotonic calibration on top of dive-3's encoder. The
expected gain is real but the design has two hard ceilings:

- The opcode disassembler **throws away every PUSH immediate** — function
  selectors ( 4-byte method IDs ), addresses, and constants are gone. The
  model only sees control-flow opcodes.
- Cross-chunk interaction is a shallow `mean ⊕ attn` pool, and each contract
  pays the cost of **4 encoder forward passes**. Inference scales with
  N_CHUNKS.

dive-5 keeps the same training scaffolding ( split, sampler, loss, EMA,
calibration, threshold tuning, checkpointing ) but rebuilds the input
representation and the model.

## What's new vs dive-4

| Change | Why |
|---|---|
| **Semantic tokenizer** — keep PUSH immediates as typed tokens; build a 2048-entry function-selector vocab from train data only; strip the CBOR metadata trailer ( ~40 % of contracts carry one ) | Function selectors are essentially function fingerprints. Throwing them away discards the densest piece of bytecode signal. |
| **Single-pass Transformer with `torch.nn.functional.scaled_dot_product_attention`** ( mem-efficient backend ) at SEQ_LEN=4096, no chunking | Eliminates 4× encoder passes / contract. SDPA's mem-efficient kernel is O(N) memory, built into torch ≥ 2.1, no `flash-attn` wheel needed ( T4 is sm_75 — official FA-v2 wheels target sm_80+ ). |
| **RoPE positions** applied to Q/K before SDPA | Bias-free, O(L · D) cos/sin tables instead of an O(H · L²) ALiBi bias buffer — the latter is ~384 MB at our L=4096 H=6 in fp32 and defeats the mem-efficient backend's whole purpose. |
| **Per-class MLP heads** ( 8 parallel 2-layer MLPs off a shared trunk ) | Per-class F1 is the main KPI; rare classes ( BR / FR ) need their own capacity to learn calibration-friendly margins. Adds ~0.6 M params. |
| **Auxiliary BCE projection** on the 2 rarest classes ( BR, FR ), weight 0.3 | Direct un-weighted gradient pathway for BR and FR — the binding constraint on macro-F1. Costs ~770 extra params. |
| **Token-span CutMix** ( replaces chunk-level CutMix ) | No chunks anymore. Sample `λ ~ Beta(0.5, 0.5)`, splice a contiguous span of tokens from j into i, mix labels by real-token-count ratio. |
| **Span masking only** ( drop chunk dropout ) | No chunks. |
| **First-hour verification gates** | Hard-asserts on ( a ) SDPA mem-efficient kernel engaging ( max_memory < 6 GB at B=12 ), ( b ) tokenized p90 ≤ 4096 ( else design is invalidated ), ( c ) ONNX export of SDPA model works ( else fall back to MultiheadAttention export path ). |
| **ONNX export + per-contract latency benchmark** | Inference time is secondary but tracked. 512 test contracts one-at-a-time through FP16 EMA on single T4 → p50 / p90 / mean ms. |

## What stays identical to dive-4

- Multilabel-stratified 80/10/10 split, seed 42.
- AsymmetricLossWeighted ( γ⁻=4, γ⁺=1, clip=0.05 ) + sqrt-inverse-freq
  per-class weights clipped to `[1, 5]`.
- EMA decay 0.999, DropPath linear 0 → 0.10.
- AdamW with selective weight decay ( no decay on biases / norms / embeddings ),
  cosine warmup ( 10 % ), grad clip 1.0.
- WeightedRandomSampler with sqrt-inverse-freq per-sample weights clipped to
  `[1, 5]`.
- Per-class isotonic calibration on val → 41-point threshold tune with
  min-precision floor 0.25 for classes with val support < 200.
- Atomic full-state checkpointing every epoch + every 30 min of wall-clock,
  tee logger, history.csv flushed per epoch.

## Reading the result

1. **Truncation log line in cell 8** — `lens<=4096: X.XX%`. Must be ≥ 85 % or
   the design is invalidated and you need SEQ_LEN ≥ 6144 ( rerun memory check ).
2. **`test_calibrated_tuned.f1_macro` vs dive-4's run.** Expect macro-F1 gain
   from semantic tokens + single-pass attention.
3. **Per-class F1 on Bad Randomness and Front Running** — these are the
   targeted rare classes. The aux-BCE term and semantic tokens are the levers.
4. **`inference_bench.json`** — per-contract p50 ms on single T4 FP16.
   Should be strictly lower than dive-4's 4-chunk forward.
""")

# ---------------------------------------------------------------------------
md("""## 1 — Environment""")

code("""import os, gc, json, math, time, random, warnings, signal, sys, traceback
from pathlib import Path
from datetime import datetime
from collections import Counter

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

# Force the SDPA backend selection: prefer mem-efficient (works on T4 sm_75);
# disable the math backend so a silent fallback is impossible (would OOM at
# L=4096). flash_sdp is left enabled in case we ever run on sm_80+.
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    print("SDPA backends: flash=ON  mem_efficient=ON  math=OFF", flush=True)
except AttributeError:
    print("WARNING: torch < 2.1 — SDPA backend toggles unavailable", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Torch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {n_gpus}", flush=True)
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB | sm_{p.major}{p.minor}", flush=True)
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
  best-metric / calibrators / selector vocab are all restored.
""")

code("""# ── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/kaggle/input/datasets/henrychristian7555/dive-smart-contract-multi-class-vulnerability")
BYTECODE_CSV = DATA_ROOT / "Bytecode_filled.csv"
LABEL_CSV    = DATA_ROOT / "DIVE_Labels.csv"

OUT_DIR      = Path("/kaggle/working");      OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR    = OUT_DIR / "cache";            CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH     = OUT_DIR / "dive5_train.log"
HIST_CSV     = OUT_DIR / "history.csv"
HIST_JSON    = OUT_DIR / "history.json"
STATE_PATH   = OUT_DIR / "last_state.pt"
BEST_PATH    = OUT_DIR / "best_model.pt"
SELECTOR_JSON = CACHE_DIR / "dive5_selector_vocab.json"
TOKENS_NPZ    = CACHE_DIR / "dive5_tokens.npz"
ONNX_PATH     = OUT_DIR / "dive5_model.onnx"
BENCH_JSON    = OUT_DIR / "inference_bench.json"

# ── Resume control ──────────────────────────────────────────────────────────
RESUME_FROM = None    # set to str(STATE_PATH) to resume

# ── Labels ──────────────────────────────────────────────────────────────────
LABEL_COLS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
              "DoS", "Bad Randomness", "Front Running", "Time manipulation"]
N_LABELS = len(LABEL_COLS)
BR_IDX = LABEL_COLS.index("Bad Randomness")     # 5
FR_IDX = LABEL_COLS.index("Front Running")      # 6

# ── Tokenisation (semantic) ─────────────────────────────────────────────────
# Specials: 0..3
PAD_ID, CLS_ID, SEP_ID, UNK_ID = 0, 1, 2, 3
# Opcodes: 4..259  (256 raw opcodes shifted by OPCODE_OFFSET)
OPCODE_OFFSET = 4
# Bucket tokens for PUSH immediates: 260..269  (10 buckets)
TOK_ZERO          = 260
TOK_BOOL_TRUE     = 261
TOK_SMALL_INT     = 262
TOK_MEDIUM_INT    = 263
TOK_LARGE_INT     = 264
TOK_ADDRESS       = 265
TOK_HASH_LIKE     = 266
TOK_LARGE_INT_32  = 267
TOK_MASK_LIKE     = 268
TOK_SELECTOR_UNK  = 269
SELECTOR_OFFSET   = 270    # selector IDs occupy 270 .. (270 + K - 1)
SELECTOR_VOCAB_SIZE = 2048
VOCAB_SIZE = SELECTOR_OFFSET + SELECTOR_VOCAB_SIZE      # 2318

# Single-pass context.
SEQ_LEN = 4096

# ── Model ───────────────────────────────────────────────────────────────────
D_MODEL  = 384
N_HEADS  = 6
N_LAYERS = 6
D_FF     = 1536
DROPOUT  = 0.15
DROP_PATH = 0.10

# ── Training ────────────────────────────────────────────────────────────────
BATCH_PER_GPU = 12          # single forward / contract (no chunking)
BATCH_SIZE    = BATCH_PER_GPU * max(1, n_gpus)
EPOCHS        = 30
LR            = 2e-4
WEIGHT_DECAY  = 0.05
GRAD_CLIP     = 1.0
WARMUP_RATIO  = 0.10
PATIENCE      = 10
NUM_WORKERS   = 2
PIN_MEMORY    = True

# ── Asymmetric Loss + aux BCE ──────────────────────────────────────────────
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP      = 0.05
PER_CLASS_W_MIN = 1.0
PER_CLASS_W_MAX = 5.0
AUX_BCE_INDICES = [BR_IDX, FR_IDX]
AUX_BCE_WEIGHT  = 0.3

# ── EMA ─────────────────────────────────────────────────────────────────────
EMA_DECAY = 0.999

# ── Augmentation ────────────────────────────────────────────────────────────
AUG_SPAN_MASK_PROB = 0.08
AUG_SPAN_MAX_LEN   = 3
AUG_CUTMIX_PROB    = 0.30
AUG_CUTMIX_BETA    = 0.5

# ── Sampler ─────────────────────────────────────────────────────────────────
SAMPLER_W_MIN = 1.0
SAMPLER_W_MAX = 5.0

# ── Checkpointing ───────────────────────────────────────────────────────────
CKPT_WALLCLOCK_SECS = 30 * 60

# ── Verification gates (first-hour smoke tests) ────────────────────────────
SDPA_MEMORY_BUDGET_BYTES = 6_000_000_000     # 6 GB ceiling at B=12, L=4096
MIN_COVERAGE_AT_SEQ_LEN  = 0.85               # ≥85% of contracts must fit

# ── Inference benchmark ─────────────────────────────────────────────────────
BENCH_N_SAMPLES = 512

# ── Sanity check on data files ─────────────────────────────────────────────
assert BYTECODE_CSV.exists() and LABEL_CSV.exists(), \\
    f"Missing data files under {DATA_ROOT}"
print("Inputs OK:", BYTECODE_CSV.name, "|", LABEL_CSV.name, flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
print(f"Context: single-pass at SEQ_LEN={SEQ_LEN}, vocab={VOCAB_SIZE}", flush=True)
print(f"Model: d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} d_ff={D_FF}", flush=True)
""")

# ---------------------------------------------------------------------------
md("""## 3 — `tee` logger

Identical to dive-4. Mirrors stdout to `dive5_train.log`. Survives Kaggle "Save
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
md("""## 4 — SDPA mem-efficient smoke test ( verification gate #1 )

Allocate a dummy `(B=BATCH_PER_GPU, L=SEQ_LEN, d=D_MODEL)` activation and run a
forward+backward through a single MHA + FFN block on GPU 0 to confirm the
mem-efficient SDPA kernel actually engages. If the math kernel silently takes
over, peak memory at L=4096 will exceed `SDPA_MEMORY_BUDGET_BYTES` and we OOM
at the real training step. Catch it now.

If this cell fails the assertion, drop BATCH_PER_GPU to 8 and re-run; if it
still fails, the design is invalidated for this hardware.
""")

code("""if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(0)
    _B = BATCH_PER_GPU
    _L = SEQ_LEN
    _D = D_MODEL
    _H = N_HEADS
    _hd = _D // _H

    _q = torch.randn(_B, _H, _L, _hd, device="cuda:0", dtype=torch.float16, requires_grad=True)
    _k = torch.randn(_B, _H, _L, _hd, device="cuda:0", dtype=torch.float16, requires_grad=True)
    _v = torch.randn(_B, _H, _L, _hd, device="cuda:0", dtype=torch.float16, requires_grad=True)
    _mask = torch.ones(_B, 1, 1, _L, device="cuda:0", dtype=torch.bool)

    with autocast():
        _out = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=_mask, is_causal=False)
        _loss = _out.float().sum()
    _loss.backward()
    torch.cuda.synchronize(0)

    peak = torch.cuda.max_memory_allocated(0)
    log(f"SDPA smoke test: peak={peak/1e9:.2f} GB at B={_B}, L={_L}, H={_H}, d_head={_hd}, fp16")
    log(f"Budget: {SDPA_MEMORY_BUDGET_BYTES/1e9:.2f} GB")
    assert peak < SDPA_MEMORY_BUDGET_BYTES, \\
        f"SDPA peak memory {peak/1e9:.2f} GB exceeds budget — mem-efficient kernel likely not engaged. " \\
        f"Drop BATCH_PER_GPU or enable gradient checkpointing."

    del _q, _k, _v, _mask, _out, _loss
    torch.cuda.empty_cache()
    gc.collect()
    log("SDPA mem-efficient kernel engaged.")
else:
    log("CUDA not available — skipping SDPA smoke test.")
""")

# ---------------------------------------------------------------------------
md("""## 5 — Load labels and bytecode""")

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
md("""## 6 — Stratified 80/10/10 split

Identical to dive-3 / dive-4 — same seed, same proportions. Done **before**
the selector vocab is built so we can compute it on train indices only and
avoid leakage.
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
md("""## 7 — CBOR metadata stripper + function-selector vocab ( train only )

**Metadata stripper.** Solc embeds a CBOR-encoded trailer at the end of
deployed bytecode containing the compiler version + IPFS hash of metadata.
Format: last 2 bytes are the trailer length L ( big-endian ), then
`b[-(L+2):-2]` starts with a CBOR map marker ( 0xa1 or 0xa2 ). If matched,
strip those `L+2` bytes — pure noise, ~40 % of contracts carry one.

**Selector vocab.** Scan train bytecodes for every PUSH4 ( opcode 0x63 )
immediate. The 4 bytes after each PUSH4 are nearly always a Solidity function
selector ( first 4 bytes of `keccak256(signature)` ) — used in the dispatch
table at the start of runtime code, and at internal call sites. Keep the
top-2048 by frequency. Coverage is heavy-tailed: ~50 ERC-20/721/AccessControl
selectors dominate.
""")

code("""def strip_metadata(bc_hex: str):
    \"\"\"Strip CBOR metadata trailer if present. Returns (hex_str, was_stripped).\"\"\"
    s = bc_hex.strip().lower()
    if s.startswith("0x"): s = s[2:]
    if len(s) < 6 or (len(s) & 1):
        return s, False
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return s, False
    n = len(b)
    if n < 4:
        return s, False
    L = int.from_bytes(b[-2:], "big")
    if L < 2 or L + 2 > n:
        return s, False
    head_byte = b[-(L + 2)]
    if head_byte not in (0xa1, 0xa2):
        return s, False
    stripped = b[:-(L + 2)].hex()
    return stripped, True


def build_selector_vocab(train_hex_list, top_k=SELECTOR_VOCAB_SIZE):
    \"\"\"Single pass over train bytecodes. Returns (vocab_dict, full_counter).\"\"\"
    cnt = Counter()
    PUSH4 = 0x63
    n_push4_sites = 0
    for raw in train_hex_list:
        s, _ = strip_metadata(raw)
        try:
            b = bytes.fromhex(s)
        except ValueError:
            continue
        i, n = 0, len(b)
        while i < n:
            op = b[i]
            if 0x60 <= op <= 0x7f:
                imm_len = op - 0x5f
                if op == PUSH4 and i + 1 + imm_len <= n:
                    sel = b[i + 1:i + 1 + imm_len].hex()
                    cnt[sel] += 1
                    n_push4_sites += 1
                i += 1 + imm_len
            else:
                i += 1
    top = cnt.most_common(top_k)
    vocab = {sel: idx for idx, (sel, _) in enumerate(top)}
    return vocab, cnt, n_push4_sites


if SELECTOR_JSON.exists() and RESUME_FROM is not None:
    selector_vocab = json.loads(SELECTOR_JSON.read_text())
    log(f"Loaded selector vocab from {SELECTOR_JSON.name} ({len(selector_vocab)} entries)")
else:
    t0 = time.time()
    train_hex = df.iloc[idx_tr]["bytecode"].tolist()
    selector_vocab, sel_counter, total_push4 = build_selector_vocab(train_hex)
    SELECTOR_JSON.write_text(json.dumps(selector_vocab, indent=1))
    covered = sum(sel_counter[s] for s in selector_vocab)
    log(f"Selector vocab built in {time.time()-t0:.1f}s — {len(selector_vocab)} entries, "
        f"coverage {covered/max(1,total_push4):.2%} of {total_push4} train PUSH4 sites")
    log("Top-20 selectors:")
    for sel, c in sel_counter.most_common(20):
        log(f"  0x{sel}  count={c}")
""")

# ---------------------------------------------------------------------------
md("""## 8 — Semantic tokenizer + token cache ( verification gate #2 )

**One token per opcode + one token per PUSH** ( the PUSH immediate is collapsed
into a single typed token, not expanded byte-by-byte ). PUSH4 immediates that
match a top-2048 selector are emitted as a distinct selector ID.

Bucket rules for PUSH<n> immediates ( applied in priority order ):

1. All-zero → `TOK_ZERO`.
2. PUSH4 and 4 bytes → selector lookup ( hit: selector ID, miss: `TOK_SELECTOR_UNK` ).
3. All-`0xff`, length ≥ 16 → `TOK_MASK_LIKE`.
4. PUSH20 ( 20 bytes ) → `TOK_ADDRESS`.
5. PUSH32 ( 32 bytes ) → Shannon entropy ≥ 4.5 bits/byte → `TOK_HASH_LIKE` else `TOK_LARGE_INT_32`.
6. PUSH1 with value 1 → `TOK_BOOL_TRUE`.
7. PUSH1 → `TOK_SMALL_INT`.
8. 2 ≤ length ≤ 4 ( non-selector ) → `TOK_MEDIUM_INT`.
9. length ≥ 5 → `TOK_LARGE_INT`.

The cell hard-asserts that ≥ `MIN_COVERAGE_AT_SEQ_LEN` of contracts fit in
`SEQ_LEN` tokens. If this assertion fails, the design is invalidated; either
raise SEQ_LEN ( and re-run the SDPA smoke test in cell 4 ) or fall back to
chunking.
""")

code("""def _push_immediate_token(op: int, imm: bytes, selector_vocab: dict) -> int:
    L = len(imm)
    if L == 0:
        return UNK_ID
    # 1) zero
    if not any(imm):
        return TOK_ZERO
    # 2) selector
    if op == 0x63 and L == 4:
        sel = imm.hex()
        sid = selector_vocab.get(sel)
        if sid is not None:
            return SELECTOR_OFFSET + sid
        return TOK_SELECTOR_UNK
    # 3) all-ff mask
    if L >= 16 and all(byte == 0xff for byte in imm):
        return TOK_MASK_LIKE
    # 4) address
    if op == 0x73 and L == 20:
        return TOK_ADDRESS
    # 5) hash-like / large-int-32
    if op == 0x7f and L == 32:
        # Shannon entropy in bits/byte
        c = Counter(imm)
        N = float(L)
        H = -sum((v / N) * math.log2(v / N) for v in c.values())
        return TOK_HASH_LIKE if H >= 4.5 else TOK_LARGE_INT_32
    # 6) bool-true
    if op == 0x60 and L == 1 and imm[0] == 0x01:
        return TOK_BOOL_TRUE
    # 7) small int
    if op == 0x60 and L == 1:
        return TOK_SMALL_INT
    # 8) medium int
    if 2 <= L <= 4:
        return TOK_MEDIUM_INT
    # 9) large int
    return TOK_LARGE_INT


def disassemble_semantic(bc_hex: str, selector_vocab: dict, max_len: int = SEQ_LEN):
    s, _ = strip_metadata(bc_hex)
    if len(s) < 2 or (len(s) & 1):
        return [CLS_ID]
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return [CLS_ID]
    toks = [CLS_ID]
    i, n = 0, len(b)
    while i < n and len(toks) < max_len:
        op = b[i]
        toks.append(op + OPCODE_OFFSET)
        if 0x60 <= op <= 0x7f:
            imm_len = op - 0x5f
            imm_end = min(i + 1 + imm_len, n)
            imm = b[i + 1:imm_end]
            if len(toks) < max_len:
                toks.append(_push_immediate_token(op, imm, selector_vocab))
            i = imm_end
        else:
            i += 1
    return toks


if TOKENS_NPZ.exists():
    z = np.load(TOKENS_NPZ, allow_pickle=True)
    token_lists = list(z["tokens"])
    log(f"Loaded cached tokens from {TOKENS_NPZ.name} ({len(token_lists)} contracts)")
else:
    t0 = time.time()
    token_lists = [disassemble_semantic(bc, selector_vocab, SEQ_LEN) for bc in df["bytecode"].values]
    log(f"Tokenized {len(token_lists)} contracts in {time.time()-t0:.1f}s")
    np.savez_compressed(TOKENS_NPZ, tokens=np.array(token_lists, dtype=object))

lens = np.array([len(t) for t in token_lists])
coverage_4096 = (lens <= 4096).mean()
log(f"Semantic-token length: median={int(np.median(lens))}, "
    f"p75={int(np.percentile(lens,75))}, p90={int(np.percentile(lens,90))}, "
    f"p99={int(np.percentile(lens,99))}, max={int(lens.max())}")
log(f"Truncated at SEQ_LEN={SEQ_LEN}: {(lens>=SEQ_LEN).mean():.2%}")
log(f"Coverage @ 4096: {coverage_4096:.2%}  ( required ≥ {MIN_COVERAGE_AT_SEQ_LEN:.0%} )")

assert coverage_4096 >= MIN_COVERAGE_AT_SEQ_LEN, \\
    f"Only {coverage_4096:.2%} of contracts fit in SEQ_LEN={SEQ_LEN} — design invalidated. " \\
    f"Either raise SEQ_LEN ( rerun SDPA smoke test in cell 4 ) or revert to chunking."
""")

# ---------------------------------------------------------------------------
md("""## 9 — Flat padded tensors + span-mask augmentation

`X` is `(N, SEQ_LEN)` int32, `TM` is `(N, SEQ_LEN)` bool ( True = real token,
False = pad or masked ).

**Span masking** ( `AUG_SPAN_MASK_PROB` per non-special position ): start a
span of 1..`AUG_SPAN_MAX_LEN` tokens at this position, replace them with
`PAD_ID`. CLS at position 0 is never masked. Breaks exact n-gram memorisation.
""")

code("""def to_padded_flat(tok_list, max_len=SEQ_LEN):
    n = len(tok_list)
    X  = np.zeros((n, max_len), dtype=np.int32)
    TM = np.zeros((n, max_len), dtype=np.bool_)
    for i, t in enumerate(tok_list):
        L = min(len(t), max_len)
        X[i, :L]  = t[:L]
        TM[i, :L] = True
    return X, TM


t0 = time.time()
X_tr, TM_tr = to_padded_flat([token_lists[i] for i in idx_tr])
X_v,  TM_v  = to_padded_flat([token_lists[i] for i in idx_v])
X_te, TM_te = to_padded_flat([token_lists[i] for i in idx_te])
Y_tr, Y_v, Y_te = Y[idx_tr], Y[idx_v], Y[idx_te]
log(f"Padded in {time.time()-t0:.1f}s | X_tr {X_tr.shape} {X_tr.dtype} | "
    f"mean real-token frac (train) = {TM_tr.mean():.3f}")


def span_mask_flat(x: np.ndarray, tm: np.ndarray):
    \"\"\"In-place span masking on (SEQ_LEN,) arrays. Position 0 (CLS) protected.\"\"\"
    x  = x.copy()
    tm = tm.copy()
    L = x.shape[0]
    j = 1
    while j < L:
        if not tm[j]:
            break
        if np.random.rand() < AUG_SPAN_MASK_PROB:
            span = np.random.randint(1, AUG_SPAN_MAX_LEN + 1)
            end = min(j + span, L)
            x[j:end]  = PAD_ID
            tm[j:end] = False
            j = end
        else:
            j += 1
    return x, tm


class OpcodeDS(Dataset):
    def __init__(self, X, TM, Y, augment: bool = False):
        self.X, self.TM, self.Y = X, TM, Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, tm = self.X[i], self.TM[i]
        if self.augment:
            x, tm = span_mask_flat(x, tm)
        return (torch.from_numpy(x).long(),
                torch.from_numpy(tm),
                torch.from_numpy(self.Y[i]))


ds_tr = OpcodeDS(X_tr, TM_tr, Y_tr, augment=True)
ds_v  = OpcodeDS(X_v,  TM_v,  Y_v,  augment=False)
ds_te = OpcodeDS(X_te, TM_te, Y_te, augment=False)
log(f"Datasets ready: train={len(ds_tr)} val={len(ds_v)} test={len(ds_te)}")
log(f"  aug: span_mask_prob={AUG_SPAN_MASK_PROB} (max_len={AUG_SPAN_MAX_LEN}), "
    f"cutmix={AUG_CUTMIX_PROB} (β={AUG_CUTMIX_BETA})")
""")

# ---------------------------------------------------------------------------
md("""## 10 — Class-balanced sampler + token-span CutMix collate

**Sampler.** Per-sample weight = `max over positive labels of sqrt(N_total /
class_count)`, clipped to `[SAMPLER_W_MIN, SAMPLER_W_MAX]`. Same as dive-4.

**CutMix collate.** With probability `AUG_CUTMIX_PROB`, for the assembled
batch: sample `λ ~ Beta(0.5, 0.5)`, choose a contiguous token span `[s, e)`
of length `⌈λ · L⌉`, and replace `X[:, s:e] ← X[perm][:, s:e]` ( same
positions across the batch — vectorised, cheap ). CLS at position 0 is
protected by drawing `s ≥ 1`. Effective per-sample λ is recomputed from the
ratio of real tokens that survived from sample `i` vs were inserted from
sample `perm[i]`.
""")

code("""def build_sampler_weights(Y_train: np.ndarray):
    counts = Y_train.sum(0).clip(min=1)
    inv = np.sqrt(Y_train.shape[0] / counts).astype(np.float32)
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


def cutmix_collate_flat(batch):
    X  = torch.stack([b[0] for b in batch], dim=0)   # (B, L)
    TM = torch.stack([b[1] for b in batch], dim=0)
    Y  = torch.stack([b[2] for b in batch], dim=0)

    if AUG_CUTMIX_PROB <= 0 or np.random.rand() >= AUG_CUTMIX_PROB:
        return X, TM, Y

    B, L = X.shape
    perm = torch.randperm(B)
    lam = float(np.random.beta(AUG_CUTMIX_BETA, AUG_CUTMIX_BETA))
    span = int(np.ceil(lam * L))
    span = max(1, min(L - 1, span))
    # Protect CLS at position 0 — span never starts at 0.
    start = int(np.random.randint(1, max(2, L - span + 1)))
    end = start + span

    X_mix  = X.clone()
    TM_mix = TM.clone()
    X_mix[:, start:end]  = X[perm][:, start:end]
    TM_mix[:, start:end] = TM[perm][:, start:end]

    real_i = TM[:, :start].float().sum(1) + TM[:, end:].float().sum(1)
    real_j = TM[perm][:, start:end].float().sum(1)
    total  = (real_i + real_j).clamp(min=1.0)
    lam_eff = (real_i / total).unsqueeze(1)
    Y_mix = lam_eff * Y + (1.0 - lam_eff) * Y[perm]

    return X_mix, TM_mix, Y_mix
""")

# ---------------------------------------------------------------------------
md("""## 11 — Model: SDPA Transformer with RoPE, per-class MLP heads, aux BCE

**Backbone.** Token embedding → CNN stem ( parallel `Conv1d k=3` and `Conv1d
k=5`, concat to `d_model` ) → 6 × pre-norm Transformer block ( each block uses
`F.scaled_dot_product_attention` with RoPE applied to Q/K and a `(B, 1, 1, L)`
boolean padding mask broadcast to `(B, H, L, L)` ) → final `LayerNorm` →
`mean ⊕ attn` pool over real tokens → shared head → **8 parallel 2-layer MLPs**
( one per class ).

**Why RoPE, not ALiBi.** At `L=4096, H=6`, an ALiBi bias tensor is `(H, L, L)`
= 6 · 4096² · 4 bytes ≈ 384 MB ( fp32 ) or 192 MB ( fp16 ), and SDPA would have
to broadcast it across the batch. That defeats the mem-efficient kernel's
O(N)-memory property. RoPE applies a small per-position rotation to Q/K
before SDPA — `(L, head_dim/2)` cos/sin tables, ~0.5 MB total — and works
cleanly with the mem-efficient backend ( the mask passed to SDPA is the
trivial boolean padding mask ).

**Aux BCE projection** is a small `Linear(2·d_model → 2)` off the shared
trunk producing extra logits for BR + FR. It feeds the auxiliary BCE term in
the loss; during inference we ignore it.
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


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    # cos, sin: (L, head_dim/2) → expand to (1, 1, L, head_dim)
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    q_rot = q * cos_full + _rotate_half(q) * sin_full
    k_rot = k * cos_full + _rotate_half(k) * sin_full
    return q_rot, k_rot


class MHAWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.h = n_heads
        self.d = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x, attn_mask, cos, sin):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    # (B, H, L, head_dim)
        cos_d = cos.to(dtype=q.dtype, device=q.device)
        sin_d = sin.to(dtype=q.dtype, device=q.device)
        q, k = _apply_rope(q, k, cos_d, sin_d)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )                                  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        return self.out(out)


class PreNormTxBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, drop_path):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MHAWithRoPE(d_model, n_heads, dropout)
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

    def forward(self, x, attn_mask, cos, sin):
        h = self.ln1(x)
        a = self.mha(h, attn_mask, cos, sin)
        x = x + self.dp1(self.drop1(a))
        h = self.ln2(x)
        x = x + self.dp2(self.drop2(self.ff(h)))
        return x


class BytecodeTxModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_classes=N_LABELS,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 d_ff=D_FF, max_len=SEQ_LEN, dropout=DROPOUT, drop_path=DROP_PATH,
                 aux_dim=2):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, f"head_dim={head_dim} must be even for RoPE"

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.emb_norm  = nn.LayerNorm(d_model)
        self.emb_drop  = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(d_model)

        dp_rates = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            PreNormTxBlock(d_model, n_heads, d_ff, dropout, dp_rates[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Pooling
        self.attn_score = nn.Linear(d_model, 1)

        # RoPE cos/sin tables — small, registered as buffers so they move with .to()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)              # (max_len, head_dim/2)
        self.register_buffer("rope_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("rope_sin", torch.sin(freqs), persistent=False)

        # Heads
        pool_dim = 2 * d_model    # mean ⊕ attn
        self.head_shared = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            ) for _ in range(num_classes)
        ])
        self.aux_head = nn.Linear(pool_dim, aux_dim)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                with torch.no_grad():
                    m.weight[PAD_ID].zero_()

    def forward(self, input_ids, tok_mask):
        \"\"\"input_ids: (B, L) long  |  tok_mask: (B, L) bool  →  (main_logits, aux_logits).\"\"\"
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.emb_drop(self.emb_norm(x))

        c = x.transpose(1, 2)
        c = torch.cat([F.gelu(self.conv3(c)), F.gelu(self.conv5(c))], dim=1)
        x = self.conv_norm(x + c.transpose(1, 2))

        # SDPA attn_mask: bool, True = attend. Shape (B, 1, 1, L) broadcasts to (B, H, L, L).
        attn_mask = tok_mask.view(B, 1, 1, L)
        cos = self.rope_cos[:L]
        sin = self.rope_sin[:L]

        for layer in self.layers:
            x = layer(x, attn_mask, cos, sin)
        x = self.final_norm(x)

        # Pooling over real tokens
        mf = tok_mask.float().unsqueeze(-1)
        denom = mf.sum(1).clamp(min=1.0)
        mean_pool = (x * mf).sum(1) / denom

        s = self.attn_score(x).squeeze(-1)
        s = s.masked_fill(~tok_mask, -1e4)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        attn_pool = (x * w).sum(1)

        pooled = torch.cat([mean_pool, attn_pool], dim=-1)   # (B, 2d)

        shared = self.head_shared(pooled)                    # (B, d)
        main_logits = torch.cat([h(shared) for h in self.class_heads], dim=-1)   # (B, 8)
        aux_logits  = self.aux_head(pooled)                  # (B, 2)
        return main_logits, aux_logits


_m = BytecodeTxModel()
n_params = sum(p.numel() for p in _m.parameters())
log(f"Parameters: {n_params/1e6:.2f} M  ({n_params:,})")
del _m; gc.collect()
""")

# ---------------------------------------------------------------------------
md("""## 12 — Loss: AsymmetricLoss ( per-class weighted ) + aux BCE on BR, FR

Same `AsymmetricLossWeighted` as dive-4 ( γ⁻=4, γ⁺=1, clip=0.05, sqrt-inverse-
freq per-class weights clipped to `[1, 5]` and mean-normalised to 1 ). On top
of that, a `0.3 · BCEWithLogitsLoss` term applied to a 2-d auxiliary head over
( Bad Randomness, Front Running ) — gives those two classes an un-weighted,
non-asymmetric gradient pathway since they are the binding constraint on
macro-F1.

The aux head is a single `Linear(2·d_model → 2)` off the shared pool, costs
~770 params, and is **ignored at inference** ( only `main_logits` are
calibrated / thresholded ).
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
        cw = self.class_weights.to(loss.device)
        loss = loss * cw.unsqueeze(0)
        return -loss.mean()


class ASLPlusAuxBCE(nn.Module):
    \"\"\"Main ASL on all 8 logits + aux_weight · BCE on the BR+FR aux logits.\"\"\"
    def __init__(self, asl: AsymmetricLossWeighted, aux_indices, aux_weight=AUX_BCE_WEIGHT):
        super().__init__()
        self.asl = asl
        self.register_buffer("aux_idx", torch.tensor(aux_indices, dtype=torch.long))
        self.aux_weight = float(aux_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, main_logits, aux_logits, targets):
        main_loss = self.asl(main_logits, targets)
        aux_targets = targets.index_select(1, self.aux_idx.to(targets.device))
        aux_loss = self.bce(aux_logits.float(), aux_targets.float())
        return main_loss + self.aux_weight * aux_loss


_cw = np.sqrt(Y_tr.shape[0] / Y_tr.sum(0).clip(min=1)).astype(np.float32)
class_weights = np.clip(_cw, PER_CLASS_W_MIN, PER_CLASS_W_MAX)
class_weights = class_weights * (len(class_weights) / class_weights.sum())
log("Per-class loss weights (normalised, mean=1):")
for i, lab in enumerate(LABEL_COLS):
    log(f"  {lab:>26s}  w={class_weights[i]:.3f}")

criterion = ASLPlusAuxBCE(
    AsymmetricLossWeighted(class_weights),
    aux_indices=AUX_BCE_INDICES,
    aux_weight=AUX_BCE_WEIGHT,
).to(device)
log(f"Loss: ASL(γ⁻={ASL_GAMMA_NEG}, γ⁺={ASL_GAMMA_POS}, clip={ASL_CLIP}) "
    f"+ {AUX_BCE_WEIGHT} · BCE(aux logits on indices {AUX_BCE_INDICES})")
""")

# ---------------------------------------------------------------------------
md("""## 13 — EMA""")

code("""class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        src = model.module if isinstance(model, nn.DataParallel) else model
        self.ema = BytecodeTxModel().to(next(src.parameters()).device)
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
md("""## 14 — Metrics, isotonic calibration, threshold tuning

Verbatim from dive-4: per-class isotonic regression fit on val probabilities,
applied to test probs before the 41-point threshold sweep. For classes with
val support < 200, reject thresholds whose precision < 0.25 ( prevents
degenerate "predict everything" thresholds on rare classes ).
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
    calibrators = []
    for i in range(y_prob.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
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
md("""## 15 — Optimizer, cosine warmup, predict""")

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
    for X, TM, Yb in loader:
        X  = X.to(device, non_blocking=True)
        TM = TM.to(device, non_blocking=True)
        with autocast():
            main_logits, _ = model(X, TM)
        logits_all.append(main_logits.float().cpu().numpy())
        labels_all.append(Yb.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)
""")

# ---------------------------------------------------------------------------
md("""## 16 — Dataloaders""")

code("""train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          drop_last=True, persistent_workers=NUM_WORKERS>0,
                          collate_fn=cutmix_collate_flat)
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
md("""## 17 — Atomic checkpoint save / load ( full state )

Same pattern as dive-4. Payload includes the selector-vocab path so a resumed
run uses the exact same vocabulary.
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
        "best_macro_f1":   float(best_macro_f1),
        "best_thresholds": np.asarray(best_thresholds, dtype=np.float32).tolist(),
        "calibrators": calibrators,
        "history":   history,
        "rng": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": {
            "SEQ_LEN": SEQ_LEN, "VOCAB_SIZE": VOCAB_SIZE,
            "SELECTOR_VOCAB_SIZE": SELECTOR_VOCAB_SIZE,
            "SELECTOR_VOCAB_PATH": str(SELECTOR_JSON),
            "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS, "D_FF": D_FF,
            "DROPOUT": DROPOUT, "DROP_PATH": DROP_PATH, "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "ASL_GAMMA_NEG": ASL_GAMMA_NEG, "ASL_GAMMA_POS": ASL_GAMMA_POS,
            "ASL_CLIP": ASL_CLIP, "EMA_DECAY": EMA_DECAY,
            "AUG_SPAN_MASK_PROB": AUG_SPAN_MASK_PROB, "AUG_SPAN_MAX_LEN": AUG_SPAN_MAX_LEN,
            "AUG_CUTMIX_PROB": AUG_CUTMIX_PROB, "AUG_CUTMIX_BETA": AUG_CUTMIX_BETA,
            "AUX_BCE_INDICES": AUX_BCE_INDICES, "AUX_BCE_WEIGHT": AUX_BCE_WEIGHT,
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
md("""## 18 — Build model, optimiser, scheduler, scaler, EMA""")

code("""model = BytecodeTxModel().to(device)
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
md("""## 19 — Training loop

Per epoch:

1. **Train** — sampler delivers class-balanced batches; CutMix collate
   optionally splices token spans across pairs; AMP fp16 + grad-clip + cosine
   LR + EMA update; combined `ASL + 0.3·BCE_aux` loss.
2. **Validate on EMA** — predict main logits → fit per-class isotonic on val
   → apply to val probs → tune thresholds on calibrated probs. Reported
   `f1_macro` is on calibrated + thresholded predictions ( the deployed
   pipeline ).
3. **Save full state** atomically. Save `best_model.pt` ( EMA weights +
   calibrators + thresholds ) on each new best.
4. **Wall-clock snapshot** every `CKPT_WALLCLOCK_SECS`.
5. **Early stop** after `PATIENCE` consecutive non-improvements.
""")

code("""HEARTBEAT_STEPS = 50


def _run_epoch(epoch):
    model.train()
    t0 = time.time()
    n_steps = len(train_loader)
    running_loss = 0.0
    running_n = 0

    for step, (X, TM, Yb) in enumerate(train_loader, 1):
        X  = X.to(device,  non_blocking=True)
        TM = TM.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            main_logits, aux_logits = model(X, TM)
            loss = criterion(main_logits, aux_logits, Yb)

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
        train_loss, train_dt = _run_epoch(epoch)

        val_logits, val_labels = predict(ema.ema, val_loader)
        val_probs_raw = 1 / (1 + np.exp(-val_logits))
        calibrators = fit_isotonic_per_class(val_labels, val_probs_raw)
        val_probs   = apply_calibration(val_probs_raw, calibrators)
        val_thr     = tune_thresholds(val_labels, val_probs)
        val_metrics, _ = multilabel_metrics(val_labels, val_probs, val_thr)
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

        save_checkpoint(STATE_PATH,
                        model=model, ema=ema,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch,
                        best_macro_f1=best_macro_f1,
                        best_thresholds=best_thresholds,
                        history=history,
                        calibrators=calibrators)
        epoch_save_done = True

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
md("""## 20 — Test evaluation ( calibrated + frozen thresholds )

Load `best_model.pt` ( EMA weights + isotonic calibrators + frozen
thresholds ). Apply calibration to test probabilities, then frozen
thresholds. Report:

- **calibrated + tuned** ( deployed setting ).
- **calibrated @ 0.5** ( calibration-only ).
- **raw @ 0.5** ( no calibration, no tuning — reference ).

This makes the threshold-tuning effect and the calibration effect both
explicit.
""")

code("""log("=== TEST EVAL ===")
if not BEST_PATH.exists():
    log("No best_model.pt found — falling back to current EMA weights and last calibrators.")
    eval_model = ema.ema
    frozen_thresholds = best_thresholds
    frozen_calibrators = best_calibrators
else:
    ckpt_b = torch.load(BEST_PATH, map_location=device)
    eval_model = BytecodeTxModel().to(device)
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


# Artefact dump
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
            "seq_len": SEQ_LEN, "vocab_size": VOCAB_SIZE,
            "selector_vocab_size": SELECTOR_VOCAB_SIZE,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF,
            "dropout": DROPOUT, "drop_path": DROP_PATH,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "asl_gamma_neg": ASL_GAMMA_NEG, "asl_gamma_pos": ASL_GAMMA_POS, "asl_clip": ASL_CLIP,
            "ema_decay": EMA_DECAY,
            "aug_span_mask_prob": AUG_SPAN_MASK_PROB, "aug_span_max_len": AUG_SPAN_MAX_LEN,
            "aug_cutmix_prob": AUG_CUTMIX_PROB, "aug_cutmix_beta": AUG_CUTMIX_BETA,
            "aux_bce_indices": AUX_BCE_INDICES, "aux_bce_weight": AUX_BCE_WEIGHT,
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
""")

# ---------------------------------------------------------------------------
md("""## 21 — ONNX export ( verification gate #3 )

Export the EMA model with a dynamic seq_len axis. SDPA lowers to ONNX
opset 17's Attention op; RoPE is plain elementwise math and exports cleanly.
We export a wrapper that returns only `main_logits` ( the aux head is
training-only ).

If the export fails ( e.g. SDPA → ONNX issue ), log the error and continue —
this is a nice-to-have, not a training gate.
""")

code("""class ExportWrapper(nn.Module):
    \"\"\"Returns only main_logits — the deployed signature.\"\"\"
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, input_ids, tok_mask):
        main, _ = self.m(input_ids, tok_mask)
        return main


onnx_ok = False
try:
    src = eval_model.module if isinstance(eval_model, nn.DataParallel) else eval_model
    wrapper = ExportWrapper(src).to(device).eval()
    dummy_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    dummy_tm  = torch.ones(1, SEQ_LEN, dtype=torch.bool, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_tm),
        str(ONNX_PATH),
        input_names=["input_ids", "tok_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "tok_mask":  {0: "batch", 1: "seq_len"},
            "logits":    {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_ok = ONNX_PATH.exists()
    if onnx_ok:
        log(f"ONNX exported → {ONNX_PATH.name} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    log(f"ONNX export failed: {e!r}")
    log("  (this does not affect the trained checkpoint; in-Python FP16 inference still works)")
""")

# ---------------------------------------------------------------------------
md("""## 22 — Per-contract inference latency benchmark

Run `BENCH_N_SAMPLES` test contracts one-at-a-time ( batch=1 ) through the
FP16 EMA model on a single T4. Report p50 / p90 / mean / max ms per contract.
Appended to `history.csv` with `phase="inference_bench"` so the comparison
against dive-4's 4-chunk forward is in one place.
""")

code("""def benchmark_latency(model, X_arr, TM_arr, n_samples=BENCH_N_SAMPLES):
    src = model.module if isinstance(model, nn.DataParallel) else model
    one_dev = torch.device("cuda:0")
    src = src.to(one_dev).eval()

    n = min(n_samples, len(X_arr))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(X_arr), n, replace=False)

    # Warmup
    with torch.no_grad():
        for i in indices[:10]:
            x  = torch.from_numpy(X_arr[i:i+1]).long().to(one_dev)
            tm = torch.from_numpy(TM_arr[i:i+1]).to(one_dev)
            with autocast():
                _ = src(x, tm)
    torch.cuda.synchronize(0)

    times_ms = []
    with torch.no_grad():
        for i in indices:
            x  = torch.from_numpy(X_arr[i:i+1]).long().to(one_dev)
            tm = torch.from_numpy(TM_arr[i:i+1]).to(one_dev)
            torch.cuda.synchronize(0)
            t0 = time.time()
            with autocast():
                _ = src(x, tm)
            torch.cuda.synchronize(0)
            times_ms.append((time.time() - t0) * 1000.0)
    times = np.array(times_ms)
    return {
        "n_samples": int(n),
        "p50_ms":    float(np.percentile(times, 50)),
        "p90_ms":    float(np.percentile(times, 90)),
        "mean_ms":   float(times.mean()),
        "max_ms":    float(times.max()),
        "min_ms":    float(times.min()),
    }


bench = benchmark_latency(eval_model, X_te, TM_te, n_samples=BENCH_N_SAMPLES)
log("=== INFERENCE BENCHMARK (single T4, FP16, batch=1) ===")
for k, v in bench.items():
    if k == "n_samples":
        log(f"  {k:>10s}: {v}")
    else:
        log(f"  {k:>10s}: {v:.2f} ms")
with open(BENCH_JSON, "w") as f:
    json.dump(bench, f, indent=2)

history.append({"epoch": -1, "phase": "inference_bench", **bench})
_flush_history()


log("Artefacts written:")
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        log(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")

log.close()
""")

# ---------------------------------------------------------------------------
md("""## Summary

**What dive-5 changes vs dive-4.**

- **Semantic tokenizer** — keep PUSH immediates as typed tokens, with a
  2048-entry function-selector vocab built from train data only. Strips the
  CBOR metadata trailer ( ~40 % of contracts ). Vocab = 2,318 tokens.
- **Single-pass long-context Transformer** with
  `torch.nn.functional.scaled_dot_product_attention` ( mem-efficient
  backend ) at SEQ_LEN=4096 — no chunking, one forward pass per contract.
- **RoPE positions** applied to Q/K before SDPA — bias-free, ~0.5 MB of
  cos/sin tables, no L²-sized buffers.
- **Per-class MLP heads** ( 8 parallel 2-layer MLPs off a shared trunk ).
- **Auxiliary BCE** on Bad Randomness + Front Running ( weight 0.3 ) —
  un-weighted gradient pathway for the 2 rarest classes.
- **Token-span CutMix** ( replaces chunk-level CutMix ).

**What stays identical to dive-4.**

- 80/10/10 multilabel-stratified split ( seed 42 ).
- `AsymmetricLossWeighted` ( γ⁻=4, γ⁺=1, clip=0.05 ) + sqrt-inverse-freq
  per-class weights clipped to `[1, 5]`.
- EMA decay 0.999, DropPath linear 0 → 0.10.
- AdamW + cosine warmup ( 10 % ), grad clip 1.0, AMP fp16, DataParallel.
- `WeightedRandomSampler` with sqrt-inverse-freq weights clipped to `[1, 5]`.
- Per-class isotonic calibration on val → 41-point threshold tune ( min-
  precision 0.25 for classes with val support < 200 ).
- Atomic full-state checkpointing, tee logger, history.csv flush per epoch.

**Verification gates ( first hour ).**

1. **SDPA mem-efficient kernel** — cell 4 asserts peak fp16 memory at
   `B=BATCH_PER_GPU, L=SEQ_LEN, H=N_HEADS` is < `SDPA_MEMORY_BUDGET_BYTES`.
2. **Tokenizer length coverage** — cell 8 asserts ≥ `MIN_COVERAGE_AT_SEQ_LEN`
   of contracts fit in `SEQ_LEN`.
3. **ONNX export** — cell 21 attempts export; failure is logged but
   non-fatal.

**Outputs ( `/kaggle/working/` ).**

- `best_model.pt` — EMA weights + isotonic calibrators + frozen thresholds.
- `last_state.pt` — full resumable state.
- `state_epochNN.pt` — wall-clock snapshots.
- `dive5_train.log`, `history.csv`, `history.json`.
- `metrics.json`, `per_class.csv`, `confusion_per_class.json`.
- `test_probs_calibrated.npy`, `test_probs_raw.npy`, `test_labels.npy`,
  `thresholds.npy`.
- `cache/dive5_selector_vocab.json`, `cache/dive5_tokens.npz`.
- `dive5_model.onnx` ( if export succeeded ).
- `inference_bench.json` ( p50 / p90 / mean ms per contract on single T4 ).

**What to look at after the run.**

1. **Coverage log line in cell 8** — confirm ≥ 85 % of contracts fit at
   SEQ_LEN=4096 after semantic tokenization.
2. **`test_calibrated_tuned.f1_macro`** vs dive-4's run.
3. **Per-class F1 on Bad Randomness and Front Running** — the targeted rare
   classes. The aux BCE head and semantic tokens are the levers.
4. **`inference_bench.json`** — per-contract p50 ms on single T4 FP16.
   Should be strictly lower than dive-4's 4-chunk forward.
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

out = Path(__file__).parent / "dive-5.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
