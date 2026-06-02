#!/usr/bin/env python3
"""Build build_dive_dedup_4.ipynb.

dive-dedup-4 = the dive-dedup-3 engine (frozen GraphCodeBERT unit embeddings +
set-Transformer aggregator) on the LOCKED dedup 70/10/20 benchmark, but with the
unit source switched from Solidity functions to **bytecode-derived readable EVM
assembly basic blocks**. It is the bytecode twin of source-modality dedup-3 on the
identical splits / engine — a clean source-vs-bytecode transfer-learning comparison.

Preprocessing (the new bit): per contract, strip the solc CBOR metadata trailer →
disassemble → decode PUSH operands (hex) + 4-byte function selectors → segment into
JUMPDEST-delimited basic blocks → emit each block as readable assembly text. Blocks
are risk-ranked and capped to MAX_UNITS, then fed to frozen GraphCodeBERT exactly as
dedup-3 fed Solidity function units.

This script loads build_dive_dedup_3.ipynb and surgically replaces the title, config
(bytecode wiring + naming), fold-load (bytecode coverage), the unit splitter (→ the
bytecode disassembler) and the narrative cells, keeping the GraphCodeBERT stage-1,
set-Transformer, loss, calibration, sampler, training, eval, graphs, inference and
arm-comparison cells verbatim. Run:  python tools/build_dive_dedup_4.py
"""
import io
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "build_dive_dedup_3.ipynb"
OUT_NB = ROOT / "build_dive_dedup_4.ipynb"


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


# ---------------------------------------------------------------------------
TITLE_MD = r"""
# dive-dedup-4 — bytecode-as-assembly into the GraphCodeBERT engine

**What this is.** The **dive-dedup-3 engine** (frozen GraphCodeBERT unit embeddings →
set-Transformer aggregator → per-class heads + aux BCE) run on the **identical locked dedup
70/10/20 benchmark** (13,143 / 1,878 / 3,756) — but the units are no longer Solidity functions.
Each contract's **bytecode** is turned into **readable EVM assembly basic blocks** and those
blocks are the units GraphCodeBERT encodes. dedup-4 is therefore the *bytecode twin* of dedup-3
on the same engine and splits: a clean **source-vs-bytecode transfer-learning** comparison.

**The new preprocessing (cell 7).** Per contract:
1. **Strip the solc CBOR metadata trailer** (non-executable; pure noise).
2. **Disassemble** the runtime bytecode → one opcode mnemonic per instruction.
3. **Decode PUSH operands** to hex, and **4-byte function selectors** (`PUSH4 0x… // selector …`,
   with a built-in name table for common ERC-20/721 selectors).
4. **Segment into basic blocks** at every `JUMPDEST` and after every terminator
   (`JUMP/JUMPI/STOP/RETURN/REVERT/INVALID/SELFDESTRUCT`); each block is emitted as readable
   labelled assembly text (`label_0x…: \n  PUSH1 0x80 \n  MSTORE \n …`).
5. **Risk-rank and cap** to `MAX_UNITS` (dispatcher/entry blocks first, then blocks by risky-opcode
   content + length) — the exact analog of dedup-3's function risk-ranking.

The dispatcher/entry blocks take unit-type 0 (the "header" slot in dedup-3); ordinary basic blocks
take type 1. Everything downstream — frozen GraphCodeBERT mean-pool, the set-Transformer, ASL +
aux-BCE loss, isotonic calibration, threshold tuning, the quality-rebalance sampler, EMA, the gates
and the arm table — is **dedup-3 verbatim**.

**Arms** (set `ARM`; run once per arm in the same Kaggle session so the per-arm
`metrics_arm_*.json` accumulate and the final cell tabulates them):

| `ARM` | Dataset | Train fold |
|---|---|---|
| `"dedup"`       | `dive-dedup`              | 13,143 real, no augmentation (baseline) |
| `"synth-dedup"` | `dive-synthesized-dedup`  | 24,640 (13,143 real + 11,497 SolidiFI synthetic) |

Both arms share the **same real val (1,878) and test (3,756)** folds — byte-identical to the locked
dedup benchmark (`Test` sha256 `53ed81a6…`) — same model, hparams, loss, calibration, thresholds,
sampler, EMA, seed. The only variable across arms is the train corpus; the only variable vs dedup-3
is the modality (assembly basic blocks instead of source functions).

**Caveat to read with.** GraphCodeBERT was pretrained on high-level source (6 languages + data flow),
**not** EVM assembly, so assembly text is out-of-distribution for it — that is exactly what this dive
measures. The selector/operand decoding nudges the text closer to source-like.

`MAX_UNITS` stays at dedup-3's value (32) for **engine parity** — deliberately, not by oversight.
Measured pre-cap basic-block counts on this corpus are **median ≈ 480 blocks/contract** (p90 ≈ 985,
max ≈ 2,800), so the cap is an *inherent* heavy-truncation regime, not a tunable that can be raised
away: cap 32 retains ~5.8% of blocks, cap 64 only ~11.4% (≈2× stage-1 encode + ~3 GB cache), cap 128
~22.4% (≈4×) — and even 128 still caps 91% of contracts. Because no reachable cap recovers coverage,
the **risk-ranking** (dispatcher/selector blocks first, then risky-opcode density) is what carries the
signal, and 32 keeps the set-Transformer input identical to source-modality dedup-3 for a clean
modality-only delta. If test F1 disappoints, raising `MAX_UNITS` (watch the 4h `STAGE1_BUDGET_SECS`
gate) is the first knob — but treat it as adding a second variable vs dedup-3, not free coverage.

**Reading the result.**
1. **Gates:** leakage + locked-test (3,756 rows, recorded support) + bytecode coverage + units/contract.
2. **`test_calibrated_tuned.f1_macro` per arm** on the 3,756 test, and **dedup-4 vs dedup-3** on the
   same splits — the source-vs-bytecode headline.
3. **`delta macro-F1 (synth-dedup − dedup)`** — does SolidiFI synthetic help on bytecode-as-assembly?
4. **Per-class F1 / overfit gap on BR / FR / DoS** — the rare classes.
"""

CONFIG_TOP = r'''# === ARM SWITCH ==============================================================
# The dive-dedup-3 GraphCodeBERT engine on the LOCKED dedup 70/10/20 benchmark,
# fed bytecode-derived readable assembly units. Identical model/config across
# arms; the ONLY variable is the train corpus. Val/test are byte-identical across
# both datasets (the locked dedup benchmark), so the two arms are directly
# comparable - and comparable to dive-dedup-3 (source modality) on the same splits.
#   "dedup"        baseline  - real-only dedup train (13,143)            [no aug]
#   "synth-dedup"  augmented - real + SolidiFI synthetic (24,640 total)  [final/]
# Run once per arm in the same Kaggle session (set ARM -> Run All; switch -> Run
# All); the final cell tabulates both on the shared real test set.
ARM = "synth-dedup"          # <- flip to "dedup" for the real-only baseline arm
assert ARM in ("dedup", "synth-dedup"), f"bad ARM {ARM!r}"

# -- Locate both datasets (Kaggle mounts or local) ----------------------------
def _has(d, *parts):
    p = Path(d)
    for x in parts:
        p = p / x
    return p.exists()

def _find_dedup_root():
    """dive-dedup root: <root>/{splits/Train_Labels.csv, Bytecode_filled.csv} and NO final/."""
    for c in ("/kaggle/input/datasets/henrychristian7555/dive-dedup/Data-before-aug",
              "/kaggle/input/dive-dedup/Data-before-aug",
              "/kaggle/input/datasets/henrychristian7555/dive-dedup",
              "/kaggle/input/dive-dedup",
              "dive-dedup"):
        if _has(c, "splits", "Train_Labels.csv") and _has(c, "Bytecode_filled.csv") and not _has(c, "final"):
            return Path(c)
    base = Path("/kaggle/input")
    if base.exists():
        for sp in sorted(base.rglob("splits")):
            r = sp.parent
            if (sp / "Train_Labels.csv").exists() and (r / "Bytecode_filled.csv").exists() and not (r / "final").exists():
                return r
    return None

def _find_synth_root():
    """dive-synthesized-dedup root: <root>/final/{Train_Labels.csv, Bytecode_filled.csv, Synthetic_Bytecode.csv}."""
    for c in ("/kaggle/input/datasets/henrychristian7555/dive-synthesized-dedup/Dataset",
              "/kaggle/input/dive-synthesized-dedup/Dataset",
              "/kaggle/input/datasets/henrychristian7555/dive-synthesized-dedup",
              "/kaggle/input/dive-synthesized-dedup",
              "dive-synthesized-dedup"):
        if _has(c, "final", "Train_Labels.csv") and _has(c, "final", "Bytecode_filled.csv"):
            return Path(c)
    base = Path("/kaggle/input")
    if base.exists():
        for fp in sorted(base.rglob("final")):
            if (fp / "Train_Labels.csv").exists() and (fp / "Bytecode_filled.csv").exists():
                return fp.parent
    return None

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

# -- Per-arm paths: split CSVs + bytecode CSVs --------------------------------
# val/test load from each arm's own root; they are byte-identical, so the gates
# below pin them to the one recorded 3,756-row dedup benchmark either way.
if ARM == "dedup":
    DATA_ROOT = _find_dedup_root()
    assert DATA_ROOT is not None, (
        "dive-dedup not attached. Add Input -> 'dive-dedup' "
        "(henrychristian7555/dive-dedup). Expected <mount>/"
        "{splits/Train_Labels.csv, Bytecode_filled.csv}.")
    TRAIN_CSV = DATA_ROOT / "splits" / "Train_Labels.csv"
    VAL_CSV   = DATA_ROOT / "splits" / "Val_Labels.csv"
    TEST_CSV  = DATA_ROOT / "splits" / "Test_Labels.csv"
    REAL_BYTECODE_CSV  = DATA_ROOT / "Bytecode_filled.csv"
    SYNTH_BYTECODE_CSV = None
else:  # "synth-dedup"
    DATA_ROOT = _find_synth_root()
    assert DATA_ROOT is not None, (
        "dive-synthesized-dedup not attached. Add Input -> 'dive-synthesized-dedup' "
        "(henrychristian7555/dive-synthesized-dedup). Expected <mount>/Dataset/final/"
        "{Train_Labels.csv, Bytecode_filled.csv, Synthetic_Bytecode.csv}.")
    TRAIN_CSV = DATA_ROOT / "final" / "Train_Labels.csv"
    VAL_CSV   = DATA_ROOT / "final" / "Val_Labels.csv"
    TEST_CSV  = DATA_ROOT / "final" / "Test_Labels.csv"
    REAL_BYTECODE_CSV  = DATA_ROOT / "final" / "Bytecode_filled.csv"
    SYNTH_BYTECODE_CSV = DATA_ROOT / "final" / "Synthetic_Bytecode.csv"

_FOLD_CSV = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}

def _fold_csv(name):
    """Return path to the split CSV for fold 'train', 'val', or 'test'."""
    p = _FOLD_CSV[name]
    if not p.exists():
        raise FileNotFoundError(f"No split CSV for fold '{name}' - tried: {p}")
    return p

# === OUTPUT NAMING ===========================================================
OUT_DIR     = Path("/kaggle/working");   OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR   = OUT_DIR / "cache";         CACHE_DIR.mkdir(parents=True, exist_ok=True)
_ARM        = ARM
RUN_TAG     = ARM
# AUG_SOURCE aliases ARM so every downstream cell (metrics, graphs, arm table)
# keys cleanly on the arm name without further edits.
AUG_SOURCE  = ARM
LOG_PATH    = OUT_DIR / f"dive_dedup4_{_ARM}_train.log"
HIST_CSV    = OUT_DIR / f"history_{_ARM}.csv"
HIST_JSON   = OUT_DIR / f"history_{_ARM}.json"
STATE_PATH  = OUT_DIR / f"last_state_{_ARM}.pt"
BEST_PATH   = OUT_DIR / f"best_model_{_ARM}.pt"
EMB_NPZ     = CACHE_DIR / f"dive_dedup4_units_{_ARM}.npz"
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
assert TRAIN_CSV.exists(), f"Missing train CSV {TRAIN_CSV}"
assert VAL_CSV.exists() and TEST_CSV.exists(), f"Missing val/test CSV under {DATA_ROOT}"
assert REAL_BYTECODE_CSV.exists(), f"Missing real bytecode CSV {REAL_BYTECODE_CSV}"
if ARM == "synth-dedup":
    assert SYNTH_BYTECODE_CSV is not None and SYNTH_BYTECODE_CSV.exists(), (
        f"ARM='synth-dedup' but synthetic bytecode CSV is missing: {SYNTH_BYTECODE_CSV}. "
        f"Either re-upload final/Synthetic_Bytecode.csv, or set ARM='dedup'.")
print(f"ARM={ARM}  DATA_ROOT={DATA_ROOT}", flush=True)
print(f"  train={TRAIN_CSV}", flush=True)
print(f"  real  bytecode={REAL_BYTECODE_CSV}", flush=True)
print(f"  synth bytecode={SYNTH_BYTECODE_CSV}", flush=True)
print("RESUME_FROM:", RESUME_FROM, flush=True)
print(f"Stage1: MAX_UNITS={MAX_UNITS} UNIT_TOKENS={UNIT_TOKENS} model={GCB_MODEL}", flush=True)
print(f"Stage2: d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} d_ff={D_FF}", flush=True)
'''

# bytecode loader prepended to the fold-load cell
BYTECODE_LOAD = r'''# === Load bytecode (real + synthetic) into one in-memory map =================
# Streams the arm's real Bytecode_filled.csv (~330 MB) and, for the synth-dedup
# arm, final/Synthetic_Bytecode.csv into a single contractID -> hex dict. Real IDs
# are numeric strings; synthetic IDs are buggy_<id>_<type>. These keys drive the
# fold-level bytecode-coverage gate below (verified 100% on all folds, both arms).
def _load_bc_csv(path, label):
    d = pd.read_csv(path, usecols=["contractID", "bytecode"], dtype={"contractID": str})
    n0 = len(d)
    d = d.dropna(subset=["bytecode"])
    d = d[d["bytecode"].astype(str).str.len() > 4]
    log(f"  {label:>10s}: {len(d):6d} / {n0:6d} rows with bytecode  ({path.name})")
    return dict(zip(d["contractID"].astype(str), d["bytecode"].astype(str)))


log("Loading bytecode CSVs...")
bytecode_map = _load_bc_csv(REAL_BYTECODE_CSV, "real")
if ARM == "synth-dedup" and SYNTH_BYTECODE_CSV is not None and SYNTH_BYTECODE_CSV.exists():
    syn_map = _load_bc_csv(SYNTH_BYTECODE_CSV, "synthetic")
    overlap = set(bytecode_map) & set(syn_map)
    if overlap:
        log(f"  WARNING: {len(overlap)} ID collision(s) real/synthetic - synthetic wins.")
    bytecode_map.update(syn_map)
log(f"Combined bytecode map: {len(bytecode_map):,} unique IDs")


'''

SPLITTER_MD = r"""
## 7 — Bytecode → readable EVM assembly basic-block units

The bytecode analog of dedup-3's Solidity function splitter. Provides the same three entry points
the stage-1 encoder calls — `read_source(cid)`, `split_solidity_units(text)`, `cap_units(units)` —
so the GraphCodeBERT stage-1 cell is reused **verbatim**; only the unit *content* changes.

1. **Strip the solc CBOR metadata trailer** (`strip_metadata`, from dive-5) — non-executable noise.
2. **Disassemble** → one opcode mnemonic per instruction; **PUSH operands decoded to hex**.
3. **Function selectors**: a `PUSH4` is annotated `// selector 0x…` with a built-in name table for
   common ERC-20/721 selectors; blocks carrying selector compares are marked dispatcher (type 0).
4. **Basic blocks**: a new block starts at `JUMPDEST` and after each terminator
   (`JUMP/JUMPI/STOP/RETURN/REVERT/INVALID/SELFDESTRUCT`); each is emitted as labelled assembly text.
5. **Risk-ranked cap** to `MAX_UNITS`: dispatcher/entry blocks first (finite high base score), then
   blocks by risky-opcode content (`CALL`, `DELEGATECALL`, `SSTORE`, `TIMESTAMP`, `SELFDESTRUCT`, …)
   + length; kept blocks returned in program-counter order.
"""

SPLITTER_CODE = r'''NL = chr(10)

# -- EVM opcode mnemonic table (0x00..0xff) -----------------------------------
_BASE_OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV", 0x05: "SDIV",
    0x06: "MOD", 0x07: "SMOD", 0x08: "ADDMOD", 0x09: "MULMOD", 0x0a: "EXP", 0x0b: "SIGNEXTEND",
    0x10: "LT", 0x11: "GT", 0x12: "SLT", 0x13: "SGT", 0x14: "EQ", 0x15: "ISZERO",
    0x16: "AND", 0x17: "OR", 0x18: "XOR", 0x19: "NOT", 0x1a: "BYTE", 0x1b: "SHL",
    0x1c: "SHR", 0x1d: "SAR", 0x20: "KECCAK256",
    0x30: "ADDRESS", 0x31: "BALANCE", 0x32: "ORIGIN", 0x33: "CALLER", 0x34: "CALLVALUE",
    0x35: "CALLDATALOAD", 0x36: "CALLDATASIZE", 0x37: "CALLDATACOPY", 0x38: "CODESIZE",
    0x39: "CODECOPY", 0x3a: "GASPRICE", 0x3b: "EXTCODESIZE", 0x3c: "EXTCODECOPY",
    0x3d: "RETURNDATASIZE", 0x3e: "RETURNDATACOPY", 0x3f: "EXTCODEHASH",
    0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP", 0x43: "NUMBER",
    0x44: "PREVRANDAO", 0x45: "GASLIMIT", 0x46: "CHAINID", 0x47: "SELFBALANCE", 0x48: "BASEFEE",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE", 0x53: "MSTORE8", 0x54: "SLOAD",
    0x55: "SSTORE", 0x56: "JUMP", 0x57: "JUMPI", 0x58: "PC", 0x59: "MSIZE", 0x5a: "GAS",
    0x5b: "JUMPDEST", 0x5f: "PUSH0",
    0xf0: "CREATE", 0xf1: "CALL", 0xf2: "CALLCODE", 0xf3: "RETURN", 0xf4: "DELEGATECALL",
    0xf5: "CREATE2", 0xfa: "STATICCALL", 0xfd: "REVERT", 0xfe: "INVALID", 0xff: "SELFDESTRUCT",
}
OPCODES = dict(_BASE_OPCODES)
for _i in range(32):
    OPCODES[0x60 + _i] = f"PUSH{_i + 1}"
for _i in range(16):
    OPCODES[0x80 + _i] = f"DUP{_i + 1}"
for _i in range(16):
    OPCODES[0x90 + _i] = f"SWAP{_i + 1}"
for _i in range(5):
    OPCODES[0xa0 + _i] = f"LOG{_i}"

BLOCK_TERMINATORS = {"JUMP", "JUMPI", "STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT"}
# risky sinks - the bytecode analog of dedup-3's source RISK_RE (drives unit_score)
RISK_OPS = {"DELEGATECALL", "CALLCODE", "SELFDESTRUCT", "CALL", "STATICCALL", "CREATE",
            "CREATE2", "SSTORE", "SLOAD", "TIMESTAMP", "NUMBER", "BLOCKHASH", "ORIGIN",
            "GASPRICE", "COINBASE", "PREVRANDAO", "KECCAK256", "CALLVALUE"}
# common ERC-20/721 selectors -> signatures (best-effort readability; offline)
KNOWN_SELECTORS = {
    "a9059cbb": "transfer(address,uint256)", "23b872dd": "transferFrom(address,address,uint256)",
    "095ea7b3": "approve(address,uint256)", "70a08231": "balanceOf(address)",
    "18160ddd": "totalSupply()", "dd62ed3e": "allowance(address,address)",
    "313ce567": "decimals()", "06fdde03": "name()", "95d89b41": "symbol()",
    "40c10f19": "mint(address,uint256)", "a0712d68": "mint(uint256)", "42966c68": "burn(uint256)",
    "f2fde38b": "transferOwnership(address)", "8da5cb5b": "owner()", "715018a6": "renounceOwnership()",
    "2e1a7d4d": "withdraw(uint256)", "3ccfd60b": "withdraw()", "d0e30db0": "deposit()",
    "6352211e": "ownerOf(uint256)", "081812fc": "getApproved(uint256)",
    "e985e9c5": "isApprovedForAll(address,address)", "a22cb465": "setApprovalForAll(address,bool)",
    "42842e0e": "safeTransferFrom(address,address,uint256)",
}


def strip_metadata(bc_hex):
    """Strip the solc CBOR metadata trailer if present. Returns hex_str. (dive-5)"""
    s = bc_hex.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) < 6 or (len(s) & 1):
        return s
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return s
    n = len(b)
    if n < 4:
        return s
    L = int.from_bytes(b[-2:], "big")
    if L < 2 or L + 2 > n:
        return s
    if b[-(L + 2)] not in (0xa1, 0xa2):
        return s
    return b[:-(L + 2)].hex()


def _disassemble(b):
    """bytes -> list of (pc, mnemonic, operand_hex|None)."""
    out = []
    i, n = 0, len(b)
    while i < n:
        op = b[i]
        mnem = OPCODES.get(op, f"UNKNOWN_0x{op:02x}")
        if 0x60 <= op <= 0x7f:                  # PUSH1..PUSH32
            k = op - 0x5f
            operand = b[i + 1:i + 1 + k].hex()
            out.append((i, mnem, operand))
            i += 1 + k
        else:
            out.append((i, mnem, None))
            i += 1
    return out


def split_solidity_units(bc_hex):
    """bytecode hex -> list of (pc_start, assembly_text, is_function). Same tuple
    shape as dedup-3's source splitter so stage-1 / inference are reused verbatim."""
    s = strip_metadata(bc_hex)
    try:
        b = bytes.fromhex(s)
    except ValueError:
        b = b""
    if not b:
        return [(0, "label_0x0:" + NL + "  STOP", False)]
    instrs = _disassemble(b)

    # block start PCs: entry, every JUMPDEST, and the instr after each terminator
    starts = {instrs[0][0]}
    for idx, (pc, mnem, _operand) in enumerate(instrs):
        if mnem == "JUMPDEST":
            starts.add(pc)
        if mnem in BLOCK_TERMINATORS and idx + 1 < len(instrs):
            starts.add(instrs[idx + 1][0])

    units = []

    def _flush(block, block_start):
        if not block:
            return
        lines = [f"label_0x{block_start:x}:"]
        is_dispatch = False
        for (pc, mnem, operand) in block:
            if operand is not None:
                line = f"  {mnem} 0x{operand}"
                if mnem == "PUSH4":
                    sig = KNOWN_SELECTORS.get(operand)
                    line += f"  // selector 0x{operand}" + (f" {sig}" if sig else "")
                    is_dispatch = True
                lines.append(line)
            else:
                lines.append(f"  {mnem}")
        is_header = is_dispatch or (block_start == 0)   # dispatcher/entry -> type 0
        units.append((block_start, NL.join(lines), not is_header))

    cur, cur_start = [], instrs[0][0]
    for (pc, mnem, operand) in instrs:
        if pc in starts and cur:
            _flush(cur, cur_start)
            cur, cur_start = [], pc
        cur.append((pc, mnem, operand))
    _flush(cur, cur_start)
    return units


def unit_score(text, is_function):
    risk = sum(text.count(op) for op in RISK_OPS)
    if not is_function:                         # dispatcher / entry block
        return HEADER_BASE_SCORE + 50.0 * risk
    return 50.0 * risk + min(len(text), 4000) / 100.0


def cap_units(units, max_units=MAX_UNITS):
    if len(units) <= max_units:
        return sorted(units, key=lambda u: u[0])
    scored = sorted(units, key=lambda u: unit_score(u[1], u[2]), reverse=True)
    return sorted(scored[:max_units], key=lambda u: u[0])


def read_source(cid):
    """Return the contract's bytecode hex (the 'source' stage-1 disassembles)."""
    return bytecode_map.get(str(cid), "")


# Quick smoke test on a handful of contracts.
_smoke = []
for _cid in df["contractID"].head(20):
    _smoke.append(len(cap_units(split_solidity_units(read_source(_cid)))))
log(f"Disassembler smoke test on 20 contracts: units = {_smoke}")
_demo = cap_units(split_solidity_units(read_source(df["contractID"].iloc[0])))
log("Sample unit[0] (first 6 lines):")
for _ln in _demo[0][1].splitlines()[:6]:
    log("    " + _ln)'''

SUMMARY_MD = r"""
## Summary

**dive-dedup-4 = the dive-dedup-3 GraphCodeBERT engine, fed bytecode-as-assembly.** Same locked
dedup benchmark, same set-Transformer / loss / calibration / sampler / EMA / gates / arm table as
source-modality dedup-3; the only change is that units are readable EVM assembly basic blocks
disassembled from the (CBOR-stripped) bytecode rather than Solidity functions.

**Pipeline.**
1. Resolve each contract's hex bytecode (real → arm `Bytecode_filled.csv`; synthetic →
   `final/Synthetic_Bytecode.csv`).
2. Strip the solc CBOR trailer → disassemble → decode PUSH operands + 4-byte selectors →
   segment into `JUMPDEST`-delimited basic blocks → risk-rank and cap to `MAX_UNITS`.
3. Frozen GraphCodeBERT mean-pools each block (≤`UNIT_TOKENS` tokens) into a 768-d vector —
   cached once to `cache/dive_dedup4_units_<arm>.npz`.
4. The set-Transformer aggregates the block set (+ dispatcher/body type embedding) and feeds
   dedup-3's exact head (8 per-class MLPs + aux BCE), trained with the same loss / sampler /
   EMA / calibration / threshold-tuning / checkpointing.

**Outputs (`/kaggle/working/`, suffixed by arm: `dedup` / `synth-dedup`).**
- `cache/dive_dedup4_units_<arm>.npz` — frozen block embeddings (reused on resume).
- `best_model_<arm>.pt`, `last_state_<arm>.pt`, `metrics_arm_<arm>.json`, `history_<arm>.csv`.
- `loss_curves_<arm>.png`, `f1_macro_curves_<arm>.png`,
  `per_class_f1_train_val_test_<arm>.png`, `per_class_overfit_gap_<arm>.png`.

**What to read first.**
1. Gates: leakage, locked-test (3,756 rows), bytecode coverage, units/contract, stage-1 timing.
2. **dedup-4 vs dedup-3** `test_calibrated_tuned.f1_macro` on the same splits — does bytecode-as-
   assembly transfer through GraphCodeBERT as well as source?
3. **`delta macro-F1 (synth-dedup − dedup)`** — augmentation effect on this modality.
4. Per-class F1 / overfit gap on Bad Randomness / Front Running / DoS — the rare classes.
"""


def main():
    nb = json.loads(io.open(SRC_NB, encoding="utf-8").read())
    cells = nb["cells"]

    # --- 1) splice config: keep dedup-3 hparams (Labels..Checkpointing) verbatim
    cfg = "".join(cells[5]["source"])
    a = cfg.index("# -- Labels")
    b = cfg.index("# -- Verification gates")
    hparam_middle = cfg[a:b].rstrip("\n")
    # the dedup-3 Stage-1 comment says "function-level units"; relabel for bytecode
    hparam_middle = hparam_middle.replace(
        "function-level units + frozen GraphCodeBERT",
        "basic-block units + frozen GraphCodeBERT")
    new_config = CONFIG_TOP + hparam_middle + "\n" + CONFIG_TAIL
    cells[5]["source"] = new_config.splitlines(keepends=True)

    # --- 2) title + narrative cells -----------------------------------------
    cells[0] = md(TITLE_MD)
    cells[14] = md(SPLITTER_MD)
    cells[15]["source"] = SPLITTER_CODE.splitlines(keepends=True)
    cells[48] = md(SUMMARY_MD)

    # --- 3) fold-load cell: prepend bytecode loader + swap coverage to bytecode
    src11 = "".join(cells[11]["source"])
    assert "sol_path(c).exists()" in src11, "fold-load coverage anchor not found"
    src11 = src11.replace(
        '    have = d["contractID"].apply(lambda c: sol_path(c).exists())',
        '    d["contractID"] = d["contractID"].astype(str)\n'
        '    have = d["contractID"].apply(lambda c: c in bytecode_map)')
    src11 = src11.replace("rows | source coverage", "rows | bytecode coverage")
    src11 = src11.replace('rows with no .sol', 'rows with no bytecode')
    cells[11]["source"] = (BYTECODE_LOAD + src11).splitlines(keepends=True)

    # --- 4) global cosmetic renames across all cells -------------------------
    for c in cells:
        s = "".join(c["source"])
        s2 = (s.replace("dive-dedup-3", "dive-dedup-4")
                .replace("dive_dedup3", "dive_dedup4")
                .replace("dive_dedup1", "dive_dedup4"))
        if s2 != s:
            c["source"] = s2.splitlines(keepends=True)

    # --- 5) clear outputs / execution counts ---------------------------------
    for c in cells:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # --- 6) normalize (cell ids) + write ------------------------------------
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
