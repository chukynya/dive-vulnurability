# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This repo is **dataset-only** — no source code, build system, package manifest, or git history. It is the data backing the sibling research project at `../sc-vulnerability-detector/` (a Jupyter/Python pipeline for Ethereum smart contract vulnerability detection). Any modeling, EDA, or training code lives there, not here. If asked to build/train/evaluate, the work belongs in that sibling directory and reads from this one.

## Dataset layout

All data sits under `Dataset/`:

- **`Source/`** — 22,330 Solidity source files named `{contractID}.sol` (IDs are contiguous: `1.sol` … `22330.sol`). These are Etherscan-verified contracts; the leading comment in each file records the verification date and original `pragma`.
- **`Bytecode_filled.csv`** (~330 MB) — columns: `contractID, contractAddress, bytecode`. One row per contract. "Filled" indicates a pass has been run to backfill missing bytecode entries — do not assume the raw on-chain bytecode is byte-identical.
- **`Transaction-based.csv`** — per-contract transaction-level features. Columns: `contractID, NoOfTransactions, blockNumber, timeStamp, hash, nonce, blockHash, transactionIndex, from, to, value, gas, gasPrice, isError, txreceipt_status, cumulativeGasUsed, gasUsed, confirmations, methodId, functionName, Proxy, Implementation`. `Proxy=1` rows carry an `Implementation` address; treat those contracts as proxy/impl pairs when joining.
- **`DIVE_Labels.csv`** — ground-truth multi-label vulnerability annotations. Columns: `contractID` plus 8 binary class columns: `Reentrancy, Access Control, Arithmetic, Unchecked Return Values, DoS, Bad Randomness, Front Running, Time manipulation`. Labels are **not mutually exclusive** — a contract can be positive in multiple columns.

## Key invariants when working across files

- **`contractID` is the universal join key** across all four sources (Source filename stem, Bytecode CSV, Transaction CSV, Labels CSV). Always join/filter on it rather than on `contractAddress`, since the same address can appear under different IDs for proxy/implementation relationships.
- **File size warning**: `Bytecode_filled.csv` is ~330 MB and `Source/` has 22k files. Never read these in full with `Read` or globs that materialize every match — stream with pandas (`chunksize=`), grep with the dedicated tool, or sample by `contractID` before loading.
- Solidity source files span a wide pragma range (0.4.x through 0.8.x). Do not assume a single compiler version when parsing.

## Commands

There are no build, lint, or test commands — this repo ships data only. If you need to regenerate `Bytecode_filled.csv` or rebuild labels, the relevant notebooks (`fill_bytecode.ipynb`, label-construction notebooks) are in `../sc-vulnerability-detector/`.

## Coding principles (Karpathy-inspired)

Source: https://github.com/multica-ai/andrej-karpathy-skills

**Think before coding** — State assumptions explicitly rather than guessing. Present multiple interpretations when ambiguity exists. Push back when a simpler approach is available. Request clarification when confused.

**Simplicity first** — Write the minimum code that solves the stated problem. Avoid speculative features or unused abstractions. No error handling for impossible scenarios.

**Surgical changes** — Modify only what the request requires. Match existing code style without unsolicited "improvements". Remove only imports/variables made unused by your own changes; mention unrelated dead code rather than deleting it.

**Goal-driven execution** — Transform tasks into verifiable success criteria. State a brief multi-step plan with verification checkpoints before acting. Use tests to define and confirm success.
