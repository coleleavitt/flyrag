# FlyRAG Claude Instructions

## Mission
FlyRAG is retrieval that learns from **verified** outcomes without forgetting, built on kenyon's
mushroom-body primitives. Its credibility rests on preregistered evaluations: every bar and every
evaluation's code is committed before the run, and missed bars are reported as missed.

## Architecture First
| Layer | Path | Owns | Must NOT own |
|---|---|---|---|
| primitives | `kenyon` (git dep, pinned rev) | FlyHash, FlyIndex, FlyBloom, codes | anything FlyRAG-specific |
| text | `src/text.rs` | tokens, hashed unigram+bigram features, IDF weights | ranking |
| readout | `src/readout.rs` | use/skip banks, bounded depression, lazy decay | text, candidates |
| store | `src/rag.rs` | add, `query`, `rerank`, offers, forward-only feedback, novelty | dataset code |
| harness | `bench/` (`flyrag-bench`, unpublished) | synthetic benchmark, SciFact, BM25, stats, study binaries | library behaviour |

## Current State
- Split out of kenyon on 2026-10-19 with history (kenyon commit `a6fbb04` removed it there).
- Studies: `research/flyrag/` (synthetic + amendments A1–A3), `research/flyrag-scifact/`,
  `research/flyrag-next/` (S1 re-ranker met; S2.2 front end 0.249 < 0.30; S3.1 novelty 0.743 < 0.90).
- All five result files reproduce byte-for-byte from this repo (`scripts/reproduce.sh`).
- Next candidates: persistence (save/load banks); the re-ranker on a second real dataset; each
  preregistered first.

## Working Rules
- Toolchain `rustc-master` (`rust-toolchain.toml`), plain `cargo`; CI uses nightly + MSRV 1.85.0.
- `unsafe_code = "forbid"`, `clippy::unwrap_used = "deny"`, errors via `thiserror` (`FlyRagError`).
- `flyrag`'s runtime deps are exactly `kenyon` and `thiserror` (`scripts/check_deps.sh`).
- kenyon is pinned by `rev` in the root `Cargo.toml`; bump it deliberately and re-run
  `scripts/reproduce.sh` to show results are unchanged (or record why they changed).
- Prereg defaults (`RagParams::prereg`) never change; new behaviour is a new option.
- Experiments follow `.claude/skills/flyrag-experiment/SKILL.md`.
- No secrets in the repo; datasets stay in `~/datasets`. No `cargo publish` without the operator.

## Validation
```sh
scripts/ci.sh                       # fmt, clippy, doc, deps, tests, MSRV
scripts/reproduce.sh                # every study (SciFact needs ~/datasets/scifact)
```

## Project Claude Layout
`.claude/CLAUDE.md` points here; `.claude/rules/` (code-style, security, testing);
`.claude/skills/flyrag-experiment/` (the preregistration workflow). `MEMORY.md` is local and ignored.
