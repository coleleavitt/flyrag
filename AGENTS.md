# FlyRAG agent guide

Read `CLAUDE.md`; this mirrors its core rules.
- Learning retrieval on kenyon (git dep, pinned rev). Library in `src/`, evaluations in `bench/`,
  preregistrations and results in `research/` and `results/`.
- Bars and evaluation code are committed before any run; changes are amendments; missed bars are
  reported as missed. `RagParams::prereg` never changes.
- `forbid(unsafe_code)`, no `unwrap` outside tests, `thiserror` errors; runtime deps exactly
  `kenyon` + `thiserror`. Run `scripts/ci.sh` before committing.
- No secrets, no datasets in the repo, no `cargo publish`.
