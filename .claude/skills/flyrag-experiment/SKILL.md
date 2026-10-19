---
name: flyrag-experiment
description: Run a new FlyRAG evaluation the predeclared way — bars, controls and code committed before the run, seeded, results written to results/, honest verdict. Use whenever a FlyRAG task produces a number that will be reported.
---

# FlyRAG experiment workflow

FlyRAG's results are trusted because every bar was fixed before the run. Keep it that way.

## 1. Predeclare (commit before any code runs)
Write `research/<study>/PREREG.md` (or an amendment to an existing one) with:
- the question, and what FlyRAG does **not** claim;
- the system: `RagParams::prereg(seed)` plus each changed field, named;
- data: source URL, sha256, splits, and any property of the data known in advance;
- arms, always including a **frozen** twin and an outcome-free control (labels independent of
  the documents at the learning arm's rate — the "independent" arm);
- metric with its exact formula, seeds (1–20; dev seeds 1000–1004 for any choice), statistics
  (paired bootstrap CIs, 10,000 resamples, over seeds or over test queries — say which);
- bars as met / not met, and limitations stated in advance.

Commit it, then commit the evaluation code (`bench/src/bin/<study>.rs`) **before its first run**.

## 2. Implement in `bench/`
- Reuse `synthetic.rs`, `scifact.rs`, `stats.rs`; every random choice from a named `Rng::derive`
  stream; print seeds and git HEAD; write `results/<study>.{md,json}`.
- Add the command to `scripts/reproduce.sh`.
- If the library changed, show the earlier studies reproduce byte-for-byte (diff the JSON).

## 3. Run and report
- Run once on the evaluation seeds. Any change after the bar is an **amendment**, committed before
  the affected run (see `research/flyrag/PREREG.md` A1–A3 for the pattern).
- Write `research/<study>/RESULTS.md`: the numbers, met / not met per bar, the reading (mark
  INFERRED explanations as such), limits. Update the README's evidence table.

## 4. Never
- tune on the evaluation seeds or the test split;
- re-score a missed bar with a friendlier metric without reporting both;
- drop a control because it made the result look worse;
- claim FlyRAG beats BM25, HNSW or learned embeddings on raw search without a predeclared
  head-to-head.
