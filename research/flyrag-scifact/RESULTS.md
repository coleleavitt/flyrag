# FlyRAG on BEIR SciFact: results

Run of `PREREG.md` (`flyrag_scifact 20`; code committed at `953dafd` before its first run). Raw
output: `results/flyrag_scifact.{md,json,log}`. Per-query nDCG@10 averaged over 20 seeds; 95%
bootstrap CIs over the 300 test queries.

| system | nDCG@10, all 300 | unseen 104 | seen 196 |
|---|---|---|---|
| FlyRAG frozen | 0.065 | 0.065 | 0.064 |
| FlyRAG learning | 0.086 | 0.068 | 0.096 |
| FlyRAG shuffled | 0.062 | 0.068 | 0.058 |
| FlyRAG independent | 0.052 | 0.067 | 0.043 |
| BM25 (context, no bar) | **0.660** | 0.678 | 0.651 |

| bar | value | verdict |
|---|---|---|
| R1. learning − frozen > 0, CI excluding 0 | +0.022 [+0.016, +0.028] | **met** |
| R2. learning − independent > 0, CI excluding 0 | +0.035 [+0.024, +0.047] | **met** |
| R3. unseen subset: learning − frozen ≥ −0.005, or CI including 0 | +0.003 [+0.000, +0.006] | **met** |
| (reference) seen subset: learning − frozen | +0.031 [+0.023, +0.040] | — |
| (reference) shuffled − frozen | −0.003 [−0.008, +0.001] | — |

## Reading
- **Outcome learning transfers to new queries on real text.** Learning from the train queries'
  simulated outcomes lifts the test queries by +0.022, almost all of it on the 196 test queries
  whose relevant documents were also relevant in training (+0.031). Queries with no such overlap
  are not harmed (+0.003). Uninformative feedback (independent) *lowers* nDCG (0.052 vs 0.065):
  demoting whatever is shown hurts, and only the outcome's information turns it into a gain.
- **FlyRAG's base retrieval is far below BM25 on real text: 0.065 vs 0.660.** Our BM25 matches
  BEIR's published SciFact BM25 (0.665), so the loader and metric are sound. The learning gain is
  real but small next to that gap. INFERRED cause (not tested): the text features have no IDF, so
  frequent words ("the", "of", "in") dominate each 1,024-bucket vector, and the k = 64
  winner-take-all code then mostly encodes them. An IDF-weighted front end is the obvious next
  design, to be preregistered on its own.

## Limits
The user is simulated from relevance labels (real text, real relevance, not real user outcomes).
Labels are sparse: 2–3% of feedback is `helped`. One dataset. FlyRAG's constants were set for the
synthetic benchmark and not tuned here.
