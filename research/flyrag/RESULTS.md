# FlyRAG evaluation — results

Run of the preregistered evaluation (`PREREG.md`, amendment A1), `flyrag_eval 20` at commit
`d50cf4c` (evaluation code committed before its first run). Raw output: `results/flyrag.json`,
`results/flyrag.md`, `results/flyrag.log`. 20 seeds; 95% bootstrap CIs over seeds.

| arm | nDCG@10 | MRR@10 |
|---|---|---|
| learning | 0.092 | 0.069 |
| frozen | 0.058 | 0.041 |
| shuffled | 0.065 | 0.046 |

| bar | value | verdict |
|---|---|---|
| 1. learning − frozen nDCG@10 ≥ +0.03, CI excluding 0 | +0.034 [+0.028, +0.040], d_z 2.47 | **met** (narrowly) |
| 2. shuffled − frozen < +0.01, or CI including 0 | +0.007 [+0.003, +0.012] | **met** (on the point estimate; its CI excludes 0) |
| 3. novelty flag AUC ≥ 0.90 | 0.507 | **not met** |
| 4. retained gain on topics 1–20 ≥ 80% | 83% [71%, 93%] | **met** |
| 5. no feedback, no change | `no_feedback_no_change` passes | **met** |

## Reading
- **Learning from verified outcomes works, and the effect is small in absolute terms.** The
  benchmark makes the helpful document lexically weaker than four distractors, so it is rarely in
  the top 3 the simulated user tries; credit is forward-only, so a document that is never shown
  cannot learn. The gain is consistent across seeds (d_z 2.47) but lifts nDCG@10 only from 0.058
  to 0.092.
- **The shuffled control is not fully clean.** It gained +0.007 with a CI excluding 0. Shuffling
  labels among the top 3 still leaves most "misled" labels on distractors, which carries some
  information. It met the preregistered bar; a stricter control (labels drawn independently of the
  documents) is future work and will be preregistered as such.
- **Novelty: the continuous score separates in- from out-of-domain queries almost perfectly (AUC
  0.997, secondary), the preregistered yes/no flag does not (0.507).** A flag AUC of 0.5 means the
  flag fires at the same rate for both. Hypothesis (INFERRED, not tested): the "best overlap < 8 of
  64" rule flags nearly every short (10-word) query against 40-word documents. Changing the
  threshold now would be tuning on the evaluation seeds, which the preregistration forbids; a fix
  will be chosen on the dev seeds (1000–1004) and re-tested as an amendment.
- **No catastrophic forgetting:** learning 20 more topics kept 83% of the earlier gain.

## Limits
Synthetic benchmark, built so that lexical similarity misleads. It shows the mechanism works when
outcomes carry information the text does not; it says nothing yet about how often that happens on
real data, and nothing about raw search quality against HNSW or learned embeddings.
