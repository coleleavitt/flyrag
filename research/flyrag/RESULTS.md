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
- **Novelty: the preregistered yes/no flag missed its bar (AUC 0.507).** The continuous score's
  AUC of 0.997 reported here was **inflated by a leak**: the in-domain probes had already been sent
  during evaluation, so the filter had stored them. With fresh probes the score's AUC is 0.643; see
  the follow-up below.
- **No catastrophic forgetting:** learning 20 more topics kept 83% of the earlier gain.

## Limits
Synthetic benchmark, built so that lexical similarity misleads. It shows the mechanism works when
outcomes carry information the text does not; it says nothing yet about how often that happens on
real data, and nothing about raw search quality against HNSW or learned embeddings.

## Follow-up (amendments A2, A3)
`flyrag_followup` (code committed at `953dafd`, before its first run). Raw output:
`results/flyrag_followup{,_dev}.{md,json}`. Seeds 1–20 unless stated; 95% bootstrap CIs over seeds.

| bar | value | verdict |
|---|---|---|
| 2b. independent − frozen < +0.01, or CI including 0 | −0.001 [−0.005, +0.003] | **met** |
| 2c. learning − independent > 0, CI excluding 0 | +0.035 [+0.029, +0.042], d_z 2.43 | **met** |
| 3 (re-test). flag AUC ≥ 0.90, fresh probes, flag chosen on dev seeds (t = 0.10, o = 4) | 0.637 [0.614, 0.658] | **not met** |
| (secondary) continuous novelty score AUC, fresh probes | 0.643 [0.618, 0.669] | — |

- **The learning gain is caused by the outcome signal.** The strict control gives the same
  amount and mix of feedback (2.1% `helped`) with labels independent of the documents, and gains
  nothing. The first run's shuffled control gained +0.007 because permuting labels among the top 3
  keeps most `misled` labels on distractors.
- **Novelty detection does not work on this benchmark.** Without the leak, a fresh in-domain query
  looks almost as novel as an out-of-domain one (score AUC 0.643; the best flag on the dev seeds
  reached 0.618). INFERRED reason: the filter stores past *queries*, and a new 10-word query
  (random topic words plus common words, hashed with bigrams) activates a mostly new set of KC
  units whatever its topic. Testing novelty against the stored *documents*, or on unigram codes,
  would be a new preregistered design; neither is claimed here.
