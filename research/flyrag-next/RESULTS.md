# FlyRAG next steps: results

Run of `PREREG.md` (`flyrag_next 20`; code committed at `b818c98` before its first run). Raw
output: `results/flyrag_next.{md,json}`. Earlier studies were re-run after the library changes and
reproduce byte-for-byte (`flyrag_eval`, `flyrag_followup`, `flyrag_scifact`).

## S1. FlyRAG as a learning re-ranker over BM25 — all bars met

| system (SciFact test, nDCG@10) | all 300 | unseen 104 |
|---|---|---|
| BM25 | 0.660 | 0.678 |
| BM25 + FlyRAG, frozen | 0.660 (identical ranking on 300/300) | 0.678 |
| **BM25 + FlyRAG, learning** | **0.676** | 0.678 |
| BM25 + FlyRAG, independent labels | 0.658 | 0.678 |

| bar | value | verdict |
|---|---|---|
| S1.1 learning − BM25 > 0, CI excluding 0 | +0.016 [+0.009, +0.023] | **met** |
| S1.2 learning − independent > 0, CI excluding 0 | +0.018 [+0.012, +0.025] | **met** |
| S1.3 unseen: learning − BM25 ≥ −0.005, or CI including 0 | −0.000 [−0.000, +0.000] | **met** |

Learning from the train queries' outcomes makes a strong lexical retriever better on new queries,
the gain needs the outcome's information (random labels slightly hurt), and queries unrelated to
the training ones are untouched. With BM25 supplying good candidates, 24% of feedback is `helped`
(vs 2–3% on FlyRAG's own candidates), so there is much more to learn from.

## S2. Weighted front end — S2.1 met, S2.2 not met

| front end (frozen) | nDCG@10 |
|---|---|
| A: d = 1024 (prereg) | 0.065 |
| B: d = 1024 + IDF | 0.104 |
| C: d = 16,384 | 0.114 |
| D: d = 16,384 + IDF | 0.249 |
| BM25 (context) | 0.660 |

| bar | value | verdict |
|---|---|---|
| S2.1 D − A > 0, CI excluding 0 | +0.184 [+0.159, +0.211] | **met** |
| S2.2 D ≥ 0.30 | 0.249 | **not met** |
| (secondary) learning − frozen under D | +0.040 [+0.032, +0.049] | — |

Both changes help and they compound: D is 3.9× A. It is still well below BM25: a k = 64 sparse
code keeps only the strongest features of each text, which costs exact-term matching. The learning
gain grows with the better front end (+0.040 vs +0.022 under A).

## S3. Novelty against stored documents — not met

| bar | value | verdict |
|---|---|---|
| S3.1 Bloom novelty AUC ≥ 0.90 (fresh probes, synthetic) | 0.743 [0.721, 0.768] | **not met** |
| (secondary) 1 − best_overlap / k | 0.674 [0.651, 0.696] | — |

Better than novelty against past queries (0.643), not good enough. INFERRED reason: short queries
and long documents share few active KC units even on topic, so "in-domain" and "out-of-domain"
codes overlap the store about equally sparsely.

## What changes
- FlyRAG's best use today is the S1 shape: **a learning layer over an existing retriever**
  (`FlyRag::rerank`), not a retriever of its own.
- The novelty claim is withdrawn for text queries until a design passes a bar.

## Limits
Simulated user from relevance labels; one real dataset; S3 synthetic only; nothing tuned.
