# FlyRAG next steps (research/flyrag-next/PREREG.md)

Seeds 1..=20; HEAD `b818c98`. SciFact: per-query nDCG@10 averaged over seeds, 95% bootstrap CIs over the 300 test queries; S3: CIs over seeds. 10,000 resamples.

## S1. Learning re-ranker over BM25 (top 100)

| system | nDCG@10 (300) | unseen 104 |
|---|---|---|
| BM25 | 0.6601 | 0.6778 |
| re-rank frozen | 0.6601 | 0.6778 |
| re-rank learning | 0.6761 | 0.6777 |
| re-rank independent | 0.6580 | 0.6777 |

Sanity: re-rank frozen differs from BM25 on 0 of 300 queries (expected 0).

| bar | value | verdict |
|---|---|---|
| S1.1 learning − BM25 > 0, CI excl. 0 | +0.0160 [+0.0092, +0.0234] | **met** |
| S1.2 learning − independent > 0, CI excl. 0 | +0.0181 [+0.0118, +0.0248] | **met** |
| S1.3 unseen: learning − BM25 ≥ −0.005 or CI incl. 0 | -0.0001 [-0.0004, +0.0000] | **met** |

## S2. Front end (frozen base retrieval)

| front end | nDCG@10 (300) |
|---|---|
| A d=1024 | 0.0646 |
| B d=1024 + IDF | 0.1042 |
| C d=16384 | 0.1135 |
| D d=16384 + IDF | 0.2489 |
| (BM25, context) | 0.6601 |

| bar | value | verdict |
|---|---|---|
| S2.1 D − A > 0, CI excl. 0 | +0.1843 [+0.1588, +0.2112] | **met** |
| S2.2 D ≥ 0.30 | 0.2489 | **not met** |
| (secondary) learning − frozen under D | +0.0401 [+0.0318, +0.0488] | — |

## S3. Novelty against stored documents (synthetic, fresh probes)

| bar | value | verdict |
|---|---|---|
| S3.1 Bloom novelty AUC ≥ 0.90 | 0.743 [0.721, 0.768] | **not met** |
| (secondary) 1 − best_overlap/k AUC | 0.674 [0.651, 0.696] | — |
