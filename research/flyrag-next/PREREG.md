# FlyRAG next steps: re-ranker, weighted front end, document novelty — preregistration

Written and committed **before** any code for these studies exists or runs. Changes after this
commit are amendments, committed before the runs they affect. Background: `research/flyrag/RESULTS.md`
(novelty against past queries fails) and `research/flyrag-scifact/RESULTS.md` (learning transfers,
but base retrieval is 0.065 nDCG@10 vs BM25 0.660).

Shared protocol unless stated: BEIR SciFact as in `research/flyrag-scifact/PREREG.md` (same data,
train/test splits, simulated user trying the top 3, `top_n = 10`, learning off for evaluation,
nDCG@10 per test query averaged over seeds 1–20, paired 95% bootstrap CIs over the 300 test
queries with 10,000 resamples, the 104-query unseen subset). FlyRAG constants are
`RagParams::prereg(seed)` except where a study changes one named field. Nothing is tuned.

## S1. FlyRAG as a learning re-ranker over BM25
**Question:** does FlyRAG's outcome learning improve a strong retriever, rather than its own weak one?

- Candidates: BM25 (k1 = 0.9, b = 0.4, FlyRAG's unigram tokens; the scorer from
  `flyrag_scifact`) top 100 per query.
- New API `FlyRag::rerank(text, candidates, top_n)`: each candidate's score is
  `bm25 / max_bm25_over_candidates + λ · learned(doc, q)`, `λ = 0.5`; the offer is the re-ranked
  top `top_n`, so feedback works exactly as for `query`.
- Arms: **BM25** alone; **frozen** (re-rank with no feedback; must equal BM25's ranking exactly,
  a sanity check, not a bar); **learning**; **independent** (labels `helped` with the learning
  arm's per-seed rate, independent of the document).
- Bars:
  - **S1.1** learning − BM25 > 0, CI excluding 0.
  - **S1.2** learning − independent > 0, CI excluding 0.
  - **S1.3** unseen subset: learning − BM25 ≥ −0.005, or CI including 0.

## S2. A weighted text front end
**Question:** how much of the gap to BM25 is the front end (no IDF, 1,024 buckets)?

- New option: IDF per feature bucket, fitted on the corpus before any document is added:
  `idf_b = ln(1 + (N − df_b + 0.5) / (df_b + 0.5))`, `df_b` = number of documents with any
  feature in bucket `b`; the feature value becomes `(1 + ln tf) · idf_b`.
- Four configurations, frozen (no learning), base retrieval with `FlyRag::query`:
  **A** prereg (d = 1024, no IDF); **B** d = 1024 + IDF; **C** d = 16,384, no IDF;
  **D** d = 16,384 + IDF. (m = 8192, fan-in 16, k = 64 unchanged.)
- Bars:
  - **S2.1** D − A > 0, CI excluding 0.
  - **S2.2** D's nDCG@10 ≥ 0.30 (point estimate; about half of BM25).
- Secondary (no bar): learning − frozen under D (does the SciFact R1 result hold with a better
  front end?), with its CI.

## S3. Novelty against stored documents
**Question:** the first design asked "have I seen a query like this?", which fails because every
new query is new. Does "is there anything like this in the store?" work?

- New option `NoveltySource::Documents`: the fly Bloom filter is filled with the codes of the
  added documents; queries are scored against it and not inserted.
  (`NoveltySource::Queries` is the original behaviour and stays the prereg default.)
- Benchmark: the synthetic one (`research/flyrag/PREREG.md`), with amendment A2.1's fresh probes
  (40 in-domain, 40 out-of-domain per seed, each sent once in a seeded order) on a store with the
  documents added and no training.
- Scores: the Bloom novelty (primary) and `1 − best_overlap / k` (secondary).
- Bar: **S3.1** the primary score's AUC (out-of-domain positive), mean over seeds 1–20, ≥ 0.90.
  It is a continuous score, so no threshold is chosen.

Reported as met / not met with the numbers; a missed bar is reported as missed.

## Limitations stated in advance
SciFact's user is simulated from relevance labels. S3 is synthetic only: SciFact has no
out-of-domain query set. S2 changes two things (IDF, dimension) in a 2×2 design so each can be seen
alone; it is not a search over front ends.
