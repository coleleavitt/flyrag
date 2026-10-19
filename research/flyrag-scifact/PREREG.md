# FlyRAG on real data (BEIR SciFact): preregistration

Written and committed **before** any code for this study exists or runs. Changes after this commit
are amendments, committed before the runs they affect.

## Question
On real text with real relevance labels, does FlyRAG's outcome learning on one set of queries
improve ranking on a *different* set of queries, and is the gain caused by the outcome signal?

## Data
BEIR SciFact (Wadden et al. 2020; BEIR, Thakur et al. 2021), downloaded from the BEIR mirror
(`scifact.zip`, sha256 `536e1444…c490fa8e0165`) into `~/datasets/scifact` (outside the repo).
- Corpus: 5,183 abstracts; document text = title + " " + abstract, added in file order.
- Train split: 809 queries, 919 relevance labels over 565 documents.
- Test split: 300 queries, 339 labels over 283 documents. No query appears in both.
- Known before running (a property of the data, not of any system): 181 of the 283 test-relevant
  documents are also relevant to some train query, and 196 of the 300 test queries have at least
  one such document. The other 104 test queries form the **unseen** subset.

## System
`flyrag` with `RagParams::prereg(seed)` unchanged (d = 1024, m = 8192, fan-in 16, k = 64, λ = 0.5,
readout constants `ReadoutParams::PREREG`). Nothing is tuned on SciFact.

## Procedure (per seed; seeds 1–20 set the FlyHash projection)
1. Index the corpus.
2. Train: send every train query once, in a seeded random order, `top_n = 10`. A simulated user
   tries the top 3; a tried document is `helped` if the train labels mark it relevant to that
   query, else `misled`.
3. Evaluate: turn learning off, send every test query, `top_n = 10`.

## Arms
- **frozen:** step 2 sends the queries but gives no feedback.
- **learning:** feedback as in step 2.
- **shuffled:** the same 3 labels, permuted at random among the 3 tried documents.
- **independent:** each tried document gets `helped` with probability `p_s` (the fraction of
  `helped` labels the learning arm gave in seed `s`), independently of the document.

## Metric
nDCG@10 per test query with binary relevance from the test labels (ideal DCG over
min(#relevant, 10) documents), averaged over the 20 seeds per query. Differences between arms are
paired by query; 95% CIs from 10,000 bootstrap resamples of the 300 test queries.

## Bars
- **R1, learning helps on new queries:** learning − frozen > 0, CI excluding 0.
- **R2, it is the outcome signal:** learning − independent > 0, CI excluding 0.
- **R3, no harm on unseen queries:** on the 104-query unseen subset, learning − frozen ≥ −0.005,
  or its CI includes 0.

Reported as met / not met with the numbers; shuffled is reported without a bar.

## Context row (no bar)
BM25 (k1 = 0.9, b = 0.4) over the same tokens (FlyRAG's `TextEncoder::tokens`, unigrams, no
stemming), so the reader can see FlyRAG's absolute quality next to a standard lexical baseline.
FlyRAG is not expected to match BM25 on raw retrieval; this study does not test that.

## Limitations stated in advance
The user is simulated from relevance labels: that is real text and real relevance, but not real
user outcomes. Labels are sparse (about 1.1 per query), so most feedback is `misled`. Train and
test queries share documents (above), which is what makes transfer possible; the unseen subset
reports the case where they don't.
