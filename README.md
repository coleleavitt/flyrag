# FlyRAG

Retrieval that **learns from verified usefulness**, built on the fruit-fly mushroom body
([kenyon](https://github.com/coleleavitt/kenyon)). Each document has two readout banks over sparse
Kenyon-cell units, "use" and "skip". A verified outcome depresses one of them (helped → skip,
misled → use) on the query's active units, with a bounded, prediction-error-sized step, credit only
for documents that were actually shown, and decay back to the prior. Nothing is overwritten, so
lessons stay in their context.

**Recommended use: a learning layer over a retriever you already have.**

```rust
use flyrag::{DocId, FlyRag, Outcome, RagParams};

fn main() -> Result<(), flyrag::FlyRagError> {
let mut rag = FlyRag::new(RagParams::prereg(7))?;
for doc in ["install with bun install", "install with npm install"] {
    rag.add(doc)?;
}
// Candidates from your retriever (BM25, vectors, …): (doc, score), higher is better.
let candidates = [(DocId(1), 3.2), (DocId(0), 3.0)];
let a = rag.rerank("how do I install", &candidates, 10)?;
// Later, a verified outcome for an offered document:
rag.feedback(a.offer, DocId(0), Outcome::Helped)?;
Ok(())
}
```

## Evidence (all preregistered: bars and code committed before each run)

| study | result |
|---|---|
| [Re-ranker over BM25, BEIR SciFact](research/flyrag-next/) | nDCG@10 0.660 → **0.676** (+0.016 [+0.009, +0.023]); random labels gain nothing; unrelated queries unchanged. All bars met. |
| [Own retrieval, SciFact](research/flyrag-scifact/) | Learning transfers to new queries (+0.022) but base retrieval is weak: 0.065 vs BM25 0.660. With IDF and 16k buckets: 0.249. |
| [Synthetic, lexically misleading](research/flyrag/) | Learning beats a frozen copy by +0.034; keeps 83% of its gain after new topics; a strict control (labels independent of documents) gains nothing. |
| Novelty detection | **Not met.** Against past queries AUC 0.64 (a leak inflated the first report to 0.997); against stored documents 0.74. Bar was 0.90. |

What it does **not** claim: better raw search than BM25, HNSW or learned embeddings.

## Layout
- `src/`: the `flyrag` crate (dependencies: `kenyon`, `thiserror`).
- `bench/`: the evaluation harness (`flyrag-bench`, unpublished): `flyrag_eval`,
  `flyrag_followup`, `flyrag_scifact`, `flyrag_next`.
- `research/`: preregistrations and results; `results/`: raw outputs.
- `scripts/ci.sh` (the CI checks), `scripts/reproduce.sh` (every study; byte-identical outputs).

kenyon is a private git dependency pinned by rev until it is published. History before the split
lives in kenyon (`github.com/coleleavitt/kenyon`, up to `70ed335`).

## License
MIT OR Apache-2.0.
