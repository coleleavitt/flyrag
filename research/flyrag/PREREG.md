# FlyRAG evaluation — preregistration

Written and committed **before** any FlyRAG evaluation run (kenyon-experiment skill). Changes after
this commit are amendments, recorded as their own commits before the runs they affect.

## Question
Does retrieval that learns from **verified usefulness** (FlyRAG) rank the truly helpful document
higher on *new* queries than the same system without learning, and is that gain caused by the
outcome signal rather than by any feedback at all?

FlyRAG's claim is learning from outcomes and novelty detection. It does **not** claim better raw
similarity search than HNSW or learned embeddings; this study does not test that.

## System under test (fixed before running)
- **Text features:** lower-cased alphanumeric tokens; unigrams and adjacent bigrams hashed (FNV-1a,
  seeded) into `d = 1024` non-negative buckets; term weight `1 + ln(tf)`.
- **Code:** `kenyon::FlyHash`, `m = 8192` KCs, fixed fan-in 16, top `k = 64` (≈ 0.8% active),
  `Preprocess::None`.
- **Candidates:** every document whose code overlaps the query's (inverted lists).
- **Readout (the learned part):** per document, two banks over KC units, "use" and "skip", each
  starting at prior `P = 0.5`, floor `F = 0.05`. `learned(doc, q) = mean over the query's active
  units u of (w_skip[u] − w_use[u]) / (P − F)`, in [−1, 1].
- **Score:** `overlap(q, doc) / k + λ · learned(doc, q)`, `λ = 0.5`.
- **Update** (only on a verified outcome for a document that was in the offer, once per offer and
  document; forward-only): target `+1` (helped) or `−1` (misled); expected = `learned(doc, q)`;
  `δ = target − expected`; if `|δ| < 0.05` nothing is learned; else step `s = min(0.25, η·|δ|)`,
  `η = 0.3`, and the bank that the outcome *depresses* (helped → skip bank, misled → use bank)
  moves `w ← max(F, w · (1 − s))` on the query's active units. Banks are never raised by an update
  (depression-only; the opposing bank is the "unlearning" trace).
- **Decay:** every bank weight relaxes toward `P` with a half-life of 2,000 queries (lazy).
- **Novelty:** a static `kenyon::FlyBloom` over query codes; a query is flagged novel when its
  novelty ≥ 0.8 or its best overlap is < 8 of 64 units.
- **Tuning rule:** constants above are final. Any tuning uses only the dev seeds 1000–1004, never
  the evaluation seeds 1–20, and is an amendment.

## Benchmark (synthetic "support knowledge base", generated per seed)
- 40 topics. Each topic has a vocabulary of 12 topic words drawn from a 4,000-word synthetic
  lexicon (topics may share a few words), plus a shared pool of 300 common words.
- Per topic, 5 documents of 40 tokens: one **fix** document (helpful) using 4 topic words, and
  four **distractors** using 9 topic words each (lexically closer to the topic, unhelpful). All
  documents are padded with common words.
- Queries: 6 topic words sampled from the topic vocabulary plus 4 common words. Per topic, 6
  training queries and 4 held-out evaluation queries, sampled independently.
- Simulated verified outcome: the user tries the top 3 offered documents; the fix document is
  `helped`, a distractor is `misled`. Documents ranked below 3 get no outcome.

## Arms (same corpus, queries and seeds)
1. **learning:** training queries with outcomes, then evaluation with learning frozen.
2. **frozen:** training queries with no outcomes (reads only), then evaluation.
3. **shuffled:** training with the same outcome labels per offer, but each label assigned to a
   random offered document of the top 3 (breaks the doc–outcome link, keeps the amount of
   feedback).

## Metrics
- Primary: mean nDCG@10 of the fix document over the evaluation queries (graded relevance 1 for
  the fix document, 0 otherwise).
- Secondary: MRR of the fix document; the novelty flag's AUC separating evaluation queries from
  out-of-domain queries (10 random lexicon words per query, 40 per seed); forgetting: after
  learning topics 1–20 then 21–40, the gain retained on topics 1–20 relative to right after
  learning them.

## Bars (20 seeds, 1–20; paired by seed; 95% bootstrap CI over seeds, 10,000 resamples)
1. **Learning helps:** learning − frozen nDCG@10 ≥ +0.03, CI excluding 0.
2. **It is the outcome signal:** shuffled − frozen nDCG@10 < +0.01, or its CI includes 0.
3. **Novelty:** AUC ≥ 0.90.
4. **No catastrophic forgetting:** retained gain on topics 1–20 ≥ 80%.
5. **No feedback, no change:** in the frozen arm every bank weight equals the prior (unit test).

Reported as met / not met per bar, with the numbers. A missed bar is reported as missed.

## Limitations stated in advance
The benchmark is synthetic and built so lexical similarity is misleading; it shows the mechanism
works when outcomes carry information that text does not, not how often that happens in real
retrieval. A real-data evaluation (e.g. fixes labelled by later reverts) is future work.

## Amendments

### A1 (before any run): sign of the readout
The readout formula above contradicts its own update rule. With `learned = (w_skip − w_use)/(P − F)`
and "helped → depress the skip bank", a helpful document would *lose* score. The intent (the
update section, and the mushroom body: dopamine depresses the synapses onto the output neuron that
drives the opposite action) is that helped documents rise. Corrected:
`learned(doc, q) = mean over the query's active units u of (w_use[u] − w_skip[u]) / (P − F)`.
No other constant, arm, metric or bar changes. Found while writing the implementation; no
evaluation code had run.

### A2 (after the first run, before any follow-up run): novelty re-test and a stricter control
The first run's results stand as reported (`RESULTS.md`). This amendment adds a follow-up,
`flyrag_followup`, run on the same benchmark and seeds. Bars 1, 4 and 5 are not re-tested.

**A2.1 The novelty measurement leaked.** In the first run, the in-domain novelty probes were
evaluation queries that `evaluate` had already sent, so the filter had stored them and their
novelty was exactly 0. The continuous score's AUC (0.997) is therefore inflated, and the flag's
in-domain rate was set by the overlap rule alone. Fix: the in-domain probes are 40 *fresh* queries
(one per topic, same generator, drawn from a separate seeded stream so the world is unchanged).
They are mixed with the 40 out-of-domain queries in a seeded random order and each is sent once,
to the learning arm's store after its evaluation. Out-of-domain probes are the positives.

**A2.2 Choosing the flag on the dev seeds.** On seeds 1000–1004 only, sweep the flag
`novel = novelty ≥ t or best_overlap < o` over `t ∈ {0.05, 0.10, …, 1.00, off}` and
`o ∈ {0, 1, 2, 4, 6, 8}`. Pick the pair with the highest mean flag AUC over the five dev seeds;
ties go to the smaller `o`, then to the `t` closest to the preregistered 0.8. The chosen pair is
committed as amendment A3 before the evaluation seeds run. Then bar 3 (flag AUC ≥ 0.90) is
re-tested once on seeds 1–20 with that pair, and the continuous score's AUC is re-reported. The
first run's 0.507 remains the preregistered result for bar 3; this is a second, separate test.

**A2.3 A stricter shuffled control, "independent".** Each of the top 3 tried documents gets
`helped` with probability `p_s` and `misled` otherwise, drawn independently of which document it
is. `p_s` is the fraction of `helped` labels the learning arm gave in seed `s` (same amount and
mix of feedback, no document–outcome link). Bars, paired over seeds 1–20 with 95% bootstrap CIs
(10,000 resamples):
- **2b:** independent − frozen nDCG@10 < +0.01, or its CI includes 0 (the control is clean).
- **2c:** learning − independent nDCG@10 > 0 with its CI excluding 0 (the outcome's information
  adds to whatever generic feedback does).
