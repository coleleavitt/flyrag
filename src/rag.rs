//! The store: add documents, query, give verified feedback.

use std::collections::{HashMap, HashSet, VecDeque};

use kenyon::{BloomParams, FlyBloom, FlyHash, FlyHashParams, FlyIndex, KcCode, Preprocess};

use crate::error::{FlyRagError, Result, invalid};
use crate::readout::{Bank, Banks, ReadoutParams};
use crate::text::TextEncoder;

/// A document id (insertion order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocId(pub u32);

/// The handle of one query's results; feedback names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OfferId(pub u64);

/// A verified outcome for an offered document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Using the document led to a verified success.
    Helped,
    /// The document was tried and led the user astray.
    Misled,
}

/// What a piece of feedback did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Learned {
    /// The bank moved by `step` (prediction error `delta`).
    Updated {
        /// `target − expected`.
        delta: f32,
        /// The step applied, `min(cap, η |δ|)`.
        step: f32,
    },
    /// The outcome was already expected (`|δ| < min_delta`): nothing changed.
    Expected,
    /// Learning is off: nothing changed.
    Frozen,
}

/// What the novelty filter remembers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoveltySource {
    /// Past queries: "have I been asked something like this?" (the prereg).
    Queries,
    /// Stored documents: "is there anything like this in the store?"
    Documents,
}

/// Every constant (the prereg's values are [`RagParams::prereg`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagParams {
    /// Feature buckets `d`.
    pub dim: usize,
    /// Kenyon cells `m`.
    pub n_kc: usize,
    /// Inputs per KC.
    pub fan_in: usize,
    /// Active KCs per code `k`.
    pub k: usize,
    /// Weight of the learned term in the score.
    pub lambda: f32,
    /// The readout constants.
    pub readout: ReadoutParams,
    /// Novelty at or above this flags the query novel.
    pub novelty_threshold: f64,
    /// A best overlap below this flags the query novel.
    pub min_overlap: u32,
    /// Offers kept for feedback; older ones are evicted.
    pub max_offers: usize,
    /// What the novelty filter is filled with.
    pub novelty_source: NoveltySource,
    /// Seed for the encoder and the projection.
    pub seed: u64,
}

impl RagParams {
    /// The preregistered system (research/flyrag/PREREG.md).
    pub fn prereg(seed: u64) -> Self {
        Self {
            dim: 1024,
            n_kc: 8192,
            fan_in: 16,
            k: 64,
            lambda: 0.5,
            readout: ReadoutParams::PREREG,
            novelty_threshold: 0.8,
            min_overlap: 8,
            max_offers: 10_000,
            novelty_source: NoveltySource::Queries,
            seed,
        }
    }

    fn validate(&self) -> Result<()> {
        let r = &self.readout;
        if !(r.floor >= 0.0 && r.floor < r.prior) {
            return Err(invalid("floor", "must satisfy 0 <= floor < prior"));
        }
        if !(r.cap > 0.0 && r.cap < 1.0) {
            return Err(invalid("cap", "must be in (0, 1)"));
        }
        if r.eta.is_nan() || r.eta <= 0.0 {
            return Err(invalid("eta", "must be > 0"));
        }
        if self.lambda.is_nan() || self.lambda < 0.0 {
            return Err(invalid("lambda", "must be >= 0"));
        }
        if self.max_offers == 0 {
            return Err(invalid("max_offers", "must be >= 1"));
        }
        Ok(())
    }
}

/// One ranked result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ranked {
    /// The document.
    pub doc: DocId,
    /// Shared active KCs with the query (of `k`).
    pub overlap: u32,
    /// The learned term in [−1, 1].
    pub learned: f32,
    /// `overlap / k + λ · learned` from [`FlyRag::query`];
    /// `base / max_base + λ · learned` from [`FlyRag::rerank`].
    pub score: f64,
}

/// A query's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The handle for feedback on these results.
    pub offer: OfferId,
    /// Ranked results, best first.
    pub hits: Vec<Ranked>,
    /// Fly Bloom novelty of the query, in [0, 1] (1 = nothing like it seen).
    pub novelty: f64,
    /// The largest code overlap of any stored document (0 if none), before
    /// ranking and truncation.
    pub best_overlap: u32,
    /// Flagged novel: high novelty, or nothing in the store overlaps enough.
    pub novel: bool,
}

#[derive(Debug, Clone)]
struct Offer {
    units: Vec<u32>,
    docs: HashSet<u32>,
    answered: HashSet<u32>,
}

/// A FlyRAG store.
#[derive(Debug, Clone)]
pub struct FlyRag {
    params: RagParams,
    encoder: TextEncoder,
    hasher: FlyHash,
    /// Per-bucket feature weights (IDF), if fitted.
    idf: Option<Vec<f32>>,
    index: FlyIndex,
    banks: Vec<Banks>,
    novelty: FlyBloom,
    clock: u64,
    offers: HashMap<u64, Offer>,
    order: VecDeque<u64>,
    next_offer: u64,
    learning: bool,
}

impl FlyRag {
    /// An empty store.
    pub fn new(params: RagParams) -> Result<Self> {
        params.validate()?;
        let hasher = FlyHash::new(FlyHashParams {
            n_inputs: params.dim,
            n_kc: params.n_kc,
            fan_in: params.fan_in,
            k: params.k,
            preprocess: Preprocess::None,
            seed: params.seed,
        })?;
        let novelty = FlyBloom::new(hasher.clone(), BloomParams::STATIC)?;
        Ok(Self {
            params,
            encoder: TextEncoder::new(params.dim, params.seed),
            index: FlyIndex::new(hasher.n_kc()),
            hasher,
            idf: None,
            banks: Vec::new(),
            novelty,
            clock: 0,
            offers: HashMap::new(),
            order: VecDeque::new(),
            next_offer: 0,
            learning: true,
        })
    }

    /// The constants.
    pub fn params(&self) -> &RagParams {
        &self.params
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }

    /// Turn learning on or off. Off, feedback is accepted and changes nothing.
    pub fn set_learning(&mut self, on: bool) {
        self.learning = on;
    }

    /// Queries served so far (the decay clock).
    pub fn clock(&self) -> u64 {
        self.clock
    }

    fn code(&self, text: &str) -> Result<KcCode> {
        let x = match &self.idf {
            Some(w) => self.encoder.encode_weighted(text, w),
            None => self.encoder.encode(text),
        };
        Ok(self.hasher.hash(&x)?)
    }

    /// Weight every feature bucket by its IDF over `corpus`,
    /// `ln(1 + (N − df + 0.5) / (df + 0.5))`, where `df` counts the texts
    /// with any feature in that bucket. Must be called before any document
    /// is added (codes depend on it); the corpus need not be what is added.
    pub fn fit_idf<S: AsRef<str>>(&mut self, corpus: &[S]) -> Result<()> {
        if !self.is_empty() {
            return Err(invalid("idf", "fit before adding documents"));
        }
        if corpus.is_empty() {
            return Err(invalid("idf", "corpus is empty"));
        }
        let mut df = vec![0u32; self.params.dim];
        for t in corpus {
            for b in self.encoder.buckets(t.as_ref()) {
                df[b] += 1;
            }
        }
        let n = corpus.len() as f64;
        self.idf = Some(
            df.into_iter()
                .map(|d| {
                    let d = f64::from(d);
                    (1.0 + (n - d + 0.5) / (d + 0.5)).ln() as f32
                })
                .collect(),
        );
        Ok(())
    }

    /// `true` once [`FlyRag::fit_idf`] has run.
    pub fn has_idf(&self) -> bool {
        self.idf.is_some()
    }

    /// Add a document.
    pub fn add(&mut self, text: &str) -> Result<DocId> {
        let code = self.code(text)?;
        if self.params.novelty_source == NoveltySource::Documents {
            self.novelty.insert_code(&code)?;
        }
        let id = self.index.insert(code)?;
        if id as usize != self.banks.len() {
            return Err(FlyRagError::Full);
        }
        self.banks.push(Banks::default());
        Ok(DocId(id))
    }

    /// Rank the documents for `text` by code overlap plus the learned term,
    /// and keep the offer for feedback. With [`NoveltySource::Queries`] the
    /// query is recorded in the novelty filter.
    pub fn query(&mut self, text: &str, top_n: usize) -> Result<Answer> {
        let k = self.params.k as f64;
        self.answer(text, top_n, |cands| {
            cands
                .iter()
                .map(|&(id, ov)| (id, f64::from(ov) / k))
                .collect()
        })
    }

    /// Re-rank another retriever's `candidates` (document, base score; higher
    /// is better): score `base / max_base + λ · learned`. The offer is the
    /// re-ranked top `top_n`, so [`FlyRag::feedback`] works as for
    /// [`FlyRag::query`]. Unknown documents are an error.
    pub fn rerank(
        &mut self,
        text: &str,
        candidates: &[(DocId, f64)],
        top_n: usize,
    ) -> Result<Answer> {
        if let Some(&(d, _)) = candidates.iter().find(|(d, _)| d.0 as usize >= self.len()) {
            return Err(invalid("candidates", format!("unknown document {}", d.0)));
        }
        let max = candidates
            .iter()
            .map(|c| c.1)
            .filter(|v| v.is_finite())
            .fold(0.0f64, f64::max);
        self.answer(text, top_n, |_| {
            candidates
                .iter()
                .map(|&(d, b)| {
                    (
                        d.0,
                        if max > 0.0 && b.is_finite() {
                            b / max
                        } else {
                            0.0
                        },
                    )
                })
                .collect()
        })
    }

    /// Shared by `query` and `rerank`: `base` maps the overlap candidates
    /// (id, overlap) to the (id, base score) list to rank.
    fn answer(
        &mut self,
        text: &str,
        top_n: usize,
        base: impl FnOnce(&[(u32, u32)]) -> Vec<(u32, f64)>,
    ) -> Result<Answer> {
        let code = self.code(text)?;
        let novelty = self.novelty.novelty_code(&code)?;
        if self.params.novelty_source == NoveltySource::Queries {
            self.novelty.insert_code(&code)?;
        }
        let units = code.indices().to_vec();
        let rp = self.params.readout;
        let cands: Vec<(u32, u32)> = self
            .index
            .query_overlap(&code, self.index.len())?
            .into_iter()
            .map(|h| (h.id, h.score as u32))
            .collect();
        let best = cands.first().map_or(0, |c| c.1);
        let overlap: HashMap<u32, u32> = cands.iter().copied().collect();
        let lambda = f64::from(self.params.lambda);
        let mut hits: Vec<Ranked> = base(&cands)
            .into_iter()
            .map(|(id, b)| {
                let learned = self.banks[id as usize].learned(&rp, &units, self.clock);
                Ranked {
                    doc: DocId(id),
                    overlap: overlap.get(&id).copied().unwrap_or(0),
                    learned,
                    score: b + lambda * f64::from(learned),
                }
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
        hits.truncate(top_n);
        self.clock += 1;
        let id = self.next_offer;
        self.next_offer += 1;
        self.offers.insert(
            id,
            Offer {
                units,
                docs: hits.iter().map(|h| h.doc.0).collect(),
                answered: HashSet::new(),
            },
        );
        self.order.push_back(id);
        while self.order.len() > self.params.max_offers {
            if let Some(old) = self.order.pop_front() {
                self.offers.remove(&old);
            }
        }
        Ok(Answer {
            offer: OfferId(id),
            hits,
            novelty,
            best_overlap: best,
            novel: novelty >= self.params.novelty_threshold || best < self.params.min_overlap,
        })
    }

    /// A verified outcome for `doc` in `offer`. Only documents that were
    /// offered can learn (forward-only credit), once per offer.
    pub fn feedback(&mut self, offer: OfferId, doc: DocId, outcome: Outcome) -> Result<Learned> {
        let o = self
            .offers
            .get_mut(&offer.0)
            .ok_or(FlyRagError::UnknownOffer(offer.0))?;
        if !o.docs.contains(&doc.0) {
            return Err(FlyRagError::NotOffered {
                offer: offer.0,
                doc: doc.0,
            });
        }
        if !o.answered.insert(doc.0) {
            return Err(FlyRagError::AlreadyAnswered {
                offer: offer.0,
                doc: doc.0,
            });
        }
        if !self.learning {
            return Ok(Learned::Frozen);
        }
        let rp = self.params.readout;
        let banks = &mut self.banks[doc.0 as usize];
        let expected = banks.learned(&rp, &o.units, self.clock);
        let (target, bank) = match outcome {
            Outcome::Helped => (1.0, Bank::Skip),
            Outcome::Misled => (-1.0, Bank::Use),
        };
        let delta = target - expected;
        if delta.abs() < rp.min_delta {
            return Ok(Learned::Expected);
        }
        let step = (rp.eta * delta.abs()).min(rp.cap);
        banks.depress(&rp, bank, &o.units, step, self.clock);
        Ok(Learned::Updated { delta, step })
    }

    /// KC units of `doc` whose bank weights are away from the prior.
    pub fn moved_units(&self, doc: DocId) -> usize {
        self.banks
            .get(doc.0 as usize)
            .map_or(0, |b| b.moved_units(&self.params.readout, self.clock))
    }

    /// Heap bytes of the hasher, index codes and novelty filter (banks grow
    /// with feedback and are not counted).
    pub fn memory_bytes(&self) -> usize {
        self.hasher.memory_bytes() + self.novelty.cell_bytes()
    }
}
