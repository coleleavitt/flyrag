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
    /// `overlap / k + λ · learned`.
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
        Ok(self.hasher.hash(&self.encoder.encode(text))?)
    }

    /// Add a document.
    pub fn add(&mut self, text: &str) -> Result<DocId> {
        let code = self.code(text)?;
        let id = self.index.insert(code)?;
        if id as usize != self.banks.len() {
            return Err(FlyRagError::Full);
        }
        self.banks.push(Banks::default());
        Ok(DocId(id))
    }

    /// Rank the documents for `text`, keep the offer for feedback, and record
    /// the query in the novelty filter.
    pub fn query(&mut self, text: &str, top_n: usize) -> Result<Answer> {
        let code = self.code(text)?;
        let novelty = self.novelty.novelty_code(&code)?;
        self.novelty.insert_code(&code)?;
        let units = code.indices().to_vec();
        let k = self.params.k as f64;
        let rp = self.params.readout;
        let candidates = self.index.query_overlap(&code, self.index.len())?;
        let best = candidates.first().map_or(0, |h| h.score as u32);
        let mut hits: Vec<Ranked> = candidates
            .into_iter()
            .map(|h| {
                let learned = self.banks[h.id as usize].learned(&rp, &units, self.clock);
                Ranked {
                    doc: DocId(h.id),
                    overlap: h.score as u32,
                    learned,
                    score: h.score / k + f64::from(self.params.lambda) * f64::from(learned),
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
