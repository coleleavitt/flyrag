//! The synthetic FlyRAG benchmark of `research/flyrag/PREREG.md`, shared by
//! `flyrag_eval` (the first run) and `flyrag_followup` (amendment A2).

use anyhow::Result;
use flyrag::{DocId, FlyRag, Outcome, RagParams};
use kenyon::Rng;

pub const TOPICS: usize = 40;
pub const LEXICON: usize = 4_000;
pub const COMMON: usize = 300;
pub const TOPIC_WORDS: usize = 12;
pub const DOC_LEN: usize = 40;
pub const FIX_TOPIC: usize = 4;
pub const DISTRACTORS: usize = 4;
pub const DISTRACTOR_TOPIC: usize = 9;
pub const Q_TOPIC: usize = 6;
pub const Q_COMMON: usize = 4;
pub const TRAIN_Q: usize = 6;
pub const EVAL_Q: usize = 4;
pub const TRIED: usize = 3;
pub const TOP_N: usize = 10;
pub const OOD_Q: usize = 40;
pub const OOD_LEN: usize = 10;

fn lex(i: usize) -> String {
    format!("lx{i}")
}
fn common(i: usize) -> String {
    format!("cw{i}")
}

/// One seed's corpus and queries.
pub struct World {
    /// Per topic: its vocabulary (lexicon indices).
    pub vocab: Vec<Vec<usize>>,
    /// Document texts in insertion order.
    pub docs: Vec<String>,
    /// Which topic's fix each document is.
    pub fix_of: Vec<Option<usize>>,
    pub train: Vec<(usize, String)>,
    pub eval: Vec<(usize, String)>,
    pub ood: Vec<String>,
}

fn text(rng: &mut Rng, words: Vec<String>, len: usize) -> String {
    let mut w = words;
    while w.len() < len {
        w.push(common(rng.below(COMMON)));
    }
    rng.shuffle(&mut w);
    w.join(" ")
}

fn pick(rng: &mut Rng, vocab: &[usize], n: usize) -> Vec<String> {
    rng.sample_distinct(vocab.len(), n)
        .into_iter()
        .map(|i| lex(vocab[i]))
        .collect()
}

/// A query for topic `t`: topic words plus common words.
pub fn make_query(rng: &mut Rng, vocab: &[Vec<usize>], t: usize) -> String {
    let mut w = pick(rng, &vocab[t], Q_TOPIC);
    w.extend((0..Q_COMMON).map(|_| common(rng.below(COMMON))));
    rng.shuffle(&mut w);
    w.join(" ")
}

pub fn world(seed: u64) -> World {
    let mut rng = Rng::derive(seed, 0x464c_5952); // "FLYR"
    let vocab: Vec<Vec<usize>> = (0..TOPICS)
        .map(|_| rng.sample_distinct(LEXICON, TOPIC_WORDS))
        .collect();
    let mut docs: Vec<(String, Option<usize>)> = Vec::new();
    for (t, v) in vocab.iter().enumerate() {
        let w = pick(&mut rng, v, FIX_TOPIC);
        docs.push((text(&mut rng, w, DOC_LEN), Some(t)));
        for _ in 0..DISTRACTORS {
            let w = pick(&mut rng, v, DISTRACTOR_TOPIC);
            docs.push((text(&mut rng, w, DOC_LEN), None));
        }
    }
    rng.shuffle(&mut docs);
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for t in 0..TOPICS {
        for _ in 0..TRAIN_Q {
            train.push((t, make_query(&mut rng, &vocab, t)));
        }
        for _ in 0..EVAL_Q {
            eval.push((t, make_query(&mut rng, &vocab, t)));
        }
    }
    rng.shuffle(&mut train);
    let ood = (0..OOD_Q)
        .map(|_| {
            (0..OOD_LEN)
                .map(|_| lex(rng.below(LEXICON)))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    World {
        vocab,
        fix_of: docs.iter().map(|d| d.1).collect(),
        docs: docs.into_iter().map(|d| d.0).collect(),
        train,
        eval,
        ood,
    }
}

/// A training arm.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Arm {
    Learning,
    Frozen,
    /// The learning arm's labels, permuted among the tried documents.
    Shuffled,
    /// Each tried document `helped` with this probability, independently of
    /// the document (amendment A2.3).
    Independent(f64),
}

pub fn store(w: &World, seed: u64) -> Result<(FlyRag, Vec<Option<usize>>)> {
    let mut r = FlyRag::new(RagParams::prereg(seed))?;
    for d in &w.docs {
        r.add(d)?;
    }
    Ok((r, w.fix_of.clone()))
}

pub fn fix_doc(fix_of: &[Option<usize>], topic: usize) -> DocId {
    DocId(fix_of.iter().position(|f| *f == Some(topic)).unwrap_or(0) as u32)
}

/// Train on `w.train` restricted to `topics`. Returns (helped labels, all
/// labels) given.
pub fn train(
    r: &mut FlyRag,
    w: &World,
    fix_of: &[Option<usize>],
    arm: Arm,
    topics: &[usize],
    rng: &mut Rng,
) -> Result<(usize, usize)> {
    let (mut helped, mut total) = (0, 0);
    for (t, q) in w.train.iter().filter(|(t, _)| topics.contains(t)) {
        let a = r.query(q, TOP_N)?;
        if arm == Arm::Frozen {
            continue;
        }
        let tried: Vec<DocId> = a.hits.iter().take(TRIED).map(|h| h.doc).collect();
        let fix = fix_doc(fix_of, *t);
        let mut labels: Vec<Outcome> = tried
            .iter()
            .map(|&d| match arm {
                Arm::Independent(p) => {
                    if rng.next_f64() < p {
                        Outcome::Helped
                    } else {
                        Outcome::Misled
                    }
                }
                _ if d == fix => Outcome::Helped,
                _ => Outcome::Misled,
            })
            .collect();
        if arm == Arm::Shuffled {
            rng.shuffle(&mut labels);
        }
        for (&d, &o) in tried.iter().zip(&labels) {
            helped += usize::from(o == Outcome::Helped);
            total += 1;
            r.feedback(a.offer, d, o)?;
        }
    }
    Ok((helped, total))
}

/// Mean nDCG@10 and MRR@10 of the fix document over `w.eval` restricted to
/// `topics`.
pub fn evaluate(
    r: &mut FlyRag,
    w: &World,
    fix_of: &[Option<usize>],
    topics: &[usize],
) -> Result<(f64, f64)> {
    let (mut ndcg, mut mrr, mut n) = (0.0, 0.0, 0.0);
    for (t, q) in w.eval.iter().filter(|(t, _)| topics.contains(t)) {
        let a = r.query(q, TOP_N)?;
        let fix = fix_doc(fix_of, *t);
        if let Some(p) = a.hits.iter().position(|h| h.doc == fix) {
            ndcg += 1.0 / ((p + 2) as f64).log2();
            mrr += 1.0 / (p + 1) as f64;
        }
        n += 1.0;
    }
    Ok((ndcg / n, mrr / n))
}

/// AUC of `pos` scores above `neg` (ties count half).
pub fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    let mut s = 0.0;
    for &p in pos {
        for &q in neg {
            s += if p > q {
                1.0
            } else if p == q {
                0.5
            } else {
                0.0
            };
        }
    }
    s / (pos.len() * neg.len()) as f64
}

/// The 40 fresh in-domain probes (one per topic) and the 40 OOD queries, in a
/// seeded random order (A2.1).
pub fn novelty_probes(w: &World, seed: u64) -> Vec<(bool, String)> {
    let mut rng = Rng::derive(seed, 0x5052_4f42); // "PROB"
    let mut p: Vec<(bool, String)> = (0..TOPICS)
        .map(|t| (false, make_query(&mut rng, &w.vocab, t)))
        .collect();
    p.extend(w.ood.iter().take(OOD_Q).map(|q| (true, q.clone())));
    rng.shuffle(&mut p);
    p
}
