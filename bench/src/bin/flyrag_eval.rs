//! FlyRAG evaluation, exactly as research/flyrag/PREREG.md (+ amendment A1).
//!
//! `flyrag_eval [n_seeds]` (default 20, seeds 1..=n). Writes
//! `results/flyrag.json` and prints the markdown report.

use anyhow::Result;
use flyrag::{DocId, FlyRag, Outcome, RagParams};
use kenyon::Rng;
use kenyon_repro::{out_dir, stats};
use rayon::prelude::*;

const TOPICS: usize = 40;
const LEXICON: usize = 4_000;
const COMMON: usize = 300;
const TOPIC_WORDS: usize = 12;
const DOC_LEN: usize = 40;
const FIX_TOPIC: usize = 4;
const DISTRACTORS: usize = 4;
const DISTRACTOR_TOPIC: usize = 9;
const Q_TOPIC: usize = 6;
const Q_COMMON: usize = 4;
const TRAIN_Q: usize = 6;
const EVAL_Q: usize = 4;
const TRIED: usize = 3;
const TOP_N: usize = 10;
const OOD_Q: usize = 40;
const OOD_LEN: usize = 10;

fn lex(i: usize) -> String {
    format!("lx{i}")
}
fn common(i: usize) -> String {
    format!("cw{i}")
}

struct World {
    /// Per topic: its vocabulary (lexicon indices).
    vocab: Vec<Vec<usize>>,
    /// Document texts in insertion order, and which topic's fix each is.
    docs: Vec<String>,
    fix_of: Vec<Option<usize>>,
    train: Vec<(usize, String)>,
    eval: Vec<(usize, String)>,
    ood: Vec<String>,
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

fn world(seed: u64) -> World {
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
    let query = |rng: &mut Rng, t: usize| {
        let mut w = pick(rng, &vocab[t], Q_TOPIC);
        w.extend((0..Q_COMMON).map(|_| common(rng.below(COMMON))));
        rng.shuffle(&mut w);
        w.join(" ")
    };
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for t in 0..TOPICS {
        for _ in 0..TRAIN_Q {
            train.push((t, query(&mut rng, t)));
        }
        for _ in 0..EVAL_Q {
            eval.push((t, query(&mut rng, t)));
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

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Learning,
    Frozen,
    Shuffled,
}

fn store(w: &World, seed: u64) -> Result<(FlyRag, Vec<Option<usize>>)> {
    let mut r = FlyRag::new(RagParams::prereg(seed))?;
    for d in &w.docs {
        r.add(d)?;
    }
    Ok((r, w.fix_of.clone()))
}

fn fix_doc(fix_of: &[Option<usize>], topic: usize) -> DocId {
    DocId(fix_of.iter().position(|f| *f == Some(topic)).unwrap_or(0) as u32)
}

fn train(
    r: &mut FlyRag,
    w: &World,
    fix_of: &[Option<usize>],
    arm: Arm,
    topics: &[usize],
    rng: &mut Rng,
) -> Result<()> {
    for (t, q) in w.train.iter().filter(|(t, _)| topics.contains(t)) {
        let a = r.query(q, TOP_N)?;
        if arm == Arm::Frozen {
            continue;
        }
        let tried: Vec<DocId> = a.hits.iter().take(TRIED).map(|h| h.doc).collect();
        let fix = fix_doc(fix_of, *t);
        let mut labels: Vec<Outcome> = tried
            .iter()
            .map(|&d| {
                if d == fix {
                    Outcome::Helped
                } else {
                    Outcome::Misled
                }
            })
            .collect();
        if arm == Arm::Shuffled {
            rng.shuffle(&mut labels);
        }
        for (&d, &o) in tried.iter().zip(&labels) {
            r.feedback(a.offer, d, o)?;
        }
    }
    Ok(())
}

/// Mean nDCG@10 and MRR@10 of the fix document over `w.eval` restricted to
/// `topics`.
fn evaluate(
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
fn auc(pos: &[f64], neg: &[f64]) -> f64 {
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

#[derive(Default, Clone)]
struct SeedResult {
    seed: u64,
    ndcg: [f64; 3],
    mrr: [f64; 3],
    auc_flag: f64,
    auc_score: f64,
    forget_m1: f64,
    forget_m2: f64,
    forget_base: f64,
}

fn run_seed(seed: u64) -> Result<SeedResult> {
    let w = world(seed);
    let all: Vec<usize> = (0..TOPICS).collect();
    let mut res = SeedResult {
        seed,
        ..Default::default()
    };
    for (i, arm) in [Arm::Learning, Arm::Frozen, Arm::Shuffled]
        .into_iter()
        .enumerate()
    {
        let (mut r, fix_of) = store(&w, seed)?;
        let mut rng = Rng::derive(seed, 0x5348_5546); // "SHUF"
        train(&mut r, &w, &fix_of, arm, &all, &mut rng)?;
        let (n, m) = evaluate(&mut r, &w, &fix_of, &all)?;
        res.ndcg[i] = n;
        res.mrr[i] = m;
        if arm == Arm::Learning {
            // Novelty: evaluation queries (in-domain, already scored above) vs
            // out-of-domain queries, on this trained store.
            let mut pos_flag = Vec::new();
            let mut pos_score = Vec::new();
            for q in &w.ood {
                let a = r.query(q, TOP_N)?;
                pos_flag.push(f64::from(u8::from(a.novel)));
                pos_score.push(a.novelty);
            }
            let mut neg_flag = Vec::new();
            let mut neg_score = Vec::new();
            for (_, q) in w.eval.iter().take(OOD_Q) {
                let a = r.query(q, TOP_N)?;
                neg_flag.push(f64::from(u8::from(a.novel)));
                neg_score.push(a.novelty);
            }
            res.auc_flag = auc(&pos_flag, &neg_flag);
            res.auc_score = auc(&pos_score, &neg_score);
        }
        if arm == Arm::Frozen {
            let first: Vec<usize> = (0..TOPICS / 2).collect();
            let (mut fz, fix_of) = store(&w, seed)?;
            res.forget_base = evaluate(&mut fz, &w, &fix_of, &first)?.0;
        }
    }
    // Forgetting: learn topics 1-20, measure; learn 21-40, re-measure 1-20.
    let first: Vec<usize> = (0..TOPICS / 2).collect();
    let second: Vec<usize> = (TOPICS / 2..TOPICS).collect();
    let (mut r, fix_of) = store(&w, seed)?;
    let mut rng = Rng::derive(seed, 0x4647_5447); // "FGTG"
    train(&mut r, &w, &fix_of, Arm::Learning, &first, &mut rng)?;
    res.forget_m1 = evaluate(&mut r, &w, &fix_of, &first)?.0;
    train(&mut r, &w, &fix_of, Arm::Learning, &second, &mut rng)?;
    res.forget_m2 = evaluate(&mut r, &w, &fix_of, &first)?.0;
    let _ = &w.vocab;
    Ok(res)
}

fn main() -> Result<()> {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20);
    let head = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    eprintln!("flyrag_eval: seeds 1..={n}, HEAD {head}");
    let mut rs: Vec<SeedResult> = (1..=n)
        .into_par_iter()
        .map(run_seed)
        .collect::<Result<_>>()?;
    rs.sort_by_key(|r| r.seed);
    let col = |f: &dyn Fn(&SeedResult) -> f64| rs.iter().map(f).collect::<Vec<f64>>();
    let (l, f, s) = (
        col(&|r| r.ndcg[0]),
        col(&|r| r.ndcg[1]),
        col(&|r| r.ndcg[2]),
    );
    let lf = stats::paired(&l, &f, 34);
    let sf = stats::paired(&s, &f, 35);
    let auc_flag = col(&|r| r.auc_flag);
    let auc_score = col(&|r| r.auc_score);
    let (m1, m2, b) = (
        col(&|r| r.forget_m1),
        col(&|r| r.forget_m2),
        col(&|r| r.forget_base),
    );
    let retained = |idx: &[usize]| {
        let g = |v: &[f64]| idx.iter().map(|&i| v[i]).sum::<f64>() / idx.len() as f64;
        let gain = g(&m1) - g(&b);
        if gain > 0.0 {
            (g(&m2) - g(&b)) / gain
        } else {
            f64::NAN
        }
    };
    let all: Vec<usize> = (0..rs.len()).collect();
    let ret = retained(&all);
    let mut boot = Vec::with_capacity(10_000);
    let mut rng = Rng::new(36);
    for _ in 0..10_000 {
        let idx: Vec<usize> = (0..rs.len()).map(|_| rng.below(rs.len())).collect();
        boot.push(retained(&idx));
    }
    boot.retain(|v| v.is_finite());
    boot.sort_by(f64::total_cmp);
    let q = |p: f64| {
        boot.get((p * (boot.len().max(1) - 1) as f64).round() as usize)
            .copied()
            .unwrap_or(f64::NAN)
    };

    let bar1 = lf.mean_diff >= 0.03 && lf.boot_lo > 0.0;
    let bar2 = sf.mean_diff < 0.01 || (sf.boot_lo <= 0.0 && sf.boot_hi >= 0.0);
    let bar3 = stats::mean(&auc_flag) >= 0.90;
    let bar4 = ret >= 0.80;
    let met = |b: bool| if b { "**met**" } else { "**not met**" };

    let mut md = String::new();
    md.push_str(&format!("# FlyRAG evaluation (research/flyrag/PREREG.md)\n\nSeeds 1..={n}; HEAD `{head}`; 95% bootstrap CIs over seeds (10,000 resamples).\n\n"));
    md.push_str("| arm | nDCG@10 | MRR@10 |\n|---|---|---|\n");
    for (i, name) in ["learning", "frozen", "shuffled"].iter().enumerate() {
        let a = col(&|r| r.ndcg[i]);
        let m = col(&|r| r.mrr[i]);
        md.push_str(&format!(
            "| {name} | {:.3} | {:.3} |\n",
            stats::mean(&a),
            stats::mean(&m)
        ));
    }
    md.push_str(&format!(
        "\n| bar | value | verdict |\n|---|---|---|\n\
         | 1. learning − frozen nDCG@10 ≥ +0.03, CI excl. 0 | {:+.3} [{:+.3}, {:+.3}] (d_z {:.2}) | {} |\n\
         | 2. shuffled − frozen < +0.01 or CI incl. 0 | {:+.3} [{:+.3}, {:+.3}] | {} |\n\
         | 3. novelty flag AUC ≥ 0.90 | {:.3} (continuous novelty score AUC {:.3}, secondary) | {} |\n\
         | 4. retained gain on topics 1–20 ≥ 80% | {:.0}% [{:.0}%, {:.0}%] (m1 {:.3}, m2 {:.3}, frozen {:.3}) | {} |\n\
         | 5. no feedback, no change | unit test `no_feedback_no_change` | see `cargo test -p flyrag` |\n",
        lf.mean_diff, lf.boot_lo, lf.boot_hi, lf.dz, met(bar1),
        sf.mean_diff, sf.boot_lo, sf.boot_hi, met(bar2),
        stats::mean(&auc_flag), stats::mean(&auc_score), met(bar3),
        100.0 * ret, 100.0 * q(0.025), 100.0 * q(0.975), stats::mean(&m1), stats::mean(&m2), stats::mean(&b), met(bar4),
    ));
    println!("{md}");
    let json = format!(
        "{{\"head\":\"{head}\",\"seeds\":{n},\"ndcg\":{{\"learning\":{l:?},\"frozen\":{f:?},\"shuffled\":{s:?}}},\
         \"learning_minus_frozen\":[{},{},{}],\"shuffled_minus_frozen\":[{},{},{}],\
         \"auc_flag\":{auc_flag:?},\"auc_score\":{auc_score:?},\"forget\":{{\"m1\":{m1:?},\"m2\":{m2:?},\"frozen\":{b:?},\"retained\":{ret}}},\
         \"bars\":[{bar1},{bar2},{bar3},{bar4}]}}\n",
        lf.mean_diff, lf.boot_lo, lf.boot_hi, sf.mean_diff, sf.boot_lo, sf.boot_hi
    );
    std::fs::write(out_dir().join("flyrag.json"), json)?;
    Ok(())
}
