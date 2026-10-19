//! FlyRAG next steps, exactly as research/flyrag-next/PREREG.md.
//!
//! `flyrag_next [n_seeds]` (default 20): S1 (learning re-ranker over BM25)
//! and S2 (front end) on BEIR SciFact (`$KENYON_DATA/scifact`), S3 (novelty
//! against documents) on the synthetic benchmark. Writes
//! `results/flyrag_next.{md,json}`.

use std::collections::HashSet;

use anyhow::{Result, ensure};
use flyrag::{DocId, FlyRag, NoveltySource, Outcome, RagParams};
use kenyon::Rng;
use kenyon_repro::flyrag_bench::{self, auc, novelty_probes};
use kenyon_repro::scifact::{Bm25, Data, load, ndcg10};
use kenyon_repro::{data_dir, out_dir, stats};
use rayon::prelude::*;

const TOP_N: usize = 10;
const TRIED: usize = 3;
const CANDIDATES: usize = 100;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Frozen,
    Learning,
    Independent(f64),
}

/// A front end: dimension and IDF on/off.
#[derive(Clone, Copy)]
struct Front {
    dim: usize,
    idf: bool,
}

/// BM25 top-100 candidates per query, as FlyRAG document ids.
type Cands = Vec<Vec<(DocId, f64)>>;

fn candidates(b: &Bm25, qs: &[(String, HashSet<usize>)]) -> Cands {
    qs.par_iter()
        .map(|(q, _)| {
            b.rank(q, CANDIDATES)
                .into_iter()
                .map(|(i, s)| (DocId(i as u32), s))
                .collect()
        })
        .collect()
}

/// One seed, one arm. `rerank` = Some((train, test)) re-ranks BM25's
/// candidates (S1); None ranks with FlyRAG's own codes (S2). Returns
/// per-test-query nDCG@10 and (helped, labels).
fn run_arm(
    d: &Data,
    seed: u64,
    front: Front,
    arm: Arm,
    rerank: Option<(&Cands, &Cands)>,
) -> Result<(Vec<f64>, usize, usize)> {
    let mut p = RagParams::prereg(seed);
    p.dim = front.dim;
    let mut r = FlyRag::new(p)?;
    if front.idf {
        r.fit_idf(&d.docs)?;
    }
    for doc in &d.docs {
        r.add(doc)?;
    }
    let mut order: Vec<usize> = (0..d.train.len()).collect();
    Rng::derive(seed, 0x4f52_4452).shuffle(&mut order); // "ORDR"
    let mut rng = Rng::derive(seed, 0x494e_4450); // "INDP"
    let (mut helped, mut total) = (0, 0);
    if arm != Arm::Frozen {
        for &qi in &order {
            let (q, rel) = &d.train[qi];
            let a = match rerank {
                Some((tr, _)) => r.rerank(q, &tr[qi], TOP_N)?,
                None => r.query(q, TOP_N)?,
            };
            let tried: Vec<DocId> = a.hits.iter().take(TRIED).map(|h| h.doc).collect();
            for doc in tried {
                let o = match arm {
                    Arm::Independent(p) if rng.next_f64() < p => Outcome::Helped,
                    Arm::Independent(_) => Outcome::Misled,
                    _ if rel.contains(&(doc.0 as usize)) => Outcome::Helped,
                    _ => Outcome::Misled,
                };
                helped += usize::from(o == Outcome::Helped);
                total += 1;
                r.feedback(a.offer, doc, o)?;
            }
        }
    }
    r.set_learning(false);
    let mut out = Vec::with_capacity(d.test.len());
    for (qi, (q, rel)) in d.test.iter().enumerate() {
        let a = match rerank {
            Some((_, te)) => r.rerank(q, &te[qi], TOP_N)?,
            None => r.query(q, TOP_N)?,
        };
        let ranked: Vec<usize> = a.hits.iter().map(|h| h.doc.0 as usize).collect();
        out.push(ndcg10(&ranked, rel));
    }
    Ok((out, helped, total))
}

/// Average per query over seeds.
fn avg(per_seed: &[Vec<f64>]) -> Vec<f64> {
    let n = per_seed[0].len();
    (0..n)
        .map(|q| per_seed.iter().map(|s| s[q]).sum::<f64>() / per_seed.len() as f64)
        .collect()
}

fn ci(p: &stats::Paired) -> String {
    format!("{:+.4} [{:+.4}, {:+.4}]", p.mean_diff, p.boot_lo, p.boot_hi)
}

fn met(b: bool) -> &'static str {
    if b { "**met**" } else { "**not met**" }
}

fn excl0(p: &stats::Paired) -> bool {
    p.mean_diff > 0.0 && p.boot_lo > 0.0
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
    let d = load(&data_dir().join("scifact"))?;
    eprintln!("flyrag_next: seeds 1..={n}; HEAD {head}");
    let train_rel: HashSet<usize> = d
        .train
        .iter()
        .flat_map(|(_, r)| r.iter().copied())
        .collect();
    let unseen: Vec<usize> = (0..d.test.len())
        .filter(|&i| d.test[i].1.is_disjoint(&train_rel))
        .collect();
    ensure!(unseen.len() == 104, "unseen subset has {}", unseen.len());
    let sub = |v: &[f64]| unseen.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let seeds: Vec<u64> = (1..=n).collect();
    let prereg = Front {
        dim: 1024,
        idf: false,
    };

    // ---- S1: learning re-ranker over BM25.
    let bm = Bm25::new(&d.docs);
    let tr = candidates(&bm, &d.train);
    let te = candidates(&bm, &d.test);
    let bm25: Vec<f64> = d
        .test
        .iter()
        .zip(&te)
        .map(|((_, rel), c)| {
            let ranked: Vec<usize> = c.iter().take(TOP_N).map(|x| x.0.0 as usize).collect();
            ndcg10(&ranked, rel)
        })
        .collect();
    let s1: Vec<[Vec<f64>; 3]> = seeds
        .par_iter()
        .map(|&s| -> Result<[Vec<f64>; 3]> {
            let rr = Some((&tr, &te));
            let (fz, _, _) = run_arm(&d, s, prereg, Arm::Frozen, rr)?;
            let (lr, h, t) = run_arm(&d, s, prereg, Arm::Learning, rr)?;
            let p = h as f64 / t.max(1) as f64;
            let (ind, _, _) = run_arm(&d, s, prereg, Arm::Independent(p), rr)?;
            eprintln!("S1 seed {s}: p(helped) {p:.4}");
            Ok([fz, lr, ind])
        })
        .collect::<Result<_>>()?;
    let col =
        |v: &[[Vec<f64>; 3]], i: usize| avg(&v.iter().map(|x| x[i].clone()).collect::<Vec<_>>());
    let (s1_fz, s1_lr, s1_ind) = (col(&s1, 0), col(&s1, 1), col(&s1, 2));
    let frozen_mismatch = s1_fz
        .iter()
        .zip(&bm25)
        .filter(|(a, b)| (*a - *b).abs() > 1e-12)
        .count();
    let s1_1 = stats::paired(&s1_lr, &bm25, 61);
    let s1_2 = stats::paired(&s1_lr, &s1_ind, 62);
    let s1_3 = stats::paired(&sub(&s1_lr), &sub(&bm25), 63);
    let b11 = excl0(&s1_1);
    let b12 = excl0(&s1_2);
    let b13 = s1_3.mean_diff >= -0.005 || (s1_3.boot_lo <= 0.0 && s1_3.boot_hi >= 0.0);

    // ---- S2: front ends A-D (frozen), plus learning under D.
    let fronts = [
        ("A d=1024", prereg),
        (
            "B d=1024 + IDF",
            Front {
                dim: 1024,
                idf: true,
            },
        ),
        (
            "C d=16384",
            Front {
                dim: 16_384,
                idf: false,
            },
        ),
        (
            "D d=16384 + IDF",
            Front {
                dim: 16_384,
                idf: true,
            },
        ),
    ];
    let mut s2 = Vec::new();
    for (_, f) in fronts {
        let per: Vec<Vec<f64>> = seeds
            .par_iter()
            .map(|&s| run_arm(&d, s, f, Arm::Frozen, None).map(|r| r.0))
            .collect::<Result<_>>()?;
        s2.push(avg(&per));
    }
    let d_learn: Vec<Vec<f64>> = seeds
        .par_iter()
        .map(|&s| run_arm(&d, s, fronts[3].1, Arm::Learning, None).map(|r| r.0))
        .collect::<Result<_>>()?;
    let d_learn = avg(&d_learn);
    let s2_1 = stats::paired(&s2[3], &s2[0], 64);
    let s2_dl = stats::paired(&d_learn, &s2[3], 65);
    let b21 = excl0(&s2_1);
    let b22 = stats::mean(&s2[3]) >= 0.30;

    // ---- S3: novelty against documents (synthetic, seeds 1..=n).
    let s3: Vec<(f64, f64)> = seeds
        .par_iter()
        .map(|&s| -> Result<(f64, f64)> {
            let w = flyrag_bench::world(s);
            let mut p = RagParams::prereg(s);
            p.novelty_source = NoveltySource::Documents;
            let mut r = FlyRag::new(p)?;
            for doc in &w.docs {
                r.add(doc)?;
            }
            let k = p.k as f64;
            let (mut pos, mut neg, mut pos_o, mut neg_o) = (vec![], vec![], vec![], vec![]);
            for (ood, q) in novelty_probes(&w, s) {
                let a = r.query(&q, TOP_N)?;
                let ov = 1.0 - f64::from(a.best_overlap) / k;
                if ood {
                    pos.push(a.novelty);
                    pos_o.push(ov);
                } else {
                    neg.push(a.novelty);
                    neg_o.push(ov);
                }
            }
            Ok((auc(&pos, &neg), auc(&pos_o, &neg_o)))
        })
        .collect::<Result<_>>()?;
    let s3_bloom: Vec<f64> = s3.iter().map(|x| x.0).collect();
    let s3_ov: Vec<f64> = s3.iter().map(|x| x.1).collect();
    let s3_ci = stats::bootstrap_ci(&s3_bloom, 10_000, 66);
    let s3_ov_ci = stats::bootstrap_ci(&s3_ov, 10_000, 67);
    let b31 = stats::mean(&s3_bloom) >= 0.90;

    let mut md = format!(
        "# FlyRAG next steps (research/flyrag-next/PREREG.md)\n\nSeeds 1..={n}; HEAD `{head}`. SciFact: per-query nDCG@10 averaged over seeds, 95% bootstrap CIs over the 300 test queries; S3: CIs over seeds. 10,000 resamples.\n\n\
         ## S1. Learning re-ranker over BM25 (top {CANDIDATES})\n\n| system | nDCG@10 (300) | unseen 104 |\n|---|---|---|\n"
    );
    for (name, v) in [
        ("BM25", &bm25),
        ("re-rank frozen", &s1_fz),
        ("re-rank learning", &s1_lr),
        ("re-rank independent", &s1_ind),
    ] {
        md.push_str(&format!(
            "| {name} | {:.4} | {:.4} |\n",
            stats::mean(v),
            stats::mean(&sub(v))
        ));
    }
    md.push_str(&format!(
        "\nSanity: re-rank frozen differs from BM25 on {frozen_mismatch} of 300 queries (expected 0).\n\n\
         | bar | value | verdict |\n|---|---|---|\n\
         | S1.1 learning − BM25 > 0, CI excl. 0 | {} | {} |\n\
         | S1.2 learning − independent > 0, CI excl. 0 | {} | {} |\n\
         | S1.3 unseen: learning − BM25 ≥ −0.005 or CI incl. 0 | {} | {} |\n\n\
         ## S2. Front end (frozen base retrieval)\n\n| front end | nDCG@10 (300) |\n|---|---|\n",
        ci(&s1_1), met(b11), ci(&s1_2), met(b12), ci(&s1_3), met(b13)
    ));
    for ((name, _), v) in fronts.iter().zip(&s2) {
        md.push_str(&format!("| {name} | {:.4} |\n", stats::mean(v)));
    }
    md.push_str(&format!(
        "| (BM25, context) | {:.4} |\n\n| bar | value | verdict |\n|---|---|---|\n\
         | S2.1 D − A > 0, CI excl. 0 | {} | {} |\n\
         | S2.2 D ≥ 0.30 | {:.4} | {} |\n\
         | (secondary) learning − frozen under D | {} | — |\n\n\
         ## S3. Novelty against stored documents (synthetic, fresh probes)\n\n| bar | value | verdict |\n|---|---|---|\n\
         | S3.1 Bloom novelty AUC ≥ 0.90 | {:.3} [{:.3}, {:.3}] | {} |\n\
         | (secondary) 1 − best_overlap/k AUC | {:.3} [{:.3}, {:.3}] | — |\n",
        stats::mean(&bm25),
        ci(&s2_1), met(b21),
        stats::mean(&s2[3]), met(b22),
        ci(&s2_dl),
        stats::mean(&s3_bloom), s3_ci.0, s3_ci.1, met(b31),
        stats::mean(&s3_ov), s3_ov_ci.0, s3_ov_ci.1,
    ));
    println!("{md}");
    std::fs::write(out_dir().join("flyrag_next.md"), &md)?;
    let json = format!(
        "{{\"head\":\"{head}\",\"seeds\":{n},\
         \"s1\":{{\"bm25\":{},\"frozen\":{},\"learning\":{},\"independent\":{},\"frozen_mismatch\":{frozen_mismatch},\
         \"s1_1\":[{},{},{}],\"s1_2\":[{},{},{}],\"s1_3\":[{},{},{}]}},\
         \"s2\":{{\"A\":{},\"B\":{},\"C\":{},\"D\":{},\"D_learning\":{},\"s2_1\":[{},{},{}],\"d_learn\":[{},{},{}]}},\
         \"s3\":{{\"bloom_auc\":{s3_bloom:?},\"overlap_auc\":{s3_ov:?}}},\
         \"bars\":{{\"S1.1\":{b11},\"S1.2\":{b12},\"S1.3\":{b13},\"S2.1\":{b21},\"S2.2\":{b22},\"S3.1\":{b31}}}}}\n",
        stats::mean(&bm25),
        stats::mean(&s1_fz),
        stats::mean(&s1_lr),
        stats::mean(&s1_ind),
        s1_1.mean_diff,
        s1_1.boot_lo,
        s1_1.boot_hi,
        s1_2.mean_diff,
        s1_2.boot_lo,
        s1_2.boot_hi,
        s1_3.mean_diff,
        s1_3.boot_lo,
        s1_3.boot_hi,
        stats::mean(&s2[0]),
        stats::mean(&s2[1]),
        stats::mean(&s2[2]),
        stats::mean(&s2[3]),
        stats::mean(&d_learn),
        s2_1.mean_diff,
        s2_1.boot_lo,
        s2_1.boot_hi,
        s2_dl.mean_diff,
        s2_dl.boot_lo,
        s2_dl.boot_hi,
    );
    std::fs::write(out_dir().join("flyrag_next.json"), json)?;
    Ok(())
}
