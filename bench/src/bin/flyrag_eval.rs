//! FlyRAG evaluation, exactly as research/flyrag/PREREG.md (+ amendment A1).
//!
//! `flyrag_eval [n_seeds]` (default 20, seeds 1..=n). Writes
//! `results/flyrag.json` and prints the markdown report.

use anyhow::Result;
use kenyon::Rng;
use kenyon_repro::flyrag_bench::{Arm, OOD_Q, TOP_N, TOPICS, auc, evaluate, store, train, world};
use kenyon_repro::{out_dir, stats};
use rayon::prelude::*;

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
