//! FlyRAG follow-up, research/flyrag/PREREG.md amendment A2.
//!
//! `flyrag_followup dev` sweeps the novelty flag on the dev seeds 1000–1004
//! and prints the chosen pair (A2.2). `flyrag_followup eval <t|off> <o>`
//! runs seeds 1–20 with that pair: the novelty re-test (A2.1/A2.2) and the
//! "independent" control (A2.3). Writes `results/flyrag_followup[_dev].{md,json}`.

use anyhow::{Result, bail};
use kenyon::Rng;
use kenyon_repro::flyrag_bench::{
    Arm, OOD_Q, TOP_N, TOPICS, World, auc, evaluate, make_query, store, train, world,
};
use kenyon_repro::{out_dir, stats};
use rayon::prelude::*;

/// One novelty probe: out-of-domain?, novelty, best overlap.
#[derive(Clone, Copy)]
struct Probe {
    ood: bool,
    novelty: f64,
    best: u32,
}

/// The 40 fresh in-domain probes (one per topic) and the 40 OOD queries, in a
/// seeded random order (A2.1).
fn probes(w: &World, seed: u64) -> Vec<(bool, String)> {
    let mut rng = Rng::derive(seed, 0x5052_4f42); // "PROB"
    let mut p: Vec<(bool, String)> = (0..TOPICS)
        .map(|t| (false, make_query(&mut rng, &w.vocab, t)))
        .collect();
    p.extend(w.ood.iter().take(OOD_Q).map(|q| (true, q.clone())));
    rng.shuffle(&mut p);
    p
}

fn flag_auc(ps: &[Probe], t: f64, o: u32) -> f64 {
    let f = |p: &Probe| f64::from(u8::from(p.novelty >= t || p.best < o));
    let pos: Vec<f64> = ps.iter().filter(|p| p.ood).map(f).collect();
    let neg: Vec<f64> = ps.iter().filter(|p| !p.ood).map(f).collect();
    auc(&pos, &neg)
}

fn score_auc(ps: &[Probe]) -> f64 {
    let pos: Vec<f64> = ps.iter().filter(|p| p.ood).map(|p| p.novelty).collect();
    let neg: Vec<f64> = ps.iter().filter(|p| !p.ood).map(|p| p.novelty).collect();
    auc(&pos, &neg)
}

struct SeedResult {
    seed: u64,
    probes: Vec<Probe>,
    /// nDCG@10: learning, frozen, shuffled, independent.
    ndcg: [f64; 4],
    p_helped: f64,
}

fn run_seed(seed: u64, arms: bool) -> Result<SeedResult> {
    let w = world(seed);
    let all: Vec<usize> = (0..TOPICS).collect();
    // Learning arm exactly as the first run, then the fresh probes.
    let (mut r, fix_of) = store(&w, seed)?;
    let mut rng = Rng::derive(seed, 0x5348_5546); // "SHUF"
    let (h, n) = train(&mut r, &w, &fix_of, Arm::Learning, &all, &mut rng)?;
    let p_helped = h as f64 / n.max(1) as f64;
    let mut ndcg = [f64::NAN; 4];
    ndcg[0] = evaluate(&mut r, &w, &fix_of, &all)?.0;
    let mut ps = Vec::new();
    for (ood, q) in probes(&w, seed) {
        let a = r.query(&q, TOP_N)?;
        ps.push(Probe {
            ood,
            novelty: a.novelty,
            best: a.best_overlap,
        });
    }
    if arms {
        let others = [
            (1, Arm::Frozen, 0x5348_5546),
            (2, Arm::Shuffled, 0x5348_5546),
            (3, Arm::Independent(p_helped), 0x494e_4450), // "INDP"
        ];
        for (i, arm, stream) in others {
            let (mut r, fix_of) = store(&w, seed)?;
            let mut rng = Rng::derive(seed, stream);
            train(&mut r, &w, &fix_of, arm, &all, &mut rng)?;
            ndcg[i] = evaluate(&mut r, &w, &fix_of, &all)?.0;
        }
    }
    Ok(SeedResult {
        seed,
        probes: ps,
        ndcg,
        p_helped,
    })
}

fn head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn run(seeds: std::ops::RangeInclusive<u64>, arms: bool) -> Result<Vec<SeedResult>> {
    let mut rs: Vec<SeedResult> = seeds
        .into_par_iter()
        .map(|s| run_seed(s, arms))
        .collect::<Result<_>>()?;
    rs.sort_by_key(|r| r.seed);
    Ok(rs)
}

fn fmt_t(t: f64) -> String {
    if t.is_finite() {
        format!("{t:.2}")
    } else {
        "off".into()
    }
}

fn dev() -> Result<()> {
    let head = head();
    let rs = run(1000..=1004, false)?;
    let ts: Vec<f64> = (1..=20)
        .map(|i| f64::from(i) * 0.05)
        .chain([f64::INFINITY])
        .collect();
    let os = [0u32, 1, 2, 4, 6, 8];
    let mut grid = Vec::new();
    for &o in &os {
        for &t in &ts {
            let m = rs.iter().map(|r| flag_auc(&r.probes, t, o)).sum::<f64>() / rs.len() as f64;
            grid.push((t, o, m));
        }
    }
    // Highest mean AUC; ties: smaller o, then t closest to 0.8 ("off" = inf).
    let mut best = grid[0];
    for &g in &grid[1..] {
        let better = g.2 > best.2 + 1e-12
            || ((g.2 - best.2).abs() <= 1e-12
                && (g.1 < best.1
                    || (g.1 == best.1 && (g.0 - 0.8).abs() < (best.0 - 0.8).abs() - 1e-12)));
        if better {
            best = g;
        }
    }
    let mut md = format!(
        "# FlyRAG follow-up: novelty flag sweep on dev seeds 1000–1004 (PREREG A2.2)\n\nHEAD `{head}`. Mean flag AUC over the 5 dev seeds; rows o (min overlap), columns t (novelty threshold).\n\n| o \\ t |"
    );
    for &t in &ts {
        md.push_str(&format!(" {} |", fmt_t(t)));
    }
    md.push_str("\n|---|");
    md.push_str(&"---|".repeat(ts.len()));
    md.push('\n');
    for &o in &os {
        md.push_str(&format!("| {o} |"));
        for g in grid.iter().filter(|g| g.1 == o) {
            md.push_str(&format!(" {:.3} |", g.2));
        }
        md.push('\n');
    }
    let sa = rs.iter().map(|r| score_auc(&r.probes)).sum::<f64>() / rs.len() as f64;
    md.push_str(&format!(
        "\n**Chosen:** t = {}, o = {} (mean dev flag AUC {:.3}). Continuous novelty score AUC on dev: {:.3}.\n",
        fmt_t(best.0),
        best.1,
        best.2,
        sa
    ));
    println!("{md}");
    std::fs::write(out_dir().join("flyrag_followup_dev.md"), &md)?;
    let json = format!(
        "{{\"head\":\"{head}\",\"seeds\":[1000,1004],\"chosen\":{{\"t\":\"{}\",\"o\":{},\"auc\":{}}},\"score_auc\":{sa},\"grid\":[{}]}}\n",
        fmt_t(best.0),
        best.1,
        best.2,
        grid.iter()
            .map(|g| format!("[\"{}\",{},{}]", fmt_t(g.0), g.1, g.2))
            .collect::<Vec<_>>()
            .join(",")
    );
    std::fs::write(out_dir().join("flyrag_followup_dev.json"), json)?;
    Ok(())
}

fn eval(t: f64, o: u32) -> Result<()> {
    let head = head();
    let rs = run(1..=20, true)?;
    let col = |i: usize| rs.iter().map(|r| r.ndcg[i]).collect::<Vec<f64>>();
    let (l, f, s, ind) = (col(0), col(1), col(2), col(3));
    let i_f = stats::paired(&ind, &f, 41);
    let l_i = stats::paired(&l, &ind, 42);
    let s_f = stats::paired(&s, &f, 43);
    let fa: Vec<f64> = rs.iter().map(|r| flag_auc(&r.probes, t, o)).collect();
    let sa: Vec<f64> = rs.iter().map(|r| score_auc(&r.probes)).collect();
    let fa_ci = stats::bootstrap_ci(&fa, 10_000, 44);
    let sa_ci = stats::bootstrap_ci(&sa, 10_000, 45);
    let ph: Vec<f64> = rs.iter().map(|r| r.p_helped).collect();
    let bar2b = i_f.mean_diff < 0.01 || (i_f.boot_lo <= 0.0 && i_f.boot_hi >= 0.0);
    let bar2c = l_i.mean_diff > 0.0 && l_i.boot_lo > 0.0;
    let bar3 = stats::mean(&fa) >= 0.90;
    let met = |b: bool| if b { "**met**" } else { "**not met**" };
    let mut md = format!(
        "# FlyRAG follow-up (PREREG amendment A2)\n\nSeeds 1–20; HEAD `{head}`; flag t = {}, o = {o} (chosen on dev seeds, A3); 95% bootstrap CIs over seeds (10,000 resamples).\n\n| arm | nDCG@10 |\n|---|---|\n",
        fmt_t(t)
    );
    for (name, v) in [
        ("learning", &l),
        ("frozen", &f),
        ("shuffled", &s),
        ("independent", &ind),
    ] {
        md.push_str(&format!("| {name} | {:.3} |\n", stats::mean(v)));
    }
    md.push_str(&format!(
        "\nIndependent arm's p(helped): mean {:.3} (per seed, the learning arm's rate).\n\n\
         | bar | value | verdict |\n|---|---|---|\n\
         | 2b. independent − frozen < +0.01 or CI incl. 0 | {:+.3} [{:+.3}, {:+.3}] | {} |\n\
         | 2c. learning − independent > 0, CI excl. 0 | {:+.3} [{:+.3}, {:+.3}] (d_z {:.2}) | {} |\n\
         | 3 (re-test). flag AUC ≥ 0.90, fresh probes | {:.3} [{:.3}, {:.3}] | {} |\n\
         | (secondary) continuous novelty AUC, fresh probes | {:.3} [{:.3}, {:.3}] | — |\n\
         | (reference) shuffled − frozen | {:+.3} [{:+.3}, {:+.3}] | — |\n",
        stats::mean(&ph),
        i_f.mean_diff, i_f.boot_lo, i_f.boot_hi, met(bar2b),
        l_i.mean_diff, l_i.boot_lo, l_i.boot_hi, l_i.dz, met(bar2c),
        stats::mean(&fa), fa_ci.0, fa_ci.1, met(bar3),
        stats::mean(&sa), sa_ci.0, sa_ci.1,
        s_f.mean_diff, s_f.boot_lo, s_f.boot_hi,
    ));
    println!("{md}");
    std::fs::write(out_dir().join("flyrag_followup.md"), &md)?;
    let json = format!(
        "{{\"head\":\"{head}\",\"t\":\"{}\",\"o\":{o},\"ndcg\":{{\"learning\":{l:?},\"frozen\":{f:?},\"shuffled\":{s:?},\"independent\":{ind:?}}},\
         \"p_helped\":{ph:?},\"flag_auc\":{fa:?},\"score_auc\":{sa:?},\
         \"independent_minus_frozen\":[{},{},{}],\"learning_minus_independent\":[{},{},{}],\"bars\":{{\"2b\":{bar2b},\"2c\":{bar2c},\"3\":{bar3}}}}}\n",
        fmt_t(t),
        i_f.mean_diff,
        i_f.boot_lo,
        i_f.boot_hi,
        l_i.mean_diff,
        l_i.boot_lo,
        l_i.boot_hi
    );
    std::fs::write(out_dir().join("flyrag_followup.json"), json)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("dev") => dev(),
        Some("eval") if args.len() == 3 => {
            let t = if args[1] == "off" {
                f64::INFINITY
            } else {
                args[1].parse()?
            };
            eval(t, args[2].parse()?)
        }
        _ => bail!("usage: flyrag_followup dev | eval <t|off> <o>"),
    }
}
