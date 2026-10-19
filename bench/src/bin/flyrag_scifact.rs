//! FlyRAG on BEIR SciFact, exactly as research/flyrag-scifact/PREREG.md.
//!
//! `flyrag_scifact [n_seeds]` (default 20). Data: `$KENYON_DATA/scifact`
//! (`corpus.jsonl`, `queries.jsonl`, `qrels/{train,test}.tsv`). Writes
//! `results/flyrag_scifact.{md,json}`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use flyrag::{DocId, FlyRag, Outcome, RagParams, TextEncoder};
use kenyon::Rng;
use kenyon_repro::{data_dir, out_dir, stats};
use rayon::prelude::*;

const TOP_N: usize = 10;
const TRIED: usize = 3;

struct Data {
    /// Document texts (title + " " + abstract), in file order.
    docs: Vec<String>,
    /// Train and test queries: text and relevant document indices.
    train: Vec<(String, HashSet<usize>)>,
    test: Vec<(String, HashSet<usize>)>,
}

fn jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

fn field(v: &serde_json::Value, k: &str) -> String {
    v.get(k)
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_owned()
}

fn load(dir: &Path) -> Result<Data> {
    let mut docs = Vec::new();
    let mut doc_ix = HashMap::new();
    for v in jsonl(&dir.join("corpus.jsonl"))? {
        doc_ix.insert(field(&v, "_id"), docs.len());
        docs.push(format!("{} {}", field(&v, "title"), field(&v, "text")));
    }
    let queries: HashMap<String, String> = jsonl(&dir.join("queries.jsonl"))?
        .iter()
        .map(|v| (field(v, "_id"), field(v, "text")))
        .collect();
    let split = |name: &str| -> Result<Vec<(String, HashSet<usize>)>> {
        let p = dir.join("qrels").join(name);
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        // Keep file order of first appearance.
        let mut order: Vec<String> = Vec::new();
        let mut rel: HashMap<String, HashSet<usize>> = HashMap::new();
        for line in s.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            ensure!(f.len() == 3, "bad qrels line {line:?}");
            if f[2].trim().parse::<i32>().unwrap_or(0) <= 0 {
                continue;
            }
            let d = *doc_ix
                .get(f[1])
                .with_context(|| format!("unknown doc {}", f[1]))?;
            if !rel.contains_key(f[0]) {
                order.push(f[0].to_owned());
            }
            rel.entry(f[0].to_owned()).or_default().insert(d);
        }
        order
            .into_iter()
            .map(|q| {
                let text = queries
                    .get(&q)
                    .with_context(|| format!("unknown query {q}"))?
                    .clone();
                let r = rel.remove(&q).unwrap_or_default();
                Ok((text, r))
            })
            .collect()
    };
    let train = split("train.tsv")?;
    let test = split("test.tsv")?;
    ensure!(
        docs.len() == 5183 && train.len() == 809 && test.len() == 300,
        "unexpected SciFact sizes: {} docs, {} train, {} test",
        docs.len(),
        train.len(),
        test.len()
    );
    Ok(Data { docs, train, test })
}

/// nDCG@10 of a ranking with binary relevance.
fn ndcg10(ranked: &[usize], rel: &HashSet<usize>) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(10)
        .enumerate()
        .filter(|(_, d)| rel.contains(d))
        .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
        .sum();
    let ideal: f64 = (0..rel.len().min(10))
        .map(|i| 1.0 / ((i + 2) as f64).log2())
        .sum();
    if ideal > 0.0 { dcg / ideal } else { 0.0 }
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Frozen,
    Learning,
    Shuffled,
    Independent(f64),
}

/// One seed, one arm: per-test-query nDCG@10 and (helped, labels) given.
fn run_arm(d: &Data, seed: u64, arm: Arm) -> Result<(Vec<f64>, usize, usize)> {
    let mut r = FlyRag::new(RagParams::prereg(seed))?;
    for doc in &d.docs {
        r.add(doc)?;
    }
    let mut order: Vec<usize> = (0..d.train.len()).collect();
    Rng::derive(seed, 0x4f52_4452).shuffle(&mut order); // "ORDR"
    let mut rng = Rng::derive(
        seed,
        match arm {
            Arm::Independent(_) => 0x494e_4450, // "INDP"
            _ => 0x5348_5546,                   // "SHUF"
        },
    );
    let (mut helped, mut total) = (0, 0);
    for &qi in &order {
        let (q, rel) = &d.train[qi];
        let a = r.query(q, TOP_N)?;
        if arm == Arm::Frozen {
            continue;
        }
        let tried: Vec<DocId> = a.hits.iter().take(TRIED).map(|h| h.doc).collect();
        let mut labels: Vec<Outcome> = tried
            .iter()
            .map(|dd| match arm {
                Arm::Independent(p) => {
                    if rng.next_f64() < p {
                        Outcome::Helped
                    } else {
                        Outcome::Misled
                    }
                }
                _ if rel.contains(&(dd.0 as usize)) => Outcome::Helped,
                _ => Outcome::Misled,
            })
            .collect();
        if arm == Arm::Shuffled {
            rng.shuffle(&mut labels);
        }
        for (&doc, &o) in tried.iter().zip(&labels) {
            helped += usize::from(o == Outcome::Helped);
            total += 1;
            r.feedback(a.offer, doc, o)?;
        }
    }
    r.set_learning(false);
    let mut out = Vec::with_capacity(d.test.len());
    for (q, rel) in &d.test {
        let a = r.query(q, TOP_N)?;
        let ranked: Vec<usize> = a.hits.iter().map(|h| h.doc.0 as usize).collect();
        out.push(ndcg10(&ranked, rel));
    }
    Ok((out, helped, total))
}

/// BM25 (k1 0.9, b 0.4) over FlyRAG's unigram tokens; per-test-query nDCG@10.
fn bm25(d: &Data) -> Vec<f64> {
    let (k1, b) = (0.9, 0.4);
    let toks: Vec<Vec<String>> = d.docs.iter().map(|t| TextEncoder::tokens(t)).collect();
    let n = toks.len() as f64;
    let avgdl = toks.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let mut df: HashMap<&str, f64> = HashMap::new();
    let mut tf: Vec<HashMap<&str, f64>> = Vec::with_capacity(toks.len());
    for t in &toks {
        let mut m: HashMap<&str, f64> = HashMap::new();
        for w in t {
            *m.entry(w.as_str()).or_default() += 1.0;
        }
        for w in m.keys() {
            *df.entry(w).or_default() += 1.0;
        }
        tf.push(m);
    }
    d.test
        .iter()
        .map(|(q, rel)| {
            let qt: HashSet<String> = TextEncoder::tokens(q).into_iter().collect();
            let mut scores: Vec<(usize, f64)> = tf
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let dl = toks[i].len() as f64;
                    let s = qt
                        .iter()
                        .filter_map(|w| {
                            let f = *m.get(w.as_str())?;
                            let dfw = df.get(w.as_str()).copied().unwrap_or(0.0);
                            let idf = (1.0 + (n - dfw + 0.5) / (dfw + 0.5)).ln();
                            Some(idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * dl / avgdl)))
                        })
                        .sum();
                    (i, s)
                })
                .collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let ranked: Vec<usize> = scores.iter().take(10).map(|s| s.0).collect();
            ndcg10(&ranked, rel)
        })
        .collect()
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
    eprintln!(
        "flyrag_scifact: {} docs, {} train, {} test; seeds 1..={n}; HEAD {head}",
        d.docs.len(),
        d.train.len(),
        d.test.len()
    );
    let train_rel: HashSet<usize> = d
        .train
        .iter()
        .flat_map(|(_, r)| r.iter().copied())
        .collect();
    let unseen: Vec<usize> = (0..d.test.len())
        .filter(|&i| d.test[i].1.is_disjoint(&train_rel))
        .collect();
    ensure!(
        unseen.len() == 104,
        "unseen subset has {} queries, expected 104",
        unseen.len()
    );

    // Per seed: frozen, learning, shuffled, then independent at the learning
    // arm's rate.
    let per_seed: Vec<[Vec<f64>; 4]> = (1..=n)
        .into_par_iter()
        .map(|s| -> Result<[Vec<f64>; 4]> {
            let (fz, _, _) = run_arm(&d, s, Arm::Frozen)?;
            let (lr, h, t) = run_arm(&d, s, Arm::Learning)?;
            let (sh, _, _) = run_arm(&d, s, Arm::Shuffled)?;
            let p = h as f64 / t.max(1) as f64;
            let (ind, _, _) = run_arm(&d, s, Arm::Independent(p))?;
            eprintln!("seed {s}: p(helped) {p:.4}");
            Ok([fz, lr, sh, ind])
        })
        .collect::<Result<_>>()?;
    // Average over seeds per test query.
    let avg = |a: usize| -> Vec<f64> {
        (0..d.test.len())
            .map(|q| per_seed.iter().map(|s| s[a][q]).sum::<f64>() / per_seed.len() as f64)
            .collect()
    };
    let (fz, lr, sh, ind) = (avg(0), avg(1), avg(2), avg(3));
    let bm = bm25(&d);
    let sub = |v: &[f64]| unseen.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let r1 = stats::paired(&lr, &fz, 51);
    let r2 = stats::paired(&lr, &ind, 52);
    let r3 = stats::paired(&sub(&lr), &sub(&fz), 53);
    let sf = stats::paired(&sh, &fz, 54);
    let seen: Vec<usize> = (0..d.test.len()).filter(|i| !unseen.contains(i)).collect();
    let pick = |v: &[f64]| seen.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let rs = stats::paired(&pick(&lr), &pick(&fz), 55);
    let bar1 = r1.mean_diff > 0.0 && r1.boot_lo > 0.0;
    let bar2 = r2.mean_diff > 0.0 && r2.boot_lo > 0.0;
    let bar3 = r3.mean_diff >= -0.005 || (r3.boot_lo <= 0.0 && r3.boot_hi >= 0.0);
    let met = |b: bool| if b { "**met**" } else { "**not met**" };

    let mut md = format!(
        "# FlyRAG on BEIR SciFact (research/flyrag-scifact/PREREG.md)\n\nSeeds 1..={n}; HEAD `{head}`; per-query nDCG@10 averaged over seeds; 95% bootstrap CIs over the 300 test queries (10,000 resamples).\n\n| system | nDCG@10 (all 300) | unseen 104 | seen 196 |\n|---|---|---|---|\n"
    );
    for (name, v) in [
        ("FlyRAG frozen", &fz),
        ("FlyRAG learning", &lr),
        ("FlyRAG shuffled", &sh),
        ("FlyRAG independent", &ind),
        ("BM25 (context, no bar)", &bm),
    ] {
        md.push_str(&format!(
            "| {name} | {:.3} | {:.3} | {:.3} |\n",
            stats::mean(v),
            stats::mean(&sub(v)),
            stats::mean(&pick(v))
        ));
    }
    md.push_str(&format!(
        "\n| bar | value | verdict |\n|---|---|---|\n\
         | R1. learning − frozen > 0, CI excl. 0 | {:+.4} [{:+.4}, {:+.4}] | {} |\n\
         | R2. learning − independent > 0, CI excl. 0 | {:+.4} [{:+.4}, {:+.4}] | {} |\n\
         | R3. unseen: learning − frozen ≥ −0.005 or CI incl. 0 | {:+.4} [{:+.4}, {:+.4}] | {} |\n\
         | (reference) seen: learning − frozen | {:+.4} [{:+.4}, {:+.4}] | — |\n\
         | (reference) shuffled − frozen | {:+.4} [{:+.4}, {:+.4}] | — |\n",
        r1.mean_diff,
        r1.boot_lo,
        r1.boot_hi,
        met(bar1),
        r2.mean_diff,
        r2.boot_lo,
        r2.boot_hi,
        met(bar2),
        r3.mean_diff,
        r3.boot_lo,
        r3.boot_hi,
        met(bar3),
        rs.mean_diff,
        rs.boot_lo,
        rs.boot_hi,
        sf.mean_diff,
        sf.boot_lo,
        sf.boot_hi,
    ));
    println!("{md}");
    std::fs::write(out_dir().join("flyrag_scifact.md"), &md)?;
    let json = format!(
        "{{\"head\":\"{head}\",\"seeds\":{n},\"mean\":{{\"frozen\":{},\"learning\":{},\"shuffled\":{},\"independent\":{},\"bm25\":{}}},\
         \"r1\":[{},{},{}],\"r2\":[{},{},{}],\"r3\":[{},{},{}],\"bars\":[{bar1},{bar2},{bar3}],\
         \"per_query\":{{\"frozen\":{fz:?},\"learning\":{lr:?},\"shuffled\":{sh:?},\"independent\":{ind:?},\"bm25\":{bm:?}}}}}\n",
        stats::mean(&fz),
        stats::mean(&lr),
        stats::mean(&sh),
        stats::mean(&ind),
        stats::mean(&bm),
        r1.mean_diff,
        r1.boot_lo,
        r1.boot_hi,
        r2.mean_diff,
        r2.boot_lo,
        r2.boot_hi,
        r3.mean_diff,
        r3.boot_lo,
        r3.boot_hi
    );
    std::fs::write(out_dir().join("flyrag_scifact.json"), json)?;
    Ok(())
}
