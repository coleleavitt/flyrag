//! BEIR SciFact loading, nDCG@10 and the BM25 baseline, shared by
//! `flyrag_scifact` and `flyrag_next`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use flyrag::TextEncoder;

/// The corpus and the train/test splits.
pub struct Data {
    /// Document texts (title + " " + abstract), in file order.
    pub docs: Vec<String>,
    /// Train and test queries: text and relevant document indices.
    pub train: Vec<(String, HashSet<usize>)>,
    pub test: Vec<(String, HashSet<usize>)>,
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

pub fn load(dir: &Path) -> Result<Data> {
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
pub fn ndcg10(ranked: &[usize], rel: &HashSet<usize>) -> f64 {
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

/// BM25 (k1 = 0.9, b = 0.4) over FlyRAG's unigram tokens.
pub struct Bm25 {
    toks_len: Vec<f64>,
    tf: Vec<HashMap<String, f64>>,
    df: HashMap<String, f64>,
    avgdl: f64,
}

impl Bm25 {
    pub fn new(docs: &[String]) -> Self {
        let toks: Vec<Vec<String>> = docs.iter().map(|t| TextEncoder::tokens(t)).collect();
        let n = toks.len() as f64;
        let avgdl = toks.iter().map(Vec::len).sum::<usize>() as f64 / n;
        let mut df: HashMap<String, f64> = HashMap::new();
        let mut tf = Vec::with_capacity(toks.len());
        for t in &toks {
            let mut m: HashMap<String, f64> = HashMap::new();
            for w in t {
                *m.entry(w.clone()).or_default() += 1.0;
            }
            for w in m.keys() {
                *df.entry(w.clone()).or_default() += 1.0;
            }
            tf.push(m);
        }
        Self {
            toks_len: toks.iter().map(|t| t.len() as f64).collect(),
            tf,
            df,
            avgdl,
        }
    }

    /// The top `n` documents for `q`, (index, score), best first; ties by index.
    pub fn rank(&self, q: &str, n: usize) -> Vec<(usize, f64)> {
        let (k1, b) = (0.9, 0.4);
        let nd = self.tf.len() as f64;
        let qt: HashSet<String> = TextEncoder::tokens(q).into_iter().collect();
        let mut scores: Vec<(usize, f64)> = self
            .tf
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let dl = self.toks_len[i];
                let s = qt
                    .iter()
                    .filter_map(|w| {
                        let f = *m.get(w.as_str())?;
                        let dfw = self.df.get(w.as_str()).copied().unwrap_or(0.0);
                        let idf = (1.0 + (nd - dfw + 0.5) / (dfw + 0.5)).ln();
                        Some(idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * dl / self.avgdl)))
                    })
                    .sum();
                (i, s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scores.truncate(n);
        scores
    }
}
