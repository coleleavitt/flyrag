//! FlyRAG's evaluation harness: the synthetic benchmark, BEIR SciFact and
//! small statistics helpers shared by the `src/bin/*` studies in `research/`.
//! Not published.

pub mod scifact;
pub mod stats;
pub mod synthetic;

use std::path::PathBuf;

/// Dataset root (`$KENYON_DATA`, default `~/datasets`).
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("KENYON_DATA") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("datasets")
}

/// Output directory for result files (`$KENYON_OUT`, default `results/`).
pub fn out_dir() -> PathBuf {
    let p = std::env::var("KENYON_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("results"));
    let _ = std::fs::create_dir_all(&p);
    p
}
