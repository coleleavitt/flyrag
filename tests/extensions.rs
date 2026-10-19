//! The extensions of research/flyrag-next/PREREG.md: re-ranking another
//! retriever's candidates, IDF weighting, novelty against documents.

use flyrag::{DocId, FlyRag, FlyRagError, NoveltySource, Outcome, RagParams};

const DOCS: [&str; 3] = [
    "install the dependencies with bun install, not npm install",
    "install the dependencies with npm install and npm ci",
    "rotate the server logs every night with logrotate and gzip",
];
const Q: &str = "how do I install the dependencies";

fn store(params: RagParams) -> FlyRag {
    let mut r = FlyRag::new(params).expect("valid params");
    for d in DOCS {
        r.add(d).expect("add");
    }
    r
}

/// Without feedback, re-ranking keeps the base retriever's order (ties by id).
#[test]
fn rerank_without_feedback_keeps_the_base_order() {
    let mut r = store(RagParams::prereg(3));
    let base = [(DocId(2), 9.0), (DocId(0), 4.0), (DocId(1), 4.0)];
    let a = r.rerank(Q, &base, 10).unwrap();
    let order: Vec<u32> = a.hits.iter().map(|h| h.doc.0).collect();
    assert_eq!(order, [2, 0, 1]);
    assert!((a.hits[0].score - 1.0).abs() < 1e-12, "base / max_base");
    assert!(a.hits.iter().all(|h| h.learned == 0.0));
}

/// A verified "helped" on a re-ranked offer lifts that document above the
/// base retriever's favourite for the same kind of query.
#[test]
fn rerank_learns_from_outcomes() {
    let mut r = store(RagParams::prereg(3));
    let base = [(DocId(1), 1.0), (DocId(0), 0.9)];
    for _ in 0..6 {
        let a = r.rerank(Q, &base, 2).unwrap();
        r.feedback(a.offer, DocId(0), Outcome::Helped).unwrap();
        r.feedback(a.offer, DocId(1), Outcome::Misled).unwrap();
    }
    let a = r.rerank(Q, &base, 2).unwrap();
    assert_eq!(a.hits[0].doc, DocId(0));
    // Only offered documents can learn.
    let a = r.rerank(Q, &base[..1], 1).unwrap();
    assert!(matches!(
        r.feedback(a.offer, DocId(0), Outcome::Helped),
        Err(FlyRagError::NotOffered { .. })
    ));
}

#[test]
fn rerank_rejects_unknown_documents() {
    let mut r = store(RagParams::prereg(3));
    assert!(r.rerank(Q, &[(DocId(99), 1.0)], 1).is_err());
}

/// IDF is fitted before documents are added, and a bucket every text shares
/// gets the lowest weight.
#[test]
fn idf_fits_before_adding_and_changes_the_codes() {
    let mut late = store(RagParams::prereg(3));
    assert!(
        late.fit_idf(&DOCS).is_err(),
        "codes already depend on the weights"
    );
    let mut empty = FlyRag::new(RagParams::prereg(3)).unwrap();
    assert!(empty.fit_idf::<&str>(&[]).is_err());

    let mut plain = store(RagParams::prereg(3));
    let mut weighted = FlyRag::new(RagParams::prereg(3)).unwrap();
    weighted.fit_idf(&DOCS).unwrap();
    assert!(weighted.has_idf());
    for d in DOCS {
        weighted.add(d).unwrap();
    }
    let p = plain.query(Q, 3).unwrap();
    let w = weighted.query(Q, 3).unwrap();
    assert_ne!(
        p.hits.iter().map(|h| h.overlap).collect::<Vec<_>>(),
        w.hits.iter().map(|h| h.overlap).collect::<Vec<_>>()
    );
}

/// Document novelty: a stored document's own text is not novel, unrelated
/// text is, and queries are not remembered.
#[test]
fn novelty_against_documents() {
    let mut p = RagParams::prereg(3);
    p.novelty_source = NoveltySource::Documents;
    let mut r = store(p);
    let own = r.query(DOCS[2], 3).unwrap();
    assert_eq!(own.novelty, 0.0);
    let far = "quantum chromodynamics of gluon plasma at high temperature";
    let first = r.query(far, 3).unwrap().novelty;
    assert!(first > 0.5, "unrelated text is novel: {first}");
    assert_eq!(
        r.query(far, 3).unwrap().novelty,
        first,
        "queries are not stored"
    );
}
