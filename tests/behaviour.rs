//! FlyRAG's promised behaviours (research/flyrag/PREREG.md), one test each.

use flyrag::{FlyRag, FlyRagError, Learned, OfferId, Outcome, RagParams};

const INSTALL: &str = "install the dependencies with bun install, not npm install";
const NPM: &str = "install the dependencies with npm install and npm ci";
const LOGS: &str = "rotate the server logs every night with logrotate and gzip";
const Q_INSTALL: &str = "how do I install the dependencies";
const Q_LOGS: &str = "how do I rotate the server logs";

fn store() -> (FlyRag, [flyrag::DocId; 3]) {
    let mut r = FlyRag::new(RagParams::prereg(11)).expect("prereg params are valid");
    let a = r.add(INSTALL).expect("add");
    let b = r.add(NPM).expect("add");
    let c = r.add(LOGS).expect("add");
    (r, [a, b, c])
}

/// Prereg bar 5: queries without feedback change no bank weight.
#[test]
fn no_feedback_no_change() {
    let (mut r, docs) = store();
    for _ in 0..50 {
        r.query(Q_INSTALL, 10).unwrap();
        r.query(Q_LOGS, 10).unwrap();
    }
    for d in docs {
        assert_eq!(r.moved_units(d), 0);
    }
    let a = r.query(Q_INSTALL, 10).unwrap();
    assert!(a.hits.iter().all(|h| h.learned == 0.0));
}

/// Credit is forward-only and once per offer and document.
#[test]
fn feedback_only_for_offered_documents_once() {
    let (mut r, docs) = store();
    // Offer just the best document; any other one was never shown.
    let a = r.query(Q_INSTALL, 1).unwrap();
    assert_eq!(a.hits.len(), 1);
    let shown = a.hits[0].doc;
    let hidden = docs.into_iter().find(|&d| d != shown).unwrap();
    assert!(matches!(
        r.feedback(a.offer, hidden, Outcome::Helped),
        Err(FlyRagError::NotOffered { .. })
    ));
    assert_eq!(r.moved_units(hidden), 0);
    assert!(matches!(
        r.feedback(OfferId(999), shown, Outcome::Helped),
        Err(FlyRagError::UnknownOffer(999))
    ));
    assert!(matches!(
        r.feedback(a.offer, shown, Outcome::Helped),
        Ok(Learned::Updated { .. })
    ));
    assert!(matches!(
        r.feedback(a.offer, shown, Outcome::Helped),
        Err(FlyRagError::AlreadyAnswered { .. })
    ));
}

/// A verified "helped" raises the document for queries like this one, and
/// the learned term stays bounded however often it is repeated.
#[test]
fn helped_rises_and_stays_bounded() {
    let (mut r, [install, npm, _]) = store();
    let before = r.query(Q_INSTALL, 10).unwrap();
    let rank = |a: &flyrag::Answer, d| a.hits.iter().position(|h| h.doc == d);
    assert!(
        rank(&before, npm) < rank(&before, install),
        "npm starts ahead"
    );
    for _ in 0..200 {
        let a = r.query(Q_INSTALL, 10).unwrap();
        for h in &a.hits {
            let o = if h.doc == install {
                Outcome::Helped
            } else {
                Outcome::Misled
            };
            r.feedback(a.offer, h.doc, o).unwrap();
        }
    }
    let after = r.query(Q_INSTALL, 10).unwrap();
    assert_eq!(after.hits[0].doc, install);
    for h in &after.hits {
        assert!((-1.0..=1.0).contains(&h.learned));
    }
}

/// An outcome that is already expected teaches nothing.
#[test]
fn an_expected_outcome_teaches_nothing() {
    let (mut r, [install, ..]) = store();
    let mut last = None;
    for _ in 0..200 {
        let a = r.query(Q_INSTALL, 10).unwrap();
        last = Some(r.feedback(a.offer, install, Outcome::Helped).unwrap());
    }
    assert_eq!(last, Some(Learned::Expected));
}

/// Learning is context-specific: a lesson for install queries does not move
/// the document for an unrelated query.
#[test]
fn lessons_stay_in_their_context() {
    let (mut r, [install, ..]) = store();
    for _ in 0..20 {
        let a = r.query(Q_INSTALL, 10).unwrap();
        r.feedback(a.offer, install, Outcome::Helped).unwrap();
    }
    let unrelated = r
        .query("install the dependencies of the logrotate job", 10)
        .unwrap();
    let related = r.query(Q_INSTALL, 10).unwrap();
    let l = |a: &flyrag::Answer| {
        a.hits
            .iter()
            .find(|h| h.doc == install)
            .map_or(0.0, |h| h.learned)
    };
    assert!(
        l(&related) > l(&unrelated),
        "{} vs {}",
        l(&related),
        l(&unrelated)
    );
}

/// Frozen: feedback is accepted and changes nothing.
#[test]
fn frozen_changes_nothing() {
    let (mut r, [install, ..]) = store();
    r.set_learning(false);
    let a = r.query(Q_INSTALL, 10).unwrap();
    assert_eq!(
        r.feedback(a.offer, install, Outcome::Helped).unwrap(),
        Learned::Frozen
    );
    assert_eq!(r.moved_units(install), 0);
}

/// A query unlike anything in the store is flagged novel; a familiar one is
/// not.
#[test]
fn novelty_flags_the_unfamiliar() {
    let (mut r, _) = store();
    for _ in 0..3 {
        r.query(Q_INSTALL, 10).unwrap();
    }
    assert!(!r.query(Q_INSTALL, 10).unwrap().novel);
    let odd = r
        .query("quaternion spline interpolation for drone attitude", 10)
        .unwrap();
    assert!(odd.novel);
    assert!(odd.novelty > 0.8);
}
