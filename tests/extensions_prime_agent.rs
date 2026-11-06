//! Extensions required for first-class prime-agent integration: serde
//! round-trips, `FlyRag::tombstone` / `is_tombstoned`, and `doc_weights`
//! introspection.

use flyrag::{FlyRag, Outcome, RagParams};

fn seeded() -> FlyRag {
    let mut r = FlyRag::new(RagParams::prereg(17)).expect("new");
    r.fit_idf(&[
        "install the dependencies with bun install, not npm install",
        "install the dependencies with npm install and npm ci",
        "the test runner panics if the daemon is not restarted",
        "restarting the daemon is destructive; it kills active agents",
    ])
    .expect("fit_idf");
    for text in [
        "install with bun install not npm",
        "install with npm install or npm ci",
        "the daemon restarts kill active agents",
        "the test runner needs a daemon restart to see new code",
    ] {
        r.add(text).expect("add");
    }
    r
}

#[test]
fn tombstone_hides_a_document_from_future_queries_but_keeps_feedback_working() {
    let mut rag = seeded();
    let b = rag.query("how do I install the dependencies", 5).unwrap();
    assert!(!b.hits.is_empty());
    let target = b.hits[0].doc;

    assert!(!rag.is_tombstoned(target));
    assert!(rag.tombstone(target).unwrap());
    assert!(rag.is_tombstoned(target));
    assert!(!rag.tombstone(target).unwrap(), "idempotent");

    let after = rag.query("how do I install the dependencies", 5).unwrap();
    assert!(
        after.hits.iter().all(|h| h.doc != target),
        "tombstoned doc hidden from future queries"
    );

    // An outstanding offer from BEFORE the tombstone still settles: feedback
    // on the original offer (which carried `target`) is still accepted and
    // still moves banks.
    let moved_before = rag.moved_units(target);
    rag.feedback(b.offer, target, Outcome::Misled).unwrap();
    let moved_after = rag.moved_units(target);
    assert!(
        moved_after > moved_before,
        "feedback after tombstone still trains the readout (moved {moved_before} -> {moved_after})"
    );

    // The doc's weights are introspectable after the tombstone.
    let w = rag.doc_weights(target).expect("weights after tombstone");
    assert!(
        !w.use_weights.is_empty() || !w.skip_weights.is_empty(),
        "at least one bank moved"
    );
}

#[test]
fn doc_weights_reports_only_touched_units_in_order() {
    let mut rag = seeded();
    let q = rag.query("daemon restarts kill active agents", 5).unwrap();
    let doc = q.hits[0].doc;
    // Untouched: both banks are empty (prior is implicit).
    let before = rag.doc_weights(doc).unwrap();
    assert!(before.use_weights.is_empty() && before.skip_weights.is_empty());

    rag.feedback(q.offer, doc, Outcome::Helped).unwrap();
    let after = rag.doc_weights(doc).unwrap();
    // Helped depresses the "skip" bank: use_weights stays empty, skip_weights
    // lists the KC units of the query's code with w below the prior.
    assert!(after.use_weights.is_empty(), "Helped never touches `use`");
    assert!(
        !after.skip_weights.is_empty(),
        "Helped moves the skip bank on the query's units"
    );
    // Weights come back in ascending-unit order, per contract.
    for w in after.skip_weights.windows(2) {
        assert!(w[0].0 < w[1].0, "unit order: {w:?}");
    }
    let p = rag.params().readout;
    assert!(
        after
            .skip_weights
            .iter()
            .all(|&(_, v)| (p.floor..=p.prior).contains(&v)),
        "skip weights stay in [floor, prior]"
    );
}

#[test]
fn doc_weights_is_none_for_an_unknown_id() {
    let rag = seeded();
    assert!(rag.doc_weights(flyrag::DocId(999)).is_none());
    assert!(!rag.is_tombstoned(flyrag::DocId(999)));
}

#[test]
fn compact_reclaims_tombstoned_docs_and_remaps_live_ids() {
    use flyrag::{DocId, Outcome};
    let mut rag = seeded();
    // `seeded` adds 4 docs with ids 0..=3 by construction.
    let (victim_a, victim_b, survivor_a, survivor_b) =
        (DocId(0), DocId(1), DocId(2), DocId(3));

    // Train a readout on a survivor so we can verify it comes along
    // through the compact: ids change but the banks follow.
    rag.feedback_detached(survivor_a, Outcome::Helped).unwrap();
    let before = rag.doc_weights(survivor_a).unwrap();

    rag.tombstone(victim_a).unwrap();
    rag.tombstone(victim_b).unwrap();
    assert_eq!(rag.len(), 4, "len() counts slots including tombstoned");

    let (mapping, dropped) = rag.compact();
    assert_eq!(dropped, 2);
    assert_eq!(rag.len(), 2, "two slots freed");
    assert!(!mapping.contains_key(&victim_a));
    assert!(!mapping.contains_key(&victim_b));

    let survivor_a_new = *mapping.get(&survivor_a).expect("remapped");
    let survivor_b_new = *mapping.get(&survivor_b).expect("remapped");
    // Ascending-old-id order: survivor_a (old DocId(2)) lands at new
    // DocId(0); survivor_b (old DocId(3)) lands at new DocId(1).
    assert_eq!(survivor_a_new, DocId(0));
    assert_eq!(survivor_b_new, DocId(1));

    // The learned weights for survivor_a travel to its new id.
    let after = rag.doc_weights(survivor_a_new).unwrap();
    assert_eq!(after.use_weights, before.use_weights);

    // Further feedback on the NEW id still trains. Training uses the
    // post-compact ids; the old DocId is now a stale reference.
    let moved_before = rag.moved_units(survivor_b_new);
    rag.feedback_detached(survivor_b_new, Outcome::Misled).unwrap();
    assert!(rag.moved_units(survivor_b_new) > moved_before);
}

#[test]
fn feedback_detached_moves_the_readout_without_an_offer() {
    use flyrag::{DocId, Learned, Outcome};
    let mut rag = seeded();
    // Pick a doc without opening an offer at all — simulate a cert-stream
    // supersede that cites the doc after its last rerank is long evicted.
    let q = rag.query("install dependencies", 3).unwrap();
    let doc = q.hits[0].doc;
    let before = rag.moved_units(doc);
    assert_eq!(before, 0);

    // A detached `Misled` (supersede) must train the readout: the
    // regular `feedback()` path refuses an unknown offer; detached never
    // touches the offer book.
    let learned = rag
        .feedback_detached(doc, Outcome::Misled)
        .expect("detached feedback accepted");
    assert!(matches!(learned, Learned::Updated { .. }));
    let after = rag.moved_units(doc);
    assert!(
        after > before,
        "detached feedback moved units: {before} -> {after}"
    );

    // An unknown doc errors cleanly, doesn't panic.
    let err = rag.feedback_detached(DocId(9_999), Outcome::Helped).unwrap_err();
    assert!(matches!(err, flyrag::FlyRagError::UnknownDocument(9_999)));
}

#[test]
fn feedback_detached_still_works_after_tombstone() {
    use flyrag::Outcome;
    let mut rag = seeded();
    let q = rag.query("daemon restart", 3).unwrap();
    let doc = q.hits[0].doc;
    assert!(rag.tombstone(doc).unwrap());
    // Tombstoned docs still carry their banks and their doc_units, so a
    // supersede cite CAN still train the readout even after the doc is
    // retired. Detached feedback accepts it without error.
    rag.feedback_detached(doc, Outcome::Misled)
        .expect("detached feedback on a tombstoned doc still trains");
    assert!(rag.moved_units(doc) > 0);
}

#[test]
fn feedback_detached_honours_prediction_error_bounds() {
    use flyrag::{Learned, Outcome};
    let mut rag = seeded();
    let q = rag.query("install dependencies", 3).unwrap();
    let doc = q.hits[0].doc;

    // Repeatedly pushing `Helped` eventually saturates — same cap as the
    // regular feedback path. The caller never sees a step larger than
    // `readout.cap`.
    let cap = rag.params().readout.cap;
    for _ in 0..200 {
        if let Learned::Updated { step, .. } = rag.feedback_detached(doc, Outcome::Helped).unwrap()
        {
            assert!(step <= cap + 1e-6, "step {step} <= cap {cap}");
        }
    }
    // Once saturated, the expected matches target → further feedback
    // reports `Expected`, not another `Updated`.
    let final_update = rag.feedback_detached(doc, Outcome::Helped).unwrap();
    assert!(matches!(
        final_update,
        Learned::Expected | Learned::Updated { .. }
    ));
}

#[cfg(feature = "serde")]
mod serde_roundtrip {
    use super::*;

    #[test]
    fn flyrag_roundtrips_after_feedback_and_tombstone() {
        let mut rag = seeded();
        // One helped feedback and one tombstone before serialising.
        let q = rag.query("how do I install the dependencies", 5).unwrap();
        let helped = q.hits[0].doc;
        rag.feedback(q.offer, helped, Outcome::Helped).unwrap();
        let q2 = rag.query("the daemon restart story", 5).unwrap();
        let retired = q2.hits.last().unwrap().doc;
        rag.tombstone(retired).unwrap();

        let bytes = serde_json::to_vec(&rag).unwrap();
        let mut back: FlyRag = serde_json::from_slice(&bytes).unwrap();

        // The learned weights round-trip exactly.
        assert_eq!(rag.doc_weights(helped), back.doc_weights(helped));
        assert_eq!(rag.doc_weights(retired), back.doc_weights(retired));
        // Tombstone status survives.
        assert!(back.is_tombstoned(retired));
        assert!(!back.is_tombstoned(helped));
        // Queries after reload see the same hits (minus the retired one).
        let a = rag.query("how do I install the dependencies", 3).unwrap();
        let b = back.query("how do I install the dependencies", 3).unwrap();
        let ids = |a: &flyrag::Answer| a.hits.iter().map(|h| h.doc.0).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
        assert!(!ids(&a).contains(&retired.0));
    }

    /// A FlyRag with no feedback, no tombstones, still round-trips.
    #[test]
    fn flyrag_roundtrips_with_no_feedback_and_queries_the_same() {
        let rag = seeded();
        let bytes = serde_json::to_vec(&rag).unwrap();
        let mut back: FlyRag = serde_json::from_slice(&bytes).unwrap();
        let mut fresh = seeded();
        let a = fresh.query("install", 3).unwrap();
        let b = back.query("install", 3).unwrap();
        assert_eq!(
            a.hits.iter().map(|h| h.doc.0).collect::<Vec<_>>(),
            b.hits.iter().map(|h| h.doc.0).collect::<Vec<_>>()
        );
    }
}
