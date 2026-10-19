# testing (flyrag)
- Behaviour tests (`tests/behaviour.rs`, `tests/extensions.rs`) cover each promised property:
  no feedback → no change, forward-only credit once per offer, bounded learning, frozen changes
  nothing, re-rank keeps the base order without feedback.
- Experiments are not unit tests: they live in `bench/` and report against committed bars.
- After any library change, re-run `scripts/reproduce.sh` into a scratch `KENYON_OUT` and diff the
  JSON against `results/` (ignoring `head`). Prereg paths must reproduce byte-for-byte.
- Never `#[ignore]` a failing test, weaken an assertion, or `#[allow]` a lint to get green.
