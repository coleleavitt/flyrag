# security (flyrag)
- No secrets in the repo, results or CI logs. CI reads kenyon with the read-only deploy key secret
  `KENYON_DEPLOY_KEY`; never print it.
- Datasets live outside the repo (`~/datasets`), with source URL and sha256 in the study's PREREG.
  Never load pickle files.
- The only git dependency is kenyon, pinned by rev. Other dependencies come from crates.io with the
  lockfile committed; adding one to `flyrag` itself needs the operator's approval.
- No `cargo publish`; push only to `origin` when the operator asks.
