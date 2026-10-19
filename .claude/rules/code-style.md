# code style (flyrag)
- Edition 2024, MSRV 1.85; `#![forbid(unsafe_code)]`; clippy `-D warnings`; no `unwrap` outside tests.
- Errors are `FlyRagError` variants; invalid parameters return errors, never panic.
- New behaviour is an option with the old behaviour as default (`RagParams::prereg` is frozen).
- Every `.rs` file is reachable from a `mod`; formatting follows the repo `rustfmt.toml`.
