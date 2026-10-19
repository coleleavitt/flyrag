#!/usr/bin/env bash
# The CI checks (.github/workflows/ci.yml), run locally, plus the 1.85.0 MSRV.
set -euo pipefail
cd "$(dirname "$0")/.."
step() { echo "== $*"; }
step fmt;     timeout 300  cargo fmt --check
step clippy;  timeout 1200 cargo clippy --workspace --all-targets -- -D warnings
step doc;     RUSTDOCFLAGS="-D warnings" timeout 600 cargo doc -p flyrag --no-deps
step deps;    scripts/check_deps.sh
step test;    timeout 1800 cargo test --workspace
if rustup toolchain list | grep -q '^1\.85\.0'; then
  step msrv;  timeout 1500 cargo +1.85.0 test -p flyrag
else
  echo "== msrv: skipped (rustup toolchain install 1.85.0 to run it)"
fi
echo "== ci ok"
