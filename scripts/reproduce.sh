#!/usr/bin/env bash
# Re-run every FlyRAG study into $KENYON_OUT (default results/). SciFact needs
# $KENYON_DATA/scifact (BEIR scifact.zip, sha256 in research/flyrag-scifact/PREREG.md).
set -euo pipefail
cd "$(dirname "$0")/.."
export KENYON_OUT=${KENYON_OUT:-results}
timeout 1800 cargo build --release -p flyrag-bench
B=target/release
timeout 1800 $B/flyrag_eval 20 > "$KENYON_OUT/flyrag.md" 2> "$KENYON_OUT/flyrag.log"
timeout 1800 $B/flyrag_followup dev > /dev/null
timeout 1800 $B/flyrag_followup eval 0.10 4 > /dev/null
timeout 3000 $B/flyrag_scifact 20 > /dev/null 2> "$KENYON_OUT/flyrag_scifact.log"
timeout 3600 $B/flyrag_next 20 > /dev/null 2> "$KENYON_OUT/flyrag_next.log"
echo "reproduce ok"
