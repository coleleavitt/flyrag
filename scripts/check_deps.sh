#!/usr/bin/env bash
# flyrag keeps exactly two runtime dependencies: kenyon and thiserror.
set -euo pipefail
cd "$(dirname "$0")/.."
deps=$(cargo tree -p flyrag -e normal --depth 1 --prefix none | tail -n +2 | awk '{print $1}' | sort -u | tr '\n' ' ')
if [ "$deps" != "kenyon thiserror " ]; then
  echo "flyrag runtime dependencies must be exactly 'kenyon thiserror'; found: $deps" >&2
  exit 1
fi
echo "deps ok: $deps"
