#!/usr/bin/env bash
# Demo 3: tree update on an honest write vs. detection of an unrecorded write.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo run --release --quiet --example demo_write -- demo/work/clean.onnx
