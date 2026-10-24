#!/usr/bin/env bash
# Demo 4: per-model cost of one verified read vs. whole-file re-hash.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
cargo run --release --quiet --example demo_cost -- \
  "TFLite LSTM (int8)=models/tiny.tflite" \
  "MobileNetV2 (TFLite)=models/mid.tflite" \
  "ResNet-50 (ONNX)=models/large.onnx"
