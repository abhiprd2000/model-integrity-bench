#!/usr/bin/env bash
# Demo 2: mib verify-read refuses the tampered model before serving the bad block.
# Needs demo/work/{clean,tampered}.onnx (run demo1_silent_failure.py first).
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mib="$root/target/release/mib"; w="$root/demo/work"
[ -x "$mib" ] || cargo build --release --quiet --manifest-path "$root/Cargo.toml" --bin mib || exit 1
[ -f "$w/demo.pub" ] || "$mib" keygen "$w/demo.seed" "$w/demo.pub" >/dev/null 2>&1

# Same model id for both files so only the bytes differ. The tampered file gets the clean sidecar.
seal=$("$mib" seal "$w/clean.onnx" --key "$w/demo.seed" --version 1 --model-id clean.onnx --out "$w/clean.onnx.mib" 2>&1)
n=$(sed -n 's/.*: \([0-9]*\) blocks.*/\1/p' <<<"$seal")
echo "Sealed clean model: $n blocks, signed root"

"$mib" verify-read "$w/clean.onnx" --pubkey "$w/demo.pub" --sidecar "$w/clean.onnx.mib" --model-id clean.onnx >/dev/null 2>"$w/clean.err"
[ $? -eq 0 ] && echo "Reading clean model ... OK" || { echo "Reading clean model ... FAILED"; exit 1; }

echo "Reading tampered model ..."
"$mib" verify-read "$w/tampered.onnx" --pubkey "$w/demo.pub" --sidecar "$w/clean.onnx.mib" --model-id clean.onnx >"$w/tampered.out" 2>"$w/tampered.err"
rc=$?
i=$(sed -n 's/^mib: block \([0-9]*\) .*/\1/p' "$w/tampered.err")
[ "$rc" -eq 3 ] && [ -n "$i" ] || { echo "unexpected: exit $rc"; cat "$w/tampered.err"; exit 1; }
echo "Block $i / $n FAILED: read refused (exit $rc)"

# Measured, not assumed: bytes emitted past the end of the last good block, and that
# everything emitted is byte-identical to the clean file.
out=$(stat -c %s "$w/tampered.out"); bad=$(( out > i*4096 ? out - i*4096 : 0 ))
cmp -s -n "$out" "$w/tampered.out" "$w/clean.onnx" || { echo "served bytes differ from clean file"; exit 1; }
echo "Bytes served from the bad block: $bad"
