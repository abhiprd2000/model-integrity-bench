#!/usr/bin/env bash
# Run the model-integrity benchmarks on an aarch64 Android phone (Termux).
#
#   tar xzf arm_bundle.tar.gz && cd arm_bundle && bash run_arm.sh
#
# Extract under Termux's home ($HOME), NOT /sdcard or ~/storage: Android mounts
# shared storage noexec, so the binaries could not start there.
# Needs: wget, sha256sum, tar (pkg install wget coreutils tar). No root.
# Env: MODELS_DIR (default ./models), RESULTS_DIR (default ./results_arm),
#      OUT_DIR (default ~/storage/downloads; run `termux-setup-storage` once).
# Skipped on ARM by design: E5 cold-cache load (needs root drop_caches and a
# 4 GiB eviction file), r_workloads head (needs the python-made model copy),
# crash/fsync tests, taskset. Nothing here needs root.
# Stops on the first failure and says which step failed; the tarball is still
# written and contains STATUS.txt so a partial run is never mistaken for a full one.
set -u
cd "$(dirname "$0")"
export MODELS_DIR="${MODELS_DIR:-./models}"
export RESULTS_DIR="${RESULTS_DIR:-./results_arm}"
OUT_DIR="${OUT_DIR:-$HOME/storage/downloads}"

case "$(pwd -P)" in
  /sdcard*|/storage*|/mnt/*) echo "ERROR: running from shared storage ($(pwd -P)); copy the bundle to \$HOME first (noexec)."; exit 2;;
esac
[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: uname -m is '$(uname -m)', expected aarch64"; exit 2; }

mkdir -p "$MODELS_DIR" "$RESULTS_DIR"
STATUS="$RESULTS_DIR/STATUS.txt"
echo "started $(date -Is)" > "$STATUS"

# Expected sha256 of the three model files = the laptop's copies (the data the
# paper's numbers were measured on).
declare -A URL SHA FILE
FILE[tiny]=tiny.tflite
URL[tiny]="https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite"
SHA[tiny]=91e94316d1a4747dd3bddcf58423927e97213913a9b4210edca3f9402b2ff0fe
FILE[mid]=mid.tflite
URL[mid]="https://huggingface.co/litert-community/MobileNet-v2/resolve/main/mobilenet_v2.tflite"
SHA[mid]=34853c702f8da7298c6ae6c100b8a23df98b11d4fb41aa200e6fa45e96d52ea6
FILE[large]=large.onnx
URL[large]="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx"
SHA[large]=3f03fdef724b22947eed826f1eef1dc5c34151bb4c37d634f1db89dfa2dd1526

finish() {  # tar results (also on failure) then exit with $1
  local rc=$1
  [ "$rc" -eq 0 ] && echo "finished OK $(date -Is)" >> "$STATUS"
  mkdir -p "$OUT_DIR" 2>/dev/null || true
  [ -d "$OUT_DIR" ] && [ -w "$OUT_DIR" ] || { echo "NOTE: $OUT_DIR not writable (run termux-setup-storage); writing to $PWD instead"; OUT_DIR="$PWD"; }
  tar czf "$OUT_DIR/results_arm.tar.gz" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")" \
    && echo "results: $OUT_DIR/results_arm.tar.gz ($(wc -c < "$OUT_DIR/results_arm.tar.gz") bytes)"
  exit "$rc"
}
fail() { echo "FAILED: $*"; echo "FAILED at $(date -Is): $*" >> "$STATUS"; finish 1; }

echo "== integrity of the bundled binaries"
[ -f SHA256SUMS ] && { sha256sum -c SHA256SUMS || fail "binary checksum mismatch (corrupted transfer?)"; }

echo "== models (${MODELS_DIR})"
for m in tiny mid large; do
  f="$MODELS_DIR/${FILE[$m]}"
  if [ ! -s "$f" ]; then
    echo "downloading ${FILE[$m]}"
    wget -O "$f.part" "${URL[$m]}" || { rm -f "$f.part"; fail "download of ${FILE[$m]}"; }
    mv "$f.part" "$f"
  fi
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" != "${SHA[$m]}" ]; then
    fail "sha256 mismatch for ${FILE[$m]}: got $got, laptop has ${SHA[$m]} (not running benchmarks on different data)"
  fi
  echo "ok  ${FILE[$m]}  $got"
done

echo "== device"
{
  echo "date: $(date -Is)"
  echo "uname: $(uname -a)"
  echo "nproc: $(nproc)"
  command -v getprop >/dev/null && for k in ro.product.manufacturer ro.product.model ro.soc.manufacturer ro.soc.model ro.board.platform ro.build.version.release; do echo "$k: $(getprop $k)"; done
  echo "--- /proc/cpuinfo (summary of distinct lines)"
  grep -E "^(Hardware|model name|CPU implementer|CPU architecture|CPU variant|CPU part|Features)" /proc/cpuinfo | sort | uniq -c
  echo "--- cpufreq (may be unreadable without root)"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    [ -r "$c/cpufreq/scaling_governor" ] && echo "$(basename $c): $(cat $c/cpufreq/scaling_governor) max=$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
  done
} 2>&1 | tee "$RESULTS_DIR/device_info.txt"
echo "NOTE: keep the screen on, plugged in, and other apps closed; phone SoCs throttle."

run() {  # run <name> <binary> [args...]
  local name=$1; shift
  echo "== $name  ($(date +%T))"
  "$@" > "$RESULTS_DIR/$name.log" 2>&1 || { tail -5 "$RESULTS_DIR/$name.log"; fail "$name exited non-zero (see $name.log)"; }
  echo "   done ($(date +%T))"
}
run main_bench ./model-integrity-bench
run experiments ./experiments
run r_workloads_sweep ./r_workloads sweep

finish 0
