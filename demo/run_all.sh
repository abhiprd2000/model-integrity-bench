#!/usr/bin/env bash
# Run the four demo beats in order for screen recording. Press Enter after each beat.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; cd "$root"
mkdir -p demo/work; log=demo/work/build.log; : >"$log"
py=.venv/bin/python

# Silent prep: assets and builds, so nothing noisy appears on camera.
if [ ! -f demo/assets/dog.jpg ] || [ ! -f demo/assets/labels.txt ]; then
  demo/get_assets.sh >>"$log" 2>&1 || { echo "asset download failed, see $log"; exit 1; }
fi
cargo build --release --bins --examples >>"$log" 2>&1 || { echo "build failed, see $log"; exit 1; }

beat() {  # beat <title> <command...>
  clear; echo "$1"; echo
  shift; "$@" 2>>"$log"; echo
  read -r -p "" _
}

beat "1/4  A single bit flip"               $py -I demo/demo1_silent_failure.py
beat "2/4  Caught on read"                  demo/demo2_verify_on_read.sh
beat "3/4  Honest writes vs tampering"      demo/demo3_write_path.sh
beat "4/4  The cost"                        demo/demo4_cost.sh
