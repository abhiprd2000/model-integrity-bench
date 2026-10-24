# Model integrity demo

Four short beats showing why per-block integrity checking matters for on-device models.

1. **A single bit flip.** Flip one bit in one weight of ResNet-50 on disk. The model still loads and runs, raises no error, and the top-1 label changes (Samoyed → bannister).
2. **Caught on read.** `mib verify-read` checks each 4 KiB block against a signed Merkle tree and refuses the tampered file at the bad block, serving none of its bytes.
3. **Honest writes vs tampering.** An honest write updates one leaf and its path to the root (then the root is re-signed); a write that bypasses the tree is caught by `first_bad_block`.
4. **The cost.** One verified 4 KiB read vs. re-hashing the whole file, for three models.

## Run

```bash
demo/get_assets.sh     # downloads dog.jpg and labels.txt into demo/assets/
demo/run_all.sh        # builds silently, then runs the 4 beats; press Enter after each
```

Needs `cargo` and the repo's `.venv` (onnxruntime, onnx, numpy, Pillow). Beats can also be run on their own: `demo1_silent_failure.py` (run first, it creates `demo/work/{clean,tampered}.onnx`), `demo2_verify_on_read.sh`, `demo3_write_path.sh`, `demo4_cost.sh`. Build output goes to `demo/work/build.log`.

## Honest notes

- The bit flip took **1 try**: bit 30 of the first weight (`resnetv17_conv0_weight[0]`) of the first conv layer. No search over other weights was needed.
- `verify-read` served **8 verified blocks (32768 bytes) and 0 bytes of the bad block** (block 8 of 25044). The script measures this by comparing the output size with the bad block's offset.
- An honest write costs the tree update **4.71 µs** plus re-signing the root **29.13 µs** (about 33.8 µs together). Re-signing is reported separately and dominates. A whole-file re-hash of the same 97.8 MiB takes about 44 ms.
- Timings use `bench_ns` from `src/mib.rs`: warm-up, then the median over trials (101 trials of 200 iterations for per-block operations and signing; 11 single runs after 3 warm-ups for whole-file hashing).
- `tiny.tflite` is an unnamed int8 model (an LSTM plus a dense layer, input `[1,1,257]`). Its metadata gives no name or purpose, so it is not labelled as any known model. `mid.tflite` is MobileNetV2 (float32).
- `dog.jpg`'s licence is unconfirmed, so it is downloaded by `get_assets.sh` rather than committed. `demo/assets/` and `demo/work/` are git-ignored.
- All numbers come from an Intel Core i5-5200U; exact values vary run to run.
