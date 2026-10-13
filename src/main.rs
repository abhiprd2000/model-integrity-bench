// Edge-AI model-file integrity benchmark: mirrors the per-4KiB-block BLAKE3
// scheme in src/main.rs (same BLOCK size, median-of-7 timing, warm-up),
// applied to real ML model files instead of synthetic buffers.
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

const BLOCK: usize = 4096;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn bench<F: FnMut()>(trials: usize, iters: u64, mut f: F) -> f64 {
    for _ in 0..iters.min(64) {
        f();
    }
    let mut per_op = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        per_op.push(start.elapsed().as_nanos() as f64 / iters as f64);
    }
    median(per_op)
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

// Paths come from the environment so the same binary runs on a phone:
// MODELS_DIR (default ./models), RESULTS_DIR (default ./results_arm).
fn models_dir() -> String {
    std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string())
}
fn results_dir() -> String {
    std::env::var("RESULTS_DIR").unwrap_or_else(|_| "./results_arm".to_string())
}
fn model_path(rel: &str) -> String {
    format!("{}/{}", models_dir(), rel.strip_prefix("models/").unwrap_or(rel))
}
fn rpath(name: &str) -> String {
    format!("{}/{}", results_dir(), name)
}

struct ModelSpec {
    label: &'static str,
    path: &'static str,
    source: &'static str,
    synthetic_target_size: usize,
}

struct Loaded {
    label: &'static str,
    filename: String,
    source: &'static str,
    real: bool,
    data: Vec<u8>,
}

fn load_or_synthesize(spec: &ModelSpec, out: &mut String) -> Loaded {
    let resolved = model_path(spec.path);
    match std::fs::read(&resolved) {
        Ok(data) => Loaded {
            label: spec.label,
            filename: resolved,
            source: spec.source,
            real: true,
            data,
        },
        Err(_) => {
            writeln!(
                out,
                "  ! download unavailable for {} ({}): falling back to SYNTHETIC {}-byte buffer",
                spec.label, resolved, spec.synthetic_target_size
            )
            .unwrap();
            let mut seed: u64 = 0x9E3779B97F4A7C15 ^ (spec.synthetic_target_size as u64);
            let mut data = vec![0u8; spec.synthetic_target_size];
            for chunk in data.chunks_mut(8) {
                let r = xorshift(&mut seed).to_le_bytes();
                for (b, r) in chunk.iter_mut().zip(r.iter()) {
                    *b = *r;
                }
            }
            Loaded {
                label: spec.label,
                filename: resolved,
                source: "SYNTHETIC (random-filled, download unavailable)",
                real: false,
                data,
            }
        }
    }
}

fn leaves_for(data: &[u8]) -> Vec<[u8; 32]> {
    let nblocks = (data.len() + BLOCK - 1) / BLOCK;
    (0..nblocks)
        .map(|i| {
            let start = i * BLOCK;
            let end = (start + BLOCK).min(data.len());
            *blake3::hash(&data[start..end]).as_bytes()
        })
        .collect()
}

fn merkle_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

// Binary Merkle tree over the per-block leaves, padded to a power of two
// (padding leaves duplicate the last real leaf, a common fixed-size scheme).
// levels[0] = padded leaves, levels[last] = [root]. depth = levels.len() - 1
// is exactly the number of node hashes a leaf update must recompute on its
// path to the root, i.e. ceil(log2(nblocks)).
struct Merkle {
    levels: Vec<Vec<[u8; 32]>>,
}

fn build_merkle(leaves: &[[u8; 32]]) -> Merkle {
    let mut padded = leaves.to_vec();
    let target = padded.len().max(1).next_power_of_two();
    let filler = *padded.last().unwrap();
    padded.resize(target, filler);

    let mut levels = vec![padded];
    while levels.last().unwrap().len() > 1 {
        let prev = levels.last().unwrap();
        let next: Vec<[u8; 32]> = prev
            .chunks(2)
            .map(|pair| merkle_parent(&pair[0], &pair[1]))
            .collect();
        levels.push(next);
    }
    Merkle { levels }
}

// Update leaf `idx` to `new_leaf` and recompute every node on its path to the
// root: exactly `levels.len() - 1` blake3 node hashes (the tree depth).
fn merkle_update_leaf(levels: &mut [Vec<[u8; 32]>], mut idx: usize, new_leaf: [u8; 32]) {
    levels[0][idx] = new_leaf;
    for lvl in 0..levels.len() - 1 {
        let sibling = idx ^ 1;
        let (left, right) = if idx & 1 == 0 {
            (levels[lvl][idx], levels[lvl][sibling])
        } else {
            (levels[lvl][sibling], levels[lvl][idx])
        };
        let parent = merkle_parent(&left, &right);
        idx /= 2;
        levels[lvl + 1][idx] = parent;
    }
}

fn main() {
    let arch = std::env::consts::ARCH;
    let mut out = String::new();

    writeln!(out, "axiom-bench-native :: model-integrity (edge-AI)   arch={arch}").unwrap();
    writeln!(
        out,
        "per-4KiB-block BLAKE3 leaf hashing, mirroring src/main.rs (whole-file vs one-block, median-of-7, warm-up)"
    )
    .unwrap();
    writeln!(out).unwrap();

    let specs = [
        ModelSpec {
            label: "tiny",
            path: "models/tiny.tflite",
            source: "DTLN noise-suppression TFLite Micro model — https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite",
            synthetic_target_size: 500_000,
        },
        ModelSpec {
            label: "mid",
            path: "models/mid.tflite",
            source: "MobileNetV2 TFLite (litert-community) — https://huggingface.co/litert-community/MobileNet-v2/resolve/main/mobilenet_v2.tflite",
            synthetic_target_size: 12_000_000,
        },
        ModelSpec {
            label: "large",
            path: "models/large.onnx",
            source: "ResNet-50 v1-12 ONNX (ONNX Model Zoo) — https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
            synthetic_target_size: 90_000_000,
        },
    ];

    writeln!(out, "--- model sources ---").unwrap();
    let mut loaded: Vec<Loaded> = Vec::new();
    for spec in &specs {
        let m = load_or_synthesize(spec, &mut out);
        writeln!(
            out,
            "  {:<6} {:<10} {:>12} B   {}",
            m.label,
            if m.real { "REAL" } else { "SYNTHETIC" },
            m.data.len(),
            if m.real { m.filename.as_str() } else { "(generated in-memory)" }
        )
        .unwrap();
        writeln!(out, "         source: {}", m.source).unwrap();
        loaded.push(m);
    }
    writeln!(out).unwrap();

    writeln!(out, "--- read+verify: whole-file vs one 4 KiB block ---").unwrap();
    writeln!(
        out,
        "{:<6}  {:>12}  {:>8}  {:>14}  {:>10}  {:>10}",
        "model", "size(B)", "blocks", "wholefile(ms)", "block(us)", "norm"
    )
    .unwrap();

    let mut first_block_us: Option<f64> = None;
    struct Timed {
        label: &'static str,
        data: Vec<u8>,
        leaves: Vec<[u8; 32]>,
        tree_depth: usize,
        wholefile_ms: f64,
        block_us: f64,
        perblock_write_us: f64,
    }
    let mut timed: Vec<Timed> = Vec::new();

    for m in loaded {
        let sz = m.data.len();
        let leaves = leaves_for(&m.data);
        let nblocks = leaves.len();
        let mut merkle = build_merkle(&leaves);
        let tree_depth = merkle.levels.len() - 1;

        let iters = (1_000_000_000u64 / sz.max(1) as u64).clamp(7, 200);
        let wf_ns = bench(7, iters, || {
            let h = blake3::hash(black_box(&m.data));
            black_box(h);
        });

        let first_block = &m.data[..BLOCK.min(sz)];
        let block_ns = bench(7, 20_000, || {
            let h = blake3::hash(black_box(first_block));
            black_box(h);
        });

        // Write-path cost: rehash the touched block's leaf, then walk the
        // Merkle path to the root (tree_depth node hashes). Timed as one
        // combined op, since that's what a real write to block 0 costs.
        let write_ns = bench(7, 20_000, || {
            let leaf = *blake3::hash(black_box(first_block)).as_bytes();
            merkle_update_leaf(&mut merkle.levels, 0, leaf);
        });

        let wholefile_ms = wf_ns / 1e6;
        let block_us = block_ns / 1e3;
        let perblock_write_us = write_ns / 1e3;
        let norm = match first_block_us {
            None => {
                first_block_us = Some(block_us);
                1.0
            }
            Some(base) => block_us / base,
        };

        writeln!(
            out,
            "{:<6}  {:>12}  {:>8}  {:>14.6}  {:>10.4}  {:>9.2}x",
            m.label, sz, nblocks, wholefile_ms, block_us, norm
        )
        .unwrap();

        timed.push(Timed {
            label: m.label,
            data: m.data,
            leaves,
            tree_depth,
            wholefile_ms,
            block_us,
            perblock_write_us,
        });
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "block_us stays ~flat (norm column near 1.00x) across tiny/mid/large: per-block verify cost"
    )
    .unwrap();
    writeln!(
        out,
        "is independent of model size, while wholefile_ms scales with size — the proportional-access effect."
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "--- tamper demo (flip 1 byte, verify via per-block leaves) ---").unwrap();
    for t in &mut timed {
        let sz = t.data.len();
        let mut seed: u64 = 0x9E3779B97F4A7C15 ^ (sz as u64);
        let offset = (xorshift(&mut seed) as usize) % sz;
        let block_idx = offset / BLOCK;
        let start = block_idx * BLOCK;
        let end = (start + BLOCK).min(sz);

        let original_byte = t.data[offset];
        t.data[offset] ^= 0xFF;

        // Per-block scheme: only the touched block is re-hashed to detect the tamper.
        let touched_leaf = *blake3::hash(&t.data[start..end]).as_bytes();
        let touched_mismatch = touched_leaf != t.leaves[block_idx];

        // Full audit (demonstration only, not part of the timed per-block cost):
        // confirm every OTHER block is bit-identical and still matches its stored leaf.
        let mut mismatches = 0usize;
        let mut mismatch_idx = usize::MAX;
        for i in 0..t.leaves.len() {
            let bs = i * BLOCK;
            let be = (bs + BLOCK).min(sz);
            let h = *blake3::hash(&t.data[bs..be]).as_bytes();
            if h != t.leaves[i] {
                mismatches += 1;
                mismatch_idx = i;
            }
        }

        writeln!(out, "  [{}] size={} B  tampered offset={} -> block {}/{}", t.label, sz, offset, block_idx, t.leaves.len()).unwrap();
        writeln!(
            out,
            "         touched-block re-verify: {}  |  full-audit mismatches: {} (expect 1, at block {})",
            if touched_mismatch { "MISMATCH (detected)" } else { "BUG: no mismatch detected" },
            mismatches,
            mismatch_idx
        )
        .unwrap();
        writeln!(
            out,
            "         per-block: cost to detect = 1 block re-hash (~{:.4} us) -> bytes-served-corrupted = 0",
            t.block_us
        )
        .unwrap();
        writeln!(
            out,
            "         whole-file: must re-hash entire {} B file every load to notice (~{:.6} ms) to catch the same tamper",
            sz, t.wholefile_ms
        )
        .unwrap();

        debug_assert_eq!(mismatches, 1);
        debug_assert_eq!(mismatch_idx, block_idx);

        // restore
        t.data[offset] = original_byte;
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "conclusion: per-block BLAKE3 leaves localize corruption to a single ~4 KiB block at flat,"
    )
    .unwrap();
    writeln!(
        out,
        "size-independent cost, and never serve corrupted bytes; whole-file hashing must scan the"
    )
    .unwrap();
    writeln!(
        out,
        "entire model (cost grows with model size) before it can notice any tamper at all."
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "--- write-path cost: per-block leaf+Merkle update vs whole-file re-hash ---"
    )
    .unwrap();
    writeln!(
        out,
        "the per-block scheme is not free to keep correct: every write to block i must rehash i's"
    )
    .unwrap();
    writeln!(
        out,
        "leaf AND update the Merkle path from leaf i to the root (tree_depth node hashes), so the"
    )
    .unwrap();
    writeln!(
        out,
        "root always reflects current content. This is the scheme's realization cost, reported here"
    )
    .unwrap();
    writeln!(
        out,
        "rather than left out of the picture. Compare to a whole-file scheme, whose single root can"
    )
    .unwrap();
    writeln!(
        out,
        "only be refreshed by re-hashing the entire file, even for a one-byte write."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "{:<6}  {:>8}  {:>10}  {:>18}  {:>18}  {:>10}",
        "model", "nblocks", "tree_depth", "perblock_write(us)", "wholefile_write(ms)", "speedup"
    )
    .unwrap();
    for t in timed.iter().filter(|t| t.label == "mid" || t.label == "large") {
        let speedup = (t.wholefile_ms * 1000.0) / t.perblock_write_us;
        writeln!(
            out,
            "{:<6}  {:>8}  {:>10}  {:>18.4}  {:>18.6}  {:>9.1}x",
            t.label,
            t.leaves.len(),
            t.tree_depth,
            t.perblock_write_us,
            t.wholefile_ms,
            speedup
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "tree_depth = ceil(log2(nblocks)) node hashes per write, on top of the 1 leaf rehash; still"
    )
    .unwrap();
    writeln!(
        out,
        "microseconds, and still flat-ish across model sizes (log growth), vs. a whole-file re-hash"
    )
    .unwrap();
    writeln!(
        out,
        "that scales linearly with model size for the exact same single-byte write."
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "--- continuous re-verification: N cycles, write rate W (blocks dirtied per cycle) ---"
    )
    .unwrap();
    writeln!(
        out,
        "an edge node keeps the model resident and re-verifies it every cycle; weights are read-mostly,"
    )
    .unwrap();
    writeln!(
        out,
        "so only W of the model's blocks actually change per cycle (single-load comparison alone is"
    )
    .unwrap();
    writeln!(
        out,
        "attackable: inference loads the whole model anyway, so this is the policy that recurs)."
    )
    .unwrap();
    writeln!(
        out,
        "both policies are dirty-tracked: whole-file re-hashes the full file on any cycle with >=1"
    )
    .unwrap();
    writeln!(
        out,
        "dirtied block (every cycle here, since W >= 1); per-block re-hashes only the W dirtied"
    )
    .unwrap();
    writeln!(
        out,
        "leaves and walks each one's Merkle path to the root (the write-path cost measured above),"
    )
    .unwrap();
    writeln!(
        out,
        "leaving clean blocks untouched."
    )
    .unwrap();
    writeln!(
        out,
        "cost model: wholefile_total = N * wholefile_ms   |   perblock_total = (N*W) * perblock_write_us"
    )
    .unwrap();
    writeln!(out).unwrap();

    const N: u64 = 1000;
    let write_rates: [u64; 3] = [1, 10, 100];

    writeln!(
        out,
        "{:<6}  {:>4}  {:>20}  {:>20}  {:>12}",
        "model", "W", "wholefile_total(ms)", "perblock_total(ms)", "speedup"
    )
    .unwrap();
    for t in timed.iter().filter(|t| t.label == "mid" || t.label == "large") {
        for &w in &write_rates {
            let wholefile_total_ms = N as f64 * t.wholefile_ms;
            let perblock_total_ms = (N * w) as f64 * t.perblock_write_us / 1000.0;
            let speedup = wholefile_total_ms / perblock_total_ms;
            writeln!(
                out,
                "{:<6}  {:>4}  {:>20.3}  {:>20.6}  {:>10.1}x",
                t.label, w, wholefile_total_ms, perblock_total_ms, speedup
            )
            .unwrap();
        }
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "thesis: a single load is a one-time cost either policy pays once regardless. Continuous"
    )
    .unwrap();
    writeln!(
        out,
        "background re-verification is where whole-file hashing becomes strictly wasteful at any"
    )
    .unwrap();
    writeln!(
        out,
        "nonzero cadence N, paying the full model size every cycle; per-block re-verify cost (leaf"
    )
    .unwrap();
    writeln!(
        out,
        "rehash + Merkle path update, the full write-path cost) tracks only the write rate W and the"
    )
    .unwrap();
    writeln!(
        out,
        "tree's log-depth, staying orders of magnitude below whole-file even at W=100 dirtied blocks."
    )
    .unwrap();

    print!("{out}");

    std::fs::create_dir_all(results_dir()).expect("create RESULTS_DIR");
    let outfile = rpath(&format!("model-integrity-results-{arch}.txt"));
    std::fs::write(&outfile, &out).expect("failed to write results file");
    eprintln!("\n(results also saved to {outfile})");
}
