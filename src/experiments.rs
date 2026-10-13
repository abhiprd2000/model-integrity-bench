// Real-measurement experiment harness for the model-integrity benchmark.
//
// Everything written to results/*.csv is either (a) a direct wall-clock timing
// of real work on the real model files in models/ (mmap'd, median-of-N with
// warm-up, release build), or (b) exact integer/float arithmetic over measured
// quantities, explicitly labeled basis=computed / basis=derived in the CSV.
// No synthetic buffers, no hardcoded numbers: if a model file is missing the
// run aborts rather than inventing data.
//
// Experiments:
//   E1 block-size sweep (per-block verify, write-update, whole-file)   [measured]
//   E2 Merkle tree metadata per (model, block size)                    [computed]
//   E3 BLAKE3 vs SHA-256 on the SAME operation                         [measured]
//   E4 write-path breakdown (leaf rehash vs leaf+Merkle path)          [measured]
//   E5 model load overhead: plain vs whole-file hash vs per-block
//      verify, warm and cold (1.2 GiB eviction read before each trial) [measured]
//   E6 max re-verification cadence per CPU budget                      [derived from measured costs]
//   E7 cost model vs measurement + break-even W* + measured crossover  [measured + derived]
//
// Phases: default runs everything except E5 (the memory-heavy load test).
// Set MIB_ONLY=load to run only E5 (keeps each invocation under ~10 min).
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

const BLOCK: usize = 4096;

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
const SWEEP: [usize; 11] = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
// NOTE: /tmp is tmpfs (RAM) on this machine, so the eviction file must live on
// the real disk. The home filesystem (SSD, 96 GB, ~15 GB free) is used instead.
// The eviction file must EXCEED the machine's page-cache occupancy so one LRU
// pass evicts the model's pages: /proc/meminfo shows ~3.3 GiB Cached here, so
// the file is 4 GiB. Path overridable via MIB_EVICT_PATH.
const EVICT_PATH_DEFAULT: &str = ".cache/mib-evict.bin";
const EVICT_BYTES: u64 = 4096 * 1024 * 1024; // 4 GiB on the SSD (> peak page cache)

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// Median per-op nanoseconds over `trials` trials of `iters` inner repetitions,
// after `warmup` warm-up calls. Data is already resident (mmap + touch pass or
// freshly read Vec), so timed repetitions measure pure CPU hashing work.
fn bench_ns(warmup: usize, trials: usize, iters: u64, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
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

// Small deterministic RNG for choosing dirty-block schedules (outside timed regions).
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

// Merkle tree over per-block leaves, padded to a power of two.
// levels[0] = padded leaves, levels[last] = [root]; len(levels)-1 = tree depth.
fn build_merkle(leaves: &[[u8; 32]]) -> Vec<Vec<[u8; 32]>> {
    let mut padded = leaves.to_vec();
    let target = padded.len().max(1).next_power_of_two();
    let filler = *padded.last().unwrap();
    padded.resize(target, filler);
    let mut levels = vec![padded];
    while levels.last().unwrap().len() > 1 {
        let prev = levels.last().unwrap();
        let next: Vec<[u8; 32]> = prev
            .chunks(2)
            .map(|p| {
                let mut h = blake3::Hasher::new();
                h.update(&p[0]);
                h.update(&p[1]);
                *h.finalize().as_bytes()
            })
            .collect();
        levels.push(next);
    }
    levels
}

// Update leaf `idx` and recompute every node on its path to the root.
fn merkle_update_leaf(levels: &mut [Vec<[u8; 32]>], mut idx: usize, new_leaf: [u8; 32]) {
    levels[0][idx] = new_leaf;
    for lvl in 0..levels.len() - 1 {
        let sib = idx ^ 1;
        let (l, r) = if idx & 1 == 0 {
            (levels[lvl][idx], levels[lvl][sib])
        } else {
            (levels[lvl][sib], levels[lvl][idx])
        };
        let mut h = blake3::Hasher::new();
        h.update(&l);
        h.update(&r);
        let parent = *h.finalize().as_bytes();
        idx /= 2;
        levels[lvl + 1][idx] = parent;
    }
}

fn main() {
    let run_note = std::env::var("MIB_RUN_NOTE").unwrap_or_default();
    let only_load = std::env::var("MIB_ONLY").as_deref() == Ok("load");

    // ---- load the real model files (abort if missing: no synthetic data) ----
    let specs = [
        ("tiny", model_path("models/tiny.tflite")),
        ("mid", model_path("models/mid.tflite")),
        ("large", model_path("models/large.onnx")),
    ];
    let mut files: Vec<(&str, String, memmap2::Mmap)> = Vec::new();
    for (label, path) in specs {
        let f = fs::File::open(&path)
            .unwrap_or_else(|e| panic!("model file missing ({path}): {e} — refusing to substitute synthetic data"));
        let mmap = unsafe { memmap2::Mmap::map(&f).expect("mmap failed") };
        files.push((label, path.to_string(), mmap));
    }

    // Touch every page once so timed passes measure CPU hashing, not page faults.
    for (_, _, m) in &files {
        for page in m.chunks(4096) {
            black_box(page[0]);
        }
    }

    if only_load {
        if cfg!(target_arch = "aarch64") {
            eprintln!("E5 (cold-cache load) is skipped on aarch64 builds: it needs a 4 GiB eviction file / root drop_caches.");
            return;
        }
        run_e5(&files, &run_note);
        return;
    }

    let mut csv_e1 = String::from("model,block_size,verify_us,write_update_us,wholefile_us\n");
    let mut csv_e2 = String::from("model,block_size,nblocks,padded_leaves,total_tree_nodes,tree_bytes,tree_pct_of_model,basis\n");
    let mut csv_e3 = String::from("model,operation,algorithm,time_us\n");
    let mut csv_e4 = String::from("model,nblocks,tree_depth,leaf_rehash_us,leaf_plus_path_us\n");
    let mut csv_e6 = String::from("model,policy,write_rate_W,cpu_budget_frac,cycle_cost_us,max_check_freq_hz,basis\n");
    let mut csv_e7 = String::from("model,cycles_N,write_rate_W,predicted_ms,measured_ms,percent_error\n");
    let mut csv_x = String::from("model,W,W_pct,perblock_ms,wholefile_ms,winner\n");
    let mut e7_summary = String::new();

    for (label, _path, m) in &files {
        let data: &[u8] = m;
        let sz = data.len();
        let nblocks = sz.div_ceil(BLOCK);

        // Leaves and tree, per 4 KiB block.
        let leaves: Vec<[u8; 32]> = (0..nblocks)
            .map(|i| {
                let s = i * BLOCK;
                *blake3::hash(&data[s..(s + BLOCK).min(sz)]).as_bytes()
            })
            .collect();
        let mut levels = build_merkle(&leaves);
        let tree_depth = levels.len() - 1;
        let first_block = &data[..BLOCK.min(sz)];

        // ---- E1: block-size sweep ----
        for &bs in &SWEEP {
            let nb = sz.div_ceil(bs);
            let region = &data[..(nb * bs).min(sz)];

            let iters_v = ((1 << 30) / bs.max(1) as u64).clamp(20, 50_000);
            let verify_us = bench_ns(16, 7, iters_v, || {
                let h = blake3::hash(black_box(&region[..bs.min(sz)]));
                black_box(h);
            }) / 1e3;

            // Write-update at this block size: rehash the touched leaf, then
            // walk the path of a tree over nb blocks (depth ceil(log2(nb))).
            let depth_bs = usize::BITS - nb.max(2).next_power_of_two().leading_zeros() - 1;
            let leaf_new = [7u8; 32];
            let iters_w = ((1 << 27) / (bs.max(1) as u64 * (depth_bs as u64 + 1))).clamp(50, 20_000);
            let write_update_us = bench_ns(16, 7, iters_w, || {
                let l = *blake3::hash(black_box(&region[..bs.min(sz)])).as_bytes();
                black_box(&leaf_new);
                black_box(depth_bs);
                let mut acc = l;
                for _ in 0..depth_bs {
                    let mut h = blake3::Hasher::new();
                    h.update(&acc);
                    h.update(&leaf_new);
                    acc = *h.finalize().as_bytes();
                }
                black_box(acc);
            }) / 1e3;

            let iters_f = ((1 << 30) / sz.max(1) as u64).clamp(7, 200);
            let wholefile_us = bench_ns(3, 7, iters_f, || {
                let h = blake3::hash(black_box(data));
                black_box(h);
            }) / 1e3;

            let _ = writeln!(csv_e1, "{label},{bs},{verify_us:.4},{write_update_us:.4},{wholefile_us:.4}");

            // ---- E2: tree metadata per block size (exact arithmetic, computed) ----
            let padded = nb.max(1).next_power_of_two();
            let total_nodes = 2 * padded - 1;
            let tree_bytes = 32 * total_nodes;
            let tree_pct = tree_bytes as f64 / sz as f64 * 100.0;
            let _ = writeln!(
                csv_e2,
                "{label},{bs},{nb},{padded},{total_nodes},{tree_bytes},{tree_pct:.4},computed"
            );
        }

        // ---- E3: BLAKE3 vs SHA-256 on the SAME operations ----
        let iters_f = ((1 << 30) / sz.max(1) as u64).clamp(7, 200);
        let b3_wf = bench_ns(3, 7, iters_f, || {
            let h = blake3::hash(black_box(data));
            black_box(h);
        });
        let sha_wf = bench_ns(3, 7, iters_f, || {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(black_box(data));
            let d = h.finalize();
            black_box(d);
        });
        let _ = writeln!(csv_e3, "{label},whole_file,BLAKE3,{:.4}", b3_wf / 1e3);
        let _ = writeln!(csv_e3, "{label},whole_file,SHA-256,{:.4}", sha_wf / 1e3);

        let b3_blk = bench_ns(16, 7, 20_000, || {
            let h = blake3::hash(black_box(first_block));
            black_box(h);
        });
        let sha_blk = bench_ns(16, 7, 20_000, || {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(black_box(first_block));
            let d = h.finalize();
            black_box(d);
        });
        let _ = writeln!(csv_e3, "{label},one_block,BLAKE3,{:.4}", b3_blk / 1e3);
        let _ = writeln!(csv_e3, "{label},one_block,SHA-256,{:.4}", sha_blk / 1e3);

        // write-update vs write-update: leaf rehash + Merkle path to root,
        // identical operation for both algorithms; only the hash function differs.
        let depth = tree_depth;
        let leaf_new = [7u8; 32];
        let write_op = |leaf_fn: &dyn Fn(&[u8]) -> [u8; 32],
                        node_fn: &dyn Fn(&[u8], &[u8]) -> [u8; 32]|
         -> [u8; 32] {
            let l = leaf_fn(black_box(first_block));
            let mut acc = l;
            for _ in 0..depth {
                acc = node_fn(&acc, &leaf_new);
            }
            acc
        };
        let b3_leaf: &dyn Fn(&[u8]) -> [u8; 32] = &|b| *blake3::hash(b).as_bytes();
        let sha_leaf: &dyn Fn(&[u8]) -> [u8; 32] = &|b| {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b);
            h.finalize().into()
        };
        let b3_node: &dyn Fn(&[u8], &[u8]) -> [u8; 32] = &|a, b| {
            let mut h = blake3::Hasher::new();
            h.update(a);
            h.update(b);
            *h.finalize().as_bytes()
        };
        let sha_node: &dyn Fn(&[u8], &[u8]) -> [u8; 32] = &|a, b| {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(a);
            h.update(b);
            h.finalize().into()
        };
        let iters_w = ((1 << 27) / (BLOCK as u64 * (depth as u64 + 1))).clamp(50, 20_000);
        let b3_wu = bench_ns(16, 7, iters_w, || {
            black_box(write_op(b3_leaf, b3_node));
        });
        let sha_wu = bench_ns(16, 7, iters_w, || {
            black_box(write_op(sha_leaf, sha_node));
        });
        let _ = writeln!(csv_e3, "{label},write_update,BLAKE3,{:.4}", b3_wu / 1e3);
        let _ = writeln!(csv_e3, "{label},write_update,SHA-256,{:.4}", sha_wu / 1e3);

        // ---- E4: write-path breakdown (real timings) ----
        let leaf_only = bench_ns(16, 7, 20_000, || {
            let l = *blake3::hash(black_box(first_block)).as_bytes();
            black_box(l);
        });
        let leaf_plus_path = bench_ns(32, 15, iters_w, || {
            let leaf = *blake3::hash(black_box(first_block)).as_bytes();
            merkle_update_leaf(&mut levels, 0, leaf);
        });
        let _ = writeln!(
            csv_e4,
            "{label},{nblocks},{tree_depth},{:.4},{:.4}",
            leaf_only / 1e3,
            leaf_plus_path / 1e3
        );

        // ---- E6: max check frequency per CPU budget (derived from measured costs) ----
        let wu_us = leaf_plus_path / 1e3;
        let wf_us = b3_wf / 1e3;
        for &w in &[1u64, 10, 100] {
            let cycle_perblock = w as f64 * wu_us;
            let freq = 0.01 * 1e6 / cycle_perblock;
            let _ = writeln!(
                csv_e6,
                "{label},perblock,{w},0.01,{cycle_perblock:.4},{freq:.2},derived_from_measured_costs"
            );
            let freq10 = 0.10 * 1e6 / cycle_perblock;
            let _ = writeln!(
                csv_e6,
                "{label},perblock,{w},0.10,{cycle_perblock:.4},{freq10:.2},derived_from_measured_costs"
            );
        }
        for &budget in &[0.01f64, 0.10] {
            let freq = budget * 1e6 / wf_us;
            let _ = writeln!(
                csv_e6,
                "{label},wholefile,0,{budget:.2},{wf_us:.4},{freq:.2},derived_from_measured_costs"
            );
        }

        // ---- E7: predicted vs measured totals + break-even W* ----
        const N: u64 = 1000;
        let mut line = format!("  [{label}]");
        for &w in &[1u64, 10, 100, 1000] {
            // Predicted from the single-op measurement above.
            let predicted_ms = N as f64 * w as f64 * leaf_plus_path / 1e6;
            // Measured: run the per-cycle work end-to-end — exactly N*W
            // write-path ops (leaf rehash + Merkle path each), median of 3.
            let total_ops = N * w;
            let mut totals_ms = Vec::with_capacity(3);
            for _ in 0..3 {
                for _ in 0..64 {
                    let leaf = *blake3::hash(black_box(first_block)).as_bytes();
                    merkle_update_leaf(&mut levels, 0, leaf);
                }
                let start = Instant::now();
                for _ in 0..total_ops {
                    let leaf = *blake3::hash(black_box(first_block)).as_bytes();
                    merkle_update_leaf(&mut levels, 0, leaf);
                }
                totals_ms.push(start.elapsed().as_nanos() as f64 / 1e6);
            }
            let measured_ms = median(totals_ms);
            let err = 100.0 * (measured_ms - predicted_ms).abs() / predicted_ms;
            let _ = writeln!(
                csv_e7,
                "{label},{N},{w},{predicted_ms:.6},{measured_ms:.6},{err:.3}"
            );
            let _ = write!(line, "  W={w}: pred={predicted_ms:.3}ms meas={measured_ms:.3}ms err={err:.2}%");
        }
        println!("{line}");

        // Break-even W* = whole_file_us / write_update_us (both in microseconds;
        // whole_file measured in ns above, converted here). Per-block write-path
        // verification stays cheaper than one whole-file re-hash up to W* dirty
        // blocks per cycle.
        let w_star = (wf_us / wu_us).floor();
        let w_star_pct = w_star / nblocks as f64 * 100.0;
        let _ = writeln!(
            e7_summary,
            "  [{label}] break-even W* = floor({:.4} us / {:.4} us) = {} dirty blocks/cycle = {:.1}% of nblocks ({nblocks})",
            wf_us, wu_us, w_star, w_star_pct
        );

        // ---- E7 crossover: N=100 cycles, W as % of nblocks (measured) ----
        // mid and large only. Each cycle dirties W distinct random blocks
        // (schedule chosen up front, outside the timed region); per-block
        // policy pays leaf rehash + Merkle path per dirty block, whole-file
        // policy re-hashes the entire file once per cycle.
        if *label == "mid" || *label == "large" {
            const NC: u64 = 100;
            let mut rng: u64 = 0x9E3779B97F4A7C15 ^ (nblocks as u64);
            // Whole-file policy: N whole-file hashes, timed once per model
            // (median of 3 repetitions) — the operation is identical for every
            // W row, so it is measured once and reused across rows.
            let mut wf_ms = Vec::with_capacity(3);
            for _ in 0..3 {
                let start = Instant::now();
                for _ in 0..NC {
                    let h = blake3::hash(black_box(data));
                    black_box(h);
                }
                wf_ms.push(start.elapsed().as_nanos() as f64 / 1e6);
            }
            let wholefile_ms = median(wf_ms);
            println!(
                "  [{label}] crossover whole-file policy: {wholefile_ms:.1}ms for {NC} cycles (measured once, reused across W rows)"
            );
            for &pct in &[1u64, 5, 10, 25, 35, 50, 75, 100] {
                let w = ((nblocks as u128 * pct as u128) / 100).max(1) as usize;
                // Precompute per-cycle dirty-block schedules (untimed).
                let mut schedules: Vec<Vec<usize>> = Vec::with_capacity(NC as usize);
                for _ in 0..NC {
                    let mut idxs: Vec<usize> = (0..nblocks).collect();
                    // partial Fisher-Yates: W distinct random indices
                    for k in 0..w {
                        let j = k + (xorshift(&mut rng) as usize) % (nblocks - k);
                        idxs.swap(k, j);
                    }
                    schedules.push(idxs[..w].to_vec());
                }
                // per-block policy: N cycles x W dirty blocks, timed end-to-end
                let mut pb_ms = Vec::with_capacity(3);
                for _ in 0..3 {
                    let start = Instant::now();
                    for cycle in &schedules {
                        for &bi in cycle {
                            let s = bi * BLOCK;
                            let leaf =
                                *blake3::hash(black_box(&data[s..(s + BLOCK).min(sz)])).as_bytes();
                            merkle_update_leaf(&mut levels, bi, leaf);
                        }
                    }
                    pb_ms.push(start.elapsed().as_nanos() as f64 / 1e6);
                }
                let perblock_ms = median(pb_ms);
                let winner = if perblock_ms < wholefile_ms { "perblock" } else { "wholefile" };
                let _ = writeln!(
                    csv_x,
                    "{label},{w},{pct},{perblock_ms:.3},{wholefile_ms:.3},{winner}"
                );
                println!("  [{label}] crossover W={w} ({pct}%): perblock={perblock_ms:.1}ms wholefile={wholefile_ms:.1}ms -> {winner}");
            }
        }
    }

    let _ = writeln!(
        e7_summary,
        "\nW* = whole-file re-hash time / per-block write-update time. For W <= W* the per-block\npolicy costs less than one whole-file re-hash per cycle; above W* it costs more.\nCrossover rows in results/e7_crossover.csv are measured end-to-end (N=100 cycles,\nW distinct random dirty blocks per cycle, median of 3 repetitions)."
    );
    if !run_note.is_empty() {
        let _ = writeln!(e7_summary, "\nrun note: {run_note}");
    }

    fs::create_dir_all(results_dir()).expect("create RESULTS_DIR");
    fs::write(rpath("e1_blocksize.csv"), csv_e1).unwrap();
    fs::write(rpath("e2_metadata.csv"), csv_e2).unwrap();
    fs::write(rpath("e3_blake3_vs_sha256.csv"), csv_e3).unwrap();
    fs::write(rpath("e4_writepath.csv"), csv_e4).unwrap();
    fs::write(rpath("e6_checkfreq.csv"), csv_e6).unwrap();
    fs::write(rpath("e7_predicted_vs_measured.csv"), csv_e7).unwrap();
    fs::write(rpath("e7_crossover.csv"), csv_x).unwrap();
    fs::write(rpath("e7_summary.txt"), e7_summary).unwrap();

    println!("\nwrote {}/ : e1_blocksize.csv e2_metadata.csv e3_blake3_vs_sha256.csv e4_writepath.csv e6_checkfreq.csv e7_predicted_vs_measured.csv e7_crossover.csv e7_summary.txt", results_dir());
}

// ---- E5: model load overhead (measured) ----
// Three load modes on the real files, warm (page cache hot, median of 7) and
// cold-ish (3 trials, median; immediately before EACH trial, a 4 GiB file on
// the SSD is read sequentially in 4 MiB chunks with every page touched,
// evicting the model's pages from the page cache via LRU pressure). Modes are
// measured ONE PER TRIAL so no mode re-primes the cache for another.
// overhead_pct_vs_plain is vs the plain median under the SAME cache condition.
// Read buffers are preallocated OUTSIDE the timed regions.
fn run_e5(files: &[(&str, String, memmap2::Mmap)], run_note: &str) {
    let evict_path =
        std::env::var("MIB_EVICT_PATH").unwrap_or_else(|_| EVICT_PATH_DEFAULT.to_string());

    // Create the eviction file if missing (real bytes, on the SSD).
    if fs::metadata(&evict_path).map(|m| m.len() < EVICT_BYTES).unwrap_or(true) {
        if let Some(dir) = std::path::Path::new(&evict_path).parent() {
            fs::create_dir_all(dir).ok();
        }
        println!("creating {evict_path} ({EVICT_BYTES} B) for cache eviction...");
        let mut f = fs::File::create(&evict_path).expect("create eviction file");
        let chunk = vec![0xA5u8; 4 * 1024 * 1024];
        let mut left = EVICT_BYTES;
        while left > 0 {
            let n = left.min(chunk.len() as u64) as usize;
            use std::io::Write;
            f.write_all(&chunk[..n]).expect("write eviction file");
            left -= n as u64;
        }
        f.sync_all().ok();
    }
    assert_eq!(
        fs::metadata(&evict_path).expect("stat eviction file").len(),
        EVICT_BYTES,
        "eviction file has wrong size"
    );

    // Read the eviction file sequentially in 4 MiB chunks, touching every page.
    let evict = || {
        let f = fs::File::open(&evict_path).expect("open eviction file");
        let mut reader = std::io::BufReader::with_capacity(4 * 1024 * 1024, f);
        let mut buf = vec![0u8; 4 * 1024 * 1024];
        let mut total = 0u64;
        loop {
            use std::io::Read;
            let n = reader.read(&mut buf).expect("read eviction file");
            if n == 0 {
                break;
            }
            black_box(buf[0]);
            total += n as u64;
        }
        assert_eq!(total, EVICT_BYTES, "eviction read incomplete");
    };

    let mut csv = String::from("model,mode,cache,total_ms,overhead_pct_vs_plain\n");
    for (label, path, m) in files {
        let sz = m.len();
        // Precomputed per-block leaves (untimed) for the verify mode.
        let leaves: Vec<[u8; 32]> = (0..sz.div_ceil(BLOCK))
            .map(|i| {
                let s = i * BLOCK;
                *blake3::hash(&m[s..(s + BLOCK).min(sz)]).as_bytes()
            })
            .collect();

        // Preallocated destination buffers (reserved once, reused every trial;
        // no allocation or grow faults inside the timed regions).
        let mut bufs: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for b in bufs.iter_mut() {
            b.reserve_exact(sz);
        }

        // One measured mode per trial: each mode re-opens and reads the file
        // itself, so a cold trial measures THAT mode's I/O cost, not a cache
        // state re-primed by an earlier mode running first in the same trial.
        // (mode: 0 = plain read, 1 = whole-file hash, 2 = per-block verify)
        let mut measure = |mode: usize, cache: &str, trials: usize| -> f64 {
            let mut times_ms = Vec::with_capacity(trials);
            for _ in 0..trials {
                if cache == "cold" {
                    evict();
                }
                let buf = &mut bufs[mode];
                buf.clear();
                let start = Instant::now();
                match mode {
                    // (a) plain read into a preallocated buffer
                    0 => {
                        let n = std::io::Read::read_to_end(
                            &mut fs::File::open(path).expect("open model"),
                            buf,
                        );
                        assert_eq!(n.unwrap(), sz, "short read");
                        black_box(&buf[..]);
                    }
                    // (b) read + whole-file BLAKE3
                    1 => {
                        let n = std::io::Read::read_to_end(
                            &mut fs::File::open(path).expect("open model"),
                            buf,
                        );
                        assert_eq!(n.unwrap(), sz, "short read");
                        let h = blake3::hash(black_box(&buf[..]));
                        black_box(h);
                    }
                    // (c) read + per-block verify against precomputed leaves
                    _ => {
                        let n = std::io::Read::read_to_end(
                            &mut fs::File::open(path).expect("open model"),
                            buf,
                        );
                        assert_eq!(n.unwrap(), sz, "short read");
                        let mut ok = true;
                        for (i, leaf) in leaves.iter().enumerate() {
                            let s = i * BLOCK;
                            let hh = *blake3::hash(&buf[s..(s + BLOCK).min(sz)]).as_bytes();
                            if hh != *leaf {
                                ok = false;
                            }
                        }
                        black_box(ok);
                    }
                }
                times_ms.push(start.elapsed().as_nanos() as f64 / 1e6);
            }
            median(times_ms)
        };

        let p_warm = measure(0, "warm", 7);
        let w_warm = measure(1, "warm", 7);
        let v_warm = measure(2, "warm", 7);
        let p_cold = measure(0, "cold", 3);
        let w_cold = measure(1, "cold", 3);
        let v_cold = measure(2, "cold", 3);
        for (cache, (p, w, v)) in [("warm", (p_warm, w_warm, v_warm)), ("cold", (p_cold, w_cold, v_cold))] {
            let _ = writeln!(csv, "{label},plain,{cache},{p:.3},0.000");
            let _ = writeln!(csv, "{label},wholefile_hash,{cache},{w:.3},{:.2}", (w - p) / p * 100.0);
            let _ = writeln!(csv, "{label},perblock_verify,{cache},{v:.3},{:.2}", (v - p) / p * 100.0);
        }
        println!(
            "[{label}] E5 warm: plain={p_warm:.2}ms whole={w_warm:.2}ms verify={v_warm:.2}ms | cold: plain={p_cold:.2}ms whole={w_cold:.2}ms verify={v_cold:.2}ms"
        );
    }

    fs::write(rpath("e5_load.csv"), csv).unwrap();
    let mut note = String::new();
    let _ = writeln!(note, "E5 method: fs::File::open + read_to_end of the real model file into a");
    let _ = writeln!(note, "preallocated buffer (cleared between trials, capacity retained);");
    let _ = writeln!(note, "whole = + one whole-file BLAKE3 over the bytes read; perblock = + re-hash");
    let _ = writeln!(note, "every 4 KiB block and compare to precomputed leaves.");
    let _ = writeln!(note, "warm = median of 7 consecutive trials, page cache hot.");
    let _ = writeln!(note, "cold = median of 3 trials. Immediately before EACH trial the eviction file");
    let _ = writeln!(note, "is read sequentially in 4 MiB chunks with every page touched, evicting the");
    let _ = writeln!(note, "model's pages from the page cache via LRU pressure (no fadvise). The eviction");
    let _ = writeln!(note, "file is on the real SSD filesystem (4 GiB, exceeding this machine's ~3.3 GiB");
    let _ = writeln!(note, "page cache; /tmp is tmpfs, i.e. RAM, and would NOT evict anything) at {evict_path}.");
    let _ = writeln!(note, "Modes are measured one per trial (each mode re-opens and reads the file itself),");
    let _ = writeln!(note, "so no mode re-primes the cache for another.");
    let _ = writeln!(note, "overhead_pct_vs_plain is vs the plain-read median under the same cache condition.");
    if !run_note.is_empty() {
        let _ = writeln!(note, "run note: {run_note}");
    }
    fs::write(rpath("e5_method.txt"), note).unwrap();
    println!("wrote {0}/e5_load.csv {0}/e5_method.txt", results_dir());
}
