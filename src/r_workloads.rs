// Part A: batched updates + realistic workloads.
//
//   r_workloads sweep  -> results/r_workloads.csv, results/r_workloads_crossover.csv
//   r_workloads head   -> results/r_workloads_head.csv  (real ResNet-50 FC update)
//
// Timing rules: release build, real model files, warm-up, median of 3
// end-to-end runs (100 cycles each) for the sweeps, median of 7 single
// operations for the head update. Every number in the CSVs comes from the
// timing code below (or, for the crossover file, from linear interpolation
// between two measured rows, labelled as such).
use mib::*;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

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

const NC: usize = 100; // cycles
const PCTS: [usize; 8] = [1, 5, 10, 25, 35, 50, 75, 100];

fn mmap(path: &str) -> memmap2::Mmap {
    let f = fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e} (no synthetic substitute)"));
    let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
    for p in m.chunks(4096) {
        black_box(p[0]); // fault every page in once, outside timing
    }
    m
}

// End-to-end: one untimed warm-up run, then median of 3 timed runs (ms).
fn e2e_ms(mut run: impl FnMut()) -> f64 {
    run();
    let mut t = Vec::with_capacity(3);
    for _ in 0..3 {
        let s = Instant::now();
        run();
        t.push(s.elapsed().as_nanos() as f64 / 1e6);
    }
    median(t)
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("sweep") => sweep(),
        Some("head") => head(),
        _ => {
            eprintln!("usage: r_workloads sweep|head");
            std::process::exit(1);
        }
    }
}

fn schedule(pattern: &str, nb: usize, w: usize, rng: &mut u64) -> Vec<usize> {
    if pattern == "random" {
        // W distinct random blocks (partial Fisher-Yates), returned sorted.
        let mut idxs: Vec<usize> = (0..nb).collect();
        for k in 0..w {
            let j = k + (xorshift(rng) as usize) % (nb - k);
            idxs.swap(k, j);
        }
        let mut v = idxs[..w].to_vec();
        v.sort_unstable();
        v
    } else {
        // one contiguous run of W blocks at a random offset
        let start = (xorshift(rng) as usize) % (nb - w + 1);
        (start..start + w).collect()
    }
}

fn sweep() {
    fs::create_dir_all(results_dir()).expect("create RESULTS_DIR");
    let mut csv = String::from(
        "model,pattern,W_pct,W,nblocks,cycles,unbatched_ms,batched_ms,wholefile_ms,\
         leaf_hashes_per_cycle,unbatched_node_hashes_per_cycle,batched_node_hashes_per_cycle,\
         roots_match_rebuild,winner_unbatched,winner_batched\n",
    );
    // rows kept for the crossover computation: (model, pattern, pct, un, ba, wf)
    let mut rows: Vec<(String, String, f64, f64, f64, f64)> = Vec::new();

    for (label, path) in [("mid", model_path("models/mid.tflite")), ("large", model_path("models/large.onnx"))] {
        let m = mmap(&path);
        let data: &[u8] = &m;
        let nb = nblocks(data.len());
        let base_leaves = leaves_of(data);
        let base_levels = build_tree(&base_leaves);
        let mut lv_un = base_levels.clone();
        let mut lv_ba = base_levels.clone();
        let mut rng: u64 = 0x9E3779B97F4A7C15 ^ (nb as u64);

        for pattern in ["random", "contiguous"] {
            for &pct in &PCTS {
                let w = ((nb * pct) / 100).max(1);
                let sched: Vec<Vec<usize>> =
                    (0..NC).map(|_| schedule(pattern, nb, w, &mut rng)).collect();

                // ---- correctness (untimed): modify the dirty blocks of cycle 0
                // in a private copy, apply both updates, compare to a rebuild.
                let mut copy = data.to_vec();
                for &i in &sched[0] {
                    copy[i * BLOCK] ^= 0x5A;
                }
                let mut a = base_levels.clone();
                let mut b = base_levels.clone();
                let nodes_un = update_unbatched(&mut a, &sched[0], &copy);
                let nodes_ba = update_batched(&mut b, &sched[0], &copy);
                let rebuilt = build_tree(&leaves_of(&copy));
                let ok = a == rebuilt && b == rebuilt;
                assert!(ok, "update does not reproduce the rebuilt tree ({label} {pattern} {pct}%)");
                drop((copy, a, b, rebuilt));

                // ---- timed: N cycles end-to-end, data unchanged (hash cost
                // is content-independent; correctness is checked above).
                let un_ms = e2e_ms(|| {
                    for cyc in &sched {
                        black_box(update_unbatched(&mut lv_un, cyc, black_box(data)));
                    }
                });
                let ba_ms = e2e_ms(|| {
                    for cyc in &sched {
                        black_box(update_batched(&mut lv_ba, cyc, black_box(data)));
                    }
                });
                let wf_ms = e2e_ms(|| {
                    for _ in 0..NC {
                        black_box(blake3::hash(black_box(data)));
                    }
                });
                let wu = if un_ms < wf_ms { "perblock" } else { "wholefile" };
                let wb = if ba_ms < wf_ms { "perblock" } else { "wholefile" };
                let _ = writeln!(
                    csv,
                    "{label},{pattern},{pct},{w},{nb},{NC},{un_ms:.3},{ba_ms:.3},{wf_ms:.3},{w},{nodes_un},{nodes_ba},{ok},{wu},{wb}"
                );
                println!("[{label}] {pattern:<10} {pct:>3}% W={w:<6} unbatched={un_ms:>10.1}ms batched={ba_ms:>10.1}ms wholefile={wf_ms:>9.1}ms ({nodes_un} vs {nodes_ba} node hashes/cycle)");
                rows.push((label.into(), pattern.into(), pct as f64, un_ms, ba_ms, wf_ms));
            }
        }
    }

    // ---- crossover: interpolate where (per-block - whole-file) changes sign.
    let mut xcsv = String::from(
        "model,pattern,policy,crossover_pct,bracket_lo_pct,bracket_hi_pct,basis\n",
    );
    for label in ["mid", "large"] {
        for pattern in ["random", "contiguous"] {
            for policy in ["unbatched", "batched"] {
                let pts: Vec<(f64, f64)> = rows
                    .iter()
                    .filter(|r| r.0 == label && r.1 == pattern)
                    .map(|r| (r.2, (if policy == "unbatched" { r.3 } else { r.4 }) - r.5))
                    .collect();
                let mut line = format!("{label},{pattern},{policy},");
                if pts[0].1 >= 0.0 {
                    let _ = write!(line, "<{},,{},per-block already loses at lowest measured W", pts[0].0, pts[0].0);
                } else if let Some(k) = (1..pts.len()).find(|&k| pts[k].1 >= 0.0) {
                    let (x0, d0) = pts[k - 1];
                    let (x1, d1) = pts[k];
                    let x = x0 + (x1 - x0) * (-d0) / (d1 - d0);
                    let _ = write!(line, "{x:.1},{x0},{x1},linear interpolation between measured rows");
                } else {
                    let _ = write!(line, "none<=100,{},{},per-block wins at every measured W up to 100%", pts[pts.len() - 1].0, 100);
                }
                println!("crossover {line}");
                let _ = writeln!(xcsv, "{line}");
            }
        }
    }

    fs::write(rpath("r_workloads.csv"), csv).unwrap();
    fs::write(rpath("r_workloads_crossover.csv"), xcsv).unwrap();
    println!("wrote {} {}", rpath("r_workloads.csv"), rpath("r_workloads_crossover.csv"));
}

// ---- real update: ResNet-50 classifier head ----
fn head() {
    if cfg!(target_arch = "aarch64") {
        eprintln!("r_workloads head is skipped on aarch64 builds: it needs the python-generated perturbed model copy.");
        return;
    }
    fs::create_dir_all(results_dir()).expect("create RESULTS_DIR");
    let meta_txt = fs::read_to_string(".cache/head/head_meta.txt")
        .expect("run scripts/make_head_update.py first");
    let get = |k: &str| -> String {
        meta_txt
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .unwrap_or_else(|| panic!("missing {k}"))
            .to_string()
    };
    let (tensor, off, blen) = (
        get("tensor_name"),
        get("byte_offset").parse::<usize>().unwrap(),
        get("byte_len").parse::<usize>().unwrap(),
    );
    let old_m = mmap(&model_path("models/large.onnx"));
    let new_m = mmap(&get("perturbed_path"));
    let (old, new): (&[u8], &[u8]) = (&old_m, &new_m);
    assert_eq!(old.len(), new.len());
    assert_eq!(old.len(), get("file_len").parse::<usize>().unwrap());
    // the byte range must hold exactly the perturbed region
    assert_eq!(&old[..off], &new[..off]);
    assert_eq!(&old[off + blen..], &new[off + blen..]);
    assert_ne!(&old[off..off + blen], &new[off..off + blen]);

    let nb = nblocks(old.len());
    let range_blocks = (off + blen - 1) / BLOCK - off / BLOCK + 1;
    // dirty set = blocks whose bytes actually differ (untimed diff)
    let dirty: Vec<usize> = (0..nb)
        .filter(|&i| {
            let (s, e) = block_range(i, old.len());
            old[s..e] != new[s..e]
        })
        .collect();
    assert!(dirty.iter().all(|&i| i >= off / BLOCK && i <= (off + blen - 1) / BLOCK));

    let old_levels = build_tree(&leaves_of(old));
    let old_root = *old_levels.last().unwrap().first().unwrap();

    // Single-operation timings: median of 7, 2 warm-up, fresh tree clone
    // before every call (clone is outside the timed region).
    let time_update = |batched: bool| -> f64 {
        let mut t = Vec::new();
        for k in 0..9 {
            let mut lv = old_levels.clone();
            let s = Instant::now();
            let n = if batched {
                update_batched(&mut lv, &dirty, black_box(new))
            } else {
                update_unbatched(&mut lv, &dirty, black_box(new))
            };
            let el = s.elapsed().as_nanos() as f64 / 1e6;
            black_box((n, &lv));
            if k >= 2 {
                t.push(el);
            }
        }
        median(t)
    };
    let unb_ms = time_update(false);
    let bat_ms = time_update(true);
    let mut t = Vec::new();
    for k in 0..9 {
        let s = Instant::now();
        black_box(blake3::hash(black_box(new)));
        if k >= 2 {
            t.push(s.elapsed().as_nanos() as f64 / 1e6);
        }
    }
    let wf_ms = median(t);

    // Verification (untimed): batched root == unbatched root == root rebuilt
    // from scratch over the new file, and every block of the new file hashes
    // to its updated leaf.
    let mut lv_b = old_levels.clone();
    let mut lv_u = old_levels.clone();
    let nodes_b = update_batched(&mut lv_b, &dirty, new);
    let nodes_u = update_unbatched(&mut lv_u, &dirty, new);
    let rebuilt = build_tree(&leaves_of(new));
    let new_root = *rebuilt.last().unwrap().first().unwrap();
    let roots_match = lv_b.last() == rebuilt.last() && lv_u.last() == rebuilt.last();
    let trees_match = lv_b == rebuilt && lv_u == rebuilt;
    let audit_ok = first_bad_block(new, &lv_b[0][..nb]).is_none();
    // and the OLD file must NOT verify against the new leaves
    let old_rejected = first_bad_block(old, &lv_b[0][..nb]).is_some();
    assert!(roots_match && trees_match && audit_ok && old_rejected && old_root != new_root);

    let csv = format!(
        "model,tensor_name,byte_offset,byte_len,file_len,nblocks,range_blocks,dirty_blocks,dirty_pct_of_file,\
unbatched_ms,batched_ms,wholefile_ms,unbatched_node_hashes,batched_node_hashes,\
old_root,new_root,rebuilt_root_matches_updated,new_file_audit_ok,old_file_rejected_by_new_leaves\n\
large,{tensor},{off},{blen},{},{nb},{range_blocks},{},{:.4},{unb_ms:.4},{bat_ms:.4},{wf_ms:.4},{nodes_u},{nodes_b},{},{},{roots_match},{audit_ok},{old_rejected}\n",
        old.len(),
        dirty.len(),
        dirty.len() as f64 * BLOCK as f64 / old.len() as f64 * 100.0,
        hex(&old_root),
        hex(&new_root),
    );
    fs::write(rpath("r_workloads_head.csv"), &csv).unwrap();
    print!("{csv}");
}
