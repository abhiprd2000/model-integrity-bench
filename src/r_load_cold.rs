// True cold-cache model load (replaces the eviction-by-reading method of E5,
// which did not evict: the ResNet-50 "cold" read there took 41 ms = 2.5 GB/s).
//
//   sudo ./target/release/r_load_cold      -> results/r_load_cold.csv
//
// Before EVERY cold trial: `sync`, then write 3 to /proc/sys/vm/drop_caches
// (page cache + dentries/inodes). Needs root; the binary refuses to run
// otherwise and never falls back to an eviction heuristic.
// Modes (same definitions as E5): plain = open + read_to_end into a
// preallocated buffer; wholefile_hash = plain + one BLAKE3 over the bytes;
// perblock_verify = plain + hash each 4 KiB block against precomputed leaves
// (leaves computed untimed). One mode per trial. Warm: 1 untimed warm-up then
// median of 7. Cold: median of 5. effective_MBps = file bytes / that mode's
// median time, written so an impossible "cold" number is visible in the data.
use mib::*;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::os::unix::fs::chown;
use std::time::Instant;

const MODELS: [(&str, &str); 3] = [
    ("tiny", "models/tiny.tflite"),
    ("mid", "models/mid.tflite"),
    ("large", "models/large.onnx"),
];

fn drop_caches() {
    let _ = std::process::Command::new("sync").status();
    fs::write("/proc/sys/vm/drop_caches", b"3")
        .expect("cannot write /proc/sys/vm/drop_caches (run as root); refusing to fake a cold cache");
}

fn main() {
    // Fail fast if we cannot drop caches.
    drop_caches();
    let mut csv = String::from(
        "model,file_bytes,mode,cache,trials,median_ms,min_ms,max_ms,overhead_pct_vs_plain,effective_MBps\n",
    );
    for (label, path) in MODELS {
        let sz = fs::metadata(path).unwrap().len() as usize;
        let leaves = leaves_of(&fs::read(path).unwrap());
        let mut bufs: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for b in bufs.iter_mut() {
            b.reserve_exact(sz);
        }
        let mut run = |mode: usize, cache: &str, trials: usize| -> (f64, f64, f64) {
            let mut t = Vec::with_capacity(trials);
            let n = if cache == "warm" { trials + 1 } else { trials };
            for k in 0..n {
                if cache == "cold" {
                    drop_caches();
                }
                let buf = &mut bufs[mode];
                buf.clear();
                let start = Instant::now();
                let got = std::io::Read::read_to_end(&mut fs::File::open(path).unwrap(), buf).unwrap();
                assert_eq!(got, sz);
                match mode {
                    0 => {
                        black_box(&buf[..]);
                    }
                    1 => {
                        black_box(blake3::hash(black_box(&buf[..])));
                    }
                    _ => {
                        let mut ok = true;
                        for (i, leaf) in leaves.iter().enumerate() {
                            let (s, e) = block_range(i, sz);
                            ok &= leaf_hash(&buf[s..e]) == *leaf;
                        }
                        assert!(ok);
                    }
                }
                let ms = start.elapsed().as_nanos() as f64 / 1e6;
                if cache == "cold" || k > 0 {
                    t.push(ms); // warm: first iteration is the untimed warm-up
                }
            }
            let (mn, mx) = (t.iter().cloned().fold(f64::MAX, f64::min), t.iter().cloned().fold(0.0, f64::max));
            (median(t), mn, mx)
        };
        for (cache, trials) in [("warm", 7usize), ("cold", 5usize)] {
            let mut plain = 0.0;
            for (mode, name) in ["plain", "wholefile_hash", "perblock_verify"].iter().enumerate() {
                let (med, mn, mx) = run(mode, cache, trials);
                if mode == 0 {
                    plain = med;
                }
                let mbps = sz as f64 / (med / 1e3) / 1e6;
                let _ = writeln!(
                    csv,
                    "{label},{sz},{name},{cache},{trials},{med:.3},{mn:.3},{mx:.3},{:.2},{mbps:.1}",
                    (med - plain) / plain * 100.0
                );
                println!("{label:<6} {name:<16} {cache:<5} median {med:>9.3} ms  [{mn:.1}..{mx:.1}]  {mbps:.0} MB/s");
            }
        }
    }
    fs::write("results/r_load_cold.csv", csv).unwrap();
    // Hand the output back to the invoking user when run under sudo.
    if let (Ok(u), Ok(g)) = (std::env::var("SUDO_UID"), std::env::var("SUDO_GID")) {
        let _ = chown("results/r_load_cold.csv", u.parse().ok(), g.parse().ok());
    }
    println!("wrote results/r_load_cold.csv");
}
