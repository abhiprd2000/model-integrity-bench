// Demo 4: cost of one verified read vs. re-hashing the whole file, per model.
//   cargo run --release --example demo_cost -- "Label=path" ...
use mib::*;
use std::hint::black_box;

fn cpu_name() -> String {
    let s = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let m = s.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(": ").nth(1)).unwrap_or("unknown");
    let m = m.replace("(R)", "").replace("(TM)", "");
    m.split(" CPU").next().unwrap().split_whitespace().collect::<Vec<_>>().join(" ")
}

fn main() {
    println!("{:<24} {:<10} {:<8} {:<19} {}", "Model", "Size", "Blocks", "1 read (verified)", "Whole file");
    for arg in std::env::args().skip(1) {
        let (label, path) = arg.split_once('=').expect("Label=path");
        let data = std::fs::read(path).expect("read model");
        let nb = nblocks(data.len());
        let levels = build_tree(&leaves_of(&data));
        let root = levels.last().unwrap()[0];
        let b = nb / 2;
        let (s, e) = block_range(b, data.len());
        let block = &data[s..e];

        let read_ns = bench_ns(16, 101, 200, || {
            let ok = verify_path(black_box(&levels), b, black_box(block), &root);
            assert!(black_box(ok));
        });
        let whole_ns = bench_ns(3, 11, 1, || {
            black_box(blake3::hash(black_box(&data)));
        });
        println!(
            "{:<24} {:<10} {:<8} {:<19} {:.2} ms",
            label,
            format!("{:.1} MiB", data.len() as f64 / 1048576.0),
            nb,
            format!("{:.2} µs", read_ns / 1e3),
            whole_ns / 1e6
        );
    }
    println!("Measured on: {}", cpu_name());
}
