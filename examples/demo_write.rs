// Demo 3: write path. Uses only mib's public library functions.
//   cargo run --release --example demo_write -- demo/work/clean.onnx
use ed25519_dalek::SigningKey;
use mib::*;
use std::hint::black_box;

fn main() {
    let path = std::env::args().nth(1).expect("usage: demo_write <model>");
    let mut data = std::fs::read(&path).expect("read model");
    let nb = nblocks(data.len());

    // 1. Build the tree.
    let mut levels = build_tree(&leaves_of(&data));
    let old_root = *levels.last().unwrap().first().unwrap();
    println!("Tree: {nb} blocks, depth {}", levels.len() - 1);

    // 2. Honest write: overwrite one float32 inside a middle block, update that leaf + its path.
    let b = nb / 2;
    let (s, e) = block_range(b, data.len());
    data[s + 100..s + 104].copy_from_slice(&0.5f32.to_le_bytes());
    let mut ns = bench_ns(16, 101, 200, || {
        let leaf = leaf_hash(black_box(&data[s..e]));
        black_box(update_leaf(black_box(&mut levels), b, leaf));
    });
    let new_root = levels.last().unwrap()[0];
    assert_ne!(old_root, new_root, "honest write must change the root");
    println!("Honest write to block {b}: tree updated in {:.2} µs (median of 101)", ns / 1e3);

    // 5. Re-sign the new root (reported on its own).
    let sk = SigningKey::from_bytes(&random_bytes());
    ns = bench_ns(500, 101, 200, || {
        black_box(RootRecord::sign(black_box(&sk), new_root, 2, data.len() as u64, "clean.onnx"));
    });
    println!("Re-signing the root: {:.2} µs", ns / 1e3);

    // 6. Whole-file re-hash for comparison.
    ns = bench_ns(3, 11, 1, || {
        black_box(blake3::hash(black_box(&data)));
    });
    println!("Whole-file re-hash instead: {:.2} ms", ns / 1e6);

    // 3. Verify every block against the new root.
    let leaves = &levels[0][..nb];
    let ok = first_bad_block(&data, leaves).is_none() && root_of_leaves(&leaves_of(&data)) == new_root;
    println!("Verify after honest write ... {}", if ok { "OK" } else { "FAILED" });
    assert!(ok);

    // 4. Tamper one byte in a different block, without updating the tree.
    let t = b + 1000;
    let ts = block_range(t, data.len()).0;
    data[ts + 7] ^= 0x01;
    match first_bad_block(&data, leaves) {
        Some(i) => println!("Tamper 1 byte in block {t} (no update) ... FAILED at block {i}"),
        None => println!("Tamper 1 byte in block {t} (no update) ... NOT DETECTED"),
    }
}
