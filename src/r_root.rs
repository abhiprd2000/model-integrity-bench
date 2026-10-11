// Part B: signed root + rollback protection (ed25519-dalek).
//
// Root record = (merkle_root, version, model_id) signed by an "update service"
// key, plus file_len (see src/mib.rs: it binds the file length so tree padding
// cannot be used to extend a file). The device holds the service public key
// and a stored minimum version.
//
// Writes results/r_root.csv: timings (measured here) and pass/fail tests.
use ed25519_dalek::SigningKey;
use mib::*;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;

const ID: &str = "mobilenet_v2";

fn main() {
    let mut csv = String::from("kind,name,model,value,unit,expected,observed,pass\n");
    let mut failures = 0;
    let mut test = |csv: &mut String, name: &str, expected: &str, observed: String| {
        let pass = expected == observed;
        if !pass {
            failures += 1;
        }
        let _ = writeln!(csv, "test,{name},mid,,,{expected},{observed},{pass}");
        println!("{:<44} expected={expected:<22} observed={observed:<22} {}", name, if pass { "PASS" } else { "FAIL" });
    };

    let sk = SigningKey::from_bytes(&random_bytes());
    let pk = sk.verifying_key();
    let attacker = SigningKey::from_bytes(&random_bytes());

    // ---------------- timings
    let root: Hash = leaf_hash(b"timing root");
    let model_len = 13_978_000u64;
    let sign_ns = bench_ns(500, 7, 2000, || {
        black_box(RootRecord::sign(black_box(&sk), root, 5, model_len, ID));
    });
    let rec = RootRecord::sign(&sk, root, 5, model_len, ID);
    let verify_ns = bench_ns(500, 7, 2000, || {
        black_box(black_box(&rec).verify(black_box(&pk)));
    });
    let _ = writeln!(csv, "timing,sign,,{:.3},us,,,", sign_ns / 1e3);
    let _ = writeln!(csv, "timing,verify_strict,,{:.3},us,,,", verify_ns / 1e3);
    println!("sign   median {:.2} us\nverify median {:.2} us", sign_ns / 1e3, verify_ns / 1e3);

    // Device open: signature + id + floor + length + leaf file -> signed root.
    for (label, path) in [("mid", "models/mid.tflite"), ("large", "models/large.onnx")] {
        let data = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let sc = seal(&data, &sk, 5, ID);
        let open_ns = bench_ns(2, 7, 5, || {
            black_box(open_verified(black_box(&sc), data.len() as u64, ID, &pk, 5)).unwrap();
        });
        let _ = writeln!(csv, "timing,open_verified,{label},{:.3},us,,,", open_ns / 1e3);
        println!("open_verified [{label}] median {:.1} us ({} leaves)", open_ns / 1e3, sc.leaves.len());
    }

    // ---------------- tests (mid)
    let data = fs::read("models/mid.tflite").expect("models/mid.tflite");
    let len = data.len() as u64;
    let cur = seal(&data, &sk, 5, ID);
    let min_v = 5;
    let res = |r: Result<(), OpenError>| match r {
        Ok(()) => "accepted".to_string(),
        Err(e) => format!("rejected:{e}"),
    };
    let reload = |sc: &Sidecar| Sidecar::from_bytes(&sc.to_bytes()).unwrap(); // via serialized form

    // 1. valid record accepted, and every block then verifies
    let sc = reload(&cur);
    test(&mut csv, "valid_record_accepted", "accepted", res(open_verified(&sc, len, ID, &pk, min_v)));
    test(
        &mut csv,
        "valid_record_all_blocks_verify",
        "no_bad_block",
        match first_bad_block(&data, &sc.leaves) {
            None => "no_bad_block".into(),
            Some(i) => format!("bad_block:{i}"),
        },
    );

    // 2. tampered leaf file rejected at open
    let mut sc = reload(&cur);
    sc.leaves[1234][7] ^= 0x01;
    test(&mut csv, "tampered_leaf_file_rejected_at_open", "rejected:RootMismatch", res(open_verified(&sc, len, ID, &pk, min_v)));
    // 2b. attacker edits a model block AND rewrites its leaf to match
    let mut sc = reload(&cur);
    let mut evil = data.clone();
    evil[4096 * 77 + 5] ^= 0xFF;
    sc.leaves[77] = leaf_hash(&evil[4096 * 77..4096 * 78]);
    test(&mut csv, "edited_block_with_rewritten_leaf_rejected", "rejected:RootMismatch", res(open_verified(&sc, evil.len() as u64, ID, &pk, min_v)));
    // 2c. leaf file truncated
    let mut sc = reload(&cur);
    sc.leaves.pop();
    test(&mut csv, "truncated_leaf_file_rejected", "rejected:LengthMismatch", res(open_verified(&sc, len, ID, &pk, min_v)));

    // 3. rollback: validly signed OLD record (version-1) is refused.
    // 3a. a genuine older release: different bytes, signed at version 4.
    let mut old_data = data.clone();
    old_data[4096 * 9 + 100] ^= 0x42;
    let old_sc = seal(&old_data, &sk, 4, ID);
    test(
        &mut csv,
        "old_release_snapshot_v4_rejected_as_rollback",
        "rejected:Rollback",
        res(open_verified(&reload(&old_sc), old_data.len() as u64, ID, &pk, min_v)),
    );
    // sanity: that old snapshot is otherwise perfectly valid (accepted at floor 4)
    test(
        &mut csv,
        "old_release_snapshot_v4_valid_at_floor_4",
        "accepted",
        res(open_verified(&reload(&old_sc), old_data.len() as u64, ID, &pk, 4)),
    );
    // 3b. literally (version-1): current root re-signed at version 4.
    let mut sc = reload(&cur);
    sc.rec = RootRecord::sign(&sk, cur.rec.root, 4, len, ID);
    test(&mut csv, "record_version_minus_1_rejected_as_rollback", "rejected:Rollback", res(open_verified(&sc, len, ID, &pk, min_v)));
    // 3c. new version accepted; device then raises its floor, and v5 is refused.
    let next = seal(&data, &sk, 6, ID);
    test(&mut csv, "next_version_v6_accepted", "accepted", res(open_verified(&reload(&next), len, ID, &pk, min_v)));
    test(&mut csv, "v5_rejected_after_floor_raised_to_6", "rejected:Rollback", res(open_verified(&reload(&cur), len, ID, &pk, 6)));

    // 4. extra attacks
    let mut sc = reload(&cur);
    sc.rec.sig[10] ^= 0x01;
    test(&mut csv, "flipped_signature_bit_rejected", "rejected:BadSignature", res(open_verified(&sc, len, ID, &pk, min_v)));
    let forged = seal(&data, &attacker, 99, ID);
    test(&mut csv, "record_signed_by_other_key_rejected", "rejected:BadSignature", res(open_verified(&forged, len, ID, &pk, min_v)));
    let mut sc = reload(&cur);
    sc.rec.version = 50; // bump version without re-signing
    test(&mut csv, "version_edited_without_resign_rejected", "rejected:BadSignature", res(open_verified(&sc, len, ID, &pk, min_v)));
    test(&mut csv, "wrong_model_id_rejected", "rejected:WrongModel", res(open_verified(&reload(&cur), len, "other_model", &pk, min_v)));
    let mut ext = data.clone();
    ext.extend_from_slice(&data[..BLOCK]);
    test(&mut csv, "file_extended_by_one_block_rejected", "rejected:LengthMismatch", res(open_verified(&reload(&cur), ext.len() as u64, ID, &pk, min_v)));
    // model-file tamper: open succeeds (leaf file is intact) but the block read fails
    let mut tam = data.clone();
    let off = 4096 * 2000 + 321;
    tam[off] ^= 0x01;
    let sc = reload(&cur);
    test(&mut csv, "tampered_model_file_open_ok", "accepted", res(open_verified(&sc, tam.len() as u64, ID, &pk, min_v)));
    test(
        &mut csv,
        "tampered_model_file_block_read_rejected",
        "bad_block:2000",
        match first_bad_block(&tam, &sc.leaves) {
            None => "no_bad_block".into(),
            Some(i) => format!("bad_block:{i}"),
        },
    );

    fs::create_dir_all("results").unwrap();
    fs::write("results/r_root.csv", csv).unwrap();
    println!("wrote results/r_root.csv ({failures} failed tests)");
    if failures > 0 {
        std::process::exit(1);
    }
}
