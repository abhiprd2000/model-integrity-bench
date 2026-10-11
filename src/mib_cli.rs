// mib: sealing + verified-read CLI.
//
//   mib keygen <seed-file> <pub-file>
//   mib seal <model> --key <seed-file> --version N [--model-id ID] [--out SIDECAR]
//   mib verify-read <model> --pubkey <pub-file> [--sidecar SIDECAR]
//                   [--min-version N] [--model-id ID]
//
// verify-read opens the sidecar (signature, model id, rollback floor, length,
// leaf file -> signed root), then streams the model to stdout 4 KiB block at a
// time. A block is written only after it hashes to its leaf. On any mismatch it
// writes nothing further, prints the reason to stderr and exits non-zero (2 =
// open-time refusal, 3 = block mismatch, 1 = usage/IO).
use ed25519_dalek::{SigningKey, VerifyingKey};
use mib::*;
use std::io::{Read, Write};
use std::process::exit;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn die(code: i32, msg: &str) -> ! {
    eprintln!("mib: {msg}");
    exit(code)
}

fn default_id(path: &str) -> String {
    std::path::Path::new(path).file_name().unwrap().to_string_lossy().into_owned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("keygen") => {
            let (seed_p, pub_p) = (&args[2], &args[3]);
            let seed: [u8; 32] = random_bytes();
            let sk = SigningKey::from_bytes(&seed);
            std::fs::write(seed_p, seed).unwrap();
            std::fs::write(pub_p, sk.verifying_key().to_bytes()).unwrap();
        }
        Some("seal") => {
            let model = &args[2];
            let key_p = arg(&args, "--key").unwrap_or_else(|| die(1, "--key required"));
            let version: u64 = arg(&args, "--version").unwrap_or_else(|| die(1, "--version required")).parse().unwrap();
            let id = arg(&args, "--model-id").unwrap_or_else(|| default_id(model));
            let out = arg(&args, "--out").unwrap_or_else(|| format!("{model}.mib"));
            let seed: [u8; 32] = std::fs::read(key_p).unwrap().try_into().unwrap();
            let data = std::fs::read(model).unwrap_or_else(|e| die(1, &format!("{model}: {e}")));
            let sc = seal(&data, &SigningKey::from_bytes(&seed), version, &id);
            std::fs::write(&out, sc.to_bytes()).unwrap();
            eprintln!("sealed {model}: {} blocks, root {}, version {version}, id {id} -> {out}", sc.leaves.len(), hex(&sc.rec.root));
        }
        Some("verify-read") => verify_read(&args),
        _ => die(1, "usage: mib keygen|seal|verify-read ..."),
    }
}

fn verify_read(args: &[String]) {
    let model = &args[2];
    let pub_p = arg(args, "--pubkey").unwrap_or_else(|| die(1, "--pubkey required"));
    let sc_p = arg(args, "--sidecar").unwrap_or_else(|| format!("{model}.mib"));
    let min_version: u64 = arg(args, "--min-version").map(|s| s.parse().unwrap()).unwrap_or(0);
    let id = arg(args, "--model-id").unwrap_or_else(|| default_id(model));

    let pk_bytes: [u8; 32] = std::fs::read(&pub_p).unwrap_or_else(|e| die(1, &format!("{pub_p}: {e}"))).try_into().unwrap_or_else(|_| die(1, "bad pubkey length"));
    let pk = VerifyingKey::from_bytes(&pk_bytes).unwrap_or_else(|_| die(1, "bad pubkey"));
    let sc_bytes = std::fs::read(&sc_p).unwrap_or_else(|e| die(1, &format!("{sc_p}: {e}")));
    let sc = Sidecar::from_bytes(&sc_bytes).unwrap_or_else(|| die(2, "open refused: Malformed sidecar"));
    let mut f = std::fs::File::open(model).unwrap_or_else(|e| die(1, &format!("{model}: {e}")));
    let len = f.metadata().unwrap().len();
    if let Err(e) = open_verified(&sc, len, &id, &pk, min_version) {
        die(2, &format!("open refused: {e}"));
    }

    // Stream in 1 MiB reads, verify and emit 4 KiB blocks one at a time.
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut buf = vec![0u8; 1 << 20];
    let mut block_idx = 0usize;
    let mut remaining = len as usize;
    while remaining > 0 {
        let want = buf.len().min(remaining);
        f.read_exact(&mut buf[..want]).unwrap_or_else(|e| die(1, &format!("read: {e}")));
        for chunk in buf[..want].chunks(BLOCK) {
            if leaf_hash(chunk) != sc.leaves[block_idx] {
                let _ = out.flush(); // everything written so far was verified
                die(3, &format!("block {block_idx} (offset {}) failed verification", block_idx * BLOCK));
            }
            out.write_all(chunk).unwrap_or_else(|e| die(1, &format!("write: {e}")));
            block_idx += 1;
        }
        remaining -= want;
    }
    out.flush().unwrap();
}
