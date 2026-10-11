// Part C: crash consistency + concurrency.
//
//   r_crash crash   -> results/r_crash.csv          (crash injection matrix)
//   r_crash cost    -> results/r_crash_cost.csv     (per-update cost, journal + fsync)
//   r_crash conc    -> results/r_crash_concurrency.csv
//   r_crash all     -> all three
//
// On-disk store (a directory):
//   data.bin     the model file
//   tree.bin     all Merkle levels flattened (level 0 = padded leaves ... root)
//   root.rec     the signed root record (merkle_root, version, model_id, file_len)
//   journal.bin  exists only while an update is in flight
//   root.tmp     temp file for the atomic root replace
//
// Update protocol (device side; the "update service" signs the new root and
// ships {block, new data, new leaf, signed new record} as the update package):
//   1 append journal entry {block, data, leaf, new signed record} + fsync
//   2 write the data block
//   3 write the new leaf and the path nodes to tree.bin
//   4 write the new signed root to root.tmp + fsync + rename over root.rec
//   5 clear (unlink) the journal
// Recovery: if a valid journal exists, re-apply it (idempotent), then audit.
// Audit: signature + version floor on the root, stored tree == tree recomputed
// from its leaves, leaves reproduce the signed root, every data block hashes
// to its leaf, file length matches.
//
// Deviation from the one-line spec, deliberately: the journal entry also holds
// the signed new root record. The device has no signing key, so after a crash
// between steps 1 and 4 it could not otherwise produce a validly signed root.
//
// Crash injection is process abort (SIGABRT) of a child at a boundary: it
// tests ordering/atomicity logic, NOT loss of the SSD's volatile write cache
// (the page cache survives a process abort). Fsync costs are real SSD costs.
use ed25519_dalek::{SigningKey, VerifyingKey};
use mib::*;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::hint::black_box;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

const MODEL: &str = "models/mid.tflite";
const MODEL_ID: &str = "mobilenet_v2";
const WORK: &str = ".cache/crash";

// ------------------------------------------------------------ store layout
fn level_starts(padded: usize) -> Vec<usize> {
    let (mut v, mut s, mut n) = (vec![], 0, padded);
    loop {
        v.push(s);
        s += n;
        if n == 1 {
            break;
        }
        n /= 2;
    }
    v
}

fn flatten(levels: &[Vec<Hash>]) -> Vec<u8> {
    levels.iter().flatten().flat_map(|h| h.iter().copied()).collect()
}

fn unflatten(b: &[u8], padded: usize) -> Vec<Vec<Hash>> {
    let mut out = vec![];
    let mut n = padded;
    let mut pos = 0;
    loop {
        out.push(b[pos..pos + 32 * n].chunks_exact(32).map(|c| c.try_into().unwrap()).collect());
        pos += 32 * n;
        if n == 1 {
            break;
        }
        n /= 2;
    }
    out
}

struct Entry {
    block: u64,
    data: Vec<u8>,
    leaf: Hash,
    rec: RootRecord,
}

impl Entry {
    fn to_bytes(&self) -> Vec<u8> {
        let rec = self.rec.to_bytes();
        let mut b = Vec::new();
        b.extend_from_slice(b"MIBJRNL1");
        b.extend_from_slice(&self.block.to_le_bytes());
        b.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.data);
        b.extend_from_slice(&self.leaf);
        b.extend_from_slice(&(rec.len() as u32).to_le_bytes());
        b.extend_from_slice(&rec);
        let ck = *blake3::hash(&b).as_bytes();
        b.extend_from_slice(&ck);
        b
    }

    fn parse(b: &[u8]) -> Option<Entry> {
        if b.len() < 8 + 8 + 4 + 32 + 4 + 32 || &b[..8] != b"MIBJRNL1" {
            return None;
        }
        let (body, ck) = b.split_at(b.len() - 32);
        if blake3::hash(body).as_bytes() != ck {
            return None;
        }
        let block = u64::from_le_bytes(body[8..16].try_into().ok()?);
        let dl = u32::from_le_bytes(body[16..20].try_into().ok()?) as usize;
        let data = body.get(20..20 + dl)?.to_vec();
        let leaf: Hash = body.get(20 + dl..52 + dl)?.try_into().ok()?;
        let rl = u32::from_le_bytes(body.get(52 + dl..56 + dl)?.try_into().ok()?) as usize;
        let rec = RootRecord::from_bytes(body.get(56 + dl..56 + dl + rl)?)?;
        Some(Entry { block, data, leaf, rec })
    }
}

fn p(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

fn sync_dir(dir: &Path) {
    File::open(dir).unwrap().sync_all().unwrap();
}

fn write_root_atomic(dir: &Path, rec: &RootRecord, strict: bool, crash: &str) {
    let tmp = p(dir, "root.tmp");
    let mut f = File::create(&tmp).unwrap();
    f.write_all(&rec.to_bytes()).unwrap();
    f.sync_all().unwrap();
    maybe_crash("after_4a", crash);
    fs::rename(&tmp, p(dir, "root.rec")).unwrap();
    if strict {
        sync_dir(dir);
    }
}

fn maybe_crash(point: &str, crash: &str) {
    if point == crash {
        std::process::abort();
    }
}

fn init_store(dir: &Path, key: &SigningKey, model: &[u8]) -> Vec<Vec<Hash>> {
    fs::create_dir_all(dir).unwrap();
    fs::write(p(dir, "data.bin"), model).unwrap();
    let levels = build_tree(&leaves_of(model));
    fs::write(p(dir, "tree.bin"), flatten(&levels)).unwrap();
    let rec = RootRecord::sign(key, levels.last().unwrap()[0], 1, model.len() as u64, MODEL_ID);
    fs::write(p(dir, "root.rec"), rec.to_bytes()).unwrap();
    levels
}

struct Prepared {
    nodes: Vec<(usize, usize, Hash)>, // (level, index, hash), leaf first
    root: Hash,
}

fn read_node(tree: &File, starts: &[usize], lvl: usize, i: usize) -> Hash {
    let mut b = [0u8; 32];
    tree.read_exact_at(&mut b, 32 * (starts[lvl] + i) as u64).unwrap();
    b
}

fn prepare(tree: &File, starts: &[usize], idx: usize, new_leaf: Hash) -> Prepared {
    let depth = starts.len() - 1;
    let (mut cur, mut i) = (new_leaf, idx);
    let mut nodes = vec![(0, i, cur)];
    for lvl in 0..depth {
        let sib = read_node(tree, starts, lvl, i ^ 1);
        cur = if i & 1 == 0 { node_hash(&cur, &sib) } else { node_hash(&sib, &cur) };
        i >>= 1;
        nodes.push((lvl + 1, i, cur));
    }
    Prepared { nodes, root: cur }
}

#[derive(Default, Clone)]
struct StepTimes {
    prepare_us: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
}

fn us(a: Instant, b: Instant) -> f64 {
    b.duration_since(a).as_nanos() as f64 / 1e3
}

fn apply_update(
    dir: &Path,
    e: &Entry,
    pk: &VerifyingKey,
    journaling: bool,
    strict: bool,
    crash: &str,
) -> Result<StepTimes, String> {
    let mut st = StepTimes::default();
    let t0 = Instant::now();
    // -- accept the package: signature, id, exact next version, leaf, root
    let cur = RootRecord::from_bytes(&fs::read(p(dir, "root.rec")).map_err(|e| e.to_string())?)
        .ok_or("bad root.rec")?;
    if !e.rec.verify(pk) {
        return Err("package signature invalid".into());
    }
    if e.rec.model_id != cur.model_id || e.rec.version != cur.version + 1 || e.rec.file_len != cur.file_len {
        return Err("package not the next version of this model".into());
    }
    let nb = nblocks(cur.file_len as usize);
    let (s, en) = block_range(e.block as usize, cur.file_len as usize);
    if e.block as usize >= nb || e.data.len() != en - s || leaf_hash(&e.data) != e.leaf {
        return Err("package block/leaf invalid".into());
    }
    let starts = level_starts(nb.next_power_of_two());
    let tree = OpenOptions::new().read(true).write(true).open(p(dir, "tree.bin")).unwrap();
    let prep = prepare(&tree, &starts, e.block as usize, e.leaf);
    if prep.root != e.rec.root {
        return Err("package root does not match the tree after update".into());
    }
    let t1 = Instant::now();
    st.prepare_us = us(t0, t1);
    maybe_crash("before_1", crash);

    // -- step 1: journal
    if journaling {
        let bytes = e.to_bytes();
        let mut j = File::create(p(dir, "journal.bin")).unwrap();
        if crash == "torn_1" {
            j.write_all(&bytes[..bytes.len() / 2]).unwrap();
            std::process::abort();
        }
        j.write_all(&bytes).unwrap();
        j.sync_all().unwrap();
        if strict {
            sync_dir(dir);
        }
    } else if crash == "torn_1" || crash == "after_1" {
        return Err("crash point not applicable without a journal".into());
    }
    let t2 = Instant::now();
    st.s1 = us(t1, t2);
    maybe_crash("after_1", crash);

    // -- step 2: data block
    let data = OpenOptions::new().write(true).open(p(dir, "data.bin")).unwrap();
    if crash == "torn_2" {
        data.write_all_at(&e.data[..e.data.len() / 2], s as u64).unwrap();
        std::process::abort();
    }
    data.write_all_at(&e.data, s as u64).unwrap();
    let t3 = Instant::now();
    st.s2 = us(t2, t3);
    maybe_crash("after_2", crash);

    // -- step 3: leaf + path nodes
    for (k, (lvl, i, h)) in prep.nodes.iter().enumerate() {
        if crash == "torn_3" && k == prep.nodes.len() / 2 {
            std::process::abort();
        }
        tree.write_all_at(h, 32 * (starts[*lvl] + i) as u64).unwrap();
    }
    if strict {
        data.sync_data().unwrap();
        tree.sync_data().unwrap();
    }
    let t4 = Instant::now();
    st.s3 = us(t3, t4);
    maybe_crash("after_3", crash);

    // -- step 4: signed root, temp + fsync + rename
    write_root_atomic(dir, &e.rec, strict, crash);
    let t5 = Instant::now();
    st.s4 = us(t4, t5);
    maybe_crash("after_4", crash);

    // -- step 5: clear journal
    if journaling {
        fs::remove_file(p(dir, "journal.bin")).unwrap();
        if strict {
            sync_dir(dir);
        }
    }
    let t6 = Instant::now();
    st.s5 = us(t5, t6);
    maybe_crash("after_5", crash);
    Ok(st)
}

// Open-time recovery: re-apply a valid journal (idempotent), discard a torn one.
fn recover(dir: &Path, pk: &VerifyingKey) -> Result<&'static str, String> {
    let _ = fs::remove_file(p(dir, "root.tmp"));
    let bytes = match fs::read(p(dir, "journal.bin")) {
        Ok(b) => b,
        Err(_) => return Ok("no_journal"),
    };
    let Some(e) = Entry::parse(&bytes) else {
        // Journal not durable/complete: step 1 never finished, so steps 2-4
        // never started. Nothing to re-apply.
        fs::remove_file(p(dir, "journal.bin")).unwrap();
        sync_dir(dir);
        return Ok("torn_journal_discarded");
    };
    if !e.rec.verify(pk) {
        return Err("journal record signature invalid".into());
    }
    let cur = RootRecord::from_bytes(&fs::read(p(dir, "root.rec")).map_err(|e| e.to_string())?)
        .ok_or("bad root.rec")?;
    if e.rec.model_id != cur.model_id || e.rec.file_len != cur.file_len {
        return Err("journal does not belong to this model".into());
    }
    let nb = nblocks(cur.file_len as usize);
    if e.block as usize >= nb {
        return Err("journal block out of range".into());
    }
    let starts = level_starts(nb.next_power_of_two());
    let (s, _) = block_range(e.block as usize, cur.file_len as usize);
    let data = OpenOptions::new().write(true).open(p(dir, "data.bin")).unwrap();
    data.write_all_at(&e.data, s as u64).unwrap();
    let tree = OpenOptions::new().read(true).write(true).open(p(dir, "tree.bin")).unwrap();
    let prep = prepare(&tree, &starts, e.block as usize, leaf_hash(&e.data));
    if prep.root != e.rec.root {
        return Err("journal root does not match recomputed root".into());
    }
    for (lvl, i, h) in &prep.nodes {
        tree.write_all_at(h, 32 * (starts[*lvl] + i) as u64).unwrap();
    }
    data.sync_data().unwrap();
    tree.sync_data().unwrap();
    if cur.version < e.rec.version {
        write_root_atomic(dir, &e.rec, true, "none");
    } else if cur.version > e.rec.version || cur.root != e.rec.root {
        return Err("journal older than/inconsistent with current root".into());
    }
    fs::remove_file(p(dir, "journal.bin")).unwrap();
    sync_dir(dir);
    Ok("journal_replayed")
}

fn audit(dir: &Path, pk: &VerifyingKey, min_version: u64) -> Result<u64, String> {
    let rec = RootRecord::from_bytes(&fs::read(p(dir, "root.rec")).map_err(|e| format!("root.rec: {e}"))?)
        .ok_or("root.rec malformed")?;
    if !rec.verify(pk) {
        return Err("root signature invalid".into());
    }
    if rec.version < min_version {
        return Err("root version below rollback floor".into());
    }
    let data = fs::read(p(dir, "data.bin")).map_err(|e| e.to_string())?;
    if data.len() as u64 != rec.file_len {
        return Err("data length != signed length".into());
    }
    let nb = nblocks(data.len());
    let padded = nb.next_power_of_two();
    let tb = fs::read(p(dir, "tree.bin")).map_err(|e| e.to_string())?;
    if tb.len() != 32 * (2 * padded - 1) {
        return Err("tree.bin wrong size".into());
    }
    let levels = unflatten(&tb, padded);
    if levels[0][nb..].iter().any(|h| *h != ZERO) {
        return Err("padding leaves not zero".into());
    }
    if build_tree(&levels[0][..nb]) != levels {
        return Err("stored tree inconsistent with its leaves".into());
    }
    if levels.last().unwrap()[0] != rec.root {
        return Err("leaves do not reproduce the signed root".into());
    }
    if let Some(i) = first_bad_block(&data, &levels[0][..nb]) {
        return Err(format!("data block {i} does not match its leaf"));
    }
    Ok(rec.version)
}

// ------------------------------------------------------- update service
struct Service {
    levels: Vec<Vec<Hash>>,
    version: u64,
    file_len: usize,
    key: SigningKey,
}

impl Service {
    fn package(&mut self, block: usize, new_data: Vec<u8>) -> Entry {
        let leaf = leaf_hash(&new_data);
        update_leaf(&mut self.levels, block, leaf);
        self.version += 1;
        let rec = RootRecord::sign(
            &self.key,
            self.levels.last().unwrap()[0],
            self.version,
            self.file_len as u64,
            MODEL_ID,
        );
        Entry { block: block as u64, data: new_data, leaf, rec }
    }
}

fn rand_block(rng: &mut u64, nb: usize, len: usize) -> (usize, Vec<u8>) {
    let b = (xorshift(rng) as usize) % nb;
    let (s, e) = block_range(b, len);
    let mut d = vec![0u8; e - s];
    for c in d.chunks_mut(8) {
        let r = xorshift(rng).to_le_bytes();
        c.copy_from_slice(&r[..c.len()]);
    }
    (b, d)
}

// ------------------------------------------------------------ child mode
fn child_main(a: &[String]) {
    // child <dir> <pkg> <pubfile> <protocol> <point>
    let pk = VerifyingKey::from_bytes(&fs::read(&a[4]).unwrap().try_into().unwrap()).unwrap();
    let e = Entry::parse(&fs::read(&a[3]).unwrap()).expect("package");
    let journaling = a[5] == "journal";
    match apply_update(Path::new(&a[2]), &e, &pk, journaling, true, &a[6]) {
        Ok(_) => std::process::exit(0),
        Err(m) => {
            eprintln!("child: {m}");
            std::process::exit(9);
        }
    }
}

// ------------------------------------------------------------- crash matrix
fn crash_matrix(key: &SigningKey, pk: &VerifyingKey, model: &[u8]) {
    let base = PathBuf::from(WORK).join("base");
    let work = PathBuf::from(WORK).join("work");
    let pubf = PathBuf::from(WORK).join("pub.key");
    let pkgf = PathBuf::from(WORK).join("pkg.bin");
    fs::write(&pubf, pk.to_bytes()).unwrap();
    let base_levels = init_store(&base, key, model);
    let nb = nblocks(model.len());
    let exe = std::env::current_exe().unwrap();

    // protocol, point, kind
    let mut plan: Vec<(&str, &str, &str)> = vec![];
    for (pt, kind) in [
        ("before_1", "extra"),
        ("after_1", "spec_boundary_1"),
        ("after_2", "spec_boundary_2"),
        ("after_3", "spec_boundary_3"),
        ("after_4", "spec_boundary_4"),
        ("after_5", "spec_boundary_5"),
        ("torn_1", "extra_torn_journal"),
        ("torn_2", "extra_torn_data_block"),
        ("torn_3", "extra_torn_tree_path"),
        ("after_4a", "extra_before_rename"),
    ] {
        plan.push(("journal", pt, kind));
    }
    // ablation: same crash points, no journal (steps 2,3,4 only)
    for (pt, kind) in [
        ("after_2", "ablation"),
        ("torn_2", "ablation"),
        ("torn_3", "ablation"),
        ("after_3", "ablation"),
        ("after_4a", "ablation"),
        ("after_4", "ablation"),
    ] {
        plan.push(("nojournal", pt, kind));
    }

    let mut csv = String::from(
        "protocol,crash_point,kind,trials,consistent_old,consistent_new,detected_inconsistent,UNDETECTED,\
recovery_replayed,torn_journal_discarded,example_detection\n",
    );
    let trials = 100;
    let mut rng: u64 = 0xC0FFEE_u64 ^ 0x9E3779B97F4A7C15;
    let mut total_undetected = 0;
    for (proto, point, kind) in plan {
        let (mut old, mut new, mut det, mut und) = (0, 0, 0, 0);
        let (mut replayed, mut discarded) = (0, 0);
        let mut example = String::new();
        for _ in 0..trials {
            let (blk, newd) = rand_block(&mut rng, nb, model.len());
            let mut svc = Service { levels: base_levels.clone(), version: 1, file_len: model.len(), key: key.clone() };
            let e = svc.package(blk, newd.clone());
            fs::write(&pkgf, e.to_bytes()).unwrap();
            let _ = fs::remove_dir_all(&work);
            fs::create_dir_all(&work).unwrap();
            for f in ["data.bin", "tree.bin", "root.rec"] {
                fs::copy(p(&base, f), p(&work, f)).unwrap();
            }
            // Untimed harness setup: flush the freshly copied store so the
            // child's fsyncs below only pay for the update's own writes.
            let _ = std::process::Command::new("sync").status();
            let st = std::process::Command::new(&exe)
                .args(["child", work.to_str().unwrap(), pkgf.to_str().unwrap(), pubf.to_str().unwrap(), proto, point])
                .status()
                .unwrap();
            // The child must have really been killed at the point (SIGABRT).
            assert_eq!(st.signal(), Some(6), "child did not abort at {proto}/{point}: {st:?}");

            // "Reboot": recovery, then full audit, then classify against the
            // two legal states (all-old, all-new) known to the harness.
            let outcome = match recover(&work, pk) {
                Err(m) => Err(m),
                Ok(r) => {
                    match r {
                        "journal_replayed" => replayed += 1,
                        "torn_journal_discarded" => discarded += 1,
                        _ => {}
                    }
                    audit(&work, pk, 1)
                }
            };
            match outcome {
                Err(m) => {
                    det += 1;
                    if example.is_empty() {
                        example = m;
                    }
                }
                Ok(ver) => {
                    let d = fs::read(p(&work, "data.bin")).unwrap();
                    let (s, en) = block_range(blk, model.len());
                    let same_outside = d.len() == model.len() && d[..s] == model[..s] && d[en..] == model[en..];
                    let is_old = same_outside && d[s..en] == model[s..en] && ver == 1;
                    let is_new = same_outside && d[s..en] == newd[..] && ver == 2;
                    if is_old {
                        old += 1;
                    } else if is_new {
                        new += 1;
                    } else {
                        und += 1;
                    }
                }
            }
        }
        total_undetected += und;
        println!("{proto:<10} {point:<9} old={old:<3} new={new:<3} detected={det:<3} UNDETECTED={und} replayed={replayed} torn_discarded={discarded}");
        let _ = writeln!(csv, "{proto},{point},{kind},{trials},{old},{new},{det},{und},{replayed},{discarded},\"{example}\"");
    }
    fs::write("results/r_crash.csv", csv).unwrap();
    println!("wrote results/r_crash.csv (UNDETECTED total = {total_undetected})");
    let _ = fs::remove_dir_all(&work);
}

// --------------------------------------------------------------- cost
fn cost(key: &SigningKey, pk: &VerifyingKey, model: &[u8]) {
    let dir = PathBuf::from(WORK).join("cost");
    let mut csv = String::from(
        "variant,journal,strict_fsync,samples,total_us_median,total_us_min,total_us_max,\
prepare_us,s1_journal_us,s2_data_us,s3_tree_us,s4_root_us,s5_clear_us\n",
    );
    let nb = nblocks(model.len());
    for (variant, journaling, strict, samples) in [
        ("journal_spec", true, false, 7usize),
        ("journal_strict", true, true, 7),
        ("nojournal_spec", false, false, 7),
        ("nojournal_strict", false, true, 7),
        ("journal_spec_n101", true, false, 101),
        ("journal_strict_n101", true, true, 101),
        ("nojournal_strict_n101", false, true, 101),
    ] {
        let _ = fs::remove_dir_all(&dir);
        let levels = init_store(&dir, key, model);
        let mut svc = Service { levels, version: 1, file_len: model.len(), key: key.clone() };
        let mut rng: u64 = 0xBADC0FFEE ^ 0x9E3779B97F4A7C15;
        let mut tot = vec![];
        let mut steps: Vec<StepTimes> = vec![];
        for k in 0..(3 + samples) {
            let (b, d) = rand_block(&mut rng, nb, model.len());
            let e = svc.package(b, d);
            let t = Instant::now();
            let st = apply_update(&dir, &e, pk, journaling, strict, "none_never").unwrap();
            let el = t.elapsed().as_nanos() as f64 / 1e3;
            if k >= 3 {
                tot.push(el);
                steps.push(st);
            }
        }
        let col = |f: fn(&StepTimes) -> f64| median(steps.iter().map(f).collect());
        let (mn, mx) = (tot.iter().cloned().fold(f64::MAX, f64::min), tot.iter().cloned().fold(0.0, f64::max));
        let med = median(tot);
        let _ = writeln!(
            csv,
            "{variant},{journaling},{strict},{samples},{med:.1},{mn:.1},{mx:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            col(|s| s.prepare_us), col(|s| s.s1), col(|s| s.s2), col(|s| s.s3), col(|s| s.s4), col(|s| s.s5)
        );
        println!("{variant:<24} median {med:>9.1} us  (min {mn:.1}, max {mx:.1})");
        // consistency: the store we just hammered must audit clean
        audit(&dir, pk, 1).expect("cost store audit");
    }
    // in-memory reference: leaf rehash + path update, no I/O
    let mut levels = build_tree(&leaves_of(model));
    let blk = &model[..BLOCK];
    let mem = bench_ns(16, 7, 5000, || {
        let l = leaf_hash(black_box(blk));
        update_leaf(&mut levels, 0, l);
    }) / 1e3;
    let _ = writeln!(csv, "in_memory_no_io,false,false,7,{mem:.1},,,,,,,,");
    println!("in_memory_no_io          median {mem:.2} us");
    fs::write("results/r_crash_cost.csv", csv).unwrap();
    println!("wrote results/r_crash_cost.csv");
    let _ = fs::remove_dir_all(&dir);
}

// -------------------------------------------------------- concurrency
struct Snap {
    data: Vec<u8>,
    levels: Vec<Vec<Hash>>,
    rec: RootRecord,
}

fn concurrency(key: &SigningKey, pk: &VerifyingKey, model: &[u8]) {
    const UPDATES: usize = 10_000;
    const READERS: usize = 4;
    let nb = nblocks(model.len());
    let levels0 = build_tree(&leaves_of(model));
    let rec0 = RootRecord::sign(key, levels0.last().unwrap()[0], 1, model.len() as u64, MODEL_ID);

    // negative control: the checker must reject a single flipped bit.
    let control = {
        let mut blk = model[..BLOCK].to_vec();
        blk[17] ^= 1;
        let good = verify_path(&levels0, 0, &model[..BLOCK], &rec0.root);
        let bad = verify_path(&levels0, 0, &blk, &rec0.root);
        good && !bad
    };
    assert!(control, "checked-read negative control failed");

    let mut csv = String::from(
        "scenario,writer_updates,reader_threads,duration_s,total_reads,failed_reads,sig_failures,\
version_regressions,reads_per_s_total,reads_per_s_per_thread,writer_updates_per_s,final_version,negative_control_detects_tamper\n",
    );
    // scenario: (name, writer_updates, readers, durable_writer, readers_only_seconds)
    for (name, updates, readers, durable, secs) in [
        ("readers_only", 0usize, READERS, false, 5.0f64),
        ("writer_only", UPDATES, 0, false, 0.0),
        ("writer_plus_4_readers", UPDATES, READERS, false, 0.0),
        ("durable_writer_plus_4_readers", UPDATES, READERS, true, 0.0),
    ] {
        let shared = Arc::new(RwLock::new(Snap {
            data: model.to_vec(),
            levels: levels0.clone(),
            rec: RootRecord::from_bytes(&rec0.to_bytes()).unwrap(),
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        let mut handles = vec![];
        for r in 0..readers {
            let (sh, st, pkc) = (shared.clone(), stop.clone(), *pk);
            handles.push(std::thread::spawn(move || {
                let mut rng: u64 = 0x1234_5678_9ABC_DEF0 ^ (r as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
                let (mut reads, mut fails, mut sigf, mut regress) = (0u64, 0u64, 0u64, 0u64);
                let (mut last_ver, mut last_sig_ver) = (0u64, 0u64);
                while !st.load(Ordering::Relaxed) {
                    let g = sh.read().unwrap();
                    let idx = (xorshift(&mut rng) as usize) % nb;
                    let (s, e) = block_range(idx, g.data.len());
                    let mut ok = verify_path(&g.levels, idx, &g.data[s..e], &g.rec.root)
                        && g.levels.last().unwrap()[0] == g.rec.root;
                    let ver = g.rec.version;
                    if ver != last_sig_ver {
                        // verify the snapshot's signed record once per new version seen
                        if !g.rec.verify(&pkc) {
                            sigf += 1;
                            ok = false;
                        }
                        last_sig_ver = ver;
                    }
                    drop(g);
                    if ver < last_ver {
                        regress += 1;
                    }
                    last_ver = ver;
                    reads += 1;
                    if !ok {
                        fails += 1;
                    }
                }
                (reads, fails, sigf, regress)
            }));
        }
        // writer (this thread)
        let w0 = Instant::now();
        let mut wrng: u64 = 0xFEED_FACE_CAFE_BEEF;
        let ddir = PathBuf::from(WORK).join("conc");
        let mut svc = if durable {
            let _ = fs::remove_dir_all(&ddir);
            let lv = init_store(&ddir, key, model);
            Some(Service { levels: lv, version: 1, file_len: model.len(), key: key.clone() })
        } else {
            None
        };
        for _ in 0..updates {
            let (b, d) = rand_block(&mut wrng, nb, model.len());
            if let Some(svc) = svc.as_mut() {
                // durable protocol runs OUTSIDE the lock; readers never wait on fsync
                let e = svc.package(b, d.clone());
                apply_update(&ddir, &e, pk, true, false, "none_never").unwrap();
            }
            let mut g = shared.write().unwrap();
            let (s, e) = block_range(b, g.data.len());
            g.data[s..e].copy_from_slice(&d);
            let leaf = leaf_hash(&d);
            update_leaf(&mut g.levels, b, leaf);
            let ver = g.rec.version + 1;
            let root = g.levels.last().unwrap()[0];
            g.rec = RootRecord::sign(key, root, ver, model.len() as u64, MODEL_ID);
        }
        if updates == 0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        }
        let wdur = w0.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
        let (mut reads, mut fails, mut sigf, mut regress) = (0, 0, 0, 0);
        for h in handles {
            let (a, b, c, d) = h.join().unwrap();
            reads += a;
            fails += b;
            sigf += c;
            regress += d;
        }
        let dur = t0.elapsed().as_secs_f64();
        let final_ver = shared.read().unwrap().rec.version;
        // final state must be fully self-consistent too
        {
            let g = shared.read().unwrap();
            assert_eq!(build_tree(&leaves_of(&g.data)), g.levels, "final snapshot inconsistent");
            assert_eq!(g.levels.last().unwrap()[0], g.rec.root);
        }
        let rps = reads as f64 / dur;
        let per = if readers > 0 { rps / readers as f64 } else { 0.0 };
        let ups = if updates > 0 { updates as f64 / wdur } else { 0.0 };
        let _ = writeln!(
            csv,
            "{name},{updates},{readers},{dur:.3},{reads},{fails},{sigf},{regress},{rps:.0},{per:.0},{ups:.0},{final_ver},{control}"
        );
        println!("{name:<32} reads={reads} failed={fails} sigfail={sigf} regress={regress} reads/s={rps:.0} writer/s={ups:.0} final_ver={final_ver}");
    }
    fs::write("results/r_crash_concurrency.csv", csv).unwrap();
    println!("wrote results/r_crash_concurrency.csv");
    let _ = fs::remove_dir_all(PathBuf::from(WORK).join("conc"));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("child") {
        child_main(&a);
        return;
    }
    let phase = a.get(1).map(String::as_str).unwrap_or("all");
    let model = fs::read(MODEL).unwrap_or_else(|e| panic!("{MODEL}: {e} (no synthetic substitute)"));
    let key = SigningKey::from_bytes(&random_bytes());
    let pk = key.verifying_key();
    fs::create_dir_all(WORK).unwrap();
    fs::create_dir_all("results").unwrap();
    if phase == "all" || phase == "crash" {
        crash_matrix(&key, &pk, &model);
    }
    if phase == "all" || phase == "cost" {
        cost(&key, &pk, &model);
    }
    if phase == "all" || phase == "conc" {
        concurrency(&key, &pk, &model);
    }
}
