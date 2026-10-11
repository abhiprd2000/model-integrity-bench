// Shared library for the revision experiments (A-D): per-4 KiB-block BLAKE3
// leaves, a binary Merkle tree over them (unbatched and batched updates), an
// ed25519-signed root record with a version (rollback protection), and the
// sidecar file format used by `mib seal` / `mib verify-read`.
//
// The earlier experiments (src/experiments.rs, src/main.rs) keep their own
// copies of the tree code and are untouched.
//
// Padding: the tree is padded to a power of two with ZERO leaves (not with a
// copy of the last real leaf as in experiments.rs). A duplicated-last-leaf pad
// would have to be rewritten whenever the last real block changes; a constant
// pad never changes, so a dirty set only ever contains real blocks. File length
// is bound by the signed record (file_len), so padding cannot be abused to
// extend a file.
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use std::time::Instant;

pub const BLOCK: usize = 4096;
pub type Hash = [u8; 32];
pub const ZERO: Hash = [0u8; 32];

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

// Median per-op nanoseconds over `trials` trials of `iters` repetitions after
// `warmup` warm-up calls.
pub fn bench_ns(warmup: usize, trials: usize, iters: u64, mut f: impl FnMut()) -> f64 {
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

pub fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

pub fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    use std::io::Read;
    let mut b = [0u8; N];
    std::fs::File::open("/dev/urandom").unwrap().read_exact(&mut b).unwrap();
    b
}

// ---------------------------------------------------------------- hashing
#[inline]
pub fn leaf_hash(b: &[u8]) -> Hash {
    *blake3::hash(b).as_bytes()
}

#[inline]
pub fn node_hash(l: &Hash, r: &Hash) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(l);
    h.update(r);
    *h.finalize().as_bytes()
}

pub fn nblocks(len: usize) -> usize {
    len.div_ceil(BLOCK)
}

#[inline]
pub fn block_range(i: usize, len: usize) -> (usize, usize) {
    let s = i * BLOCK;
    (s, (s + BLOCK).min(len))
}

pub fn leaves_of(data: &[u8]) -> Vec<Hash> {
    (0..nblocks(data.len()))
        .map(|i| {
            let (s, e) = block_range(i, data.len());
            leaf_hash(&data[s..e])
        })
        .collect()
}

// levels[0] = padded leaves, levels[last] = [root].
pub fn build_tree(leaves: &[Hash]) -> Vec<Vec<Hash>> {
    let mut padded = leaves.to_vec();
    let target = padded.len().max(1).next_power_of_two();
    padded.resize(target, ZERO);
    let mut levels = vec![padded];
    while levels.last().unwrap().len() > 1 {
        let next: Vec<Hash> = levels
            .last()
            .unwrap()
            .chunks(2)
            .map(|p| node_hash(&p[0], &p[1]))
            .collect();
        levels.push(next);
    }
    levels
}

pub fn root_of_leaves(leaves: &[Hash]) -> Hash {
    build_tree(leaves).last().unwrap()[0]
}

// ---------------------------------------------------------------- updates
// Update one leaf and every node on its path (depth node hashes). Returns the
// number of node hashes computed.
pub fn update_leaf(levels: &mut [Vec<Hash>], mut idx: usize, new_leaf: Hash) -> usize {
    levels[0][idx] = new_leaf;
    for lvl in 0..levels.len() - 1 {
        let (l, r) = if idx & 1 == 0 {
            (levels[lvl][idx], levels[lvl][idx + 1])
        } else {
            (levels[lvl][idx - 1], levels[lvl][idx])
        };
        idx >>= 1;
        levels[lvl + 1][idx] = node_hash(&l, &r);
    }
    levels.len() - 1
}

// Unbatched: for each dirty block rehash its leaf and walk its full path.
// Returns node hashes computed (= dirty.len() * depth).
pub fn update_unbatched(levels: &mut [Vec<Hash>], dirty: &[usize], data: &[u8]) -> usize {
    let mut nodes = 0;
    for &i in dirty {
        let (s, e) = block_range(i, data.len());
        nodes += update_leaf(levels, i, leaf_hash(&data[s..e]));
    }
    nodes
}

// Batched: rehash all dirty leaves, then recompute each affected parent ONCE
// per level (shared upper nodes hashed once). `dirty` must be sorted + unique.
// Returns node hashes computed.
pub fn update_batched(levels: &mut [Vec<Hash>], dirty: &[usize], data: &[u8]) -> usize {
    let mut idxs: Vec<usize> = Vec::with_capacity(dirty.len());
    for &i in dirty {
        let (s, e) = block_range(i, data.len());
        levels[0][i] = leaf_hash(&data[s..e]);
        idxs.push(i);
    }
    let mut nodes = 0;
    for lvl in 0..levels.len() - 1 {
        // idxs is sorted, so equal parents are adjacent: dedup in place.
        let mut w = 0;
        for r in 0..idxs.len() {
            let p = idxs[r] >> 1;
            if w == 0 || idxs[w - 1] != p {
                idxs[w] = p;
                w += 1;
            }
        }
        idxs.truncate(w);
        for &p in &idxs {
            levels[lvl + 1][p] = node_hash(&levels[lvl][2 * p], &levels[lvl][2 * p + 1]);
            nodes += 1;
        }
    }
    nodes
}

// Checked read: block must hash to its leaf AND the leaf's authentication path
// must reach `root`.
pub fn verify_path(levels: &[Vec<Hash>], idx: usize, block: &[u8], root: &Hash) -> bool {
    let mut cur = leaf_hash(block);
    if cur != levels[0][idx] {
        return false;
    }
    let mut i = idx;
    for lvl in 0..levels.len() - 1 {
        let sib = levels[lvl][i ^ 1];
        cur = if i & 1 == 0 { node_hash(&cur, &sib) } else { node_hash(&sib, &cur) };
        i >>= 1;
    }
    cur == *root
}

// First mismatching block index, if any.
pub fn first_bad_block(data: &[u8], leaves: &[Hash]) -> Option<usize> {
    (0..leaves.len()).find(|&i| {
        let (s, e) = block_range(i, data.len());
        leaf_hash(&data[s..e]) != leaves[i]
    })
}

// ------------------------------------------------------------ signed root
pub struct RootRecord {
    pub root: Hash,
    pub version: u64,
    pub file_len: u64,
    pub model_id: String,
    pub sig: [u8; 64],
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OpenError {
    Malformed,
    BadSignature,
    WrongModel,
    Rollback,
    LengthMismatch,
    RootMismatch,
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl RootRecord {
    fn message(root: &Hash, version: u64, file_len: u64, model_id: &str) -> Vec<u8> {
        let mut m = Vec::with_capacity(64 + model_id.len());
        m.extend_from_slice(b"MIB-ROOT-V1\0");
        m.extend_from_slice(&version.to_le_bytes());
        m.extend_from_slice(&file_len.to_le_bytes());
        m.extend_from_slice(root);
        m.extend_from_slice(&(model_id.len() as u16).to_le_bytes());
        m.extend_from_slice(model_id.as_bytes());
        m
    }

    pub fn sign(key: &SigningKey, root: Hash, version: u64, file_len: u64, model_id: &str) -> Self {
        let sig = key.sign(&Self::message(&root, version, file_len, model_id)).to_bytes();
        RootRecord { root, version, file_len, model_id: model_id.to_string(), sig }
    }

    pub fn verify(&self, pk: &VerifyingKey) -> bool {
        let msg = Self::message(&self.root, self.version, self.file_len, &self.model_id);
        pk.verify_strict(&msg, &Signature::from_bytes(&self.sig)).is_ok()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(114 + self.model_id.len());
        b.extend_from_slice(&self.root);
        b.extend_from_slice(&self.version.to_le_bytes());
        b.extend_from_slice(&self.file_len.to_le_bytes());
        b.extend_from_slice(&self.sig);
        b.extend_from_slice(&(self.model_id.len() as u16).to_le_bytes());
        b.extend_from_slice(self.model_id.as_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 114 {
            return None;
        }
        let id_len = u16::from_le_bytes(b[112..114].try_into().ok()?) as usize;
        if b.len() != 114 + id_len {
            return None;
        }
        Some(RootRecord {
            root: b[0..32].try_into().ok()?,
            version: u64::from_le_bytes(b[32..40].try_into().ok()?),
            file_len: u64::from_le_bytes(b[40..48].try_into().ok()?),
            sig: b[48..112].try_into().ok()?,
            model_id: String::from_utf8(b[114..].to_vec()).ok()?,
        })
    }
}

// Sidecar = signed record + the per-block leaf hashes (the "leaf file").
pub struct Sidecar {
    pub rec: RootRecord,
    pub leaves: Vec<Hash>,
}

impl Sidecar {
    pub fn to_bytes(&self) -> Vec<u8> {
        let rec = self.rec.to_bytes();
        let mut b = Vec::with_capacity(20 + rec.len() + 32 * self.leaves.len());
        b.extend_from_slice(b"MIBSIDE1");
        b.extend_from_slice(&(rec.len() as u32).to_le_bytes());
        b.extend_from_slice(&rec);
        b.extend_from_slice(&(self.leaves.len() as u64).to_le_bytes());
        for l in &self.leaves {
            b.extend_from_slice(l);
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 12 || &b[..8] != b"MIBSIDE1" {
            return None;
        }
        let rl = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
        let rec = RootRecord::from_bytes(b.get(12..12 + rl)?)?;
        let n = u64::from_le_bytes(b.get(12 + rl..20 + rl)?.try_into().ok()?) as usize;
        let body = b.get(20 + rl..)?;
        if body.len() != n.checked_mul(32)? {
            return None;
        }
        let leaves = body.chunks_exact(32).map(|c| c.try_into().unwrap()).collect();
        Some(Sidecar { rec, leaves })
    }
}

// Open-time check of the device: signature, model id, rollback floor, file
// length, and that the leaf file reproduces the signed Merkle root.
pub fn open_verified(
    sc: &Sidecar,
    file_len: u64,
    model_id: &str,
    pk: &VerifyingKey,
    min_version: u64,
) -> Result<(), OpenError> {
    if !sc.rec.verify(pk) {
        return Err(OpenError::BadSignature);
    }
    if sc.rec.model_id != model_id {
        return Err(OpenError::WrongModel);
    }
    if sc.rec.version < min_version {
        return Err(OpenError::Rollback);
    }
    if sc.rec.file_len != file_len || nblocks(file_len as usize) != sc.leaves.len() {
        return Err(OpenError::LengthMismatch);
    }
    if root_of_leaves(&sc.leaves) != sc.rec.root {
        return Err(OpenError::RootMismatch);
    }
    Ok(())
}

pub fn seal(data: &[u8], key: &SigningKey, version: u64, model_id: &str) -> Sidecar {
    let leaves = leaves_of(data);
    let root = root_of_leaves(&leaves);
    let rec = RootRecord::sign(key, root, version, data.len() as u64, model_id);
    Sidecar { rec, leaves }
}
