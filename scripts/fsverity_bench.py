#!/usr/bin/env python3
"""fs-verity head-to-head -> results/r_fsverity.csv.   Run as root:
    sudo .venv/bin/python scripts/fsverity_bench.py

1 GiB ext4 image (mkfs.ext4 -O verity, 4 KiB blocks) loop-mounted under .cache/;
each model is copied in twice (a "verity" copy that gets fsverity-enabled and a
"plain" copy). Everything is timed with perf_counter around the real operation;
every value in the CSV comes from those timers or from existing results CSVs
(read at run time, never typed in). fs-verity here is SHA-256, ours is BLAKE3.

experiments written:
  env            kernel / config / fsverity version / image features / device
  enable         `fsverity enable --hash-alg=sha256 --block-size=4096` (median of
                 3 fresh copies, warm and cold) + the null-CLI spawn baseline
  read           full sequential read, verity vs plain copy (warm: 1 untimed +
                 median of 7; cold: drop_caches before each, median of 5)
  write_attempt  exact errno of write-opening / truncating a verity file
  update         legit one-block update: copy, change one block, fsync, enable,
                 rename (+ directory fsync); median of 3 end-to-end, warm and cold
  ours_update    our one-block update costs copied from existing results CSVs
"""
import csv, errno, hashlib, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "r_fsverity.csv"
IMG = ROOT / ".cache" / "verity.img"
MNT = ROOT / ".cache" / "verity.mnt"
MODELS = [("tiny", "tiny.tflite"), ("mid", "mid.tflite"), ("large", "large.onnx")]
rows = []
BLK = 4096


def add(exp, model, variant, cache, trials, med=None, mn=None, mx=None, detail=""):
    f = lambda v: "" if v is None else f"{v:.3f}"
    rows.append([exp, model, variant, cache, trials, f(med), f(mn), f(mx), detail])


def stats(xs):
    return statistics.median(xs), min(xs), max(xs)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def drop_caches():
    subprocess.run(["sync"], check=True)
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3")


def timed_cli(cmd):
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = (time.perf_counter() - t) * 1e3
    if p.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}: {p.stderr.strip()}")
    return dt


def measure(path):
    return run(["fsverity", "measure", str(path)]).stdout.split()[0]


def read_all(path, buf):
    fd = os.open(path, os.O_RDONLY)
    mv, n, sz = memoryview(buf), 0, len(buf)
    t = time.perf_counter()
    while n < sz:
        r = os.readv(fd, [mv[n:]])
        if r == 0:
            break
        n += r
    dt = (time.perf_counter() - t) * 1e3
    os.close(fd)
    assert n == sz, (n, sz)
    return dt


if os.geteuid() != 0:
    sys.exit("run as root (loop mount + drop_caches)")
if shutil.which("fsverity") is None:
    sys.exit("fsverity not installed: sudo apt install fsverity")

cfg = [l.strip() for l in open(f"/boot/config-{os.uname().release}") if l.startswith("CONFIG_FS_VERITY")]
backing = run(["findmnt", "-T", str(IMG.parent), "-no", "SOURCE,FSTYPE"]).stdout.strip()
add("env", "-", "kernel", "-", 0, detail=os.uname().release)
add("env", "-", "kernel_config", "-", 0, detail=";".join(cfg))
add("env", "-", "fsverity_version", "-", 0, detail=run(["fsverity", "--version"]).stdout.strip())
add("env", "-", "image_backing_fs", "-", 0, detail=backing)
dev = backing.split()[0].replace("/dev/", "")
base = "".join(c for c in dev if not c.isdigit()) if dev.startswith("sd") else dev
try:
    rot = Path(f"/sys/block/{base}/queue/rotational").read_text().strip()
    mdl = Path(f"/sys/block/{base}/device/model").read_text().strip()
    add("env", "-", "backing_device", "-", 0, detail=f"{base} rotational={rot} model={mdl}")
except OSError:
    pass

try:
    for p in (IMG,):
        p.unlink(missing_ok=True)
    MNT.mkdir(parents=True, exist_ok=True)
    with open(IMG, "wb") as f:
        f.truncate(1 << 30)
    run(["mkfs.ext4", "-q", "-F", "-O", "verity", "-b", "4096", str(IMG)])
    feats = [l for l in run(["dumpe2fs", "-h", str(IMG)]).stdout.splitlines() if l.startswith("Filesystem features")][0]
    add("env", "-", "image_features", "-", 0, detail=feats.split(":", 1)[1].strip())
    run(["mount", "-o", "loop", str(IMG), str(MNT)])

    # null-CLI baseline: spawn + exit of the fsverity binary doing no work
    base_ms = [timed_cli(["fsverity", "--version"]) for _ in range(7)]
    add("enable", "-", "cli_spawn_baseline(fsverity --version)", "warm", 7, *stats(base_ms))

    for label, fname in MODELS:
        src = ROOT / "models" / fname
        sz = src.stat().st_size
        src_digest = hashlib.sha256(src.read_bytes()).hexdigest()
        vpath, ppath = MNT / f"{fname}.verity", MNT / f"{fname}.plain"
        shutil.copyfile(src, ppath)
        shutil.copyfile(src, vpath)
        os.sync()

        # ---- a) tree build: fresh copy each trial
        for cache, trials in (("warm", 3), ("cold", 3)):
            ts = []
            for k in range(trials):
                tmp = MNT / f"{fname}.enable{k}"
                shutil.copyfile(ppath, tmp)
                os.sync()
                if cache == "cold":
                    drop_caches()
                ts.append(timed_cli(["fsverity", "enable", "--hash-alg=sha256", "--block-size=4096", str(tmp)]))
                dg = measure(tmp)
                tmp.unlink()
            add("enable", label, "fsverity enable sha256 4096", cache, trials, *stats(ts),
                detail=f"file_bytes={sz};digest={dg}")
        run(["fsverity", "enable", "--hash-alg=sha256", "--block-size=4096", str(vpath)])
        vdigest = measure(vpath)
        os.sync()

        # content identical to the source through both copies (untimed)
        for pth in (vpath, ppath):
            assert hashlib.sha256(pth.read_bytes()).hexdigest() == src_digest, pth

        # ---- b) sequential read
        buf = bytearray(sz)
        for cache, trials in (("warm", 7), ("cold", 5)):
            for variant, pth in (("plain_copy", ppath), ("verity_copy", vpath)):
                if cache == "warm":
                    read_all(pth, buf)  # untimed warm-up
                ts = []
                for _ in range(trials):
                    if cache == "cold":
                        drop_caches()
                    ts.append(read_all(pth, buf))
                med, mn, mx = stats(ts)
                add("read", label, variant, cache, trials, med, mn, mx,
                    detail=f"file_bytes={sz};MBps={sz / (med / 1e3) / 1e6:.1f}")

        # ---- c) write attempts on the verity file
        before = measure(vpath)
        attempts = [
            ("open O_WRONLY", lambda: os.close(os.open(vpath, os.O_WRONLY))),
            ("open O_RDWR", lambda: os.close(os.open(vpath, os.O_RDWR))),
            ("open O_WRONLY|O_APPEND", lambda: os.close(os.open(vpath, os.O_WRONLY | os.O_APPEND))),
            ("truncate(0)", lambda: os.truncate(vpath, 0)),
        ]
        for name, fn in attempts:
            try:
                fn()
                res = "SUCCEEDED (unexpected)"
            except OSError as e:
                res = f"{errno.errorcode[e.errno]}: {e.strerror}"
            add("write_attempt", label, name, "-", 1, detail=res)
        # one-block write through a write-opened fd, if it could be opened at all
        try:
            fd = os.open(vpath, os.O_WRONLY)
            os.pwrite(fd, b"\0" * BLK, 0)
            os.close(fd)
            res = "block write SUCCEEDED (unexpected)"
        except OSError as e:
            res = f"not reached: open failed {errno.errorcode[e.errno]}: {e.strerror}"
        add("write_attempt", label, "write one 4 KiB block at offset 0", "-", 1, detail=res)
        after = measure(vpath)
        add("write_attempt", label, "digest unchanged after attempts", "-", 1, detail=str(before == after))

        # ---- d) legitimate update: copy, change a block, fsync, enable, rename
        nb = (sz + BLK - 1) // BLK
        rng = int.from_bytes(os.urandom(8), "little")
        for cache in ("warm", "cold"):
            steps = {k: [] for k in ("copy", "write_block", "fsync", "enable", "rename", "dirsync", "total")}
            for k in range(3):
                blk = (rng + 7919 * k) % nb
                off = blk * BLK
                newb = os.urandom(min(BLK, sz - off))
                old_digest = measure(vpath)
                os.sync()
                if cache == "cold":
                    drop_caches()
                tmp = MNT / f"{fname}.new"
                t0 = time.perf_counter()
                shutil.copyfile(vpath, tmp)
                t1 = time.perf_counter()
                fd = os.open(tmp, os.O_WRONLY)
                os.pwrite(fd, newb, off)
                t2 = time.perf_counter()
                os.fsync(fd)
                os.close(fd)
                t3 = time.perf_counter()
                subprocess.run(["fsverity", "enable", "--hash-alg=sha256", "--block-size=4096", str(tmp)], check=True)
                t4 = time.perf_counter()
                os.rename(tmp, vpath)
                t5 = time.perf_counter()
                dfd = os.open(MNT, os.O_RDONLY)
                os.fsync(dfd)
                os.close(dfd)
                t6 = time.perf_counter()
                for key, (a, b) in zip(steps, [(t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t5, t6), (t0, t6)]):
                    steps[key].append((b - a) * 1e3)
                # correctness: new block is there, still verity, digest changed
                with open(vpath, "rb") as f:
                    f.seek(off)
                    assert f.read(len(newb)) == newb
                assert measure(vpath) != old_digest
            for key, v in steps.items():
                add("update", label, f"step:{key}" if key != "total" else "TOTAL_sequence", cache, 3, *stats(v),
                    detail=f"file_bytes={sz}")
        for g in MNT.glob(f"{fname}.*"):
            g.unlink()
        print(f"[{label}] done (verity digest {vdigest[:16]}...)", flush=True)
finally:
    subprocess.run(["umount", str(MNT)], capture_output=True)
    IMG.unlink(missing_ok=True)
    try:
        MNT.rmdir()
    except OSError:
        pass

# ---- e) our one-block update costs, from existing result files
def csvrows(p):
    return list(csv.DictReader(open(ROOT / "results" / p, newline="")))

for r in csvrows("e4_writepath.csv"):
    add("ours_update", r["model"], "in-memory leaf rehash + path (BLAKE3)", "-", 7,
        float(r["leaf_plus_path_us"]) / 1e3,
        detail="source=results/e4_writepath.csv:leaf_plus_path_us; no I/O")
for r in csvrows("r_crash_cost.csv"):
    if r["variant"] in ("journal_spec", "journal_strict", "nojournal_spec", "nojournal_strict"):
        add("ours_update", "mid", f"durable {r['variant']} (BLAKE3)", "-", int(r["samples"]),
            float(r["total_us_median"]) / 1e3, float(r["total_us_min"]) / 1e3, float(r["total_us_max"]) / 1e3,
            detail="source=results/r_crash_cost.csv:total_us_median; HDD, includes fsync; model measured=mid only")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "model", "variant", "cache", "trials", "median_ms", "min_ms", "max_ms", "detail"])
    w.writerows(rows)
if os.environ.get("SUDO_UID"):
    os.chown(OUT, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
print(f"wrote {OUT}")
