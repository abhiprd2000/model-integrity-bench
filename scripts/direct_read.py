#!/usr/bin/env python3
"""Sequential O_DIRECT read throughput of one file (substitute for
`dd iflag=direct`; the installed dd is uutils and rejects iflag=direct).
Output goes to stdout only; nothing is written to results/."""
import mmap, os, sys, time
path = sys.argv[1]; bs = 1 << 20; trials = int(sys.argv[2]) if len(sys.argv) > 2 else 3
size = os.path.getsize(path)
buf = mmap.mmap(-1, bs)          # page-aligned buffer for O_DIRECT
for t in range(trials):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    n = 0; t0 = time.perf_counter()
    while True:
        r = os.readv(fd, [buf])
        if r == 0: break
        n += r
    dt = time.perf_counter() - t0
    os.close(fd)
    assert n == size, (n, size)
    print(f"{path}: {n} B in {dt*1e3:.1f} ms = {n/dt/1e6:.1f} MB/s (O_DIRECT, 1 MiB reads)")
