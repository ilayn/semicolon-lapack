# Benchmarking and Profiling

This directory measures how our LAPACK routines perform against the vendor LAPACK and helps identify where time is spent. There are three layers, each serving a different purpose:

1. **C benchmark binaries** — the actual workloads. Each binary exercises one routine (ours or the vendor's) with configurable matrix sizes and iteration counts.
2. **pyperf** — wraps the binaries and provides statistically rigorous timing with calibration, process isolation, and outlier detection.
3. **samply** — a sampling profiler that produces interactive flame graphs in your browser, for understanding *where* time is spent.

You can use each layer independently. The C binaries work on their own for quick checks; pyperf adds statistical rigor; samply adds profiling depth.


## Building the Benchmarks

Benchmarks are not built by default. Enable them with the `-Dbenchmarks=true` option:

```bash
meson setup builddir -Dbenchmarks=true
ninja -C builddir
```

If you already have a builddir, pass the flag when reconfiguring:

```bash
meson setup builddir --reconfigure -Dbenchmarks=true
```

This produces three executables per routine under `builddir/benchmark/`:

| Binary | What it calls |
|--------|--------------|
| `bench_dgetrf` | Our `dgetrf()` via `semicolon_lapack_double.h` |
| `bench_dgetrf_fortran` | The vendor `dgetrf_()` via the Fortran LAPACK symbol |
| `bench_dgetrf_lapacke` | The vendor `LAPACKE_dgetrf()` via the LAPACKE C interface |

The Fortran and LAPACKE binaries call the standard interfaces, so they work with whatever BLAS/LAPACK library is linked (OpenBLAS, MKL, BLIS, etc.).

All three generate the same matrix (`srand(42)`, diagonally dominant), so the comparison is apples-to-apples. The only difference is which `dgetrf` implementation gets called.

Note: "Fortran" here refers to the calling convention (`dgetrf_` with trailing underscore, pass-by-pointer arguments), not the implementation language. Some vendors (e.g. OpenBLAS) implement these routines in C but expose them through the Fortran symbol name for compatibility. OpenBLAS replaces the following LAPACK routines with optimized C implementations that call architecture-specific kernels directly, bypassing the standard BLAS interface:

- **LU**: `gesv`, `getf2`, `getrf`, `getrs`, `laswp`
- **Cholesky**: `potf2`, `potrf`, `potri`
- **Triangular**: `lauu2`, `lauum`, `trti2`, `trtri`, `trtrs`
- **Other**: `laed3`

All other LAPACK routines (QR, symmetric indefinite, eigenvalue, SVD, etc.) fall through to the reference Fortran implementation. When benchmarking against these non-rewritten routines, the comparison is our C against the vendor's bundled Fortran — a more direct test of translation quality.


## Layer 1: C Benchmark Binaries

The vendor library is a pre-built release binary (typically `-O3`) that we cannot change. Our library is built at whatever optimization level the builddir is configured with — the default is `release` (`-O3`), so the comparison is fair. The benchmark harness files themselves are additionally compiled with `-fno-omit-frame-pointer` to support profiling, but this does not affect the library code being measured.

### Single size

```bash
./builddir/benchmark/bench_dgetrf 1000 100
```

This runs 3 warmup iterations (untimed), then 100 timed iterations of LU factorization on a 1000x1000 matrix. Each iteration is timed individually with a cache flush between iterations, and the output reports both the minimum and median times:

```
Benchmarking dgetrf: m=1000, n=1000, iters=100 (warmup=3)
Min time:    12.340 ms  (54.12 GFLOP/s)
Median time: 12.890 ms  (51.86 GFLOP/s)
```

GFLOP/s are computed using precise LAWN 41 FLOP formulas (see `bench_flops.h`).

### Rectangular matrices

Pass three positional arguments for non-square matrices:

```bash
./builddir/benchmark/bench_dgeqrf 2000 200 50     # m=2000, n=200, 50 iterations
```

With one or two positional arguments, the old convention is preserved: `bench 1000 100` means n=1000, iters=100 (square).

### Size sweep

```bash
./builddir/benchmark/bench_dgetrf --sweep
```

Runs 42 sizes (35 square from n=4 to n=4000, plus tall-skinny and short-wide rectangular shapes) with automatically chosen iteration counts (more iterations for small sizes, fewer for large). Prints a table:

```
m      n      iters   min(ms)   med(ms)  min(GF/s)  med(GF/s)
---    ---    -----   -------   -------  ---------  ---------
32     32      5000     0.010     0.012      10.44       8.70
64     64      3000     0.040     0.042      20.42      19.50
...
4000   4000      10   350.000   376.139     122.00     113.43
1000   100     500      0.680     0.710      22.50      21.55
```

### CSV output

Add `--csv` to any invocation for machine-readable output, suitable for plotting or further analysis:

```bash
./builddir/benchmark/bench_dgetrf --sweep --csv > ours.csv
./builddir/benchmark/bench_dgetrf_fortran --sweep --csv > vendor.csv
```

### Correctness check (QR only)

The `bench_dgeqrf` and `bench_dgeqrf_fortran` binaries support a `--check` flag that verifies the factorization after benchmarking:

```bash
./builddir/benchmark/bench_dgeqrf 1000 50 --check
```

This computes `||A - Q*R|| / (max(m,n) * ||A|| * eps)` and reports PASS/FAIL to stderr.


## Layer 2: Statistical Benchmarking with pyperf

The C binaries give you one timing number. That number varies between runs due to OS scheduling, CPU frequency scaling, cache state, and other noise. [pyperf](https://pyperf.readthedocs.io/) addresses this by running the binary multiple times across multiple processes, computing mean, median, standard deviation, and interquartile range, and flagging outliers.

### Install

```bash
conda install -c conda-forge pyperf  # or: pip install pyperf
```

### Compare ours vs vendor at a single size

```bash
python benchmark/run_bench.py dgetrf 500 50
```

This runs both `bench_dgetrf` and `bench_dgetrf_fortran` through pyperf (5 processes, 3 values each), saves JSON result files to `benchmark/results/`, and prints a statistical comparison showing whether the difference is significant.

### Sweep across sizes

```bash
python benchmark/run_bench.py dgetrf --sweep
```

Runs all sizes for both implementations. At the end, prints a summary table:

```
     n     ours (ms)  fortran (ms)     ratio
   ---     --------   ------------     -----
    32        0.002          0.002      ~1x
   ...
  4000      376.139        358.236      1.05x
```

### Working with saved results

pyperf saves everything as JSON. You can re-analyze without re-running:

```bash
# Compare two saved results
python benchmark/run_bench.py --compare benchmark/results/dgetrf_ours_500.json \
                                       benchmark/results/dgetrf_fortran_500.json

# Detailed statistics (mean, median, std dev, IQR, outliers)
python benchmark/run_bench.py --stats benchmark/results/dgetrf_ours_500.json
```

### Other options

```bash
# Only benchmark our implementation (skip vendor comparison)
python benchmark/run_bench.py dgetrf 500 50 --ours-only

# Custom build directory
python benchmark/run_bench.py dgetrf 500 50 --builddir path/to/builddir
```


## Layer 3: Profiling with samply

Benchmarking tells you *how fast*. Profiling tells you *why*. [samply](https://github.com/mstange/samply) is a sampling profiler that records stack traces while your program runs, then opens the Firefox Profiler UI in your browser for interactive analysis.

### Install

```bash
conda install -c conda-forge samply  # or: cargo install samply
```

### Kernel settings (one-time setup)

Both samply and perf need two kernel settings that Fedora restricts by default. Without these you'll get "Permission denied" or "mmap failed" errors. The instructions below are tested on Fedora; other distributions may have different defaults (Ubuntu and Arch typically ship with `perf_event_paranoid=1` or lower already).

**Permanent** (create once, survives reboots):

```bash
sudo tee /etc/sysctl.d/99-perf.conf << 'EOF'
kernel.perf_event_paranoid=1
kernel.perf_event_mlock_kb=2048
EOF
sudo sysctl --system
```

**Or temporary** (resets on reboot):

```bash
sudo sysctl kernel.perf_event_paranoid=1
sudo sysctl kernel.perf_event_mlock_kb=2048
```

What these do:
- `perf_event_paranoid=1` allows non-root users to read performance counters for their own processes. This is safe for development workstations — it does not expose other users' data or allow system-wide tracing (that would require `perf_event_paranoid=-1`). Fedora defaults to `2` which blocks even per-process profiling.
- `perf_event_mlock_kb=2048` raises the per-user locked memory limit for the perf ring buffer from 516 KB to 2 MB. This only affects perf event buffers, not general memory limits. samply needs this to allocate its sampling buffer.

Verify:

```bash
cat /proc/sys/kernel/perf_event_paranoid   # should be 1
cat /proc/sys/kernel/perf_event_mlock_kb   # should be 2048
```

### Profile a benchmark

```bash
samply record ./builddir/benchmark/bench_dgetrf 1000 50
```

Your browser opens automatically. The Firefox Profiler UI lets you:

- **Flame graph**: see which functions consume the most time, with full call stacks
- **Call tree**: hierarchical breakdown of time per function
- **Timeline**: sample distribution over the program's execution
- **Search/filter**: find specific functions by name

To compare implementations, profile each one separately and look at where they differ:

```bash
samply record ./builddir/benchmark/bench_dgetrf 1000 50
# inspect in browser, close when done

samply record ./builddir/benchmark/bench_dgetrf_fortran 1000 50
# inspect the vendor profile for comparison
```

For single-threaded profiling (removes vendor threading noise):

```bash
OPENBLAS_NUM_THREADS=1 samply record ./builddir/benchmark/bench_dgetrf 1000 200
```

Note: the thread-count environment variable is vendor-specific. Use `OPENBLAS_NUM_THREADS` for OpenBLAS, `MKL_NUM_THREADS` for MKL, `BLIS_NUM_THREADS` for BLIS.

### What to look for

When comparing our implementation against the vendor:

- **Panel factorization cost** — How much time is in `dgetrf2` (unblocked) vs the Level 3 BLAS calls (`cblas_dgemm`, `cblas_dtrsm`)?
- **CBLAS dispatch overhead** — Our code calls `cblas_dgemv`, `cblas_dscal` etc. which have parameter validation and dispatch logic. Vendor libraries may call internal kernels directly with near-zero overhead. This matters most for small matrices.
- **Memory traffic** — Time in `memcpy` (matrix restoration in the benchmark loop) relative to actual computation.
- **Call depth** — Deep call stacks (our `dgetrf` → `dgetrf2` → CBLAS wrappers → vendor kernels) vs flat hot loops (vendor's internal path).

## Troubleshooting

### Missing symbols in profiles

Vendor BLAS libraries installed via conda are release builds without debug symbols, so some internal functions will appear as hex addresses (e.g. `0x7f01c11c0a8ff`). This is expected and unavoidable without building the vendor library from source with `-g`. Our own function names will always be visible since our library is compiled with debug info in the benchmark targets.

### Incomplete call stacks

The benchmark `meson.build` compiles with `-fno-omit-frame-pointer` so that profilers can unwind the stack. If stacks still look truncated (especially through optimized BLAS code), this is typically due to vendor libraries being compiled without frame pointers.
