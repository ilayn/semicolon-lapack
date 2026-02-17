#!/usr/bin/env python
"""
Benchmark runner using pyperf for rigorous statistical comparison.

Usage:
    # Compare our dgetrf vs vendor Fortran at n=500, 50 iterations
    python run_bench.py dgetrf 500 50

    # Sweep across sizes
    python run_bench.py dgetrf --sweep

    # Compare saved results
    python run_bench.py --compare results/dgetrf_ours_500.json results/dgetrf_fortran_500.json

    # Just run ours (no comparison)
    python run_bench.py dgetrf 500 50 --ours-only

    # Custom build directory
    python run_bench.py dgetrf 500 50 --builddir ../builddir
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SWEEP_SIZES = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

DEFAULT_ITERS = {
    32: 500, 64: 500, 96: 500, 128: 500,
    192: 200, 256: 200, 384: 200, 512: 200,
    768: 100, 1024: 100,
    1536: 30, 2048: 30,
    3072: 10, 4096: 10,
}


def find_builddir():
    """Find the meson build directory."""
    script_dir = Path(__file__).parent
    candidates = [
        script_dir.parent / "builddir",
        script_dir.parent / "build",
        script_dir.parent / "builddir-release",
    ]
    for c in candidates:
        if (c / "build.ninja").exists():
            return c
    return None


def get_binary(builddir, routine, variant):
    """Get path to benchmark binary."""
    name = f"bench_{routine}"
    if variant != "ours":
        name += f"_{variant}"
    path = builddir / "benchmark" / name
    if not path.exists():
        return None
    return path


def iters_for_size(n):
    """Get iteration count for a given matrix size."""
    if n in DEFAULT_ITERS:
        return DEFAULT_ITERS[n]
    if n <= 128:
        return 500
    if n <= 512:
        return 200
    if n <= 1024:
        return 100
    if n <= 2048:
        return 30
    return 10


def run_pyperf(binary, n, iters, output_json, label=None):
    """Run pyperf command on a benchmark binary."""
    cmd = [
        sys.executable, "-m", "pyperf", "command",
        "--processes", "5",
        "--warmups", "1",
        "--values", "3",
        "-o", str(output_json),
    ]
    if label:
        cmd.extend(["--name", label])
    cmd.extend(["--", str(binary), str(n), str(iters)])

    print(f"  Running: {' '.join(str(c) for c in cmd[-4:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: pyperf failed:\n{result.stderr}", file=sys.stderr)
        return False
    # Print pyperf's summary line
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    return True


def run_compare(json_a, json_b):
    """Run pyperf compare_to on two result files."""
    cmd = [
        sys.executable, "-m", "pyperf", "compare_to",
        str(json_a), str(json_b),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)


def run_stats(json_file):
    """Print detailed statistics for a result file."""
    cmd = [sys.executable, "-m", "pyperf", "stats", str(json_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)


def bench_single(builddir, routine, n, iters, results_dir, ours_only=False):
    """Benchmark a single size, ours vs vendor Fortran."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Both benchmarks use the same --name so pyperf compare_to can pair them.
    bench_name = f"{routine}_n{n}"

    ours_bin = get_binary(builddir, routine, "ours")
    if ours_bin is None:
        print(f"ERROR: bench_{routine} not found in {builddir}/benchmark/")
        print(f"  Build with: meson configure {builddir} -Dbenchmarks=true && ninja -C {builddir}")
        return False

    ours_json = results_dir / f"{routine}_ours_{n}.json"
    print(f"\n[ours] bench_{routine} n={n} iters={iters}")
    if not run_pyperf(ours_bin, n, iters, ours_json, label=bench_name):
        return False

    if ours_only:
        print(f"\nDetailed stats:")
        run_stats(ours_json)
        return True

    fortran_bin = get_binary(builddir, routine, "fortran")
    if fortran_bin is None:
        print(f"WARNING: bench_{routine}_fortran not found, skipping comparison")
        return True

    fortran_json = results_dir / f"{routine}_fortran_{n}.json"
    print(f"\n[fortran] bench_{routine}_fortran n={n} iters={iters}")
    if not run_pyperf(fortran_bin, n, iters, fortran_json, label=bench_name):
        return False

    print(f"\n{'='*60}")
    print(f"Comparison: n={n}")
    print(f"{'='*60}")
    run_compare(ours_json, fortran_json)

    return True


def bench_sweep(builddir, routine, results_dir, ours_only=False):
    """Sweep across sizes."""
    print(f"Sweeping bench_{routine} across {len(SWEEP_SIZES)} sizes\n")

    ours_results = []
    fortran_results = []

    for n in SWEEP_SIZES:
        iters = iters_for_size(n)
        results_dir.mkdir(parents=True, exist_ok=True)

        ours_bin = get_binary(builddir, routine, "ours")
        if ours_bin is None:
            print(f"ERROR: bench_{routine} not found")
            return False

        bench_name = f"n{n}"
        ours_json = results_dir / f"{routine}_ours_{n}.json"
        print(f"[n={n:>5}] ours  ...", end="", flush=True)
        if run_pyperf(ours_bin, n, iters, ours_json, label=bench_name):
            ours_results.append((n, ours_json))
            print(" done")
        else:
            print(" FAILED")
            continue

        if not ours_only:
            fortran_bin = get_binary(builddir, routine, "fortran")
            if fortran_bin:
                fortran_json = results_dir / f"{routine}_fortran_{n}.json"
                print(f"[n={n:>5}] oblas ...", end="", flush=True)
                if run_pyperf(fortran_bin, n, iters, fortran_json, label=bench_name):
                    fortran_results.append((n, fortran_json))
                    print(" done")
                else:
                    print(" FAILED")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Summary: {routine}")
    print(f"{'='*70}")
    print(f"{'n':>6}  {'ours (ms)':>12}  {'fortran (ms)':>14}  {'ratio':>8}")
    print(f"{'---':>6}  {'--------':>12}  {'------------':>14}  {'-----':>8}")

    for n, ours_json in ours_results:
        ours_mean = _get_mean_ms(ours_json)
        fortran_mean = None
        for on, oj in fortran_results:
            if on == n:
                fortran_mean = _get_mean_ms(oj)
                break

        if fortran_mean is not None:
            ratio = ours_mean / fortran_mean
            print(f"{n:>6}  {ours_mean:>12.3f}  {fortran_mean:>14.3f}  {ratio:>8.2f}x")
        else:
            print(f"{n:>6}  {ours_mean:>12.3f}  {'N/A':>14}  {'N/A':>8}")

    return True


def _get_mean_ms(json_path):
    """Extract mean time in milliseconds from a pyperf JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    # pyperf stores values in seconds
    values = data["benchmarks"][0]["runs"]
    # Filter to only value runs (not warmup/calibration)
    times = []
    for run in values:
        if "values" in run:
            times.extend(run["values"])
    if not times:
        return 0.0
    return (sum(times) / len(times)) * 1000.0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner using pyperf for statistical comparison"
    )
    parser.add_argument("routine", nargs="?", default="dgetrf",
                        help="Routine to benchmark (default: dgetrf)")
    parser.add_argument("n", nargs="?", type=int, default=None,
                        help="Matrix size")
    parser.add_argument("iters", nargs="?", type=int, default=None,
                        help="Iterations per benchmark invocation")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep across standard sizes")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Compare two existing pyperf result files")
    parser.add_argument("--stats", metavar="FILE.json",
                        help="Show detailed stats for a result file")
    parser.add_argument("--ours-only", action="store_true",
                        help="Only benchmark our implementation")
    parser.add_argument("--builddir", type=str, default=None,
                        help="Path to meson build directory")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to store pyperf JSON results")

    args = parser.parse_args()

    # Handle --compare mode
    if args.compare:
        run_compare(Path(args.compare[0]), Path(args.compare[1]))
        return

    # Handle --stats mode
    if args.stats:
        run_stats(Path(args.stats))
        return

    # Find build directory
    if args.builddir:
        builddir = Path(args.builddir)
    else:
        builddir = find_builddir()
    if builddir is None or not builddir.exists():
        print("ERROR: Cannot find build directory.")
        print("  Specify with --builddir or run from project root with builddir/ present")
        sys.exit(1)

    # Results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent / "results"

    if args.sweep:
        bench_sweep(builddir, args.routine, results_dir, args.ours_only)
    else:
        n = args.n or 1000
        iters = args.iters or iters_for_size(n)
        bench_single(builddir, args.routine, n, iters, results_dir, args.ours_only)


if __name__ == "__main__":
    main()
