"""
Benchmark for stateprop.

Measures per-call evaluation time for the key kernels on CO2.
If Numba is installed, the JIT-compiled code is used automatically.
Otherwise the pure-Python fallback runs (so you can compare the two
by running this script twice: once with and once without Numba).

Usage:
    python examples/99_benchmark.py            # single pass, prints a table
    python examples/99_benchmark.py --warm     # include a warm-up JIT phase

Typical speedups on a modern laptop (Numba vs pure Python):
    alpha_r + derivatives      ~80x
    pressure                   ~80x
    full property kernel       ~60x

Absolute numbers (CO2, 39-term residual + 8 ideal terms, Numba enabled):
    alpha_r derivatives        ~1 us / call
    pressure                   ~1 us / call
    full property kernel       ~2 us / call
"""
import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from stateprop import load_fluid
from stateprop.core import alpha_r_derivs, alpha_0_derivs, alpha_derivs
from stateprop.properties import _pressure_kernel, _all_props_kernel


def _time(func, n=10000):
    """Run `func()` `n` times and return seconds per call."""
    t0 = time.perf_counter()
    for _ in range(n):
        func()
    t1 = time.perf_counter()
    return (t1 - t0) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", action="store_true",
                        help="Warm up the JIT cache first (recommended if Numba is on)")
    parser.add_argument("-n", "--niter", type=int, default=50000,
                        help="Iterations per measurement (default 50000)")
    args = parser.parse_args()

    # Detect numba
    try:
        import numba
        numba_version = numba.__version__
    except ImportError:
        numba_version = None

    print("=" * 60)
    print("stateprop benchmark")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Numba:  {numba_version or 'not installed (pure-Python fallback)'}")
    print()

    # Load CO2 and pack once
    fl = load_fluid("carbondioxide")
    print(f"Fluid: {fl}")
    print()

    # Prepare args
    pack = fl.pack()
    R, rho_c, T_c = pack[0], pack[1], pack[2]
    # New pack layout (28 items total):
    #   [0:3]   scalars (R, rho_c, T_c)
    #   [3:25]  residual arrays (3 poly + 4 exp + 7 gauss + 8 nonanalytic)
    #   [25:28] ideal arrays (codes, a, b)
    res_args = pack[3:25]    # residual arrays
    ideal_args = pack[25:]   # ideal arrays

    # A test state point
    T = 320.0
    rho = 12000.0
    delta = rho / rho_c
    tau = T_c / T

    # Warm-up: one call to each to trigger JIT compilation
    if args.warm:
        print("Warming up JIT cache ...")
        alpha_r_derivs(delta, tau, *res_args)
        alpha_0_derivs(delta, tau, *ideal_args)
        alpha_derivs(delta, tau, *res_args, *ideal_args)
        _pressure_kernel(rho, T, R, rho_c, T_c, *res_args)
        _all_props_kernel(rho, T, R, rho_c, T_c, *res_args, *ideal_args)
        print("Done.")
        print()

    n = args.niter

    # Bind closures so the tight loop avoids argument-tuple construction overhead
    def f_alpha_r():
        alpha_r_derivs(delta, tau, *res_args)

    def f_alpha_0():
        alpha_0_derivs(delta, tau, *ideal_args)

    def f_alpha():
        alpha_derivs(delta, tau, *res_args, *ideal_args)

    def f_pressure():
        _pressure_kernel(rho, T, R, rho_c, T_c, *res_args)

    def f_all_props():
        _all_props_kernel(rho, T, R, rho_c, T_c, *res_args, *ideal_args)

    print(f"Running {n} iterations of each kernel ...")
    print()
    results = {
        "alpha_r + 5 derivs":   _time(f_alpha_r,  n),
        "alpha_0 + 5 derivs":   _time(f_alpha_0,  n),
        "alpha + 5 derivs":     _time(f_alpha,    n),
        "pressure":             _time(f_pressure, n),
        "all properties":       _time(f_all_props, n),
    }

    print(f"{'kernel':<28} {'time/call':>15} {'calls/sec':>15}")
    print("-" * 60)
    for name, t in results.items():
        us_per_call = t * 1e6
        calls_per_sec = 1.0 / t
        print(f"{name:<28} {us_per_call:>11.3f} us  {calls_per_sec:>11.2e}")

    print()

    # Vectorized benchmark: property evaluation at N state points via the
    # public API. This is what typical user code does.
    print("Vectorized (Python-for-loop inside wrapper):")
    import stateprop as h
    N = 10000
    rho_arr = np.linspace(1.0, 15000.0, N)
    T_arr = np.full(N, 320.0)

    t0 = time.perf_counter()
    p = h.pressure(rho_arr, T_arr, fl)
    t1 = time.perf_counter()
    dt = (t1 - t0) / N
    print(f"   pressure (N={N}):           {dt*1e6:.3f} us/point  ({N/(t1-t0):.2e}/s)")

    t0 = time.perf_counter()
    cp = h.cp(rho_arr, T_arr, fl)
    t1 = time.perf_counter()
    dt = (t1 - t0) / N
    print(f"   cp       (N={N}):           {dt*1e6:.3f} us/point  ({N/(t1-t0):.2e}/s)")

    t0 = time.perf_counter()
    w = h.speed_of_sound(rho_arr, T_arr, fl)
    t1 = time.perf_counter()
    dt = (t1 - t0) / N
    print(f"   sound    (N={N}):           {dt*1e6:.3f} us/point  ({N/(t1-t0):.2e}/s)")

    print()
    if numba_version is None:
        print("NOTE: Numba is not installed. Install it with `pip install numba` to")
        print("      get an expected 50-100x speedup on these kernels.")


if __name__ == "__main__":
    main()
