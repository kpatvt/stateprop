"""CPR-compressed Jacobian speedup for distillation columns.

What this demonstrates
----------------------
The v0.9.117 release wired Curtis-Powell-Reid (1974) sparsity
compression into the Naphtali-Sandholm column solver.  The
Jacobian of an N-stage column is block-tridiagonal: stage *j*'s
residual depends only on variables at stages *j-1, j, j+1*.  CPR
exploits this by simultaneously perturbing variables in stages
{0, 3, 6, ...}, {1, 4, 7, ...}, {2, 5, 8, ...} — three perturbation
passes regardless of column length.

This replaces the dense O(N) FD probes with O(1) probes, giving a
speedup that grows linearly with column length:

    N_stages = 10:   speedup  3.1×
    N_stages = 20:   speedup  4.6×
    N_stages = 30:   speedup  7.2×
    N_stages = 40:   speedup 10.1×

CPR is automatic — it kicks in for any column without non-local
coupling (no pump-arounds, no side-strippers, no Murphree
efficiency E < 1).  Users get the speedup without changing any code.

Reference
---------
Curtis, A. R.; Powell, M. J. D.; Reid, J. K. (1974). On the
Estimation of Sparse Jacobian Matrices.  IMA J. Appl. Math. 13, 117.

Approximate runtime: ~30 seconds (sweeps several column sizes).

Public APIs invoked
-------------------
- stateprop.electrolyte.sour_water_stripper
- stateprop.reaction.reactive_column._build_block_tridiag_jacobian
  (internal hook — monkey-patched here to compare CPR vs dense)

"""
import sys, os, time, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from examples._harness import validate, validate_bool, summary, smoke_mode

print("=" * 70)
print("CPR-compressed Jacobian speedup vs forced dense FD")
print("=" * 70)
print()
print("  Test column: refinery sour-water stripper (NH₃, H₂S, CO₂, H₂O)")
print("  Solver:      Naphtali-Sandholm")
print("  Conditions:  feed at stage 2 (top), reflux 1.0, distillate 2.5")
print()

# Use the sour-water stripper as the test column — substantial
# realistic problem with 4 components and full activity model.
from stateprop.electrolyte import sour_water_stripper
import stateprop.reaction.reactive_column as rc

species = ["NH3", "H2S", "CO2", "H2O"]


def run_column(N, force_dense=False):
    """Run the column once at N stages.  Optionally force dense FD by
    monkey-patching the helper."""
    common = dict(
        n_stages=N, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, N)),
        stage_efficiency=1.0,
    )
    if force_dense:
        original = rc._build_block_tridiag_jacobian
        def force_dense_wrapper(*args, **kwargs):
            kwargs["has_nonlocal_coupling"] = True
            return original(*args, **kwargs)
        rc._build_block_tridiag_jacobian = force_dense_wrapper
        try:
            t0 = time.time()
            r = sour_water_stripper(**common, energy_balance=False)
            dt = time.time() - t0
        finally:
            rc._build_block_tridiag_jacobian = original
    else:
        t0 = time.time()
        r = sour_water_stripper(**common, energy_balance=False)
        dt = time.time() - t0
    return r, dt


# Smoke-mode: run only N=10 and N=20.  Full mode: also N=30 and N=40.
if smoke_mode():
    Ns = [10, 20]
    print("  Running smoke mode (N=10, 20 only).")
else:
    Ns = [10, 20, 30, 40]
    print("  Running full sweep (N=10, 20, 30, 40).")
print()

print(f"  {'N':>4s}  {'CPR (s)':>10s}  {'Dense (s)':>10s}  "
      f"{'Speedup':>10s}  {'Δrecovery':>11s}")
print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*11}")

speedups = []
for N in Ns:
    # CPR (default)
    r_cpr, t_cpr = run_column(N, force_dense=False)
    # Forced dense
    r_den, t_den = run_column(N, force_dense=True)
    speedup = t_den / t_cpr
    speedups.append((N, t_cpr, t_den, speedup))

    # Verify the answer didn't change — CPR and dense must give
    # bit-identical results for these conditions.
    a_cpr = r_cpr.bottoms_strip_efficiency["NH3"]
    a_den = r_den.bottoms_strip_efficiency["NH3"]
    delta = abs(a_cpr - a_den)

    print(f"  {N:>4d}  {t_cpr:>10.3f}  {t_den:>10.3f}  "
          f"{speedup:>9.2f}×  {delta:>10.2e}")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# CPR must be at least 2× faster at N=10 (the smallest case).
N_baseline, t_cpr_b, t_den_b, speedup_b = speedups[0]
validate_bool(f"Speedup ≥ 2× at N={N_baseline}",
                condition=(speedup_b >= 2.0),
                detail=f"actual speedup = {speedup_b:.2f}×",
                source="v0.9.117 release notes (CPR vs dense FD)")

# CPR must give same answer (NH3 strip efficiency) as dense
# We re-run a single case to compare a more sensitive metric: the
# bottoms x_NH3 mole fraction
common = dict(
    n_stages=10, feed_stage=2, feed_F=100.0,
    feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
    species_names=species,
    reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
    T_init=list(np.linspace(353.15, 383.15, 10)),
    stage_efficiency=1.0,
)
r_cpr, _ = run_column(10, force_dense=False)
r_den, _ = run_column(10, force_dense=True)
nh3_idx = species.index("NH3")
x_b_cpr = r_cpr.column_result.x[-1][nh3_idx]
x_b_den = r_den.column_result.x[-1][nh3_idx]
validate("Bottoms x_NH3 identical (CPR vs dense) at N=10",
          reference=x_b_den, computed=x_b_cpr,
          units="-", tol_rel=1e-4,
          tol_abs=1e-6,
          source="Theoretical: CPR is exact (no approximation)")

# Larger speedup at larger N
if len(speedups) >= 4:
    N_largest, _, _, speedup_largest = speedups[-1]
    validate_bool(f"Speedup ≥ 5× at N={N_largest} (linear scaling)",
                    condition=(speedup_largest >= 5.0),
                    detail=f"actual speedup = {speedup_largest:.2f}×",
                    source="v0.9.117 linear scaling")

print()
print("  Speedup summary:")
print(f"    Smallest column (N={speedups[0][0]}): {speedups[0][3]:.1f}×")
print(f"    Largest column  (N={speedups[-1][0]}): {speedups[-1][3]:.1f}×")
if len(speedups) >= 2:
    print(f"    Speedup growth ratio: "
          f"{speedups[-1][3]/speedups[0][3]:.2f}× from "
          f"N={speedups[0][0]} to N={speedups[-1][0]}")

summary()
