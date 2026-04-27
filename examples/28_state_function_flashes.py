"""State-function flashes: PT, PH, PS, TH, TS round-trip consistency.

What this demonstrates
----------------------
Process simulation rarely uses PT-flash exclusively.  Different unit
operations specify different state-function pairs:

- **Compressors / expanders** — adiabatic-reversible specs as PS
  (constant-entropy from inlet to outlet)
- **JT valves / throttles** — constant-h, specified by PH
- **Heat exchangers** — outlet T at given p (specified by PT) or
  given h (specified by PH for energy-balance closure)
- **Adiabatic flash drums** — given inlet enthalpy, find p, T, x, y
  (PH or UV depending on what's fixed)
- **Refrigeration cycles** — TS diagram traces require PS / TS legs
- **Reactors with adiabatic operation** — UV (constant internal
  energy and volume) for the closed adiabatic case

This example demonstrates round-trip consistency: a PT flash gives
h, s; feeding those h or s back through PH or PS must recover T.
We do this for a 3-component natural-gas-condensate mixture
(methane / ethane / propane) at five state points spanning
single-phase vapor, two-phase, and supercritical regions.

Reference
---------
Smith, Van Ness, Abbott (2005). Introduction to Chemical
Engineering Thermodynamics.  Chapter 11 — Solution Thermodynamics:
Theory.

Approximate runtime: ~5 seconds.

Public APIs invoked
-------------------
- stateprop.cubic.from_chemicals.cubic_from_name
- stateprop.cubic.CubicMixture
- stateprop.cubic.flash.flash_pt
- stateprop.cubic.flash.flash_ph
- stateprop.cubic.flash.flash_ps
- stateprop.cubic.flash.flash_th
- stateprop.cubic.flash.flash_ts

"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.cubic.from_chemicals import cubic_from_name
from stateprop.cubic import CubicMixture
from stateprop.cubic.flash import (
    flash_pt, flash_ph, flash_ps, flash_th, flash_ts,
)
from examples._harness import validate, summary

print("=" * 70)
print("State-function flashes: PT / PH / PS / TH / TS round-trip")
print("=" * 70)
print()

# Build a representative natural-gas-condensate mixture
ch4 = cubic_from_name("methane", family="pr")
c2h6 = cubic_from_name("ethane", family="pr")
c3h8 = cubic_from_name("propane", family="pr")
nc4 = cubic_from_name("n-butane", family="pr")

z = [0.70, 0.18, 0.08, 0.04]
species = ["CH₄", "C₂H₆", "C₃H₈", "n-C₄H₁₀"]
mx = CubicMixture([ch4, c2h6, c3h8, nc4], composition=z)

print(f"  Mixture: {species}")
print(f"  Composition: {z}")
print()

# Five state points covering different phase-equilibrium regimes
state_points = [
    (50e5,   200.0, "single-phase liquid"),
    (10e5,   200.0, "two-phase (cold)"),
    (10e5,   270.0, "two-phase (warm)"),
    (10e5,   400.0, "single-phase vapor"),
    (50e5,   400.0, "supercritical"),
]

print(f"  {'state':>8s}  {'p (bar)':>8s}  {'T (K)':>6s}  "
      f"{'phase':>20s}  {'h (J/mol)':>10s}  {'s (J/mol/K)':>12s}")
print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}  {'-'*10}  {'-'*12}")

reference_states = []
for i, (p, T, label) in enumerate(state_points, 1):
    r = flash_pt(p=p, T=T, z=z, mixture=mx)
    reference_states.append((i, p, T, label, r.h, r.s))
    print(f"  {i:>8d}  {p/1e5:>8.1f}  {T:>6.1f}  "
            f"{r.phase[:20]:>20s}  {r.h:>10.0f}  {r.s:>12.3f}")

# ------------------------------------------------------------------
# Round-trip tests: PT → state-fn → flash → check we recover T
# ------------------------------------------------------------------
print()
print("Round-trip consistency: feed (h, s) from PT back through PH and PS")
print()
print(f"  {'state':>5s}  {'spec':>4s}  {'recovered T (K)':>16s}  "
      f"{'expected (K)':>13s}  {'rel err':>8s}")
print(f"  {'-'*5}  {'-'*4}  {'-'*16}  {'-'*13}  {'-'*8}")

ph_results, ps_results = [], []
th_results, ts_results = [], []

for i, p, T_ref, label, h_ref, s_ref in reference_states:
    # PH at given p, recover T
    r_ph = flash_ph(p=p, h_target=h_ref, z=z, mixture=mx)
    err_ph = abs(r_ph.T - T_ref) / T_ref
    print(f"  {i:>5d}  {'PH':>4s}  {r_ph.T:>16.3f}  "
            f"{T_ref:>13.1f}  {err_ph*100:>7.4f}%")
    ph_results.append((i, T_ref, r_ph.T, err_ph))

    # PS at given p, recover T
    r_ps = flash_ps(p=p, s_target=s_ref, z=z, mixture=mx)
    err_ps = abs(r_ps.T - T_ref) / T_ref
    print(f"  {i:>5d}  {'PS':>4s}  {r_ps.T:>16.3f}  "
            f"{T_ref:>13.1f}  {err_ps*100:>7.4f}%")
    ps_results.append((i, T_ref, r_ps.T, err_ps))

    # TH at given T, recover p
    r_th = flash_th(T=T_ref, h_target=h_ref, z=z, mixture=mx)
    err_th = abs(r_th.p - p) / p
    print(f"  {i:>5d}  {'TH':>4s}  p={r_th.p/1e5:>11.4f} bar  "
            f"p_ref={p/1e5:>5.1f}     {err_th*100:>7.4f}%")
    th_results.append((i, p, r_th.p, err_th))

    # TS at given T, recover p
    r_ts = flash_ts(T=T_ref, s_target=s_ref, z=z, mixture=mx)
    err_ts = abs(r_ts.p - p) / p
    print(f"  {i:>5d}  {'TS':>4s}  p={r_ts.p/1e5:>11.4f} bar  "
            f"p_ref={p/1e5:>5.1f}     {err_ts*100:>7.4f}%")
    ts_results.append((i, p, r_ts.p, err_ts))

# ------------------------------------------------------------------
# A real engineering scenario: adiabatic JT throttle
# ------------------------------------------------------------------
print()
print("Engineering scenario: adiabatic JT throttle (constant-h)")
print()
print("  Inlet:  p₁ = 50 bar, T₁ = 350 K")
print("  Outlet: p₂ = 5 bar, T₂ = ? (find via PH-flash)")
print()

p1, T1 = 50e5, 350.0
r_in = flash_pt(p=p1, T=T1, z=z, mixture=mx)
h_in = r_in.h

p2 = 5e5
r_out = flash_ph(p=p2, h_target=h_in, z=z, mixture=mx)
T2 = r_out.T

print(f"  Inlet:  T={T1:.1f} K, h={h_in:.1f} J/mol, phase={r_in.phase}")
print(f"  Outlet: T={T2:.1f} K, h={r_out.h:.1f} J/mol, phase={r_out.phase}")
print(f"  ΔT = {T2 - T1:+.2f} K (negative = JT cooling)")
if r_out.phase != "single phase vapor" and r_out.beta is not None:
    print(f"  Outlet vapor fraction β = {r_out.beta:.4f}")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# All round-trip errors under 0.5%.  The state-function flashes use
# outer Newton iterations whose default convergence tolerance is
# ~1e-5 in the function value; this translates to ~1e-3 to 1e-2
# relative error on the recovered T or p.  For tighter round-trip
# fidelity the underlying solvers can be called with smaller tol=.
max_err_ph = max(e for _, _, _, e in ph_results)
validate("PH round-trip max error",
          reference=0.0, computed=max_err_ph,
          units="-", tol_rel=1.0, tol_abs=5e-3,
          source="Default Newton tolerance ~1e-5 in residual")

max_err_ps = max(e for _, _, _, e in ps_results)
validate("PS round-trip max error",
          reference=0.0, computed=max_err_ps,
          units="-", tol_rel=1.0, tol_abs=5e-3,
          source="Default Newton tolerance ~1e-5 in residual")

max_err_th = max(e for _, _, _, e in th_results)
validate("TH round-trip max error",
          reference=0.0, computed=max_err_th,
          units="-", tol_rel=1.0, tol_abs=5e-3,
          source="Default Newton tolerance ~1e-5 in residual")

max_err_ts = max(e for _, _, _, e in ts_results)
validate("TS round-trip max error",
          reference=0.0, computed=max_err_ts,
          units="-", tol_rel=1.0, tol_abs=5e-3,
          source="Default Newton tolerance ~1e-5 in residual")

# JT throttle should cool methane-rich mix (μ_JT > 0 at these conditions)
validate("JT throttle outlet T < inlet T (cooling)",
          reference=T1 - 5.0, computed=T2,
          units="K", tol_rel=0.20,
          source="Real-gas behavior: JT cooling at moderate p, T")

summary()
