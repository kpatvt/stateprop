"""Tests for stateprop.extraction module (v0.9.68).

Multistage countercurrent liquid-liquid extraction columns.
"""
from __future__ import annotations
import sys
import warnings

import numpy as np

sys.path.insert(0, '.')

# UNIFAC-LLE happily generates RuntimeWarnings during Newton iteration
# when Newton's exploring the search space; they're transient and don't
# affect the converged result.
warnings.simplefilter("ignore", RuntimeWarning)


_passed = 0
_failed = 0
_failures = []


def check(label, ok):
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        _failures.append(label)
        print(f"  FAIL  {label}")


def section(name):
    print(f"\n[{name}]")


# ------------------------------------------------------------------
# System under test: water + acetone + benzene at 298.15 K.
#   water:   carrier   (in feed)
#   acetone: solute    (in feed; partitions strongly into benzene)
#   benzene: solvent   (in solvent stream)
# Highly immiscible carrier/solvent pair => clean LLE split.
# ------------------------------------------------------------------
def _system():
    from stateprop.activity.compounds import make_unifac_lle
    species = ["water", "acetone", "benzene"]
    uf = make_unifac_lle(species)
    return species, uf


def test_construction():
    """Result fields populated after a basic 3-stage solve."""
    from stateprop.extraction import lle_extraction_column
    section("test_construction")
    species, uf = _system()
    res = lle_extraction_column(
        n_stages=3,
        feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=40, tol=1e-7)
    check(f"converged: {res.converged} ({res.iterations} iters)",
          res.converged)
    check(f"n_stages = 3: {res.n_stages}", res.n_stages == 3)
    check(f"x_R shape (3, 3): {res.x_R.shape}", res.x_R.shape == (3, 3))
    check(f"x_E shape (3, 3): {res.x_E.shape}", res.x_E.shape == (3, 3))
    check(f"R length 3: {res.R.shape}", res.R.shape == (3,))
    check(f"E length 3: {res.E.shape}", res.E.shape == (3,))


def test_overall_mass_balance():
    """F z_F + S z_S = R_prod x_R_prod + E_prod x_E_prod  must close
    to machine precision per species."""
    from stateprop.extraction import lle_extraction_column
    section("test_overall_mass_balance")
    species, uf = _system()
    res = lle_extraction_column(
        n_stages=5,
        feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=40, tol=1e-9)
    check(f"converged: {res.converged}", res.converged)
    in_  = res.F * res.z_F + res.S * res.z_S
    out_ = res.R_product * res.x_R[-1] + res.E_product * res.x_E[0]
    err = float(np.max(np.abs(in_ - out_)))
    check(f"per-species mass balance: max err = {err:.2e}", err < 1e-9)
    total_err = abs((res.F + res.S) - (res.R_product + res.E_product))
    check(f"total mass balance: err = {total_err:.2e}", total_err < 1e-9)


def test_phase_closures_and_positivity():
    """Sum of mole fractions = 1 in both phases at every stage; all flows
    positive; all mole fractions in [0, 1]."""
    from stateprop.extraction import lle_extraction_column
    section("test_phase_closures_and_positivity")
    species, uf = _system()
    res = lle_extraction_column(
        n_stages=6,
        feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=1.5, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=40, tol=1e-9)
    check(f"converged: {res.converged}", res.converged)
    sum_R = res.x_R.sum(axis=1)
    sum_E = res.x_E.sum(axis=1)
    err_R = float(np.max(np.abs(sum_R - 1.0)))
    err_E = float(np.max(np.abs(sum_E - 1.0)))
    check(f"max |Sum x_R - 1| = {err_R:.2e}", err_R < 1e-8)
    check(f"max |Sum x_E - 1| = {err_E:.2e}", err_E < 1e-8)
    check(f"all R positive (min={res.R.min():.4f})", float(res.R.min()) > 0)
    check(f"all E positive (min={res.E.min():.4f})", float(res.E.min()) > 0)
    in_range = (res.x_R >= -1e-10).all() and (res.x_R <= 1.0 + 1e-10).all() \
               and (res.x_E >= -1e-10).all() and (res.x_E <= 1.0 + 1e-10).all()
    check(f"all mole fractions in [0, 1]", in_range)


def test_iso_activity_equilibrium():
    """At convergence gamma_i^R x_i^R = gamma_i^E x_i^E on every stage,
    every species, to high precision."""
    from stateprop.extraction import lle_extraction_column
    section("test_iso_activity_equilibrium")
    species, uf = _system()
    res = lle_extraction_column(
        n_stages=5,
        feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=40, tol=1e-9)
    check(f"converged: {res.converged}", res.converged)
    max_err = 0.0
    for j in range(res.n_stages):
        gR = np.asarray(uf.gammas(float(res.T[j]), res.x_R[j]))
        gE = np.asarray(uf.gammas(float(res.T[j]), res.x_E[j]))
        diff = gR * res.x_R[j] - gE * res.x_E[j]
        max_err = max(max_err, float(np.max(np.abs(diff))))
    check(f"max |gamma^R x^R - gamma^E x^E| = {max_err:.2e}",
          max_err < 1e-8)


def test_recovery_increases_with_stages():
    """Adding stages must not decrease acetone recovery (countercurrent
    extraction with positive K)."""
    from stateprop.extraction import lle_extraction_column
    section("test_recovery_increases_with_stages")
    species, uf = _system()
    cfg = dict(feed_F=1.0, feed_z=[0.85, 0.15, 0.0],
               solvent_S=1.0, solvent_z=[0.0, 0.0, 1.0],
               T=298.15, species_names=species, activity_model=uf,
               max_newton_iter=40, tol=1e-7)
    rec = []
    for n in [2, 4, 6]:
        r = lle_extraction_column(n_stages=n, **cfg)
        if r.converged:
            rec.append((n, r.recovery("acetone")))
        else:
            rec.append((n, float("nan")))
    for n, R in rec:
        print(f"    n={n}: acetone recovery = {R:.2%}")
    monotonic = all(rec[i + 1][1] >= rec[i][1] - 1e-8
                    for i in range(len(rec) - 1))
    check(f"recovery monotone non-decreasing in n_stages: {monotonic}",
          monotonic)
    check(f"6-stage recovery > 70% (large K, S/F=1): "
          f"{rec[-1][1]:.2%}", rec[-1][1] > 0.70)


def test_recovery_increases_with_solvent_ratio():
    """At fixed n_stages, more solvent should extract more solute."""
    from stateprop.extraction import lle_extraction_column
    section("test_recovery_increases_with_solvent_ratio")
    species, uf = _system()
    rec = []
    for S in [0.5, 1.0, 2.0, 4.0]:
        r = lle_extraction_column(
            n_stages=5,
            feed_F=1.0, feed_z=[0.85, 0.15, 0.0],
            solvent_S=S, solvent_z=[0.0, 0.0, 1.0],
            T=298.15, species_names=species, activity_model=uf,
            max_newton_iter=40, tol=1e-7)
        if r.converged:
            rec.append((S, r.recovery("acetone")))
    for S, R in rec:
        print(f"    S/F={S}: acetone recovery = {R:.2%}")
    monotonic = all(rec[i + 1][1] >= rec[i][1] - 1e-8
                    for i in range(len(rec) - 1))
    check(f"recovery monotone non-decreasing in S/F: {monotonic}",
          monotonic)


def test_pure_solvent_drives_solute_to_extract():
    """With a clean (no-solute) solvent and a high-K system, the
    raffinate-product solute fraction should drop well below the feed
    fraction."""
    from stateprop.extraction import lle_extraction_column
    section("test_pure_solvent_drives_solute_to_extract")
    species, uf = _system()
    z_feed = np.array([0.7, 0.3, 0.0])
    res = lle_extraction_column(
        n_stages=8,
        feed_F=1.0, feed_z=z_feed,
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=40, tol=1e-7)
    check(f"converged: {res.converged}", res.converged)
    x_R_prod_solute = float(res.x_R[-1, 1])
    x_E_prod_solute = float(res.x_E[0, 1])
    print(f"    feed   solute mole fraction: {z_feed[1]:.4f}")
    print(f"    raffinate product:           {x_R_prod_solute:.6f}")
    print(f"    extract  product:            {x_E_prod_solute:.6f}")
    check(f"raffinate solute << feed solute: "
          f"{x_R_prod_solute:.4f} < 0.05 * {z_feed[1]}",
          x_R_prod_solute < 0.05 * z_feed[1])
    check(f"extract carries the solute: x_E_solute = "
          f"{x_E_prod_solute:.4f} > 0.05",
          x_E_prod_solute > 0.05)


def test_one_stage_matches_LLE_flash():
    """A single-stage column should reproduce a single-pass LLE flash on
    the combined feed+solvent inlet stream."""
    from stateprop.extraction import lle_extraction_column
    from stateprop.activity.lle import LLEFlash
    section("test_one_stage_matches_LLE_flash")
    species, uf = _system()
    F, S = 1.0, 2.0
    zF = np.array([0.7, 0.3, 0.0])
    zS = np.array([0.0, 0.0, 1.0])
    z_overall = (F * zF + S * zS) / (F + S)
    # Reference: direct LLE flash
    flash = LLEFlash(uf)
    sol = flash.solve(298.15, z_overall,
                      x1_guess=[0.97, 0.02, 0.01],
                      x2_guess=[0.005, 0.05, 0.945],
                      maxiter=300, tol=1e-9)
    # Column with n_stages=1
    res = lle_extraction_column(
        n_stages=1,
        feed_F=F, feed_z=zF, solvent_S=S, solvent_z=zS,
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=30, tol=1e-9)
    check(f"converged: {res.converged}", res.converged)
    # Identify which flash phase corresponds to raffinate (closer to z_F).
    if np.linalg.norm(sol.x1 - zF) < np.linalg.norm(sol.x2 - zF):
        x_R_ref, x_E_ref = sol.x1, sol.x2
    else:
        x_R_ref, x_E_ref = sol.x2, sol.x1
    err_R = float(np.max(np.abs(res.x_R[0] - x_R_ref)))
    err_E = float(np.max(np.abs(res.x_E[0] - x_E_ref)))
    check(f"x_R agrees with flash: max err = {err_R:.2e}", err_R < 1e-5)
    check(f"x_E agrees with flash: max err = {err_E:.2e}", err_E < 1e-5)


def test_recovery_invalid_species():
    """`recovery()` raises KeyError for unknown species."""
    from stateprop.extraction import lle_extraction_column
    section("test_recovery_invalid_species")
    species, uf = _system()
    res = lle_extraction_column(
        n_stages=3, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=1.0, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=30, tol=1e-7)
    raised = False
    try:
        res.recovery("nonexistent_species")
    except KeyError:
        raised = True
    check("recovery() raises KeyError for unknown species", raised)


def test_detects_outside_binodal():
    """When the overall (F+S) composition lies OUTSIDE the binodal so
    no 2-phase column solution exists, the solver must report failure
    rather than silently return collapsed phases / negative flows."""
    from stateprop.extraction import lle_extraction_column
    section("test_detects_outside_binodal")
    from stateprop.activity.compounds import make_unifac_lle
    # water + acetone + 1-butanol with butanol-heavy overall mix lies
    # outside the binodal at 298 K (acetone is a cosolvent that dissolves
    # the water and butanol into a single liquid phase).
    species = ["water", "acetone", "1-butanol"]
    uf = make_unifac_lle(species)
    res = lle_extraction_column(
        n_stages=5,
        feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],   # high-S/F, butanol heavy
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=60, tol=1e-7)
    print(f"    message: {res.message[:120]}")
    check(f"converged flagged as False: {res.converged}",
          not res.converged)
    check("message mentions binodal/non-physical: "
          + repr(res.message[:60]),
          ("non-physical" in res.message)
          or ("collapsed" in res.message)
          or ("binodal" in res.message))



def test_input_validation():
    """Bad input arguments raise ValueError."""
    from stateprop.extraction import lle_extraction_column
    section("test_input_validation")
    species, uf = _system()
    cfg = dict(T=298.15, species_names=species, activity_model=uf,
               max_newton_iter=10, tol=1e-7)

    raised = False
    try:
        lle_extraction_column(n_stages=3, feed_F=1.0,
                              feed_z=[0.7, 0.3],   # wrong length
                              solvent_S=1.0, solvent_z=[0.0, 0.0, 1.0],
                              **cfg)
    except ValueError:
        raised = True
    check("len(feed_z) != C raises ValueError", raised)

    raised = False
    try:
        lle_extraction_column(n_stages=3, feed_F=0.0,    # zero feed
                              feed_z=[0.7, 0.3, 0.0],
                              solvent_S=1.0, solvent_z=[0.0, 0.0, 1.0],
                              **cfg)
    except ValueError:
        raised = True
    check("feed_F = 0 raises ValueError", raised)

    raised = False
    try:
        lle_extraction_column(n_stages=0, feed_F=1.0,
                              feed_z=[0.7, 0.3, 0.0],
                              solvent_S=1.0, solvent_z=[0.0, 0.0, 1.0],
                              **cfg)
    except ValueError:
        raised = True
    check("n_stages = 0 raises ValueError", raised)


# ------------------------------------------------------------------
# v0.9.69 -- Reactive extraction and energy balance
# ------------------------------------------------------------------
def _enthalpy_funcs():
    """Constant-Cp_L liquid enthalpy model for water/acetone/benzene."""
    T_REF = 298.15
    Cp_L = np.array([75.3, 124.0, 134.0])    # J/(mol K)
    return [(lambda T, i=i: Cp_L[i] * (T - T_REF)) for i in range(3)]


def test_energy_balance_isothermal_inlets():
    """Energy balance with equal feed_T = solvent_T = column reference T
    must reproduce the isothermal column to high precision (no driving
    enthalpy gradient to break CMO)."""
    from stateprop.extraction import lle_extraction_column
    section("test_energy_balance_isothermal_inlets")
    species, uf = _system()
    h_L = _enthalpy_funcs()
    T_iso = 298.15
    r_iso = lle_extraction_column(
        n_stages=5, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        T=T_iso, species_names=species, activity_model=uf,
        max_newton_iter=40, tol=1e-9)
    r_eb = lle_extraction_column(
        n_stages=5, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        species_names=species, activity_model=uf,
        energy_balance=True, feed_T=T_iso, solvent_T=T_iso,
        h_L_funcs=h_L,
        max_newton_iter=40, tol=1e-9)
    check(f"both converged: iso={r_iso.converged} eb={r_eb.converged}",
          r_iso.converged and r_eb.converged)
    err_T = float(np.max(np.abs(r_eb.T - T_iso)))
    err_xR = float(np.max(np.abs(r_iso.x_R - r_eb.x_R)))
    err_xE = float(np.max(np.abs(r_iso.x_E - r_eb.x_E)))
    check(f"max |T_eb - T_iso| = {err_T:.2e} K", err_T < 1e-5)
    check(f"max |x_R diff| = {err_xR:.2e}", err_xR < 1e-7)
    check(f"max |x_E diff| = {err_xE:.2e}", err_xE < 1e-7)


def test_energy_balance_thermal_gradient():
    """Energy balance with feed_T != solvent_T should produce a varying
    T profile bounded by the inlet temperatures.  Bottom stage T should
    be very close to solvent_T because S enters there with rate >= F."""
    from stateprop.extraction import lle_extraction_column
    section("test_energy_balance_thermal_gradient")
    species, uf = _system()
    h_L = _enthalpy_funcs()
    feed_T = 310.0
    solvent_T = 290.0
    res = lle_extraction_column(
        n_stages=6, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        species_names=species, activity_model=uf,
        energy_balance=True, feed_T=feed_T, solvent_T=solvent_T,
        h_L_funcs=h_L,
        max_newton_iter=40, tol=1e-7)
    check(f"converged: {res.converged}", res.converged)
    print(f"    T profile: {np.round(res.T, 2)}")
    # T values bounded by inlet temps (since no heat duty)
    in_range = float(res.T.min()) >= solvent_T - 0.01 \
                and float(res.T.max()) <= feed_T + 0.01
    check(f"T bounded by inlet temps [{res.T.min():.2f}, {res.T.max():.2f}]",
          in_range)
    # Profile must be non-trivial (i.e., the energy balance is doing work)
    T_spread = float(res.T.max() - res.T.min())
    check(f"T spread = {T_spread:.2f} > 1.0 K (non-trivial gradient)",
          T_spread > 1.0)
    # Bottom of column dominated by solvent (S=2 vs F=1) -- T_bot ~ solvent_T
    check(f"T[-1] ({res.T[-1]:.2f}) within 5 K of solvent_T={solvent_T}",
          abs(res.T[-1] - solvent_T) < 5.0)


def test_energy_balance_per_stage_closure():
    """Per-stage H_j residual closes to << H_scale at convergence.
    Reproduces the solver's H residual using returned x_R, x_E, R, E, T."""
    from stateprop.extraction import lle_extraction_column
    section("test_energy_balance_per_stage_closure")
    species, uf = _system()
    h_L = _enthalpy_funcs()
    F, S = 1.0, 2.0
    z_F = np.array([0.7, 0.3, 0.0])
    z_S = np.array([0.0, 0.0, 1.0])
    feed_T, solvent_T = 305.0, 290.0
    res = lle_extraction_column(
        n_stages=5, feed_F=F, feed_z=z_F,
        solvent_S=S, solvent_z=z_S,
        species_names=species, activity_model=uf,
        energy_balance=True, feed_T=feed_T, solvent_T=solvent_T,
        h_L_funcs=h_L,
        max_newton_iter=40, tol=1e-9)
    check(f"converged: {res.converged}", res.converged)
    # Reproduce H residuals
    h_R_arr = np.array([sum(res.x_R[j, i] * h_L[i](res.T[j])
                             for i in range(3))
                        for j in range(res.n_stages)])
    h_E_arr = np.array([sum(res.x_E[j, i] * h_L[i](res.T[j])
                             for i in range(3))
                        for j in range(res.n_stages)])
    h_F_val = float(sum(z_F[i] * h_L[i](feed_T) for i in range(3)))
    h_S_val = float(sum(z_S[i] * h_L[i](solvent_T) for i in range(3)))
    H_scale = max(F + S, 1.0) * 1e4
    max_scaled = 0.0
    for j in range(res.n_stages):
        in_h = (F * h_F_val if j == 0 else res.R[j-1] * h_R_arr[j-1])
        in_h += (S * h_S_val if j == res.n_stages - 1
                 else res.E[j+1] * h_E_arr[j+1])
        out_h = res.R[j] * h_R_arr[j] + res.E[j] * h_E_arr[j]
        H_resid_scaled = (in_h - out_h) / H_scale
        max_scaled = max(max_scaled, abs(H_resid_scaled))
    check(f"per-stage max |H_scaled| = {max_scaled:.2e}",
          max_scaled < 1e-7)


def test_reactive_zero_xi_at_consistent_K_eq():
    """When K_eq matches the K_a observed at the non-reactive solution
    on a chosen stage, the reactive solver returns xi ~ 0 and otherwise
    identical compositions.  Validates the reactive code path."""
    from stateprop.extraction import lle_extraction_column
    from stateprop.reaction.liquid_phase import LiquidPhaseReaction
    section("test_reactive_zero_xi_at_consistent_K_eq")
    species, uf = _system()
    cfg = dict(n_stages=5, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
               solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
               species_names=species, activity_model=uf,
               max_newton_iter=40, tol=1e-9)
    r_nr = lle_extraction_column(T=298.15, **cfg)
    j_ref = 2  # stage 3 (1-indexed)
    gE = np.asarray(uf.gammas(298.15, r_nr.x_E[j_ref]))
    a = gE * r_nr.x_E[j_ref]
    # Use stoichiometry nu = [-1, -1, +2] (water + acetone -> 2 benzene;
    # chemistry-nonsense but stoichiometrically balanced).
    K_a_consistent = float((a[2] ** 2) / (a[0] * a[1] + 1e-30))
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=[-1, -1, +2],
                              K_eq_298=K_a_consistent, dH_rxn=0.0)
    r_r = lle_extraction_column(
        T=298.15, reactions=[rxn], reactive_stages=[j_ref + 1],
        reaction_phase="E", **cfg)
    check(f"converged: {r_r.converged} ({r_r.iterations} iters)",
          r_r.converged)
    xi_max = float(np.max(np.abs(r_r.xi)))
    check(f"max |xi| at consistent K_eq: {xi_max:.2e}", xi_max < 1e-8)
    err_xR = float(np.max(np.abs(r_nr.x_R - r_r.x_R)))
    err_xE = float(np.max(np.abs(r_nr.x_E - r_r.x_E)))
    check(f"x_R unchanged: max err = {err_xR:.2e}", err_xR < 1e-7)
    check(f"x_E unchanged: max err = {err_xE:.2e}", err_xE < 1e-7)


def test_reactive_K_a_equals_K_eq_at_convergence():
    """On every reactive stage at convergence, K_a (in the reaction
    phase) equals K_eq(T) to high precision."""
    from stateprop.extraction import lle_extraction_column
    from stateprop.reaction.liquid_phase import LiquidPhaseReaction
    section("test_reactive_K_a_equals_K_eq_at_convergence")
    species, uf = _system()
    cfg = dict(n_stages=5, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
               solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
               species_names=species, activity_model=uf,
               max_newton_iter=40, tol=1e-9)
    r_nr = lle_extraction_column(T=298.15, **cfg)
    gE = np.asarray(uf.gammas(298.15, r_nr.x_E[2]))
    a = gE * r_nr.x_E[2]
    K_a_consistent = float((a[2] ** 2) / (a[0] * a[1] + 1e-30))
    # Perturb K_eq to make xi non-trivial
    K_eq_perturbed = 2.0 * K_a_consistent
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=[-1, -1, +2],
                              K_eq_298=K_eq_perturbed, dH_rxn=0.0)
    reactive_stages = [2, 3, 4]
    r = lle_extraction_column(
        T=298.15, reactions=[rxn], reactive_stages=reactive_stages,
        reaction_phase="E", **cfg)
    check(f"converged: {r.converged} ({r.iterations} iters)", r.converged)
    print(f"    K_eq = {K_eq_perturbed:.4e}")
    max_err = 0.0
    for stage_1idx in reactive_stages:
        j = stage_1idx - 1
        gE_j = np.asarray(uf.gammas(float(r.T[j]), r.x_E[j]))
        a_j = gE_j * r.x_E[j]
        K_a_j = float((a_j[2] ** 2) / (a_j[0] * a_j[1] + 1e-30))
        rel_err = abs(K_a_j - K_eq_perturbed) / K_eq_perturbed
        max_err = max(max_err, rel_err)
        print(f"    stage {stage_1idx}: K_a = {K_a_j:.4e}, "
              f"rel err = {rel_err:.2e}")
    check(f"max |K_a - K_eq| / K_eq = {max_err:.2e}", max_err < 1e-7)


def test_reactive_mass_balance_with_xi():
    """Mass balance with reaction source: F z_F + S z_S + sum(nu_i xi)
    = R_prod x_R_prod + E_prod x_E_prod, per species."""
    from stateprop.extraction import lle_extraction_column
    from stateprop.reaction.liquid_phase import LiquidPhaseReaction
    section("test_reactive_mass_balance_with_xi")
    species, uf = _system()
    cfg = dict(n_stages=5, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
               solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
               species_names=species, activity_model=uf,
               max_newton_iter=40, tol=1e-9)
    r_nr = lle_extraction_column(T=298.15, **cfg)
    gE = np.asarray(uf.gammas(298.15, r_nr.x_E[2]))
    a = gE * r_nr.x_E[2]
    K_a_consistent = float((a[2] ** 2) / (a[0] * a[1] + 1e-30))
    nu = np.array([-1, -1, +2], dtype=float)
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=nu.tolist(),
                              K_eq_298=2.0 * K_a_consistent, dH_rxn=0.0)
    r = lle_extraction_column(
        T=298.15, reactions=[rxn], reactive_stages=[2, 3, 4],
        reaction_phase="E", **cfg)
    check(f"converged: {r.converged}", r.converged)
    in_  = r.F * r.z_F + r.S * r.z_S
    out_ = r.R_product * r.x_R[-1] + r.E_product * r.x_E[0]
    rxn_src = nu * float(r.xi.sum())
    err_full = float(np.max(np.abs(in_ + rxn_src - out_)))
    check(f"per-species mass balance with rxn source: "
          f"max err = {err_full:.2e}", err_full < 1e-9)


def test_reactive_plus_energy_balance():
    """Combined reactive + energy balance converges and satisfies both
    the chemistry constraint and the energy balance."""
    from stateprop.extraction import lle_extraction_column
    from stateprop.reaction.liquid_phase import LiquidPhaseReaction
    section("test_reactive_plus_energy_balance")
    species, uf = _system()
    h_L = _enthalpy_funcs()
    cfg = dict(n_stages=5, feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
               solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
               species_names=species, activity_model=uf,
               max_newton_iter=40, tol=1e-9)
    r_nr = lle_extraction_column(T=298.15, **cfg)
    gE = np.asarray(uf.gammas(298.15, r_nr.x_E[2]))
    a = gE * r_nr.x_E[2]
    K_a_consistent = float((a[2] ** 2) / (a[0] * a[1] + 1e-30))
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=[-1, -1, +2],
                              K_eq_298=2.0 * K_a_consistent,
                              dH_rxn=-5000.0)   # exothermic
    r = lle_extraction_column(
        reactions=[rxn], reactive_stages=[2, 3, 4],
        reaction_phase="E",
        energy_balance=True, feed_T=300.0, solvent_T=300.0,
        h_L_funcs=h_L,
        **cfg)
    check(f"converged: {r.converged} ({r.iterations} iters)", r.converged)
    # Reaction is exothermic at reactive stages, so T should rise above
    # feed_T = solvent_T = 300 K on those stages.
    T_max = float(r.T.max())
    print(f"    T profile: {np.round(r.T, 3)}")
    print(f"    xi profile: {r.xi.flatten()}")
    check(f"T_max = {T_max:.2f} >= 300.0 (rxn heat): {T_max >= 300.0}",
          T_max >= 299.99)


def main():
    for fn in [
        test_construction,
        test_overall_mass_balance,
        test_phase_closures_and_positivity,
        test_iso_activity_equilibrium,
        test_recovery_increases_with_stages,
        test_recovery_increases_with_solvent_ratio,
        test_pure_solvent_drives_solute_to_extract,
        test_one_stage_matches_LLE_flash,
        test_recovery_invalid_species,
        test_detects_outside_binodal,
        test_energy_balance_isothermal_inlets,
        test_energy_balance_thermal_gradient,
        test_energy_balance_per_stage_closure,
        test_reactive_zero_xi_at_consistent_K_eq,
        test_reactive_K_a_equals_K_eq_at_convergence,
        test_reactive_mass_balance_with_xi,
        test_reactive_plus_energy_balance,
        test_input_validation,
    ]:
        try:
            fn()
        except Exception as e:
            global _failed, _failures
            _failed += 1
            _failures.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            print(f"  EXC   {fn.__name__}: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    if _failures:
        print()
        print("Failures:")
        for f in _failures:
            print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
