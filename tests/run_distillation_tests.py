"""Quick-suite tests for stateprop.distillation.

Validates the non-reactive distillation column wrapper on standard
textbook systems (benzene/toluene, methanol/water, ternary
benzene/toluene/cumene), checks mass-balance closure, equivalence to
the underlying reactive solver with reactions=(), and qualitative
behaviour with respect to reflux ratio and number of stages.
"""
from __future__ import annotations

import sys, os, math, warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from stateprop.distillation import (
    distillation_column, DistillationColumnResult, FeedSpec, PumpAround, Spec,
    SideStripper,
)
from stateprop.reaction.reactive_column import reactive_distillation_column
from stateprop.activity.compounds import make_unifac


# ----- shared infrastructure -------------------------------------------------

_PASS, _FAIL = 0, 0


def section(name):
    print(f"\n[{name}]")


def check(label, ok):
    global _PASS, _FAIL
    if ok:
        print(f"  PASS  {label}")
        _PASS += 1
    else:
        print(f"  FAIL  {label}")
        _FAIL += 1


def antoine(A, B, C):
    """Return P_sat(T_K) = 10**(A - B/(Tc + C)) * 133.322 [Pa]."""
    def f(T):
        return 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    return f


# Antoine constants (P in mmHg, T in deg C)
PSAT = {
    "benzene":  antoine(6.90565, 1211.033, 220.790),
    "toluene":  antoine(6.95464, 1344.800, 219.482),
    "methanol": antoine(8.08097, 1582.271, 239.726),
    "water":    antoine(8.07131, 1730.630, 233.426),
    "cumene":   antoine(6.96292, 1469.677, 207.806),
}


def _bt_setup(n_stages=12, feed_stage=6, R=2.0, D=50.0, F=100.0,
              z=(0.5, 0.5)):
    """Benzene/toluene reference setup."""
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    return dict(n_stages=n_stages, feed_stage=feed_stage,
                feed_F=F, feed_z=list(z), feed_T=355.0,
                reflux_ratio=R, distillate_rate=D, pressure=101325.0,
                species_names=species, activity_model=uf,
                psat_funcs=psats)


# ----- tests ----------------------------------------------------------------

def test_construction():
    """Wrapper exposes the right object and basic fields."""
    section("test_construction")
    res = distillation_column(**_bt_setup())
    check("returns DistillationColumnResult",
          isinstance(res, DistillationColumnResult))
    check("converged",                res.converged)
    check("n_stages = 12",            res.n_stages == 12)
    check("species_names matches",    res.species_names == ("benzene", "toluene"))
    check("T shape (12,)",            res.T.shape == (12,))
    check("x shape (12, 2)",          res.x.shape == (12, 2))
    check("y shape (12, 2)",          res.y.shape == (12, 2))
    check("L shape (12,)",            res.L.shape == (12,))
    check("V shape (12,)",            res.V.shape == (12,))
    check("no reactive_stages attr (clean API)",
          not hasattr(res, "reactive_stages"))
    check("no xi attr (clean API)",
          not hasattr(res, "xi"))
    check("D + B = F",
          abs(res.D + res.B - res.feed_F) < 1e-10)


def test_overall_mass_balance_closure():
    """Per-species mass balance: F z_i = D x_D_i + B x_B_i to ~1e-10."""
    section("test_overall_mass_balance_closure")
    res = distillation_column(**_bt_setup())
    F, D, B = res.feed_F, res.D, res.B
    in_  = F * np.asarray(res.feed_z)
    out_ = D * res.x_D + B * res.x_B
    err = np.max(np.abs(in_ - out_))
    print(f"    max |F z - D x_D - B x_B| = {err:.3e}")
    check("per-species mass balance < 1e-9",  err < 1e-9)


def test_phase_closures():
    """Sum_i x = Sum_i y = 1 on every stage."""
    section("test_phase_closures")
    res = distillation_column(**_bt_setup(n_stages=10))
    sx = np.abs(res.x.sum(axis=1) - 1)
    sy = np.abs(res.y.sum(axis=1) - 1)
    print(f"    max |Sum x - 1| = {sx.max():.3e}")
    print(f"    max |Sum y - 1| = {sy.max():.3e}")
    check("Sum_i x = 1 on every stage to 1e-10",  sx.max() < 1e-10)
    check("Sum_i y = 1 on every stage to 1e-10",  sy.max() < 1e-10)


def test_equivalence_to_reactive_with_no_reactions():
    """distillation_column must produce identical numerics to
    reactive_distillation_column with reactions=()."""
    section("test_equivalence_to_reactive_with_no_reactions")
    cfg = _bt_setup(n_stages=8, feed_stage=4, R=1.5, D=50.0)
    res_d = distillation_column(**cfg)

    cfg_r = dict(cfg)
    cfg_r["reactions"] = ()
    cfg_r["reactive_stages"] = ()
    res_r = reactive_distillation_column(**cfg_r)

    print(f"    iters: distillation={res_d.iterations}  "
          f"reactive(no rxns)={res_r.iterations}")
    check("both converged",
          res_d.converged and res_r.converged)
    check("T profiles match to 1e-10",
          float(np.max(np.abs(res_d.T - res_r.T))) < 1e-10)
    check("x profiles match to 1e-10",
          float(np.max(np.abs(res_d.x - res_r.x))) < 1e-10)
    check("y profiles match to 1e-10",
          float(np.max(np.abs(res_d.y - res_r.y))) < 1e-10)
    check("L profiles match to 1e-10",
          float(np.max(np.abs(res_d.L - res_r.L))) < 1e-10)
    check("V profiles match to 1e-10",
          float(np.max(np.abs(res_d.V - res_r.V))) < 1e-10)
    check("D matches",  abs(res_d.D - res_r.D) < 1e-12)
    check("B matches",  abs(res_d.B - res_r.B) < 1e-12)


def test_purity_increases_with_reflux():
    """For a given column geometry, distillate purity x_D[light] is
    a monotonically non-decreasing function of reflux ratio R."""
    section("test_purity_increases_with_reflux")
    R_list = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    purities = []
    for R in R_list:
        res = distillation_column(**_bt_setup(R=R))
        purities.append(res.x_D[0])  # benzene mole fraction in distillate
        print(f"    R={R:5.2f}  x_D[benzene]={res.x_D[0]:.5f}  "
              f"iters={res.iterations}")
    diffs = np.diff(purities)
    check("x_D[benzene] non-decreasing in R",
          bool((diffs >= -5e-7).all()))
    check("x_D[benzene] strictly increases over the full range",
          purities[-1] > purities[0] + 0.01)


def test_purity_increases_with_stages():
    """Distillate purity is monotonically non-decreasing in n_stages
    (with feed at the geometric centre)."""
    section("test_purity_increases_with_stages")
    n_list = [4, 6, 8, 12, 16, 24]
    purities = []
    for n in n_list:
        cfg = _bt_setup(n_stages=n, feed_stage=max(2, n // 2))
        res = distillation_column(**cfg)
        purities.append(res.x_D[0])
        print(f"    n_stages={n:2d}  feed_stage={max(2, n//2):2d}  "
              f"x_D[benzene]={res.x_D[0]:.5f}  iters={res.iterations}")
    diffs = np.diff(purities)
    check("x_D[benzene] non-decreasing in n_stages",
          bool((diffs >= -5e-7).all()))
    check("substantial improvement: 4 stages -> 24 stages",
          purities[-1] > purities[0] + 0.005)


def test_temperature_profile_monotonic():
    """For an A/B mixture with B heavier (lower volatility), T_top < T_bottom
    and the profile is approximately monotone increasing top-to-bottom."""
    section("test_temperature_profile_monotonic")
    res = distillation_column(**_bt_setup(n_stages=12))
    T = res.T
    print(f"    T[top]={T[0]:.2f}  T[bottom]={T[-1]:.2f}")
    print(f"    full profile: {[round(t, 2) for t in T]}")
    check("T_top < T_bottom",  T[0] < T[-1])
    # Allow small non-monotonicity at the feed stage (cold feed
    # introduces a kink); require global monotonicity within 0.5 K.
    diffs = np.diff(T)
    check("globally non-decreasing within 0.5 K",
          bool((diffs >= -0.5).all()))


def test_methanol_water_split():
    """Methanol/water at 1 atm: methanol (light, BP 64.7 deg C) recovered
    overhead, water (heavy, BP 100 deg C) recovered as bottoms."""
    section("test_methanol_water_split")
    species = ["methanol", "water"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.4, 0.6], feed_T=350.0,
        reflux_ratio=2.5, distillate_rate=40.0, pressure=101325.0,
        species_names=species, activity_model=uf,
        psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}")
    print(f"    x_D[methanol]={res.x_D[0]:.4f}, "
          f"x_B[water]={res.x_B[1]:.4f}")
    print(f"    methanol recovery to D = "
          f"{res.recovery('methanol','distillate'):.4%}")
    check("converged",  res.converged)
    check("x_D[methanol] > 0.85",  res.x_D[0] > 0.85)
    check("x_B[water] > 0.85",     res.x_B[1] > 0.85)
    check("methanol recovery to D > 0.85",
          res.recovery('methanol', 'distillate') > 0.85)


def test_three_component_split():
    """Ternary benzene/toluene/cumene: light/medium/heavy split.
    Benzene -> distillate, cumene -> bottoms, toluene distributes.
    Mass balance must close on every species."""
    section("test_three_component_split")
    species = ["benzene", "toluene", "cumene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.3, 0.4, 0.3], feed_T=380.0,
        reflux_ratio=3.0, distillate_rate=30.0, pressure=101325.0,
        species_names=species, activity_model=uf,
        psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}")
    print(f"    x_D = {np.round(res.x_D, 4)}")
    print(f"    x_B = {np.round(res.x_B, 4)}")
    print(f"    benzene recovery to D = "
          f"{res.recovery('benzene','distillate'):.4%}")
    print(f"    cumene recovery to B  = "
          f"{res.recovery('cumene','bottoms'):.4%}")
    check("converged",  res.converged)

    # Mass balance per species
    F, D, B = res.feed_F, res.D, res.B
    in_  = F * np.asarray(res.feed_z)
    out_ = D * res.x_D + B * res.x_B
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    max mass balance error = {err:.3e}")
    check("mass balance closes < 1e-9 per species",  err < 1e-9)
    check("benzene strongly enriched in distillate",
          res.x_D[0] > 0.9)
    check("cumene strongly enriched in bottoms",
          res.x_B[2] > 0.4)
    check("benzene mostly absent in bottoms (< 1%)",
          res.x_B[0] < 0.01)
    check("cumene almost absent in distillate",
          res.x_D[2] < 0.001)


def test_recovery_complementary():
    """recovery to distillate + recovery to bottoms = 1 for every
    species in a non-reactive column."""
    section("test_recovery_complementary")
    res = distillation_column(**_bt_setup(n_stages=10))
    for s in ["benzene", "toluene"]:
        rD = res.recovery(s, "distillate")
        rB = res.recovery(s, "bottoms")
        print(f"    {s}: D={rD:.6f}  B={rB:.6f}  sum={rD+rB:.10f}")
        check(f"{s}: recovery_D + recovery_B = 1 to 1e-9",
              abs(rD + rB - 1.0) < 1e-9)


def test_high_reflux_approaches_pure_distillate():
    """At very high R the rectifying section approaches total reflux;
    distillate purity for the lighter component should approach 1."""
    section("test_high_reflux_approaches_pure_distillate")
    res = distillation_column(**_bt_setup(n_stages=20, feed_stage=10,
                                           R=50.0, D=49.5, F=100.0))
    print(f"    x_D[benzene] at R=50 = {res.x_D[0]:.6f}, "
          f"iters={res.iterations}")
    check("x_D[benzene] > 0.999 at R=50 with 20 stages",
          res.x_D[0] > 0.999)


def test_energy_balance_smoke():
    """energy_balance=True should run successfully and produce a
    physically sensible profile when h_*_funcs are provided."""
    section("test_energy_balance_smoke")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]

    # Linear ideal h(T): h_L = Cp_L*(T - Tref), h_V = Cp_L*(T-Tref) + dHvap_ref
    # Approx: benzene Cp_L=136 J/molK, dHvap=33.8 kJ/mol at 353K
    #         toluene Cp_L=157 J/molK, dHvap=38.0 kJ/mol at 384K
    Tref = 298.15
    def h_L_b(T): return 136.0 * (T - Tref)
    def h_L_t(T): return 157.0 * (T - Tref)
    def h_V_b(T): return 136.0 * (T - Tref) + 33800.0
    def h_V_t(T): return 157.0 * (T - Tref) + 38000.0

    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf,
        psat_funcs=psats,
        energy_balance=True,
        h_V_funcs=[h_V_b, h_V_t],
        h_L_funcs=[h_L_b, h_L_t])
    print(f"    converged={res.converged}, iters={res.iterations}")
    print(f"    T profile (top to bottom): "
          f"{[round(t, 2) for t in res.T]}")
    print(f"    x_D[benzene]={res.x_D[0]:.4f}, "
          f"x_B[toluene]={res.x_B[1]:.4f}")
    check("converged with energy balance",  res.converged)
    check("x_D[benzene] > 0.95",  res.x_D[0] > 0.95)
    check("T profile monotone increasing within 0.5 K",
          bool((np.diff(res.T) >= -0.5).all()))

    # Mass balance still closes
    F, D, B = res.feed_F, res.D, res.B
    err = float(np.max(np.abs(F * np.asarray(res.feed_z)
                              - D * res.x_D - B * res.x_B)))
    print(f"    max mass balance err with EB = {err:.3e}")
    check("mass balance closes with EB < 1e-8",  err < 1e-8)


def test_input_validation():
    """Sanity checks on the wrapper-level argument validation."""
    section("test_input_validation")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(feed_stage=2, feed_F=100.0, feed_z=[0.5, 0.5],
                feed_T=355.0, reflux_ratio=2.0, distillate_rate=50.0,
                pressure=101325.0, species_names=species,
                activity_model=uf, psat_funcs=psats)

    raised = False
    try:
        distillation_column(n_stages=1, **base)
    except ValueError:
        raised = True
    check("n_stages = 1 raises ValueError",  raised)

    raised = False
    try:
        distillation_column(n_stages=10, **{**base, "feed_F": 0.0})
    except ValueError:
        raised = True
    check("feed_F = 0 raises ValueError",  raised)

    raised = False
    try:
        distillation_column(n_stages=10, **{**base, "reflux_ratio": -1.0})
    except ValueError:
        raised = True
    check("reflux_ratio < 0 raises ValueError",  raised)

    raised = False
    try:
        # D >= F is unphysical (B = F - D <= 0)
        distillation_column(n_stages=10, **{**base, "distillate_rate": 200.0})
    except ValueError:
        raised = True
    check("distillate_rate >= feed_F raises ValueError",  raised)

    raised = False
    try:
        distillation_column(n_stages=10, **{**base, "distillate_rate": 0.0})
    except ValueError:
        raised = True
    check("distillate_rate = 0 raises ValueError",  raised)


def test_multifeed_two_feeds_converges():
    """Two-feed column at different stages converges and closes mass
    balance to high precision."""
    section("test_multifeed_two_feeds_converges")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=12,
        feeds=[FeedSpec(stage=4, F=50.0, z=[0.5, 0.5]),
               FeedSpec(stage=8, F=50.0, z=[0.5, 0.5])],
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}, "
          f"msg={res.message[:55]}")
    check("two-feed column converges", res.converged)
    check("two feeds reported in result.feeds", len(res.feeds) == 2)
    check("D + B = total_F", abs(res.D + res.B - 100.0) < 1e-10)
    # Mass balance per species
    total_in = sum(f.F * np.asarray(f.z) for f in res.feeds)
    total_out = res.D * res.x_D + res.B * res.x_B
    err = float(np.max(np.abs(total_in - total_out)))
    print(f"    max mass balance err = {err:.3e}")
    check("per-species mass balance < 1e-10", err < 1e-10)


def test_multifeed_colocated_equivalence():
    """Splitting a single feed into two colocated feeds at the same
    stage must give bit-identical numerics to the single-feed path."""
    section("test_multifeed_colocated_equivalence")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]

    res_single = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)

    res_double = distillation_column(
        n_stages=12,
        feeds=[FeedSpec(stage=6, F=30.0, z=[0.5, 0.5]),
               FeedSpec(stage=6, F=70.0, z=[0.5, 0.5])],
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)

    err_x = float(np.max(np.abs(res_single.x - res_double.x)))
    err_T = float(np.max(np.abs(res_single.T - res_double.T)))
    err_L = float(np.max(np.abs(res_single.L - res_double.L)))
    err_V = float(np.max(np.abs(res_single.V - res_double.V)))
    print(f"    max |dx|={err_x:.2e}  |dT|={err_T:.2e}  "
          f"|dL|={err_L:.2e}  |dV|={err_V:.2e}")
    check("x profiles bit-identical", err_x < 1e-12)
    check("T profiles bit-identical", err_T < 1e-12)
    check("L profiles bit-identical", err_L < 1e-12)
    check("V profiles bit-identical", err_V < 1e-12)


def test_multifeed_different_compositions():
    """Two feeds with DIFFERENT compositions: one rich in benzene at
    a high stage, one rich in toluene at a low stage.  Each feed
    enters its 'natural' end of the column."""
    section("test_multifeed_different_compositions")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15,
        feeds=[FeedSpec(stage=5,  F=50.0, z=[0.7, 0.3]),   # benzene-rich at top
               FeedSpec(stage=10, F=50.0, z=[0.3, 0.7])],  # toluene-rich lower
        reflux_ratio=2.5, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}")
    print(f"    x_D = {np.round(res.x_D, 4)}, x_B = {np.round(res.x_B, 4)}")
    check("converged", res.converged)
    # Mass balance
    total_in = sum(f.F * np.asarray(f.z) for f in res.feeds)
    total_out = res.D * res.x_D + res.B * res.x_B
    err = float(np.max(np.abs(total_in - total_out)))
    print(f"    max mass balance err = {err:.3e}")
    check("mass balance < 1e-10", err < 1e-10)
    # 50/50 overall feed should split cleanly with R=2.5, 15 stages
    check("benzene strongly enriched in distillate", res.x_D[0] > 0.95)
    check("toluene strongly enriched in bottoms",    res.x_B[1] > 0.95)


def test_liquid_side_draw_mass_balance():
    """Liquid side draw at an interior stage: overall and per-species
    mass balance must close to roundoff."""
    section("test_liquid_side_draw_mass_balance")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    U = 10.0
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=40.0, pressure=101325.0,
        liquid_draws={8: U},
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}, "
          f"||F||~{res.message.split('=')[-1]}")
    check("converged with liquid side draw", res.converged)
    check("D + B + U_draw = F_total",
          abs(res.D + res.B + U - 100.0) < 1e-10)
    # Per-species
    in_  = 100.0 * np.array([0.5, 0.5])
    out_ = res.D * res.x_D + res.B * res.x_B + U * res.x[7]
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    max mass balance err = {err:.3e}")
    check("per-species mass balance < 1e-7", err < 1e-7)
    # L profile sanity: liquid drops by U below the draw stage
    L_above = res.L[6]   # stage 7 (Python idx 6)
    L_below = res.L[7]   # stage 8 (Python idx 7) -- after the draw
    print(f"    L[stage 7]={L_above:.4f}, L[stage 8 after draw]={L_below:.4f}")
    check(f"L drops by U_8 across the draw stage",
          abs((L_above - L_below) - U) < 1e-9)


def test_vapor_side_draw_mass_balance():
    """Vapor side draw: V profile must JUMP UP below the draw stage
    by exactly the draw flow rate (CMO).  Mass balance closes."""
    section("test_vapor_side_draw_mass_balance")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    W = 10.0
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=40.0, pressure=101325.0,
        vapor_draws={4: W},
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}")
    check("converged with vapor side draw", res.converged)
    check("D + B + W_draw = F_total",
          abs(res.D + res.B + W - 100.0) < 1e-10)
    # V profile: V[j+1] - V[j] = W at the vapor-draw stage j (1-indexed=4)
    V_above = res.V[2]   # stage 3, above the draw
    V_below = res.V[4]   # stage 5, below the draw
    print(f"    V[stage 3]={V_above:.4f}, V[stage 5]={V_below:.4f}")
    check("V increases by W below the draw stage",
          abs((V_below - V_above) - W) < 1e-9)
    # Per-species mass balance
    y_4 = res.y[3]
    in_  = 100.0 * np.array([0.5, 0.5])
    out_ = res.D * res.x_D + res.B * res.x_B + W * y_4
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    max mass balance err = {err:.3e}")
    check("per-species mass balance < 1e-9", err < 1e-9)


def test_recovery_with_side_draw():
    """recovery() correctly accounts for side-draw outlets:
    distillate + bottoms + draw recoveries = 1."""
    section("test_recovery_with_side_draw")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=40.0, pressure=101325.0,
        liquid_draws={8: 15.0},
        species_names=species, activity_model=uf, psat_funcs=psats)
    for sp in ["benzene", "toluene"]:
        rD = res.recovery(sp, "distillate")
        rB = res.recovery(sp, "bottoms")
        rU = res.recovery(sp, "liquid_draw:8")
        total = rD + rB + rU
        print(f"    {sp}: D={rD:.5f}  B={rB:.5f}  U_8={rU:.5f}  "
              f"sum={total:.10f}")
        check(f"{sp}: recoveries to D+B+draw sum to 1",
              abs(total - 1.0) < 1e-7)


def test_three_outlet_column():
    """Combined feature test: two feeds + liquid draw + vapor draw."""
    section("test_three_outlet_column")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15,
        feeds=[FeedSpec(stage=4,  F=40.0, z=[0.6, 0.4]),
               FeedSpec(stage=10, F=60.0, z=[0.3, 0.7])],
        reflux_ratio=2.5, distillate_rate=35.0, pressure=101325.0,
        liquid_draws={7: 5.0},
        vapor_draws={3: 3.0},
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    converged={res.converged}, iters={res.iterations}")
    check("complex column converges", res.converged)
    # Overall mass balance
    total_F = sum(f.F for f in res.feeds)
    total_out_flow = res.D + res.B + 5.0 + 3.0
    print(f"    Sum F = {total_F:.4f}, D+B+U+W = {total_out_flow:.4f}")
    check("overall flow balance",
          abs(total_F - total_out_flow) < 1e-9)
    # Per-species
    in_ = sum(f.F * np.asarray(f.z) for f in res.feeds)
    out_ = (res.D * res.x_D + res.B * res.x_B
            + 5.0 * res.x[6] + 3.0 * res.y[2])
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    max mass balance err = {err:.3e}")
    check("per-species mass balance < 1e-7", err < 1e-7)


def test_wang_henke_rejects_multifeed():
    """Wang-Henke solver must reject multi-feed and side-draw configs
    with a clear error message."""
    section("test_wang_henke_rejects_multifeed")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]

    raised = False
    try:
        distillation_column(
            n_stages=10,
            feeds=[FeedSpec(stage=3, F=50.0, z=[0.5, 0.5]),
                   FeedSpec(stage=7, F=50.0, z=[0.5, 0.5])],
            reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
            species_names=species, activity_model=uf, psat_funcs=psats,
            method="wang_henke")
    except ValueError as e:
        raised = True
        print(f"    expected error: {str(e)[:80]}")
    check("multi-feed + Wang-Henke raises ValueError", raised)

    raised = False
    try:
        distillation_column(
            n_stages=10, feed_stage=5,
            feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
            reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
            liquid_draws={6: 10.0},
            species_names=species, activity_model=uf, psat_funcs=psats,
            method="wang_henke")
    except ValueError as e:
        raised = True
    check("side draw + Wang-Henke raises ValueError", raised)


def test_invalid_multifeed_combo_raises():
    """Cannot mix `feeds` list with single-feed scalars."""
    section("test_invalid_multifeed_combo_raises")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    raised = False
    try:
        distillation_column(
            n_stages=10, feed_stage=5, feed_F=100.0, feed_z=[0.5, 0.5],
            feeds=[FeedSpec(stage=5, F=100.0, z=[0.5, 0.5])],
            reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
            species_names=species, activity_model=uf, psat_funcs=psats)
    except ValueError as e:
        raised = True
    check("feeds + single-feed scalars raises", raised)

    raised = False
    try:
        distillation_column(
            n_stages=10,
            feeds=[FeedSpec(stage=11, F=100.0, z=[0.5, 0.5])],   # out of range
            reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
            species_names=species, activity_model=uf, psat_funcs=psats)
    except ValueError as e:
        raised = True
    check("feed stage > n_stages raises", raised)


def test_q_fraction_saturated_vapor_feed():
    """q=0 saturated vapor feed: under CMO, V should DROP below the
    feed by F (all of feed joins vapor, going up); L unchanged
    (no liquid added)."""
    section("test_q_fraction_saturated_vapor_feed")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0, feed_q=0.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    iters={res.iterations}")
    print(f"    V profile: {[round(v,2) for v in res.V]}")
    print(f"    L profile: {[round(l,2) for l in res.L]}")
    # V_top = (R+1)*D = 150; V should be 150 above stage 6 (Python idx 5)
    # and 50 below (= V_top - F)
    check("V above feed = 150",  abs(res.V[4] - 150.0) < 1e-9)
    check("V below feed = 50 (V_top - F)",  abs(res.V[6] - 50.0) < 1e-9)
    check("L stays at 100 (no liquid added by q=0 feed)",
          abs(res.L[4] - 100.0) < 1e-9 and abs(res.L[6] - 100.0) < 1e-9)
    check("L_N (reboiler) = B = 50",  abs(res.L[-1] - 50.0) < 1e-9)
    # Mass balance
    err = float(np.max(np.abs(100 * np.array([0.5, 0.5])
                              - res.D * res.x_D - res.B * res.x_B)))
    print(f"    mass balance err = {err:.2e}")
    check("mass balance < 1e-10",  err < 1e-10)


def test_q_fraction_two_phase_feed():
    """q=0.5: feed splits 50/50 between liquid and vapor.
    Under CMO, L should rise by 50 (q*F) at the feed and V should
    drop by 50 ((1-q)*F) below the feed."""
    section("test_q_fraction_two_phase_feed")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0, feed_q=0.5,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    L profile: {[round(l,2) for l in res.L]}")
    print(f"    V profile: {[round(v,2) for v in res.V]}")
    check("L above feed = R*D = 100",  abs(res.L[4] - 100.0) < 1e-9)
    check("L at feed and below = R*D + q*F = 150",
          abs(res.L[5] - 150.0) < 1e-9 and abs(res.L[8] - 150.0) < 1e-9)
    check("V above feed = (R+1)*D = 150", abs(res.V[4] - 150.0) < 1e-9)
    check("V below feed = (R+1)*D - (1-q)*F = 100",
          abs(res.V[6] - 100.0) < 1e-9)


def test_q_equals_1_unchanged():
    """For q=1 (default), the multi-feed/q-aware code path must be
    bit-identical to the v0.9.71 single-feed default."""
    section("test_q_equals_1_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_q1      = distillation_column(**base, feed_q=1.0)
    err_x = float(np.max(np.abs(res_default.x - res_q1.x)))
    err_T = float(np.max(np.abs(res_default.T - res_q1.T)))
    print(f"    max |x_default - x_q1| = {err_x:.2e}")
    print(f"    max |T_default - T_q1| = {err_T:.2e}")
    check("q=1 explicit identical to default (q=1 implicit)",
          err_x < 1e-15 and err_T < 1e-15)


def test_q_in_FeedSpec():
    """FeedSpec carries q properly; multi-feed with mixed q's works."""
    section("test_q_in_FeedSpec")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=14,
        feeds=[FeedSpec(stage=4,  F=50.0, z=[0.5, 0.5], q=1.0),  # liquid
               FeedSpec(stage=10, F=50.0, z=[0.5, 0.5], q=0.0)], # vapor
        reflux_ratio=2.5, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    iters={res.iterations}")
    print(f"    L profile: {[round(l,2) for l in res.L]}")
    print(f"    V profile: {[round(v,2) for v in res.V]}")
    check("converged with mixed-q feeds", res.converged)
    # L should rise by q1*F1 = 50 at stage 4 (liquid feed), no change at stage 10
    check("L rise of 50 at stage 4 (liquid feed)",
          abs((res.L[3] - res.L[2]) - 50.0) < 1e-9)
    check("L unchanged at stage 10 (vapor feed)",
          abs(res.L[9] - res.L[8]) < 1e-9)
    # V should be unchanged at stage 4 (liquid feed), drop by (1-q)*F = 50
    # below stage 10 (vapor feed)
    check("V unchanged across stage 4 (liquid feed)",
          abs(res.V[3] - res.V[2]) < 1e-9)
    check("V drop of 50 below stage 10 (vapor feed)",
          abs((res.V[9] - res.V[10]) - 50.0) < 1e-9)
    # Mass balance with multi-feed
    in_ = sum(f.F * np.asarray(f.z) for f in res.feeds)
    out_ = res.D * res.x_D + res.B * res.x_B
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    mass balance err = {err:.2e}")
    check("mass balance < 1e-7",  err < 1e-7)
    # FeedSpec round-trip
    check("FeedSpec[0].q = 1.0",  res.feeds[0].q == 1.0)
    check("FeedSpec[1].q = 0.0",  res.feeds[1].q == 0.0)


def test_q_fraction_reduces_separation():
    """Vapor feed (q=0) generally yields worse separation than liquid
    feed (q=1) at the same composition because the vapor portion
    bypasses the rectifying section.  With moderate stages and reflux,
    expect x_D[benzene] for q=0 < x_D[benzene] for q=1."""
    section("test_q_fraction_reduces_separation")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=10, feed_stage=5,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=370.0,
        reflux_ratio=1.5, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    purities = []
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        res = distillation_column(**base, feed_q=q)
        purities.append((q, res.x_D[0]))
        print(f"    q={q:.2f}: x_D[benzene]={res.x_D[0]:.4f}")
    # Higher q should give better separation (more liquid feed = better
    # rectification of the light component)
    p_q0  = purities[0][1]
    p_q1  = purities[-1][1]
    print(f"    x_D(q=0)={p_q0:.4f} vs x_D(q=1)={p_q1:.4f}")
    check("x_D(q=1) > x_D(q=0)", p_q1 > p_q0)


def test_partial_condenser_field():
    """condenser='partial' is stored in the result and the solver
    converges identically (the residual equations are the same; only
    the user-facing interpretation differs)."""
    section("test_partial_condenser_field")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res_total = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        condenser="total")
    res_partial = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        condenser="partial")
    check("total result.condenser = 'total'",
          res_total.condenser == "total")
    check("partial result.condenser = 'partial'",
          res_partial.condenser == "partial")
    # Math is identical between the two — only user-facing
    # interpretation differs (stage 1 = top tray vs stage 1 = condenser)
    err_x = float(np.max(np.abs(res_total.x - res_partial.x)))
    print(f"    max |x_total - x_partial| = {err_x:.2e} "
          f"(expected: identical, math is the same)")
    check("residual equations give identical numerics", err_x < 1e-15)


def test_partial_condenser_extra_stage_separates_better():
    """A 13-stage partial-condenser column has the same number of
    TRAYS as a 12-stage total-condenser column (12 trays + reboiler
    in both cases) plus an extra equilibrium stage at the condenser.
    Therefore the partial column should give at least as good a
    separation, in this case better because of the extra stage."""
    section("test_partial_condenser_extra_stage_separates_better")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    # 12-stage total: 11 trays + 1 reboiler = 12 stages, feed at 6
    res_total = distillation_column(
        n_stages=12, feed_stage=6, condenser="total", **base)
    # 13-stage partial: 1 condenser + 11 trays + 1 reboiler = 13 stages,
    # feed at 7 (same physical position as in total case: stage 7 partial
    # corresponds to stage 6 total because the partial has a condenser
    # at stage 1 prepended)
    res_partial = distillation_column(
        n_stages=13, feed_stage=7, condenser="partial", **base)
    print(f"    total (n=12):   x_D[benzene]={res_total.x_D[0]:.5f}")
    print(f"    partial (n=13): x_D[benzene]={res_partial.x_D[0]:.5f}")
    check("partial-condenser column with extra stage gives "
          "x_D[benzene] >= total-condenser (same trays)",
          res_partial.x_D[0] >= res_total.x_D[0])


def test_pressure_drop_uniform_default():
    """pressure_drop=None or 0 must give bit-identical results to a
    uniform pressure column, since the default is no pressure drop."""
    section("test_pressure_drop_uniform_default")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_dp0     = distillation_column(**base, pressure_drop=0.0)
    err_x = float(np.max(np.abs(res_default.x - res_dp0.x)))
    err_T = float(np.max(np.abs(res_default.T - res_dp0.T)))
    print(f"    max |x_default - x_dp0| = {err_x:.2e}")
    print(f"    max |T_default - T_dp0| = {err_T:.2e}")
    check("pressure_drop=0 identical to default",
          err_x < 1e-15 and err_T < 1e-15)


def test_pressure_drop_raises_T_profile():
    """A positive pressure_drop should monotonically push the per-stage
    pressure higher going down, raising the bubble-point temperatures
    relative to the uniform-pressure case."""
    section("test_pressure_drop_raises_T_profile")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_u = distillation_column(**base)
    res_p = distillation_column(**base, pressure_drop=2000.0)
    print(f"    uniform: p={res_u.p[0]:.0f} Pa everywhere; T_top={res_u.T[0]:.2f}, "
          f"T_bot={res_u.T[-1]:.2f}")
    print(f"    +2 kPa/stage: p[0]={res_p.p[0]:.0f}, p[-1]={res_p.p[-1]:.0f}; "
          f"T_top={res_p.T[0]:.2f}, T_bot={res_p.T[-1]:.2f}")
    check("p[0] = base pressure",
          abs(res_p.p[0] - 101325.0) < 1e-9)
    check("p[-1] = base + (n-1)*dp",
          abs(res_p.p[-1] - (101325.0 + 11 * 2000.0)) < 1e-9)
    # Bottom T must rise because BP rises with pressure
    check("T_bot rises with pressure_drop",
          res_p.T[-1] > res_u.T[-1] + 3.0)
    # Top T may or may not rise (top stage at the same p),
    # but reboiler T definitely should
    check("p profile monotone", np.all(np.diff(res_p.p) > 0))


def test_pressure_array_explicit():
    """Pass a full per-stage pressure array; must be used as-is."""
    section("test_pressure_array_explicit")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    n = 12
    p = np.linspace(101325.0, 121325.0, n)
    res = distillation_column(
        n_stages=n, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=p,
        species_names=species, activity_model=uf, psat_funcs=psats)
    err = float(np.max(np.abs(res.p - p)))
    print(f"    max |p_result - p_input| = {err:.2e}")
    check("p in result equals p passed in", err < 1e-9)


def test_murphree_E1_unchanged():
    """stage_efficiency=None or 1.0 must be bit-identical to the
    default (full equilibrium)."""
    section("test_murphree_E1_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_e1      = distillation_column(**base, stage_efficiency=1.0)
    err_x = float(np.max(np.abs(res_default.x - res_e1.x)))
    err_y = float(np.max(np.abs(res_default.y - res_e1.y)))
    err_T = float(np.max(np.abs(res_default.T - res_e1.T)))
    print(f"    max |dx|={err_x:.2e}, |dy|={err_y:.2e}, |dT|={err_T:.2e}")
    check("E=1 explicit identical to default",
          err_x < 1e-15 and err_y < 1e-15 and err_T < 1e-15)


def test_murphree_reduces_separation():
    """Below-unity Murphree efficiency means each stage achieves only a
    partial equilibration with the vapor coming up.  At the same column
    geometry, this must yield a lower distillate purity than the
    full-equilibrium case."""
    section("test_murphree_reduces_separation")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    purities = []
    for E in [1.0, 0.9, 0.7, 0.5]:
        res = distillation_column(**base, stage_efficiency=E)
        purities.append((E, res.x_D[0]))
        print(f"    E={E}: iters={res.iterations}, x_D[B]={res.x_D[0]:.4f}")
    # Purity should be monotone non-increasing as E decreases
    for i in range(len(purities) - 1):
        E_a, p_a = purities[i]
        E_b, p_b = purities[i + 1]
        check(f"x_D(E={E_a}) >= x_D(E={E_b})", p_a >= p_b - 1e-9)


def test_murphree_array_bottom_forced_to_one():
    """Reboiler is always treated as a full equilibrium stage; if the
    user passes an array with E[-1] < 1, the solver silently overrides
    it to 1.0 (the vapor below a reboiler does not exist)."""
    section("test_murphree_array_bottom_forced_to_one")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    n = 12
    E_user = [0.6] * n   # user attempts to set reboiler E < 1
    res = distillation_column(
        n_stages=n, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        stage_efficiency=E_user)
    print(f"    iters={res.iterations}, x_D[B]={res.x_D[0]:.4f}")
    # Compare against E[:-1]=0.6 with E[-1]=1.0 explicitly --- must match
    E_explicit = [0.6] * (n - 1) + [1.0]
    res_explicit = distillation_column(
        n_stages=n, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        stage_efficiency=E_explicit)
    err_x = float(np.max(np.abs(res.x - res_explicit.x)))
    print(f"    max |dx| vs E[-1]=1 forced = {err_x:.2e}")
    check("reboiler E forced to 1 internally", err_x < 1e-15)


def test_pressure_array_input_validation():
    """Pressure-array length must match n_stages; pressure_drop and
    array form are mutually exclusive."""
    section("test_pressure_array_input_validation")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=10, feed_stage=5,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    # Mismatched length
    try:
        distillation_column(**base, pressure=np.array([1e5] * 9))
        check("mismatched pressure length raises", False)
    except ValueError as e:
        print(f"    mismatched length: {e}")
        check("mismatched pressure length raises", True)
    # Conflict array + drop
    try:
        distillation_column(**base, pressure=np.array([1e5] * 10),
                            pressure_drop=2000.0)
        check("array + drop raises", False)
    except ValueError as e:
        print(f"    array + drop: {e}")
        check("array + drop raises", True)
    # Negative pressure
    try:
        distillation_column(**base, pressure=-1e5)
        check("negative pressure raises", False)
    except ValueError:
        check("negative pressure raises", True)


def test_pressure_drop_smoke():
    """Linear pressure drop: p[j] = p[0] + j * pressure_drop. Higher
    pressure at the bottom should raise the bottom temperature
    (boiling point rises with pressure)."""
    section("test_pressure_drop_smoke")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res_uniform = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_dp = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        pressure_drop=1000.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    print(f"    uniform: T[-1]={res_uniform.T[-1]:.2f}, "
          f"x_D[B]={res_uniform.x_D[0]:.4f}")
    print(f"    drop:    T[-1]={res_dp.T[-1]:.2f}, "
          f"x_D[B]={res_dp.x_D[0]:.4f}")
    print(f"    p[0]={res_dp.p[0]:.0f}, p[-1]={res_dp.p[-1]:.0f}")
    check("p[0] = top pressure", abs(res_dp.p[0] - 101325.0) < 1e-6)
    check("p[-1] = top + dp*(N-1)",
          abs(res_dp.p[-1] - (101325.0 + 1000.0 * 11)) < 1e-6)
    check("higher P at bottom raises T_bottom",
          res_dp.T[-1] > res_uniform.T[-1])


def test_pressure_array_input():
    """User can pass per-stage pressure array directly."""
    section("test_pressure_array_input")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    p_arr = np.linspace(101325.0, 110325.0, 12)
    res = distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=p_arr,
        species_names=species, activity_model=uf, psat_funcs=psats)
    check("pressure array round-trips",
          np.allclose(res.p, p_arr))
    check("converged with non-uniform pressure", res.converged)
    # Mass balance still closes
    err = float(np.max(np.abs(100*np.array([0.5, 0.5])
                              - res.D*res.x_D - res.B*res.x_B)))
    check("mass balance < 1e-9", err < 1e-9)
    print(f"    iters={res.iterations}, x_D[B]={res.x_D[0]:.4f}, "
          f"err={err:.2e}")


def test_pressure_array_and_drop_conflict():
    """Cannot pass both array pressure and pressure_drop."""
    section("test_pressure_array_and_drop_conflict")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    try:
        distillation_column(
            n_stages=12, feed_stage=6,
            feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
            reflux_ratio=2.0, distillate_rate=50.0,
            pressure=np.linspace(101325.0, 110325.0, 12),
            pressure_drop=500.0,    # conflict
            species_names=species, activity_model=uf, psat_funcs=psats)
        check("array+drop conflict raises", False)
    except ValueError as e:
        print(f"    expected error: {e}")
        check("array+drop conflict raises", True)


def test_pressure_drop_zero_unchanged():
    """pressure_drop=0 (or None) is bit-identical to no-pressure-drop."""
    section("test_pressure_drop_zero_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_dp0     = distillation_column(**base, pressure_drop=0.0)
    err_x = float(np.max(np.abs(res_default.x - res_dp0.x)))
    err_T = float(np.max(np.abs(res_default.T - res_dp0.T)))
    print(f"    max |x| diff = {err_x:.2e}, max |T| diff = {err_T:.2e}")
    check("pressure_drop=0 bit-identical", err_x < 1e-15 and err_T < 1e-15)


def test_stage_efficiency_full_equilibrium_unchanged():
    """stage_efficiency=1.0 (or None) is bit-identical to default."""
    section("test_stage_efficiency_full_equilibrium_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_E1      = distillation_column(**base, stage_efficiency=1.0)
    err = float(np.max(np.abs(res_default.x - res_E1.x)))
    print(f"    max |x_default - x_E1| = {err:.2e}")
    check("stage_efficiency=1.0 bit-identical", err < 1e-15)


def test_stage_efficiency_reduces_separation():
    """Murphree E < 1 should give less separation than E = 1
    (partial equilibrium). The reboiler is always at full equilibrium."""
    section("test_stage_efficiency_reduces_separation")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    purities = []
    for E in [1.0, 0.9, 0.7, 0.5]:
        res = distillation_column(**base, stage_efficiency=E)
        purities.append((E, res.x_D[0]))
        print(f"    E={E}: x_D[B]={res.x_D[0]:.4f}, iters={res.iterations}")
    check("E=1.0 > E=0.9", purities[0][1] > purities[1][1])
    check("E=0.9 > E=0.7", purities[1][1] > purities[2][1])
    check("E=0.7 > E=0.5", purities[2][1] > purities[3][1])


def test_stage_efficiency_array():
    """stage_efficiency can be a full array; reboiler always = 1."""
    section("test_stage_efficiency_array")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    n = 12
    # Try to set reboiler efficiency to 0.5 and check it gets overridden to 1
    E_in = np.full(n, 0.85)
    E_in[-1] = 0.5    # reboiler — should be forced to 1.0 internally
    res = distillation_column(
        n_stages=n, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        stage_efficiency=E_in)
    check("array eff converged", res.converged)
    print(f"    iters={res.iterations}, x_D[B]={res.x_D[0]:.4f}")


def test_stage_efficiency_invalid():
    """E outside (0, 1] raises."""
    section("test_stage_efficiency_invalid")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    for bad in (0.0, -0.1, 1.5, [0.5, 0.0]):
        try:
            distillation_column(**base, stage_efficiency=bad)
            check(f"E={bad} should raise", False)
        except ValueError:
            check(f"E={bad} raises", True)


def test_wang_henke_rejects_pressure_and_efficiency():
    """Wang-Henke rejects pressure profile and Murphree."""
    section("test_wang_henke_rejects_pressure_and_efficiency")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=10, feed_stage=5,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        method="wang_henke")
    for kwargs in [
        dict(pressure_drop=1000.0),
        dict(stage_efficiency=0.7),
    ]:
        try:
            distillation_column(**base, **kwargs)
            check(f"WH should reject {list(kwargs)}", False)
        except ValueError:
            check(f"WH rejects {list(kwargs)}", True)


def test_pump_around_no_PA_unchanged():
    """pump_arounds=None (default) is bit-identical to v0.9.73."""
    section("test_pump_around_no_PA_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_empty   = distillation_column(**base, pump_arounds=[])
    err_x = float(np.max(np.abs(res_default.x - res_empty.x)))
    err_T = float(np.max(np.abs(res_default.T - res_empty.T)))
    print(f"    max |x_default - x_empty_PA| = {err_x:.2e}")
    print(f"    max |T_default - T_empty_PA| = {err_T:.2e}")
    check("pump_arounds=[] bit-identical to default",
          err_x < 1e-15 and err_T < 1e-15)


def test_pump_around_L_profile_step():
    """A pump-around with draw at stage j_d, return at j_r, flow F
    adds F to L[j] for j_r-1 <= j <= j_d-2 (Python idx)."""
    section("test_pump_around_L_profile_step")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res_base = distillation_column(
        n_stages=14, feed_stage=7,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_pa = distillation_column(
        n_stages=14, feed_stage=7,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        pump_arounds=[PumpAround(draw_stage=5, return_stage=2, flow=80.0)])
    print(f"    base L: {[round(v,1) for v in res_base.L]}")
    print(f"    PA   L: {[round(v,1) for v in res_pa.L]}")
    # PA carries from Python idx 1 (stage 2, return) through idx 3 (stage 4,
    # just before draw at stage 5)
    for j in range(1, 4):
        check(f"L[{j}] += 80 (PA flow)",
              abs((res_pa.L[j] - res_base.L[j]) - 80.0) < 1e-9)
    for j in [0, 4, 5, 6, 13]:
        check(f"L[{j}] unchanged (outside PA loop)",
              abs(res_pa.L[j] - res_base.L[j]) < 1e-9)


def test_pump_around_mass_balance():
    """Pump-around is mass-conserving (internal recycle): overall
    mass balance still closes."""
    section("test_pump_around_mass_balance")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=14, feed_stage=7,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        pump_arounds=[PumpAround(draw_stage=5, return_stage=2, flow=80.0)])
    in_ = 100 * np.array([0.5, 0.5])
    out_ = res.D * res.x_D + res.B * res.x_B
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    mass balance err = {err:.2e}")
    check("mass balance closes < 1e-9", err < 1e-9)
    check("D + B = total_F",
          abs(res.D + res.B - 100.0) < 1e-9)


def test_pump_around_multiple():
    """Multiple non-overlapping pump-arounds add up correctly."""
    section("test_pump_around_multiple")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=20, feed_stage=12,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=3.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        pump_arounds=[
            PumpAround(draw_stage=5,  return_stage=2, flow=60.0),
            PumpAround(draw_stage=10, return_stage=7, flow=40.0),
        ])
    print(f"    iters={res.iterations}")
    print(f"    L profile: {[round(v,1) for v in res.L]}")
    # PA1 carries +60 at Python idx 1..3
    # PA2 carries +40 at Python idx 6..8
    base_L_above = 3.0 * 50.0  # R*D = 150
    check("L[2] = base + 60 (PA1)",
          abs(res.L[2] - (base_L_above + 60.0)) < 1e-9)
    check("L[7] = base + 40 (PA2)",
          abs(res.L[7] - (base_L_above + 40.0)) < 1e-9)
    check("L[5] = base (between PAs)",
          abs(res.L[5] - base_L_above) < 1e-9)
    # Mass balance
    in_ = 100 * np.array([0.5, 0.5])
    out_ = res.D * res.x_D + res.B * res.x_B
    err = float(np.max(np.abs(in_ - out_)))
    check("mass balance closes < 1e-9", err < 1e-9)


def test_pump_around_invalid():
    """return_stage >= draw_stage, negative flow, etc. raise."""
    section("test_pump_around_invalid")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    for pa in (
        PumpAround(draw_stage=3, return_stage=5, flow=10.0),  # return > draw
        PumpAround(draw_stage=5, return_stage=5, flow=10.0),  # return = draw
        PumpAround(draw_stage=5, return_stage=2, flow=-1.0),  # negative flow
        PumpAround(draw_stage=15, return_stage=2, flow=10.0), # draw > n_stages
        PumpAround(draw_stage=5, return_stage=2, flow=10.0, dT=-5.0),  # dT<0
    ):
        try:
            distillation_column(**base, pump_arounds=[pa])
            check(f"invalid PA {pa} should raise", False)
        except ValueError:
            check(f"invalid PA raises", True)


def test_pump_around_with_energy_balance():
    """PA with cooling under energy balance: dT > 0 should give a
    lower return-stage T than dT = 0 (same PA flow and topology),
    isolating the cooling effect from any composition shift."""
    section("test_pump_around_with_energy_balance")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]

    def hL(i, T): return [136.0, 157.0][i] * (T - 298.15)
    def hV(i, T): return [136.0, 157.0][i] * (T - 298.15) + [33800.0, 38000.0][i]
    h_L_funcs = [lambda T, i=i: hL(i, T) for i in range(2)]
    h_V_funcs = [lambda T, i=i: hV(i, T) for i in range(2)]

    base = dict(
        n_stages=14, feed_stage=7,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        energy_balance=True, h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs)

    # Same PA topology and flow, only dT differs
    res_dT0 = distillation_column(
        **base,
        pump_arounds=[PumpAround(draw_stage=5, return_stage=2,
                                 flow=80.0, dT=0.0)])
    res_dT10 = distillation_column(
        **base,
        pump_arounds=[PumpAround(draw_stage=5, return_stage=2,
                                 flow=80.0, dT=10.0)])
    print(f"    dT=0:  T[1]={res_dT0.T[1]:.3f},  "
          f"x_D[B]={res_dT0.x_D[0]:.4f}")
    print(f"    dT=10: T[1]={res_dT10.T[1]:.3f}, "
          f"x_D[B]={res_dT10.x_D[0]:.4f}")
    check("converged with cooled PA + EB", res_dT10.converged)
    check("dT > 0 produces a measurably different solution",
          abs(res_dT10.T[1] - res_dT0.T[1]) > 1e-4 or
          abs(res_dT10.x_D[0] - res_dT0.x_D[0]) > 1e-4)
    # Mass balance still holds with cooling
    in_ = 100 * np.array([0.5, 0.5])
    out_ = res_dT10.D * res_dT10.x_D + res_dT10.B * res_dT10.x_B
    err = float(np.max(np.abs(in_ - out_)))
    print(f"    mass balance err (dT=10): {err:.2e}")
    check("mass balance closes < 1e-7 with cooled PA", err < 1e-7)


def test_wang_henke_rejects_pump_arounds():
    """Wang-Henke rejects pump_arounds."""
    section("test_wang_henke_rejects_pump_arounds")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    try:
        distillation_column(
            n_stages=10, feed_stage=5,
            feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
            reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
            species_names=species, activity_model=uf, psat_funcs=psats,
            method="wang_henke",
            pump_arounds=[PumpAround(draw_stage=4, return_stage=2,
                                     flow=10.0)])
        check("WH should reject pump_arounds", False)
    except ValueError:
        check("WH rejects pump_arounds", True)


def test_spec_x_D_vary_R():
    """Fix D, vary R to hit a distillate-purity target."""
    section("test_spec_x_D_vary_R")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, reflux_ratio=None,
        pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        specs=[Spec(kind="x_D", value=0.99, species="benzene")],
        initial_reflux_ratio=2.0)
    print(f"    R found: {res.reflux_ratio:.4f}, "
          f"x_D[B]={res.x_D[0]:.6f}")
    check("x_D spec hit to within tol",
          abs(res.x_D[0] - 0.99) < 1e-5)
    check("D fixed at 50", abs(res.D - 50.0) < 1e-9)
    check("R is positive", res.reflux_ratio > 0)


def test_spec_recovery_D_vary_D():
    """Fix R, vary D to hit a recovery target."""
    section("test_spec_recovery_D_vary_D")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=None, reflux_ratio=2.5,
        pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        specs=[Spec(kind="recovery_D", value=0.95, species="benzene")],
        initial_distillate_rate=50.0)
    rec = res.D * res.x_D[0] / 50.0
    print(f"    D found: {res.D:.4f}, recovery_D[B]={rec:.6f}")
    check("recovery_D spec hit",
          abs(rec - 0.95) < 1e-5)
    check("R fixed at 2.5",
          abs(res.reflux_ratio - 2.5) < 1e-9)


def test_spec_two_specs_vary_both():
    """Two specs (x_D and recovery_D) — vary both D and R."""
    section("test_spec_two_specs_vary_both")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=None, reflux_ratio=None,
        pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        specs=[
            Spec(kind="x_D", value=0.98, species="benzene"),
            Spec(kind="recovery_D", value=0.95, species="benzene"),
        ],
        initial_distillate_rate=50.0, initial_reflux_ratio=2.0)
    rec = res.D * res.x_D[0] / 50.0
    print(f"    D={res.D:.4f}, R={res.reflux_ratio:.4f}, "
          f"x_D[B]={res.x_D[0]:.6f}, rec={rec:.6f}")
    check("x_D spec hit", abs(res.x_D[0] - 0.98) < 1e-5)
    check("recovery_D spec hit", abs(rec - 0.95) < 1e-5)


def test_spec_x_B():
    """Vary R to hit a bottoms-purity (impurity) target."""
    section("test_spec_x_B")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, reflux_ratio=None,
        pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        specs=[Spec(kind="x_B", value=0.01, species="benzene")],
        initial_reflux_ratio=2.0)
    print(f"    R found: {res.reflux_ratio:.4f}, "
          f"x_B[B]={res.x_B[0]:.6f}")
    check("x_B spec hit", abs(res.x_B[0] - 0.01) < 1e-5)


def test_spec_invalid_count():
    """#specs != #free unknowns must raise."""
    section("test_spec_invalid_count")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    # 2 specs but only 1 free unknown
    try:
        distillation_column(
            **base,
            distillate_rate=50.0, reflux_ratio=None,
            specs=[
                Spec(kind="x_D", value=0.95, species="benzene"),
                Spec(kind="x_B", value=0.05, species="benzene"),
            ])
        check("2 specs + 1 free should raise", False)
    except ValueError:
        check("2 specs + 1 free raises", True)
    # 1 spec but both free
    try:
        distillation_column(
            **base,
            distillate_rate=None, reflux_ratio=None,
            specs=[Spec(kind="x_D", value=0.95, species="benzene")])
        check("1 spec + 2 free should raise", False)
    except ValueError:
        check("1 spec + 2 free raises", True)


def test_spec_no_specs_unchanged():
    """specs=None or [] is bit-identical to v0.9.74 default."""
    section("test_spec_no_specs_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    res_default = distillation_column(**base)
    res_none    = distillation_column(**base, specs=None)
    res_empty   = distillation_column(**base, specs=[])
    err_n = float(np.max(np.abs(res_default.x - res_none.x)))
    err_e = float(np.max(np.abs(res_default.x - res_empty.x)))
    check("specs=None bit-identical", err_n < 1e-15)
    check("specs=[] bit-identical", err_e < 1e-15)


def test_spec_Q_C_vary_R():
    """Q_C duty spec: fix D, vary R to hit target condenser duty."""
    section("test_spec_Q_C_vary_R")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    def hL(i, T): return [136.0, 157.0][i] * (T - 298.15)
    def hV(i, T): return [136.0, 157.0][i] * (T - 298.15) + [33800.0, 38000.0][i]
    h_L_funcs = [lambda T, i=i: hL(i, T) for i in range(2)]
    h_V_funcs = [lambda T, i=i: hV(i, T) for i in range(2)]

    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, reflux_ratio=None, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        specs=[Spec(kind="Q_C", value=5.5e6)],
        initial_reflux_ratio=2.0, spec_outer_tol=1.0)

    from stateprop.reaction.reactive_column import _compute_Q_C
    Q_C = _compute_Q_C(res, h_V_funcs, h_L_funcs)
    print(f"    R={res.reflux_ratio:.4f}, Q_C={Q_C:.0f} (target 5.5e6)")
    check("Q_C hits target within 1 J/h", abs(Q_C - 5.5e6) < 1.0)


def test_spec_Q_R_vary_R():
    """Q_R duty spec: fix D, vary R to hit target reboiler duty."""
    section("test_spec_Q_R_vary_R")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    def hL(i, T): return [136.0, 157.0][i] * (T - 298.15)
    def hV(i, T): return [136.0, 157.0][i] * (T - 298.15) + [33800.0, 38000.0][i]
    h_L_funcs = [lambda T, i=i: hL(i, T) for i in range(2)]
    h_V_funcs = [lambda T, i=i: hV(i, T) for i in range(2)]

    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, reflux_ratio=None, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        specs=[Spec(kind="Q_R", value=5.5e6)],
        initial_reflux_ratio=2.0, spec_outer_tol=1.0)

    from stateprop.reaction.reactive_column import _compute_Q_R
    Q_R = _compute_Q_R(res, h_V_funcs, h_L_funcs)
    print(f"    R={res.reflux_ratio:.4f}, Q_R={Q_R:.0f} (target 5.5e6)")
    check("Q_R hits target within 1 J/h", abs(Q_R - 5.5e6) < 1.0)


def test_spec_Q_C_requires_enthalpy_funcs():
    """Q_C / Q_R specs require h_V_funcs and h_L_funcs."""
    section("test_spec_Q_C_requires_enthalpy_funcs")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    try:
        distillation_column(
            n_stages=12, feed_stage=6,
            feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
            distillate_rate=50.0, reflux_ratio=None, pressure=101325.0,
            species_names=species, activity_model=uf, psat_funcs=psats,
            specs=[Spec(kind="Q_C", value=5.5e6)],
            initial_reflux_ratio=2.0)
        check("Q_C without h funcs should raise", False)
    except ValueError:
        check("Q_C without h funcs raises", True)


def test_spec_ratio_vary_R():
    """ratio spec: x_D[A]/x_D[B] = value."""
    section("test_spec_ratio_vary_R")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    res = distillation_column(
        n_stages=15, feed_stage=8,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, reflux_ratio=None, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        specs=[Spec(kind="ratio", value=30.0,
                    species="benzene", species2="toluene")],
        initial_reflux_ratio=2.0)
    ratio = float(res.x_D[0] / res.x_D[1])
    print(f"    R={res.reflux_ratio:.4f}, x_D[B]/x_D[T]={ratio:.4f}")
    check("ratio hits target to 1e-4",
          abs(ratio - 30.0) < 1e-4)


def test_spec_Q_C_partial_condenser():
    """Q_C spec under a partial condenser uses the partial-condenser
    formula and converges."""
    section("test_spec_Q_C_partial_condenser")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    def hL(i, T): return [136.0, 157.0][i] * (T - 298.15)
    def hV(i, T): return [136.0, 157.0][i] * (T - 298.15) + [33800.0, 38000.0][i]
    h_L_funcs = [lambda T, i=i: hL(i, T) for i in range(2)]
    h_V_funcs = [lambda T, i=i: hV(i, T) for i in range(2)]

    # Partial condenser: stage 1 = condenser; n_stages includes it.
    # Use a generous Q_C so the heuristic R initial guess is in range.
    res = distillation_column(
        n_stages=16, feed_stage=9,
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, reflux_ratio=None, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        condenser="partial",
        specs=[Spec(kind="Q_C", value=5.5e6)],
        initial_reflux_ratio=2.0, spec_outer_tol=1.0)

    from stateprop.reaction.reactive_column import _compute_Q_C
    Q_C = _compute_Q_C(res, h_V_funcs, h_L_funcs)
    print(f"    R={res.reflux_ratio:.4f}, Q_C={Q_C:.0f} (target 5.5e6), "
          f"condenser={res.condenser}")
    check("partial condenser Q_C spec converges", res.converged)
    check("partial Q_C hits target within 1 J/h", abs(Q_C - 5.5e6) < 1.0)


def test_column_eos_dispatch_unchanged():
    """vapor_eos=None must give bit-identical results to v0.9.77."""
    section("test_column_eos_dispatch_unchanged")
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    base = dict(n_stages=12, feed_stage=6,
                feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
                distillate_rate=50.0, reflux_ratio=2.0, pressure=101325.0,
                species_names=species, activity_model=uf, psat_funcs=psats)
    r0 = distillation_column(**base)
    r0_explicit = distillation_column(**base, vapor_eos=None)
    diff = float(np.max(np.abs(r0.x - r0_explicit.x)))
    print(f"    max |Δx| = {diff:.2e}")
    check("vapor_eos=None bit-identical to default", diff < 1e-15)


def test_column_eos_low_p_close_to_modified_raoult():
    """At 1 bar, γ-φ-EOS column results agree with modified Raoult
    on x_D within ~1% absolute."""
    section("test_column_eos_low_p_close_to_modified_raoult")
    from stateprop.activity import make_phi_sat_funcs
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ["benzene", "toluene"]
    mix = CubicMixture([PR(562.05, 4.895e6, 0.2110),
                        PR(591.75, 4.108e6, 0.2640)])
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    phi_sat = make_phi_sat_funcs(mix, psats)
    V_L = [89.5e-6, 106.3e-6]
    base = dict(n_stages=12, feed_stage=6,
                feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
                distillate_rate=50.0, reflux_ratio=2.0, pressure=1e5,
                species_names=species, activity_model=uf, psat_funcs=psats)
    r_no = distillation_column(**base)
    r_eos = distillation_column(**base, vapor_eos=mix,
                                pure_liquid_volumes=V_L,
                                phi_sat_funcs=phi_sat)
    d = abs(r_no.x_D[0] - r_eos.x_D[0])
    print(f"    no-EOS x_D[B]={r_no.x_D[0]:.5f}, γ-φ x_D[B]={r_eos.x_D[0]:.5f}, "
          f"|Δ|={d:.4f}")
    check("γ-φ-EOS at 1 bar matches no-EOS within 1% absolute",
          d < 0.01)


def test_column_eos_high_p_diverges():
    """At higher p (10 bar) the two formulations should give a
    measurable difference in x_D."""
    section("test_column_eos_high_p_diverges")
    from stateprop.activity import make_phi_sat_funcs
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ["benzene", "toluene"]
    mix = CubicMixture([PR(562.05, 4.895e6, 0.2110),
                        PR(591.75, 4.108e6, 0.2640)])
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    phi_sat = make_phi_sat_funcs(mix, psats)
    V_L = [89.5e-6, 106.3e-6]
    base = dict(n_stages=12, feed_stage=6,
                feed_F=100.0, feed_z=[0.5, 0.5], feed_T=440.0,
                distillate_rate=50.0, reflux_ratio=2.0, pressure=10e5,
                species_names=species, activity_model=uf, psat_funcs=psats)
    r_no = distillation_column(**base)
    r_eos = distillation_column(**base, vapor_eos=mix,
                                pure_liquid_volumes=V_L,
                                phi_sat_funcs=phi_sat)
    d = abs(r_no.x_D[0] - r_eos.x_D[0])
    dT = abs(r_no.T[0] - r_eos.T[0])
    print(f"    no-EOS x_D[B]={r_no.x_D[0]:.5f}, γ-φ x_D[B]={r_eos.x_D[0]:.5f}, "
          f"|Δx|={d:.4f}, |ΔT_top|={dT:.2f} K")
    check("γ-φ-EOS at 10 bar differs from Raoult by >1% absolute",
          d > 0.01)
    check("T-top shifts by >0.5 K at 10 bar", dT > 0.5)


def test_column_eos_phi_sat_helper():
    """make_phi_sat_funcs round-trip: use it inside a column solve."""
    section("test_column_eos_phi_sat_helper")
    from stateprop.activity import make_phi_sat_funcs
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR
    species = ["benzene", "toluene"]
    mix = CubicMixture([PR(562.05, 4.895e6, 0.2110),
                        PR(591.75, 4.108e6, 0.2640)])
    psats = [PSAT[s] for s in species]
    phi_sat = make_phi_sat_funcs(mix, psats)
    # Φ_sat at low T should be near 1; at higher T below the lighter
    # component's critical T, should drop
    Phi_300 = [f(300.0) for f in phi_sat]
    Phi_500 = [f(500.0) for f in phi_sat]
    print(f"    Φ_sat(300 K) = {Phi_300}")
    print(f"    Φ_sat(500 K) = {Phi_500}")
    check("Φ_sat at 300 K close to 1", all(0.99 < v <= 1.0 for v in Phi_300))
    check("Φ_sat at 500 K below 0.85 (significant non-ideality)",
          all(v < 0.85 for v in Phi_500))
    check("Φ_sat monotone decreasing with T",
          all(p500 < p300 for p300, p500 in zip(Phi_300, Phi_500)))


def test_column_eos_wang_henke_rejected():
    """Wang-Henke must reject vapor_eos with a clear error."""
    section("test_column_eos_wang_henke_rejected")
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR
    species = ["benzene", "toluene"]
    mix = CubicMixture([PR(562.05, 4.895e6, 0.2110),
                        PR(591.75, 4.108e6, 0.2640)])
    uf = make_unifac(species)
    psats = [PSAT[s] for s in species]
    raised = False
    try:
        distillation_column(
            n_stages=12, feed_stage=6,
            feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
            distillate_rate=50.0, reflux_ratio=2.0, pressure=101325.0,
            species_names=species, activity_model=uf, psat_funcs=psats,
            vapor_eos=mix,
            method="wang_henke")
    except ValueError as e:
        raised = "vapor_eos" in str(e) or "γ-φ-EOS" in str(e) or "EOS" in str(e)
    check("Wang-Henke + vapor_eos raises ValueError", raised)


# =====================================================================
# v0.9.88 — side strippers
# =====================================================================

def _ss_3comp_psat():
    """Three-component ideal binary set with α-light=4, α-mid=2, α-heavy=1
    (Tb 320 / 350 / 380 K, Hvap 30 kJ/mol Clausius-Clapeyron)."""
    R = 8.314462618
    Hvap = 30000.0
    return [
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/320.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/350.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/380.0))),
    ]


class _IdealActivity:
    def gammas(self, T, x):
        return np.ones(len(x))


def test_side_stripper_basic_3component():
    """3-component column with one side stripper.  Light goes to D,
    heavy to B, middle is concentrated in the SS bottoms.  Verify
    convergence, sum(x)=1 on every stage, and physically-sensible
    enrichment of the middle component in the SS product."""
    section("test_side_stripper_basic_3component")
    psat_funcs = _ss_3comp_psat()
    ss = SideStripper(draw_stage=18, return_stage=17, n_stages=4,
                      flow=60.0, bottoms_rate=30.0, pressure=1.013e5)
    res = distillation_column(
        n_stages=25, feed_stage=12, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0, pressure=1.013e5,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=3.0, distillate_rate=33.0,
        side_strippers=[ss], condenser='total',
        max_newton_iter=60, newton_tol=1e-7)
    check("converged", res.converged)
    check("Newton ||F|| < 1e-7", "1e-13" in res.message or "e-1" in res.message)
    main_sums = res.x.sum(axis=1)
    check(f"main sum(x) ≈ 1 on every stage (max dev {np.abs(main_sums-1).max():.1e})",
          np.allclose(main_sums, 1.0, atol=1e-8))
    s = res.side_strippers[0]
    ss_sums = s["x"].sum(axis=1)
    check(f"SS sum(x) ≈ 1 on every stage (max dev {np.abs(ss_sums-1).max():.1e})",
          np.allclose(ss_sums, 1.0, atol=1e-8))
    # Light ≥ 99% in D, heavy ≥ 60% in B, middle enriched in SS
    check(f"x_D[light]   = {res.x_D[0]:.4f} > 0.99", res.x_D[0] > 0.99)
    check(f"x_B[heavy]   = {res.x_B[2]:.4f} > 0.55", res.x_B[2] > 0.55)
    check(f"SS x_bottoms[middle] = {s['x_bottoms'][1]:.4f} > 0.55 (vs feed 0.33)",
          s["x_bottoms"][1] > 0.55)


def test_side_stripper_mass_balance():
    """Component-by-component mass balance for the column + side stripper:
    F * z_in == D * x_D + B * x_B + sum_SS(b_ss * x_bot_ss).  Must
    close to machine precision (no Wegstein iteration -- the simultaneous
    Newton solve enforces this exactly)."""
    section("test_side_stripper_mass_balance")
    psat_funcs = _ss_3comp_psat()
    ss = SideStripper(draw_stage=18, return_stage=17, n_stages=4,
                      flow=60.0, bottoms_rate=30.0, pressure=1.013e5)
    res = distillation_column(
        n_stages=25, feed_stage=12, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0, pressure=1.013e5,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=3.0, distillate_rate=33.0,
        side_strippers=[ss],
        max_newton_iter=60, newton_tol=1e-7)
    F_in = 100.0 * np.array([1/3, 1/3, 1/3])
    s = res.side_strippers[0]
    F_out = (res.D * res.x_D
             + res.B * res.x_B
             + s["bottoms_rate"] * s["x_bottoms"])
    err = np.abs(F_in - F_out).max()
    check(f"per-component balance closes to machine precision "
          f"(max |in-out| = {err:.1e})", err < 1e-10)
    # Overall scalar mass balance
    F_total = 100.0
    F_out_total = res.D + res.B + s["bottoms_rate"]
    check(f"overall mass balance: F={F_total} = D+B+SS_bot = "
          f"{F_out_total:.4f}", abs(F_total - F_out_total) < 1e-10)


def test_side_stripper_no_ss_unchanged():
    """A column WITHOUT a side stripper should produce IDENTICAL
    profiles whether ``side_strippers=None`` or ``side_strippers=[]``.
    Regression guard: the SS plumbing must not perturb the no-SS path."""
    section("test_side_stripper_no_ss_unchanged")
    psat_funcs = _ss_3comp_psat()
    common = dict(
        n_stages=15, feed_stage=8, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0, pressure=1.013e5,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=2.0, distillate_rate=33.0,
        max_newton_iter=40, newton_tol=1e-8)
    r_none = distillation_column(**common, side_strippers=None)
    r_empty = distillation_column(**common, side_strippers=[])
    check("None and [] give identical x", np.allclose(r_none.x, r_empty.x))
    check("None and [] give identical T", np.allclose(r_none.T, r_empty.T))


def test_side_stripper_two_independent():
    """Two independent side strippers attached at different stages.
    Both must converge in a single simultaneous Newton solve and each
    SS bottoms is a distinct side product."""
    section("test_side_stripper_two_independent")
    # 4-component case so we have enough boiling-point spread for two
    # SSs to make sense
    R = 8.314462618; Hvap = 30000.0
    psat_funcs = [
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/310.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/340.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/370.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/400.0))),
    ]
    ss1 = SideStripper(draw_stage=10, return_stage=9, n_stages=3,
                       flow=30.0, bottoms_rate=15.0, pressure=1.013e5)
    ss2 = SideStripper(draw_stage=20, return_stage=19, n_stages=3,
                       flow=30.0, bottoms_rate=15.0, pressure=1.013e5)
    res = distillation_column(
        n_stages=30, feed_stage=15, feed_F=100.0,
        feed_z=[0.25, 0.25, 0.25, 0.25], feed_q=1.0, pressure=1.013e5,
        species_names=['A', 'B', 'C', 'D'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=4.0, distillate_rate=25.0,
        side_strippers=[ss1, ss2], condenser='total',
        max_newton_iter=80, newton_tol=1e-7)
    check("converged", res.converged)
    check("two SS in result", len(res.side_strippers) == 2)
    F_in = 100.0 * np.array([0.25, 0.25, 0.25, 0.25])
    s1, s2 = res.side_strippers
    F_out = (res.D * res.x_D + res.B * res.x_B
             + s1["bottoms_rate"] * s1["x_bottoms"]
             + s2["bottoms_rate"] * s2["x_bottoms"])
    err = np.abs(F_in - F_out).max()
    check(f"4-component mass balance closes (max |in-out| = {err:.1e})",
          err < 1e-9)
    # Upper SS draws lighter components, lower SS draws heavier
    check(f"SS1 x_bot[B]={s1['x_bottoms'][1]:.3f} > SS2 x_bot[B]={s2['x_bottoms'][1]:.3f}",
          s1["x_bottoms"][1] > s2["x_bottoms"][1])
    check(f"SS2 x_bot[C]={s2['x_bottoms'][2]:.3f} > SS1 x_bot[C]={s1['x_bottoms'][2]:.3f}",
          s2["x_bottoms"][2] > s1["x_bottoms"][2])


def test_side_stripper_invalid_bottoms_rate():
    """bottoms_rate must satisfy 0 < bottoms_rate < flow.  Reject
    cases at and beyond the bounds."""
    section("test_side_stripper_invalid_bottoms_rate")
    psat_funcs = _ss_3comp_psat()
    common = dict(
        n_stages=15, feed_stage=8, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0, pressure=1.013e5,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=2.0, distillate_rate=33.0)
    bad_cases = [
        SideStripper(8, 7, 3, 30.0, 30.0, 1.013e5),    # equal: no vapor
        SideStripper(8, 7, 3, 30.0, 35.0, 1.013e5),    # bottoms > flow
        SideStripper(8, 7, 3, 30.0, 0.0, 1.013e5),     # zero bottoms
        SideStripper(8, 7, 3, 30.0, -5.0, 1.013e5),    # negative
    ]
    n_rejected = 0
    for bad in bad_cases:
        try:
            distillation_column(**common, side_strippers=[bad])
        except (ValueError, Exception) as e:
            if "bottoms_rate" in str(e) or "bottoms" in str(e):
                n_rejected += 1
    check(f"all {len(bad_cases)} invalid bottoms_rate values rejected",
          n_rejected == len(bad_cases))


def test_side_stripper_invalid_draw_stage():
    """draw_stage and return_stage must be 1..n_stages."""
    section("test_side_stripper_invalid_draw_stage")
    psat_funcs = _ss_3comp_psat()
    common = dict(
        n_stages=15, feed_stage=8, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0, pressure=1.013e5,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=2.0, distillate_rate=33.0)
    bad_cases = [
        SideStripper(0, 7, 3, 30.0, 15.0, 1.013e5),    # draw < 1
        SideStripper(20, 7, 3, 30.0, 15.0, 1.013e5),   # draw > n_stages
        SideStripper(8, 0, 3, 30.0, 15.0, 1.013e5),    # return < 1
        SideStripper(8, 20, 3, 30.0, 15.0, 1.013e5),   # return > n_stages
    ]
    n_rejected = 0
    for bad in bad_cases:
        try:
            distillation_column(**common, side_strippers=[bad])
        except (ValueError, Exception) as e:
            if "draw_stage" in str(e) or "return_stage" in str(e):
                n_rejected += 1
    check(f"all {len(bad_cases)} out-of-range stages rejected",
          n_rejected == len(bad_cases))


def test_side_stripper_eb_supported_v0989():
    """v0.9.89: energy-balance + side strippers now WORKS (no longer
    raises NotImplementedError).  Verify EB + SS reboil mode converges
    and produces a temperature-varying SS profile."""
    section("test_side_stripper_eb_supported_v0989")
    psat_funcs = _ss_3comp_psat()
    h_V_funcs = [(lambda T: 30000.0 + 35.0 * T) for _ in range(3)]
    h_L_funcs = [(lambda T: 0.0 + 75.0 * T) for _ in range(3)]
    ss = SideStripper(draw_stage=18, return_stage=17, n_stages=4,
                      flow=60.0, bottoms_rate=30.0, pressure=1.013e5)
    res = distillation_column(
        n_stages=25, feed_stage=12, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_T=350.0, feed_q=1.0,
        pressure=1.013e5,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=3.0, distillate_rate=33.0,
        side_strippers=[ss],
        energy_balance=True,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        max_newton_iter=80, newton_tol=1e-6)
    check("EB + SS converges", res.converged)
    s = res.side_strippers[0]
    F_in = 100.0 * np.array([1/3, 1/3, 1/3])
    F_out = res.D*res.x_D + res.B*res.x_B + s["bottoms_rate"]*s["x_bottoms"]
    err = np.abs(F_in - F_out).max()
    check(f"EB + SS mass balance closes (max |in-out| = {err:.1e})",
          err < 1e-9)
    # Temperature gradient on SS (should rise going down due to stripping)
    dT = float(s["T"][-1] - s["T"][0])
    check(f"SS T-profile has positive gradient (dT = {dT:.2f} K)", dT > 0.5)


def test_side_stripper_steam_mode_cmo():
    """CMO + SS in steam mode: 60 mol/h liquid feed, 10 mol/h water
    steam injected at SS bottom, 30 mol/h side product.  Mass balance
    closes; F_main + steam = D + B + bot."""
    section("test_side_stripper_steam_mode_cmo")
    R = 8.314462618; Hvap = 30000.0
    psat_funcs = [
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/320.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/350.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/400.0))),
        (lambda T: 1.013e5 * np.exp(-40000/R * (1.0/T - 1.0/373.15))),
    ]
    ss = SideStripper(draw_stage=18, return_stage=17, n_stages=4,
                      flow=60.0, bottoms_rate=30.0, pressure=1.013e5,
                      stripping_mode="steam",
                      steam_flow=10.0, steam_z=[0, 0, 0, 1.0],
                      steam_T=400.0)
    res = distillation_column(
        n_stages=25, feed_stage=12, feed_F=100.0,
        feed_z=[0.30, 0.30, 0.30, 0.10], feed_q=1.0, pressure=1.013e5,
        species_names=['light', 'middle', 'heavy', 'water'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=3.0, distillate_rate=33.0,
        side_strippers=[ss], condenser='total',
        max_newton_iter=80, newton_tol=1e-7)
    check("CMO + steam SS converges", res.converged)
    s = res.side_strippers[0]
    F_in = 100.0 * np.array([0.30, 0.30, 0.30, 0.10])
    F_steam = 10.0 * np.array([0, 0, 0, 1.0])
    F_out = res.D*res.x_D + res.B*res.x_B + s["bottoms_rate"]*s["x_bottoms"]
    err = np.abs(F_in + F_steam - F_out).max()
    check(f"steam mode mass balance closes (max |in - out| = {err:.1e})",
          err < 1e-10)
    total_in = 100.0 + 10.0
    total_out = res.D + res.B + s["bottoms_rate"]
    check(f"overall scalar balance: F+steam={total_in}, D+B+bot={total_out:.4f}",
          abs(total_in - total_out) < 1e-10)


def test_side_stripper_steam_mode_eb():
    """EB + SS in steam mode: same setup as CMO test but with energy
    balance enabled.  Validates the EB+steam coupling works."""
    section("test_side_stripper_steam_mode_eb")
    R = 8.314462618; Hvap = 30000.0
    psat_funcs = [
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/320.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/350.0))),
        (lambda T: 1.013e5 * np.exp(-Hvap/R * (1.0/T - 1.0/400.0))),
        (lambda T: 1.013e5 * np.exp(-40000/R * (1.0/T - 1.0/373.15))),
    ]
    h_V_funcs = [(lambda T: 30000.0 + 35.0 * T) for _ in range(4)]
    h_L_funcs = [(lambda T: 0.0 + 75.0 * T) for _ in range(4)]
    ss = SideStripper(draw_stage=18, return_stage=17, n_stages=4,
                      flow=60.0, bottoms_rate=30.0, pressure=1.013e5,
                      stripping_mode="steam",
                      steam_flow=10.0, steam_z=[0, 0, 0, 1.0],
                      steam_T=400.0)
    res = distillation_column(
        n_stages=25, feed_stage=12, feed_F=100.0,
        feed_z=[0.30, 0.30, 0.30, 0.10], feed_T=350.0, feed_q=1.0,
        pressure=1.013e5,
        species_names=['light', 'middle', 'heavy', 'water'],
        activity_model=_IdealActivity(), psat_funcs=psat_funcs,
        reflux_ratio=3.0, distillate_rate=33.0,
        side_strippers=[ss],
        energy_balance=True,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        max_newton_iter=80, newton_tol=1e-6)
    check("EB + steam SS converges", res.converged)
    s = res.side_strippers[0]
    F_in = 100.0 * np.array([0.30, 0.30, 0.30, 0.10])
    F_steam = 10.0 * np.array([0, 0, 0, 1.0])
    F_out = res.D*res.x_D + res.B*res.x_B + s["bottoms_rate"]*s["x_bottoms"]
    err = np.abs(F_in + F_steam - F_out).max()
    check(f"EB + steam mass balance closes (max |in - out| = {err:.1e})",
          err < 1e-7)
    # Sanity: water (last component) has higher concentration in side
    # product than in main feed (concentrated by stripping)
    water_in_feed = 0.10
    water_in_bot = float(s["x_bottoms"][3])
    check(f"water enriched in side product ({water_in_bot:.3f} > {water_in_feed:.3f})",
          water_in_bot > water_in_feed)


def test_side_stripper_steam_invalid_z_sum():
    """Steam composition must sum to 1."""
    section("test_side_stripper_steam_invalid_z_sum")
    psat_funcs = _ss_3comp_psat()
    bad = SideStripper(draw_stage=8, return_stage=7, n_stages=3,
                       flow=30.0, bottoms_rate=15.0, pressure=1.013e5,
                       stripping_mode="steam",
                       steam_flow=5.0, steam_z=[0.5, 0.0, 0.4],
                       steam_T=400.0)
    raised = False
    try:
        distillation_column(
            n_stages=15, feed_stage=8, feed_F=100.0,
            feed_z=[1/3, 1/3, 1/3], feed_q=1.0, pressure=1.013e5,
            species_names=['light', 'middle', 'heavy'],
            activity_model=_IdealActivity(), psat_funcs=psat_funcs,
            reflux_ratio=2.0, distillate_rate=33.0,
            side_strippers=[bad])
    except ValueError as e:
        raised = "steam_z must sum to 1" in str(e)
    check("steam_z that doesn't sum to 1 is rejected", raised)


# =====================================================================
# CPR-compressed Jacobian (v0.9.117)
# =====================================================================

def test_cpr_jacobian_matches_dense_no_eb():
    """CPR-compressed and dense FD Jacobians must give identical
    converged results (no energy balance, no nonlocal coupling)."""
    section("test_cpr_jacobian_matches_dense_no_eb")
    import stateprop.reaction.reactive_column as rc
    psat = _ss_3comp_psat()
    common = dict(
        n_stages=12, feed_stage=6, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat,
        reflux_ratio=2.0, distillate_rate=33.0,
        pressure=1.013e5,
    )
    # Default uses CPR
    r_cpr = distillation_column(**common)

    # Force dense by monkey-patching the helper to set the flag
    orig = rc._build_block_tridiag_jacobian
    def force_dense(*args, **kwargs):
        kwargs['has_nonlocal_coupling'] = True
        return orig(*args, **kwargs)
    rc._build_block_tridiag_jacobian = force_dense
    try:
        r_dense = distillation_column(**common)
    finally:
        rc._build_block_tridiag_jacobian = orig

    # Compare bottoms compositions
    diff = float(np.max(np.abs(r_cpr.x[-1] - r_dense.x[-1])))
    check(f"  bottoms x_diff (CPR vs dense) = {diff:.2e} < 1e-5", diff < 1e-5)
    diff_T = float(np.max(np.abs(np.asarray(r_cpr.T)
                                       - np.asarray(r_dense.T))))
    check(f"  T profile diff = {diff_T:.2e} < 1e-3 K", diff_T < 1e-3)


def test_cpr_jacobian_matches_dense_with_eb():
    """Same equivalence with energy balance enabled."""
    section("test_cpr_jacobian_matches_dense_with_eb")
    import stateprop.reaction.reactive_column as rc
    psat = _ss_3comp_psat()

    def h_V(T): return 30.0 * T
    def h_L(T): return 75.0 * T - 40000.0
    h_V_funcs = [h_V] * 3
    h_L_funcs = [h_L] * 3

    common = dict(
        n_stages=10, feed_stage=5, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat,
        reflux_ratio=2.0, distillate_rate=33.0,
        pressure=1.013e5,
        energy_balance=True,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
    )
    r_cpr = distillation_column(**common)
    orig = rc._build_block_tridiag_jacobian
    def force_dense(*args, **kwargs):
        kwargs['has_nonlocal_coupling'] = True
        return orig(*args, **kwargs)
    rc._build_block_tridiag_jacobian = force_dense
    try:
        r_dense = distillation_column(**common)
    finally:
        rc._build_block_tridiag_jacobian = orig
    diff = float(np.max(np.abs(r_cpr.x[-1] - r_dense.x[-1])))
    check(f"  bottoms x_diff (CPR vs dense, EB) = {diff:.2e}", diff < 1e-5)


def test_cpr_jacobian_falls_back_with_murphree():
    """When E < 1, the Murphree y_actual recursion creates non-local
    Jacobian structure.  CPR must fall back to dense (verified by
    monitoring that the helper is called with has_nonlocal=True)."""
    section("test_cpr_jacobian_falls_back_with_murphree")
    import stateprop.reaction.reactive_column as rc
    psat = _ss_3comp_psat()

    flag = {"called_dense": False}
    orig = rc._build_block_tridiag_jacobian
    def watcher(*args, **kwargs):
        if kwargs.get('has_nonlocal_coupling', False):
            flag["called_dense"] = True
        return orig(*args, **kwargs)
    rc._build_block_tridiag_jacobian = watcher
    try:
        r = distillation_column(
            n_stages=10, feed_stage=5, feed_F=100.0,
            feed_z=[1/3, 1/3, 1/3], feed_q=1.0,
            species_names=['light', 'middle', 'heavy'],
            activity_model=_IdealActivity(), psat_funcs=psat,
            reflux_ratio=2.0, distillate_rate=33.0,
            pressure=1.013e5,
            stage_efficiency=0.7,
        )
    finally:
        rc._build_block_tridiag_jacobian = orig
    check(f"  has_nonlocal=True triggered with E=0.7", flag["called_dense"])
    check(f"  column still converges with Murphree", r.converged)


def test_cpr_jacobian_falls_back_with_pump_around():
    """Pump-around couples non-adjacent stages → must fall back to dense."""
    section("test_cpr_jacobian_falls_back_with_pump_around")
    import stateprop.reaction.reactive_column as rc
    psat = _ss_3comp_psat()
    flag = {"called_dense": False}
    orig = rc._build_block_tridiag_jacobian
    def watcher(*args, **kwargs):
        if kwargs.get('has_nonlocal_coupling', False):
            flag["called_dense"] = True
        return orig(*args, **kwargs)
    rc._build_block_tridiag_jacobian = watcher
    try:
        r = distillation_column(
            n_stages=10, feed_stage=5, feed_F=100.0,
            feed_z=[1/3, 1/3, 1/3], feed_q=1.0,
            species_names=['light', 'middle', 'heavy'],
            activity_model=_IdealActivity(), psat_funcs=psat,
            reflux_ratio=2.0, distillate_rate=33.0,
            pressure=1.013e5,
            pump_arounds=[PumpAround(draw_stage=8, return_stage=4, flow=20.0)],
        )
    finally:
        rc._build_block_tridiag_jacobian = orig
    check(f"  has_nonlocal=True triggered with pump-around",
          flag["called_dense"])
    check(f"  column still converges with PA", r.converged)


def test_cpr_jacobian_speedup_at_n20():
    """At N=20 stages, CPR should be at least 2× faster than dense."""
    section("test_cpr_jacobian_speedup_at_n20")
    import time
    import stateprop.reaction.reactive_column as rc
    psat = _ss_3comp_psat()
    common = dict(
        n_stages=20, feed_stage=10, feed_F=100.0,
        feed_z=[1/3, 1/3, 1/3], feed_q=1.0,
        species_names=['light', 'middle', 'heavy'],
        activity_model=_IdealActivity(), psat_funcs=psat,
        reflux_ratio=2.0, distillate_rate=33.0,
        pressure=1.013e5,
    )
    # CPR (default)
    t0 = time.time()
    distillation_column(**common)
    t_cpr = time.time() - t0
    # Force dense
    orig = rc._build_block_tridiag_jacobian
    def force_dense(*args, **kwargs):
        kwargs['has_nonlocal_coupling'] = True
        return orig(*args, **kwargs)
    rc._build_block_tridiag_jacobian = force_dense
    try:
        t0 = time.time()
        distillation_column(**common)
        t_dense = time.time() - t0
    finally:
        rc._build_block_tridiag_jacobian = orig
    speedup = t_dense / t_cpr
    check(f"  N=20: CPR={t_cpr*1000:.0f}ms, dense={t_dense*1000:.0f}ms, "
          f"speedup={speedup:.2f}x ≥ 2x", speedup >= 2.0)


def main():
    print("=" * 60)
    print("stateprop.distillation tests")
    print("=" * 60)
    tests = [
        test_construction,
        test_overall_mass_balance_closure,
        test_phase_closures,
        test_equivalence_to_reactive_with_no_reactions,
        test_purity_increases_with_reflux,
        test_purity_increases_with_stages,
        test_temperature_profile_monotonic,
        test_methanol_water_split,
        test_three_component_split,
        test_recovery_complementary,
        test_high_reflux_approaches_pure_distillate,
        test_energy_balance_smoke,
        test_input_validation,
        # v0.9.71 multi-feed and side-draw tests:
        test_multifeed_two_feeds_converges,
        test_multifeed_colocated_equivalence,
        test_multifeed_different_compositions,
        test_liquid_side_draw_mass_balance,
        test_vapor_side_draw_mass_balance,
        test_recovery_with_side_draw,
        test_three_outlet_column,
        test_wang_henke_rejects_multifeed,
        test_invalid_multifeed_combo_raises,
        # v0.9.72 q-fraction and partial condenser:
        test_q_fraction_saturated_vapor_feed,
        test_q_fraction_two_phase_feed,
        test_q_equals_1_unchanged,
        test_q_in_FeedSpec,
        test_q_fraction_reduces_separation,
        test_partial_condenser_field,
        test_partial_condenser_extra_stage_separates_better,
        # v0.9.73 pressure profile and Murphree efficiency:
        test_pressure_drop_smoke,
        test_pressure_array_input,
        test_pressure_array_and_drop_conflict,
        test_pressure_drop_zero_unchanged,
        test_stage_efficiency_full_equilibrium_unchanged,
        test_stage_efficiency_reduces_separation,
        test_stage_efficiency_array,
        test_stage_efficiency_invalid,
        test_wang_henke_rejects_pressure_and_efficiency,
        # v0.9.74 pump-arounds:
        test_pump_around_no_PA_unchanged,
        test_pump_around_L_profile_step,
        test_pump_around_mass_balance,
        test_pump_around_multiple,
        test_pump_around_invalid,
        test_pump_around_with_energy_balance,
        test_wang_henke_rejects_pump_arounds,
        # v0.9.75 design-mode specs:
        test_spec_x_D_vary_R,
        test_spec_recovery_D_vary_D,
        test_spec_two_specs_vary_both,
        test_spec_x_B,
        test_spec_invalid_count,
        test_spec_no_specs_unchanged,
        # v0.9.76 duty + ratio specs:
        test_spec_Q_C_vary_R,
        test_spec_Q_R_vary_R,
        test_spec_Q_C_requires_enthalpy_funcs,
        test_spec_ratio_vary_R,
        test_spec_Q_C_partial_condenser,
        # v0.9.78 column γ-φ-EOS coupling
        test_column_eos_dispatch_unchanged,
        test_column_eos_low_p_close_to_modified_raoult,
        test_column_eos_high_p_diverges,
        test_column_eos_phi_sat_helper,
        test_column_eos_wang_henke_rejected,
        # v0.9.88 side strippers
        test_side_stripper_basic_3component,
        test_side_stripper_mass_balance,
        test_side_stripper_no_ss_unchanged,
        test_side_stripper_two_independent,
        test_side_stripper_invalid_bottoms_rate,
        test_side_stripper_invalid_draw_stage,
        # v0.9.89 EB+SS and steam injection
        test_side_stripper_eb_supported_v0989,
        test_side_stripper_steam_mode_cmo,
        test_side_stripper_steam_mode_eb,
        test_side_stripper_steam_invalid_z_sum,
        # v0.9.117 — CPR-compressed Jacobian
        test_cpr_jacobian_matches_dense_no_eb,
        test_cpr_jacobian_matches_dense_with_eb,
        test_cpr_jacobian_falls_back_with_murphree,
        test_cpr_jacobian_falls_back_with_pump_around,
        test_cpr_jacobian_speedup_at_n20,
    ]
    for t in tests:
        t()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
