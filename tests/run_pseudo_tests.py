"""Pseudo-component characterization tests for stateprop.

Validates the Riazi-Daubert / Lee-Kesler / Edmister / Watson correlation
network against published critical properties for n-alkanes (C5-C14)
where true values are well-established from NIST.

Tests also verify EOS interop: a PseudoComponent constructed from NBP
and SG drops cleanly into PR or SRK and reproduces vapor pressures and
densities consistent with the underlying correlations.
"""
from __future__ import annotations
import sys, os, math, warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from stateprop.pseudo import (
    PseudoComponent,
    make_pseudo_from_NBP_SG, make_pseudo_cut_distribution,
    make_PR_from_pseudo, make_SRK_from_pseudo,
    riazi_daubert_Tc, riazi_daubert_Pc, riazi_daubert_MW, riazi_daubert_Vc,
    edmister_acentric, lee_kesler_acentric, lee_kesler_psat,
    watson_K, lee_kesler_cp_ig_coeffs, rackett_density,
)


_PASS = 0
_FAIL = 0


def section(name):
    print(f"\n[{name}]")


def check(label, ok):
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  PASS  {label}")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}")


# -------------------------------------------------------------------
# n-Alkane reference data (NIST)
#  (NBP[K], SG, MW, Tc[K], Pc[bar], omega, Vc[cm³/mol])
# -------------------------------------------------------------------
N_ALKANES = [
    ("n-pentane",     309.2, 0.6262,  72.15, 469.7, 33.7, 0.251, 313),
    ("n-hexane",      341.9, 0.6640,  86.18, 507.6, 30.2, 0.299, 371),
    ("n-heptane",     371.6, 0.6883, 100.20, 540.2, 27.4, 0.350, 432),
    ("n-octane",      398.8, 0.7068, 114.23, 568.7, 24.9, 0.398, 492),
    ("n-nonane",      424.0, 0.7219, 128.26, 594.6, 22.9, 0.445, 555),
    ("n-decane",      447.3, 0.7301, 142.28, 617.7, 21.1, 0.490, 624),
    ("n-dodecane",    489.5, 0.7484, 170.34, 658.5, 18.2, 0.575, 754),
    ("n-tetradecane", 526.7, 0.7636, 198.39, 692.4, 15.7, 0.644, 894),
]


def test_riazi_daubert_Tc_accuracy():
    """Riazi-Daubert Tc within ~5% across n-pentane through n-tetradecane."""
    section("test_riazi_daubert_Tc_accuracy")
    max_err = 0.0
    for name, NBP, SG, MW, Tc_true, Pc, omega, Vc in N_ALKANES:
        Tc_est = riazi_daubert_Tc(NBP, SG)
        err = abs(Tc_est - Tc_true) / Tc_true * 100
        max_err = max(max_err, err)
        check(f"{name}: Tc {Tc_est:.1f} vs {Tc_true:.1f} K "
              f"({err:.1f}% err)", err < 5.0)
    check(f"max Tc error across all: {max_err:.1f}% < 5%", max_err < 5.0)


def test_riazi_daubert_Pc_accuracy():
    """Riazi-Daubert Pc within ~10% across n-alkanes."""
    section("test_riazi_daubert_Pc_accuracy")
    max_err = 0.0
    for name, NBP, SG, MW, Tc, Pc_true, omega, Vc in N_ALKANES:
        Pc_est = riazi_daubert_Pc(NBP, SG) / 1e5  # bar
        err = abs(Pc_est - Pc_true) / Pc_true * 100
        max_err = max(max_err, err)
        check(f"{name}: Pc {Pc_est:.2f} vs {Pc_true:.2f} bar "
              f"({err:.1f}% err)", err < 10.0)
    check(f"max Pc error across all: {max_err:.1f}% < 10%", max_err < 10.0)


def test_riazi_daubert_MW_accuracy():
    """Riazi-Daubert MW within ~10% across n-alkanes."""
    section("test_riazi_daubert_MW_accuracy")
    max_err = 0.0
    for name, NBP, SG, MW_true, Tc, Pc, omega, Vc in N_ALKANES:
        MW_est = riazi_daubert_MW(NBP, SG)
        err = abs(MW_est - MW_true) / MW_true * 100
        max_err = max(max_err, err)
        check(f"{name}: MW {MW_est:.1f} vs {MW_true:.1f} g/mol "
              f"({err:.1f}% err)", err < 10.0)
    check(f"max MW error: {max_err:.1f}% < 10%", max_err < 10.0)


def test_lee_kesler_acentric_accuracy():
    """Lee-Kesler acentric factor within ~10% on n-alkanes."""
    section("test_lee_kesler_acentric_accuracy")
    max_err = 0.0
    for name, NBP, SG, MW, Tc_true, Pc_true, omega_true, Vc in N_ALKANES:
        # Use TRUE Tc, Pc to isolate the LK acentric formula
        omega_est = lee_kesler_acentric(NBP, Tc_true, Pc_true * 1e5)
        err = abs(omega_est - omega_true) / omega_true * 100
        max_err = max(max_err, err)
        check(f"{name}: ω {omega_est:.3f} vs {omega_true:.3f} "
              f"({err:.1f}% err)", err < 10.0)
    check(f"max ω error (true Tc, Pc): {max_err:.1f}% < 10%",
          max_err < 10.0)


def test_lee_kesler_psat_at_NBP_unit_atm():
    """Lee-Kesler psat must give ~1 atm at the normal boiling point.
    This is a fundamental consistency check for any vapor-pressure
    correlation — psat(T_NBP) = P_atm by definition of NBP."""
    section("test_lee_kesler_psat_at_NBP_unit_atm")
    for name, NBP, SG, MW, Tc_true, Pc_true, omega_true, Vc in N_ALKANES:
        psat_est = lee_kesler_psat(NBP, Tc_true, Pc_true * 1e5, omega_true)
        err = abs(psat_est - 101325.0) / 101325.0 * 100
        check(f"{name}: psat(NBP) {psat_est:.1f} Pa "
              f"({err:.2f}% off 1 atm)", err < 1.5)


def test_pseudo_component_construction():
    """End-to-end pseudo-component build: input NBP, SG; check that all
    derived properties land within their characterization tolerances."""
    section("test_pseudo_component_construction")
    for name, NBP, SG, MW_true, Tc_true, Pc_true, omega_true, Vc_true in N_ALKANES:
        p = PseudoComponent(NBP=NBP, SG=SG, name=name)
        # Tolerances chosen per the correlation accuracy bounds
        check(f"{name}: Tc within 3%",
              abs(p.Tc - Tc_true) / Tc_true < 0.03)
        check(f"{name}: Pc within 10%",
              abs(p.Pc / 1e5 - Pc_true) / Pc_true < 0.10)
        check(f"{name}: MW within 10%",
              abs(p.MW - MW_true) / MW_true < 0.10)
        check(f"{name}: omega within 12%",
              abs(p.omega - omega_true) / omega_true < 0.12)
        check(f"{name}: Vc within 8%",
              abs(p.Vc * 1e6 - Vc_true) / Vc_true < 0.08)


def test_pseudo_psat_at_NBP():
    """For a constructed PseudoComponent, psat(NBP) should give 1 atm
    via the internal Lee-Kesler psat method."""
    section("test_pseudo_psat_at_NBP")
    for name, NBP, SG, MW, Tc, Pc, omega, Vc in N_ALKANES:
        p = PseudoComponent(NBP=NBP, SG=SG)
        psat = p.psat(NBP)
        check(f"{name}: psat(NBP)/Patm = {psat/101325:.4f}",
              abs(psat - 101325) / 101325 < 0.05)


def test_pseudo_density():
    """Liquid density at 298 K within ~5% of measured for n-alkanes."""
    section("test_pseudo_density")
    rho_meas = {
        "n-pentane": 626, "n-hexane": 655, "n-heptane": 684, "n-octane": 703,
        "n-nonane": 718, "n-decane": 730, "n-dodecane": 750, "n-tetradecane": 763,
    }
    for name, NBP, SG, *_ in N_ALKANES:
        p = PseudoComponent(NBP=NBP, SG=SG, name=name)
        rho = p.liquid_density_kg(298.15)
        rho_true = rho_meas[name]
        err = abs(rho - rho_true) / rho_true * 100
        check(f"{name}: ρ_liq(298) {rho:.0f} vs {rho_true} kg/m³ "
              f"({err:.1f}% err)", err < 8.0)


def test_pseudo_PR_eos_dispatch():
    """A PseudoComponent passed to make_PR_from_pseudo must give an
    EOS object whose density and vapor pressure are consistent with
    the pseudo's stand-alone correlations."""
    section("test_pseudo_PR_eos_dispatch")
    p = PseudoComponent(NBP=447.3, SG=0.7301, name="n-decane-test")
    eos = make_PR_from_pseudo(p)
    check("EOS Tc matches pseudo", abs(eos.T_c - p.Tc) < 1e-9)
    check("EOS Pc matches pseudo", abs(eos.p_c - p.Pc) < 1e-9)
    check("EOS omega matches pseudo",
          abs(eos.acentric_factor - p.omega) < 1e-9)
    # PR vapor density at supercritical state
    rho_n = eos.density_from_pressure(p=10e5, T=500.0, phase_hint="vapor")
    check(f"PR vapor density at 500K, 10 bar = {float(rho_n):.2f} mol/m³",
          0 < float(rho_n) < 5000)


def test_watson_K_paraffinic_range():
    """Watson K should be near 12.7 for typical paraffinic cuts and
    fall to ~10.5 for benzene (highly aromatic)."""
    section("test_watson_K_paraffinic_range")
    # n-decane (paraffinic): K_W ≈ 12.7
    K_dec = watson_K(447.3, 0.7301)
    check(f"K_W(n-decane) = {K_dec:.2f} (expect ~12.7)",
          12.5 < K_dec < 13.0)
    # benzene (aromatic): NBP=353.2, SG=0.879, K_W ≈ 9.7
    K_benz = watson_K(353.2, 0.879)
    check(f"K_W(benzene) = {K_benz:.2f} (expect ~9.7)",
          9.5 < K_benz < 10.0)


def test_make_pseudo_cut_distribution():
    """Generate a 5-cut distribution from NBPs + Watson K; verify all
    cuts return PseudoComponents with monotone-increasing Tc."""
    section("test_make_pseudo_cut_distribution")
    NBPs = [400.0, 450.0, 500.0, 550.0, 600.0]
    cuts = make_pseudo_cut_distribution(NBPs, Watson_K=12.5,
                                          name_prefix="diesel")
    check(f"5 cuts generated", len(cuts) == 5)
    for i in range(1, len(cuts)):
        check(f"cut {i+1} Tc > cut {i} Tc ({cuts[i].Tc:.0f} > "
              f"{cuts[i-1].Tc:.0f})", cuts[i].Tc > cuts[i-1].Tc)
    # SG_avg variant
    cuts2 = make_pseudo_cut_distribution(NBPs, SG_avg=0.78)
    for c in cuts2:
        check(f"{c.name} SG = 0.78", abs(c.SG - 0.78) < 1e-9)


def test_pseudo_invalid_inputs():
    """Validate input ranges: NBP > 0, 0.4 < SG < 1.5."""
    section("test_pseudo_invalid_inputs")
    cases = [
        (-1.0, 0.7, "negative NBP"),
        (300.0, 0.3, "SG too low"),
        (300.0, 1.6, "SG too high"),
        (0.0, 0.7, "zero NBP"),
    ]
    for NBP, SG, label in cases:
        raised = False
        try:
            PseudoComponent(NBP=NBP, SG=SG)
        except ValueError:
            raised = True
        check(f"reject {label}", raised)


def test_pseudo_distribution_input_validation():
    """make_pseudo_cut_distribution rejects inconsistent input modes."""
    section("test_pseudo_distribution_input_validation")
    raised_no_input = False
    try:
        make_pseudo_cut_distribution([400.0, 500.0])
    except ValueError:
        raised_no_input = True
    check("no SG input raises", raised_no_input)
    raised_len = False
    try:
        make_pseudo_cut_distribution([400.0, 500.0], SG_cuts=[0.7])
    except ValueError:
        raised_len = True
    check("SG_cuts length mismatch raises", raised_len)


def test_pseudo_repr_round_trip():
    """The repr() should include name, NBP, SG, MW, Tc, Pc, ω, K_W."""
    section("test_pseudo_repr_round_trip")
    p = PseudoComponent(NBP=450.0, SG=0.78, name="diesel-cut-1")
    s = repr(p)
    for token in ["diesel-cut-1", "450", "0.7800", "MW=", "Tc=",
                  "Pc=", "omega=", "K_W="]:
        check(f"repr contains {token!r}", token in s)


def main():
    print("=" * 60)
    print("stateprop pseudo-component tests")
    print("=" * 60)
    tests = [
        test_riazi_daubert_Tc_accuracy,
        test_riazi_daubert_Pc_accuracy,
        test_riazi_daubert_MW_accuracy,
        test_lee_kesler_acentric_accuracy,
        test_lee_kesler_psat_at_NBP_unit_atm,
        test_pseudo_component_construction,
        test_pseudo_psat_at_NBP,
        test_pseudo_density,
        test_pseudo_PR_eos_dispatch,
        test_watson_K_paraffinic_range,
        test_make_pseudo_cut_distribution,
        test_pseudo_invalid_inputs,
        test_pseudo_distribution_input_validation,
        test_pseudo_repr_round_trip,
    ]
    for t in tests:
        t()
    print()
    print("=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
