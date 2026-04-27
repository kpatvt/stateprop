"""TBP discretization tests for stateprop."""
from __future__ import annotations
import sys, os, math, warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from stateprop.tbp import (
    discretize_TBP, TBPDiscretization,
    discretize_from_D86, discretize_from_D2887,
    interpolate_TBP, D86_to_TBP, D2887_to_TBP,
    API_to_SG, SG_to_API, watson_K_to_SG,
)
from stateprop.pseudo import PseudoComponent

_PASS = 0
_FAIL = 0

def section(n): print(f"\n[{n}]")
def check(label, ok):
    global _PASS, _FAIL
    if ok: _PASS += 1; print(f"  PASS  {label}")
    else: _FAIL += 1; print(f"  FAIL  {label}")


# Standard diesel TBP curve for use across tests
DIESEL_VOLS = [0, 10, 30, 50, 70, 90, 100]
DIESEL_NBPs = [380, 430, 480, 510, 540, 580, 620]


def test_API_SG_round_trip():
    section("test_API_SG_round_trip")
    for SG in [0.65, 0.75, 0.85, 0.95]:
        API = SG_to_API(SG)
        SG_back = API_to_SG(API)
        check(f"SG={SG} -> API={API:.2f} -> SG={SG_back:.6f}",
              abs(SG_back - SG) < 1e-12)
    # Specific landmark: water = 10° API
    check(f"water (SG=1.0) = 10°API", abs(SG_to_API(1.0) - 10.0) < 1e-9)
    # Light crude (35° API)
    check(f"35°API → SG=0.8498",
          abs(API_to_SG(35.0) - 0.8498) < 1e-3)


def test_watson_K_to_SG_inverse():
    section("test_watson_K_to_SG_inverse")
    from stateprop.pseudo import watson_K
    for NBP in [350, 450, 550, 650]:
        for K_W in [10.5, 11.5, 12.5, 13.0]:
            SG = watson_K_to_SG(NBP, K_W)
            K_back = watson_K(NBP, SG)
            check(f"NBP={NBP}, K_W={K_W}: round-trip K={K_back:.4f}",
                  abs(K_back - K_W) < 1e-6)


def test_interpolate_TBP_endpoints():
    section("test_interpolate_TBP_endpoints")
    T0 = interpolate_TBP(0.0, DIESEL_VOLS, DIESEL_NBPs)
    T100 = interpolate_TBP(100.0, DIESEL_VOLS, DIESEL_NBPs)
    check(f"T(0%) = {T0} (expect 380)", abs(T0 - 380) < 1e-9)
    check(f"T(100%) = {T100} (expect 620)", abs(T100 - 620) < 1e-9)
    T50 = interpolate_TBP(50.0, DIESEL_VOLS, DIESEL_NBPs)
    check(f"T(50%) = {T50} (expect 510)", abs(T50 - 510) < 1e-9)
    # In-between point
    T20 = interpolate_TBP(20.0, DIESEL_VOLS, DIESEL_NBPs)
    # 20% is midway between (10%, 430) and (30%, 480) → 455
    check(f"T(20%) = {T20} (expect 455 by linear interp)",
          abs(T20 - 455) < 1e-9)


def test_interpolate_TBP_out_of_range():
    section("test_interpolate_TBP_out_of_range")
    raised_low = False; raised_high = False
    try: interpolate_TBP(-5, DIESEL_VOLS, DIESEL_NBPs)
    except ValueError: raised_low = True
    try: interpolate_TBP(105, DIESEL_VOLS, DIESEL_NBPs)
    except ValueError: raised_high = True
    check("rejects volume < range_min", raised_low)
    check("rejects volume > range_max", raised_high)


def test_discretize_equal_volume_basic():
    section("test_discretize_equal_volume_basic")
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=6,
                          API_gravity=35.0)
    check(f"6 cuts created", res.n_cuts == 6)
    check(f"method recorded", res.method == "equal_volume")
    # Sum of fractions = 1
    check(f"vol_frac sums to 1", abs(res.volume_fractions.sum() - 1) < 1e-12)
    check(f"mass_frac sums to 1", abs(res.mass_fractions.sum() - 1) < 1e-12)
    check(f"mole_frac sums to 1", abs(res.mole_fractions.sum() - 1) < 1e-12)
    # Equal volume → all vol_frac equal
    check(f"all vol_frac == 1/6",
          np.allclose(res.volume_fractions, 1/6, atol=1e-12))
    # NBP_lo[0] = T_min, NBP_hi[-1] = T_max
    check(f"NBP_lo[0] = T_min (380)", abs(res.NBP_lower[0] - 380) < 1e-9)
    check(f"NBP_hi[-1] = T_max (620)", abs(res.NBP_upper[-1] - 620) < 1e-9)
    # Cuts NBP monotone increasing
    NBPs_cut = [c.NBP for c in res.cuts]
    check(f"cut NBPs monotone increasing",
          all(NBPs_cut[i] < NBPs_cut[i+1] for i in range(5)))


def test_discretize_volume_continuity():
    section("test_discretize_volume_continuity")
    """Adjacent cuts must share NBP at their boundary."""
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=8,
                          API_gravity=35.0)
    for i in range(7):
        check(f"NBP_hi[{i}] == NBP_lo[{i+1}]",
              abs(res.NBP_upper[i] - res.NBP_lower[i+1]) < 1e-9)


def test_discretize_equal_NBP():
    section("test_discretize_equal_NBP")
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=6,
                          API_gravity=35.0, method="equal_NBP")
    check(f"method = equal_NBP", res.method == "equal_NBP")
    check(f"6 cuts", res.n_cuts == 6)
    # Cut NBP differences should all be ~equal: (620-380)/6 = 40 K
    cut_NBPs = [c.NBP for c in res.cuts]
    NBP_widths = res.NBP_upper - res.NBP_lower
    check(f"NBP widths roughly equal (~40 K each)",
          np.allclose(NBP_widths, 40.0, atol=1e-9))
    # Sum of fractions still = 1
    check(f"vol_frac sums to 1", abs(res.volume_fractions.sum() - 1) < 1e-12)


def test_discretize_with_SG_table():
    section("test_discretize_with_SG_table")
    SG_table = [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=6,
                          SG_table=SG_table)
    check(f"6 cuts with per-volume SG", res.n_cuts == 6)
    # SG should be monotone increasing
    SGs = [c.SG for c in res.cuts]
    check(f"SGs monotone increasing across cuts",
          all(SGs[i] < SGs[i+1] for i in range(5)))


def test_discretize_with_Watson_K():
    section("test_discretize_with_Watson_K")
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=5,
                          Watson_K=12.0)
    # Each cut's SG must satisfy K = (1.8*NBP)^(1/3)/SG = 12.0
    for c in res.cuts:
        K_computed = (1.8 * c.NBP) ** (1.0/3.0) / c.SG
        check(f"cut {c.name}: K_W = {K_computed:.4f}",
              abs(K_computed - 12.0) < 1e-9)


def test_discretize_invalid_inputs():
    section("test_discretize_invalid_inputs")
    # Multiple SG specs → reject
    raised = False
    try:
        discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, 5,
                        SG_avg=0.85, Watson_K=12.0)
    except ValueError: raised = True
    check("rejects multiple SG specs", raised)

    # No SG spec → reject
    raised = False
    try:
        discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, 5)
    except ValueError: raised = True
    check("rejects missing SG spec", raised)

    # n_cuts <= 0 → reject
    raised = False
    try:
        discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, 0, SG_avg=0.85)
    except ValueError: raised = True
    check("rejects n_cuts=0", raised)

    # Mismatched SG_table length → reject
    raised = False
    try:
        discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, 5,
                        SG_table=[0.7, 0.8, 0.9])
    except ValueError: raised = True
    check("rejects SG_table length mismatch", raised)


def test_discretize_invalid_TBP_table():
    section("test_discretize_invalid_TBP_table")
    # Non-monotone volumes
    raised = False
    try: discretize_TBP([400, 500, 600], [0, 30, 20], 3, SG_avg=0.8)
    except ValueError: raised = True
    check("rejects non-monotone volumes", raised)

    # Negative temperature
    raised = False
    try: discretize_TBP([-5, 500, 600], [0, 50, 100], 3, SG_avg=0.8)
    except ValueError: raised = True
    check("rejects negative temperature", raised)

    # Non-monotone temperatures
    raised = False
    try: discretize_TBP([500, 400, 600], [0, 50, 100], 3, SG_avg=0.8)
    except ValueError: raised = True
    check("rejects non-monotone temperatures", raised)

    # Single-point table
    raised = False
    try: discretize_TBP([500], [50], 1, SG_avg=0.8)
    except ValueError: raised = True
    check("rejects single-point table", raised)


def test_discretize_n_cuts_scaling():
    section("test_discretize_n_cuts_scaling")
    """As n_cuts increases, the spread of NBP within each cut shrinks
    and the average cut NBP approaches the linearly-interpolated curve."""
    for n in [3, 6, 12, 20]:
        res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n,
                              API_gravity=35.0)
        widths = res.NBP_upper - res.NBP_lower
        # Equal-volume cuts → max width should not exceed (240 K total range)
        # / N times some factor (because the curve isn't linear in volume).
        avg_width = float(widths.mean())
        # Theoretical avg width: total NBP range (240) / N
        expected = 240.0 / n
        # Allow 30% tolerance for curve nonlinearity
        check(f"n={n}: avg cut width {avg_width:.1f} ≈ expected {expected:.1f}",
              abs(avg_width - expected) / expected < 0.30)


def test_pseudo_components_in_cuts_valid():
    section("test_pseudo_components_in_cuts_valid")
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=6,
                          API_gravity=35.0)
    for c in res.cuts:
        check(f"{c.name}: PseudoComponent",
              isinstance(c, PseudoComponent))
        check(f"{c.name}: Tc > NBP", c.Tc > c.NBP)
        check(f"{c.name}: Pc > 0", c.Pc > 0)
        check(f"{c.name}: 0.4 < SG < 1.5", 0.4 < c.SG < 1.5)
        check(f"{c.name}: MW > 0", c.MW > 0)


def test_D2887_to_TBP_close():
    section("test_D2887_to_TBP_close")
    """D2887 should be within ~5 K of TBP."""
    D2887 = [380, 430, 480, 510, 540, 580, 620]
    TBP = D2887_to_TBP(DIESEL_VOLS, D2887)
    deltas = [abs(TBP[i] - D2887[i]) for i in range(len(D2887))]
    check(f"max |D2887 - TBP| = {max(deltas):.2f} K (expect < 5 K)",
          max(deltas) < 5)
    # Endpoint corrections: at 0% TBP < D2887, at 100% TBP > D2887
    check(f"TBP[0] < D2887[0] (light end correction)", TBP[0] < D2887[0])
    check(f"TBP[-1] > D2887[-1] (heavy end correction)",
          TBP[-1] > D2887[-1])


def test_D86_to_TBP_runs():
    section("test_D86_to_TBP_runs")
    D86 = [380, 430, 480, 510, 540, 580, 620]
    TBP = D86_to_TBP(DIESEL_VOLS, D86)
    check(f"D86_to_TBP returns same length", len(TBP) == len(D86))
    check(f"all positive", (TBP > 0).all())


def test_discretize_from_D86():
    section("test_discretize_from_D86")
    D86 = [380, 430, 480, 510, 540, 580, 620]
    res = discretize_from_D86(DIESEL_VOLS, D86, n_cuts=5, API_gravity=35.0)
    check(f"5 cuts via D86", res.n_cuts == 5)
    check(f"vol_frac sums to 1",
          abs(res.volume_fractions.sum() - 1) < 1e-12)


def test_summary_includes_all_cuts():
    section("test_summary_includes_all_cuts")
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=4,
                          API_gravity=35.0, name_prefix="diesel")
    s = res.summary()
    for i in range(1, 5):
        check(f"summary contains diesel_{i}", f"diesel_{i}" in s)


def test_eos_dispatch_from_cuts():
    section("test_eos_dispatch_from_cuts")
    """Each cut's PseudoComponent must build a valid PR-EOS object."""
    from stateprop.pseudo import make_PR_from_pseudo
    res = discretize_TBP(DIESEL_NBPs, DIESEL_VOLS, n_cuts=4,
                          API_gravity=35.0)
    for c in res.cuts:
        eos = make_PR_from_pseudo(c)
        check(f"{c.name}: EOS Tc matches", abs(eos.T_c - c.Tc) < 1e-9)
        check(f"{c.name}: EOS Pc matches", abs(eos.p_c - c.Pc) < 1e-9)


def main():
    print("=" * 60)
    print("stateprop TBP discretization tests")
    print("=" * 60)
    tests = [
        test_API_SG_round_trip,
        test_watson_K_to_SG_inverse,
        test_interpolate_TBP_endpoints,
        test_interpolate_TBP_out_of_range,
        test_discretize_equal_volume_basic,
        test_discretize_volume_continuity,
        test_discretize_equal_NBP,
        test_discretize_with_SG_table,
        test_discretize_with_Watson_K,
        test_discretize_invalid_inputs,
        test_discretize_invalid_TBP_table,
        test_discretize_n_cuts_scaling,
        test_pseudo_components_in_cuts_valid,
        test_D2887_to_TBP_close,
        test_D86_to_TBP_runs,
        test_discretize_from_D86,
        test_summary_includes_all_cuts,
        test_eos_dispatch_from_cuts,
    ]
    for t in tests: t()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
