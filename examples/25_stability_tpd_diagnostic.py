"""4-quadrant Michelsen TPD stability diagnostic.

Demonstrates the 4 stability tests that together provide unambiguous
phase-count detection (the foundation under v0.9.53 auto_isothermal_full_tpd):

  1. L->L stability test (v0.9.48): liquid candidate, liquid trial,
                                       gamma-based.
  2. V->V stability test (v0.9.51): vapor candidate, vapor trial,
                                       phi-based.
  3. L->V cross-phase test (v0.9.52): liquid candidate, vapor trial,
                                         gamma + phi jointly.
  4. V->L cross-phase test (v0.9.52): vapor candidate, liquid trial,
                                         gamma + phi jointly.

For each system, all four flags are reported. The combination of stable
(S) / unstable (U) flags gives the phase-count regime per the 5-phase
truth table.
"""
from __future__ import annotations

import numpy as np
from stateprop.activity import (UNIFAC_LLE, AntoinePsat, stability_test,
                                 vapor_phase_stability_test,
                                 cross_phase_stability_test)
from stateprop.activity.compounds import make_unifac
from stateprop.cubic import PR, CubicMixture


def run_4_tests(name, T, p, z, activity_model, eos, psat_funcs, true_phase=""):
    """Run all 4 Michelsen TPD tests and print a result line."""
    z = np.asarray(z)
    print(f"\n  {name}: T={T:.1f} K, p={p/1e5:.3f} bar, z={z}")
    if true_phase:
        print(f"    Expected: {true_phase}")

    # 1. L->L
    r_LL = stability_test(activity_model, T, z)
    flag_LL = "S" if r_LL.stable else "U"

    # 2. V->V
    r_VV = vapor_phase_stability_test(eos, T, p, z)
    flag_VV = "S" if r_VV.stable else "U"

    # 3. L->V (liquid candidate, vapor trial)
    r_LV = cross_phase_stability_test(activity_model, eos, psat_funcs,
                                          T, p, z, candidate_phase='liquid')
    flag_LV = "S" if r_LV.stable else "U"

    # 4. V->L (vapor candidate, liquid trial)
    r_VL = cross_phase_stability_test(activity_model, eos, psat_funcs,
                                          T, p, z, candidate_phase='vapor')
    flag_VL = "S" if r_VL.stable else "U"

    pattern = f"{flag_LL}{flag_VV}{flag_LV}{flag_VL}"
    table = {
        "SSSU": "1L (single liquid)",
        "SSUS": "1V (single vapor)",
        "SSUU": "2VL (vapor + liquid)",
        "USSU": "2LL (two liquids, no vapor)",
        "USUU": "3VLL (vapor + two liquids)",
        "SSSS": "1-phase (degenerate; needs bubble-p check)",
    }
    interpretation = table.get(pattern, "(unrecognized pattern)")

    print(f"    LL={flag_LL}  VV={flag_VV}  LV={flag_LV}  VL={flag_VL}  ->  "
          f"{interpretation}")
    print(f"      LL TPD_min={r_LL.tpd_min:.3e}, "
          f"VV TPD_min={r_VV.tpd_min:.3e}")
    print(f"      LV TPD_min={r_LV.tpd_min:.3e}, "
          f"VL TPD_min={r_VL.tpd_min:.3e}")


def main():
    print("=" * 70)
    print("4-quadrant Michelsen TPD stability diagnostic")
    print("=" * 70)

    # System A: ethanol + water (fully miscible)
    eth_psat = AntoinePsat(A=5.37229, B=1670.409, C=-40.191)
    wa_psat  = AntoinePsat(A=4.6543,  B=1435.264, C=-64.848)
    ethanol = PR(T_c=513.92, p_c=6.137e6, acentric_factor=0.6452)
    water   = PR(T_c=647.096, p_c=22.064e6, acentric_factor=0.3443)
    pr_A = CubicMixture([ethanol, water], composition=[0.5, 0.5])
    uf_A = make_unifac(["ethanol", "water"])

    print("\nSystem A: Ethanol + water (fully miscible)")
    print("-" * 70)
    run_4_tests("Compressed liquid", T=298.0, p=10e5, z=[0.5, 0.5],
                  activity_model=uf_A, eos=pr_A,
                  psat_funcs=[eth_psat, wa_psat],
                  true_phase="1L")
    run_4_tests("Superheated vapor", T=400.0, p=0.5e5, z=[0.5, 0.5],
                  activity_model=uf_A, eos=pr_A,
                  psat_funcs=[eth_psat, wa_psat],
                  true_phase="1V")
    run_4_tests("Atmospheric VLE", T=355.0, p=101325.0, z=[0.5, 0.5],
                  activity_model=uf_A, eos=pr_A,
                  psat_funcs=[eth_psat, wa_psat],
                  true_phase="2VL")

    # System B: n-butanol + water (partially miscible)
    bu_psat = AntoinePsat(A=4.55139, B=1351.555, C=-93.34)
    butanol = PR(T_c=563.05, p_c=4.414e6, acentric_factor=0.589)
    pr_B = CubicMixture([butanol, water], composition=[0.5, 0.5])
    uf_B = UNIFAC_LLE([{'CH3': 1, 'CH2': 3, 'OH': 1}, {'H2O': 1}])

    print("\nSystem B: n-butanol + water (partially miscible)")
    print("-" * 70)
    run_4_tests("Below boiling, atmospheric", T=298.0, p=101325.0,
                  z=[0.5, 0.5],
                  activity_model=uf_B, eos=pr_B,
                  psat_funcs=[bu_psat, wa_psat],
                  true_phase="2LL")
    run_4_tests("High pressure equimolar", T=298.0, p=10e5, z=[0.5, 0.5],
                  activity_model=uf_B, eos=pr_B,
                  psat_funcs=[bu_psat, wa_psat],
                  true_phase="2LL")

    print("\n" + "=" * 70)
    print("Reference: 4-test pattern truth table")
    print("=" * 70)
    print("""
  +------+------+------+------+------------------------+
  |  LL  |  VV  |  LV  |  VL  | Phase count            |
  +------+------+------+------+------------------------+
  |  S   |  S   |  S   |  U   | 1L (single liquid)     |
  |  S   |  S   |  U   |  S   | 1V (single vapor)      |
  |  S   |  S   |  U   |  U   | 2VL                    |
  |  U   |  S   |  S   |  U   | 2LL                    |
  |  U   |  S   |  U   |  U   | 3VLL                   |
  +------+------+------+------+------------------------+
""")


if __name__ == "__main__":
    main()
