"""Automatic phase-count detection via Michelsen TPD framework.

Demonstrates the full 4-quadrant Michelsen tangent-plane-distance (TPD)
machinery (v0.9.48 + v0.9.51 + v0.9.52 + v0.9.53) that automatically
identifies which of 5 phase-count regimes a system is in:

  - 1L:    single liquid
  - 1V:    single vapor
  - 2VL:   ordinary VLE (vapor + liquid)
  - 2LL:   liquid-liquid only (no vapor present at this p)
  - 3VLL:  three-phase VLLE (vapor + two liquids)

Two systems are used to walk through the regimes:
  (A) ethanol + water (fully miscible) -> shows 1L, 1V, 2VL
  (B) n-butanol + water (partially miscible) -> shows 2LL, 3VLL
"""
from __future__ import annotations

import numpy as np
from stateprop.activity import (UNIFAC_LLE, AntoinePsat,
                                 GammaPhiEOSThreePhaseFlash)
from stateprop.activity.compounds import make_unifac
from stateprop.cubic import PR, CubicMixture


def _run_case(flash, T, p, z, desc):
    """Run auto-flash and print summary line."""
    try:
        r = flash.auto_isothermal_full_tpd(T=T, p=p, z=np.asarray(z),
                                              tol=1e-5, maxiter=80)
        zstr = f"({z[0]:.2f},{z[1]:.2f})"
        print(f"  {T:>6.1f} {p/1e5:>8.4f} {zstr:>14s}  "
              f"{r.phase_type:<6s}  {desc:<32s}")
        # Show compositions for multi-phase cases
        inner = r.result
        if r.phase_type == "2VL" and hasattr(inner, "x"):
            beta_v = getattr(inner, 'V', getattr(inner, 'beta', None))
            print(f"         beta_V={beta_v:.3f}, "
                  f"x={np.array2string(inner.x, precision=4, suppress_small=True)}, "
                  f"y={np.array2string(inner.y, precision=4, suppress_small=True)}")
        elif r.phase_type == "2LL" and hasattr(inner, "x1"):
            print(f"         beta={inner.beta:.3f}, "
                  f"x1={np.array2string(inner.x1, precision=4, suppress_small=True)}, "
                  f"x2={np.array2string(inner.x2, precision=4, suppress_small=True)}")
        elif r.phase_type == "3VLL" and hasattr(inner, "beta_V"):
            print(f"         beta_V={inner.beta_V:.3f}, "
                  f"beta_L1={inner.beta_L1:.3f}, "
                  f"beta_L2={inner.beta_L2:.3f}")
    except Exception as e:
        zstr = f"({z[0]:.2f},{z[1]:.2f})"
        print(f"  {T:>6.1f} {p/1e5:>8.4f} {zstr:>14s}  "
              f"failed - {str(e)[:40]}")


def main():
    # System A: ethanol + water (fully miscible)
    eth_psat = AntoinePsat(A=5.37229, B=1670.409, C=-40.191)
    wa_psat  = AntoinePsat(A=4.6543,  B=1435.264, C=-64.848)
    ethanol = PR(T_c=513.92, p_c=6.137e6, acentric_factor=0.6452)
    water   = PR(T_c=647.096, p_c=22.064e6, acentric_factor=0.3443)
    pr_A = CubicMixture([ethanol, water], composition=[0.5, 0.5])

    flash_A = GammaPhiEOSThreePhaseFlash(
        activity_model=make_unifac(["ethanol", "water"]),
        psat_funcs=[eth_psat, wa_psat],
        vapor_eos=pr_A,
    )

    print("=" * 70)
    print("System A: Ethanol + water (fully miscible) -> 1L, 1V, 2VL")
    print("=" * 70)
    print(f"\n  {'T(K)':>6s} {'p(bar)':>8s} {'z':>14s}  {'phase':<6s}  "
          f"description")
    print("  " + "-" * 70)
    _run_case(flash_A, T=298.0, p=10e5,    z=[0.5, 0.5], desc="298K, 10 bar (compressed liquid)")
    _run_case(flash_A, T=400.0, p=0.5e5,   z=[0.5, 0.5], desc="400K, 0.5 bar (superheated)")
    _run_case(flash_A, T=355.0, p=101325,  z=[0.5, 0.5], desc="355K, 1 atm (VLE)")
    _run_case(flash_A, T=358.0, p=101325,  z=[0.2, 0.8], desc="358K, 1 atm (water-rich VLE)")

    # System B: n-butanol + water (partially miscible)
    bu_psat = AntoinePsat(A=4.55139, B=1351.555, C=-93.34)
    butanol = PR(T_c=563.05, p_c=4.414e6, acentric_factor=0.589)
    pr_B = CubicMixture([butanol, water], composition=[0.5, 0.5])

    flash_B = GammaPhiEOSThreePhaseFlash(
        activity_model=UNIFAC_LLE([
            {'CH3': 1, 'CH2': 3, 'OH': 1}, {'H2O': 1}
        ]),
        psat_funcs=[bu_psat, wa_psat],
        vapor_eos=pr_B,
    )

    print("\n" + "=" * 70)
    print("System B: n-butanol + water (partially miscible) -> 2LL, 3VLL")
    print("=" * 70)
    print(f"\n  {'T(K)':>6s} {'p(bar)':>8s} {'z':>14s}  {'phase':<6s}  "
          f"description")
    print("  " + "-" * 70)
    _run_case(flash_B, T=298.0, p=10e5,    z=[0.5, 0.5], desc="298K, 10 bar (LLE only)")
    _run_case(flash_B, T=298.0, p=101325,  z=[0.5, 0.5], desc="298K, 1 atm (LLE only)")
    _run_case(flash_B, T=370.0, p=101325,  z=[0.5, 0.5], desc="370K, 1 atm (3-phase region)")

    print("\n" + "=" * 70)
    print("How auto-detection works: 4-test pattern matching")
    print("=" * 70)
    print("""
  +------+------+------+------+------------------------+
  |  LL  |  VV  |  LV  |  VL  | Phase count            |
  +------+------+------+------+------------------------+
  |  S   |  S   |  S   |  U   | 1L (single liquid)     |
  |  S   |  S   |  U   |  S   | 1V (single vapor)      |
  |  S   |  S   |  U   |  U   | 2VL                    |
  |  U   |  S   |  S   |  U   | 2LL (no vapor present) |
  |  U   |  S   |  U   |  U   | 3VLL                   |
  +------+------+------+------+------------------------+
  (S = stable, U = unstable; LL/VV/LV/VL refer to the 4 trial phases)

  Each test asks: "is this trial phase truly stable against splitting?"
    LL: liquid-against-liquid (gamma-based)              v0.9.48
    VV: vapor-against-vapor (phi-based)                  v0.9.51
    LV: liquid candidate, vapor trial (cross-phase)      v0.9.52
    VL: vapor candidate, liquid trial (cross-phase)      v0.9.52
""")


if __name__ == "__main__":
    main()
