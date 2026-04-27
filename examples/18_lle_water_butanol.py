"""LLE flash with UNIFAC_LLE — water + n-butanol mutual solubility.

Demonstrates:
  - UNIFAC_LLE class with Magnussen 1981 LLE-fitted parameters (v0.9.54)
  - LLEFlash for 2-phase liquid-liquid equilibrium
  - Coverage reporting (`lle_coverage`) for diagnosing parameter applicability
  - Validation against published mutual solubilities (`validate_against_benchmarks`)
  - Comparison vs standard VLE-fitted UNIFAC for the same system

System: n-butanol + water — canonical hard LLE system. VLE-fitted UNIFAC
fails to find a real 2-phase split for this mixture; LLE-fitted UNIFAC
succeeds.
"""
from __future__ import annotations

import numpy as np
from stateprop.activity import (UNIFAC, UNIFAC_LLE, LLEFlash,
                                 lle_coverage,
                                 validate_against_benchmarks,
                                 format_benchmark_results,
                                 LLE_OVERRIDES)


def main():
    # n-butanol + water groups
    groups = [
        {'CH3': 1, 'CH2': 3, 'OH': 1},   # n-butanol
        {'H2O': 1},                       # water
    ]

    print("=" * 65)
    print("Water + n-butanol LLE at 298 K")
    print("=" * 65)

    # 1. Coverage report -- which group-pair interactions are LLE-fitted?
    report = lle_coverage(groups)
    print("\n  Coverage of the LLE_OVERRIDES set for this system:")
    print("  " + "\n  ".join(str(report).split("\n")))

    # 2. Build models with both VLE-UNIFAC and LLE-UNIFAC
    uf_vle = UNIFAC(groups)
    uf_lle = UNIFAC_LLE(groups)

    # Print activity coefficients in dilute-organic limit (where LLE matters)
    g_vle = uf_vle.gammas(298.15, [0.001, 0.999])[0]
    g_lle = uf_lle.gammas(298.15, [0.001, 0.999])[0]
    print(f"\n  Infinite-dilution activity coefficient gamma_BuOH^inf in water:")
    print(f"    VLE-UNIFAC: {g_vle:.2f}")
    print(f"    LLE-UNIFAC: {g_lle:.2f}  (larger -> stronger LLE driving force)")

    # 3. LLE flash with each model
    print(f"\n  LLE flash (z = (0.5, 0.5)):")
    print(f"    Published DDBST data: x_BuOH = 0.018 in water-rich phase,")
    print(f"                          x_BuOH = 0.485 in BuOH-rich phase\n")

    for label, model in [("VLE-UNIFAC", uf_vle), ("LLE-UNIFAC", uf_lle)]:
        lle = LLEFlash(model)
        try:
            r = lle.solve(298.15, [0.5, 0.5],
                            x1_guess=[0.01, 0.99], x2_guess=[0.5, 0.5])
            if 0.05 < r.beta < 0.95:
                # Valid 2-phase split
                if r.x1[0] < r.x2[0]:
                    water_phase, org_phase = r.x1, r.x2
                else:
                    water_phase, org_phase = r.x2, r.x1
                print(f"    {label}: water-rich x_BuOH = {water_phase[0]:.4f},  "
                      f"BuOH-rich x_BuOH = {org_phase[0]:.4f},  beta = {r.beta:.3f}")
            else:
                print(f"    {label}: COLLAPSED to single phase (beta = {r.beta:.3f})")
        except Exception as e:
            print(f"    {label}: failed - {str(e)[:60]}")

    # 4. Benchmark validation against canonical aqueous-organic LLE systems
    print("\n" + "=" * 65)
    print("Benchmark validation: LLE_OVERRIDES vs published data (4 systems)")
    print("=" * 65 + "\n")
    results = validate_against_benchmarks(verbose=False)
    print(format_benchmark_results(results))

    print(f"\n  Bundled LLE_OVERRIDES contains {len(LLE_OVERRIDES)} parameter pairs")
    print(f"  (Magnussen, Rasmussen, Fredenslund 1981).")
    print(f"  For systems whose main groups fall outside this set, parameters")
    print(f"  fall back to standard VLE-fitted UNIFAC values from Hansen 1991.")
    print(f"\n  To extend, pass `extra_overrides={{(m,n): (a_mn, a_nm)}}` to")
    print(f"  UNIFAC_LLE() or use load_overrides_from_json('your_table.json').")


if __name__ == "__main__":
    main()
