"""NRTL parameter regression to LLE tie-line data.

Demonstrates v0.9.47 parameter regression machinery — fitting NRTL
binary interaction parameters to a set of liquid-liquid equilibrium
tie-line measurements.

Workflow:
  1. Generate "experimental" tie lines (here from UNIFAC_LLE so we have
     a known target for the fit; in practice these would be lab data).
  2. Build a model_factory that returns an NRTL with parameters from
     the trial vector.
  3. Call regress_lle() with the activity-residual objective.
  4. Verify the fitted NRTL reproduces the tie lines.

System: n-butanol + water at 5 temperatures.
"""
from __future__ import annotations

import numpy as np
from stateprop.activity import (NRTL, UNIFAC_LLE, LLEFlash)
from stateprop.activity.regression import regress_lle


def main():
    # 1. Generate "experimental" tie lines using UNIFAC_LLE
    #    (in real work, these would be from DDBST or your own lab data).
    uf_lle = UNIFAC_LLE([
        {'CH3': 1, 'CH2': 3, 'OH': 1},   # n-butanol
        {'H2O': 1},
    ])
    lle_uf = LLEFlash(uf_lle)

    print("=" * 70)
    print("NRTL parameter regression to LLE tie-line data")
    print("=" * 70)
    print("\nGenerating 'experimental' tie lines (from UNIFAC_LLE)...")

    Ts = [283.15, 293.15, 298.15, 303.15, 313.15]
    tie_lines = []
    print(f"\n  {'T (K)':>7s}   {'x1 (water-rich)':>22s}   "
          f"{'x2 (BuOH-rich)':>22s}")
    for T in Ts:
        r = lle_uf.solve(T, [0.5, 0.5],
                          x1_guess=[0.01, 0.99], x2_guess=[0.5, 0.5])
        # Determine which phase is which
        if r.x1[0] < r.x2[0]:
            x_water_rich, x_org_rich = r.x1, r.x2
        else:
            x_water_rich, x_org_rich = r.x2, r.x1
        tie_lines.append((T, x_water_rich, x_org_rich))
        wr = np.array2string(x_water_rich, precision=4, suppress_small=True)
        org = np.array2string(x_org_rich, precision=4, suppress_small=True)
        print(f"  {T:>7.2f}   {wr:>22s}   {org:>22s}")

    # 2. Define NRTL factory: alpha is fixed, b is fitted.
    alpha = np.array([[0.0, 0.3], [0.3, 0.0]])

    def make_nrtl(params):
        b12, b21 = params
        b = np.array([[0.0, b12], [b21, 0.0]])
        return NRTL(alpha=alpha, b=b)

    # 3. Run regression
    print("\n" + "=" * 70)
    print("Regressing NRTL b_12 and b_21 against tie-line data")
    print("=" * 70)

    # Initial guess: rough values suggesting positive deviations
    x0 = [400.0, 1500.0]
    print(f"\n  Initial guess: b_12 = {x0[0]:.2f}, b_21 = {x0[1]:.2f}")

    result = regress_lle(make_nrtl, tie_lines, x0=x0,
                            objective='activity', verbose=0)

    b12_fit, b21_fit = result.x
    print(f"\n  Fitted parameters:")
    print(f"    b_12 = {b12_fit:.3f}")
    print(f"    b_21 = {b21_fit:.3f}")
    print(f"  Final objective (sum of squared residuals): {result.cost:.4e}")
    print(f"  Termination: {result.message}")

    # 4. Validate the fit by running LLE flash with the fitted NRTL
    print("\n" + "=" * 70)
    print("Validating fit: LLE flash with fitted NRTL vs experimental")
    print("=" * 70)
    nrtl_fit = make_nrtl([b12_fit, b21_fit])
    lle_nrtl = LLEFlash(nrtl_fit)

    print(f"\n  {'T (K)':>7s}   {'exp x_BuOH (W-rich)':>22s}   "
          f"{'fit x_BuOH (W-rich)':>22s}   "
          f"{'exp x_BuOH (O-rich)':>22s}   "
          f"{'fit x_BuOH (O-rich)':>22s}")
    err_water = []
    err_org = []
    for T, x_w_exp, x_o_exp in tie_lines:
        try:
            r = lle_nrtl.solve(T, [0.5, 0.5],
                                  x1_guess=x_w_exp, x2_guess=x_o_exp)
            if r.x1[0] < r.x2[0]:
                x_w_fit, x_o_fit = r.x1, r.x2
            else:
                x_w_fit, x_o_fit = r.x2, r.x1
            ew = abs(x_w_fit[0] - x_w_exp[0])
            eo = abs(x_o_fit[0] - x_o_exp[0])
            err_water.append(ew)
            err_org.append(eo)
            print(f"  {T:>7.2f}   {x_w_exp[0]:>22.4f}   {x_w_fit[0]:>22.4f}   "
                  f"{x_o_exp[0]:>22.4f}   {x_o_fit[0]:>22.4f}")
        except Exception as e:
            print(f"  {T:>7.2f}   flash failed: {str(e)[:40]}")

    if err_water:
        print(f"\n  RMSE water-rich phase: {np.sqrt(np.mean(np.square(err_water))):.4f}")
        print(f"  RMSE org-rich phase:   {np.sqrt(np.mean(np.square(err_org))):.4f}")

    print("""
  Interpretation:
  - The 'activity' objective minimizes ln(x1*gamma1) - ln(x2*gamma2),
    which is smooth in the NRTL parameters (no flash is run during
    regression). This makes Levenberg-Marquardt converge reliably
    from poor initial guesses.
  - For high-precision fits, follow with `objective='flash'` after
    'activity' has produced a reasonable starting point.
""")


if __name__ == "__main__":
    main()
