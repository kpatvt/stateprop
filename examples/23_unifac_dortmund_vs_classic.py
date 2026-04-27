"""UNIFAC variants compared: classic vs Dortmund-modified.

What this demonstrates
----------------------
The UNIFAC group-contribution method has multiple parameter sets,
fit to different experimental databases over the years.  The two
most commonly used variants:

- **Classic UNIFAC** (Fredenslund-Jones-Prausnitz 1975, with periodic
  parameter updates by Hansen et al.). Single-temperature parameters,
  fit broadly across VLE data.

- **UNIFAC-Dortmund** (Weidlich-Gmehling 1987, periodically updated).
  Temperature-dependent parameters, expanded to cover infinite-
  dilution activity coefficients (γ∞), excess enthalpies (H_E), and
  other properties simultaneously.

For VLE prediction at moderate composition both give similar results.
The differences emerge at the *extremes* of composition (γ∞) and
across temperature ranges (H_E reflecting dγ/dT).

This example compares both variants on three binaries spanning a
range of non-ideality:

1. **n-Hexane / benzene** — alkane + aromatic, mild non-ideality
2. **Acetone / water** — strongly non-ideal H-bonding + dipolar
3. **Ethanol / n-hexane** — strongly non-ideal H-bonding + non-polar

We compute γ across composition at 25 °C and 75 °C for each system,
showing where the two variants agree and disagree.

Reference
---------
Fredenslund, A.; Jones, R. L.; Prausnitz, J. M. (1975). Group
contribution estimation of activity coefficients in non-ideal liquid
mixtures.  AIChE J. 21, 1086.

Weidlich, U.; Gmehling, J. (1987). A modified UNIFAC model. 1.
Prediction of VLE, h^E, and γ^∞.  Ind. Eng. Chem. Res. 26, 1372.

Approximate runtime: ~2 seconds.

Public APIs invoked
-------------------
- stateprop.activity.UNIFAC, UNIFAC_Dortmund
- stateprop.activity.compounds.make_unifac
- stateprop.activity.compounds.make_unifac_dortmund
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.activity.compounds import make_unifac, make_unifac_dortmund
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("UNIFAC variants compared: classic vs Dortmund-modified")
print("=" * 70)
print()

binaries = [
    (["n-hexane", "benzene"],
        "n-Hexane / benzene  (alkane + aromatic, mild non-ideality)"),
    (["acetone", "water"],
        "Acetone / water     (H-bonding + dipolar, strongly non-ideal)"),
    (["ethanol", "n-hexane"],
        "Ethanol / n-hexane  (H-bonding + non-polar, strongly non-ideal)"),
]

# T sweep at fixed midpoint composition
T_sweep = [298.15, 333.15, 373.15]   # 25, 60, 100 °C


def gammas_both(names, T, x):
    uf = make_unifac(names)
    ud = make_unifac_dortmund(names)
    return uf.gammas(T, x), ud.gammas(T, x)


# ------------------------------------------------------------------
# Per-binary tables
# ------------------------------------------------------------------
gamma_inf_data = {}   # (system, T) -> (γ∞ classic, γ∞ Dortmund)

for names, label in binaries:
    print(f"\n{label}")
    print()
    print(f"  T = 25 °C, x sweep:")
    print(f"  {'x_1':>5s}  {'γ_1 UF':>7s}  {'γ_2 UF':>7s}  "
          f"{'γ_1 UD':>7s}  {'γ_2 UD':>7s}  "
          f"{'Δγ_1 %':>8s}  {'Δγ_2 %':>8s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
    for x1 in [0.05, 0.20, 0.50, 0.80, 0.95]:
        x = [x1, 1.0 - x1]
        g_uf, g_ud = gammas_both(names, 298.15, x)
        d1 = (g_ud[0] - g_uf[0]) / g_uf[0] * 100
        d2 = (g_ud[1] - g_uf[1]) / g_uf[1] * 100
        print(f"  {x1:>5.2f}  {g_uf[0]:>7.3f}  {g_uf[1]:>7.3f}  "
                f"{g_ud[0]:>7.3f}  {g_ud[1]:>7.3f}  "
                f"{d1:>+7.1f}%  {d2:>+7.1f}%")

    # Infinite-dilution γ across T
    print()
    print(f"  Infinite-dilution γ vs T:")
    print(f"  {'T (°C)':>7s}  {'γ_1∞ UF':>8s}  {'γ_1∞ UD':>8s}  "
          f"{'Δ%':>6s}  {'γ_2∞ UF':>8s}  {'γ_2∞ UD':>8s}  {'Δ%':>6s}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}")
    for T in T_sweep:
        # γ∞_1: x_1 → 0
        x_dilute_1 = [1e-5, 1 - 1e-5]
        g_uf_1, g_ud_1 = gammas_both(names, T, x_dilute_1)
        # γ∞_2: x_2 → 0
        x_dilute_2 = [1 - 1e-5, 1e-5]
        g_uf_2, g_ud_2 = gammas_both(names, T, x_dilute_2)
        d1 = (g_ud_1[0] - g_uf_1[0]) / g_uf_1[0] * 100
        d2 = (g_ud_2[1] - g_uf_2[1]) / g_uf_2[1] * 100
        gamma_inf_data[(names[0], T)] = (g_uf_1[0], g_ud_1[0])
        gamma_inf_data[(names[1] + "/" + names[0], T)] = (g_uf_2[1], g_ud_2[1])
        print(f"  {T-273.15:>+7.0f}  {g_uf_1[0]:>8.3f}  {g_ud_1[0]:>8.3f}  "
                f"{d1:>+5.1f}%  {g_uf_2[1]:>8.3f}  {g_ud_2[1]:>8.3f}  "
                f"{d2:>+5.1f}%")

# ------------------------------------------------------------------
# Engineering takeaway
# ------------------------------------------------------------------
print()
print("=" * 70)
print("Engineering takeaway:")
print()
print("- For mild non-ideality (alkane-aromatic), classic and Dortmund")
print("  UNIFAC differ by <5 %.  Either is fine for distillation design.")
print("- For strongly non-ideal H-bonding systems (acetone/water,")
print("  ethanol/hexane), differences reach 20-50 % at γ∞.  This")
print("  matters for trace-component recovery, where γ∞ controls the")
print("  separation factor.")
print("- Classic UNIFAC tends to *under-predict* γ∞ in associating")
print("  systems; Dortmund — fit explicitly to γ∞ data — is")
print("  typically more accurate at the extremes of composition.")
print("- Both methods OVER-predict γ∞ for systems like ethanol/water")
print("  (UNIFAC γ∞_EtOH ≈ 7-11 vs. experimental ~4); for accurate VLE")
print("  in such systems, regress NRTL or UNIQUAC parameters against")
print("  binary data (see example 20).")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1. At equimolar in n-hexane / benzene at 25 °C, both variants give
#    γ in 1.0-1.3 (mild non-ideality)
g_uf_hb, g_ud_hb = gammas_both(["n-hexane", "benzene"], 298.15, [0.5, 0.5])
validate_bool("n-Hexane/benzene equimolar γ in 1.0-1.3 (both variants)",
                condition=(1.0 < g_uf_hb[0] < 1.3 and 1.0 < g_uf_hb[1] < 1.3
                            and 1.0 < g_ud_hb[0] < 1.3
                            and 1.0 < g_ud_hb[1] < 1.3),
                detail=f"UF=({g_uf_hb[0]:.3f},{g_uf_hb[1]:.3f}); "
                f"UD=({g_ud_hb[0]:.3f},{g_ud_hb[1]:.3f})")

# 2. Hexane/benzene shows mild divergence between variants (<5 %)
diff_hb = (g_ud_hb[0] - g_uf_hb[0]) / g_uf_hb[0] * 100
validate_bool("Hexane/benzene: Δγ_1 between variants < 5 %",
                condition=(abs(diff_hb) < 5.0),
                detail=f"Δγ_1 = {diff_hb:+.2f}%",
                source="Mild non-ideality → classical UNIFAC works fine")

# 3. Acetone/water γ∞_acetone is large in both variants (>5 at 25 °C)
g_uf_aw, g_ud_aw = gammas_both(["acetone", "water"], 298.15,
                                              [1e-5, 1 - 1e-5])
validate_bool("Acetone/water γ∞(acetone) > 5 (strong non-ideality)",
                condition=(g_uf_aw[0] > 5 and g_ud_aw[0] > 5),
                detail=f"UF γ∞={g_uf_aw[0]:.2f}, UD γ∞={g_ud_aw[0]:.2f}",
                source="DECHEMA ~8.0; both variants over-predict")

# 4. Dortmund and classic differ significantly for strongly non-ideal
#    systems at γ∞
diff_aw = (g_ud_aw[0] - g_uf_aw[0]) / g_uf_aw[0] * 100
validate_bool("Acetone/water: variants differ >10 % at γ∞",
                condition=(abs(diff_aw) > 10),
                detail=f"Δγ∞_acetone = {diff_aw:+.1f}%",
                source="UD includes γ∞ data in its regression set")

# 5. Both variants give γ → 1 as x → 1 (return to ideal limit at
#    pure component)
g_uf_a99, g_ud_a99 = gammas_both(["acetone", "water"], 298.15,
                                              [0.999, 0.001])
validate("γ_acetone → 1 as x_acetone → 1 (UNIFAC)",
          reference=1.0, computed=g_uf_a99[0],
          units="-", tol_rel=0.01,
          source="Theoretical: γ_i(x_i → 1) = 1 by definition")

# 6. Temperature dependence: γ_∞ should change with T (for both, but
#    Dortmund has explicit T-dependent parameters)
g_uf_25, _ = gammas_both(["acetone", "water"], 298.15, [1e-5, 1-1e-5])
g_uf_100, _ = gammas_both(["acetone", "water"], 373.15, [1e-5, 1-1e-5])
delta_with_T_uf = abs((g_uf_100[0] - g_uf_25[0]) / g_uf_25[0])
validate_bool("Classic UNIFAC γ∞ changes < 30 % over 25-100 °C",
                condition=(delta_with_T_uf < 0.30),
                detail=f"γ∞(25°C)={g_uf_25[0]:.2f}, γ∞(100°C)={g_uf_100[0]:.2f}, "
                f"Δ={delta_with_T_uf*100:.1f}%",
                source="Classic UNIFAC has weak T-dependence (only via combinatorial)")

# 7. Ethanol/hexane should show LARGER γ∞ for ethanol than for hexane
#    (asymmetric: ethanol's H-bond network is destroyed in hexane)
g_uf_eh, _ = gammas_both(["ethanol", "n-hexane"], 298.15, [1e-5, 1-1e-5])
g_uf_he, _ = gammas_both(["ethanol", "n-hexane"], 298.15, [1-1e-5, 1e-5])
validate_bool("Ethanol/hexane: γ∞_ethanol > γ∞_hexane",
                condition=(g_uf_eh[0] > g_uf_he[1]),
                detail=f"γ∞_EtOH={g_uf_eh[0]:.2f}, γ∞_hex={g_uf_he[1]:.2f}",
                source="Asymmetric H-bonding cost vs dispersion cost")

summary()
