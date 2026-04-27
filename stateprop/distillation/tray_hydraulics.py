"""Tray hydraulics for distillation columns (v0.9.113).

Given a converged column profile (V, L, T, x, y, P), this module
computes per-stage hydraulic checks for a sieve-tray column:

  * **Vapor flooding** — vapor velocity vs Souders-Brown (Fair 1961)
    flooding velocity.  Stages above ~0.80 of flooding will entrain
    excessively.
  * **Weeping** — vapor velocity vs the minimum needed to support the
    liquid on the tray.  Below this the liquid drains through the
    holes instead of overflowing the weir.
  * **Weir crest height** (Francis weir formula) — head of liquid
    flowing over the outlet weir.
  * **Dry-tray pressure drop** (Liebson 1957) and total wet drop.
  * **Downcomer froth height** — Bennett (1983) clear-liquid +
    froth holdup.

Two top-level entry points:

  * :func:`tray_hydraulics` — analyse an existing column with a
    given :class:`TrayDesign`.
  * :func:`size_tray_diameter` — find the smallest column diameter
    that keeps all stages below a target flooding fraction.

Unit conventions (SI throughout):
  * Flows in mol/s
  * T in K, P in Pa
  * Density in kg/m³
  * Length in m, area in m²
  * Velocity in m/s
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


# Universal gas constant
_R_GAS: float = 8.31446
_PI: float = np.pi


# =====================================================================
# Tray geometry
# =====================================================================

@dataclass
class TrayDesign:
    """Sieve-tray geometry.

    Parameters
    ----------
    diameter : float
        Inside tower diameter [m].
    spacing : float, default 0.6
        Tray-to-tray spacing [m].  Typical 0.5-0.8 m.
    weir_height : float, default 0.05
        Outlet weir height [m].  Typical 25-100 mm.
    hole_area_frac : float, default 0.10
        Hole area fraction of *active* tray area.  Typical 0.06-0.13.
    downcomer_area_frac : float, default 0.10
        Downcomer area as a fraction of total tray area.  Typical
        0.10-0.20.
    weir_length_frac : float, default 0.73
        Weir length as a fraction of tower diameter.  For a typical
        circular segment downcomer at 10 % area fraction.
    """
    diameter: float
    spacing: float = 0.6
    weir_height: float = 0.05
    hole_area_frac: float = 0.10
    downcomer_area_frac: float = 0.10
    weir_length_frac: float = 0.73

    @property
    def total_area(self) -> float:
        return _PI * self.diameter ** 2 / 4.0

    @property
    def downcomer_area(self) -> float:
        return self.downcomer_area_frac * self.total_area

    @property
    def active_area(self) -> float:
        return self.total_area - 2.0 * self.downcomer_area

    @property
    def hole_area(self) -> float:
        return self.hole_area_frac * self.active_area

    @property
    def weir_length(self) -> float:
        return self.weir_length_frac * self.diameter


# =====================================================================
# Fluid-property helpers
# =====================================================================

# Standard molecular weights [kg/mol]
_MW_DEFAULT: dict = {
    "H2O": 0.01802, "NH3": 0.01703, "H2S": 0.03408, "CO2": 0.04401,
    "N2": 0.02801, "O2": 0.03200, "CH4": 0.01604, "Air": 0.02897,
    "Ar": 0.03995,
}


def _vapor_density_ideal(P: float, T: float, y: np.ndarray,
                            mw: np.ndarray) -> float:
    """Ideal-gas vapor density [kg/m³]:  ρ = P · MW_avg / (R · T)."""
    MW_avg = float(np.dot(y, mw))      # kg/mol
    return P * MW_avg / (_R_GAS * T)


def _liquid_density_water(T: float) -> float:
    """Water-only liquid density approximation [kg/m³]
    (good for dilute aqueous columns).  Polynomial fit to NIST data,
    accurate to <1 % over 0-150 °C.

    Reference points (NIST):
      0 °C → 999.84,  25 °C → 997.05,  50 °C → 988.05,
      75 °C → 974.85, 100 °C → 958.39, 130 °C → 934.6
    """
    T_C = T - 273.15
    return float(999.97 - 0.05586 * T_C - 0.003662 * T_C ** 2)


def _liquid_density_dilute_aqueous(T: float, x: np.ndarray,
                                          mw: np.ndarray) -> float:
    """Dilute-aqueous liquid density [kg/m³] approximated by water.

    For mostly-water systems (mole fraction H2O > 0.9) the deviation
    from pure-water density is < 1 % at typical sour-water and
    amine-stripper conditions.  More accurate models (Pitzer
    apparent molar volumes) are deferred.
    """
    return _liquid_density_water(T)


def _surface_tension_water(T: float) -> float:
    """Water surface tension [N/m].

    From the CRC-handbook fit anchored to NIST data:
        σ = 235.8 mN/m · (1 − T/Tc)^1.256 · [1 − 0.625 · (1 − T/Tc)]
    Accurate to ~0.5 % over 10-180 °C.  Tc = 647.15 K.
    """
    Tc = 647.15
    if T >= Tc:
        return 0.0
    tau = 1.0 - T / Tc
    sigma_mN = 235.8 * tau ** 1.256 * (1.0 - 0.625 * tau)
    return float(sigma_mN * 1e-3)   # mN/m → N/m


# =====================================================================
# Flooding correlation: Fair (1961)
# =====================================================================

def _C_sb_fair(F_LV: float, tray_spacing: float) -> float:
    """Souders-Brown C-factor [m/s] for sieve trays from Fair (1961).

    Combines the F_LV flow-parameter correlation with the tray-spacing
    multiplier.  F_LV = (L/V) · sqrt(ρ_V/ρ_L) is the dimensionless
    flow parameter.  Returns C in m/s, tied to σ = 0.020 N/m and to
    the actual tray spacing.  Use ``v_flood = C · (σ/0.020)^0.2 ·
    sqrt((ρ_L − ρ_V) / ρ_V)``.

    Fit to Fair's 1961 chart over F_LV in [0.01, 1.0]:
    """
    F = max(min(F_LV, 1.0), 0.01)
    # Log-log linear fit to Fair's chart at T_s = 0.61 m, σ = 0.020 N/m
    # C_61 ≈ 0.061 at F=0.01, 0.045 at F=0.1, 0.025 at F=0.5, 0.014 at F=1
    # Power-law fit C_61 = 0.040 · F^(-0.30)
    C_61 = 0.040 * F ** (-0.30)
    # Tray-spacing scaling (Fair):
    #   C(T_s) = C(0.61) · (T_s / 0.61) ** 0.5  for T_s in [0.30, 0.92]
    return C_61 * (tray_spacing / 0.61) ** 0.5


def flooding_velocity(rho_L: float, rho_V: float, F_LV: float,
                          tray_spacing: float, sigma: float) -> float:
    """Souders-Brown flooding superficial vapor velocity [m/s].

    v_flood = C_sb(F_LV, T_s) · (σ / 0.020)^0.2 · √((ρ_L − ρ_V) / ρ_V)
    """
    C_sb = _C_sb_fair(F_LV, tray_spacing)
    sigma_corr = (sigma / 0.020) ** 0.2 if sigma > 0 else 1.0
    return C_sb * sigma_corr * np.sqrt(max(rho_L - rho_V, 0) / max(rho_V, 1e-9))


# =====================================================================
# Per-stage hydraulics
# =====================================================================

@dataclass
class StageHydraulics:
    """Per-stage hydraulic snapshot.

    All quantities are at the steady-state operating point; ``flooding``
    and ``weeping`` are advisory booleans (≥ 80 % and < 60 % of design,
    respectively, are common warning thresholds).
    """
    stage: int
    rho_V: float                        # kg/m³
    rho_L: float                        # kg/m³
    sigma: float                        # N/m
    F_LV: float                         # flow parameter
    Q_V: float                          # vapor volumetric flow [m³/s]
    Q_L: float                          # liquid volumetric flow [m³/s]
    velocity_actual: float              # superficial vapor velocity m/s
    velocity_flood: float               # Fair-correlation flooding velocity m/s
    pct_flood: float                    # 100 · v / v_flood
    velocity_min_weep: float            # min. velocity to avoid weeping m/s
    weir_crest: float                   # Francis weir head [m]
    dry_pressure_drop: float            # Pa
    total_pressure_drop: float          # Pa (dry + wet)
    downcomer_froth: float              # m
    weeping: bool
    flooding: bool


@dataclass
class TrayHydraulicsResult:
    """Top-level hydraulics analysis for a column."""
    tray_design: TrayDesign
    per_stage: List[StageHydraulics] = field(default_factory=list)
    max_pct_flood: float = 0.0
    max_pct_flood_stage: int = 0
    flooding_stages: List[int] = field(default_factory=list)
    weeping_stages: List[int] = field(default_factory=list)
    total_pressure_drop: float = 0.0   # Pa across all trays


# =====================================================================
# Public API
# =====================================================================

def tray_hydraulics(
    V_profile: Sequence[float],
    L_profile: Sequence[float],
    T_profile: Sequence[float],
    x_profile: np.ndarray,
    y_profile: np.ndarray,
    P: float,
    species_names: Sequence[str],
    tray_design: TrayDesign,
    *,
    rho_L_func: Optional[Callable[[float, np.ndarray, np.ndarray], float]] = None,
    sigma_func: Optional[Callable[[float], float]] = None,
    flood_warn_frac: float = 0.80,
    weep_min_velocity_frac: float = 0.50,
) -> TrayHydraulicsResult:
    """Analyse a column profile against a tray design.

    Parameters
    ----------
    V_profile, L_profile : length-N (per stage) molar flows [mol/s].
        Constant-V columns can pass [V]*N.  For variable-V output of
        the v0.9.109+ stripper, pass the per-stage V from the
        result's ``V_profile`` attribute (drop interface notation:
        use the average of consecutive interface values).
    T_profile : length-N stage temperatures [K].
    x_profile, y_profile : (N, C) liquid and vapor mole-fraction arrays.
    P : column pressure [Pa].
    species_names : length-C species ordering.
    tray_design : TrayDesign instance.
    rho_L_func : callable(T, x, mw) -> rho_L  [kg/m³], optional
        Override default dilute-aqueous water-density approximation.
    sigma_func : callable(T) -> sigma  [N/m], optional
        Override default water surface tension.
    flood_warn_frac : default 0.80
        Fraction of flooding velocity above which a stage is flagged.
    weep_min_velocity_frac : default 0.50
        Fraction of design flooding velocity below which weeping is
        flagged.

    Returns
    -------
    TrayHydraulicsResult
    """
    if rho_L_func is None:
        rho_L_func = _liquid_density_dilute_aqueous
    if sigma_func is None:
        sigma_func = _surface_tension_water

    # Build MW vector from species_names
    mw = np.array([_MW_DEFAULT.get(sp, 0.030) for sp in species_names])

    N = len(V_profile)
    if not (len(L_profile) == len(T_profile) == N):
        raise ValueError(
            f"V_profile length ({N}) must match L_profile "
            f"({len(L_profile)}) and T_profile ({len(T_profile)})")

    A_active = tray_design.active_area
    A_hole = tray_design.hole_area
    A_dc = tray_design.downcomer_area
    L_w = tray_design.weir_length
    h_w = tray_design.weir_height

    per_stage: List[StageHydraulics] = []
    flooding_stages: List[int] = []
    weeping_stages: List[int] = []
    max_pct = 0.0
    max_pct_stage = 0
    total_dp = 0.0

    for j in range(N):
        T_j = float(T_profile[j])
        V_j = float(V_profile[j])
        L_j = float(L_profile[j])
        x_j = np.asarray(x_profile[j], dtype=float)
        y_j = np.asarray(y_profile[j], dtype=float)

        rho_V = _vapor_density_ideal(P, T_j, y_j, mw)
        rho_L = float(rho_L_func(T_j, x_j, mw))
        sigma = float(sigma_func(T_j))

        # Vapor / liquid molar mass
        MW_V = float(np.dot(y_j, mw))      # kg/mol
        MW_L = float(np.dot(x_j, mw))

        # Volumetric flows
        Q_V = V_j * MW_V / max(rho_V, 1e-9)        # m³/s
        Q_L = L_j * MW_L / max(rho_L, 1e-9)

        # Superficial vapor velocity over active area
        v_act = Q_V / max(A_active, 1e-9)

        # Flow parameter and Souders-Brown flooding velocity
        if V_j > 1e-9:
            F_LV = (L_j * MW_L) / (V_j * MW_V) * np.sqrt(rho_V / max(rho_L, 1e-9))
        else:
            F_LV = 0.0
        v_flood = flooding_velocity(rho_L, rho_V, F_LV,
                                          tray_design.spacing, sigma)
        pct_flood = 100.0 * v_act / max(v_flood, 1e-9)

        # Weeping minimum velocity (Liebson 1957 simplified):
        #   v_min ~ 0.32 · sqrt((rho_L − rho_V) / rho_V)
        # for sieve trays at typical hole diameters.  This sets the
        # threshold below which liquid drains through the holes.
        v_min_weep = 0.32 * np.sqrt(max(rho_L - rho_V, 0)
                                          / max(rho_V, 1e-9))

        # Francis weir crest h_ow [m] (Q_L in m³/s, L_w in m)
        # Standard form: h_ow = 0.664 · (Q_L / L_w)^(2/3)   [m]  (Q in m³/s, L in m)
        # Note: this is "metric Francis" (no segmental correction)
        h_ow = 0.664 * (Q_L / max(L_w, 1e-3)) ** (2.0 / 3.0)

        # Dry hole velocity → dry tray pressure drop (Liebson)
        # u_h = Q_V / A_hole; ΔP_dry = (1/2) · ρ_V · u_h² · K_orifice
        # K_orifice ≈ 1.7 (sharp-edge sieve hole)
        u_h = Q_V / max(A_hole, 1e-9)
        dP_dry = 0.5 * rho_V * u_h ** 2 * 1.7

        # Total pressure drop = dry + (h_w + h_ow) · ρ_L · g
        dP_total = dP_dry + (h_w + h_ow) * rho_L * 9.81

        # Bennett (1983) downcomer froth height (simplified):
        # h_dc ≈ h_w + h_ow + dP_total/(ρ_L g)
        h_dc = (dP_total / (rho_L * 9.81)) + h_w + h_ow

        is_flooding = pct_flood > 100.0 * flood_warn_frac
        is_weeping = v_act < weep_min_velocity_frac * v_min_weep

        per_stage.append(StageHydraulics(
            stage=j, rho_V=rho_V, rho_L=rho_L, sigma=sigma,
            F_LV=float(F_LV),
            Q_V=Q_V, Q_L=Q_L,
            velocity_actual=v_act,
            velocity_flood=v_flood,
            pct_flood=pct_flood,
            velocity_min_weep=v_min_weep,
            weir_crest=float(h_ow),
            dry_pressure_drop=float(dP_dry),
            total_pressure_drop=float(dP_total),
            downcomer_froth=float(h_dc),
            weeping=bool(is_weeping),
            flooding=bool(is_flooding),
        ))

        if pct_flood > max_pct:
            max_pct = pct_flood
            max_pct_stage = j
        if is_flooding:
            flooding_stages.append(j)
        if is_weeping:
            weeping_stages.append(j)
        total_dp += dP_total

    return TrayHydraulicsResult(
        tray_design=tray_design,
        per_stage=per_stage,
        max_pct_flood=max_pct,
        max_pct_flood_stage=max_pct_stage,
        flooding_stages=flooding_stages,
        weeping_stages=weeping_stages,
        total_pressure_drop=total_dp,
    )


def size_tray_diameter(
    V_profile: Sequence[float],
    L_profile: Sequence[float],
    T_profile: Sequence[float],
    x_profile: np.ndarray,
    y_profile: np.ndarray,
    P: float,
    species_names: Sequence[str],
    *,
    spacing: float = 0.6,
    weir_height: float = 0.05,
    hole_area_frac: float = 0.10,
    target_flood_frac: float = 0.75,
    rho_L_func: Optional[Callable] = None,
    sigma_func: Optional[Callable] = None,
) -> float:
    """Find the minimum tower diameter [m] keeping all stages below
    ``target_flood_frac`` of flooding velocity.

    Solves by bisection on the diameter; smaller diameter gives higher
    velocity and higher % flood.
    """
    def _max_pct_at(D: float) -> float:
        td = TrayDesign(
            diameter=D, spacing=spacing,
            weir_height=weir_height, hole_area_frac=hole_area_frac)
        r = tray_hydraulics(
            V_profile, L_profile, T_profile,
            x_profile, y_profile, P, species_names,
            tray_design=td,
            rho_L_func=rho_L_func, sigma_func=sigma_func,
        )
        return r.max_pct_flood

    target_pct = 100.0 * target_flood_frac
    # Initial bracket: 0.1 m (small, will flood) up to 5 m (large, fine)
    D_lo, D_hi = 0.1, 5.0
    pct_lo = _max_pct_at(D_lo)
    pct_hi = _max_pct_at(D_hi)
    if pct_hi > target_pct:
        # Even 5 m is not enough → enlarge
        while pct_hi > target_pct and D_hi < 50.0:
            D_hi *= 2.0
            pct_hi = _max_pct_at(D_hi)
    if pct_lo < target_pct:
        return D_lo

    # Bisection
    for _ in range(60):
        D_mid = 0.5 * (D_lo + D_hi)
        pct_mid = _max_pct_at(D_mid)
        if pct_mid > target_pct:
            D_lo = D_mid
        else:
            D_hi = D_mid
        if D_hi - D_lo < 1e-3:
            break
    return float(D_hi)
