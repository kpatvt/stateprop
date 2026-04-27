"""Two-stage sour-water stripper flowsheet (v0.9.113).

Industrial sour-water plants typically use two strippers in series:

  Stage 1 — Acid stripper (low pH, ~3-5):
    Acid (HCl, H2SO4) shifts the equilibrium so that
        NH3 + H+ → NH4+        (non-volatile)
        HS-  + H+ → H2S        (volatile)
        HCO3- + H+ → CO2       (volatile)
    Result: H2S and CO2 strip out efficiently, NH3 stays as NH4+
    in the bottoms.

  Stage 2 — Caustic stripper (high pH, ~10-11):
    Base (NaOH, KOH) reverses the speciation:
        NH4+ + OH- → NH3 + H2O   (volatile)
        H2S       → HS-          (non-volatile)  ← but H2S already gone
    Result: NH3 strips, the residual sulfide and carbonate stay as
    ions but they were already removed in stage 1.

This module wires two :func:`sour_water_stripper` calls into a
single integrator that respects the inter-stage Cl⁻ carry-over,
applies the user-specified acid/base molality, and reports
combined recoveries and energy KPIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .sour_water_column import (
    sour_water_stripper, SourWaterStripperResult,
    _M_H2O_PER_KG,
)


# =====================================================================
# Result
# =====================================================================

@dataclass
class SourWaterFlowsheetResult:
    """Two-stage sour-water flowsheet result.

    Attributes
    ----------
    stage1_result : SourWaterStripperResult
        Acid-stripper column.
    stage2_result : SourWaterStripperResult
        Caustic-stripper column.
    overall_recovery : Dict[str, float]
        Total feed → final-bottoms removal fraction for each volatile.
    acid_dose_mol_per_kg : float
        HCl dose specified by user [mol/kg H2O in stage 1 feed].
    base_dose_mol_per_kg : float
        NaOH dose specified by user [mol/kg H2O in stage 2 feed].
    acid_consumption_kg_per_h : Optional[float]
        HCl mass flow [kg/s · 3600] consumed by stage 1.
    base_consumption_kg_per_h : Optional[float]
        NaOH mass flow [kg/h] consumed by stage 2.
    Q_R_total : Optional[float]
        Sum of reboiler duties (W).
    steam_ratio_total : Optional[float]
        Combined steam consumption per kg feed water (kg/kg).
    converged : bool
    """
    stage1_result: SourWaterStripperResult
    stage2_result: SourWaterStripperResult
    overall_recovery: Dict[str, float]
    acid_dose_mol_per_kg: float
    base_dose_mol_per_kg: float
    acid_consumption_kg_per_h: Optional[float] = None
    base_consumption_kg_per_h: Optional[float] = None
    Q_R_total: Optional[float] = None
    steam_ratio_total: Optional[float] = None
    converged: bool = True


# =====================================================================
# Public API
# =====================================================================

def sour_water_two_stage_flowsheet(
        feed_F: float,
        feed_z: Sequence[float],
        feed_T: float,
        species_names: Sequence[str],
        acid_dose_mol_per_kg: float,
        base_dose_mol_per_kg: float,
        # Stage-1 column inputs
        n_stages_acid: int = 8,
        feed_stage_acid: int = 2,
        reflux_ratio_acid: float = 0.5,
        distillate_rate_acid: float = 2.0,
        pressure_acid: float = 1.5e5,
        T_top_acid: Optional[float] = None,
        T_bot_acid: Optional[float] = None,
        # Stage-2 column inputs
        n_stages_base: int = 8,
        feed_stage_base: int = 2,
        reflux_ratio_base: float = 0.5,
        distillate_rate_base: float = 2.0,
        pressure_base: float = 1.5e5,
        T_top_base: Optional[float] = None,
        T_bot_base: Optional[float] = None,
        # Inter-stage handling
        T_intermediate_cooler: Optional[float] = None,
        # Common kwargs
        energy_balance: bool = True,
        stage_efficiency: object = 0.65,
        verbose: bool = False,
) -> SourWaterFlowsheetResult:
    """Solve the two-stage sour-water flowsheet.

    Parameters
    ----------
    feed_F : float
        Total feed molar flow [mol/s].
    feed_z : sequence of float
        Feed mole fractions, same ordering as ``species_names``.  Must
        include H2O and at least one of NH3/H2S/CO2.
    feed_T : float
        Feed temperature [K].
    species_names : sequence of str
    acid_dose_mol_per_kg : float
        HCl molality applied as ``extra_strong_anions`` in stage 1.
        Typical industrial range: 0.05-1.0 mol/kg.
    base_dose_mol_per_kg : float
        NaOH molality added at the inlet of stage 2.  The Cl⁻ from
        stage 1 also carries over to stage 2 (so stage 2 sees both
        the residual Cl⁻ and the new Na⁺).
    n_stages_acid, n_stages_base : default 8 each.
    reflux_ratio_acid, reflux_ratio_base : default 0.5 (light reflux
        for stripper-only configuration).
    distillate_rate_acid, distillate_rate_base : default 2.0 mol/s.
    T_top_acid/bot_acid, T_top_base/bot_base : optional column-T initial
        bounds.  Default top = feed_T, bottom = feed_T + 30.
    T_intermediate_cooler : float, optional [K]
        Cool stage 1 bottoms to this T before feeding stage 2.  Default:
        no cooling (pass through at stage 1 bottom T).
    energy_balance : bool, default True (turn on heat balance).
    stage_efficiency : 0.65 by default (industrial Murphree).
    verbose : bool

    Returns
    -------
    SourWaterFlowsheetResult
    """
    if "H2O" not in species_names:
        raise ValueError("species_names must contain 'H2O'")
    if acid_dose_mol_per_kg < 0 or base_dose_mol_per_kg < 0:
        raise ValueError("doses must be non-negative")

    feed_z_arr = np.asarray(feed_z, dtype=float)
    i_h2o = list(species_names).index("H2O")
    if T_top_acid is None:
        T_top_acid = feed_T
    if T_bot_acid is None:
        T_bot_acid = feed_T + 30.0
    if T_top_base is None:
        T_top_base = T_top_acid
    if T_bot_base is None:
        T_bot_base = T_bot_acid

    # --------------- Stage 1: acid stripper ---------------
    if verbose:
        print(f"  [stage 1] acid dose = {acid_dose_mol_per_kg:.3f} mol/kg")
    r1 = sour_water_stripper(
        n_stages=n_stages_acid, feed_stage=feed_stage_acid,
        feed_F=feed_F, feed_z=list(feed_z), feed_T=feed_T,
        species_names=list(species_names),
        reflux_ratio=reflux_ratio_acid,
        distillate_rate=distillate_rate_acid,
        pressure=pressure_acid,
        T_init=list(np.linspace(T_top_acid, T_bot_acid, n_stages_acid)),
        extra_strong_anions=acid_dose_mol_per_kg,
        extra_strong_cations=0.0,
        energy_balance=energy_balance,
        stage_efficiency=stage_efficiency,
        verbose=False,
    )

    # --------------- Inter-stage: cool & re-mix ---------------
    bottoms_z = r1.column_result.x[-1, :].copy()
    bottoms_F = float(r1.column_result.B)
    T_bot1 = float(r1.column_result.T[-1])
    T_feed2 = T_intermediate_cooler if T_intermediate_cooler is not None else T_bot1

    if verbose:
        print(f"  [intermediate] stage 1 bottoms: F={bottoms_F:.2f} mol/s, "
               f"T={T_bot1-273.15:.1f}°C → {T_feed2-273.15:.1f}°C")

    # --------------- Stage 2: caustic stripper ---------------
    # Stage 2 inherits Cl⁻ from stage 1 (no chloride volatilises) and
    # gains Na⁺ from the added NaOH.  Both go into the activity model
    # background.
    if verbose:
        print(f"  [stage 2] base dose = {base_dose_mol_per_kg:.3f} mol/kg")
    r2 = sour_water_stripper(
        n_stages=n_stages_base, feed_stage=feed_stage_base,
        feed_F=bottoms_F, feed_z=bottoms_z.tolist(), feed_T=T_feed2,
        species_names=list(species_names),
        reflux_ratio=reflux_ratio_base,
        distillate_rate=distillate_rate_base,
        pressure=pressure_base,
        T_init=list(np.linspace(T_top_base, T_bot_base, n_stages_base)),
        extra_strong_anions=acid_dose_mol_per_kg,            # Cl⁻ carryover
        extra_strong_cations=base_dose_mol_per_kg,           # Na⁺ added
        energy_balance=energy_balance,
        stage_efficiency=stage_efficiency,
        verbose=False,
    )

    # --------------- Combined recovery ---------------
    final_bottoms_z = r2.column_result.x[-1, :]
    final_bottoms_F = float(r2.column_result.B)
    overall: Dict[str, float] = {}
    for sp in ("NH3", "H2S", "CO2"):
        if sp in species_names:
            i = list(species_names).index(sp)
            if feed_z_arr[i] > 1e-12:
                m_feed = feed_F * feed_z_arr[i]
                m_bot = final_bottoms_F * final_bottoms_z[i]
                overall[sp] = float(max(0.0, (m_feed - m_bot) / m_feed))
            else:
                overall[sp] = float("nan")

    # --------------- Energy KPIs ---------------
    Q_total = None
    steam_total = None
    if energy_balance and (r1.Q_R is not None) and (r2.Q_R is not None):
        Q_total = float(r1.Q_R + r2.Q_R)

        # Combined steam ratio: total steam consumption / kg water in feed
        kg_water_in_feed_per_s = feed_F * feed_z_arr[i_h2o] * 18.015e-3
        if kg_water_in_feed_per_s > 0 and Q_total > 0:
            from .sour_water_column import _delta_H_vap_water
            dHvap = _delta_H_vap_water(feed_T)
            kg_steam_per_s = Q_total / dHvap * 18.015e-3
            steam_total = float(kg_steam_per_s / kg_water_in_feed_per_s)

    # Acid / base consumption (mol/s of feed water → mol/s of HCl/NaOH,
    # then convert to kg/h).
    kg_water_in_feed_per_s = feed_F * feed_z_arr[i_h2o] * 18.015e-3
    acid_kg_per_h = (acid_dose_mol_per_kg * kg_water_in_feed_per_s
                          * 36.46e-3 * 3600.0)
    base_kg_per_h = (base_dose_mol_per_kg * float(r2.column_result.feed_F)
                          * feed_z_arr[i_h2o] * 18.015e-3
                          * 40.00e-3 * 3600.0)

    if verbose:
        print(f"  Overall recovery: {overall}")
        if Q_total is not None:
            print(f"  Total Q_R = {Q_total/1e3:.1f} kW, steam ratio "
                   f"{steam_total:.3f} kg/kg")

    return SourWaterFlowsheetResult(
        stage1_result=r1,
        stage2_result=r2,
        overall_recovery=overall,
        acid_dose_mol_per_kg=acid_dose_mol_per_kg,
        base_dose_mol_per_kg=base_dose_mol_per_kg,
        acid_consumption_kg_per_h=acid_kg_per_h,
        base_consumption_kg_per_h=base_kg_per_h,
        Q_R_total=Q_total,
        steam_ratio_total=steam_total,
        converged=(r1.column_result.converged
                       and r2.column_result.converged),
    )


# =====================================================================
# Auto-dose: bisect on dose to hit a recovery target
# =====================================================================

def find_acid_dose_for_h2s_recovery(
        target_recovery: float,
        feed_F: float,
        feed_z: Sequence[float],
        feed_T: float,
        species_names: Sequence[str],
        n_stages_acid: int = 8,
        feed_stage_acid: int = 2,
        reflux_ratio_acid: float = 0.5,
        distillate_rate_acid: float = 2.0,
        pressure_acid: float = 1.5e5,
        T_top_acid: Optional[float] = None,
        T_bot_acid: Optional[float] = None,
        stage_efficiency: object = 0.65,
        dose_min: float = 0.0,
        dose_max: float = 2.0,
        tol: float = 1e-3,
        max_iter: int = 30,
        verbose: bool = False,
) -> float:
    """Bisect on acid_dose to hit a target H2S recovery in stage 1.

    Returns the acid dose [mol HCl / kg H2O in feed] giving the
    requested H2S recovery in the acid stripper.  Useful when the
    plant operating spec is "99.5 % H2S" and the engineer wants the
    minimum acid consumption.
    """
    feed_z_arr = np.asarray(feed_z, dtype=float)
    if "H2S" not in species_names:
        raise ValueError("H2S must be in species_names")

    if T_top_acid is None:
        T_top_acid = feed_T
    if T_bot_acid is None:
        T_bot_acid = feed_T + 30.0

    def _recovery_at(dose: float) -> float:
        r = sour_water_stripper(
            n_stages=n_stages_acid, feed_stage=feed_stage_acid,
            feed_F=feed_F, feed_z=list(feed_z), feed_T=feed_T,
            species_names=list(species_names),
            reflux_ratio=reflux_ratio_acid,
            distillate_rate=distillate_rate_acid,
            pressure=pressure_acid,
            T_init=list(np.linspace(T_top_acid, T_bot_acid,
                                            n_stages_acid)),
            extra_strong_anions=dose, extra_strong_cations=0.0,
            energy_balance=False,
            stage_efficiency=stage_efficiency,
        )
        return r.bottoms_strip_efficiency.get("H2S", 0.0)

    e_lo = _recovery_at(dose_min)
    e_hi = _recovery_at(dose_max)
    if e_lo >= target_recovery:
        return float(dose_min)
    if e_hi < target_recovery:
        # Even the largest dose is not enough → try doubling
        for _ in range(5):
            dose_max *= 2.0
            e_hi = _recovery_at(dose_max)
            if e_hi >= target_recovery:
                break
        if e_hi < target_recovery:
            return float(dose_max)

    for it in range(max_iter):
        dose_mid = 0.5 * (dose_min + dose_max)
        e_mid = _recovery_at(dose_mid)
        if verbose:
            print(f"  iter {it}: dose={dose_mid:.4f}, "
                   f"H2S recovery={e_mid:.4f} (target {target_recovery})")
        if e_mid < target_recovery:
            dose_min = dose_mid
        else:
            dose_max = dose_mid
        if abs(dose_max - dose_min) < tol:
            break
    return float(dose_max)
