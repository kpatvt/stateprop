"""Lean-rich heat exchanger for amine CO2 capture cycles (v0.9.106).

The cross-exchanger is the canonical energy-recovery feature of every
industrial amine capture process.  It uses the hot lean amine leaving
the stripper bottom (at ~120 °C) to preheat the cold rich amine going
to the stripper top (at ~50 °C), simultaneously cooling the lean amine
back to absorber-feed temperature (~40 °C after a final trim cooler).

Heat recovery is typically 50-70% of the rich-side sensible heating
duty, which is itself ~20% of the full reboiler duty.  In practical
terms, the lean-rich HX reduces total reboiler duty from ~5-6 GJ/ton
CO2 (no HX) to ~3.5-4 GJ/ton CO2 (with HX) — the "industry
benchmark" range.

Counter-current shell-and-tube physics
---------------------------------------
The exchanger is sized by either:

(a) **Approach-T design (this module's primary mode):** specify a
    minimum approach temperature ΔT_min between hot and cold streams
    (typical 5-10 K).  The duty is determined by which end pinches:

    - Hot-end pinch: T_cold_out ≤ T_hot_in - ΔT_min
    - Cold-end pinch: T_hot_out ≥ T_cold_in + ΔT_min

    Q is limited by the side that pinches first.

(b) **Effectiveness-NTU:** specify an effectiveness ε ∈ (0, 1).
    Q = ε · C_min · (T_hot_in - T_cold_in)
    where C_min = min(m·cp_hot, m·cp_cold).

For balanced flows (C_hot ≈ C_cold), the pinch can be at either end,
and ε_max = 1 - ΔT_min/(T_hot_in - T_cold_in).

For unbalanced flows:
* C_hot < C_cold: hot stream constrains; pinch at hot exit
  (T_hot_out_min = T_cold_in + ΔT_min)
* C_hot > C_cold: cold stream constrains; pinch at cold exit
  (T_cold_out_max = T_hot_in - ΔT_min)

UA sizing
---------
Given the duty and stream temperatures, the required UA (heat-transfer
coefficient × area) is computed by counter-current LMTD:

    Q = UA · LMTD
    LMTD = (ΔT_1 - ΔT_2) / ln(ΔT_1 / ΔT_2)
    where ΔT_1 = T_hot_in - T_cold_out
          ΔT_2 = T_hot_out - T_cold_in

UA is the canonical engineering proxy for HX size; typical values for
amine cross-exchangers are 50-200 kW/K.

References
----------
* Kakaç, S., Liu, H. (2002).  Heat Exchangers: Selection, Rating,
  and Thermal Design (2nd ed.).  CRC Press.
* Cousins, A. et al. (2011).  A survey of process flow sheet
  modifications for energy-efficient post-combustion capture of CO2.
  IJGGC 5, 605.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CrossExchangerResult:
    """Result of a CrossHeatExchanger.solve() call.

    Attributes
    ----------
    Q : float
        Heat duty exchanged [W] (always ≥ 0).
    T_hot_out : float
        Hot stream outlet temperature [K].
    T_cold_out : float
        Cold stream outlet temperature [K].
    delta_T_hot_end : float
        Approach temperature at hot end (T_hot_in - T_cold_out).
    delta_T_cold_end : float
        Approach temperature at cold end (T_hot_out - T_cold_in).
    pinch_at_hot_end : bool
        True if the constraint binds at the hot end (T_cold_out closer
        to T_hot_in than ΔT_min away); False if pinch is at cold end.
    LMTD : float
        Log-mean temperature difference [K] for the exchanged duty.
    UA_required : float
        Required UA [W/K] = Q / LMTD; the canonical sizing parameter.
    effectiveness : float
        ε = Q / Q_max where Q_max = C_min · (T_hot_in - T_cold_in).
        For a typical cross-exchanger ε is 0.6-0.85.
    C_hot : float
        Hot stream heat-capacity flow rate [W/K] = m_hot · cp_hot.
    C_cold : float
        Cold stream heat-capacity flow rate [W/K] = m_cold · cp_cold.
    """
    Q: float
    T_hot_out: float
    T_cold_out: float
    delta_T_hot_end: float
    delta_T_cold_end: float
    pinch_at_hot_end: bool
    LMTD: float
    UA_required: float
    effectiveness: float
    C_hot: float
    C_cold: float


class CrossHeatExchanger:
    """Counter-current cross heat exchanger.

    Designed for the lean-rich exchanger in amine CO2 capture, but
    works for any single-phase counter-current sensible-heat exchanger.

    Parameters
    ----------
    delta_T_min : float, default 5.0
        Minimum approach temperature [K].  Standard industrial values:
        5-10 K for compact plate exchangers, 10-15 K for shell-and-tube.
    effectiveness : float, optional
        If specified, overrides delta_T_min and sizes the exchanger to
        achieve this ε.  Must be in (0, 1).

    Examples
    --------
    >>> from stateprop.electrolyte import CrossHeatExchanger
    >>> hx = CrossHeatExchanger(delta_T_min=5.0)
    >>> # Lean amine 393 K → cooled by rich amine 313 K
    >>> result = hx.solve(
    ...     T_hot_in=393.15,    # lean from stripper at 120 °C
    ...     m_hot=10.0,         # mass flow [kg/s]
    ...     cp_hot=3700.0,      # J/(kg·K)
    ...     T_cold_in=313.15,   # rich from absorber at 40 °C
    ...     m_cold=10.0,
    ...     cp_cold=3700.0,
    ... )
    >>> result.Q / 1e6
    2.78    # MW recovered
    >>> result.T_hot_out - 273.15
    44.85   # °C — lean cooled to ~45 °C
    >>> result.T_cold_out - 273.15
    115.0   # °C — rich preheated to 115 °C
    """

    def __init__(self,
                  delta_T_min: float = 5.0,
                  effectiveness: Optional[float] = None):
        self.delta_T_min = float(delta_T_min)
        if self.delta_T_min <= 0:
            raise ValueError("delta_T_min must be > 0")
        if effectiveness is not None:
            if not 0.0 < effectiveness < 1.0:
                raise ValueError("effectiveness must be in (0, 1)")
        self.effectiveness = effectiveness

    def solve(self,
                T_hot_in: float,
                m_hot: float,
                cp_hot: float,
                T_cold_in: float,
                m_cold: float,
                cp_cold: float) -> CrossExchangerResult:
        """Solve the heat exchanger.

        Parameters
        ----------
        T_hot_in, T_cold_in : float
            Inlet temperatures [K].  T_hot_in must be > T_cold_in.
        m_hot, m_cold : float
            Mass flow rates [kg/s] (or any consistent flow-rate unit).
        cp_hot, cp_cold : float
            Heat capacities [J/(kg·K)] (or consistent with flow-rate
            unit).

        Returns
        -------
        CrossExchangerResult
        """
        if T_hot_in <= T_cold_in:
            raise ValueError(
                f"T_hot_in ({T_hot_in} K) must exceed T_cold_in "
                f"({T_cold_in} K) for heat to flow")
        if min(m_hot, m_cold, cp_hot, cp_cold) <= 0:
            raise ValueError("flows and cp's must be > 0")

        C_hot = m_hot * cp_hot     # W/K
        C_cold = m_cold * cp_cold
        C_min = min(C_hot, C_cold)
        Q_max = C_min * (T_hot_in - T_cold_in)

        # Determine duty
        if self.effectiveness is not None:
            Q = self.effectiveness * Q_max
            T_hot_out = T_hot_in - Q / C_hot
            T_cold_out = T_cold_in + Q / C_cold
        else:
            # Approach-T design: compute Q limited by ΔT_min
            # Hot-end pinch: T_cold_out_max = T_hot_in - ΔT_min
            #   → Q_cold_max = C_cold · (T_cold_out_max - T_cold_in)
            #               = C_cold · ((T_hot_in - T_cold_in) - ΔT_min)
            # Cold-end pinch: T_hot_out_min = T_cold_in + ΔT_min
            #   → Q_hot_max = C_hot · ((T_hot_in - T_cold_in) - ΔT_min)
            available = (T_hot_in - T_cold_in) - self.delta_T_min
            if available <= 0:
                raise ValueError(
                    f"ΔT_min ({self.delta_T_min} K) ≥ available "
                    f"ΔT ({T_hot_in - T_cold_in:.1f} K); no Q possible")
            Q_hot_max = C_hot * available
            Q_cold_max = C_cold * available
            Q = min(Q_hot_max, Q_cold_max)
            T_hot_out = T_hot_in - Q / C_hot
            T_cold_out = T_cold_in + Q / C_cold

        # Approach Ts at each end
        dT_hot_end = T_hot_in - T_cold_out
        dT_cold_end = T_hot_out - T_cold_in

        # Pinch indicator
        pinch_at_hot_end = (dT_hot_end < dT_cold_end)

        # LMTD (counter-current)
        if abs(dT_hot_end - dT_cold_end) < 1e-9:
            # Equal ΔT at both ends — limit case, LMTD = ΔT
            lmtd = 0.5 * (dT_hot_end + dT_cold_end)
        elif dT_hot_end > 0 and dT_cold_end > 0:
            lmtd = ((dT_hot_end - dT_cold_end)
                     / np.log(dT_hot_end / dT_cold_end))
        else:
            lmtd = float("nan")
        UA = Q / lmtd if lmtd and lmtd > 0 else float("nan")

        # Effectiveness
        eff = Q / Q_max if Q_max > 0 else 0.0

        return CrossExchangerResult(
            Q=Q, T_hot_out=T_hot_out, T_cold_out=T_cold_out,
            delta_T_hot_end=dT_hot_end, delta_T_cold_end=dT_cold_end,
            pinch_at_hot_end=pinch_at_hot_end,
            LMTD=lmtd, UA_required=UA,
            effectiveness=eff,
            C_hot=C_hot, C_cold=C_cold,
        )


# =====================================================================
# Convenience: amine-specific lean-rich exchanger
# =====================================================================

def lean_rich_exchanger(amine, total_amine: float,
                          T_lean_in: float, T_rich_in: float,
                          L_lean: float, L_rich: float = None,
                          delta_T_min: float = 5.0,
                          wt_frac_amine: float = 0.30,
                          ) -> CrossExchangerResult:
    """Convenience: solve a lean-rich exchanger for an amine system.

    Computes mass flows and cp from molar flows and the bundled
    Amine.cp_solution() helper, then runs the cross exchanger.

    Parameters
    ----------
    amine : Amine or str
        Amine name or instance.  Used to derive cp_solution and MW.
    total_amine : float
        Liquid amine concentration [mol/kg solvent].
    T_lean_in : float
        Hot lean amine inlet temperature [K] (from stripper bottom).
    T_rich_in : float
        Cold rich amine inlet temperature [K] (from absorber bottom).
    L_lean : float
        Lean amine molar flow [mol amine / s].
    L_rich : float, optional
        Rich amine molar flow [mol amine / s].  If None, defaults to
        L_lean (typical for steady-state cycles).
    delta_T_min : float, default 5.0
        Minimum approach temperature [K].
    wt_frac_amine : float, default 0.30

    Returns
    -------
    CrossExchangerResult
    """
    from .amines import lookup_amine
    if isinstance(amine, str):
        amine = lookup_amine(amine)
    if L_rich is None:
        L_rich = L_lean

    # cp at average T
    T_avg = 0.5 * (T_lean_in + T_rich_in)
    cp_sol = amine.cp_solution(wt_frac_amine, T_avg)

    # Mass flow per mol amine = MW_amine [g/mol]/1000 / wt_frac_amine
    kg_sol_per_mol = (amine.MW * 1e-3) / wt_frac_amine
    m_lean = L_lean * kg_sol_per_mol
    m_rich = L_rich * kg_sol_per_mol

    hx = CrossHeatExchanger(delta_T_min=delta_T_min)
    return hx.solve(
        T_hot_in=T_lean_in, m_hot=m_lean, cp_hot=cp_sol,
        T_cold_in=T_rich_in, m_cold=m_rich, cp_cold=cp_sol,
    )
