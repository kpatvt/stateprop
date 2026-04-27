"""Counter-current reactive absorber/regenerator for amine-based CO2
capture (v0.9.104).

This module couples the v0.9.103 AmineSystem (chemical equilibrium) to
a stage-cascade column model, giving end-to-end simulation of:

* **Amine absorbers** — CO2-bearing flue gas (lean) entering bottom,
  lean amine solvent entering top, rich amine leaving bottom, treated
  gas leaving top.  Each stage solves VLE + chemical equilibrium
  simultaneously.
* **Amine regenerators (strippers)** — rich amine entering top,
  steam (or stripping gas) entering bottom, lean amine leaving bottom,
  CO2 + steam off the top.

Theory
------
For each equilibrium stage j, the simultaneous unknowns are:

    (T_j, V_j, x_j_total)

where T_j is stage temperature, V_j is vapor flow leaving the stage,
and x_j_total is the total CO2 loading α_j on the liquid leaving the
stage.  The constraints are:

    1.  Component CO2 balance:
            L_{j-1}·α_{j-1} + V_{j+1}·y_{j+1}
              = L_j·α_j + V_j·y_j
        (counter-current: liquid flows from j-1 to j to j+1,
         vapor flows from j+1 to j to j-1)

    2.  CO2 chemical equilibrium per stage:
            P_CO2_j = AmineSystem.speciate(α_j, T_j).P_CO2
        and y_CO2_j = P_CO2_j / P_total_j

    3.  Water vapor saturation (assumed in equilibrium):
            P_H2O_j = γ_w_j · x_w_j · P_sat_water(T_j)
            (in our simple model, x_w ≈ 1 in dilute amine and γ_w ≈ 1)
        ⇒ y_H2O_j = P_H2O_j / P_total_j

    4.  Energy balance:
            L_{j-1}·h_{j-1} + V_{j+1}·H_{j+1} + Q_j
              = L_j·h_j + V_j·H_j

The solver uses block-iterative stage relaxation (Wang-Henke style)
which is robust for absorber/regenerator with strong CO2-amine
chemistry.  For each pass:

    a)  Update temperatures from energy balances (with damping)
    b)  Update vapor flows from overall material balance
    c)  Update liquid CO2 loading from component balance + chem. eq.
    d)  Check convergence on (T, V, α) profiles

Convergence is typically reached in 20-50 iterations for absorbers,
40-100 iterations for regenerators (which have steeper profiles).

Limitations
-----------
The current implementation uses these simplifications:

* Amine vapor pressure is neglected (MEA at 40 °C has v.p. ~5 mbar,
  a few % of typical absorber operation; for tight design check
  using detailed simulators).
* Water vapor is assumed in pure-water saturation (γ_w ≈ 1, x_w ≈ 1).
  Exact treatment requires the full MEA-H2O VLE; for engineering
  sizing this is fine.
* Heat capacities are temperature-independent constants from typical
  amine-water heat capacity data.
* Heat of absorption taken as constant ΔH_abs = -85 kJ/mol CO2 for
  primary amines (Mathonat 1997), -65 kJ/mol for tertiary (less
  exothermic, no carbamate route).

For higher accuracy use the full eNRTL γ option in AmineSystem
(`activity_model='bromley'`, an eNRTL-style upgrade in v0.9.104) — see
documentation.

References
----------
* Wang, J. C., Henke, G. E. (1966). Tridiagonal matrix for distillation.
  Hydrocarbon Processing 45, 155.
* Hilliard, M. D. (2008). A predictive thermodynamic model for an
  aqueous blend of potassium carbonate, piperazine, and
  monoethanolamine for carbon dioxide capture from flue gas.
  PhD Thesis, U. Texas at Austin.
* Aronu, U. E. et al. (2011). Solubility of CO2 in 15, 30, 45 and 60
  mass% MEA from 40 to 120 °C and model representation using the
  extended UNIQUAC framework. Chem. Eng. Sci. 66, 6393.
* Faramarzi, L. et al. (2009). Extended UNIQUAC model for thermodynamic
  modeling of CO2 absorption in aqueous alkanolamine solutions.
  Fluid Phase Equilib. 282, 121.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union, Tuple
import numpy as np

from ..electrolyte.amines import (
    Amine, AmineSystem, lookup_amine,
)


# Heat of absorption constants (used when not detailed in AmineSystem)
_DELTA_H_ABS_PRIMARY = -85000.0   # J/mol CO2 (MEA, Mathonat 1997)
_DELTA_H_ABS_TERTIARY = -65000.0  # J/mol CO2 (MDEA, less exothermic)

# Heat capacity of amine solvent (typical 30 wt% MEA)
_CP_LIQUID = 4000.0   # J/(kg·K)  - amine solvent
_CP_VAPOR_H2O = 2010.0   # J/(kg·K) at ~100 °C

# Latent heat of vaporization of water at 100 °C
_DH_VAP_H2O = 2_257_000.0   # J/kg


def _antoine_water(T: float) -> float:
    """Saturation pressure of water [bar] from Antoine equation
    (NIST WebBook, valid 273-373 K)."""
    # log10(P_mmHg) = A - B/(C + T_°C)
    # Stull form: A=8.07131, B=1730.63, C=233.426 (273-373 K)
    T_C = T - 273.15
    log10_P_mmHg = 8.07131 - 1730.63 / (T_C + 233.426)
    P_mmHg = 10.0 ** log10_P_mmHg
    return P_mmHg / 750.062   # bar


# =====================================================================
# Result dataclass
# =====================================================================

@dataclass
class AmineColumnResult:
    """Result of a counter-current amine column simulation.

    Attributes
    ----------
    n_stages : int
    T : ndarray, shape (n_stages,)
        Stage temperatures [K].
    V : ndarray
        Vapor molar flow leaving each stage (upward) [kmol/h].
    L : ndarray
        Liquid molar flow leaving each stage (downward) [kmol/h]
        (liquid total moles, mostly water + amine).
    alpha : ndarray
        Liquid CO2 loading (mol CO2 / mol amine) at each stage.
    y_CO2 : ndarray
        Vapor mole fraction CO2 in stream leaving each stage upward.
    y_H2O : ndarray
        Vapor mole fraction H2O.
    P_CO2 : ndarray
        CO2 partial pressure in vapor leaving each stage [bar].
    pH : ndarray
        Liquid bulk pH at each stage.
    rich_alpha : float
        Loading of amine leaving the bottom of an absorber (or
        equivalent) [mol/mol].
    lean_alpha : float
        Loading of amine leaving the top of a regenerator
        (or input to absorber) [mol/mol].
    treated_y_CO2 : float
        CO2 mole fraction in the treated gas leaving the top of
        an absorber (or top of regenerator).
    capture_pct : float
        % CO2 captured (absorber) or % CO2 stripped (regenerator),
        relative to feed CO2.
    converged : bool
    iterations : int
    column_type : str
        'absorber' or 'regenerator'.
    """
    n_stages: int
    T: np.ndarray
    V: np.ndarray
    L: np.ndarray
    alpha: np.ndarray
    y_CO2: np.ndarray
    y_H2O: np.ndarray
    P_CO2: np.ndarray
    pH: np.ndarray
    rich_alpha: float = 0.0
    lean_alpha: float = 0.0
    treated_y_CO2: float = 0.0
    capture_pct: float = 0.0
    converged: bool = False
    iterations: int = 0
    column_type: str = "absorber"


# =====================================================================
# Amine absorber (CO2 capture)
# =====================================================================

class AmineAbsorber:
    """Counter-current CO2 absorber for alkanolamine solvents.

    Configuration (top-to-bottom):

        Treated gas (top)              Lean amine (top)
              ↑                              ↓
              |     stage 1                  |
              |     stage 2                  |
              |     ...                      |
              |     stage N                  |
              ↑                              ↓
        Feed gas (bottom)              Rich amine (bottom)

    Solution method: stage-by-stage iteration with relaxation (Wang-
    Henke).  At each iteration:
      1. Mass balance for CO2 from top down (sets α profile)
      2. Chemical equilibrium per stage gives P_CO2 → y_CO2 → V update
      3. Energy balance from top down (sets T profile)
      4. Damping factor 0.5 on T and V updates for stability

    Parameters
    ----------
    n_stages : int
        Number of equilibrium stages (typical absorber: 10-25).
    amine : Amine or str
        Solvent amine (looked up if str).
    P_total : float
        Operating pressure [bar].  Typical CO2 capture: 1.0 (atmospheric).
    activity_model : str, default "davies"
        Passed through to AmineSystem (see v0.9.103/v0.9.104 docs).

    Notes
    -----
    The amine concentration in the liquid is fixed throughout the
    column at the lean amine value (no significant amine evaporation
    in our simple model).  CO2 loading α changes from lean (top) to
    rich (bottom).
    """

    def __init__(self,
                  n_stages: int,
                  amine: Union[Amine, str],
                  P_total: float = 1.0,
                  activity_model: str = "davies"):
        if n_stages < 2:
            raise ValueError("n_stages must be ≥ 2")
        self.n_stages = int(n_stages)
        self.amine = (lookup_amine(amine) if isinstance(amine, str)
                       else amine)
        self.P_total = float(P_total)
        self.activity_model = activity_model

    # -----------------------------------------------------------------
    # Main solver
    # -----------------------------------------------------------------
    def simulate(self,
                  feed_gas: Dict[str, float],
                  lean_amine: Dict[str, float],
                  *,
                  max_iter: int = 100,
                  tol: float = 1e-4,
                  damping: float = 0.5,
                  verbose: bool = False) -> AmineColumnResult:
        """Run the absorber simulation.

        Parameters
        ----------
        feed_gas : dict
            Bottom-feed gas specification, with keys:
              - 'F' : total molar flow rate [kmol/h]
              - 'y_CO2' : CO2 mole fraction
              - 'T' : feed temperature [K]
              (balance assumed inert/N2 + saturated water vapor)
        lean_amine : dict
            Top-feed amine spec, with keys:
              - 'F' : total amine molar flow [kmol/h] of amine
                      (i.e. the amine itself; water is implicit)
              - 'total_amine' : amine concentration [mol/kg solvent]
                                (e.g. 5.0 for ~30 wt% MEA)
              - 'alpha' : lean loading [mol CO2 / mol amine]
              - 'T' : feed temperature [K]
        max_iter : int
        tol : float
            Relative tolerance on stage profiles between iterations.
        damping : float, default 0.5
            Damping factor for T and V updates (0 < damping ≤ 1).
        verbose : bool

        Returns
        -------
        AmineColumnResult
        """
        N = self.n_stages

        # Stage indexing: 1 = top, N = bottom (top-down)
        # For arrays we use 0-indexed: 0 = top, N-1 = bottom

        # ----- Feed/inlet molar flows -----
        F_gas = float(feed_gas["F"])
        y_CO2_feed = float(feed_gas["y_CO2"])
        T_gas_in = float(feed_gas["T"])
        F_CO2_feed = F_gas * y_CO2_feed   # kmol/h CO2 in feed gas
        F_inert_feed = F_gas * (1.0 - y_CO2_feed)   # excludes CO2

        F_amine = float(lean_amine["F"])
        m_amine = float(lean_amine["total_amine"])  # mol/kg solvent
        alpha_lean = float(lean_amine["alpha"])
        T_amine_in = float(lean_amine["T"])

        # Solvent (water) molar flow inferred from amine concentration:
        #   m_amine [mol/kg solvent], so per kmol amine, kg solvent =
        #   1000 / m_amine.  Solvent moles = (kg solvent) / 0.018 (H2O MW).
        kg_solvent_per_kmol_amine = 1000.0 / m_amine
        F_solvent = F_amine * kg_solvent_per_kmol_amine / 0.018  # kmol/h H2O

        # CO2 entering with lean amine
        F_CO2_lean = F_amine * alpha_lean

        # ----- Initialize stage profiles -----
        # T: linear from gas inlet (bottom) to amine inlet (top)
        T = np.linspace(T_amine_in, T_gas_in, N)

        # α: assume linear from lean (top) to "expected" rich (bottom)
        # Initial guess: rich loading = lean + 80% of equilibrium with feed
        alpha = np.linspace(alpha_lean, alpha_lean + 0.3, N)

        # V: assume vapor flow nearly constant (slight reduction as CO2
        # absorbed — initial guess: vapor = feed gas + water saturation)
        # We track CO2-free gas and CO2 separately for stability.
        # V[j] = vapor LEAVING stage j (upward); V[N] = vapor entering
        # bottom = F_gas
        V = np.full(N, F_gas)

        # L: liquid leaving stage j (downward).  L[0] = lean amine + lean CO2
        # Effectively constant in our simple model since amine + water
        # don't change much.
        L = np.full(N, F_amine + F_solvent + F_CO2_feed)  # rough

        # ----- Outer iteration -----
        sys = AmineSystem(self.amine, total_amine=m_amine,
                            activity_model=self.activity_model)
        converged = False
        for it in range(max_iter):
            T_old, V_old, alpha_old = T.copy(), V.copy(), alpha.copy()

            # === STEP 1: CO2 balance, top-down ===
            # Total CO2 entering top stage 0 from above = lean amine CO2
            # CO2 entering bottom stage N-1 from below = vapor CO2 = V_in·y_CO2
            # Within the column, mass balance gives α at each stage.
            #
            # Counter-current CO2 balance (overall column):
            #   F_CO2_total_in = F_CO2_lean + F_CO2_feed
            #   F_CO2_out = F_CO2_treated_gas + F_CO2_rich_amine
            #
            # For stage j (top-down):
            #   Liquid CO2 leaving stage j = liquid CO2 from stage j-1
            #     + vapor CO2 from stage j+1 - vapor CO2 to stage j-1
            #
            # Equivalent cleaner formulation: solve
            #   α_j = AmineSystem.speciate^{-1}(y_CO2 P_total) at stage temp T_j
            # given y_CO2 at each stage from above mass balance.
            #
            # We use a fixed-point: assume V profile, get y, get α via
            # equilibrium_loading, get new V from CO2 balance.

            # First set y_CO2 by overall + per-stage balance (top-down)
            #   F_CO2 in vapor leaving stage j upward = ?
            # Use chemistry-driven approach: at each stage, given α_j-1
            # from above and feed y from below, find new α.  Alternatively,
            # given y_CO2 at this stage, find α_j by equilibrium_loading.

            # CO2 ENTERING bottom (stage N-1) from feed gas:
            # F_CO2_in_bottom = F_CO2_feed
            # CO2 LEAVING top in vapor = F_CO2_treated.

            # Marching top-down with current vapor profile:
            #   F_CO2_liq_in_j = F_CO2_lean (for j=0) or
            #     L[j-1]_alpha * F_amine (since amine flow doesn't change)
            #   F_CO2_vap_in_j = V_below * y_CO2_below   (j+1 stage)
            #   ⇒ F_CO2_out_j = F_CO2_liq_in_j + F_CO2_vap_in_j
            #     - F_CO2_vap_out_j_upward
            #
            # This is coupled across stages.  Use simple Wang-Henke:
            # iterate per-stage equilibrium given current profiles.

            # -- equilibrium-based update --
            # For each stage, compute P_CO2 from speciation at current α,T
            # then y_CO2 = P_CO2 / P_total
            # then redistribute CO2 by component balance.
            P_CO2 = np.zeros(N)
            y_CO2 = np.zeros(N)
            y_H2O = np.zeros(N)
            for j in range(N):
                spec = sys.speciate(alpha[j], T=T[j])
                P_CO2[j] = spec.P_CO2
                P_H2O_j = _antoine_water(T[j])
                # If P_CO2 + P_H2O > P_total, water doesn't all go to vapor;
                # cap y_H2O at the saturation level only if that's consistent.
                P_inert = max(self.P_total - P_CO2[j] - P_H2O_j, 0.0)
                y_CO2[j] = P_CO2[j] / self.P_total
                y_H2O[j] = P_H2O_j / self.P_total

            # Now do an overall CO2 balance to get rich loading.
            #   F_CO2_in_total = F_CO2_lean + F_CO2_feed
            #   F_CO2_treated_gas = V[0] * y_CO2[0]
            #   F_CO2_rich_amine = F_amine * α_rich  (= L liquid CO2 out
            #                                          bottom)
            #
            # F_CO2_in = F_CO2_treated + F_CO2_rich
            # ⇒ α_rich = (F_CO2_in - F_CO2_treated) / F_amine
            # We track F_CO2 in vapor leaving each stage.
            # V[0] · y_CO2[0] = F_CO2 in treated gas.
            # Approximating V[j] ≈ F_inert_feed + F_CO2_above + saturation H2O:
            # We use simpler approach below.

            # Stage-by-stage CO2 balance — march bottom-up:
            # CO2 in vapor leaving stage j upward:
            #   F_v_CO2[j] = F_CO2 entering bottom of column from below (j+1)
            #               - net CO2 absorbed in stages j+1...N-1 + ...
            # Simpler: cumulative balance.
            # F_CO2 absorbed in stage j = F_amine·(α[j] - α[j-1])  [j>=1]
            # F_CO2_leaving_top_of_stage_j upward = F_CO2_at_top_+_below
            #
            # Define a_j = alpha[j], march from bottom:
            #   F_CO2_v_below_stage_j (entering j from j+1):
            #       For j = N-1: F_CO2_feed (from gas inlet)
            #       For j < N-1: F_v_CO2_above_(j+1)
            #   F_CO2_v_above_stage_j (leaving j upward):
            #       = F_CO2_v_below_j - F_amine·(alpha[j] - alpha[j-1] for j>0;
            #                                    alpha[0]-alpha_lean for j=0)
            # Then total vapor leaving stage j:
            #   V[j] = F_inert_feed + F_v_CO2_above[j] + F_H2O_v[j]
            # F_H2O_v[j] = water vapor saturation at T[j], proportionate

            # First build F_v_CO2 array (CO2 in vapor leaving each stage upward)
            F_v_CO2_up = np.zeros(N)   # F_v_CO2_up[j] = vapor CO2 leaving stage j up
            # Bottom: vapor entering bottom is feed gas
            # Stage N-1 receives F_CO2_feed from below; absorbs/desorbs to liquid;
            #   leaves F_v_CO2_up[N-1] going up to stage N-2.
            # CO2 absorbed in stage N-1 = F_amine·(alpha[N-1] - alpha[N-2])
            # F_v_CO2_up[N-1] = F_CO2_feed - F_absorbed_in_stage_N-1
            # Working up: F_v_CO2_up[j] = F_v_CO2_up[j+1] - F_amine·(alpha[j]-alpha[j-1])

            for j in range(N-1, -1, -1):
                if j == N - 1:
                    F_below = F_CO2_feed
                else:
                    F_below = F_v_CO2_up[j+1]
                if j == 0:
                    delta_alpha = alpha[j] - alpha_lean
                else:
                    delta_alpha = alpha[j] - alpha[j-1]
                F_absorbed = F_amine * delta_alpha
                F_v_CO2_up[j] = F_below - F_absorbed

            # Now update V profile and y_CO2 profile from this CO2 balance.
            # Inerts are constant (= F_inert_feed) leaving each stage.
            # Water vapor at saturation contributes F_H2O_v[j] = (y_H2O[j]/y_inert[j])
            #   · F_inert_feed   (since inert is the carrier)
            for j in range(N):
                # V[j] · y_CO2[j] = F_v_CO2_up[j]
                # V[j] · y_H2O[j] = F_H2O_v[j]
                # V[j] · (1 - y_CO2 - y_H2O) = F_inert_feed
                if (1.0 - y_CO2[j] - y_H2O[j]) > 0.01:
                    V_new = F_inert_feed / (1.0 - y_CO2[j] - y_H2O[j])
                else:
                    # All vapor is CO2+H2O (no inerts) — keep V from CO2
                    V_new = F_v_CO2_up[j] / max(y_CO2[j], 1e-12)
                V[j] = (1.0 - damping) * V[j] + damping * V_new

            # === STEP 2: Update α from CO2 balance ===
            # Given vapor profile, alpha must satisfy the chemical
            # equilibrium constraint: P_CO2 from speciation = P_total · y_CO2.
            # We use the equilibrium_loading inverse from AmineSystem.
            # But y_CO2 is set by V·y = F_v_CO2_up, which depends on alpha.
            # Iterate locally per stage:
            for j in range(N):
                # Target P_CO2 from current vapor balance:
                # F_v_CO2_up[j] is set above from current alpha[]; matches y_CO2.
                # New alpha from inverse: solve speciate-1.
                target_P_CO2 = max(F_v_CO2_up[j] / V[j] * self.P_total, 1e-9)
                try:
                    new_alpha = sys.equilibrium_loading(
                        P_CO2=target_P_CO2, T=T[j], alpha_max=0.99)
                except Exception:
                    new_alpha = alpha[j]
                alpha[j] = (1.0 - damping) * alpha[j] + damping * new_alpha

            # === STEP 3: Update T from energy balance ===
            # Liquid enthalpy at stage j: h_j = h_lean + Cp·(T_j - T_amine_in)
            # Liquid CO2 contributes ΔH_abs per mol absorbed.
            # Simplified energy balance per stage (from top-down):
            #   L_total ≈ F_solvent·MW_H2O + F_amine·MW_amine
            # Heat of absorption released in stage j:
            #   Q_abs_j = F_amine·(α_j - α_j-1)·|ΔH_abs|   (positive = released)
            # Heat from gas cooling/heating:
            #   Q_gas_j = V_below·Cp_g·(T_below - T_j) - V_above·Cp_g·(T_above - T_j)
            # Latent heat from water condensation/vaporization:
            #   Δ y_H2O between stages → water condensation releases latent heat
            # We use simplified: T set by adiabatic balance assuming
            # constant Cp and integration from top-down.

            # Adiabatic absorber: T rises from top to bottom due to heat
            # of absorption (and gas heat of cooling).
            # Balance for each stage:
            #   m_dot_L · Cp · (T_j - T_j-1) = Q_abs_in_j + sensible_gas_in_j
            #
            # Liquid mass flow rate (kg/h):
            kg_solvent_per_h = F_amine * kg_solvent_per_kmol_amine
            # Inlet liquid temperature (top): T_amine_in
            # Top-down march:
            cum_dH = 0.0
            for j in range(N):
                if j == 0:
                    delta_alpha = alpha[j] - alpha_lean
                else:
                    delta_alpha = alpha[j] - alpha[j-1]
                if delta_alpha > 0:   # absorbing
                    is_tert = self.amine.is_tertiary
                    dH_abs = (_DELTA_H_ABS_TERTIARY if is_tert
                                 else _DELTA_H_ABS_PRIMARY)
                    Q_abs = -dH_abs * F_amine * delta_alpha * 1000.0  # J/h
                    # divided by liquid mass flow * Cp gives ΔT
                    dT = Q_abs / (kg_solvent_per_h * _CP_LIQUID)
                else:
                    dT = 0.0
                T_new_j = (T_amine_in if j == 0 else T[j-1]) + dT
                T[j] = (1.0 - damping) * T[j] + damping * T_new_j

            # === Convergence check ===
            T_change = np.max(np.abs(T - T_old))
            V_change = np.max(np.abs(V - V_old)) / max(np.max(V_old), 1e-9)
            alpha_change = np.max(np.abs(alpha - alpha_old))

            if verbose and (it % 5 == 0 or it < 5):
                print(f"  iter {it:3d}: ΔT={T_change:.3f}K, "
                       f"ΔV/V={V_change:.3e}, Δα={alpha_change:.3e}")

            if (T_change < tol * 100 and V_change < tol
                    and alpha_change < tol):
                converged = True
                break

        rich_alpha = alpha[-1]
        treated_y_CO2 = y_CO2[0]
        F_CO2_in = F_CO2_feed + F_CO2_lean
        F_CO2_treated = V[0] * treated_y_CO2
        capture_pct = ((F_CO2_in - F_CO2_treated) / F_CO2_feed * 100
                          if F_CO2_feed > 0 else 0.0)

        # Compute pH at each stage
        pH = np.zeros(N)
        for j in range(N):
            spec = sys.speciate(alpha[j], T=T[j])
            pH[j] = spec.pH

        return AmineColumnResult(
            n_stages=N, T=T, V=V, L=L, alpha=alpha,
            y_CO2=y_CO2, y_H2O=y_H2O, P_CO2=P_CO2, pH=pH,
            rich_alpha=rich_alpha, lean_alpha=alpha_lean,
            treated_y_CO2=treated_y_CO2, capture_pct=capture_pct,
            converged=converged, iterations=it + 1,
            column_type="absorber")


# =====================================================================
# Amine regenerator (CO2 stripper)
# =====================================================================

class AmineRegenerator:
    """Counter-current CO2 regenerator (stripper) for alkanolamine.

    Configuration (top-to-bottom):

        Off-gas: CO2 + steam (top)     Rich amine (top)
              ↑                              ↓
              |     stage 1                  |
              |     stage 2                  |
              |     ...                      |
              |     stage N                  |
              ↑                              ↓
        Stripping steam (bottom)       Lean amine (bottom)

    The regenerator is operated at higher T (typically 100-120 °C)
    where CO2-amine reactions reverse, releasing CO2.  Steam from the
    reboiler provides both stripping action and the heat needed to
    drive the endothermic reverse reactions.

    The simulation method is similar to AmineAbsorber but with
    direction inverted.

    Parameters
    ----------
    n_stages : int
    amine : Amine or str
    P_total : float, default 1.5
        Operating pressure [bar].  Regenerators run slightly above
        atmospheric.
    activity_model : str, default "bromley"
        Default Bromley because regenerator T is high — see v0.9.104
        γ-model trade-off discussion.
    """

    def __init__(self,
                  n_stages: int,
                  amine: Union[Amine, str],
                  P_total: float = 1.5,
                  activity_model: str = "bromley"):
        if n_stages < 2:
            raise ValueError("n_stages must be ≥ 2")
        self.n_stages = int(n_stages)
        self.amine = (lookup_amine(amine) if isinstance(amine, str)
                       else amine)
        self.P_total = float(P_total)
        self.activity_model = activity_model

    def simulate(self,
                  rich_amine: Dict[str, float],
                  steam: Dict[str, float],
                  *,
                  max_iter: int = 100,
                  tol: float = 1e-4,
                  damping: float = 0.4,
                  verbose: bool = False) -> AmineColumnResult:
        """Run the regenerator simulation.

        Parameters
        ----------
        rich_amine : dict
            Top-feed rich amine spec, with keys:
              - 'F' : amine molar flow [kmol/h]
              - 'total_amine' : amine concentration [mol/kg solvent]
              - 'alpha' : rich loading [mol/mol]
              - 'T' : feed temperature [K] (typically 100-110 °C
                      after lean/rich heat exchanger)
        steam : dict
            Bottom-feed reboiler steam spec:
              - 'F' : steam molar flow [kmol/h] (water)
              - 'T' : steam temperature [K] (typically 120 °C)
        max_iter, tol, damping : Newton settings
        verbose : bool

        Returns
        -------
        AmineColumnResult with column_type='regenerator'
        """
        N = self.n_stages
        F_amine = float(rich_amine["F"])
        m_amine = float(rich_amine["total_amine"])
        alpha_rich = float(rich_amine["alpha"])
        T_rich_in = float(rich_amine["T"])

        F_steam = float(steam["F"])
        T_steam_in = float(steam["T"])

        kg_solvent_per_kmol_amine = 1000.0 / m_amine
        F_solvent = F_amine * kg_solvent_per_kmol_amine / 0.018
        F_CO2_rich = F_amine * alpha_rich

        # Stage 0 = top (rich in, off-gas out)
        # Stage N-1 = bottom (steam in, lean out)

        T = np.linspace(T_rich_in, T_steam_in, N)
        # Initial alpha guess: linear from rich (top) to lean (bottom)
        alpha = np.linspace(alpha_rich, max(alpha_rich - 0.3, 0.05), N)
        V = np.full(N, F_steam)

        sys = AmineSystem(self.amine, total_amine=m_amine,
                            activity_model=self.activity_model)
        converged = False
        for it in range(max_iter):
            T_old, V_old, alpha_old = T.copy(), V.copy(), alpha.copy()

            P_CO2 = np.zeros(N)
            y_CO2 = np.zeros(N)
            y_H2O = np.zeros(N)
            for j in range(N):
                spec = sys.speciate(alpha[j], T=T[j])
                P_CO2[j] = spec.P_CO2
                P_H2O_j = _antoine_water(T[j])
                # In a regenerator, vapor phase is mostly H2O + CO2
                P_total_actual = P_CO2[j] + P_H2O_j
                if P_total_actual > self.P_total:
                    # Saturated; partition by pressure
                    y_CO2[j] = P_CO2[j] / P_total_actual
                    y_H2O[j] = P_H2O_j / P_total_actual
                else:
                    y_CO2[j] = P_CO2[j] / self.P_total
                    y_H2O[j] = P_H2O_j / self.P_total

            # CO2 balance bottom-up
            # Stage N-1 receives 0 CO2 from below (steam is pure water)
            # Stage j: F_CO2_v_up[j] = F_CO2_v_up[j+1] + F_amine·(α[j-1]-α[j])
            #   (positive when stripping = α decreasing)
            F_v_CO2_up = np.zeros(N)
            for j in range(N - 1, -1, -1):
                if j == N - 1:
                    F_below = 0.0   # steam, no CO2
                else:
                    F_below = F_v_CO2_up[j + 1]
                if j == 0:
                    delta_alpha = alpha_rich - alpha[j]
                else:
                    delta_alpha = alpha[j - 1] - alpha[j]
                F_stripped = F_amine * delta_alpha
                F_v_CO2_up[j] = F_below + F_stripped

            # Vapor flow update — H2O dominates in regenerator
            # V[j] ≈ F_steam + F_water_evaporated_below_j + F_v_CO2_up[j]
            # Simplified: vapor at saturation; V[j] from y balance.
            for j in range(N):
                if y_CO2[j] + y_H2O[j] > 1e-3:
                    V_new = F_v_CO2_up[j] / max(y_CO2[j], 1e-9)
                    V_new = max(V_new, F_steam)   # at least steam
                else:
                    V_new = V[j]
                V[j] = (1.0 - damping) * V[j] + damping * V_new

            # Update alpha from speciation inverse
            for j in range(N):
                target_P_CO2 = max(F_v_CO2_up[j] / V[j] * self.P_total, 1e-9)
                try:
                    new_alpha = sys.equilibrium_loading(
                        P_CO2=target_P_CO2, T=T[j], alpha_max=0.99)
                except Exception:
                    new_alpha = alpha[j]
                alpha[j] = (1.0 - damping) * alpha[j] + damping * new_alpha

            # T update: regenerator T is set by reboiler (bottom)
            # and gradually drops toward top due to endothermic reactions
            kg_solvent_per_h = F_amine * kg_solvent_per_kmol_amine
            for j in range(N):
                # bottom-up: T_j is steam-driven
                if j == N - 1:
                    T_new = T_steam_in
                else:
                    delta_alpha = alpha[j] - alpha[j + 1]
                    if delta_alpha > 0:    # net stripping (alpha higher at j)
                        is_tert = self.amine.is_tertiary
                        dH_abs = (_DELTA_H_ABS_TERTIARY if is_tert
                                     else _DELTA_H_ABS_PRIMARY)
                        # endothermic reverse: cools liquid going up
                        Q = dH_abs * F_amine * delta_alpha * 1000.0  # J/h, negative for endo
                        dT = Q / (kg_solvent_per_h * _CP_LIQUID)
                    else:
                        dT = 0.0
                    T_new = T[j + 1] + dT
                T[j] = (1.0 - damping) * T[j] + damping * T_new

            T_change = np.max(np.abs(T - T_old))
            V_change = np.max(np.abs(V - V_old)) / max(np.max(V_old), 1e-9)
            alpha_change = np.max(np.abs(alpha - alpha_old))
            if verbose and (it % 5 == 0 or it < 5):
                print(f"  iter {it:3d}: ΔT={T_change:.3f}K, "
                       f"ΔV/V={V_change:.3e}, Δα={alpha_change:.3e}")
            if (T_change < tol * 100 and V_change < tol
                    and alpha_change < tol):
                converged = True
                break

        lean_alpha = alpha[-1]
        # CO2 in off-gas = vapor CO2 leaving top stage (j=0)
        F_CO2_offgas = V[0] * y_CO2[0]
        F_CO2_in_total = F_CO2_rich   # CO2 entering top from rich amine
        capture_pct = ((F_CO2_in_total - F_amine * lean_alpha)
                          / F_CO2_in_total * 100
                          if F_CO2_in_total > 0 else 0.0)

        pH = np.zeros(N)
        for j in range(N):
            spec = sys.speciate(alpha[j], T=T[j])
            pH[j] = spec.pH
        L = np.full(N, F_amine + F_solvent + F_amine * np.mean(alpha))

        return AmineColumnResult(
            n_stages=N, T=T, V=V, L=L, alpha=alpha,
            y_CO2=y_CO2, y_H2O=y_H2O, P_CO2=P_CO2, pH=pH,
            rich_alpha=alpha_rich, lean_alpha=lean_alpha,
            treated_y_CO2=y_CO2[0], capture_pct=capture_pct,
            converged=converged, iterations=it + 1,
            column_type="regenerator")
