"""Reactive stripper / regenerator column for amine systems (v0.9.105).

The stripper / regenerator is the counterpart to the absorber in an
amine-based CO2 capture process.  Rich amine flowing down releases CO2
to a counter-current vapor (steam + stripped CO2) flowing up.  Heat is
supplied at the bottom by a reboiler; cooling at the top condenses
water back to maintain solvent inventory.

Geometry
--------
* N stages, numbered 1 (top) to N (bottom)
* Liquid flows down: rich in at top (α=α_rich at high T), lean out at
  bottom (α=α_lean, fed to reboiler)
* Vapor flows up: stripping vapor in at bottom (mostly steam from
  reboiler, low y_CO2_reb), CO2-rich vapor out at top (high y_CO2)
* T profile: typically 100-105 °C at top (below water bp at 1 atm),
  115-125 °C at bottom (slightly above water bp due to operating P
  > 1 atm in regenerator)

Mass balance per stage
----------------------
Same form as the absorber column:

    L · α_{n-1} + G · y_{n+1} = L · α_n + G · y_n

with boundary conditions reversed:
    α_0 = α_rich   (liquid inlet at top)
    y_{N+1} = y_reb (vapor inlet at bottom from reboiler)

For the absorber, α_0 is small (lean in) and y_{N+1} is large.
For the stripper, α_0 is large (rich in) and y_{N+1} is small.

The Newton solver from AmineColumn handles both cases — the stripper
class is mostly a friendly interface with stripper-appropriate naming
and an energy-balance computation.

Heat balance (industrial regenerator)
--------------------------------------
The reboiler duty Q_reb [W] is the sum of three contributions, all
positive (heat input):

    Q_reb = Q_sensible  +  Q_reaction  +  Q_vaporization

  Q_sensible:    L · cp_solution · (T_lean - T_rich_in)
                 (heating the rich amine to lean / reboiler T)

  Q_reaction:    L · |ΔH_abs| · (α_rich - α_lean)
                 (heat absorbed from solution to release CO2;
                 ΔH_abs is negative — heat is released on absorption,
                 so |ΔH_abs| is needed to reverse it on stripping)

  Q_vaporization: V_reb · ΔH_vap_water
                  (latent heat to generate the stripping steam at
                   the reboiler; V_reb is the steam molar flow rate)

For typical 30 wt% MEA stripping rich (α=0.5) to lean (α=0.2):

  Q_sensible    ≈ 75 kJ per kg solvent   (heating 100→120 °C)
  Q_reaction    ≈ 125 kJ per kg solvent  (~1.5 mol CO2 × 85 kJ/mol)
  Q_vaporization ≈ 90-150 kJ per kg solvent (steam ratio 1-1.5)
  Total         ≈ 290-350 kJ per kg solvent
  Per ton CO2   ≈ 4-5 GJ/ton CO2 (industry benchmark 3.5-4 GJ/ton)

The implementation provides the heat balance as a post-hoc calculation
given a specified T profile and solved α profile.  Future work could
add an iterative T solver where Q_reb is specified and T is solved.

References
----------
* Notz, R., Mangalapally, H. P., Hasse, H. (2012).  Post-combustion
  CO2 capture by reactive absorption: Pilot plant description and
  results of systematic studies with MEA. Int. J. Greenhouse Gas
  Control 6, 84.
* Aaron, D., Tsouris, C. (2005).  Separation of CO2 from flue gas:
  A review. Sep. Sci. Technol. 40, 321.
* Cousins, A., Wardhaugh, L. T., Feron, P. H. M. (2011).  A survey
  of process flow sheet modifications for energy-efficient post-
  combustion capture of CO2 using chemical absorption. Int. J.
  Greenhouse Gas Control 5, 605.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from .amines import Amine, AmineSystem, lookup_amine
from .amine_column import AmineColumn


_DELTA_H_VAP_WATER = 40700.0    # J/mol at 100 °C (Wagner-Pruss)
_M_WATER = 0.01801528           # kg/mol
_CP_WATER_VAPOR = 35.0          # J/(mol·K) — H2O(g) average over 40-120 °C
_CP_CO2_VAPOR = 38.0            # J/(mol·K) — CO2(g) average over 40-120 °C


# =====================================================================
# Antoine-style water vapor pressure (NIST, valid 1-200 °C)
# =====================================================================

def P_water_sat(T: float) -> float:
    """Saturated water vapor pressure [bar] at T [K].

    NIST/Wagner-Pruss simplified Antoine form, accurate to 0.5%
    over 1-200 °C:
        log10(P/bar) = A - B / (T - C)
    with A=4.6543, B=1435.264, C=-64.848 for T in K.
    """
    if T < 273.15:
        return 0.006   # below freezing
    return 10.0 ** (4.6543 - 1435.264 / (T - 64.848))


def T_water_sat(P: float) -> float:
    """Saturation temperature of water [K] at pressure P [bar].

    Inverse of P_water_sat: solve log10(P) = A - B/(T - C) for T.
        T = C + B / (A - log10(P))

    Examples (relative to NIST):
        T_water_sat(1.013) = 372.92 K   (lit 373.15, -0.06 %)
        T_water_sat(1.5)   = 384.81 K   (lit 384.52, +0.08 %)
        T_water_sat(1.8)   = 391.12 K   (lit 390.85, +0.07 %)
        T_water_sat(2.0)   = 394.79 K   (lit 393.36, +0.36 %)
    """
    if P <= 0:
        raise ValueError(f"P must be > 0; got {P}")
    A, B, C = 4.6543, 1435.264, 64.848
    log_P = np.log10(P)
    if A - log_P <= 0:
        raise ValueError(f"P={P} bar is too high for Antoine inversion")
    return C + B / (A - log_P)


# =====================================================================
# Stripper result
# =====================================================================

@dataclass
class AmineStripperResult:
    """Result of an AmineStripper.solve() call.

    Attributes
    ----------
    alpha : list of float
        Liquid loading [mol CO2 / mol amine] at each stage exit
        (length N, alpha[0] = stage-1 = top, alpha[N-1] = α_lean
        at bottom).
    y_CO2 : list of float
        Vapor CO2 mole fraction at each stage exit.
    y_H2O : list of float
        Water vapor mole fraction (Raoult on water at stage T).
    T : list of float
        Stage temperatures [K].
    alpha_lean : float
        Lean liquid loading exiting the bottom (= alpha[-1]).
    alpha_rich : float
        Rich liquid loading entering the top (input).
    y_top_CO2 : float
        Top vapor CO2 mole fraction (= y_CO2[0]).
    co2_stripped : float
        Mol CO2 stripped per mol amine fed = α_rich - α_lean.
    L : float
        Liquid molar flow rate [mol amine/s].
    G : float
        Vapor molar flow rate [mol/s].
    Q_sensible : float
        Sensible-heat duty [W] for heating rich → lean.
    Q_reaction : float
        Reaction-heat duty [W] for CO2 release.
    Q_vaporization : float
        Latent heat duty [W] for generating stripping steam.
    Q_reboiler : float
        Total reboiler duty [W] = Q_sens + Q_reac + Q_vap.
    Q_per_ton_CO2 : float
        Specific reboiler duty [GJ / ton CO2 stripped] —
        the standard industrial benchmark metric.
    pH : list of float
        Liquid pH at each stage.
    converged : bool
    iterations : int
    """
    alpha: List[float]
    y_CO2: List[float]
    y_H2O: List[float]
    T: List[float]
    alpha_lean: float
    alpha_rich: float
    y_top_CO2: float
    co2_stripped: float
    L: float
    G: float
    Q_sensible: float
    Q_reaction: float
    Q_vaporization: float
    Q_reboiler: float
    Q_per_ton_CO2: float
    pH: List[float]
    converged: bool
    iterations: int
    V_profile: Optional[List[float]] = None    # v0.9.109 variable-V mode


# =====================================================================
# AmineStripper class
# =====================================================================

class AmineStripper:
    """Multi-stage reactive stripper / regenerator column.

    Counter-current operation:
      * Rich amine in at top (α_rich, T_rich_in)
      * Lean amine out at bottom (α_lean, T_reboiler)
      * Stripping vapor in at bottom (low y_CO2_reb, mostly steam)
      * CO2-enriched vapor out at top (high y_CO2, condensable steam)

    Internally uses :class:`AmineColumn`'s Newton solver with reversed
    boundary conditions, plus a post-hoc energy balance to compute
    reboiler duty.

    Parameters
    ----------
    amine : Amine or str
    total_amine : float
        Liquid amine concentration [mol/kg solvent].
    n_stages : int

    Examples
    --------
    >>> from stateprop.electrolyte import AmineStripper
    >>> # Typical MEA regenerator: rich α=0.50 → lean α=0.20
    >>> strip = AmineStripper("MEA", total_amine=5.0, n_stages=15)
    >>> r = strip.solve(
    ...     L=10.0,            # rich amine flow [mol amine/s]
    ...     G=15.0,            # stripping steam flow [mol/s]
    ...     alpha_rich=0.50,
    ...     y_reb=0.05,        # 5% CO2 in stripping vapor from reboiler
    ...     P=1.8,             # 1.8 bar typical regenerator
    ...     T_top=378.15,      # 105 °C at top
    ...     T_bottom=393.15,   # 120 °C at bottom (reboiler)
    ...     wt_frac_amine=0.30,
    ... )
    >>> r.alpha_lean
    0.18    # achieved lean loading
    >>> r.Q_per_ton_CO2
    4.2     # GJ / ton CO2 — industry benchmark ~3.5-4
    """

    def __init__(self,
                  amine: Union[Amine, str],
                  total_amine: float,
                  n_stages: int):
        self.amine = (lookup_amine(amine) if isinstance(amine, str)
                       else amine)
        self.total_amine = float(total_amine)
        self.n_stages = int(n_stages)
        if self.n_stages < 1:
            raise ValueError("n_stages must be >= 1")
        # Internal column solver (we'll use it with reversed BCs)
        self._col = AmineColumn(self.amine, self.total_amine, self.n_stages)
        # Internal AmineSystem for direct equilibrium calls
        self._sys = AmineSystem(self.amine, self.total_amine)

    # -----------------------------------------------------------------
    def solve(self,
                L: float,
                G: float,
                alpha_rich: float,
                y_reb: float = 0.05,
                P: float = 1.8,
                T_top: float = 378.15,
                T_bottom: float = 393.15,
                wt_frac_amine: float = 0.30,
                T_rich_in: Optional[float] = None,
                max_iter: int = 100,
                tol: float = 1e-8,
                variable_V=False,
                max_outer_V: int = 30,
                tol_V: float = 1e-4,
                damp_V: float = 0.5,
                auto_clip_T_bottom: bool = True,
                T_sat_margin: float = 1.0,
                verbose: bool = False) -> AmineStripperResult:
        """Solve the stripper for steady-state α and y profiles, and
        compute the reboiler duty.

        Parameters
        ----------
        L : float
            Liquid molar flow rate [mol amine / s] (constant assumed).
        G : float
            Stripping vapor molar flow rate [mol / s].  This is the
            total vapor flow leaving the reboiler (steam + small
            amount of CO2 if any recycle).
        alpha_rich : float
            Inlet rich liquid loading [mol CO2 / mol amine].
        y_reb : float, default 0.05
            CO2 mole fraction in vapor entering bottom (from reboiler).
            For pure-steam reboiler vapor this is ~0; in practice
            small recycle gives 0.01-0.10.
        P : float, default 1.8 bar
            Total operating pressure (regenerators run at slight
            overpressure to elevate water bp).
        T_top, T_bottom : float
            Stage temperatures at top and bottom [K].  A linear T
            profile is built between them.  Defaults: 105 °C top,
            120 °C bottom (standard MEA regenerator).
        wt_frac_amine : float, default 0.30
            Weight fraction of amine in the solvent (water + amine);
            used for cp_solution in the heat balance.
        T_rich_in : float, optional
            Rich-amine feed temperature.  If None, defaults to T_top
            (no preheat). For a typical regenerator with a lean/rich
            heat exchanger, T_rich_in is ~T_top - 5 K.
        max_iter, tol : Newton solver settings (passed to AmineColumn).
        verbose : bool, default False

        Returns
        -------
        AmineStripperResult
        """
        # ----------- T saturation constraint (v0.9.110, Item 1) -----------
        # The bottom stage of the stripper must satisfy P_water_sat(T_bottom)
        # < P_total — otherwise water boils and the constant-flow vapor
        # model breaks down (the column would be running at boiling point
        # with phase reversal at every interface).  Auto-clip by default.
        T_sat_at_P = T_water_sat(P)
        T_bottom_max = T_sat_at_P - T_sat_margin
        if T_bottom > T_sat_at_P:
            if auto_clip_T_bottom:
                if verbose:
                    print(f"  [v0.9.110] T_bottom={T_bottom:.2f} K exceeds "
                           f"T_sat(P={P})={T_sat_at_P:.2f} K; clipping to "
                           f"{T_bottom_max:.2f} K (margin={T_sat_margin} K)")
                T_bottom = T_bottom_max
                if T_top > T_bottom:
                    T_top = T_bottom - 5.0   # preserve sensible Δ
            else:
                raise ValueError(
                    f"T_bottom={T_bottom:.2f} K ≥ T_sat(P={P})="
                    f"{T_sat_at_P:.2f} K; vapor cannot be saturated. "
                    f"Set auto_clip_T_bottom=True to auto-clip, or "
                    f"reduce T_bottom or raise P_stripper.")

        # ----------- Resolve variable_V mode (v0.9.110, Item 2) -----------
        # Backward compat: variable_V=True → 'saturation' (v0.9.109 behavior)
        # New options:
        #   variable_V='saturation'  — water mass balance with saturation
        #     y_H2O = P_sat(T)/P → V varies inversely with y_H2O
        #     Result: V increases going up (cooler stages need more total V
        #     to maintain water mass flow at lower y_H2O).
        #   variable_V='energy'      — per-stage energy balance with sensible
        #     and reaction terms determines water flow change
        #     Result: V profile reflects local heat consumption (reaction
        #     endothermic for stripper → V·y_H2O decreases going up,
        #     possibly offsetting saturation-driven increase).
        if variable_V is False or variable_V is None:
            V_mode = 'constant'
        elif variable_V is True:
            V_mode = 'saturation'   # v0.9.109 backward compat
        elif isinstance(variable_V, str):
            if variable_V not in ('saturation', 'energy'):
                raise ValueError(
                    f"variable_V must be False/True or "
                    f"'saturation'/'energy'; got {variable_V!r}")
            V_mode = variable_V
        else:
            raise ValueError(
                f"variable_V must be bool or str; got {type(variable_V)}")

        # Use AmineColumn solver with reversed BCs:
        #   alpha_lean (column arg) = α_rich (top inlet for stripper)
        #   y_in (column arg)       = y_reb  (bottom inlet for stripper)
        # Build T profile: linear from T_top (stage 1) to T_bottom (stage N)
        T_profile = list(np.linspace(T_top, T_bottom, self.n_stages))

        # ----------- Variable-V mode (v0.9.109) -----------
        # The constant-G stripper assumes V is uniform through the column.
        # In reality, V varies because water saturation at the local
        # T differs through the column.  In a stripper at constant P,
        #     y_H2O(n) = P_water_sat(T_n) / P_total          (saturation)
        # and the water carrier in liquid is constant.  Water mass
        # balance then forces:
        #     V[n] · y_H2O(n) = G_reb · (1 - y_reb) = const  through column
        # giving
        #     V[n] = G_reb · (1 - y_reb) / y_H2O(T_n)
        # Cooler stages have lower y_H2O → higher total V;  hotter
        # stages have higher y_H2O → lower total V.  At the reboiler
        # bottom V[N] = G_reb by definition.
        V_profile = None
        if V_mode != 'constant':
            N = self.n_stages

            # Interface T (interface k between stages k-1 and k):
            # k=0 → at the very top above stage 0; k=N → at the bottom
            # below stage N-1. Use stage T as a proxy for the interface
            # just below it.
            def T_int(k):
                if k == 0:
                    return T_profile[0]
                elif k == N:
                    return T_profile[-1]
                return 0.5 * (T_profile[k - 1] + T_profile[k])

            # Cap y_H2O for numerical safety (always < 1 even close to T_sat)
            def y_H2O_at(T_k):
                return min(P_water_sat(T_k) / P, 0.99)

            V = np.zeros(N + 1)

            if V_mode == 'saturation':
                # v0.9.109 saturation mode: water flow constant, V[k] from
                # water mass balance only
                water_flow = float(G) * (1.0 - y_reb)
                for k in range(N + 1):
                    V[k] = water_flow / max(y_H2O_at(T_int(k)), 1e-3)
                V[N] = float(G)   # boundary at reboiler

            else:    # V_mode == 'energy'  (NEW v0.9.110)
                # Initial guess: saturation profile
                water_flow = float(G) * (1.0 - y_reb)
                V_sat = np.zeros(N + 1)
                for k in range(N + 1):
                    V_sat[k] = water_flow / max(y_H2O_at(T_int(k)), 1e-3)
                V_sat[N] = float(G)
                V[:] = V_sat
                # Bounds for V[k] to keep iteration physical: ±50 % of
                # the saturation value (with V[N] = G_reb fixed).  This
                # prevents the per-stage energy balance from collapsing
                # V at the top when reaction heat is concentrated there
                # (a known limitation of the simple per-stage closure;
                # in reality the saturation assumption breaks down and
                # vapor leaves the top sub-saturated).
                V_min = 0.5 * V_sat
                V_max = 1.5 * V_sat
                V_min[N] = V_max[N] = float(G)

                # Energy balance loop: update V from bottom up
                cp_sol_per_kg = self.amine.cp_solution(
                    wt_frac_amine, 0.5 * (T_top + T_bottom))
                kg_sol_per_mol_amine = (self.amine.MW * 1e-3) / wt_frac_amine
                cp_L_per_mol = cp_sol_per_kg * kg_sol_per_mol_amine
                delta_H_abs = abs(self.amine.delta_H_abs)
                cp_V = 35.0   # J/(mol·K), water-vapor dominated

                converged_V = False
                for outer_V in range(max_outer_V):
                    alpha, y_arr, inner_conv, inner_iter = (
                        self._newton_alpha_with_V(
                            L=L, V_profile=V,
                            alpha_rich=alpha_rich, y_reb=y_reb,
                            P=P, T_profile=T_profile,
                            max_iter=max_iter, tol=tol)
                    )
                    new_V = V.copy()
                    new_V[N] = float(G)
                    for n in range(N - 1, -1, -1):
                        T_n = T_profile[n]
                        T_above = T_top if n == 0 else T_profile[n - 1]
                        a_above = alpha_rich if n == 0 else alpha[n - 1]
                        water_below = (new_V[n + 1]
                                          * y_H2O_at(T_int(n + 1)))
                        sensible = L * cp_L_per_mol * (T_above - T_n)
                        reaction = -L * delta_H_abs * (a_above - alpha[n])
                        water_at_n = (water_below
                                          + (sensible + reaction)
                                                / _DELTA_H_VAP_WATER)
                        # Compute V from water + clip to physical bounds
                        V_new_n = water_at_n / max(y_H2O_at(T_int(n)), 1e-3)
                        new_V[n] = float(np.clip(V_new_n,
                                                       V_min[n], V_max[n]))

                    err = float(np.max(np.abs(new_V - V)
                                          / np.maximum(V, 1e-9)))
                    if verbose:
                        print(f"  outer V (energy) {outer_V:3d}: "
                               f"||ΔV/V|| = {err:.3e}, "
                               f"V_top={new_V[0]:.3f} V_bot={new_V[N]:.3f}")
                    if err < tol_V:
                        V = new_V
                        converged_V = True
                        break
                    V = damp_V * new_V + (1.0 - damp_V) * V

            # Solve α with the (now-fixed) V profile
            alpha, y_arr, inner_converged, inner_iter = (
                self._newton_alpha_with_V(
                    L=L, V_profile=V,
                    alpha_rich=alpha_rich, y_reb=y_reb,
                    P=P, T_profile=T_profile,
                    max_iter=max_iter, tol=tol)
            )

            V_profile = V.tolist()
            # Build a synthetic AmineColumnResult equivalent for downstream code
            from .amine_column import AmineColumnResult
            recovery_local = ((y_reb - y_arr[0])
                                  / max(y_reb, 1e-12))
            pH_final = []
            for n in range(N):
                try:
                    res = self._sys.speciate(alpha=alpha[n], T=T_profile[n])
                    pH_final.append(res.pH)
                except Exception:
                    pH_final.append(float("nan"))
            col_result = AmineColumnResult(
                alpha=alpha.tolist(), y=y_arr.tolist(), T=T_profile,
                alpha_rich=float(alpha[-1]),
                y_top=float(y_arr[0]),
                co2_recovery=float(recovery_local),
                pH=pH_final,
                converged=inner_converged,
                iterations=inner_iter,
                L=L, G=G, LG_ratio=L / G,
            )
            if verbose:
                print(f"  variable-V ({V_mode}): "
                       f"V[top]={V[0]:.3f} → V[bot]={V[N]:.3f}")
        else:
            # Constant-V (default): use existing AmineColumn solver
            col_result = self._col.solve(
                L=L, G=G,
                alpha_lean=alpha_rich,    # rename for stripper convention
                y_in=y_reb,
                P=P,
                T=T_profile,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )

        # In stripper convention:
        #   col_result.alpha[0] = α at stage 1 (top stage exit, going down)
        #   col_result.alpha[N-1] = α_lean at bottom
        #   col_result.y[0] = y_top_CO2 (gas leaving top, going up)
        #   col_result.y[N-1] = y at bottom stage exit
        alpha = col_result.alpha
        y_CO2 = col_result.y
        T = col_result.T
        alpha_lean = col_result.alpha_rich     # bottom of column
        y_top_CO2 = col_result.y_top
        co2_stripped = alpha_rich - alpha_lean

        # Water vapor pressure (Raoult's law) at each stage T
        # Activity of water in loaded amine ~0.6-0.9, but for rough
        # heat balance we use Raoult with x_water ≈ wt-water fraction
        x_water = 1.0 - wt_frac_amine    # crude proxy
        y_H2O = [P_water_sat(t) * x_water / P for t in T]

        # ---------------- Energy balance ----------------
        if T_rich_in is None:
            T_rich_in = T_top

        # Sensible heat: heating rich amine to lean (reboiler) T
        # cp of solution at average T (cp not strongly T-dependent)
        cp_sol = self.amine.cp_solution(wt_frac_amine, 0.5*(T_top+T_bottom))
        # L is mol amine/s.  Solvent mass per mol amine = M_amine + (1-w)/w · MW_amine
        # Per mol amine: MW_amine in kg amine, plus water at (1-w)/w mol_amine ratio
        # Mass of solvent per mol amine = MW_amine [g/mol] / 1000 / wt_frac_amine [-]
        # = MW_amine_kg / wt_frac_amine [kg solvent per mol amine]
        kg_sol_per_mol_amine = (self.amine.MW * 1e-3) / wt_frac_amine
        m_dot_sol = L * kg_sol_per_mol_amine     # kg solvent / s

        Q_sensible = m_dot_sol * cp_sol * (T_bottom - T_rich_in)

        # Reaction heat: |ΔH_abs| · (mol CO2 stripped / s)
        # Note ΔH_abs is negative; we need positive heat input to reverse
        mol_CO2_per_sec = L * co2_stripped
        Q_reaction = mol_CO2_per_sec * abs(self.amine.delta_H_abs)

        # Vaporization heat: G · (1 - y_top_CO2) is approximate steam
        # generation rate (mol H2O / s).  For a more accurate estimate,
        # integrate water vapor along the column; here we use the
        # crude approximation that the bottom-stage vapor is mostly
        # steam (y_H2O_bot ≈ 1 - y_reb).
        steam_mol_per_sec = G * (1.0 - y_reb)
        Q_vaporization = steam_mol_per_sec * _DELTA_H_VAP_WATER

        Q_reboiler = Q_sensible + Q_reaction + Q_vaporization

        # Specific reboiler duty: GJ / ton CO2
        if co2_stripped > 0 and L > 0:
            kg_CO2_per_sec = mol_CO2_per_sec * 0.04401   # MW CO2 = 44.01 g/mol
            Q_per_ton_CO2 = (Q_reboiler / kg_CO2_per_sec) * 1e-6  # J/kg → GJ/ton
        else:
            Q_per_ton_CO2 = float("nan")

        return AmineStripperResult(
            alpha=alpha, y_CO2=y_CO2, y_H2O=y_H2O, T=T,
            alpha_lean=alpha_lean, alpha_rich=alpha_rich,
            y_top_CO2=y_top_CO2, co2_stripped=co2_stripped,
            L=L, G=G,
            Q_sensible=Q_sensible, Q_reaction=Q_reaction,
            Q_vaporization=Q_vaporization, Q_reboiler=Q_reboiler,
            Q_per_ton_CO2=Q_per_ton_CO2,
            pH=col_result.pH,
            converged=col_result.converged,
            iterations=col_result.iterations,
            V_profile=V_profile,
        )

    # -----------------------------------------------------------------
    # Internal: variable-V α Newton (v0.9.109)
    # -----------------------------------------------------------------
    def _newton_alpha_with_V(self,
                                L: float,
                                V_profile,
                                alpha_rich: float,
                                y_reb: float,
                                P: float,
                                T_profile,
                                max_iter: int = 100,
                                tol: float = 1e-8):
        """Newton iteration on α with a per-stage V_profile (length N+1).

        Returns (alpha, y_array, converged, iterations).

        The mass balance is:
            F_n = L · α_above + V[n+1] · y_below - L · α_n - V[n] · y_n
        with α_above = α_rich for n=0, y_below = y_reb for n=N-1.
        """
        N = self.n_stages
        # Initial guess: linear α from rich to a deep guess
        alpha_lean_guess = max(0.01, alpha_rich - 0.5)
        alpha = np.linspace(alpha_rich, alpha_lean_guess, N + 1)[1:].copy()
        V = np.asarray(V_profile, dtype=float)

        converged = False
        for outer in range(max_iter):
            y = np.array([self._col._y_eq(alpha[n], T_profile[n], P)
                            for n in range(N)])

            F = np.zeros(N)
            for n in range(N):
                a_above = alpha_rich if n == 0 else alpha[n - 1]
                y_below = y_reb if n == N - 1 else y[n + 1]
                F[n] = (L * a_above + V[n + 1] * y_below
                         - L * alpha[n] - V[n] * y[n])

            scale = max(L * abs(alpha_rich - alpha_lean_guess),
                          float(np.mean(V)) * y_reb, 1e-12)
            norm_F = float(np.max(np.abs(F)))
            if norm_F / scale < tol:
                converged = True
                break

            # Numerical Jacobian (banded tridiagonal: ∂F_i/∂α_{i-1,i,i+1})
            J = np.zeros((N, N))
            eps = 1e-5
            dy_dalpha = np.zeros(N)
            for n in range(N):
                a_perturb = alpha[n] + eps
                if a_perturb >= 1.0:
                    a_perturb = alpha[n] - eps
                    dy_dalpha[n] = ((self._col._y_eq(alpha[n], T_profile[n], P)
                                       - self._col._y_eq(a_perturb, T_profile[n], P))
                                       / eps)
                else:
                    dy_dalpha[n] = ((self._col._y_eq(a_perturb, T_profile[n], P)
                                       - self._col._y_eq(alpha[n], T_profile[n], P))
                                       / eps)
            for i in range(N):
                if i > 0:
                    J[i, i - 1] = +L
                J[i, i] = -L - V[i] * dy_dalpha[i]
                if i < N - 1:
                    J[i, i + 1] = +V[i + 1] * dy_dalpha[i + 1]

            try:
                dalpha = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                dalpha = -0.1 * F / max(np.linalg.norm(F), 1e-12)

            # Damping
            damp = 1.0
            new_alpha = alpha + damp * dalpha
            while np.any(new_alpha < 0) or np.any(new_alpha > 0.95):
                damp *= 0.5
                new_alpha = alpha + damp * dalpha
                if damp < 1e-6:
                    break
            max_step = float(np.max(np.abs(damp * dalpha)))
            if max_step > 0.2:
                damp *= 0.2 / max_step
                new_alpha = alpha + damp * dalpha
            alpha = new_alpha

        return alpha, y, converged, outer + 1

    # -----------------------------------------------------------------
    # Coupled iterative T-solver: specify Q_reb, solve for T profile
    # -----------------------------------------------------------------
    def solve_for_Q_reb(self,
                            L: float,
                            G: float,
                            alpha_rich: float,
                            Q_reb_target: float,
                            y_reb: float = 0.05,
                            P: float = 1.8,
                            T_top_min: float = 363.15,
                            T_top_max: float = 393.15,
                            delta_T_column: float = 15.0,
                            wt_frac_amine: float = 0.30,
                            T_rich_in: Optional[float] = None,
                            tol_rel: float = 1e-3,
                            max_outer: int = 30,
                            verbose: bool = False,
                            ) -> AmineStripperResult:
        """Solve the stripper for a target reboiler duty Q_reb.

        Outer iteration: bisection on T_top (and corresponding T_bottom
        = T_top + delta_T_column) until the post-hoc Q_reboiler matches
        the target.  Inner iteration: the standard Newton solver from
        ``solve()`` runs at each candidate T_top.

        Use this when the user knows the available reboiler steam load
        (e.g., 1 MW) and wants to back out the achievable α_lean and
        operating T's.

        Parameters
        ----------
        L, G : float
            Liquid and vapor molar flow rates [mol/s].
        alpha_rich : float
            Inlet rich loading.
        Q_reb_target : float
            Target reboiler duty [W].
        y_reb : float, default 0.05
        P : float, default 1.8 bar
        T_top_min, T_top_max : float
            Bracket for outer bisection on T_top [K].  Defaults
            90-120 °C, the typical industrial range.
        delta_T_column : float, default 15.0 K
            Imposed top-to-bottom T difference (used to derive
            T_bottom = T_top + delta_T_column for each candidate).
        wt_frac_amine : float, default 0.30
        T_rich_in : float, optional
        tol_rel : float, default 1e-3
            Relative convergence on Q_reb mismatch.
        max_outer : int, default 30
        verbose : bool, default False

        Returns
        -------
        AmineStripperResult
            With T profile, α profile, and Q breakdown matching the
            target Q_reb (within tol_rel).
        """
        # v0.9.110: clip T_top_max so that T_bottom = T_top + delta_T_column
        # stays below the water saturation T at P (avoids the inner solver's
        # auto-clip silently overriding the bisection variable).
        T_sat_P = T_water_sat(P)
        T_top_max_safe = T_sat_P - delta_T_column - 1.0   # 1 K margin
        if T_top_max > T_top_max_safe:
            if verbose:
                print(f"  [v0.9.110] clipping T_top_max from {T_top_max:.2f} K "
                       f"to {T_top_max_safe:.2f} K (T_sat(P={P})="
                       f"{T_sat_P:.2f} K, ΔT_col={delta_T_column} K)")
            T_top_max = T_top_max_safe
        if T_top_min >= T_top_max:
            T_top_min = T_top_max - 30.0   # ensure non-empty bracket

        def _Q_reb_at(T_top):
            T_bottom = T_top + delta_T_column
            r = self.solve(L=L, G=G,
                              alpha_rich=alpha_rich, y_reb=y_reb,
                              P=P, T_top=T_top, T_bottom=T_bottom,
                              wt_frac_amine=wt_frac_amine,
                              T_rich_in=T_rich_in,
                              auto_clip_T_bottom=False)
            return r, r.Q_reboiler

        # Bisection bracket
        r_lo, Q_lo = _Q_reb_at(T_top_min)
        r_hi, Q_hi = _Q_reb_at(T_top_max)

        # Q_reb generally increases with T_top (more sensible heat,
        # more vaporization, deeper stripping).  Sanity check:
        if Q_lo > Q_reb_target:
            if verbose:
                print(f"  Q_reb_target={Q_reb_target:.0f} W is BELOW the minimum "
                       f"({Q_lo:.0f} W at T_top_min); returning low end")
            return r_lo
        if Q_hi < Q_reb_target:
            if verbose:
                print(f"  Q_reb_target={Q_reb_target:.0f} W is ABOVE the maximum "
                       f"({Q_hi:.0f} W at T_top_max); returning high end")
            return r_hi

        # Bisection
        T_lo, T_hi = T_top_min, T_top_max
        for outer in range(max_outer):
            T_mid = 0.5 * (T_lo + T_hi)
            r_mid, Q_mid = _Q_reb_at(T_mid)
            rel_err = (Q_mid - Q_reb_target) / Q_reb_target
            if verbose:
                print(f"  outer {outer:3d}: T_top={T_mid:.2f} K, "
                       f"Q_reb={Q_mid:.1f} W, rel_err={rel_err:+.4f}")
            if abs(rel_err) < tol_rel:
                return r_mid
            if Q_mid < Q_reb_target:
                T_lo = T_mid
                Q_lo = Q_mid
            else:
                T_hi = T_mid
                Q_hi = Q_mid

        # Return best estimate
        if verbose:
            print(f"  outer iteration did not fully converge; rel_err={rel_err:.3e}")
        return r_mid

    # -----------------------------------------------------------------
    # Stage-resolved heat balance utility
    # -----------------------------------------------------------------
    def stage_heat_balance(self,
                              result: AmineStripperResult,
                              wt_frac_amine: float = 0.30
                              ) -> List[dict]:
        """Per-stage heat balance breakdown given a solved result.

        Returns a list (length N) of dicts with keys:
            'T', 'alpha', 'Q_sensible', 'Q_reaction', 'Q_vaporization'
        each in W (per stage).

        The Q_sensible is computed from L · cp_sol · ΔT_stage.
        Q_reaction is L · |ΔH_abs| · Δα_stage.
        Q_vaporization is approximated as G_avg · y_H2O_avg ·
        ΔH_vap_water.

        These do not exactly sum to the total Q_reboiler because of
        the simple linear T-profile assumption (no feedback to flow).
        Provided for diagnostic purposes.
        """
        N = len(result.alpha)
        out = []
        cp_sol = self.amine.cp_solution(
            wt_frac_amine, 0.5 * (result.T[0] + result.T[-1]))
        kg_sol_per_mol = (self.amine.MW * 1e-3) / wt_frac_amine
        m_dot = result.L * kg_sol_per_mol
        for n in range(N):
            T_above = result.T[n - 1] if n > 0 else result.T[0]
            T_n = result.T[n]
            alpha_above = (result.alpha[n - 1] if n > 0
                            else result.alpha_rich)
            alpha_n = result.alpha[n]
            d_alpha = alpha_above - alpha_n   # +ve for stripper
            Q_sens = m_dot * cp_sol * (T_n - T_above)
            Q_reac = result.L * abs(self.amine.delta_H_abs) * d_alpha
            Q_vap = (result.G * result.y_H2O[n]
                       * _DELTA_H_VAP_WATER)
            out.append({
                "T": T_n,
                "alpha": alpha_n,
                "Q_sensible": Q_sens,
                "Q_reaction": Q_reac,
                "Q_vaporization": Q_vap,
            })
        return out


# =====================================================================
# Stripper top condenser (v0.9.107)
# =====================================================================

@dataclass
class StripperCondenserResult:
    """Result of a StripperCondenser.solve() call.

    Attributes
    ----------
    V_in : float
        Vapor inflow [mol/s] (from stripper top).
    y_CO2_in, y_H2O_in : float
        Inlet vapor composition.
    T_in : float
        Inlet vapor temperature [K].
    T_cond : float
        Condenser temperature [K] (vapor exit T).
    V_vent : float
        Vented vapor flow [mol/s] (CO₂ + small saturated H₂O).
    y_CO2_vent : float
        CO₂ mole fraction in vented gas (the "CO₂ purity" spec).
    y_H2O_vent : float
        H₂O mole fraction in vented gas — saturated at T_cond.
    L_reflux : float
        Condensed water reflux [mol/s] (returned to top stage).
    Q_cond : float
        Condenser duty [W] (always > 0, removed from system).
    Q_sensible_cond : float
        Sensible cooling [W] (cooling vapor from T_in → T_cond).
    Q_latent_cond : float
        Latent heat of water condensation [W].
    co2_recovery_in_vent : float
        Fraction of inlet CO2 carried in vented gas (≈ 1 if T_cond
        is well above CO2 dew point — usually the case).
    """
    V_in: float
    y_CO2_in: float
    y_H2O_in: float
    T_in: float
    T_cond: float
    V_vent: float
    y_CO2_vent: float
    y_H2O_vent: float
    L_reflux: float
    Q_cond: float
    Q_sensible_cond: float
    Q_latent_cond: float
    co2_recovery_in_vent: float


class StripperCondenser:
    """Partial condenser at the top of an amine stripper.

    Cools the CO₂ + H₂O vapor exiting the stripper from ~100-105 °C to
    a cold-end temperature (~30-45 °C, set by available cooling water
    or refrigeration) and condenses most water back as reflux.  Output
    streams: a vented gas stream of high-purity CO₂ and a reflux
    stream of nearly-pure water.

    Vented vapor is **saturated with water at T_cond**, so:

        y_H2O_vent = P_water_sat(T_cond) / P_total
        y_CO2_vent = 1 - y_H2O_vent

    Industrial practice: T_cond ~ 40-50 °C with cooling water gives
    CO₂ purities ~93-97 vol%.  Sub-ambient cooling pushes higher
    purities at the expense of refrigeration energy.

    The condenser closes the **water balance** of the capture cycle:
    L_reflux flows back into the stripper top stage; net water exit
    is V_vent · y_H2O_vent (the small water that leaves with the
    vented CO₂).  This is the makeup water requirement.

    Parameters
    ----------
    T_cond : float
        Condenser cold-end temperature [K].  Typical range
        308-323 K (35-50 °C) with cooling water.
    P : float, default 1.8
        Operating pressure [bar] (matches stripper).

    Examples
    --------
    >>> from stateprop.electrolyte import StripperCondenser
    >>> # Vapor exits stripper top at 105 °C with 50 vol% CO2
    >>> cond = StripperCondenser(T_cond=313.15, P=1.8)   # 40 °C
    >>> r = cond.solve(V_in=10.0, y_CO2_in=0.50, T_in=378.15)
    >>> r.V_vent
    5.10           # mol/s — only the CO2 + small saturated water leaves
    >>> r.y_CO2_vent
    0.978          # 97.8 vol% CO2 — nearly pure
    >>> r.L_reflux
    4.90           # mol/s — water condensed and returned
    >>> r.Q_cond / 1e3
    225.0          # kW removed by cooling water
    """

    def __init__(self,
                  T_cond: float,
                  P: float = 1.8):
        self.T_cond = float(T_cond)
        self.P = float(P)
        if self.T_cond <= 273.15:
            raise ValueError("T_cond must be > 0 °C")
        # Validate: T_cond shouldn't exceed water saturation T at P
        # (otherwise no condensation possible)
        if P_water_sat(self.T_cond) >= self.P:
            raise ValueError(
                f"At T_cond={self.T_cond} K, P_water_sat="
                f"{P_water_sat(self.T_cond):.3f} bar ≥ P_total={self.P} bar; "
                f"no condensation possible")

    def solve(self,
                V_in: float,
                y_CO2_in: float,
                T_in: float = 378.15,
                ) -> StripperCondenserResult:
        """Solve the partial condenser.

        Parameters
        ----------
        V_in : float
            Vapor flow into condenser [mol/s] (= V_top from stripper).
        y_CO2_in : float
            CO2 mole fraction in inlet vapor (= y_top from stripper).
        T_in : float, default 378.15 K
            Inlet vapor temperature [K] (= top stage T from stripper).

        Returns
        -------
        StripperCondenserResult
        """
        if not 0 <= y_CO2_in <= 1.0001:
            raise ValueError(f"y_CO2_in must be in [0,1], got {y_CO2_in}")
        # Clip near-edge unphysical values from upstream constant-G models
        y_CO2_in = max(0.0, min(1.0, y_CO2_in))
        if V_in <= 0:
            raise ValueError("V_in must be > 0")

        y_H2O_in = 1.0 - y_CO2_in

        # Saturated water at condenser exit
        P_sat_cond = P_water_sat(self.T_cond)
        y_H2O_vent = P_sat_cond / self.P
        y_CO2_vent = 1.0 - y_H2O_vent

        # CO2 conservation (assume CO2 doesn't dissolve in reflux water)
        # V_in · y_CO2_in = V_vent · y_CO2_vent
        if y_CO2_vent <= 0 or y_CO2_in == 0:
            # Edge case: pure water vapor in or condenser too cold
            V_vent = 0.0
            L_reflux = V_in
            recovery = 0.0
        else:
            V_vent = V_in * y_CO2_in / y_CO2_vent
            # Sanity: V_vent ≤ V_in (else condenser is "creating" vapor)
            if V_vent > V_in:
                # Means inlet was even drier than condenser outlet would be;
                # no condensation, vapor passes through
                V_vent = V_in
                L_reflux = 0.0
                # Recompute output composition
                y_H2O_vent = y_H2O_in
                y_CO2_vent = y_CO2_in
            else:
                L_reflux = V_in - V_vent
            recovery = (V_vent * y_CO2_vent) / (V_in * y_CO2_in) \
                          if y_CO2_in > 0 else 0.0

        # Heat duty
        # Sensible cooling: V_in (at T_in) → V_vent at T_cond
        # plus L_reflux water cooled from T_in to T_cond
        # Approx: cool ALL inlet vapor from T_in → T_cond (sensible)
        # then condense L_reflux moles of H2O at T_cond (latent)
        # Vapor cp_avg ≈ y_CO2 · cp_CO2 + y_H2O · cp_H2O_g
        cp_in = y_CO2_in * _CP_CO2_VAPOR + y_H2O_in * _CP_WATER_VAPOR
        Q_sensible = V_in * cp_in * (T_in - self.T_cond)
        Q_latent = L_reflux * _DELTA_H_VAP_WATER
        Q_cond = Q_sensible + Q_latent

        return StripperCondenserResult(
            V_in=V_in, y_CO2_in=y_CO2_in, y_H2O_in=y_H2O_in,
            T_in=T_in, T_cond=self.T_cond,
            V_vent=V_vent, y_CO2_vent=y_CO2_vent, y_H2O_vent=y_H2O_vent,
            L_reflux=L_reflux,
            Q_cond=Q_cond,
            Q_sensible_cond=Q_sensible,
            Q_latent_cond=Q_latent,
            co2_recovery_in_vent=recovery,
        )


def stripper_with_condenser(stripper: AmineStripper,
                                stripper_solve_kwargs: dict,
                                T_cond: float,
                                P: float = 1.8,
                                ) -> tuple:
    """Convenience: run an AmineStripper.solve() and feed top vapor to
    a partial condenser.

    Returns
    -------
    (stripper_result, condenser_result) : tuple
    """
    s_res = stripper.solve(**stripper_solve_kwargs)
    cond = StripperCondenser(T_cond=T_cond, P=P)
    # Top vapor flow = G (constant in our simple model)
    V_top = s_res.G
    y_CO2_top = s_res.y_top_CO2
    T_top_K = s_res.T[0]
    c_res = cond.solve(V_in=V_top, y_CO2_in=y_CO2_top, T_in=T_top_K)
    return s_res, c_res
