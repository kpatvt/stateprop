"""Full amine CO2 capture flowsheet integrator (v0.9.108, rigorous mode v0.9.115).

Ties the v0.9.104-v0.9.107 unit operations into a complete plant model
with stream tearing on the lean amine recycle:

    flue gas + lean amine
         ↓
    [ABSORBER]  ────── cleaned gas →
         ↓
    rich amine
         ↓
    [HX cold side: rich heated]  ←── hot lean
         ↓
    rich (preheated)
         ↓
    [STRIPPER]  ─── top vapor → [CONDENSER] ── vent CO2
         ↓                           ↓
    lean amine                       reflux water
         ↓
    [HX hot side: lean cooled]
         ↓
    lean (warm) ─── [LEAN COOLER trim] ──→ lean (cold)
                                               ↓
                                         back to ABSORBER (recycle)

The recycle is closed by iterating on the tear stream (α_lean, T_lean
entering the absorber) until convergence.  Direct substitution with
damping is used; for typical industrial designs (L/G ~ 0.5-1.5,
α_lean ~ 0.15-0.30), convergence in 5-15 outer iterations.

The flowsheet output is the canonical industrial summary:
    * α_lean, α_rich, CO2 recovery
    * Reboiler duty Q_reb [W]
    * Condenser duty Q_cond [W]
    * Lean cooler duty Q_lean [W]
    * Heat exchanger duty Q_HX [W]
    * Per-ton CO2 metric Q_per_ton [GJ/ton]
    * Makeup water requirement [mol/s]
    * Vented CO2 flow + purity

v0.9.115 — Rigorous solver mode and tray sizing
-----------------------------------------------
The default ``solver="bespoke"`` keeps the v0.9.104 / v0.9.105 inner
solvers (α-Newton).  Pass ``solver="ns"`` to use the v0.9.114
:func:`amine_absorber_ns` and :func:`amine_stripper_ns` instead —
the same Naphtali-Sandholm engine from the distillation package, with
proper bubble-point per stage, water mass transfer, energy balance,
and (optionally) Murphree efficiency.

In N-S mode, ``size_trays=True`` triggers the v0.9.113
:func:`size_tray_diameter` on both columns, populating the result's
``absorber_diameter`` and ``stripper_diameter`` fields with the
minimum tower diameters meeting ``target_flood_frac`` (default 0.75).
This delivers the "Q/ton + tower hardware" pair that a process
engineer needs for a complete capacity-and-economics design pass.

References
----------
* Cousins, A., Wardhaugh, L. T., Feron, P. H. M. (2011).  A survey
  of process flow sheet modifications for energy-efficient post-
  combustion capture of CO2 using chemical absorption.  IJGGC 5, 605.
* Notz, R., Mangalapally, H. P., Hasse, H. (2012).  Post-combustion
  CO2 capture by reactive absorption: pilot plant description and
  results of systematic studies with MEA.  IJGGC 6, 84.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any
import numpy as np

from .amines import Amine, lookup_amine
from .amine_column import AmineColumn, AmineColumnResult
from .amine_stripper import (
    AmineStripper, AmineStripperResult,
    StripperCondenser, StripperCondenserResult,
)
from .amine_column_ns import (
    amine_absorber_ns, amine_stripper_ns, AmineNSResult,
)
from .heat_exchanger import lean_rich_exchanger, CrossExchangerResult


_DELTA_H_VAP_WATER = 40700.0
_M_CO2 = 0.04401


# =====================================================================
# Result
# =====================================================================

@dataclass
class CaptureFlowsheetResult:
    """Complete result of a CaptureFlowsheet.solve() call.

    The fields cover both the unit-operation sub-results (for detail)
    and the plant-level summary (for headline reporting).

    Attributes
    ----------
    converged : bool
    iterations : int
    alpha_lean_history : list of float
        Sequence of α_lean values across outer iterations (last is
        converged value).

    --- Stream states ---
    alpha_lean : float       Loaded amine returning to absorber.
    alpha_rich : float       Loaded amine leaving absorber.
    T_lean_to_absorber : float    [K]
    T_rich_from_absorber : float  [K]
    T_rich_to_stripper : float    [K]  (after HX preheat)
    T_lean_from_stripper : float  [K]
    T_lean_after_HX : float       [K]  (before trim cooler)

    --- Mass flows ---
    L_amine : float          [mol amine / s]
    G_flue : float           [mol total / s]
    G_strip_steam : float    [mol steam / s]
    co2_captured : float     [mol CO2 / s]
    co2_recovery : float     [-]

    --- Vent ---
    V_vent : float           [mol / s]
    y_CO2_vent : float       [-] (CO2 purity in vent gas)
    L_reflux : float         [mol H2O / s] returned to stripper top
    L_makeup_water : float   [mol H2O / s] needed to replace vented water

    --- Energy ---
    Q_reboiler : float       [W] (input)
    Q_condenser : float      [W] (output to cooling water)
    Q_lean_cooler : float    [W] (output to cooling water)
    Q_HX : float             [W] (internal, lean→rich)
    Q_per_ton_CO2 : float    [GJ / ton CO2 captured]

    --- Sub-results ---
    absorber_result : AmineColumnResult or AmineNSResult
    HX_result : CrossExchangerResult
    stripper_result : AmineStripperResult or AmineNSResult
    condenser_result : Optional[StripperCondenserResult]
        ``None`` in N-S mode where the condenser is integrated into
        the column.

    --- Solver / tray-sizing (v0.9.115) ---
    solver : str
        ``"bespoke"`` or ``"ns"``.
    absorber_diameter : Optional[float]
        Minimum tower diameter [m] meeting ``target_flood_frac``.
        Populated only when ``solver="ns"`` and ``size_trays=True``.
    stripper_diameter : Optional[float]
        Same, for the stripper.
    absorber_hydraulics : Optional[TrayHydraulicsResult]
        Per-stage flooding %, weir crest, ΔP, etc.  N-S + size_trays.
    stripper_hydraulics : Optional[TrayHydraulicsResult]
    """
    converged: bool
    iterations: int
    alpha_lean_history: List[float]
    # Streams
    alpha_lean: float
    alpha_rich: float
    T_lean_to_absorber: float
    T_rich_from_absorber: float
    T_rich_to_stripper: float
    T_lean_from_stripper: float
    T_lean_after_HX: float
    # Flows
    L_amine: float
    G_flue: float
    G_strip_steam: float
    co2_captured: float
    co2_recovery: float
    # Vent
    V_vent: float
    y_CO2_vent: float
    L_reflux: float
    L_makeup_water: float
    # Energy
    Q_reboiler: float
    Q_condenser: float
    Q_lean_cooler: float
    Q_HX: float
    Q_per_ton_CO2: float
    # Sub-results (for detail inspection)
    absorber_result: Any
    HX_result: CrossExchangerResult
    stripper_result: Any
    condenser_result: Optional[StripperCondenserResult]
    # Solver and hydraulics (v0.9.115)
    solver: str = "bespoke"
    absorber_diameter: Optional[float] = None
    stripper_diameter: Optional[float] = None
    absorber_hydraulics: Optional[Any] = None
    stripper_hydraulics: Optional[Any] = None

    def summary(self) -> str:
        """Format a human-readable plant-level summary."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"CAPTURE FLOWSHEET SUMMARY  ({'converged' if self.converged else 'NOT CONVERGED'} in {self.iterations} iter)")
        lines.append("=" * 70)
        lines.append(f"  Loadings:    α_lean={self.alpha_lean:.3f}  α_rich={self.alpha_rich:.3f}")
        lines.append(f"  CO2 capture: {self.co2_recovery*100:.1f}% "
                      f"({self.co2_captured*_M_CO2*3600:.1f} kg/h)")
        lines.append(f"  CO2 vent purity: {self.y_CO2_vent*100:.1f} vol%")
        lines.append("")
        lines.append(f"  Operating temperatures:")
        lines.append(f"    Lean to absorber:    {self.T_lean_to_absorber-273.15:.1f} °C")
        lines.append(f"    Rich from absorber:  {self.T_rich_from_absorber-273.15:.1f} °C")
        lines.append(f"    Rich to stripper:    {self.T_rich_to_stripper-273.15:.1f} °C  (HX preheat)")
        lines.append(f"    Lean from stripper:  {self.T_lean_from_stripper-273.15:.1f} °C")
        lines.append(f"    Lean after HX:       {self.T_lean_after_HX-273.15:.1f} °C")
        lines.append("")
        lines.append(f"  Energy duties [MW]:")
        lines.append(f"    Reboiler (input):    {self.Q_reboiler/1e6:>+7.3f}")
        lines.append(f"    HX  (lean→rich):     {self.Q_HX/1e6:>+7.3f}  (recovered)")
        lines.append(f"    Condenser (output):  {-self.Q_condenser/1e6:>+7.3f}")
        lines.append(f"    Lean cooler (output):{-self.Q_lean_cooler/1e6:>+7.3f}")
        lines.append(f"")
        lines.append(f"  Q per ton CO2:  {self.Q_per_ton_CO2:.2f}  GJ/ton  (industry 3.5-4)")
        lines.append("")
        lines.append(f"  Water balance:  makeup = {self.L_makeup_water*0.018*3600:.1f} kg/h "
                      f"(reflux = {self.L_reflux*0.018*3600:.1f} kg/h)")
        if self.absorber_diameter is not None or self.stripper_diameter is not None:
            lines.append("")
            lines.append(f"  Tower hardware (sized for {self.solver} solver):")
            if self.absorber_diameter is not None:
                ahyd = self.absorber_hydraulics
                lines.append(
                    f"    Absorber: D = {self.absorber_diameter:.2f} m, "
                    f"max %flood = "
                    f"{ahyd.max_pct_flood:.1f}%" if ahyd else "")
            if self.stripper_diameter is not None:
                shyd = self.stripper_hydraulics
                lines.append(
                    f"    Stripper: D = {self.stripper_diameter:.2f} m, "
                    f"max %flood = "
                    f"{shyd.max_pct_flood:.1f}%" if shyd else "")
        lines.append("=" * 70)
        return "\n".join(lines)


# =====================================================================
# CaptureFlowsheet class
# =====================================================================

class CaptureFlowsheet:
    """Complete amine CO2 capture flowsheet integrator.

    Combines absorber, lean-rich heat exchanger, stripper, top
    condenser, and lean trim cooler into a recycle loop that is solved
    by tearing the lean-amine stream and iterating to convergence.

    Parameters
    ----------
    amine : Amine or str
        The alkanolamine.
    total_amine : float
        Liquid amine concentration [mol/kg solvent].
    n_stages_absorber : int, default 20
    n_stages_stripper : int, default 15

    Examples
    --------
    >>> fs = CaptureFlowsheet("MEA", 5.0)
    >>> r = fs.solve(
    ...     G_flue=15.0, y_in_CO2=0.12,    # 12% CO2 inlet (post-comb)
    ...     L_amine=8.0,                    # 30 wt% MEA, 8 mol/s
    ...     T_absorber_feed=313.15,         # 40 °C lean feed
    ...     P_absorber=1.013,
    ...     G_strip_steam=8.0,              # stripping steam from reboiler
    ...     T_strip_top=378.15, T_strip_bottom=393.15,
    ...     P_stripper=1.8,
    ...     T_cond=313.15,                  # 40 °C cooling water
    ...     delta_T_min_HX=5.0,
    ...     wt_frac_amine=0.30,
    ... )
    >>> print(r.summary())
    """

    def __init__(self,
                  amine: Union[Amine, str],
                  total_amine: float,
                  n_stages_absorber: int = 20,
                  n_stages_stripper: int = 15):
        self.amine = (lookup_amine(amine) if isinstance(amine, str)
                       else amine)
        self.total_amine = float(total_amine)
        self._absorber = AmineColumn(self.amine, self.total_amine,
                                          n_stages_absorber)
        self._stripper = AmineStripper(self.amine, self.total_amine,
                                            n_stages_stripper)

    # -----------------------------------------------------------------
    def solve(self,
                G_flue: float,
                y_in_CO2: float,
                L_amine: float,
                T_absorber_feed: float = 313.15,
                P_absorber: float = 1.013,
                G_strip_steam: float = None,
                y_reb: float = 0.01,
                T_strip_top: float = 378.15,
                T_strip_bottom: float = 393.15,
                P_stripper: float = 1.8,
                T_cond: float = 313.15,
                delta_T_min_HX: float = 5.0,
                wt_frac_amine: float = 0.30,
                alpha_lean_init: float = 0.20,
                damp: float = 0.6,
                max_outer: int = 30,
                tol: float = 5e-4,
                adiabatic_absorber: bool = False,
                T_gas_in: Optional[float] = None,
                variable_V_stripper=False,
                # v0.9.115: rigorous solver + tray sizing
                solver: str = "bespoke",
                stage_efficiency: float = 1.0,
                size_trays: bool = False,
                target_flood_frac: float = 0.75,
                tray_spacing: float = 0.6,
                weir_height: float = 0.05,
                verbose: bool = False) -> CaptureFlowsheetResult:
        """Solve the closed-loop capture flowsheet.

        Parameters
        ----------
        G_flue : float
            Flue gas molar flow [mol/s] entering absorber bottom.
        y_in_CO2 : float
            CO2 mole fraction in flue gas (typical 0.04 = 4 % for
            natural gas combustion, 0.12-0.15 for coal).
        L_amine : float
            Liquid amine molar flow [mol amine / s].
        T_absorber_feed : float, default 313.15 (40 °C)
            Target lean amine T entering absorber.  The trim cooler
            absorbs whatever residual heat is needed to hit this.
        P_absorber : float, default 1.013 bar
        G_strip_steam : float, default = L_amine
            Stripping steam flow at reboiler [mol/s].
        y_reb : float, default 0.05
            CO2 mole fraction in reboiler vapor.
        T_strip_top, T_strip_bottom : float
            Stripper top/bottom temperatures [K].  Default 105/120 °C.
        P_stripper : float, default 1.8 bar
        T_cond : float, default 313.15 (40 °C)
            Stripper top condenser cold-end T [K].
        delta_T_min_HX : float, default 5.0 K
            Lean-rich HX minimum approach.
        wt_frac_amine : float, default 0.30
        alpha_lean_init : float, default 0.20
            Initial guess for tear-stream loading.
        damp : float, default 0.6
            Damping factor for direct-substitution updates.
        max_outer : int, default 30
        tol : float, default 5e-4
            Convergence tolerance on |Δα_lean|.
        adiabatic_absorber : bool, default False
            If True, run absorber in adiabatic mode (T_n unknown,
            energy balance per stage).  Captures the temperature
            bulge; rich liquid leaves at peak stage T (typically
            +10-20 K above feed).  Recovery is lower than isothermal
            but more realistic.
        T_gas_in : float, optional
            Gas feed T [K] for adiabatic absorber mode (defaults to
            T_absorber_feed).
        solver : str, "bespoke" (default) or "ns"
            Inner unit-op solver.  ``"bespoke"`` uses the v0.9.104
            ``AmineColumn`` and v0.9.105 ``AmineStripper`` α-Newton
            solvers.  ``"ns"`` uses the v0.9.114
            :func:`amine_absorber_ns` and :func:`amine_stripper_ns`
            built on the rigorous Naphtali-Sandholm distillation
            engine.  N-S resolves bubble-point per stage, water
            mass transfer, full multi-species balance with inert
            (N₂) carrier, and an integrated total condenser.
        stage_efficiency : float, default 1.0
            Murphree efficiency applied uniformly to both columns
            in N-S mode.  Industrial absorbers run 0.6-0.8.
            Ignored in bespoke mode.
        size_trays : bool, default False
            If True (and solver="ns"), run :func:`size_tray_diameter`
            on both columns after convergence to determine the
            minimum diameter for the requested ``target_flood_frac``.
            Populates ``absorber_diameter``, ``stripper_diameter``,
            and the per-stage hydraulics.
        target_flood_frac : float, default 0.75
            Target maximum vapor flooding fraction (75 % is a
            conventional design point, leaving 25 % capacity margin
            against turn-up).
        tray_spacing : float, default 0.6 m
            Sieve-tray spacing used for sizing.
        weir_height : float, default 0.05 m
        verbose : bool, default False
        """
        if G_strip_steam is None:
            G_strip_steam = L_amine
        if T_gas_in is None:
            T_gas_in = T_absorber_feed
        if solver not in ("bespoke", "ns"):
            raise ValueError(
                f"solver must be 'bespoke' or 'ns'; got {solver!r}")
        if size_trays and solver != "ns":
            raise ValueError(
                "size_trays=True requires solver='ns' "
                "(bespoke solvers do not expose stage profiles)")

        alpha_lean = float(alpha_lean_init)
        alpha_lean_history = [alpha_lean]

        # Solvent thermal mass for trim cooler computation
        cp_sol = self.amine.cp_solution(wt_frac_amine, T_absorber_feed)
        kg_sol_per_mol = (self.amine.MW * 1e-3) / wt_frac_amine

        converged = False
        cond_res: Optional[StripperCondenserResult] = None

        for outer in range(max_outer):

            if solver == "ns":
                # ---- N-S absorber ----
                abs_res = amine_absorber_ns(
                    amine_name=self.amine.name,
                    total_amine=self.total_amine,
                    L=L_amine, G=G_flue,
                    alpha_lean=alpha_lean, y_in_CO2=y_in_CO2,
                    wt_frac_amine=wt_frac_amine,
                    n_stages=self._absorber.n_stages,
                    T_liquid_in=T_absorber_feed,
                    T_gas_in=T_gas_in,
                    P=P_absorber * 1e5,
                    energy_balance=adiabatic_absorber,
                    stage_efficiency=stage_efficiency,
                )
                alpha_rich = abs_res.alpha_rich
                # Rich amine leaves at the bottom-stage T (whatever
                # the bubble-point / energy balance gave us).
                T_rich_from_absorber = float(abs_res.column_result.T[-1])
                y_top = float(abs_res.column_result.y[0, 0])

                # ---- HX (same as bespoke path) ----
                hx_res = lean_rich_exchanger(
                    self.amine, self.total_amine,
                    T_lean_in=T_strip_bottom,
                    T_rich_in=T_rich_from_absorber,
                    L_lean=L_amine,
                    delta_T_min=delta_T_min_HX,
                    wt_frac_amine=wt_frac_amine,
                )
                T_rich_to_stripper = hx_res.T_cold_out
                T_lean_after_HX = hx_res.T_hot_out

                # ---- N-S stripper (with integrated condenser) ----
                strip_res = amine_stripper_ns(
                    amine_name=self.amine.name,
                    total_amine=self.total_amine,
                    L=L_amine, G=G_strip_steam,
                    alpha_rich=alpha_rich,
                    wt_frac_amine=wt_frac_amine,
                    n_stages=self._stripper.n_stages,
                    T_top=T_strip_top, T_bottom=T_strip_bottom,
                    P=P_stripper * 1e5,
                    y_reb_CO2=y_reb,
                    energy_balance=True,
                    stage_efficiency=stage_efficiency,
                )
                alpha_lean_new = strip_res.alpha_lean
                Q_reb = strip_res.Q_R or 0.0
                Q_cond = strip_res.Q_C or 0.0
                T_lean_from_stripper = float(strip_res.column_result.T[-1])

                # The N-S column has a built-in total condenser → the
                # distillate D *is* the vent stream.  No separate
                # ``StripperCondenser`` flash needed.
                col_strip = strip_res.column_result
                V_vent = float(col_strip.D)
                # Distillate composition: x_D == y_top from total condenser
                if hasattr(col_strip, "x_D") and col_strip.x_D is not None:
                    y_CO2_vent = float(col_strip.x_D[0])
                    y_H2O_vent = float(col_strip.x_D[1])
                else:
                    y_CO2_vent = float(col_strip.y[0, 0])
                    y_H2O_vent = float(col_strip.y[0, 1])
                L_reflux = float(col_strip.reflux_ratio * V_vent)
                cond_res = None
                # Track approximate "Q_reboiler" semantics
                Q_reboiler_eff = Q_reb
                Q_condenser_eff = Q_cond

            else:
                # ---- Bespoke absorber + stripper + condenser ----
                if adiabatic_absorber:
                    abs_res = self._absorber.solve(
                        L=L_amine, G=G_flue,
                        alpha_lean=alpha_lean, y_in=y_in_CO2,
                        P=P_absorber,
                        adiabatic=True,
                        T_liquid_in=T_absorber_feed,
                        T_gas_in=T_gas_in,
                        wt_frac_amine=wt_frac_amine,
                    )
                    alpha_rich = abs_res.alpha_rich
                    T_rich_from_absorber = abs_res.T[-1]
                else:
                    abs_res = self._absorber.solve(
                        L=L_amine, G=G_flue,
                        alpha_lean=alpha_lean, y_in=y_in_CO2,
                        P=P_absorber, T=T_absorber_feed,
                    )
                    alpha_rich = abs_res.alpha_rich
                    T_rich_from_absorber = T_absorber_feed
                y_top = abs_res.y_top

                # HX
                hx_res = lean_rich_exchanger(
                    self.amine, self.total_amine,
                    T_lean_in=T_strip_bottom,
                    T_rich_in=T_rich_from_absorber,
                    L_lean=L_amine,
                    delta_T_min=delta_T_min_HX,
                    wt_frac_amine=wt_frac_amine,
                )
                T_rich_to_stripper = hx_res.T_cold_out
                T_lean_after_HX = hx_res.T_hot_out

                # Stripper
                strip_res = self._stripper.solve(
                    L=L_amine, G=G_strip_steam,
                    alpha_rich=alpha_rich, y_reb=y_reb,
                    P=P_stripper,
                    T_top=T_strip_top, T_bottom=T_strip_bottom,
                    wt_frac_amine=wt_frac_amine,
                    T_rich_in=T_rich_to_stripper,
                    variable_V=variable_V_stripper,
                )
                alpha_lean_new = strip_res.alpha_lean
                T_lean_from_stripper = T_strip_bottom

                # Condenser (separate flash)
                cond = StripperCondenser(T_cond=T_cond, P=P_stripper)
                y_top_for_cond = min(1.0, max(0.0, strip_res.y_top_CO2))
                cond_res = cond.solve(
                    V_in=G_strip_steam,
                    y_CO2_in=y_top_for_cond,
                    T_in=strip_res.T[0],
                )
                V_vent = cond_res.V_vent
                y_CO2_vent = cond_res.y_CO2_vent
                y_H2O_vent = cond_res.y_H2O_vent
                L_reflux = cond_res.L_reflux
                Q_reboiler_eff = strip_res.Q_reboiler
                Q_condenser_eff = cond_res.Q_cond

            # Convergence check
            err = abs(alpha_lean_new - alpha_lean)
            if verbose:
                print(f"  outer {outer:3d}: α_lean: {alpha_lean:.4f} → "
                       f"{alpha_lean_new:.4f} (Δ={err:.2e})")
            if err < tol:
                converged = True
                alpha_lean = alpha_lean_new
                alpha_lean_history.append(alpha_lean)
                break

            alpha_lean = damp * alpha_lean_new + (1 - damp) * alpha_lean
            alpha_lean_history.append(alpha_lean)

        # 5. Lean trim cooler: cool from T_lean_after_HX to T_absorber_feed
        m_lean = L_amine * kg_sol_per_mol
        Q_lean_cooler = max(0.0,
                                m_lean * cp_sol
                                * (T_lean_after_HX - T_absorber_feed))

        # Plant-level totals
        co2_captured = G_flue * (y_in_CO2 - y_top)
        recovery = co2_captured / (G_flue * y_in_CO2) if y_in_CO2 > 0 else 0.0
        if co2_captured > 1e-12:
            kg_CO2_per_s = co2_captured * _M_CO2
            Q_per_ton = Q_reboiler_eff / kg_CO2_per_s * 1e-6
        else:
            Q_per_ton = float("nan")

        # Water balance
        water_vented = V_vent * y_H2O_vent
        L_makeup_water = water_vented

        # ------------------------------------------------------------
        # 6. Tray sizing (v0.9.115; only N-S mode)
        # ------------------------------------------------------------
        absorber_diameter: Optional[float] = None
        stripper_diameter: Optional[float] = None
        absorber_hyd = None
        stripper_hyd = None
        if size_trays and solver == "ns":
            from ..distillation import (
                size_tray_diameter, tray_hydraulics, TrayDesign,
            )
            # Absorber: 4 species [CO2, H2O, amine, N2]
            sp_abs = ["CO2", "H2O", self.amine.name, "N2"]
            col_abs = abs_res.column_result
            absorber_diameter = size_tray_diameter(
                V_profile=col_abs.V, L_profile=col_abs.L,
                T_profile=col_abs.T,
                x_profile=col_abs.x, y_profile=col_abs.y,
                P=P_absorber * 1e5, species_names=sp_abs,
                spacing=tray_spacing, weir_height=weir_height,
                target_flood_frac=target_flood_frac,
            )
            absorber_hyd = tray_hydraulics(
                V_profile=col_abs.V, L_profile=col_abs.L,
                T_profile=col_abs.T,
                x_profile=col_abs.x, y_profile=col_abs.y,
                P=P_absorber * 1e5, species_names=sp_abs,
                tray_design=TrayDesign(
                    diameter=absorber_diameter,
                    spacing=tray_spacing, weir_height=weir_height),
            )
            # Stripper: 3 species [CO2, H2O, amine]
            sp_strip = ["CO2", "H2O", self.amine.name]
            col_str = strip_res.column_result
            stripper_diameter = size_tray_diameter(
                V_profile=col_str.V, L_profile=col_str.L,
                T_profile=col_str.T,
                x_profile=col_str.x, y_profile=col_str.y,
                P=P_stripper * 1e5, species_names=sp_strip,
                spacing=tray_spacing, weir_height=weir_height,
                target_flood_frac=target_flood_frac,
            )
            stripper_hyd = tray_hydraulics(
                V_profile=col_str.V, L_profile=col_str.L,
                T_profile=col_str.T,
                x_profile=col_str.x, y_profile=col_str.y,
                P=P_stripper * 1e5, species_names=sp_strip,
                tray_design=TrayDesign(
                    diameter=stripper_diameter,
                    spacing=tray_spacing, weir_height=weir_height),
            )

        return CaptureFlowsheetResult(
            converged=converged, iterations=outer + 1,
            alpha_lean_history=alpha_lean_history,
            alpha_lean=alpha_lean, alpha_rich=alpha_rich,
            T_lean_to_absorber=T_absorber_feed,
            T_rich_from_absorber=T_rich_from_absorber,
            T_rich_to_stripper=T_rich_to_stripper,
            T_lean_from_stripper=T_lean_from_stripper,
            T_lean_after_HX=T_lean_after_HX,
            L_amine=L_amine, G_flue=G_flue, G_strip_steam=G_strip_steam,
            co2_captured=co2_captured, co2_recovery=recovery,
            V_vent=V_vent, y_CO2_vent=y_CO2_vent,
            L_reflux=L_reflux, L_makeup_water=L_makeup_water,
            Q_reboiler=Q_reboiler_eff,
            Q_condenser=Q_condenser_eff,
            Q_lean_cooler=Q_lean_cooler,
            Q_HX=hx_res.Q,
            Q_per_ton_CO2=Q_per_ton,
            absorber_result=abs_res,
            HX_result=hx_res,
            stripper_result=strip_res,
            condenser_result=cond_res,
            solver=solver,
            absorber_diameter=absorber_diameter,
            stripper_diameter=stripper_diameter,
            absorber_hydraulics=absorber_hyd,
            stripper_hydraulics=stripper_hyd,
        )
