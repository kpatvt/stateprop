"""Amine absorber and stripper via Naphtali-Sandholm (v0.9.114).

This module wires the v0.9.103 :mod:`amines` carbamate-equilibrium
chemistry into the v0.9.108 :func:`distillation_column` Naphtali-Sandholm
solver, replacing the bespoke :class:`AmineColumn` and
:class:`AmineStripper` Newton solvers with the rigorous N-S engine.

The mapping
-----------
For a 3-species [CO2, H2O, amine] system, the modified Raoult K-value
for CO2 must reproduce the amine equilibrium partial pressure at each
stage:

    K_CO2 · x_CO2 = y_CO2 = P_CO2_eq(α, T) / P_total

where α = x_CO2 / x_amine is the CO2 loading.  Setting

    P_sat_CO2_pseudo(T) = 1 bar = 1e5 Pa     (constant)

we get

    γ_CO2 = K_CO2 · P_total / P_sat_CO2_pseudo
          = (y_CO2 · P) / (x_CO2 · P_sat_CO2_pseudo)
          = P_CO2_eq(α, T) [bar] / (x_CO2)         (when P_sat = 1 bar)

i.e. γ_CO2 carries all the amine-equilibrium non-ideality.  For water,
γ_H2O = 1 with the standard Antoine P_sat.  For the amine itself we
use a very small constant P_sat (treating it as essentially
non-volatile under typical absorber/stripper conditions).

Why use N-S?
------------
The bespoke :class:`AmineColumn` solver works well, but the N-S engine
brings a number of features that come "for free":

* **Energy balance** with proper enthalpy callables — the v0.9.106
  adiabatic absorber + lean-rich HX previously used a side-mode
  ``adiabatic=True`` flag in AmineColumn; with N-S, energy balance is
  the standard mode.
* **Murphree efficiency** at the column level (real columns run
  60-80 % efficiency).
* **Multiple feeds, side draws, side strippers** — the architectural
  pieces a real plant flowsheet needs.
* **Tray-hydraulics integration** — V_profile / L_profile / x / y are
  the standard outputs from N-S, ready to feed
  :func:`tray_hydraulics`.

Limitations
-----------
* The activity model performs a full speciation each query → slower
  per Newton step than the bespoke α-Newton.  Typical 5-10× per
  column solve, but the rigorous engine more than makes up for it
  on subsequent flowsheet integration.
* Absorber configuration: N-S assumes a top condenser + bottom
  reboiler.  For a pure absorber (no reboiler/condenser duty) we
  set ``reflux_ratio=0``, partial-condenser, and rely on mass
  balance to drive the reboiler vaporisation to ≈ 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .amines import AmineSystem, lookup_amine
from .amine_stripper import P_water_sat as _P_water_sat_bar


# Convention: pseudo P_sat for CO2 at all T = 1 bar.  This makes
# γ_CO2 numerically equal to P_CO2_eq[bar] / x_CO2.
_P_SAT_CO2_PSEUDO_PA: float = 1.0e5

# Treat amine as effectively non-volatile via tiny P_sat
_P_SAT_AMINE_PA: float = 1.0    # 1 Pa — y_amine << 1e-5 at any P

# Treat non-condensable gases (N2, O2, etc.) via HUGE P_sat → x ≈ 0
_P_SAT_INERT_GAS_PA: float = 1.0e10    # very large → K huge → x tiny in liquid

# Universal molality conversion (mol H2O / kg water solvent)
_M_H2O_PER_KG: float = 1000.0 / 18.0153


# =====================================================================
# Activity model: amine carbamate chemistry → N-S K-values
# =====================================================================

class AmineActivityModel:
    """Activity-coefficient model for the [CO2, H2O, amine] system.

    Each call to :meth:`gammas` performs an amine speciation at the
    current (α, T) and returns γ values that make the N-S column's
    modified-Raoult K_i reproduce the amine equilibrium partial
    pressure of CO2.

    Parameters
    ----------
    amine_system : AmineSystem
        Pre-instantiated amine equilibrium system.
    species_names : sequence of str
        Ordered species names.  Must contain "CO2", "H2O", and the
        amine's name (e.g. "MEA", "MDEA", "DEA", "PZ", "AMP").
    P_sat_CO2_pseudo : float, default 1e5 Pa = 1 bar
        Constant pseudo P_sat for CO2.  Pairs with P_sat_funcs.

    Notes
    -----
    The 3-species model assumes the amine is fully present as the
    "amine" lump — molecular MEA + protonated MEAH⁺ + carbamate
    MEA-COO⁻ are all rolled together as "amine" in the column's
    mole-fraction accounting.  α = x_CO2 / x_amine then matches
    the carbamate-equilibrium loading variable used by AmineSystem.
    """

    def __init__(self,
                  amine_system: AmineSystem,
                  species_names: Sequence[str],
                  P_sat_CO2_pseudo: float = _P_SAT_CO2_PSEUDO_PA):
        self.system = amine_system
        self.species_names = list(species_names)
        self.P_sat_CO2 = float(P_sat_CO2_pseudo)
        if "CO2" not in self.species_names:
            raise ValueError(
                "species_names must include 'CO2'")
        if "H2O" not in self.species_names:
            raise ValueError(
                "species_names must include 'H2O'")
        amine_name = amine_system.amine.name
        if amine_name not in self.species_names:
            raise ValueError(
                f"species_names must include the amine name "
                f"{amine_name!r}; got {self.species_names}")
        self._idx_CO2 = self.species_names.index("CO2")
        self._idx_H2O = self.species_names.index("H2O")
        self._idx_amine = self.species_names.index(amine_name)
        self._amine_name = amine_name

    # -----------------------------------------------------------------
    def loading(self, x: Sequence[float]) -> float:
        """Compute α = x_CO2 / x_amine at the given liquid composition."""
        x = np.asarray(x, dtype=float)
        x_CO2 = max(float(x[self._idx_CO2]), 1e-12)
        x_amine = max(float(x[self._idx_amine]), 1e-12)
        return min(max(x_CO2 / x_amine, 0.0), 0.95)

    # -----------------------------------------------------------------
    def equilibrium_P_CO2(self, alpha: float, T: float) -> float:
        """P_CO2 in Pa from amine carbamate equilibrium at (α, T)."""
        if alpha < 1e-9:
            return 0.0
        try:
            res = self.system.speciate(alpha=alpha, T=T)
            return float(res.P_CO2) * 1.0e5    # bar → Pa
        except Exception:
            return 1.0e10                        # blow-up signal

    # -----------------------------------------------------------------
    def gammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Activity coefficients γ_i for [CO2, H2O, amine].

        γ_CO2 = P_CO2_eq(α, T) / (x_CO2 · P_sat_CO2_pseudo)
        γ_H2O = 1   (water at saturation, standard Antoine)
        γ_amine = 1 (paired with very low P_sat, so K_amine ~ 0)
        """
        x = np.asarray(x, dtype=float)
        gammas = np.ones(len(self.species_names))

        x_CO2 = max(float(x[self._idx_CO2]), 1e-12)
        alpha = self.loading(x)

        P_CO2_eq = self.equilibrium_P_CO2(alpha, T)
        gamma_CO2 = P_CO2_eq / (x_CO2 * self.P_sat_CO2)
        # Bound γ_CO2 to a sensible range; very large values are a
        # numerical signal that α is outside the equilibrium-fit range.
        gamma_CO2 = float(np.clip(gamma_CO2, 1e-6, 1e8))
        gammas[self._idx_CO2] = gamma_CO2
        # γ_H2O = 1 (default), γ_amine = 1 (default)
        return gammas


# =====================================================================
# P_sat builders
# =====================================================================

def _make_psat_CO2_pseudo() -> Callable[[float], float]:
    val = _P_SAT_CO2_PSEUDO_PA
    def psat(T: float) -> float:
        return val
    psat.__name__ = "psat_CO2_pseudo"
    return psat


def _psat_water_Pa(T: float) -> float:
    return _P_water_sat_bar(T) * 1.0e5


def _make_psat_amine() -> Callable[[float], float]:
    val = _P_SAT_AMINE_PA
    def psat(T: float) -> float:
        return val
    psat.__name__ = "psat_amine_nonvolatile"
    return psat


def _make_psat_inert_gas() -> Callable[[float], float]:
    val = _P_SAT_INERT_GAS_PA
    def psat(T: float) -> float:
        return val
    psat.__name__ = "psat_inert_gas"
    return psat


def build_amine_psat_funcs(
        species_names: Sequence[str],
        amine_name: str = "MEA",
) -> List[Callable[[float], float]]:
    """Return a list of P_sat(T) callables for the amine-system column.

    For ``CO2``: constant 1 bar (pseudo-P_sat).
    For ``H2O``: standard water Antoine (Pa).
    For the amine: constant 1 Pa (effectively non-volatile).
    For any other species (N2, O2, Ar, …): constant 1e10 Pa (insoluble
    non-condensable gas — K = 1e10/P >> 1, so x ≈ 0 in liquid).
    """
    funcs: List[Callable[[float], float]] = []
    for sp in species_names:
        if sp == "CO2":
            funcs.append(_make_psat_CO2_pseudo())
        elif sp == "H2O":
            funcs.append(_psat_water_Pa)
        elif sp == amine_name:
            funcs.append(_make_psat_amine())
        else:
            funcs.append(_make_psat_inert_gas())
    return funcs


# =====================================================================
# Enthalpy callables (energy balance)
# =====================================================================

# Reference state: ideal gas at T_ref = 298.15 K
_T_REF: float = 298.15

# Heat of CO2 absorption into amine (exothermic, ~80 kJ/mol)
_DELTA_H_ABS_DEFAULT: float = -85_000.0

# Ideal-gas cp_p [J/(mol·K)]
_CP_V_CO2: float = 37.1
_CP_V_H2O: float = 33.6
_CP_V_AMINE: float = 90.0   # rough cp_p for typical amine vapor (rare in column)

# Liquid cp_p [J/(mol·K)]
_CP_L_H2O: float = 75.3
_CP_L_AMINE: float = 175.0   # MEA cp_l
_CP_L_CO2_AQ: float = 40.0   # CO2(aq) approx

# Water heat of vaporization at T_ref [J/mol]
_DELTA_H_VAP_WATER_REF: float = 43_990.0
_T_C_WATER: float = 647.1


def _delta_H_vap_water(T: float) -> float:
    if T >= _T_C_WATER:
        return 0.0
    return _DELTA_H_VAP_WATER_REF * ((_T_C_WATER - T) / (_T_C_WATER - _T_REF)) ** 0.38


def _h_V_water(T: float) -> float:
    return _delta_H_vap_water(T) + _CP_V_H2O * (T - _T_REF)


def _h_L_water(T: float) -> float:
    return _CP_L_H2O * (T - _T_REF)


def _make_h_V_CO2() -> Callable[[float], float]:
    def f(T: float) -> float:
        return _CP_V_CO2 * (T - _T_REF)
    return f


def _make_h_L_CO2(amine_system: AmineSystem) -> Callable[[float], float]:
    """h_L_CO2 includes the heat-of-absorption of CO2 into amine."""
    dH = float(getattr(amine_system.amine, "delta_H_abs",
                          _DELTA_H_ABS_DEFAULT))
    def f(T: float) -> float:
        return dH + _CP_L_CO2_AQ * (T - _T_REF)
    return f


def _make_h_V_amine() -> Callable[[float], float]:
    def f(T: float) -> float:
        # Treat as ideal gas if it ever appears in vapor
        return _CP_V_AMINE * (T - _T_REF)
    return f


def _make_h_L_amine() -> Callable[[float], float]:
    def f(T: float) -> float:
        return _CP_L_AMINE * (T - _T_REF)
    return f


def build_amine_enthalpy_funcs(
        species_names: Sequence[str],
        amine_system: AmineSystem,
) -> Tuple[List[Callable[[float], float]],
                 List[Callable[[float], float]]]:
    """Return (h_V_funcs, h_L_funcs) for the amine N-S column.

    Reference state: ideal gas at 298.15 K.

    * Water: standard ΔH_vap(T) (Watson-reduced) + cp_p sensible.
    * CO2:   ideal-gas vapor enthalpy in V; absorption-heat-offset
             liquid enthalpy in L.  ΔH_abs is taken from the amine
             attribute :attr:`Amine.delta_H_abs` (negative for
             exothermic).
    * Amine: cp_p sensible (vapor barely populates).
    """
    h_V: List[Callable[[float], float]] = []
    h_L: List[Callable[[float], float]] = []
    for sp in species_names:
        if sp == "CO2":
            h_V.append(_make_h_V_CO2())
            h_L.append(_make_h_L_CO2(amine_system))
        elif sp == "H2O":
            h_V.append(_h_V_water)
            h_L.append(_h_L_water)
        else:
            h_V.append(_make_h_V_amine())
            h_L.append(_make_h_L_amine())
    return h_V, h_L


# =====================================================================
# Result wrappers
# =====================================================================

@dataclass
class AmineNSResult:
    """Result of an amine column solved by N-S.

    Attributes
    ----------
    column_result : DistillationColumnResult
        Raw N-S output (T, x, y, V, L profiles).
    alpha : List[float]
        Per-stage CO2 loading (= x_CO2 / x_amine).
    P_CO2_eq : List[float]
        Per-stage equilibrium CO2 partial pressure [Pa].
    alpha_lean : float
        Loading at the bottom-most liquid stream (lean for stripper,
        rich for absorber).
    alpha_rich : float
        Loading at the top-most liquid stream (rich for stripper,
        lean for absorber).
    co2_recovery : float
        For absorber: fraction of incoming CO2 absorbed.  For
        stripper: fraction of incoming CO2 in rich stream that
        leaves as vapor.
    Q_R, Q_C : Optional[float]
        Reboiler / condenser duty [W] when energy_balance=True.
    """
    column_result: object
    alpha: List[float]
    P_CO2_eq: List[float]
    alpha_lean: float
    alpha_rich: float
    co2_recovery: float
    Q_R: Optional[float] = None
    Q_C: Optional[float] = None


# =====================================================================
# High-level wrappers
# =====================================================================

def amine_stripper_ns(
        amine_name: str,
        total_amine: float,
        L: float,
        G: float,
        alpha_rich: float,
        wt_frac_amine: float = 0.30,
        n_stages: int = 15,
        feed_stage: int = 1,
        T_top: float = 378.15,
        T_bottom: float = 388.15,
        P: float = 1.5e5,
        y_reb_CO2: float = 0.05,
        energy_balance: bool = True,
        stage_efficiency: object = 1.0,
        max_outer_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
) -> AmineNSResult:
    """Solve an amine stripper as a Naphtali-Sandholm column.

    Equivalent to :class:`AmineStripper.solve` but uses the rigorous
    N-S engine.  The reboiler at stage N corresponds to the steam
    stripper, the condenser at the top to the overhead reflux drum.

    Parameters
    ----------
    amine_name : "MEA" / "MDEA" / "DEA" / "PZ" / "AMP" etc.
    total_amine : amine concentration in liquid [mol/kg solvent].
        Used by the underlying :class:`AmineSystem`.
    L : mol amine / s.
    G : steam feed rate at the reboiler [mol/s].
    alpha_rich : feed loading [mol CO2 / mol amine].
    wt_frac_amine : weight fraction in solvent, default 0.30.
    n_stages : default 15.
    feed_stage : 1-indexed; default 1 (top).
    T_top, T_bottom : initial T profile bounds [K].
    P : column pressure [Pa].
    y_reb_CO2 : CO2 mole fraction in the reboiler steam.
    energy_balance : default True.
    stage_efficiency : Murphree (default 1.0 — rigorous stages).
    """
    from ..distillation import distillation_column

    amine = lookup_amine(amine_name)
    system = AmineSystem(amine, total_amine)
    species_names = ["CO2", "H2O", amine_name]

    activity = AmineActivityModel(system, species_names)
    psat_funcs = build_amine_psat_funcs(species_names, amine_name)

    # Build feed composition from rich stream
    # Per mol amine: 1 mol amine + α mol CO2 + N_water mol H2O
    # where N_water = (1 - wt_frac) / wt_frac · MW_amine / MW_water
    MW_amine = float(amine.MW)
    MW_water = 18.0153
    N_water_per_amine = ((1.0 - wt_frac_amine) / wt_frac_amine
                              * MW_amine / MW_water)
    moles_per_amine = 1.0 + alpha_rich + N_water_per_amine
    z_CO2 = alpha_rich / moles_per_amine
    z_amine = 1.0 / moles_per_amine
    z_H2O = N_water_per_amine / moles_per_amine
    feed_z = [z_CO2, z_H2O, z_amine]
    feed_F = L * moles_per_amine     # total feed flow [mol/s]

    # Distillate rate ≈ steam + CO2 stripped (small rise, but bounded by G)
    distillate_rate = G + L * alpha_rich * 0.95     # over-allocate slightly

    if energy_balance:
        h_V_funcs, h_L_funcs = build_amine_enthalpy_funcs(species_names,
                                                                  system)
    else:
        h_V_funcs = h_L_funcs = None

    T_init = list(np.linspace(T_top, T_bottom, n_stages))

    col = distillation_column(
        n_stages=n_stages,
        feed_stage=feed_stage,
        feed_F=feed_F,
        feed_z=feed_z,
        feed_T=T_top,
        reflux_ratio=0.5,
        distillate_rate=distillate_rate,
        pressure=P,
        species_names=species_names,
        activity_model=activity,
        psat_funcs=psat_funcs,
        T_init=T_init,
        max_outer_iter=max_outer_iter,
        tol=tol,
        energy_balance=energy_balance,
        h_V_funcs=h_V_funcs,
        h_L_funcs=h_L_funcs,
        stage_efficiency=stage_efficiency,
        verbose=verbose,
    )

    # Per-stage α and P_CO2_eq
    alpha_list: List[float] = []
    P_CO2_list: List[float] = []
    for j in range(n_stages):
        x_j = col.x[j, :]
        T_j = col.T[j]
        a = activity.loading(x_j)
        alpha_list.append(a)
        P_CO2_list.append(activity.equilibrium_P_CO2(a, T_j))

    alpha_lean = alpha_list[-1]
    alpha_rich_top = alpha_list[0]
    co2_recovery = max(0.0, (alpha_rich - alpha_lean) / max(alpha_rich, 1e-9))

    # Compute Q_R, Q_C post-hoc using the same boundary balances
    Q_R = Q_C = None
    if energy_balance and h_V_funcs is not None:
        C = len(species_names)
        T0 = float(col.T[0])
        y0 = col.y[0, :]
        V_top_eff = (col.reflux_ratio + 1.0) * col.D
        h_V_top = sum(y0[i] * h_V_funcs[i](T0) for i in range(C))
        h_L_top = sum(y0[i] * h_L_funcs[i](T0) for i in range(C))
        Q_C = float(V_top_eff * (h_V_top - h_L_top))

        N = n_stages
        Tb = float(col.T[N - 1])
        Tabove = float(col.T[N - 2])
        h_V_N = sum(col.y[N - 1, i] * h_V_funcs[i](Tb) for i in range(C))
        h_L_N = sum(col.x[N - 1, i] * h_L_funcs[i](Tb) for i in range(C))
        h_L_above = sum(col.x[N - 2, i] * h_L_funcs[i](Tabove)
                            for i in range(C))
        Q_R = float(col.V[N - 1] * h_V_N
                       + col.B * h_L_N
                       - col.L[N - 2] * h_L_above)

    return AmineNSResult(
        column_result=col,
        alpha=alpha_list,
        P_CO2_eq=P_CO2_list,
        alpha_lean=float(alpha_lean),
        alpha_rich=float(alpha_rich_top),
        co2_recovery=float(co2_recovery),
        Q_R=Q_R, Q_C=Q_C,
    )


def amine_absorber_ns(
        amine_name: str,
        total_amine: float,
        L: float,
        G: float,
        alpha_lean: float,
        y_in_CO2: float,
        wt_frac_amine: float = 0.30,
        n_stages: int = 10,
        T_liquid_in: float = 313.15,
        T_gas_in: float = 313.15,
        P: float = 1.013e5,
        energy_balance: bool = False,
        stage_efficiency: object = 1.0,
        inert_name: str = "N2",
        y_in_H2O: Optional[float] = None,
        max_outer_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
) -> AmineNSResult:
    """Solve an amine CO2 absorber as a Naphtali-Sandholm column.

    The N-S column is configured with two feeds: lean amine at the
    top (stage 1) and flue gas at the bottom (stage N).  Reflux
    ratio is set to zero and the distillate rate is the gas feed
    rate, which mass-balances the reboiler stage to V_R ≈ 0.

    The flue gas carries an **inert non-condensable** (default
    "N2") at the balance of CO2 and water — without this the
    N-S bubble-point relations cannot accommodate the high water
    vapor content at the absorber's low operating T (water would
    condense).  The inert is given a very low pseudo-P_sat so it
    stays in vapor phase throughout.

    Parameters
    ----------
    amine_name : str
    total_amine : amine concentration [mol/kg solvent]
    L : mol amine flow [mol/s].
    G : flue gas flow [mol/s].
    alpha_lean : lean amine loading entering the top.
    y_in_CO2 : CO2 mole fraction in flue gas.
    wt_frac_amine : weight fraction; default 0.30.
    n_stages : default 10.
    T_liquid_in, T_gas_in : default 40 °C feed temperatures.
    P : column pressure [Pa]; default 1.013e5 (atmospheric).
    energy_balance : default False (isothermal absorber).
    stage_efficiency : Murphree, default 1.0.
    inert_name : str, default "N2".  Species name for the
        non-condensable carrier (must not be CO2/H2O/amine).
    y_in_H2O : float, optional.  Water mole fraction in flue gas.
        Default is water-saturated at T_gas_in:
        y_H2O_sat = P_water_sat(T_gas_in) / P.
    """
    from ..distillation import distillation_column
    from ..distillation.column import FeedSpec

    amine = lookup_amine(amine_name)
    system = AmineSystem(amine, total_amine)
    species_names = ["CO2", "H2O", amine_name, inert_name]

    activity = AmineActivityModel(system, species_names)
    psat_funcs = build_amine_psat_funcs(species_names, amine_name)

    # Liquid feed (lean amine) at top
    MW_amine = float(amine.MW)
    MW_water = 18.0153
    N_water_per_amine = ((1.0 - wt_frac_amine) / wt_frac_amine
                              * MW_amine / MW_water)
    moles_per_amine_lean = 1.0 + alpha_lean + N_water_per_amine
    z_liquid = [
        alpha_lean / moles_per_amine_lean,           # CO2
        N_water_per_amine / moles_per_amine_lean,    # H2O
        1.0 / moles_per_amine_lean,                  # amine
        0.0,                                           # inert
    ]
    F_liquid = L * moles_per_amine_lean

    # Gas feed at bottom: CO2, H2O (saturated), amine=0, inert=balance
    if y_in_H2O is None:
        y_in_H2O = _P_water_sat_bar(T_gas_in) * 1e5 / P
        y_in_H2O = min(max(y_in_H2O, 0.0), 0.5)   # cap for sanity
    y_in_inert = max(0.0, 1.0 - y_in_CO2 - y_in_H2O)
    z_gas = [y_in_CO2, y_in_H2O, 0.0, y_in_inert]
    F_gas = G

    if energy_balance:
        h_V_funcs, h_L_funcs = build_amine_enthalpy_funcs(species_names,
                                                                  system)
    else:
        h_V_funcs = h_L_funcs = None

    T_init = list(np.linspace(T_liquid_in, T_gas_in, n_stages))

    feeds = [
        FeedSpec(stage=1, F=F_liquid, z=list(z_liquid),
                  T=T_liquid_in, q=1.0),
        FeedSpec(stage=n_stages, F=F_gas, z=list(z_gas),
                  T=T_gas_in, q=0.0),
    ]

    col = distillation_column(
        n_stages=n_stages,
        reflux_ratio=0.001,
        distillate_rate=F_gas,
        pressure=P,
        species_names=species_names,
        activity_model=activity,
        psat_funcs=psat_funcs,
        feeds=feeds,
        T_init=T_init,
        max_outer_iter=max_outer_iter,
        tol=tol,
        energy_balance=energy_balance,
        h_V_funcs=h_V_funcs,
        h_L_funcs=h_L_funcs,
        stage_efficiency=stage_efficiency,
        condenser="partial",
        verbose=verbose,
    )

    # Per-stage α
    alpha_list: List[float] = []
    P_CO2_list: List[float] = []
    for j in range(n_stages):
        x_j = col.x[j, :]
        T_j = col.T[j]
        a = activity.loading(x_j)
        alpha_list.append(a)
        P_CO2_list.append(activity.equilibrium_P_CO2(a, T_j))

    alpha_top = alpha_list[0]
    alpha_bot = alpha_list[-1]

    # CO2 recovery
    co2_in = G * y_in_CO2
    y_out_CO2 = float(col.y[0, 0])
    V_top = (col.reflux_ratio + 1.0) * col.D
    co2_out = V_top * y_out_CO2
    co2_recovery = max(0.0, (co2_in - co2_out) / max(co2_in, 1e-9))

    return AmineNSResult(
        column_result=col,
        alpha=alpha_list,
        P_CO2_eq=P_CO2_list,
        alpha_lean=float(alpha_top),
        alpha_rich=float(alpha_bot),
        co2_recovery=float(co2_recovery),
        Q_R=None, Q_C=None,
    )
