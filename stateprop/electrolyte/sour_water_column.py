"""Sour-water stripper column model (v0.9.111).

Couples the v0.9.97 :mod:`sour_water` aqueous-phase chemistry into the
:func:`stateprop.distillation.distillation_column` Naphtali-Sandholm
solver.  This lets users model a real industrial sour-water stripper
as a multi-stage distillation column with the correct effective
volatilities — accounting for the partial dissociation of NH₃/H₂S/CO₂
in water at each stage's local pH.

Background
----------
A sour-water stripper removes dissolved NH₃, H₂S, and CO₂ from
process water (refinery condensate, ammonia-plant condensate, etc.)
by counter-current contact with steam.  The volatile species are
*partially* dissociated in solution:

    NH₃ + H₂O ⇌ NH₄⁺ + OH⁻     pK_b ≈ 4.75
    H₂S      ⇌ HS⁻ + H⁺        pK_a1 ≈ 7.0
    CO₂ + H₂O ⇌ HCO₃⁻ + H⁺     pK_a1 ≈ 6.35

The ionic forms are non-volatile, so the *effective* Henry's-law
coefficient at a given pH is:

    H_eff(T, pH) = H_molecular(T) · α_molecular(T, pH)

where α is the molecular-form fraction.  For an N-S column built
around modified Raoult's law (K_i = γ_i · P_sat_i(T) / P), this maps
to:

    γ_i = α_molecular_i(T, pH)
    P_sat_i(T) = H_henry_i(T) · 55.51   [mol H₂O / kg]

The factor 55.51 = 1000/18.015 converts the molality-based Henry's
coefficient (Pa·kg/mol) into a mole-fraction-basis pseudo-P_sat for
dilute aqueous solutions where m_i ≈ x_i · 55.51 / x_water.

For water, γ ≈ 1 and P_sat is the standard water Antoine.

Because pH depends on the local liquid composition (volatile and ionic
species), the activity model performs a speciation calculation each
time it is queried by the column solver.  This is wrapped in
:class:`SourWaterActivityModel`, which the N-S column treats as any
other activity-coefficient model.

Energy balance (v0.9.112)
-------------------------
Per-species ideal-gas + heat-of-solution enthalpy callables are built
by :func:`build_enthalpy_funcs`.  These plug into the N-S column's
``energy_balance=True`` mode so the column closes a per-stage heat
balance, computes the reboiler duty :math:`Q_R`, the condenser duty
:math:`Q_C`, and reports a steam-to-water mass ratio.

Murphree efficiency (v0.9.112)
------------------------------
Real sour-water strippers run at 60-80 % Murphree vapor efficiency.
:func:`sour_water_stripper` exposes the column's ``stage_efficiency``
keyword (default ``0.65``) which can be set to a scalar in (0, 1] or
a per-stage sequence.

Limitations
-----------
* Dilute approximation: m_i = x_i · 55.51 / x_water assumes a dilute
  aqueous solution.  Accuracy degrades above ~5 mol/kg total volatiles.
* The strong-electrolyte background (Na⁺/Cl⁻ from process upsets) can
  be supplied via the optional ``extra_strong_cations`` /
  ``extra_strong_anions`` arguments to :class:`SourWaterActivityModel`.
* Energy balance uses constant cp_p and a Watson-style ΔH_vap(T) for
  water; the volatiles' partial-molar cp_p in solution is approximated
  by their ideal-gas cp_p (dilute-aqueous approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (Callable, Dict, List, Optional, Sequence)

import numpy as np

from .sour_water import (
    speciate, henry_constant, dissociation_K, SourWaterSpeciation,
)
from .amine_stripper import P_water_sat as _P_water_sat_bar


# Molality conversion: 1 kg H₂O = 1000/M_H2O = 55.508 mol
_M_H2O_PER_KG: float = 1000.0 / 18.0153

# Standard volatile species recognised by this module
_SUPPORTED_VOLATILES: tuple = ("NH3", "H2S", "CO2")


# =====================================================================
# Activity model wrapper
# =====================================================================

class SourWaterActivityModel:
    """Activity-coefficient model for the sour-water + steam system.

    Returns γ_i for each species given (T, x) by:

    1. Converting mole fractions to molalities (assumes water is the
       solvent and the solution is dilute):
            m_i = x_i · 55.51 / x_water       for each volatile i
    2. Calling :func:`stateprop.electrolyte.sour_water.speciate` to get
       the equilibrium pH and the molecular-fraction α for each
       volatile species.
    3. Returning γ_volatile = α_volatile and γ_water = 1.

    Parameters
    ----------
    species_names : sequence of str
        Ordered species names.  Must include "H2O" and at least one of
        "NH3", "H2S", "CO2".  Other names are passed through with γ = 1
        (treated as inert / non-condensable).
    extra_strong_cations : float, default 0.0
        Background strong-cation molality (Na⁺, K⁺, …) NOT generated
        from the volatile dissociation equilibria.
    extra_strong_anions : float, default 0.0
        Background strong-anion molality (Cl⁻, SO₄²⁻, …).
    pH_min : float, default 3.0
        Lower pH safety bound for speciation.
    pH_max : float, default 12.0
        Upper pH safety bound for speciation.
    pitzer_corrections : bool, default False  (v0.9.116)
        If True, apply Setschenow salting-out corrections to the
        molecular volatile activity coefficients to capture
        high-ionic-strength behaviour beyond the dilute Davies-equation
        regime built into :func:`speciate`.  The correction is

            γ_volatile_corrected = γ_volatile · 10^(k_s · I_strong)

        where ``I_strong = 0.5 · (m_extra_cation + m_extra_anion)``
        is the strong-ion ionic strength contribution and the
        Setschenow constants are NH3: 0.077, H2S: 0.137, CO2: 0.103
        kg/mol (Long-McDevit 1952, Schumpe 1993 NaCl-anchor values).
        Has noticeable effect above I ~ 1 mol/kg; <2 % effect below
        0.5 mol/kg.
    """

    # Setschenow salting-out coefficients [kg/mol] for molecular gases
    # in NaCl, anchor at 25 °C, weak T-dependence ignored.
    # Source: Schumpe 1993 review, also Weisenberger-Schumpe 1996.
    _K_S = {
        "NH3": 0.077,
        "H2S": 0.137,
        "CO2": 0.103,
    }

    def __init__(self,
                  species_names: Sequence[str],
                  extra_strong_cations: float = 0.0,
                  extra_strong_anions: float = 0.0,
                  pH_min: float = 3.0,
                  pH_max: float = 12.0,
                  pitzer_corrections: bool = False):
        self.species_names = list(species_names)
        if "H2O" not in self.species_names:
            raise ValueError(
                "SourWaterActivityModel requires 'H2O' in species_names; "
                f"got {self.species_names}")
        # Index map for fast lookup
        self._idx_H2O = self.species_names.index("H2O")
        self._idx_volatile: Dict[str, Optional[int]] = {}
        for sp in _SUPPORTED_VOLATILES:
            self._idx_volatile[sp] = (
                self.species_names.index(sp)
                if sp in self.species_names else None)
        # Need at least one volatile
        if all(v is None for v in self._idx_volatile.values()):
            raise ValueError(
                "SourWaterActivityModel needs at least one of "
                f"{_SUPPORTED_VOLATILES} in species_names; "
                f"got {self.species_names}")
        self.extra_strong_cations = float(extra_strong_cations)
        self.extra_strong_anions = float(extra_strong_anions)
        self.pH_min = float(pH_min)
        self.pH_max = float(pH_max)
        self.pitzer_corrections = bool(pitzer_corrections)
        # Pre-compute the Setschenow correction factor (depends only on
        # constructor-time strong-ion molalities, not on x or T).
        if self.pitzer_corrections:
            # I_strong = 0.5 (m+·z+² + m_·z_²); for 1:1 background
            # m+ ≈ m_, so I ≈ 0.5(m+ + m_).
            self._I_strong = 0.5 * (self.extra_strong_cations
                                          + self.extra_strong_anions)
        else:
            self._I_strong = 0.0

    # -----------------------------------------------------------------
    def speciate_at(self, T: float, x: Sequence[float]) -> SourWaterSpeciation:
        """Return the SourWaterSpeciation result for liquid (T, x)."""
        x = np.asarray(x, dtype=float)
        x_water = max(float(x[self._idx_H2O]), 1e-9)
        # m_i = x_i · 55.51 / x_water
        scale = _M_H2O_PER_KG / x_water
        m_NH3 = (float(x[self._idx_volatile["NH3"]]) * scale
                 if self._idx_volatile["NH3"] is not None else 0.0)
        m_H2S = (float(x[self._idx_volatile["H2S"]]) * scale
                 if self._idx_volatile["H2S"] is not None else 0.0)
        m_CO2 = (float(x[self._idx_volatile["CO2"]]) * scale
                 if self._idx_volatile["CO2"] is not None else 0.0)
        # Defensive: clamp molalities to avoid speciate() blow-ups
        m_NH3 = max(m_NH3, 0.0)
        m_H2S = max(m_H2S, 0.0)
        m_CO2 = max(m_CO2, 0.0)
        return speciate(
            T=T,
            m_NH3_total=m_NH3,
            m_H2S_total=m_H2S,
            m_CO2_total=m_CO2,
            extra_strong_cations=self.extra_strong_cations,
            extra_strong_anions=self.extra_strong_anions,
            apply_davies_gammas=self.pitzer_corrections,
        )

    # -----------------------------------------------------------------
    def gammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Activity coefficients γ_i for each species at (T, x).

        γ_volatile = α_molecular (volatile fraction)
        γ_water = 1
        γ_inert = 1

        With pitzer_corrections=True, additional Setschenow salting-out
        factor 10^(k_s · I_strong) is applied to molecular volatiles
        for high-ionic-strength sour water (background salts).
        """
        x = np.asarray(x, dtype=float)
        N = len(self.species_names)
        gammas = np.ones(N)
        try:
            sp = self.speciate_at(T, x)
        except Exception:
            # Speciation failed (e.g. all-vapor stage); return unity γ
            # so the column solver can still progress.
            return gammas

        # Setschenow factor (1.0 if pitzer_corrections=False since
        # _I_strong=0)
        if self.pitzer_corrections and self._I_strong > 0:
            ks_NH3 = self._K_S["NH3"]
            ks_H2S = self._K_S["H2S"]
            ks_CO2 = self._K_S["CO2"]
            f_NH3 = 10.0 ** (ks_NH3 * self._I_strong)
            f_H2S = 10.0 ** (ks_H2S * self._I_strong)
            f_CO2 = 10.0 ** (ks_CO2 * self._I_strong)
        else:
            f_NH3 = f_H2S = f_CO2 = 1.0

        # Bound α to (0, 1] (numerical safety) and apply Setschenow
        if self._idx_volatile["NH3"] is not None:
            gammas[self._idx_volatile["NH3"]] = max(
                min(sp.alpha_NH3 * f_NH3, 100.0), 1e-6)
        if self._idx_volatile["H2S"] is not None:
            gammas[self._idx_volatile["H2S"]] = max(
                min(sp.alpha_H2S * f_H2S, 100.0), 1e-6)
        if self._idx_volatile["CO2"] is not None:
            gammas[self._idx_volatile["CO2"]] = max(
                min(sp.alpha_CO2 * f_CO2, 100.0), 1e-6)
        return gammas


# =====================================================================
# Pseudo-P_sat builder for Henry's-law species
# =====================================================================

def _make_henry_psat(species: str) -> Callable[[float], float]:
    """Build a P_sat(T) callable for a Henry-law volatile.

    Returns a callable that produces ``H_henry(T) * 55.51`` in Pa,
    appropriate for the modified-Raoult K-value form used inside
    :func:`distillation_column` when paired with
    :class:`SourWaterActivityModel`.
    """
    def psat(T: float) -> float:
        return float(henry_constant(species, T)) * _M_H2O_PER_KG
    psat.__name__ = f"henry_psat_{species}"
    psat.__doc__ = (
        f"Pseudo-P_sat(T) for {species} based on Henry's law: "
        f"H(T) [Pa·kg/mol] · 55.51 mol H₂O/kg.")
    return psat


def _water_psat_Pa(T: float) -> float:
    """Water saturation pressure [Pa] from Antoine (NIST simplified)."""
    return _P_water_sat_bar(T) * 1.0e5


def _inert_psat(T: float) -> float:
    """Effectively non-condensable: P_sat huge so K = ∞ would be bad.
    Use a moderate value so the species stays in the vapor."""
    return 1e9    # 10 GPa


def build_psat_funcs(species_names: Sequence[str]) -> List[Callable[[float], float]]:
    """Return a list of P_sat(T) callables, one per species.

    For NH₃/H₂S/CO₂: pseudo-P_sat = H_henry · 55.51.
    For H₂O: standard water Antoine (Pa).
    For any other species: very large constant (treated as inert
    non-condensable).
    """
    funcs: List[Callable[[float], float]] = []
    for sp in species_names:
        if sp == "H2O":
            funcs.append(_water_psat_Pa)
        elif sp in _SUPPORTED_VOLATILES:
            funcs.append(_make_henry_psat(sp))
        else:
            funcs.append(_inert_psat)
    return funcs


# =====================================================================
# Energy-balance enthalpy callables (v0.9.112)
# =====================================================================
#
# Reference state: pure ideal gas at T_ref = 298.15 K.
#
#   h_V_water(T) = ΔH_vap_water(T) + cp_V_water · (T − T_ref)
#   h_L_water(T) = cp_L_water · (T − T_ref)
#
# For volatile species (NH3, H2S, CO2):
#   h_V_i(T) = cp_V_i · (T − T_ref)
#   h_L_i(T) = ΔH_diss_i + cp_L_i · (T − T_ref)
#
# ΔH_diss_i is the heat of dissolution (gas → infinitely-dilute aqueous
# solution) at T_ref.  Negative for exothermic dissolution.
#
# The volatile heat-of-solution data are taken from Wilhelm 1977 plus
# common textbook values (Smith Van Ness Abbott, Perry).  cp_V values
# are ideal-gas at 298 K; partial-molar cp_L for the dilute aqueous
# species is approximated by the gas-phase cp (justifiable in the
# dilute limit since the Setchenov / non-ideality corrections are small
# compared to ΔH_vap_water and ΔH_diss).

_T_REF: float = 298.15

# Heat-of-dissolution ΔH_diss in J/mol (gas → aqueous, exothermic = negative)
_DELTA_H_DISS: Dict[str, float] = {
    "NH3": -34_200.0,
    "H2S": -19_500.0,
    "CO2": -19_700.0,
}

# Ideal-gas cp_p at 298 K [J/(mol·K)]  — constant approximation
_CP_V: Dict[str, float] = {
    "NH3": 35.5,
    "H2S": 34.2,
    "CO2": 37.1,
    "H2O": 33.6,
}

# Liquid (aqueous solute) partial-molar cp_p approximated by gas cp_p
# for the volatiles; pure liquid cp_p for water.
_CP_L: Dict[str, float] = {
    "NH3": 35.5,
    "H2S": 34.2,
    "CO2": 37.1,
    "H2O": 75.3,
}

# Heat of vaporization of water at the reference temperature [J/mol]
_DELTA_H_VAP_WATER_REF: float = 43_990.0   # at 298.15 K
_T_C_WATER: float = 647.1                  # critical T [K]


def _delta_H_vap_water(T: float) -> float:
    """ΔH_vap of water [J/mol] at T using a Watson-correlation reduction
    from the 298.15 K reference value.

    H_vap(T) = H_vap(T_ref) · ((T_c − T) / (T_c − T_ref))^0.38
    """
    if T >= _T_C_WATER:
        return 0.0
    ratio = (_T_C_WATER - float(T)) / (_T_C_WATER - _T_REF)
    return _DELTA_H_VAP_WATER_REF * ratio ** 0.38


def _h_V_water(T: float) -> float:
    """Vapor enthalpy of water [J/mol] at T (ref: ideal gas at T_ref)."""
    return _delta_H_vap_water(T) + _CP_V["H2O"] * (T - _T_REF)


def _h_L_water(T: float) -> float:
    """Liquid enthalpy of water [J/mol] at T (ref: ideal gas at T_ref)."""
    return _CP_L["H2O"] * (T - _T_REF)


def _make_h_V_volatile(species: str) -> Callable[[float], float]:
    cp = _CP_V[species]
    def h_V(T: float) -> float:
        return cp * (T - _T_REF)
    h_V.__name__ = f"h_V_{species}"
    return h_V


def _make_h_L_volatile(species: str) -> Callable[[float], float]:
    cp = _CP_L[species]
    dH_diss = _DELTA_H_DISS[species]
    def h_L(T: float) -> float:
        return dH_diss + cp * (T - _T_REF)
    h_L.__name__ = f"h_L_{species}"
    return h_L


def _h_inert_V(T: float) -> float:
    """Vapor enthalpy of an inert (e.g., N₂); cp_p ≈ 29 J/(mol·K)."""
    return 29.1 * (T - _T_REF)


def _h_inert_L(T: float) -> float:
    """Liquid enthalpy of an inert (negligibly soluble; treat as
    bulk-vapor enthalpy plus a small dissolution penalty)."""
    return 0.0 + 29.1 * (T - _T_REF)


def build_enthalpy_funcs(
        species_names: Sequence[str]
) -> tuple:
    """Build (h_V_funcs, h_L_funcs) for the N-S energy balance.

    Returns
    -------
    h_V_funcs, h_L_funcs : lists of callables (T -> J/mol), one per
        species in ``species_names``.

    The reference state is ideal gas at 298.15 K.  For water, the
    vapor enthalpy includes the (T-dependent) heat of vaporization
    via a Watson reduction.  For volatile sour-water species, the
    liquid enthalpy is offset by the heat of dissolution (exothermic).
    """
    h_V: List[Callable[[float], float]] = []
    h_L: List[Callable[[float], float]] = []
    for sp in species_names:
        if sp == "H2O":
            h_V.append(_h_V_water)
            h_L.append(_h_L_water)
        elif sp in _SUPPORTED_VOLATILES:
            h_V.append(_make_h_V_volatile(sp))
            h_L.append(_make_h_L_volatile(sp))
        else:
            h_V.append(_h_inert_V)
            h_L.append(_h_inert_L)
    return h_V, h_L


# =====================================================================
# Convenience entry point: solve a sour-water stripper
# =====================================================================

@dataclass
class SourWaterStripperResult:
    """Result of a sour-water stripper column solve.

    Attributes
    ----------
    column_result : DistillationColumnResult
        Raw N-S column output (T profile, x/y profiles, flows).
    pH : List[float]
        Per-stage liquid pH from speciation.
    alpha_NH3, alpha_H2S, alpha_CO2 : List[float]
        Per-stage molecular fraction of each volatile.
    bottoms_strip_efficiency : Dict[str, float]
        For each volatile, (m_feed - m_bottoms) / m_feed [-].
    Q_R : Optional[float]
        Reboiler duty [W], present iff energy_balance=True.
    Q_C : Optional[float]
        Condenser duty [W], present iff energy_balance=True.
    steam_ratio_kg_per_kg_water : Optional[float]
        Reboiler steam consumption per kg of water in the feed.
        Useful industrial KPI.  None unless energy_balance=True.
    stage_efficiency_used : Optional
        Murphree vapor efficiency that was actually used (scalar,
        list, or None for the default-1.0 case).
    """
    column_result: object
    pH: List[float]
    alpha_NH3: List[float]
    alpha_H2S: List[float]
    alpha_CO2: List[float]
    bottoms_strip_efficiency: Dict[str, float]
    Q_R: Optional[float] = None
    Q_C: Optional[float] = None
    steam_ratio_kg_per_kg_water: Optional[float] = None
    stage_efficiency_used: object = None


def sour_water_stripper(
        n_stages: int,
        feed_stage: int,
        feed_F: float,
        feed_z: Sequence[float],
        feed_T: float,
        species_names: Sequence[str],
        reflux_ratio: float,
        distillate_rate: float,
        pressure: float,
        T_init: Optional[Sequence[float]] = None,
        extra_strong_cations: float = 0.0,
        extra_strong_anions: float = 0.0,
        max_outer_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
        # v0.9.112 additions
        energy_balance: bool = False,
        h_V_funcs: Optional[Sequence[Callable[[float], float]]] = None,
        h_L_funcs: Optional[Sequence[Callable[[float], float]]] = None,
        stage_efficiency=0.65,
        # v0.9.116 additions
        pitzer_corrections: bool = False,
        **column_kwargs,
) -> SourWaterStripperResult:
    """Solve a sour-water stripper as a Naphtali-Sandholm column.

    Wraps :func:`stateprop.distillation.distillation_column` with the
    :class:`SourWaterActivityModel` and Henry-derived P_sat funcs so
    the column reproduces the partial-pressure relationship

        P_partial_i = H_eff_i(T, pH) · m_i

    at each stage.

    Parameters
    ----------
    n_stages, feed_stage, feed_F, feed_z, feed_T :
        Standard distillation column inputs.
    species_names : sequence of str
        Must include "H2O" and at least one of NH3/H2S/CO2.
    reflux_ratio, distillate_rate, pressure :
        Standard column external specs.  ``pressure`` is in Pa.
    T_init : optional list[float]
        Initial T profile (length n_stages).  Defaults to a linear ramp
        from ``feed_T`` to ``feed_T + 30 K``.
    extra_strong_cations, extra_strong_anions : float
        Background non-volatile strong-electrolyte molality.
    energy_balance : bool, default False  (v0.9.112)
        If True, run the column with stage-wise energy balances.
        Auto-builds h_V_funcs/h_L_funcs from sour-water thermodynamics
        unless the user provides them.  Result will include Q_R, Q_C,
        and ``steam_ratio_kg_per_kg_water``.
    h_V_funcs, h_L_funcs : sequences of T-callables, optional
        Override the default sour-water enthalpy callables.
    stage_efficiency : None, float, or sequence of floats (v0.9.112)
        Murphree vapor efficiency for the column.  Default is 0.65,
        the typical industrial value for sour-water trayed columns.
        Pass ``1.0`` (or a list of 1.0s) for theoretical-stage solves.
    pitzer_corrections : bool, default False  (v0.9.116)
        If True, apply Setschenow salting-out corrections to the
        molecular volatile activities.  Recommended when the
        background ionic strength (from extra_strong_cations /
        anions) exceeds ~1 mol/kg, where the dilute Davies model
        underpredicts γ_volatile by 10-50 %.
    column_kwargs :
        Additional kwargs passed straight to ``distillation_column``.

    Returns
    -------
    SourWaterStripperResult
    """
    from ..distillation import distillation_column   # local import

    if T_init is None:
        T_init = list(np.linspace(feed_T, feed_T + 30.0, n_stages))

    activity = SourWaterActivityModel(
        species_names=species_names,
        extra_strong_cations=extra_strong_cations,
        extra_strong_anions=extra_strong_anions,
        pitzer_corrections=pitzer_corrections,
    )
    psat_funcs = build_psat_funcs(species_names)

    # Build enthalpy callables when the user enables energy_balance
    if energy_balance and (h_V_funcs is None or h_L_funcs is None):
        h_V_default, h_L_default = build_enthalpy_funcs(species_names)
        if h_V_funcs is None:
            h_V_funcs = h_V_default
        if h_L_funcs is None:
            h_L_funcs = h_L_default

    col_res = distillation_column(
        n_stages=n_stages,
        feed_stage=feed_stage,
        feed_F=feed_F,
        feed_z=feed_z,
        feed_T=feed_T,
        reflux_ratio=reflux_ratio,
        distillate_rate=distillate_rate,
        pressure=pressure,
        species_names=list(species_names),
        activity_model=activity,
        psat_funcs=psat_funcs,
        T_init=T_init,
        max_outer_iter=max_outer_iter,
        tol=tol,
        verbose=verbose,
        energy_balance=energy_balance,
        h_V_funcs=h_V_funcs,
        h_L_funcs=h_L_funcs,
        stage_efficiency=stage_efficiency,
        **column_kwargs,
    )

    # Per-stage pH and alpha post-processing
    pH_list: List[float] = []
    a_NH3: List[float] = []
    a_H2S: List[float] = []
    a_CO2: List[float] = []
    for j in range(n_stages):
        x_j = col_res.x[j, :]
        T_j = col_res.T[j]
        try:
            sp = activity.speciate_at(T_j, x_j)
            pH_list.append(sp.pH)
            a_NH3.append(sp.alpha_NH3)
            a_H2S.append(sp.alpha_H2S)
            a_CO2.append(sp.alpha_CO2)
        except Exception:
            pH_list.append(float("nan"))
            a_NH3.append(float("nan"))
            a_H2S.append(float("nan"))
            a_CO2.append(float("nan"))

    # Strip efficiency for each volatile (feed → bottoms)
    feed_z_arr = np.asarray(feed_z, dtype=float)
    bottoms_z = col_res.x[-1, :]
    eff: Dict[str, float] = {}
    for sp in _SUPPORTED_VOLATILES:
        if sp in species_names:
            i = list(species_names).index(sp)
            if feed_z_arr[i] > 1e-12:
                B = col_res.B
                F = float(feed_F)
                m_feed = F * feed_z_arr[i]
                m_bot = B * bottoms_z[i]
                eff[sp] = float(max(0.0, (m_feed - m_bot) / m_feed))
            else:
                eff[sp] = float("nan")

    # Energy-balance KPIs
    Q_R = None
    Q_C = None
    steam_ratio = None
    if energy_balance:
        # Compute Q_R, Q_C post-solve using the same boundary
        # enthalpy balances that the reactive_column helpers use.
        C = len(species_names)
        # Q_C: total condenser, Q_C = V_top · (h_V_top − h_L_top)
        T0 = float(col_res.T[0])
        y0 = col_res.y[0, :]
        V_top = (col_res.reflux_ratio + 1.0) * col_res.D
        h_V_top = sum(y0[i] * h_V_funcs[i](T0) for i in range(C))
        h_L_top = sum(y0[i] * h_L_funcs[i](T0) for i in range(C))
        Q_C = float(V_top * (h_V_top - h_L_top))
        # Q_R: reboiler, Q_R = V_N · h_V_N + B · h_L_N − L_{N-1} · h_L_{N-1}
        N = n_stages
        Tb = float(col_res.T[N - 1])
        Tabove = float(col_res.T[N - 2])
        h_V_N = sum(col_res.y[N - 1, i] * h_V_funcs[i](Tb) for i in range(C))
        h_L_N = sum(col_res.x[N - 1, i] * h_L_funcs[i](Tb) for i in range(C))
        h_L_above = sum(col_res.x[N - 2, i] * h_L_funcs[i](Tabove)
                            for i in range(C))
        Q_R = float(col_res.V[N - 1] * h_V_N
                       + col_res.B * h_L_N
                       - col_res.L[N - 2] * h_L_above)
        # Steam-to-water KPI
        if "H2O" in species_names:
            i_h2o = list(species_names).index("H2O")
            kg_water_in_feed_per_s = (
                feed_F * feed_z_arr[i_h2o] * 18.015e-3)
            if kg_water_in_feed_per_s > 0 and Q_R > 0:
                dHvap = _delta_H_vap_water(feed_T)
                kg_steam_per_s = abs(Q_R) / dHvap * 18.015e-3
                steam_ratio = kg_steam_per_s / kg_water_in_feed_per_s

    return SourWaterStripperResult(
        column_result=col_res,
        pH=pH_list,
        alpha_NH3=a_NH3,
        alpha_H2S=a_H2S,
        alpha_CO2=a_CO2,
        bottoms_strip_efficiency=eff,
        Q_R=Q_R,
        Q_C=Q_C,
        steam_ratio_kg_per_kg_water=steam_ratio,
        stage_efficiency_used=stage_efficiency,
    )
