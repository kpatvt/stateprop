"""Mineral solubility prediction (v0.9.101).

Computes saturation indices and equilibrium solubilities of common
minerals in aqueous solution using activity coefficients and water
activity from the multi-electrolyte Pitzer framework.

The saturation index is:

    SI = log10(IAP / K_sp)
    IAP = (γ_M · m_M)^a · (γ_A · m_A)^b · ... · a_w^n

where the mineral M_aA_b·nH2O dissolves as:

    M_aA_b·nH2O(s) ⇌ a·M^z+ + b·A^z- + n·H2O

Interpretation:
    SI > 0:  supersaturated → mineral will precipitate (scale risk)
    SI = 0:  saturated → equilibrium
    SI < 0:  undersaturated → mineral will dissolve (if present)

K_sp is the thermodynamic solubility product. T-dependence is via
van't Hoff form anchored at 25 °C:

    log_K_sp(T) = log_K_sp_25 - (ΔH_rxn / 2.303·R) · (1/T - 1/298.15)

This captures the dominant T-trend for most minerals over 0-100 °C.
For higher accuracy or wider T-ranges, the user can override
`log_K_sp_25` and `delta_H_rxn` with custom values.

Bundled minerals (25 °C unless noted):

    Highly soluble:        halite (NaCl), sylvite (KCl)
    Sulfates:              gypsum (CaSO4·2H2O), anhydrite (CaSO4),
                           barite (BaSO4), celestite (SrSO4),
                           mirabilite (Na2SO4·10H2O),
                           thenardite (Na2SO4),
                           epsomite (MgSO4·7H2O)
    Carbonates:            calcite (CaCO3), aragonite (CaCO3),
                           dolomite (CaMg(CO3)2),
                           magnesite (MgCO3)
    Hydroxides:            brucite (Mg(OH)2),
                           portlandite (Ca(OH)2)

References
----------
* Plummer, L. N., Busenberg, E. (1982). The solubilities of calcite,
  aragonite and vaterite in CO2-H2O solutions between 0 and 90 °C.
  Geochim. Cosmochim. Acta 46, 1011.
* Krumgalz, B. S., Pogorelsky, R., Pitzer, K. S. (1995). Volumetric
  ion-interaction parameters for single-solute aqueous electrolyte
  solutions at various temperatures. J. Phys. Chem. Ref. Data 25, 663.
* Reardon, E. J., Beckie, R. D. (1987). Modelling chemical equilibria
  of acid mine-drainage. Geochim. Cosmochim. Acta 51, 2355.
* Blount, C. W. (1977). Barite solubilities and thermodynamic
  quantities up to 300 °C and 1400 bars. American Mineralogist 62, 942.
* Marshall, W. L., Slusher, R. (1966). Thermodynamics of calcium
  sulfate dihydrate in aqueous sodium chloride solutions, 0-110 °C.
  J. Phys. Chem. 70, 4015.
* Parkhurst, D. L., Appelo, C. A. J. (2013). Description of input and
  examples for PHREEQC version 3 (USGS Techniques and Methods 6-A43).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from .pitzer import PitzerModel, lookup_salt
from .multi_pitzer import MultiPitzerSystem


_R = 8.314462618    # J/(mol·K)
_LN10 = np.log(10.0)
_T_REF = 298.15


# =====================================================================
# Mineral data class
# =====================================================================

@dataclass
class Mineral:
    """Solubility data for an aqueous mineral.

    Parameters
    ----------
    name : str
        Common mineral name, e.g. "halite", "gypsum", "calcite".
    formula : str
        Chemical formula, e.g. "NaCl", "CaSO4·2H2O".
    cations : dict[str, int]
        Cation stoichiometry per formula unit, e.g. {"Ca++": 1} for
        gypsum, {"Ca++": 1, "Mg++": 1} for dolomite. Keys must match
        the ion naming convention used in MultiPitzerSystem ("Na+",
        "K+", "Mg++", "Ca++", "Sr++", "Ba++", "H+", etc.).
    anions : dict[str, int]
        Anion stoichiometry, e.g. {"SO4--": 1}, {"CO3--": 2}, etc.
    n_H2O : int
        Number of waters of crystallization in the formula unit.
        E.g. 2 for gypsum (CaSO4·2H2O), 10 for mirabilite, 0 for
        anhydrite or halite.
    log_K_sp_25 : float
        log10(K_sp) at 298.15 K — the **apparent** (stoichiometric)
        solubility product calibrated against total-concentration
        solubility measurements.  Used by the simple `saturation_index`
        and `solubility_in_water` API (no explicit complexation).
    delta_H_rxn : float
        Enthalpy of dissolution [J/mol].  Used for van't Hoff
        T-dependence:  d(ln K)/dT = ΔH / (R T²).
        Positive ΔH → solubility increases with T.
        Default 0 (T-independent K_sp; OK for narrow T ranges).
    binary_salt : str, optional
        If the mineral is a simple binary salt also in the bundled
        Pitzer single-electrolyte database (NaCl, KCl, etc.), this
        is the salt name that ``solubility_in_water`` will use for
        γ_pm and water-activity evaluation.  None for ternary
        minerals (dolomite) and minerals not in the binary DB.
    log_K_sp_25_thermo : float, optional
        Thermodynamic (free-ion) log10(K_sp) at 298.15 K, used by
        `SpeciationResult.saturation_index` when the user solves
        speciation with explicit complexation (v0.9.102+).
        For minerals strongly affected by ion-pairing (gypsum,
        anhydrite), this is typically 0.2-0.4 log units lower than
        log_K_sp_25.  If None, the apparent value is used (with a
        warning that this double-counts complexation).
    """
    name: str
    formula: str
    cations: Dict[str, int]
    anions: Dict[str, int]
    n_H2O: int
    log_K_sp_25: float
    delta_H_rxn: float = 0.0
    binary_salt: Optional[str] = None
    log_K_sp_25_thermo: Optional[float] = None

    @property
    def nu(self) -> int:
        """Total stoichiometric coefficient ν = Σ(cation count) + Σ(anion count)."""
        return sum(self.cations.values()) + sum(self.anions.values())

    def log_K_sp(self, T: float) -> float:
        """log10(K_sp) at temperature T via van't Hoff form.

        log_K(T) = log_K_25 + (-ΔH / (R · ln(10))) · (1/T - 1/Tr)
        """
        if self.delta_H_rxn == 0.0:
            return self.log_K_sp_25
        return (self.log_K_sp_25
                 - (self.delta_H_rxn / (_R * _LN10))
                   * (1.0 / T - 1.0 / _T_REF))

    def K_sp(self, T: float) -> float:
        """K_sp at T (linear scale)."""
        return 10.0 ** self.log_K_sp(T)


# =====================================================================
# Bundled mineral database
# =====================================================================
#
# log_K_sp values at 25 °C from Plummer-Busenberg 1982 (carbonates),
# Reardon-Beckie 1987 (sulfates), Blount 1977 (barite), Krumgalz-
# Pogorelsky-Pitzer 1995 (chlorides), Parkhurst-Appelo 2013 PHREEQC
# llnl.dat compilation. ΔH_rxn from Plummer-Busenberg 1982 (carbonates),
# CRC Handbook 75th ed. (sulfates, hydroxides), Blount 1977 (barite).

_MINERAL_DB: Dict[str, Mineral] = {
    # ------------------------------------------------------------------
    # Highly soluble chlorides (K_sp > 1)
    # ------------------------------------------------------------------
    "halite": Mineral(
        name="halite", formula="NaCl",
        cations={"Na+": 1}, anions={"Cl-": 1}, n_H2O=0,
        log_K_sp_25=1.582, delta_H_rxn=3900.0,
        binary_salt="NaCl"),
    "sylvite": Mineral(
        name="sylvite", formula="KCl",
        cations={"K+": 1}, anions={"Cl-": 1}, n_H2O=0,
        log_K_sp_25=0.960, delta_H_rxn=18600.0,
        binary_salt="KCl"),
    # ------------------------------------------------------------------
    # Sulfates (industrial scaling concerns)
    # ------------------------------------------------------------------
    "gypsum": Mineral(
        name="gypsum", formula="CaSO4·2H2O",
        cations={"Ca++": 1}, anions={"SO4--": 1}, n_H2O=2,
        log_K_sp_25=-4.581, delta_H_rxn=1100.0,
        binary_salt="CaSO4",
        log_K_sp_25_thermo=-4.75),    # calibrated against M-S 1966 with CaSO4°
    "anhydrite": Mineral(
        name="anhydrite", formula="CaSO4",
        cations={"Ca++": 1}, anions={"SO4--": 1}, n_H2O=0,
        log_K_sp_25=-4.36, delta_H_rxn=-18000.0,
        binary_salt="CaSO4",
        log_K_sp_25_thermo=-4.36),    # apparent ≈ thermo within 0.05 log units
    "barite": Mineral(
        name="barite", formula="BaSO4",
        cations={"Ba++": 1}, anions={"SO4--": 1}, n_H2O=0,
        log_K_sp_25=-9.97, delta_H_rxn=26000.0,
        binary_salt=None),
    "celestite": Mineral(
        name="celestite", formula="SrSO4",
        cations={"Sr++": 1}, anions={"SO4--": 1}, n_H2O=0,
        log_K_sp_25=-6.63, delta_H_rxn=-8700.0,
        binary_salt=None),
    "mirabilite": Mineral(
        name="mirabilite", formula="Na2SO4·10H2O",
        cations={"Na+": 2}, anions={"SO4--": 1}, n_H2O=10,
        log_K_sp_25=-1.114, delta_H_rxn=78000.0,
        binary_salt="Na2SO4"),
    "thenardite": Mineral(
        name="thenardite", formula="Na2SO4",
        cations={"Na+": 2}, anions={"SO4--": 1}, n_H2O=0,
        log_K_sp_25=-0.179, delta_H_rxn=-8000.0,
        binary_salt="Na2SO4"),
    "epsomite": Mineral(
        name="epsomite", formula="MgSO4·7H2O",
        cations={"Mg++": 1}, anions={"SO4--": 1}, n_H2O=7,
        log_K_sp_25=-1.881, delta_H_rxn=12000.0,
        binary_salt="MgSO4"),
    # ------------------------------------------------------------------
    # Carbonates (need pH from sour_water module for full speciation)
    # ------------------------------------------------------------------
    "calcite": Mineral(
        name="calcite", formula="CaCO3",
        cations={"Ca++": 1}, anions={"CO3--": 1}, n_H2O=0,
        log_K_sp_25=-8.48, delta_H_rxn=-10500.0,
        binary_salt=None),
    "aragonite": Mineral(
        name="aragonite", formula="CaCO3",
        cations={"Ca++": 1}, anions={"CO3--": 1}, n_H2O=0,
        log_K_sp_25=-8.34, delta_H_rxn=-11000.0,
        binary_salt=None),
    "dolomite": Mineral(
        name="dolomite", formula="CaMg(CO3)2",
        cations={"Ca++": 1, "Mg++": 1}, anions={"CO3--": 2}, n_H2O=0,
        log_K_sp_25=-17.09, delta_H_rxn=-37000.0,
        binary_salt=None),
    "magnesite": Mineral(
        name="magnesite", formula="MgCO3",
        cations={"Mg++": 1}, anions={"CO3--": 1}, n_H2O=0,
        log_K_sp_25=-7.83, delta_H_rxn=-25000.0,
        binary_salt=None),
    # ------------------------------------------------------------------
    # Hydroxides
    # ------------------------------------------------------------------
    "brucite": Mineral(
        name="brucite", formula="Mg(OH)2",
        cations={"Mg++": 1}, anions={"OH-": 2}, n_H2O=0,
        log_K_sp_25=-10.88, delta_H_rxn=-114000.0,
        binary_salt=None),
    "portlandite": Mineral(
        name="portlandite", formula="Ca(OH)2",
        cations={"Ca++": 1}, anions={"OH-": 2}, n_H2O=0,
        log_K_sp_25=-5.18, delta_H_rxn=-121000.0,
        binary_salt=None),
}


def lookup_mineral(name: str) -> Mineral:
    """Look up a Mineral by name (case-insensitive)."""
    key = name.lower()
    if key not in _MINERAL_DB:
        raise KeyError(
            f"Unknown mineral {name!r}. Available: "
            f"{sorted(_MINERAL_DB.keys())}")
    return _MINERAL_DB[key]


def list_minerals() -> List[str]:
    """Return alphabetical list of bundled mineral names."""
    return sorted(_MINERAL_DB.keys())


# =====================================================================
# Saturation index calculation
# =====================================================================

def saturation_index(mineral: Union[str, Mineral],
                       molalities: Dict[str, float],
                       gammas: Dict[str, float],
                       T: float = 298.15,
                       a_w: float = 1.0) -> float:
    """Compute the saturation index SI = log10(IAP / K_sp) of a mineral
    in an aqueous solution.

    Parameters
    ----------
    mineral : Mineral or str
        The mineral whose SI to compute.  If str, looked up from the
        bundled database.
    molalities : dict
        Ion molalities {ion_name: mol/kg}. Must contain the cations
        and anions of the mineral; missing entries default to 0
        (which gives SI = -inf).
    gammas : dict
        Activity coefficients {ion_name: γ_i}. Missing entries default
        to 1.0.
    T : float, default 298.15
        Temperature [K] for K_sp(T) evaluation.
    a_w : float, default 1.0
        Water activity (only used if mineral has waters of hydration).

    Returns
    -------
    float
        SI = log10(IAP) - log10(K_sp). Returns -inf if any required
        ion has zero molality (i.e. IAP = 0).

    Examples
    --------
    >>> from stateprop.electrolyte import MultiPitzerSystem
    >>> sys = MultiPitzerSystem.from_salts(["NaCl", "CaCl2", "Na2SO4"])
    >>> m = {"Na+": 1.0, "Ca++": 0.05, "Cl-": 1.05, "SO4--": 0.025}
    >>> g = sys.gammas(m)
    >>> a_w = sys.water_activity(m)
    >>> saturation_index("gypsum", m, g, a_w=a_w)
    -0.93   # undersaturated, gypsum will dissolve
    """
    if isinstance(mineral, str):
        mineral = lookup_mineral(mineral)

    log_IAP = 0.0
    for ion, nu in mineral.cations.items():
        m_i = molalities.get(ion, 0.0)
        if m_i <= 0.0:
            return -np.inf
        g_i = gammas.get(ion, 1.0)
        log_IAP += nu * np.log10(g_i * m_i)
    for ion, nu in mineral.anions.items():
        m_i = molalities.get(ion, 0.0)
        if m_i <= 0.0:
            return -np.inf
        g_i = gammas.get(ion, 1.0)
        log_IAP += nu * np.log10(g_i * m_i)
    if mineral.n_H2O > 0:
        if a_w <= 0.0:
            return -np.inf
        log_IAP += mineral.n_H2O * np.log10(a_w)

    return float(log_IAP - mineral.log_K_sp(T))


# =====================================================================
# Solubility in pure water (binary salts)
# =====================================================================

def solubility_in_water(mineral: Union[str, Mineral],
                          T: float = 298.15,
                          max_iter: int = 100,
                          tol: float = 1e-8,
                          relax: float = 0.5) -> float:
    """Equilibrium molality of a binary-salt mineral in pure water at T.

    Solves for the formula-unit solubility S such that:

        K_sp(T) = ν_M^ν_M · ν_A^ν_A · γ_pm(S, T)^ν · S^ν · a_w(S, T)^n

    by fixed-point iteration with damping. Activity coefficients and
    water activity come from PitzerModel (single-electrolyte) on the
    mineral's `binary_salt`.  At convergence, the ion molalities are
    m_M = ν_M · S and m_A = ν_A · S.

    Parameters
    ----------
    mineral : Mineral or str
        Must have `binary_salt` set (i.e. exist in the bundled Pitzer
        single-electrolyte DB).  For minerals like dolomite (ternary),
        use `MineralSystem.solubility` instead.
    T : float, default 298.15 K
        Temperature.
    max_iter : int, default 100
        Maximum iterations of the fixed-point solver.
    tol : float, default 1e-8
        Relative convergence tolerance on S.
    relax : float, default 0.5
        Damping for the update: S_new = relax · S_new + (1-relax) · S_old.
        For some retrograde-solubility cases (anhydrite at high T) the
        default damping prevents oscillation.

    Returns
    -------
    float
        Equilibrium formula-unit solubility S [mol/kg].

    Raises
    ------
    ValueError
        If the mineral does not have a binary_salt set (use
        ``MineralSystem.saturation_index`` instead for ternary minerals).

    Examples
    --------
    >>> solubility_in_water("halite", T=298.15)
    6.156   # mol/kg, well-known halite saturation at 25 °C

    >>> solubility_in_water("gypsum", T=298.15)
    0.0152  # mol/kg, ~2.07 g/kg CaSO4·2H2O
    """
    if isinstance(mineral, str):
        mineral = lookup_mineral(mineral)
    if mineral.binary_salt is None:
        raise ValueError(
            f"{mineral.name!r} is not a binary salt; "
            f"use MineralSystem for multi-component minerals")

    pitzer = PitzerModel(mineral.binary_salt)
    K_sp = mineral.K_sp(T)
    nu_M = sum(mineral.cations.values())
    nu_A = sum(mineral.anions.values())
    nu = nu_M + nu_A
    n_H2O = mineral.n_H2O

    # Closed-form factor: K_sp / (ν_M^ν_M · ν_A^ν_A)
    factor_const = K_sp / (nu_M ** nu_M * nu_A ** nu_A)

    # Initial guess: ideal (γ=1, a_w=1)
    S = factor_const ** (1.0 / nu)

    for _ in range(max_iter):
        try:
            gamma_pm = pitzer.gamma_pm(S, T)
            a_w = pitzer.water_activity(S, T)
            # Solve K_sp = factor · γ^ν · S^ν · a_w^n  for S:
            # S = (K_sp / (ν_M^ν_M · ν_A^ν_A · γ^ν · a_w^n))^(1/ν)
            denom = (gamma_pm ** nu) * (a_w ** n_H2O)
        except (OverflowError, FloatingPointError):
            # Pitzer parameters extrapolated beyond calibration envelope
            # (typical for 2:2 salts at S > 3 mol/kg or extreme T).
            raise RuntimeError(
                f"Pitzer model extrapolated beyond valid range for "
                f"{mineral.binary_salt!r} at S~{S:.2g} mol/kg, T={T:.1f} K. "
                f"Reduce T or use a different model.")
        if denom <= 0 or not np.isfinite(denom):
            raise RuntimeError(
                f"Pitzer activity products non-finite at S={S:.2g}, T={T:.1f} K")
        S_new = (factor_const / denom) ** (1.0 / nu)
        if abs(S_new - S) / S < tol:
            S = S_new
            break
        S = relax * S_new + (1.0 - relax) * S

    return float(S)


# =====================================================================
# MineralSystem: convenience wrapper for batch SI calculation
# =====================================================================

class MineralSystem:
    """Wrap a MultiPitzerSystem with a list of minerals for batch
    saturation-index calculation.

    Examples
    --------
    >>> from stateprop.electrolyte import MultiPitzerSystem
    >>> brine = MultiPitzerSystem.from_salts(
    ...     ["NaCl", "CaCl2", "Na2SO4"])
    >>> ms = MineralSystem(brine, ["halite", "gypsum", "anhydrite",
    ...                             "thenardite"])
    >>> m = {"Na+": 1.5, "Ca++": 0.1, "Cl-": 1.7, "SO4--": 0.05}
    >>> ms.saturation_indices(m, T=323.15)
    {'halite': -0.91, 'gypsum': -0.34, 'anhydrite': -0.31,
     'thenardite': -0.71}
    >>> ms.scale_risks(m, T=323.15)   # only minerals at risk of precipitating
    {}                               # all undersaturated
    """

    def __init__(self,
                  pitzer: MultiPitzerSystem,
                  minerals: Sequence[Union[str, Mineral]]):
        self.pitzer = pitzer
        self.minerals: List[Mineral] = []
        for m in minerals:
            self.minerals.append(
                lookup_mineral(m) if isinstance(m, str) else m)

    def saturation_indices(self,
                              molalities: Dict[str, float],
                              T: float = 298.15) -> Dict[str, float]:
        """Compute SI for every mineral in the system.

        Returns a dict {mineral_name: SI}.  -inf for minerals whose
        component ions are not all present in `molalities`.
        """
        gammas = self.pitzer.gammas(molalities, T)
        a_w = self.pitzer.water_activity(molalities, T)
        return {
            m.name: saturation_index(m, molalities, gammas, T, a_w)
            for m in self.minerals
        }

    def scale_risks(self,
                       molalities: Dict[str, float],
                       T: float = 298.15,
                       threshold: float = 0.0) -> Dict[str, float]:
        """Return only minerals with SI > threshold (potential scale).

        Parameters
        ----------
        threshold : float, default 0.0
            SI threshold above which a mineral is reported.  Use a
            small positive value (e.g. 0.3) to filter out minerals
            that are only marginally supersaturated.
        """
        SI = self.saturation_indices(molalities, T)
        return {name: si for name, si in SI.items()
                 if si > threshold and np.isfinite(si)}
