"""Multi-electrolyte Pitzer model with mixing terms (v0.9.98+).

Extends the single-electrolyte Pitzer framework (``pitzer.py``) to
multi-electrolyte aqueous solutions. The full Pitzer 1991 expression
for the activity coefficient of cation M in a mixture is:

    ln γ_M = z_M² F + Σ_a m_a [2 B_Ma + Z C_Ma]
            + Σ_c m_c [2 Φ_Mc + Σ_a m_a ψ_Mca]
            + Σ_a Σ_a' < m_a m_a' ψ_Maa'
            + |z_M| Σ_c Σ_a m_c m_a C_ca

Analogous form for anion X.  The new pieces vs single-electrolyte:

* **θ_cc'**   — cation-cation interaction (e.g. Na⁺/K⁺)
* **θ_aa'**   — anion-anion interaction (e.g. Cl⁻/SO₄²⁻)
* **ψ_cc'a**  — ternary cation-cation-anion
* **ψ_caa'**  — ternary cation-anion-anion
* **E-θ_ij(I)** — unsymmetric mixing function for different-charge
  ion pairs (Na⁺/Mg²⁺, Na⁺/Ca²⁺, etc.). Computed via the
  Plummer-Parkhurst 1988 closed-form approximation to Pitzer's
  J_0 integral (active in v0.9.99+; was zero in v0.9.98).

The unsymmetric mixing E-θ is computed via the Plummer-Parkhurst 1988
closed-form fit to Pitzer's J_0 integral:

    J_0(x) = x / [4 + 4.581·x^0.7237·exp(-0.0120·x^(4/3))]

This is the form used in PHREEQC and related geochemistry codes.
Accurate to <2% over 0 < x < 100 (the practical range for typical
brine ionic strengths I < 6 mol/kg).

Parameter sources for bundled mixtures (25 °C):

* **NaCl / KCl / CaCl₂ / MgCl₂ / Na₂SO₄ / K₂SO₄ / MgSO₄ / CaSO₄
  ("seawater system")**: Harvie-Møller-Weare 1984 (Geochim.
  Cosmochim. Acta 48, 723), the canonical reference for brine-system
  thermodynamics. Validated against Pitzer-Kim 1974, Filippov 1988,
  Pabalan-Pitzer 1987.

References
----------
* Pitzer, K. S. (1991). *Activity Coefficients in Electrolyte
  Solutions* (2nd ed.), Ch. 3 §3.5.
* Pitzer, K. S. (1975). Thermodynamics of electrolytes V. Effects
  of higher-order electrostatic terms. J. Solution Chem. 4, 249.
* Plummer, L. N., Parkhurst, D. L., Fleming, G. W., Dunkle, S. A.
  (1988). A computer program incorporating Pitzer's equations for
  calculation of geochemical reactions in brines. USGS Water-
  Resources Investigations Report 88-4153.
* Harvie, C. E., Møller, N., Weare, J. H. (1984). The prediction of
  mineral solubilities in natural waters: the Na-K-Mg-Ca-H-Cl-SO₄-
  OH-HCO₃-CO₃-CO₂-H₂O system to high ionic strengths at 25 °C.
* Pitzer, K. S., Kim, J. J. (1974). Thermodynamics of electrolytes
  IV. Activity and osmotic coefficients for mixed electrolytes.
  J. Am. Chem. Soc. 96, 5701.
* Robinson, R. A., Wood, R. H. (1972). Activity coefficients of NaCl
  in aqueous KCl mixtures. J. Solution Chem. 1, 481.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np

from .pitzer import (
    PitzerSalt, lookup_salt, _g, _g_prime,
)
from .utils import debye_huckel_A, _MW_WATER


# =====================================================================
# Mixing parameters (θ, ψ) with T-dependence
# =====================================================================
#
# Pitzer mixing terms are stored as `MixingParam` named tuples with
# a value at 298.15 K and an optional first derivative dP/dT.  The
# parameter at temperature T is:
#
#     P(T) = P_25 + (dP/dT) · (T - 298.15)
#
# T-derivatives are bundled where reliable published values exist:
#   * Møller 1988: Na-Ca-Cl-SO4-H2O system to 200 °C
#       J. Phys. Chem. 92, 4660. T-derivatives for θ_NaCa, ψ_NaCaCl,
#       ψ_NaCaSO4, ψ_CaClSO4
#   * Greenberg-Møller 1989: Na-K-Ca-Cl-SO4 to 250 °C
#       Geochim. Cosmochim. Acta 53, 2503
#   * Pabalan-Pitzer 1987: Na-K-Mg-Cl-SO4-OH-HCO3 system
#       Geochim. Cosmochim. Acta 51, 2429.  ψ_NaKCl, ψ_NaKSO4
#
# For mixing terms without published T-derivatives, dP/dT defaults
# to 0 (i.e. assume value is constant at the 25 °C value across the
# range of interest — a reasonable approximation for most pairs over
# 0-100 °C, since Pitzer found the **binary β** derivatives dominate
# the T-dependence of mixed-electrolyte solutions).
#
# Conventions:
#   * Keys are sorted-tuple pairs/triples to avoid (a,b) vs (b,a)
#     ambiguity: theta[(min(c1,c2), max(c1,c2))] = MixingParam(...)
#   * User-supplied overrides may be plain floats (treated as
#     T-independent at the given value) or MixingParam objects.

from typing import NamedTuple, Union


class MixingParam(NamedTuple):
    """Pitzer mixing parameter with optional linear T-dependence.

    P(T) = value_25 + dvalue_dT · (T - 298.15)

    Attributes
    ----------
    value_25 : float
        Parameter value at the reference temperature 298.15 K.
    dvalue_dT : float
        First derivative dP/dT [parameter-units / K]. Default 0.

    Examples
    --------
    >>> # T-independent parameter (default behavior)
    >>> p = MixingParam(0.07)
    >>> p.at_T(298.15)
    0.07
    >>> p.at_T(348.15)
    0.07

    >>> # T-dependent parameter
    >>> p = MixingParam(0.07, dvalue_dT=4.09e-4)
    >>> p.at_T(298.15)
    0.07
    >>> round(p.at_T(348.15), 6)
    0.090450
    """
    value_25: float
    dvalue_dT: float = 0.0

    def at_T(self, T: float) -> float:
        """Evaluate the parameter at temperature T [K]."""
        return self.value_25 + self.dvalue_dT * (T - 298.15)


def _to_mixing_param(v: Union[float, MixingParam]) -> MixingParam:
    """Coerce float → MixingParam(value, 0); pass through if already one."""
    if isinstance(v, MixingParam):
        return v
    return MixingParam(float(v), 0.0)


def _csort(*items):
    """Return tuple with items sorted alphabetically (for canonical keys)."""
    return tuple(sorted(items))


# Cation-cation θ: same-sign mixing parameters (no E-θ if same charge)
_THETA_CC = {
    _csort("Na+", "K+"):    MixingParam(-0.012),    # Pitzer-Kim 1974; ~T-indep
    _csort("Na+", "Mg++"):  MixingParam( 0.070),    # H-M-W 1984
    # Na-Ca: Møller 1988 dθ/dT = +4.09e-4 K⁻¹
    _csort("Na+", "Ca++"):  MixingParam( 0.070, dvalue_dT=4.09e-4),
    _csort("Na+", "H+"):    MixingParam( 0.036),    # Pitzer 1991
    _csort("K+", "Mg++"):   MixingParam( 0.000),    # H-M-W 1984
    _csort("K+", "Ca++"):   MixingParam( 0.032),    # H-M-W 1984
    _csort("K+", "H+"):     MixingParam( 0.005),    # Pitzer 1991
    _csort("Mg++", "Ca++"): MixingParam( 0.007),    # H-M-W 1984
    _csort("Mg++", "H+"):   MixingParam( 0.100),    # Pitzer 1991
    _csort("Ca++", "H+"):   MixingParam( 0.092),    # Pitzer 1991
}

# Anion-anion θ
_THETA_AA = {
    _csort("Cl-", "SO4--"):    MixingParam( 0.020),  # H-M-W 1984
    _csort("Cl-", "HCO3-"):    MixingParam( 0.030),  # H-M-W 1984
    _csort("Cl-", "CO3--"):    MixingParam(-0.020),  # H-M-W 1984
    _csort("Cl-", "OH-"):      MixingParam(-0.050),  # Pitzer 1991
    _csort("SO4--", "HCO3-"):  MixingParam( 0.010),  # H-M-W 1984
    _csort("SO4--", "CO3--"):  MixingParam( 0.020),  # H-M-W 1984
    _csort("SO4--", "OH-"):    MixingParam(-0.013),  # Pitzer 1991
    _csort("HCO3-", "CO3--"):  MixingParam( 0.000),  # H-M-W 1984
}

# Ternary cation-cation-anion ψ_cc'a
_PSI_CCA = {
    # Na-K-Cl: Pabalan-Pitzer 1987 dψ/dT = -1.91e-5 K⁻¹
    (*_csort("Na+", "K+"),   "Cl-"):
        MixingParam(-0.0018, dvalue_dT=-1.91e-5),
    # Na-K-SO4: Pabalan-Pitzer 1987 small but nonzero T-derivative
    (*_csort("Na+", "K+"),   "SO4--"):
        MixingParam(-0.010,  dvalue_dT=-1.40e-4),
    (*_csort("Na+", "Mg++"), "Cl-"):    MixingParam(-0.012),
    (*_csort("Na+", "Mg++"), "SO4--"):  MixingParam(-0.015),
    # Na-Ca-Cl: Møller 1988 dψ/dT = -2.6e-4 K⁻¹
    (*_csort("Na+", "Ca++"), "Cl-"):
        MixingParam(-0.014,  dvalue_dT=-2.60e-4),
    (*_csort("Na+", "Ca++"), "SO4--"):  MixingParam(-0.055),
    (*_csort("Na+", "H+"),   "Cl-"):    MixingParam(-0.004),
    (*_csort("K+", "Mg++"),  "Cl-"):    MixingParam(-0.022),
    (*_csort("K+", "Mg++"),  "SO4--"):  MixingParam(-0.048),
    (*_csort("K+", "Ca++"),  "Cl-"):    MixingParam(-0.025),
    (*_csort("K+", "H+"),    "Cl-"):    MixingParam(-0.011),
    (*_csort("Mg++", "Ca++"),"Cl-"):    MixingParam(-0.012),
    (*_csort("Mg++", "H+"),  "Cl-"):    MixingParam(-0.011),
    (*_csort("Ca++", "H+"),  "Cl-"):    MixingParam(-0.015),
}

# Ternary cation-anion-anion ψ_caa'
_PSI_CAA = {
    ("Na+",  *_csort("Cl-", "SO4--")):    MixingParam( 0.0014),
    ("Na+",  *_csort("Cl-", "HCO3-")):    MixingParam(-0.015),
    ("Na+",  *_csort("Cl-", "CO3--")):    MixingParam( 0.0085),
    ("Na+",  *_csort("Cl-", "OH-")):      MixingParam(-0.006),
    ("Na+",  *_csort("SO4--", "HCO3-")):  MixingParam( 0.0),
    ("Na+",  *_csort("SO4--", "CO3--")):  MixingParam(-0.005),
    ("K+",   *_csort("Cl-", "SO4--")):    MixingParam(-0.0),
    ("K+",   *_csort("Cl-", "HCO3-")):    MixingParam(-0.008),
    ("Mg++", *_csort("Cl-", "SO4--")):    MixingParam(-0.008),
    # Ca-Cl-SO4: Møller 1988 small positive dψ/dT
    ("Ca++", *_csort("Cl-", "SO4--")):
        MixingParam(-0.018, dvalue_dT=+1.5e-5),
}


def _theta_pair(i: str, j: str, T: float = 298.15) -> float:
    """Look up θ_ij for ions i, j at T (returns 0 if not in database)."""
    if i == j:
        return 0.0
    p = _THETA_CC.get(_csort(i, j))
    if p is None:
        p = _THETA_AA.get(_csort(i, j))
    if p is None:
        return 0.0
    return p.at_T(T)


def _psi_triple(c1: str, c2: str, a: str,
                  cc_pair: bool = True, T: float = 298.15) -> float:
    """Look up ψ for a (c1, c2, a) or (c, a1, a2) triple at T.

    cc_pair=True: cation-cation-anion ψ_cc'a
    cc_pair=False: cation-anion-anion ψ_caa'
    """
    if cc_pair:
        if c1 == c2:
            return 0.0
        p = _PSI_CCA.get((*_csort(c1, c2), a))
    else:
        if c2 == a:
            return 0.0
        p = _PSI_CAA.get((c1, *_csort(c2, a)))
    return p.at_T(T) if p is not None else 0.0


# =====================================================================
# Unsymmetric mixing: E-θ_ij(I) for different-charge ion pairs
# =====================================================================
#
# Pitzer 1975 / Harvie 1981 form. For ions i,j of charges z_i, z_j:
#
#   E-θ_ij(I) = (z_i z_j / 4 I) [J_0(x_ij) - 0.5·J_0(x_ii) - 0.5·J_0(x_jj)]
#   E-θ'_ij(I) = -E-θ_ij/I + (z_i z_j / 8 I²) [
#                  x_ij J_1(x_ij) - 0.5·x_ii J_1(x_ii) - 0.5·x_jj J_1(x_jj)
#                ]
#
# where x_ij = 6·z_i·z_j·A_φ·√I, J_n are Pitzer's J integrals, and
# J_1(x) = x · dJ_0/dx.
#
# When z_i = z_j (same-charge ions): x_ii = x_jj = x_ij so the bracket
# is zero and E-θ = E-θ' = 0; same-charge mixing needs only the
# constant θ.
#
# **Implementation: Plummer-Parkhurst 1988 explicit approximation.**
# Source: Plummer, L. N., Parkhurst, D. L., Fleming, G. W., Dunkle,
# S. A. (1988). A computer program incorporating Pitzer's equations
# for calculation of geochemical reactions in brines. USGS
# Water-Resources Investigations Report 88-4153.
#
# Their explicit closed-form J_0:
#
#     J_0(x) = x / [4 + 4.581·x^0.7237·exp(-0.0120·x^(4/3))]
#
# This is the form used in PHREEQC and related geochemistry codes.
# Accurate to <2% over 0 < x < 100 (the practical range for typical
# brine ionic strengths I < 6 mol/kg).
#
# Constants in the bracket (J_0(x_ij) - 0.5 J_0(x_ii) - 0.5 J_0(x_jj))
# cancel by construction, so the difference between this approximation
# and the exact J_0 (which has J_0(0) = -1 rather than 0) doesn't
# affect the E-θ result.

def _J0(x: float) -> float:
    """Pitzer's J_0 integral, Plummer-Parkhurst 1988 explicit form.

    J_0(x) = x / [4 + 4.581·x^0.7237·exp(-0.0120·x^(4/3))]

    Returns 0 for x ≤ 0. Has the correct asymptote J_0(x) → x/4 for
    large x, and J_0(x) → 0 for small x (constants cancel in the
    E-θ bracket so this is equivalent to the exact J_0(0) = -1).
    """
    if x <= 0.0:
        return 0.0
    x_43 = x ** (4.0 / 3.0)
    # Guard against overflow at x = 0 (shouldn't happen given check above)
    return x / (4.0 + 4.581 * (x ** 0.7237) * np.exp(-0.0120 * x_43))


def _J1(x: float) -> float:
    """Pitzer's J_1(x) = x · dJ_0/dx, by central finite difference.

    Uses log-step finite difference for numerical stability across the
    full x range (the analytical derivative of the Plummer-Parkhurst
    form has many terms; finite difference is simpler and accurate to
    ~5e-5 with eps=1e-4 relative step).
    """
    if x <= 0.0:
        return 0.0
    eps = 1e-4
    dh = eps * x
    return x * (_J0(x + dh) - _J0(x - dh)) / (2.0 * dh)


def E_theta(z_i: int, z_j: int, I: float, T: float) -> Tuple[float, float]:
    """Unsymmetric mixing function E-θ_ij(I) and its I-derivative E-θ'_ij(I).

    Returns (E_theta, E_theta_prime). Both are zero when |z_i| = |z_j|
    (same-magnitude charges don't have asymmetric mixing).

    Uses the Pitzer 1975 expressions with J_0 from the Plummer-Parkhurst
    1988 closed-form approximation.
    """
    if abs(z_i) == abs(z_j):
        # Same charge magnitude → x_ii = x_jj = x_ij → bracket is 0
        return (0.0, 0.0)
    if I < 1e-6:
        # Pitzer's E-θ formula has a 1/√I divergence as I→0; below
        # 1e-6 mol/kg we treat the system as effectively pure water
        return (0.0, 0.0)

    A = debye_huckel_A(T)
    sI = np.sqrt(I)
    x_ij = 6.0 * abs(z_i * z_j) * A * sI
    x_ii = 6.0 * z_i * z_i * A * sI
    x_jj = 6.0 * z_j * z_j * A * sI

    J0_ij = _J0(x_ij)
    J0_ii = _J0(x_ii)
    J0_jj = _J0(x_jj)
    J1_ij = _J1(x_ij)
    J1_ii = _J1(x_ii)
    J1_jj = _J1(x_jj)

    # Pitzer 1975 Eq. 39 (using |z_i z_j| since z's are positive for cations
    # and negative for anions; for same-sign ions these are the magnitudes)
    zz = abs(z_i * z_j)
    e_theta = (zz / (4.0 * I)) * (
        J0_ij - 0.5 * J0_ii - 0.5 * J0_jj)

    # E-θ' from Pitzer 1975 Eq. 40
    e_theta_prime = (-e_theta / I
                       + (zz / (8.0 * I * I)) * (
                           x_ij * J1_ij
                           - 0.5 * x_ii * J1_ii
                           - 0.5 * x_jj * J1_jj))

    return (float(e_theta), float(e_theta_prime))


# =====================================================================
# MultiPitzerSystem class
# =====================================================================

@dataclass
class IonInfo:
    """Single ion species in a multi-electrolyte system."""
    name: str       # Canonical name, e.g. "Na+", "Mg++", "Cl-", "SO4--"
    charge: int     # Signed integer charge


class MultiPitzerSystem:
    """Multi-electrolyte Pitzer model for arbitrary cation/anion mixtures.

    Parameters
    ----------
    cations, anions : list of (name, charge)
        Ion species in the mixture.  Use canonical names matching
        the bundled mixing-parameter tables: "Na+", "K+", "H+",
        "Mg++", "Ca++"; "Cl-", "SO4--", "HCO3-", "CO3--", "OH-".
    binary_pairs : dict, optional
        Map (cation, anion) → PitzerSalt with binary β⁰, β¹, β², C^φ.
        If omitted, the constructor tries to look up a canonical
        binary salt from the single-electrolyte database
        (Na+/Cl- → NaCl; Mg++/SO4-- → MgSO4; etc.).
    theta_cc, theta_aa : dict, optional
        Override or supply additional θ-pair values.
    psi_cca, psi_caa : dict, optional
        Override or supply additional ψ-triple values.

    Examples
    --------
    >>> sys = MultiPitzerSystem.from_salts(["NaCl", "KCl"])
    >>> g = sys.gammas({"Na+": 0.5, "K+": 0.5, "Cl-": 1.0}, T=298.15)
    >>> g["Na+"], g["K+"], g["Cl-"]
    (0.6443, 0.5945, 0.6168)

    >>> sys = MultiPitzerSystem.seawater()
    >>> # Standard seawater concentrations [mol/kg], approx
    >>> m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
    ...       "Cl-": 0.566, "SO4--": 0.0293}
    >>> sys.osmotic_coefficient(m, T=298.15)
    """

    BPITZER = 1.2

    # Mapping from (cation, anion) pair → bundled binary salt name
    # Used by from_salts to auto-pull binary β values
    _PAIR_TO_SALT = {
        ("Na+", "Cl-"):    "NaCl",
        ("K+",  "Cl-"):    "KCl",
        ("Li+", "Cl-"):    "LiCl",
        ("Na+", "Br-"):    "NaBr",
        ("K+",  "Br-"):    "KBr",
        ("Na+", "OH-"):    "NaOH",
        ("K+",  "OH-"):    "KOH",
        ("H+",  "Cl-"):    "HCl",
        ("H+",  "Br-"):    "HBr",
        ("Na+", "NO3-"):   "NaNO3",
        ("Na+", "ClO4-"):  "NaClO4",
        ("Ca++", "Cl-"):   "CaCl2",
        ("Mg++", "Cl-"):   "MgCl2",
        ("Ba++", "Cl-"):   "BaCl2",
        ("Na+", "SO4--"):  "Na2SO4",
        ("K+",  "SO4--"):  "K2SO4",
        ("Mg++", "SO4--"): "MgSO4",
        ("Ca++", "SO4--"): "CaSO4",
        ("Cu++", "SO4--"): "CuSO4",
    }

    def __init__(
        self,
        cations: Sequence[Tuple[str, int]],
        anions: Sequence[Tuple[str, int]],
        binary_pairs: Optional[Dict[Tuple[str, str], PitzerSalt]] = None,
        theta_cc: Optional[Dict] = None,
        theta_aa: Optional[Dict] = None,
        psi_cca: Optional[Dict] = None,
        psi_caa: Optional[Dict] = None,
    ):
        self.cations = [IonInfo(n, z) for n, z in cations]
        self.anions = [IonInfo(n, z) for n, z in anions]
        if any(z <= 0 for n, z in cations):
            raise ValueError("Cations must have positive charge")
        if any(z >= 0 for n, z in anions):
            raise ValueError("Anions must have negative charge")

        # Auto-populate binary pairs from bundled database if not given
        if binary_pairs is None:
            binary_pairs = {}
            for c in self.cations:
                for a in self.anions:
                    key = (c.name, a.name)
                    salt_name = self._PAIR_TO_SALT.get(key)
                    if salt_name is not None:
                        binary_pairs[key] = lookup_salt(salt_name)
        self.binary_pairs = binary_pairs

        # Mixing terms: start from defaults, update with user overrides.
        # Plain-float user inputs are coerced to MixingParam (T-independent).
        self.theta_cc = dict(_THETA_CC)
        if theta_cc:
            for k, v in theta_cc.items():
                self.theta_cc[_csort(*k)] = _to_mixing_param(v)
        self.theta_aa = dict(_THETA_AA)
        if theta_aa:
            for k, v in theta_aa.items():
                self.theta_aa[_csort(*k)] = _to_mixing_param(v)
        self.psi_cca = dict(_PSI_CCA)
        if psi_cca:
            for k, v in psi_cca.items():
                # k = (c1, c2, a) — re-sort the cations
                c1, c2, a = k
                self.psi_cca[(*_csort(c1, c2), a)] = _to_mixing_param(v)
        self.psi_caa = dict(_PSI_CAA)
        if psi_caa:
            for k, v in psi_caa.items():
                c, a1, a2 = k
                self.psi_caa[(c, *_csort(a1, a2))] = _to_mixing_param(v)

        # Cache for ion properties
        self._charges = {c.name: c.charge for c in self.cations}
        self._charges.update({a.name: a.charge for a in self.anions})

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_salts(cls, salt_names: Sequence[str]) -> "MultiPitzerSystem":
        """Build a multi-electrolyte system from a list of salt names.

        Each salt is looked up in the single-electrolyte database to
        get binary β values; mixing terms are pulled from the bundled
        H-M-W 1984 / Pitzer 1991 tables for any (cat, an), (cat, cat),
        (an, an) combinations that exist there.

        Parameters
        ----------
        salt_names : list of str
            E.g. ["NaCl", "KCl"] or ["NaCl", "Na2SO4", "MgCl2"].

        Returns
        -------
        MultiPitzerSystem
        """
        # Decompose each salt into cation and anion
        cations_seen = {}   # name → charge
        anions_seen = {}
        binary_pairs = {}
        for name in salt_names:
            salt = lookup_salt(name)
            cation_name, anion_name = cls._decode_salt(name)
            cations_seen[cation_name] = salt.z_M
            anions_seen[anion_name] = salt.z_X
            binary_pairs[(cation_name, anion_name)] = salt
        cations = [(n, z) for n, z in cations_seen.items()]
        anions = [(n, z) for n, z in anions_seen.items()]
        return cls(cations=cations, anions=anions,
                   binary_pairs=binary_pairs)

    @staticmethod
    def _decode_salt(name: str) -> Tuple[str, str]:
        """Decompose a salt name into (cation, anion).  Case sensitive."""
        # Hard-coded for the bundled set
        decoder = {
            "NaCl":   ("Na+", "Cl-"),
            "KCl":    ("K+",  "Cl-"),
            "LiCl":   ("Li+", "Cl-"),
            "NaBr":   ("Na+", "Br-"),
            "KBr":    ("K+",  "Br-"),
            "NaOH":   ("Na+", "OH-"),
            "KOH":    ("K+",  "OH-"),
            "HCl":    ("H+",  "Cl-"),
            "HBr":    ("H+",  "Br-"),
            "NaNO3":  ("Na+", "NO3-"),
            "NaClO4": ("Na+", "ClO4-"),
            "CaCl2":  ("Ca++","Cl-"),
            "MgCl2":  ("Mg++","Cl-"),
            "BaCl2":  ("Ba++","Cl-"),
            "Na2SO4": ("Na+", "SO4--"),
            "K2SO4":  ("K+",  "SO4--"),
            "MgSO4":  ("Mg++","SO4--"),
            "CaSO4":  ("Ca++","SO4--"),
            "CaSO4_Moeller": ("Ca++","SO4--"),
            "CuSO4":  ("Cu++","SO4--"),
        }
        if name not in decoder:
            raise ValueError(f"Cannot decode salt name {name!r} into "
                              f"cation/anion. Add to _decode_salt or "
                              f"build the system manually.")
        return decoder[name]

    @classmethod
    def seawater(cls) -> "MultiPitzerSystem":
        """Convenience constructor for the seawater system:
        Na⁺, K⁺, Mg²⁺, Ca²⁺ + Cl⁻, SO₄²⁻.
        Includes all H-M-W 1984 mixing parameters."""
        return cls.from_salts([
            "NaCl", "KCl", "MgCl2", "CaCl2",
            "Na2SO4", "K2SO4", "MgSO4",
        ])

    # ------------------------------------------------------------------
    # Helper accessors
    # ------------------------------------------------------------------

    def ionic_strength(self, m: Dict[str, float]) -> float:
        """I = ½·Σ m_i·z_i²."""
        I = 0.0
        for ion, mol in m.items():
            z = self._charges.get(ion, 0)
            I += mol * z * z
        return 0.5 * I

    def _Z(self, m: Dict[str, float]) -> float:
        """Z = Σ_i m_i |z_i|."""
        return sum(mol * abs(self._charges.get(ion, 0))
                    for ion, mol in m.items())

    def _binary_B(self, c: str, a: str, I: float, T: float) -> float:
        """B_ca(I, T) = β⁰ + β¹·g(α₁√I) + β²·g(α₂√I) for the (c, a) pair."""
        salt = self.binary_pairs.get((c, a))
        if salt is None:
            return 0.0
        s = salt.at_T(T)
        sI = np.sqrt(I)
        out = s.beta_0 + s.beta_1 * _g(s.alpha_1 * sI)
        if s.beta_2 != 0.0:
            out += s.beta_2 * _g(s.alpha_2 * sI)
        return out

    def _binary_B_prime(self, c: str, a: str, I: float, T: float) -> float:
        """B'_ca(I, T) = β¹·g'(α₁√I)/I + β²·g'(α₂√I)/I."""
        salt = self.binary_pairs.get((c, a))
        if salt is None or I < 1e-15:
            return 0.0
        s = salt.at_T(T)
        sI = np.sqrt(I)
        out = s.beta_1 * _g_prime(s.alpha_1 * sI) / I
        if s.beta_2 != 0.0:
            out += s.beta_2 * _g_prime(s.alpha_2 * sI) / I
        return out

    def _binary_B_phi(self, c: str, a: str, I: float, T: float) -> float:
        """B^φ_ca(I, T) = β⁰ + β¹·exp(-α₁√I) + β²·exp(-α₂√I)."""
        salt = self.binary_pairs.get((c, a))
        if salt is None:
            return 0.0
        s = salt.at_T(T)
        sI = np.sqrt(I)
        out = s.beta_0 + s.beta_1 * np.exp(-s.alpha_1 * sI)
        if s.beta_2 != 0.0:
            out += s.beta_2 * np.exp(-s.alpha_2 * sI)
        return out

    def _binary_C(self, c: str, a: str) -> float:
        """C_ca = C^φ / (2√|z_c·z_a|)."""
        salt = self.binary_pairs.get((c, a))
        if salt is None:
            return 0.0
        return salt.C_phi / (2.0 * np.sqrt(abs(salt.z_M * salt.z_X)))

    def _theta(self, i: str, j: str, T: float = 298.15) -> float:
        """θ_ij(T) for ion pair (constant part, before E-θ correction).

        Evaluates the bundled MixingParam at T using its linear
        T-derivative (defaults to 0 for parameters without published
        T-dependence).
        """
        if i == j:
            return 0.0
        z_i = self._charges.get(i, 0)
        z_j = self._charges.get(j, 0)
        # Look up in cation-cation or anion-anion table
        if z_i > 0 and z_j > 0:
            p = self.theta_cc.get(_csort(i, j))
        elif z_i < 0 and z_j < 0:
            p = self.theta_aa.get(_csort(i, j))
        else:
            return 0.0   # cation-anion: no θ (binary B handles them)
        return p.at_T(T) if p is not None else 0.0

    def _Phi(self, i: str, j: str, I: float, T: float) -> float:
        """Φ_ij(I, T) = θ_ij(T) + E-θ_ij(I, T)."""
        if i == j:
            return 0.0
        theta = self._theta(i, j, T)
        z_i = self._charges.get(i, 0)
        z_j = self._charges.get(j, 0)
        e_theta, _ = E_theta(z_i, z_j, I, T)
        return theta + e_theta

    def _Phi_prime(self, i: str, j: str, I: float, T: float) -> float:
        """Φ'_ij(I, T) = E-θ'_ij(I, T)  (since θ_ij(T) is I-independent)."""
        if i == j:
            return 0.0
        z_i = self._charges.get(i, 0)
        z_j = self._charges.get(j, 0)
        _, e_theta_prime = E_theta(z_i, z_j, I, T)
        return e_theta_prime

    def _Phi_phi(self, i: str, j: str, I: float, T: float) -> float:
        """Φ^φ_ij(I, T) = θ(T) + E-θ(I, T) + I·E-θ'(I, T)."""
        if i == j:
            return 0.0
        theta = self._theta(i, j, T)
        z_i = self._charges.get(i, 0)
        z_j = self._charges.get(j, 0)
        e_theta, e_theta_prime = E_theta(z_i, z_j, I, T)
        return theta + e_theta + I * e_theta_prime

    def _psi_cca(self, c1: str, c2: str, a: str,
                   T: float = 298.15) -> float:
        """ψ_cc'a(T) ternary cation-cation-anion mixing."""
        if c1 == c2:
            return 0.0
        p = self.psi_cca.get((*_csort(c1, c2), a))
        return p.at_T(T) if p is not None else 0.0

    def _psi_caa(self, c: str, a1: str, a2: str,
                   T: float = 298.15) -> float:
        """ψ_caa'(T) ternary cation-anion-anion mixing."""
        if a1 == a2:
            return 0.0
        p = self.psi_caa.get((c, *_csort(a1, a2)))
        return p.at_T(T) if p is not None else 0.0

    # ------------------------------------------------------------------
    # F function (long-range + ion-ion derivatives)
    # ------------------------------------------------------------------

    def _F(self, m: Dict[str, float], I: float, T: float) -> float:
        """Pitzer F function (Pitzer 1991 Eq. 3.59)."""
        A = debye_huckel_A(T)
        b = self.BPITZER
        sI = np.sqrt(I)
        # Long-range Debye-Hückel
        F = -A * (sI / (1.0 + b * sI) + (2.0 / b) * np.log(1.0 + b * sI))
        # Σ_c Σ_a m_c m_a B'_ca
        for c in self.cations:
            mc = m.get(c.name, 0.0)
            if mc == 0:
                continue
            for a in self.anions:
                ma = m.get(a.name, 0.0)
                if ma == 0:
                    continue
                F += mc * ma * self._binary_B_prime(c.name, a.name, I, T)
        # Σ_c<c' m_c m_c' Φ'_cc'
        for i, c1 in enumerate(self.cations):
            for c2 in self.cations[i+1:]:
                mc1 = m.get(c1.name, 0.0)
                mc2 = m.get(c2.name, 0.0)
                if mc1 == 0 or mc2 == 0:
                    continue
                F += mc1 * mc2 * self._Phi_prime(c1.name, c2.name, I, T)
        # Σ_a<a' m_a m_a' Φ'_aa'
        for i, a1 in enumerate(self.anions):
            for a2 in self.anions[i+1:]:
                ma1 = m.get(a1.name, 0.0)
                ma2 = m.get(a2.name, 0.0)
                if ma1 == 0 or ma2 == 0:
                    continue
                F += ma1 * ma2 * self._Phi_prime(a1.name, a2.name, I, T)
        return F

    # ------------------------------------------------------------------
    # Activity coefficients
    # ------------------------------------------------------------------

    def gammas(self, m: Dict[str, float],
                T: float = 298.15) -> Dict[str, float]:
        """Activity coefficients for every ion in the mixture.

        Returns a dict {ion_name: γ_i} for each cation and anion present
        in m.  Uses the full Pitzer 1991 multi-electrolyte expressions
        with bundled mixing parameters.

        Parameters
        ----------
        m : dict
            Molality of each ion {name: mol/kg}.
        T : float
            Temperature [K].

        Returns
        -------
        dict
            {ion_name: γ_i} for each ion.
        """
        I = self.ionic_strength(m)
        Z = self._Z(m)
        F = self._F(m, I, T)
        out = {}
        # Cations
        for M in self.cations:
            mM = m.get(M.name, 0.0)
            if mM == 0:
                continue
            ln_g = M.charge ** 2 * F
            # Σ_a m_a [2 B_Ma + Z C_Ma]
            for a in self.anions:
                ma = m.get(a.name, 0.0)
                if ma == 0:
                    continue
                B = self._binary_B(M.name, a.name, I, T)
                C = self._binary_C(M.name, a.name)
                ln_g += ma * (2.0 * B + Z * C)
            # Σ_c m_c [2 Φ_Mc + Σ_a m_a ψ_Mca]
            for c in self.cations:
                if c.name == M.name:
                    continue
                mc = m.get(c.name, 0.0)
                if mc == 0:
                    continue
                Phi = self._Phi(M.name, c.name, I, T)
                ln_g += mc * 2.0 * Phi
                for a in self.anions:
                    ma = m.get(a.name, 0.0)
                    if ma == 0:
                        continue
                    ln_g += mc * ma * self._psi_cca(M.name, c.name, a.name, T)
            # Σ_a<a' m_a m_a' ψ_Maa'
            for i, a1 in enumerate(self.anions):
                for a2 in self.anions[i+1:]:
                    ma1 = m.get(a1.name, 0.0)
                    ma2 = m.get(a2.name, 0.0)
                    if ma1 == 0 or ma2 == 0:
                        continue
                    ln_g += ma1 * ma2 * self._psi_caa(M.name, a1.name, a2.name, T)
            # |z_M| Σ_c Σ_a m_c m_a C_ca
            ccsum = 0.0
            for c in self.cations:
                mc = m.get(c.name, 0.0)
                for a in self.anions:
                    ma = m.get(a.name, 0.0)
                    if mc == 0 or ma == 0:
                        continue
                    ccsum += mc * ma * self._binary_C(c.name, a.name)
            ln_g += abs(M.charge) * ccsum
            out[M.name] = float(np.exp(ln_g))
        # Anions
        for X in self.anions:
            mX = m.get(X.name, 0.0)
            if mX == 0:
                continue
            ln_g = X.charge ** 2 * F
            # Σ_c m_c [2 B_cX + Z C_cX]
            for c in self.cations:
                mc = m.get(c.name, 0.0)
                if mc == 0:
                    continue
                B = self._binary_B(c.name, X.name, I, T)
                C = self._binary_C(c.name, X.name)
                ln_g += mc * (2.0 * B + Z * C)
            # Σ_a m_a [2 Φ_Xa + Σ_c m_c ψ_Xac]
            for a in self.anions:
                if a.name == X.name:
                    continue
                ma = m.get(a.name, 0.0)
                if ma == 0:
                    continue
                Phi = self._Phi(X.name, a.name, I, T)
                ln_g += ma * 2.0 * Phi
                for c in self.cations:
                    mc = m.get(c.name, 0.0)
                    if mc == 0:
                        continue
                    ln_g += mc * ma * self._psi_caa(c.name, X.name, a.name, T)
            # Σ_c<c' m_c m_c' ψ_cc'X
            for i, c1 in enumerate(self.cations):
                for c2 in self.cations[i+1:]:
                    mc1 = m.get(c1.name, 0.0)
                    mc2 = m.get(c2.name, 0.0)
                    if mc1 == 0 or mc2 == 0:
                        continue
                    ln_g += mc1 * mc2 * self._psi_cca(c1.name, c2.name, X.name, T)
            # |z_X| Σ_c Σ_a m_c m_a C_ca
            ccsum = 0.0
            for c in self.cations:
                mc = m.get(c.name, 0.0)
                for a in self.anions:
                    ma = m.get(a.name, 0.0)
                    if mc == 0 or ma == 0:
                        continue
                    ccsum += mc * ma * self._binary_C(c.name, a.name)
            ln_g += abs(X.charge) * ccsum
            out[X.name] = float(np.exp(ln_g))
        return out

    def gamma_pm(self, salt_name: str, m: Dict[str, float],
                  T: float = 298.15) -> float:
        """Mean ionic γ_± for a specific salt in the mixture.

        γ_± = (γ_M^ν_M · γ_X^ν_X)^(1/ν)

        Useful for comparing to experimental γ_NaCl in NaCl/KCl
        mixtures, etc.  ``salt_name`` should be one of the bundled
        Pitzer salts and identifies the (cation, anion, ν_M, ν_X)
        combination.
        """
        cation_name, anion_name = self._decode_salt(salt_name)
        salt = lookup_salt(salt_name)
        gammas = self.gammas(m, T)
        g_M = gammas.get(cation_name)
        g_X = gammas.get(anion_name)
        if g_M is None or g_X is None:
            raise KeyError(
                f"Cannot compute γ_± for {salt_name!r}: "
                f"need both {cation_name!r} and {anion_name!r} in mixture")
        return float((g_M ** salt.nu_M * g_X ** salt.nu_X) ** (1.0 / salt.nu))

    def osmotic_coefficient(self, m: Dict[str, float],
                              T: float = 298.15) -> float:
        """Osmotic coefficient φ for the mixture (Pitzer 1991 Eq. 3.61).

        (φ - 1) Σ_i m_i = 2 [
            -A_φ·I^(3/2) / (1+b√I)
            + Σ_c Σ_a m_c m_a (B^φ_ca + Z C_ca)
            + Σ_c<c' m_c m_c' (Φ^φ_cc' + Σ_a m_a ψ_cc'a)
            + Σ_a<a' m_a m_a' (Φ^φ_aa' + Σ_c m_c ψ_caa')
        ]
        """
        I = self.ionic_strength(m)
        Z = self._Z(m)
        sum_m = sum(m.values())
        if sum_m < 1e-15:
            return 1.0
        A = debye_huckel_A(T)
        b = self.BPITZER
        sI = np.sqrt(I)
        # Long-range
        sum_term = -A * I ** 1.5 / (1.0 + b * sI)
        # Σ_c Σ_a m_c m_a (B^φ_ca + Z C_ca)
        for c in self.cations:
            mc = m.get(c.name, 0.0)
            for a in self.anions:
                ma = m.get(a.name, 0.0)
                if mc == 0 or ma == 0:
                    continue
                B_phi = self._binary_B_phi(c.name, a.name, I, T)
                C = self._binary_C(c.name, a.name)
                sum_term += mc * ma * (B_phi + Z * C)
        # Σ_c<c' m_c m_c' (Φ^φ_cc' + Σ_a m_a ψ_cc'a)
        for i, c1 in enumerate(self.cations):
            for c2 in self.cations[i+1:]:
                mc1 = m.get(c1.name, 0.0)
                mc2 = m.get(c2.name, 0.0)
                if mc1 == 0 or mc2 == 0:
                    continue
                Phi_phi = self._Phi_phi(c1.name, c2.name, I, T)
                inner = Phi_phi
                for a in self.anions:
                    ma = m.get(a.name, 0.0)
                    if ma == 0:
                        continue
                    inner += ma * self._psi_cca(c1.name, c2.name, a.name, T)
                sum_term += mc1 * mc2 * inner
        # Σ_a<a' m_a m_a' (Φ^φ_aa' + Σ_c m_c ψ_caa')
        for i, a1 in enumerate(self.anions):
            for a2 in self.anions[i+1:]:
                ma1 = m.get(a1.name, 0.0)
                ma2 = m.get(a2.name, 0.0)
                if ma1 == 0 or ma2 == 0:
                    continue
                Phi_phi = self._Phi_phi(a1.name, a2.name, I, T)
                inner = Phi_phi
                for c in self.cations:
                    mc = m.get(c.name, 0.0)
                    if mc == 0:
                        continue
                    inner += mc * self._psi_caa(c.name, a1.name, a2.name, T)
                sum_term += ma1 * ma2 * inner
        return float(1.0 + 2.0 * sum_term / sum_m)

    def water_activity(self, m: Dict[str, float],
                          T: float = 298.15) -> float:
        """Water activity from the osmotic coefficient.

        ln(a_w) = -M_w · φ · Σ_i m_i
        """
        sum_m = sum(m.values())
        phi = self.osmotic_coefficient(m, T)
        return float(np.exp(-_MW_WATER * phi * sum_m))
