"""ChemSep pure-component database (Kooijman-Taylor 2018).

Bundled JSON dataset converted from the ChemSep XML pure-component
database, version 8.00:

    Kooijman, H. A. and Taylor, R. (2018). ChemSep — a free pure-
    component database for chemical engineering simulations.
    http://www.perlfoundation.org/artistic_license_2_0

Coverage: 446 industrially-relevant compounds, with the following
property categories:

* **Scalars**: critical properties, normal boiling/melting points,
  molecular weight, acentric factor, solubility parameter, dipole
  moment, heat of formation, Gibbs energy of formation, absolute
  entropy, Mathias-Copeman parameters, parachor, Lennard-Jones
  parameters, UNIQUAC R/Q, van der Waals volume/area, etc.
* **Temperature-dependent equations** (DIPPR forms): liquid density,
  vapor pressure (Antoine and DIPPR), heat of vaporization, ideal
  gas heat capacity, liquid heat capacity, second virial coefficient,
  liquid/vapor viscosity, liquid/vapor thermal conductivity, surface
  tension, solid density, etc.
* **Group contributions**: UNIFAC-VLE, UNIFAC-LLE, modified UNIFAC,
  ASOG, UMR-PRU, PPR78 (via gc_method).

Lookup is by CAS number (preferred), name, or SMILES.

Usage
-----

    >>> from stateprop.chemsep import lookup_chemsep, evaluate_dippr
    >>> ch4 = lookup_chemsep(name="Methane")
    >>> ch4["critical_temperature"]["value"]
    190.56
    >>> ch4["acentric_factor"]["value"]
    0.011

    >>> # Evaluate the DIPPR-101 vapor-pressure correlation at 150 K
    >>> p_sat = evaluate_dippr(ch4["vapor_pressure"], T=150.0)
    >>> p_sat            # Pa
    1042512.6...

Unit conventions
----------------
ChemSep uses **kmol** for amount of substance throughout.  This means:

* ``molecular_weight`` is in kg/kmol (= g/mol numerically).
* Heat capacities and enthalpies are J/kmol/K and J/kmol.
* Volumes are m³/kmol.

The bundled JSON preserves these original units verbatim — values are
stored exactly as ChemSep specifies.  When a stateprop component
constructor needs SI per-mol units (e.g. ``molar_mass`` in kg/mol),
divide by 1000 explicitly. Helper functions for common conversions
are provided in this module.

DIPPR equation forms
--------------------
The ``eqno`` field selects the temperature-dependence form. The most
common are:

* eqno=1, 2, 3, 4: polynomial in T
* eqno=5: polynomial in 1/T
* eqno=10: exp(A − B/(T+C))   (Antoine for ln(p))
* eqno=12: exp(A + B*T)
* eqno=13: A + B*T + C*T²
* eqno=16: A + exp(B/T + C + D*T + E*T²)
* eqno=100: A + B*T + C*T² + D*T³ + E*T⁴   (polynomial in T)
* eqno=101: exp(A + B/T + C*ln(T) + D*T^E)   (DIPPR vapor pressure)
* eqno=102: A*T^B / (1 + C/T + D/T²)   (DIPPR vapor viscosity)
* eqno=104: A + B/T + C/T³ + D/T⁸ + E/T⁹   (DIPPR second virial)
* eqno=105: A / B^(1 + (1−T/C)^D)   (DIPPR Rackett liquid density)
* eqno=106: A * (1−Tr)^(B + C*Tr + D*Tr² + E*Tr³)   (DIPPR Watson)

The ``evaluate_dippr`` helper function below covers all of these.
"""
from __future__ import annotations
import json
from functools import lru_cache
from typing import Optional, Dict, Any, List
import importlib.resources

import numpy as np


_CHEMSEP_DB: Optional[List[Dict[str, Any]]] = None


def _data_path(filename: str) -> str:
    files = importlib.resources.files("stateprop") / "data"
    return str(files / filename)


def load_chemsep_database() -> List[Dict[str, Any]]:
    """Load the bundled ChemSep pure-component database (446 entries).

    Returns the raw JSON list. Cached; subsequent calls return the
    same list object.
    """
    global _CHEMSEP_DB
    if _CHEMSEP_DB is None:
        with open(_data_path("chemsep.json"), "r") as f:
            _CHEMSEP_DB = json.load(f)
    return _CHEMSEP_DB


# =====================================================================
# Lookup
# =====================================================================

def _normalize_name(s: Optional[str]) -> Optional[str]:
    return None if s is None else s.strip().lower()


def _get_value(entry: Dict[str, Any], key: str) -> Any:
    """Pull the .value out of a {value, units} entry. Returns None if missing."""
    sub = entry.get(key)
    if sub is None:
        return None
    if isinstance(sub, dict):
        return sub.get("value")
    return sub


def _matches(entry: Dict[str, Any],
             cas: Optional[str], name: Optional[str],
             smiles: Optional[str]) -> bool:
    if cas is not None and _get_value(entry, "cas") == cas:
        return True
    if name is not None:
        n = _get_value(entry, "name")
        if isinstance(n, str) and _normalize_name(n) == _normalize_name(name):
            return True
    if smiles is not None and _get_value(entry, "smiles") == smiles:
        return True
    return False


def lookup_chemsep(cas: Optional[str] = None,
                    name: Optional[str] = None,
                    smiles: Optional[str] = None) -> Dict[str, Any]:
    """Look up a pure-component entry in the ChemSep database.

    At least one of (cas, name, smiles) must be specified.  CAS is
    preferred (synonyms make name matching ambiguous for some species);
    name matching is case-insensitive.  Raises ``KeyError`` if no
    entry matches.

    Returns
    -------
    dict
        The full ChemSep record. See module docstring for structure.
    """
    if not any([cas, name, smiles]):
        raise ValueError(
            "Must specify at least one of: cas, name, smiles")
    db = load_chemsep_database()
    for entry in db:
        if _matches(entry, cas, name, smiles):
            return entry
    raise KeyError(
        f"No ChemSep entry found for "
        f"{ {k: v for k, v in dict(cas=cas, name=name, smiles=smiles).items() if v} }")


def chemsep_summary() -> Dict[str, int]:
    """Return summary counts of the ChemSep database."""
    db = load_chemsep_database()
    n_with_vp = sum(1 for c in db if "vapor_pressure" in c)
    n_with_cp = sum(1 for c in db if "ideal_gas_heat_capacity" in c)
    n_with_unifac = sum(1 for c in db if "unifac_vle" in c)
    n_with_smiles = sum(1 for c in db if "smiles" in c)
    n_with_cas = sum(1 for c in db if "cas" in c)
    return {
        "n_compounds": len(db),
        "n_with_vapor_pressure": n_with_vp,
        "n_with_ideal_gas_heat_capacity": n_with_cp,
        "n_with_unifac_vle_groups": n_with_unifac,
        "n_with_cas": n_with_cas,
        "n_with_smiles": n_with_smiles,
    }


# =====================================================================
# DIPPR equation evaluator
# =====================================================================

def evaluate_dippr(equation: Dict[str, Any], T: float) -> float:
    """Evaluate a DIPPR temperature-dependent property correlation.

    Parameters
    ----------
    equation : dict
        A property block from a ChemSep record, with keys:
        ``eqno``, ``coefficients`` (dict of A, B, C, D, E), and
        optionally ``Tmin``, ``Tmax``.
    T : float
        Temperature in K.

    Returns
    -------
    float
        Property value in the units specified by ``equation["units"]``.

    Raises
    ------
    ValueError
        If the eqno is not implemented.
    KeyError
        If the equation block is missing required fields.

    Notes
    -----
    Out-of-range temperatures (T < Tmin or T > Tmax) are not blocked —
    the correlation is evaluated and the user is responsible for
    deciding whether extrapolation is acceptable.  All DIPPR forms
    listed in the module docstring are supported.

    The ``Tc`` (critical temperature) is required by reduced-T forms
    (eqno 106).  Pass it as ``equation["Tc"]`` if needed; for
    convenience, this function pulls it from the parent record's
    critical-temperature entry when called via ``evaluate_property``.
    """
    eqno = equation.get("eqno")
    coefs = equation.get("coefficients", {})
    A = coefs.get("A", 0.0)
    B = coefs.get("B", 0.0)
    C = coefs.get("C", 0.0)
    D = coefs.get("D", 0.0)
    E = coefs.get("E", 0.0)

    if eqno == 1:
        return A
    elif eqno == 2:
        return A + B * T
    elif eqno == 3:
        return A + B * T + C * T**2 + D * T**3
    elif eqno == 4:
        return A + B * T + C * T**2 + D * T**3 + E * T**4
    elif eqno == 5:
        return A + B / T + C / T**2 + D / T**3 + E / T**4
    elif eqno == 10:
        # Antoine on ln(p): p = exp(A - B / (T + C))
        return np.exp(A - B / (T + C))
    elif eqno == 12:
        return np.exp(A + B * T)
    elif eqno == 13:
        return A + B * T + C * T**2
    elif eqno == 16:
        # DIPPR 16: A + exp(B/T + C + D*T + E*T^2)
        return A + np.exp(B / T + C + D * T + E * T**2)
    elif eqno == 100:
        # Polynomial in T
        return A + B * T + C * T**2 + D * T**3 + E * T**4
    elif eqno == 101:
        # DIPPR vapor pressure: exp(A + B/T + C*ln(T) + D*T^E)
        return np.exp(A + B / T + C * np.log(T) + D * T**E)
    elif eqno == 102:
        # DIPPR vapor viscosity: A * T^B / (1 + C/T + D/T^2)
        return A * T**B / (1.0 + C / T + D / (T * T))
    elif eqno == 104:
        # DIPPR second virial (m^3/kmol):
        #   B(T) = A + B/T + C/T^3 + D/T^8 + E/T^9
        return A + B / T + C / T**3 + D / T**8 + E / T**9
    elif eqno == 105:
        # DIPPR Rackett liquid density (kmol/m^3):
        #   rho = A / B^(1 + (1 - T/C)^D)
        return A / B**(1.0 + (1.0 - T / C)**D)
    elif eqno == 106:
        # DIPPR Watson heat of vaporization:
        #   ΔHvap = A * (1 - Tr)^(B + C*Tr + D*Tr^2 + E*Tr^3)
        # Requires reduced T; coefficients A is the value at Tr=0.
        # Tc must be passed in via equation["Tc"]; if not, assume the
        # equation's "Tmax" is approximately Tc (works for most ChemSep
        # entries since DIPPR-106 fits stop at Tc).
        Tc = equation.get("Tc", equation.get("Tmax", T + 1.0))
        Tr = T / Tc
        if Tr >= 1.0:
            return 0.0
        exponent = B + C * Tr + D * Tr**2 + E * Tr**3
        return A * (1.0 - Tr)**exponent
    elif eqno == 118:
        # Polynomial form occasionally used in ChemSep:
        #   A + B*T + C*T^2 + D*T^3 + E*T^4 (same as 100)
        return A + B * T + C * T**2 + D * T**3 + E * T**4
    elif eqno == 120:
        # ChemSep-specific polynomial form
        return A + B * T + C * T**2 + D * T**3 + E * T**4
    elif eqno == 121:
        # Another ChemSep polynomial form
        return A + B / T + C * T + D * T**2 + E * T**3
    elif eqno == 200:
        # Cubic spline coefficients — evaluation requires the spline
        # knot data which isn't in the simple A/B/C/D/E format. Return
        # constant A as a fallback (rare in ChemSep entries).
        return A
    else:
        raise ValueError(f"DIPPR eqno {eqno} not implemented")


def evaluate_property(entry: Dict[str, Any], property_key: str,
                       T: float) -> float:
    """Evaluate a temperature-dependent property of a ChemSep compound.

    Wrapper around ``evaluate_dippr`` that pulls Tc from the parent
    record so DIPPR-106 (Watson heat of vaporization) works without
    extra plumbing.

    Parameters
    ----------
    entry : dict
        A ChemSep compound record.
    property_key : str
        One of: 'vapor_pressure', 'liquid_density',
        'heat_of_vaporization', 'ideal_gas_heat_capacity',
        'liquid_heat_capacity', 'second_virial_coefficient',
        'liquid_viscosity', 'vapor_viscosity', 'surface_tension', etc.
    T : float
        Temperature in K.

    Returns
    -------
    float
        Property value in ChemSep units (kmol-based).

    Examples
    --------
    >>> from stateprop.chemsep import lookup_chemsep, evaluate_property
    >>> water = lookup_chemsep(name="Water")
    >>> evaluate_property(water, "vapor_pressure", 373.15)   # Pa
    101325.0...
    """
    eq = entry.get(property_key)
    if eq is None:
        raise KeyError(
            f"Compound has no '{property_key}' equation block")
    # Inject Tc so DIPPR-106 works
    Tc = _get_value(entry, "critical_temperature")
    if Tc is not None and "Tc" not in eq:
        eq = {**eq, "Tc": Tc}
    return evaluate_dippr(eq, T)


# =====================================================================
# Convenience: extract common scalars in stateprop's preferred SI units
# =====================================================================

def get_critical_constants(entry: Dict[str, Any]) -> Dict[str, float]:
    """Return Tc [K], Pc [Pa], omega [-], Vc [m³/mol], Zc [-] for a
    ChemSep entry. Volume is converted from kmol to mol.

    Useful for initializing stateprop's cubic EOS objects directly:

    >>> e = lookup_chemsep(name="Methane")
    >>> c = get_critical_constants(e)
    >>> c
    {'Tc': 190.56, 'Pc': 4599000.0, 'omega': 0.011,
     'Vc': 9.86e-5, 'Zc': 0.286}
    """
    Tc = _get_value(entry, "critical_temperature")
    Pc = _get_value(entry, "critical_pressure")
    Vc_kmol = _get_value(entry, "critical_volume")
    Zc = _get_value(entry, "critical_compressibility")
    omega = _get_value(entry, "acentric_factor")
    out: Dict[str, float] = {}
    if Tc is not None: out["Tc"] = float(Tc)
    if Pc is not None: out["Pc"] = float(Pc)
    if omega is not None: out["omega"] = float(omega)
    if Vc_kmol is not None:
        out["Vc"] = float(Vc_kmol) * 1e-3   # m³/kmol → m³/mol
    if Zc is not None: out["Zc"] = float(Zc)
    return out


def get_molar_mass(entry: Dict[str, Any]) -> Optional[float]:
    """Return molar mass in kg/mol (ChemSep stores kg/kmol).

    >>> get_molar_mass(lookup_chemsep(name="Water"))
    0.01801528
    """
    mw_kmol = _get_value(entry, "molecular_weight")
    if mw_kmol is None:
        return None
    return float(mw_kmol) * 1e-3


def get_formation_properties(entry: Dict[str, Any]) -> Dict[str, float]:
    """Return ΔHf°, ΔGf°, S°(298) in J/mol and J/(mol·K).

    ChemSep stores these per kmol; we convert to per mol to match
    stateprop's convention.
    """
    Hf = _get_value(entry, "heat_of_formation")
    Gf = _get_value(entry, "gibbs_energy_of_formation")
    S = _get_value(entry, "absolute_entropy")
    out: Dict[str, float] = {}
    if Hf is not None: out["Hf"] = float(Hf) * 1e-3   # J/kmol → J/mol
    if Gf is not None: out["Gf"] = float(Gf) * 1e-3
    if S is not None: out["S"] = float(S) * 1e-3      # J/kmol/K → J/mol/K
    return out
