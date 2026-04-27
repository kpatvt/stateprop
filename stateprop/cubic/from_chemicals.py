"""Interface between stateprop.cubic and Caleb Bell's `chemicals` library.

This module lets users build cubic equations of state and cubic mixtures
by chemical name/formula/CAS, rather than having to hard-code critical
properties (T_c, p_c, omega, M) at every call site:

    from stateprop.cubic.from_chemicals import PR_from_name, cubic_mixture_from_names

    # Single-component cubic
    eos = PR_from_name("propane")

    # 5-component mixture in one call
    mix = cubic_mixture_from_names(
        ["methane", "ethane", "propane", "nitrogen", "carbon dioxide"],
        composition=[0.85, 0.05, 0.02, 0.05, 0.03],
        family="pr",
    )

REQUIREMENTS
------------
This module needs the ``chemicals`` package to look up real substance
properties (~26,000 chemicals covered). Install it with::

    pip install chemicals

If ``chemicals`` is not installed, this module falls back to a small
built-in table covering the 21 GERG-2008 components, so basic natural-gas
workflows still work without the extra dependency. Attempts to look up
substances outside that table will raise ``ImportError`` with a clear
install hint.

Properties retrieved from ``chemicals`` (in SI units):
  - T_c   critical temperature  [K]
  - p_c   critical pressure     [Pa]
  - omega acentric factor       [-]
  - M     molar mass            [kg/mol]   (chemicals returns g/mol; we convert)

The user's identifier (name, formula, CAS) is resolved via
``chemicals.CAS_from_any`` which accepts an extremely wide range of inputs
(common names, IUPAC names, SMILES, InChI, CAS numbers, formulas).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union
import warnings

from .eos import CubicEOS, PR, PR78, SRK, RK, VDW
from .mixture import CubicMixture


# ---------------------------------------------------------------------------
# Optional chemicals-library import
# ---------------------------------------------------------------------------

try:
    from chemicals import CAS_from_any
    from chemicals import Tc as _chem_Tc
    from chemicals import Pc as _chem_Pc
    from chemicals import omega as _chem_omega
    from chemicals import MW as _chem_MW
    _HAVE_CHEMICALS = True
except ImportError:
    _HAVE_CHEMICALS = False


# ---------------------------------------------------------------------------
# Fallback table: the 21 GERG-2008 components and common aliases
# ---------------------------------------------------------------------------
#
# Values are from GERG-2008 Table A5 (critical parameters and molar masses)
# and NIST Webbook / DIPPR for acentric factors. Keys are lowercased with
# spaces/dashes removed so `normalize("Carbon Dioxide") -> "carbondioxide"`.

_FALLBACK_TABLE: dict[str, dict[str, float]] = {
    # key: {"T_c": K, "p_c": Pa, "omega": -, "M": kg/mol, "aliases": [...]}
    "methane":         {"T_c": 190.564,  "p_c": 4.5992e6, "omega": 0.01142, "M": 0.01604246},
    "nitrogen":        {"T_c": 126.192,  "p_c": 3.3958e6, "omega": 0.0372,  "M": 0.0280134},
    "carbondioxide":   {"T_c": 304.1282, "p_c": 7.3773e6, "omega": 0.22394, "M": 0.0440095},
    "ethane":          {"T_c": 305.322,  "p_c": 4.8722e6, "omega": 0.0995,  "M": 0.03006904},
    "propane":         {"T_c": 369.825,  "p_c": 4.2472e6, "omega": 0.1521,  "M": 0.04409562},
    "nbutane":         {"T_c": 425.125,  "p_c": 3.796e6,  "omega": 0.201,   "M": 0.0581222},
    "isobutane":       {"T_c": 407.817,  "p_c": 3.629e6,  "omega": 0.184,   "M": 0.0581222},
    "npentane":        {"T_c": 469.7,    "p_c": 3.3675e6, "omega": 0.251,   "M": 0.07214878},
    "isopentane":      {"T_c": 460.35,   "p_c": 3.378e6,  "omega": 0.2274,  "M": 0.07214878},
    "nhexane":         {"T_c": 507.82,   "p_c": 3.034e6,  "omega": 0.299,   "M": 0.08617536},
    "nheptane":        {"T_c": 540.13,   "p_c": 2.736e6,  "omega": 0.349,   "M": 0.10020194},
    "noctane":         {"T_c": 569.32,   "p_c": 2.497e6,  "omega": 0.398,   "M": 0.11422852},
    "nnonane":         {"T_c": 594.55,   "p_c": 2.281e6,  "omega": 0.443,   "M": 0.1282551},
    "ndecane":         {"T_c": 617.7,    "p_c": 2.103e6,  "omega": 0.4884,  "M": 0.14228168},
    "hydrogen":        {"T_c": 33.19,    "p_c": 1.315e6,  "omega": -0.219,  "M": 0.00201588},
    "oxygen":          {"T_c": 154.595,  "p_c": 5.043e6,  "omega": 0.0222,  "M": 0.0319988},
    "carbonmonoxide":  {"T_c": 132.86,   "p_c": 3.494e6,  "omega": 0.05,    "M": 0.0280101},
    "water":           {"T_c": 647.096,  "p_c": 22.064e6, "omega": 0.3443,  "M": 0.01801528},
    "hydrogensulfide": {"T_c": 373.1,    "p_c": 9.0e6,    "omega": 0.1005,  "M": 0.03408088},
    "helium":          {"T_c": 5.1953,   "p_c": 0.22746e6,"omega": -0.385,  "M": 0.004002602},
    "argon":           {"T_c": 150.687,  "p_c": 4.863e6,  "omega": -0.00219,"M": 0.039948},
}


# Aliases that should map to the canonical keys above. Common names and
# formulas are included. (CAS numbers are left to chemicals.CAS_from_any;
# this fallback table is only for the case where chemicals is NOT
# installed, and typical users then refer to fluids by name.)
_FALLBACK_ALIASES: dict[str, str] = {
    # Methane
    "ch4":            "methane",
    "c1":             "methane",
    # Nitrogen
    "n2":             "nitrogen",
    # CO2
    "co2":            "carbondioxide",
    "carbon dioxide": "carbondioxide",
    # Ethane
    "c2h6":           "ethane",
    "c2":             "ethane",
    # Propane
    "c3h8":           "propane",
    "c3":             "propane",
    # Butanes
    "nc4":            "nbutane",  "n-butane":  "nbutane",
    "butane":         "nbutane",  # "butane" usually means n-butane in industry
    "ic4":            "isobutane", "i-butane": "isobutane",
    "isobutane":      "isobutane", "2-methylpropane": "isobutane",
    # Pentanes
    "nc5":            "npentane", "n-pentane": "npentane", "pentane": "npentane",
    "ic5":            "isopentane","i-pentane": "isopentane", "2-methylbutane": "isopentane",
    # Longer alkanes
    "nc6":            "nhexane",  "n-hexane": "nhexane",  "hexane":  "nhexane",
    "nc7":            "nheptane", "n-heptane": "nheptane", "heptane": "nheptane",
    "nc8":            "noctane",  "n-octane": "noctane",  "octane":  "noctane",
    "nc9":            "nnonane",  "n-nonane": "nnonane",  "nonane":  "nnonane",
    "nc10":           "ndecane",  "n-decane": "ndecane",  "decane":  "ndecane",
    # H2, O2, CO
    "h2":             "hydrogen",
    "o2":             "oxygen",
    "co":             "carbonmonoxide",
    "carbon monoxide":"carbonmonoxide",
    # Water
    "h2o":            "water",
    "steam":          "water",
    # H2S
    "h2s":            "hydrogensulfide",
    "hydrogen sulfide":"hydrogensulfide",
    # Noble gases
    "he":             "helium",
    "ar":             "argon",
}


def _fallback_normalize(name: str) -> str:
    """Normalize an identifier for fallback-table lookup."""
    s = name.strip().lower()
    # Try alias table first (it preserves hyphens, spaces, etc.)
    if s in _FALLBACK_ALIASES:
        return _FALLBACK_ALIASES[s]
    # Then try the canonical-key form (spaces/hyphens stripped)
    s_canon = s.replace(" ", "").replace("-", "")
    if s_canon in _FALLBACK_TABLE:
        return s_canon
    if s_canon in _FALLBACK_ALIASES:
        return _FALLBACK_ALIASES[s_canon]
    raise KeyError(
        f"'{name}' is not in stateprop's built-in fallback table. "
        f"Install the `chemicals` library to enable lookup of ~26,000 chemicals: "
        f"    pip install chemicals"
    )


# ---------------------------------------------------------------------------
# The core lookup function
# ---------------------------------------------------------------------------

def lookup_pure_component(identifier: str) -> dict:
    """Look up pure-component properties needed to build a cubic EOS.

    Parameters
    ----------
    identifier : str
        Any identifier that ``chemicals.CAS_from_any`` accepts: a chemical
        name, IUPAC name, formula, SMILES, InChI, CAS number, or common
        alias. If ``chemicals`` is not installed, falls back to a small
        built-in table of ~21 common components (alkanes through n-decane,
        common inerts, water, H2S, H2, He, Ar).

    Returns
    -------
    dict with keys: ``T_c`` [K], ``p_c`` [Pa], ``omega`` [-],
    ``M`` [kg/mol], ``source`` (str: ``"chemicals"`` or ``"fallback"``),
    ``CAS`` (str or None).

    Raises
    ------
    KeyError   if the identifier is not found in either source.
    ValueError if chemicals returns None for a required property.

    Notes
    -----
    Unlike the hard-coded stateprop factories which take numeric T_c, p_c,
    omega arguments, this function expects a single string identifier. If
    you already have numeric values from a reference you want to enforce,
    build the CubicEOS directly with PR(T_c, p_c, omega) etc.
    """
    if _HAVE_CHEMICALS:
        try:
            cas = CAS_from_any(identifier)
        except Exception as e:
            raise KeyError(
                f"chemicals.CAS_from_any could not resolve '{identifier}': {e}"
            ) from e
        T_c = _chem_Tc(cas)
        p_c = _chem_Pc(cas)
        om = _chem_omega(cas)
        M_gmol = _chem_MW(cas)
        missing = [name for name, val in
                   (("T_c", T_c), ("p_c", p_c), ("omega", om), ("M", M_gmol))
                   if val is None]
        if missing:
            raise ValueError(
                f"chemicals returned None for {identifier} (CAS {cas}): "
                f"missing {missing}. This substance may not have critical-"
                f"property data in any chemicals databank, or may require a "
                f"non-default estimation method."
            )
        return {
            "T_c":   float(T_c),
            "p_c":   float(p_c),
            "omega": float(om),
            "M":     float(M_gmol) * 1e-3,    # g/mol -> kg/mol
            "source": "chemicals",
            "CAS":   cas,
        }

    # Fallback path
    key = _fallback_normalize(identifier)
    row = _FALLBACK_TABLE[key]
    return {
        "T_c":   row["T_c"],
        "p_c":   row["p_c"],
        "omega": row["omega"],
        "M":     row["M"],
        "source": "fallback",
        "CAS":   None,
    }


# ---------------------------------------------------------------------------
# Convenience factories: build a CubicEOS from an identifier
# ---------------------------------------------------------------------------

_FAMILY_FACTORY = {
    "pr":   PR,
    "pr78": PR78,
    "srk":  SRK,
    "rk":   RK,
    "vdw":  VDW,
}


def cubic_from_name(identifier: str, family: str = "pr",
                       volume_shift: object = None,
                       **kwargs) -> CubicEOS:
    """Build a single-component CubicEOS by name.

    Parameters
    ----------
    identifier : str   Chemical identifier (name, formula, CAS, etc.)
    family : str       One of "pr" (default), "pr78", "srk", "rk", "vdw".
    volume_shift : str | float | None, default None
        Volume translation policy (v0.9.119).  If supplied:

        - ``"auto"``: look up the compound in the bundled table
          (:mod:`stateprop.cubic.volume_translation`), falling back to
          the family correlation (Peneloux 1982 for SRK / RK,
          Jhaveri-Youngren 1988 for PR / PR78).
        - ``"table"``: bundled table only.  Raises KeyError if missing.
        - ``"correlation"``: family correlation only, ignoring the table.
        - ``float``: pass-through user-supplied c [m³/mol].
        - ``None`` (default): no volume shift, c=0.

        If both ``volume_shift`` and ``volume_shift_c`` (the lower-level
        kwarg) are supplied, ``volume_shift_c`` wins.  ``volume_shift``
        is a higher-level convenience that auto-resolves c from the
        bundled DB; ``volume_shift_c`` is the literal CubicEOS field.
    **kwargs           Extra arguments passed through to the EOS factory
                       (e.g., ``volume_shift_c``, ``use_pr78``).

    Examples
    --------
    >>> # Plain PR for propane
    >>> eos = cubic_from_name("propane")

    >>> # SRK with auto volume shift (table → correlation fallback)
    >>> eos = cubic_from_name("water", family="srk", volume_shift="auto")

    >>> # PR with bundled-table-only lookup (raises if compound missing)
    >>> eos = cubic_from_name("benzene", volume_shift="table")
    """
    props = lookup_pure_component(identifier)
    fam = family.lower()
    if fam not in _FAMILY_FACTORY:
        raise ValueError(
            f"Unknown cubic family '{family}'. "
            f"Expected one of {list(_FAMILY_FACTORY)}."
        )
    factory = _FAMILY_FACTORY[fam]

    # Resolve volume_shift policy (v0.9.119) into a numeric c, unless the
    # caller has already supplied volume_shift_c directly (in which case
    # that wins, preserving backward compatibility).
    if volume_shift is not None and "volume_shift_c" not in kwargs:
        from .volume_translation import resolve_volume_shift
        c = resolve_volume_shift(
            name=identifier,
            family=fam,
            omega=props.get("omega", 0.0),
            T_c=props["T_c"],
            p_c=props["p_c"],
            molar_mass=props.get("M", 0.0) or 0.0,
            mode=volume_shift,
        )
        if c is not None:
            kwargs["volume_shift_c"] = c

    # RK and VDW don't take acentric_factor
    if fam in ("rk", "vdw"):
        return factory(T_c=props["T_c"], p_c=props["p_c"], **kwargs)
    return factory(
        T_c=props["T_c"],
        p_c=props["p_c"],
        acentric_factor=props["omega"],
        **kwargs,
    )


# Shortcut wrappers for the common families
def PR_from_name(identifier: str, **kwargs) -> CubicEOS:
    """Peng-Robinson cubic EOS from a chemical identifier."""
    return cubic_from_name(identifier, family="pr", **kwargs)


def PR78_from_name(identifier: str, **kwargs) -> CubicEOS:
    """PR-1978 cubic EOS from a chemical identifier."""
    return cubic_from_name(identifier, family="pr78", **kwargs)


def SRK_from_name(identifier: str, **kwargs) -> CubicEOS:
    """Soave-Redlich-Kwong cubic EOS from a chemical identifier."""
    return cubic_from_name(identifier, family="srk", **kwargs)


def RK_from_name(identifier: str, **kwargs) -> CubicEOS:
    """Original Redlich-Kwong cubic EOS from a chemical identifier."""
    return cubic_from_name(identifier, family="rk", **kwargs)


def VDW_from_name(identifier: str, **kwargs) -> CubicEOS:
    """Van der Waals cubic EOS from a chemical identifier."""
    return cubic_from_name(identifier, family="vdw", **kwargs)


# ---------------------------------------------------------------------------
# Mixture factory
# ---------------------------------------------------------------------------

def cubic_mixture_from_names(
    identifiers: Sequence[str],
    composition: Optional[Sequence[float]] = None,
    family: str = "pr",
    k_ij: Optional[dict] = None,
    **eos_kwargs,
) -> CubicMixture:
    """Build a CubicMixture from a list of chemical identifiers.

    Parameters
    ----------
    identifiers : sequence of str
        Chemical names/formulas/CAS numbers for each component.
    composition : sequence of float, optional
        Mole fractions (must sum to 1.0, automatically normalized).
        If None, equal composition is assumed.
    family : str
        Cubic EOS family, same as `cubic_from_name`.
    k_ij : dict, optional
        Binary interaction parameters keyed by (i, j) tuples with i < j,
        e.g., ``{(0, 3): 0.025}`` for a 0.025 k_ij between the 0th and
        3rd components.
    **eos_kwargs
        Passed through to each per-component EOS factory.

    Returns
    -------
    CubicMixture

    Examples
    --------
    Build a 5-component natural-gas mixture with binary interaction
    parameters from Passut-Danner or published sources::

        mix = cubic_mixture_from_names(
            ["methane", "ethane", "propane", "nitrogen", "carbon dioxide"],
            composition=[0.85, 0.05, 0.02, 0.05, 0.03],
            family="pr",
            k_ij={(0, 3): 0.025, (0, 4): 0.09, (3, 4): -0.017},
        )

    Notes
    -----
    k_ij defaults to zero for all pairs not listed. For rigorous engineering
    calculations these values should come from published experimental fits
    (e.g., DIPPR, GPA, or NIST TDE). Stateprop does NOT supply default
    k_ij values -- there is no universally-accepted binary-parameter set
    for cubic EOS, and silently defaulting to nonzero k_ij could produce
    subtly wrong results.
    """
    components = [cubic_from_name(n, family=family, **eos_kwargs)
                  for n in identifiers]
    return CubicMixture(components, composition=composition, k_ij=k_ij)


# ---------------------------------------------------------------------------
# Module-level capability check
# ---------------------------------------------------------------------------

def chemicals_available() -> bool:
    """Returns True if the `chemicals` library is importable.

    When False, stateprop.cubic.from_chemicals still works for the ~21
    fluids in the built-in fallback table. Install `chemicals` to get
    access to the ~26,000-chemical databank.
    """
    return _HAVE_CHEMICALS
