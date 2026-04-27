"""PC-SAFT parameter database (Esper 2023 + Rehner 2023).

Bundled JSON datasets ported from the FeOS open-source library:

    * ``esper2023.json`` (1842 pure-component PC-SAFT parameters):
      Esper, T., Bursik, B., Bauer, P., Gross, J. (2023).  PCP-SAFT
      parameters of pure substances using large experimental databases
      and active learning.  Industrial & Engineering Chemistry Research,
      62(37), 15300-15310.

    * ``rehner2023_binary.json`` (7848 binary interaction parameters):
      Rehner, P., Bauer, P., Gross, J. (2023).  Equation of state and
      mixing rules for transferable PCP-SAFT parameters.  J. Chemical &
      Engineering Data, 68(7), 1604-1623.

These cover most common refinery, gas-processing, and chemical-process
species.  Each pure-component entry has at minimum (m, sigma, epsilon/k,
MW); 1090 of the 1842 components have associating-site parameters
(``na``, ``nb``, optionally ``kappa_ab``, ``epsilon_k_ab``); 457 have
dipole moments ``mu``.

Lookup is by CAS number (preferred), name, IUPAC name, SMILES, or InChI.
The first identifier match is returned; if the user has multiple
substances with the same common name (rare), they should use the CAS.

Usage
-----

    >>> from stateprop.saft.database import lookup_pcsaft, lookup_kij
    >>> meoh = lookup_pcsaft(name='methanol')
    >>> print(meoh.m, meoh.sigma, meoh.epsilon_k)
    2.25965 2.83016 183.58634
    >>> print(meoh.kappa_AB, meoh.eps_AB_k)   # has 2B-style associating
    0.08716 2465.13545

    >>> kij = lookup_kij(name1='methanol', name2='water')
    >>> print(kij)         # may be None if not in database
    -0.06xxx

    >>> # One-call constructor for a SAFTMixture
    >>> from stateprop.saft.database import make_saft_mixture
    >>> mix = make_saft_mixture(['methanol', 'water'], composition=[0.5, 0.5])
"""
from __future__ import annotations
from functools import lru_cache
import json
from typing import Optional, Sequence, Tuple, List, Dict, Any
import importlib.resources

import numpy as np

from .eos import PCSAFT


# =====================================================================
# Database loaders (cached)
# =====================================================================

_PURE_DB: Optional[List[Dict[str, Any]]] = None
_BINARY_DB: Optional[List[Dict[str, Any]]] = None


def _data_path(filename: str) -> str:
    """Resolve a path to a packaged JSON file in stateprop/data/.

    Uses ``importlib.resources`` so the resolution works whether
    stateprop is installed as a wheel or run from a source tree.
    """
    files = importlib.resources.files("stateprop") / "data"
    return str(files / filename)


def load_pure_database() -> List[Dict[str, Any]]:
    """Load the bundled Esper-2023 pure-component PC-SAFT parameters.

    Returns the raw JSON list (1842 entries).  Cached; subsequent calls
    return the same list object.  See module docstring for citation.
    """
    global _PURE_DB
    if _PURE_DB is None:
        with open(_data_path("esper2023.json"), "r") as f:
            _PURE_DB = json.load(f)
    return _PURE_DB


def load_binary_database() -> List[Dict[str, Any]]:
    """Load the bundled Rehner-2023 binary interaction parameters.

    Returns the raw JSON list (7848 entries).  Cached.
    """
    global _BINARY_DB
    if _BINARY_DB is None:
        with open(_data_path("rehner2023_binary.json"), "r") as f:
            _BINARY_DB = json.load(f)
    return _BINARY_DB


# =====================================================================
# Pure-component lookup
# =====================================================================

def _normalize(s: Optional[str]) -> Optional[str]:
    """Normalize a string for case-insensitive matching (None-safe)."""
    return None if s is None else s.strip().lower()


def _identifier_matches(entry_id: Dict[str, Any],
                          cas: Optional[str] = None,
                          name: Optional[str] = None,
                          iupac_name: Optional[str] = None,
                          smiles: Optional[str] = None,
                          inchi: Optional[str] = None) -> bool:
    """Check whether a database entry's `identifier` block matches any
    of the supplied lookup keys.  Returns True on the first match.
    Case-insensitive on names; CAS/SMILES/InChI matched exactly."""
    if cas is not None and entry_id.get("cas") == cas:
        return True
    if (name is not None
            and _normalize(entry_id.get("name")) == _normalize(name)):
        return True
    if (iupac_name is not None
            and _normalize(entry_id.get("iupac_name")) == _normalize(iupac_name)):
        return True
    if smiles is not None and entry_id.get("smiles") == smiles:
        return True
    if inchi is not None and entry_id.get("inchi") == inchi:
        return True
    return False


def _find_pure_entry(cas: Optional[str] = None,
                      name: Optional[str] = None,
                      iupac_name: Optional[str] = None,
                      smiles: Optional[str] = None,
                      inchi: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """First-match lookup in the pure-component database."""
    if not any([cas, name, iupac_name, smiles, inchi]):
        raise ValueError(
            "Must specify at least one of: cas, name, iupac_name, "
            "smiles, inchi")
    db = load_pure_database()
    for entry in db:
        if _identifier_matches(entry["identifier"],
                                  cas=cas, name=name,
                                  iupac_name=iupac_name,
                                  smiles=smiles, inchi=inchi):
            return entry
    return None


def lookup_pcsaft(cas: Optional[str] = None,
                   name: Optional[str] = None,
                   iupac_name: Optional[str] = None,
                   smiles: Optional[str] = None,
                   inchi: Optional[str] = None,
                   T_c: float = 0.0,
                   p_c: float = 0.0,
                   acentric_factor: float = 0.0) -> PCSAFT:
    """Look up a pure-component PC-SAFT parameter set and return a
    ``PCSAFT`` dataclass instance.

    Lookup keys (any one suffices, CAS preferred):
        cas         CAS registry number, e.g. ``'67-56-1'``
        name        Common name, e.g. ``'methanol'``
        iupac_name  IUPAC systematic name
        smiles      Canonical SMILES
        inchi       Standard InChI

    Optional ``T_c``, ``p_c``, ``acentric_factor`` parameters are
    forwarded directly to the ``PCSAFT`` dataclass.  These are only
    used by stateprop for Wilson K-value initialization in flash
    routines; the core SAFT calculations do not require them.  Default
    is 0.0 (sentinel — flash will fall back to non-Wilson init).

    Raises
    ------
    KeyError
        If no entry matches the given identifiers.
    """
    entry = _find_pure_entry(cas=cas, name=name, iupac_name=iupac_name,
                                smiles=smiles, inchi=inchi)
    if entry is None:
        keys = {k: v for k, v in dict(cas=cas, name=name,
                                          iupac_name=iupac_name,
                                          smiles=smiles, inchi=inchi).items()
                if v is not None}
        raise KeyError(f"No PC-SAFT entry found for {keys}")

    eid = entry["identifier"]
    # Required PC-SAFT parameters
    m = float(entry["m"])
    sigma = float(entry["sigma"])
    eps_k = float(entry["epsilon_k"])
    mw_g = float(entry["molarweight"])

    # Polar (optional)
    mu = float(entry.get("mu", 0.0))

    # Associating (optional).  The dataset uses a single-site block
    # with na, nb, kappa_ab, epsilon_k_ab.  When kappa_ab is absent
    # (induced-association entries), we leave the SAFT params at zero
    # — these compounds will associate only through cross-association
    # via binary parameters.
    sites = entry.get("association_sites", [])
    eps_AB_k = 0.0
    kappa_AB = 0.0
    if sites:
        # Use the first site's params (most entries have exactly one)
        site = sites[0]
        if "kappa_ab" in site and "epsilon_k_ab" in site:
            kappa_AB = float(site["kappa_ab"])
            eps_AB_k = float(site["epsilon_k_ab"])
        # If only nb/na present, leave at zero (induced-association)

    return PCSAFT(
        m=m, sigma=sigma, epsilon_k=eps_k,
        T_c=T_c, p_c=p_c, acentric_factor=acentric_factor,
        name=eid.get("name") or eid.get("cas") or "unknown",
        eps_AB_k=eps_AB_k, kappa_AB=kappa_AB,
        assoc_scheme="2B",
        dipole_moment=mu,
        n_polar_segments=0.0,    # auto-set to m by PCSAFT __post_init__ when mu>0
        quadrupole_moment=0.0,
        molar_mass=mw_g * 1e-3,  # g/mol → kg/mol
        T_b=0.0,
        parachor=0.0,
    )


# =====================================================================
# Binary lookup
# =====================================================================

def _binary_match(entry: Dict[str, Any],
                    cas1: Optional[str], name1: Optional[str],
                    cas2: Optional[str], name2: Optional[str]
                    ) -> bool:
    """Check both ordering directions: (id1,id2) and (id2,id1)."""
    fwd = (_identifier_matches(entry["id1"], cas=cas1, name=name1)
           and _identifier_matches(entry["id2"], cas=cas2, name=name2))
    rev = (_identifier_matches(entry["id1"], cas=cas2, name=name2)
           and _identifier_matches(entry["id2"], cas=cas1, name=name1))
    return fwd or rev


def lookup_binary(cas1: Optional[str] = None, name1: Optional[str] = None,
                    cas2: Optional[str] = None, name2: Optional[str] = None,
                    ) -> Optional[Dict[str, Any]]:
    """Look up a binary interaction record between two components.

    Returns
    -------
    dict or None
        ``{'k_ij': float | None, 'kappa_ab': float | None,
           'epsilon_k_ab': float | None}``.
        Each field is ``None`` if absent in the database.
        Returns ``None`` if no entry exists for this binary pair.

    Notes
    -----
    The Rehner-2023 binary database is symmetric: an entry covers
    both (A, B) and (B, A) pairings.  The returned ``kappa_ab`` and
    ``epsilon_k_ab`` are *cross*-association parameters, applicable
    only when both species have associating sites.
    """
    if not (cas1 or name1) or not (cas2 or name2):
        raise ValueError(
            "Must specify (cas or name) for both components 1 and 2")
    db = load_binary_database()
    for entry in db:
        if _binary_match(entry, cas1=cas1, name1=name1,
                          cas2=cas2, name2=name2):
            sites = entry.get("association_sites", [])
            kappa_ab = sites[0].get("kappa_ab") if sites else None
            eps_ab = sites[0].get("epsilon_k_ab") if sites else None
            return {
                "k_ij": entry.get("k_ij"),
                "kappa_ab": kappa_ab,
                "epsilon_k_ab": eps_ab,
            }
    return None


def lookup_kij(cas1: Optional[str] = None, name1: Optional[str] = None,
                cas2: Optional[str] = None, name2: Optional[str] = None,
                ) -> Optional[float]:
    """Convenience: look up just the k_ij scalar between two species.

    Returns ``None`` if the binary pair is not in the database OR if
    it is in the database but has no k_ij value.
    """
    rec = lookup_binary(cas1=cas1, name1=name1, cas2=cas2, name2=name2)
    if rec is None:
        return None
    return rec.get("k_ij")


# =====================================================================
# One-call mixture constructor
# =====================================================================

def make_saft_mixture(names: Sequence[str],
                        composition: Optional[Sequence[float]] = None,
                        cas_list: Optional[Sequence[str]] = None,
                        T_c_list: Optional[Sequence[float]] = None,
                        p_c_list: Optional[Sequence[float]] = None,
                        omega_list: Optional[Sequence[float]] = None,
                        ):
    """Build a ``SAFTMixture`` from common names with auto-populated kij.

    Each component is looked up in the Esper-2023 pure database; the
    binary pairs are looked up in the Rehner-2023 binary database and
    used to fill the kij matrix.  Cross-association parameters are NOT
    automatically transferred to the mixture (stateprop's SAFTMixture
    uses a Berthelot-Lorentz combining rule for cross-association by
    default; supplying database cross-assoc would require additional
    plumbing in mixture.py).

    Parameters
    ----------
    names : sequence of str
        Component names matching the Esper-2023 dataset.
    composition : sequence of float, optional
        Mole fractions (must sum to 1).  Defaults to uniform.
    cas_list : sequence of str, optional
        CAS numbers, used as primary lookup if provided (more reliable
        than name matching for compounds with synonyms).
    T_c_list, p_c_list, omega_list : sequence of float, optional
        Critical properties used by Wilson initialization in flash
        routines.  Highly recommended when the resulting mixture will
        be used in two-phase flash; not needed for pure-density work.

    Returns
    -------
    SAFTMixture

    Examples
    --------
    >>> mix = make_saft_mixture(['methanol', 'water'],
    ...                           composition=[0.3, 0.7])
    >>> rho = mix.density_from_pressure(p=1e5, T=300, phase_hint='liquid')
    """
    from .mixture import SAFTMixture

    n = len(names)
    cas_list = list(cas_list) if cas_list else [None] * n
    T_c_list = list(T_c_list) if T_c_list else [0.0] * n
    p_c_list = list(p_c_list) if p_c_list else [0.0] * n
    omega_list = list(omega_list) if omega_list else [0.0] * n
    composition = (np.full(n, 1.0 / n) if composition is None
                    else np.asarray(composition, dtype=float))
    if abs(composition.sum() - 1.0) > 1e-9:
        raise ValueError(
            f"composition must sum to 1, got {composition.sum()}")

    components = []
    for i, name in enumerate(names):
        kw = dict(name=name, T_c=T_c_list[i], p_c=p_c_list[i],
                   acentric_factor=omega_list[i])
        if cas_list[i]:
            kw = dict(cas=cas_list[i], T_c=T_c_list[i], p_c=p_c_list[i],
                       acentric_factor=omega_list[i])
        components.append(lookup_pcsaft(**kw))

    # Build kij dict from the binary database (SAFTMixture wants
    # {(i, j): kij} with i < j; symmetric pairs not duplicated)
    k_ij_dict: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            kij = lookup_kij(cas1=cas_list[i], name1=names[i],
                              cas2=cas_list[j], name2=names[j])
            if kij is not None:
                k_ij_dict[(i, j)] = float(kij)

    return SAFTMixture(components, composition, k_ij=k_ij_dict)


# =====================================================================
# Diagnostics / database statistics
# =====================================================================

def database_summary() -> Dict[str, int]:
    """Return summary counts of the bundled databases.

    Useful for documentation, sanity checks, and ``__repr__`` output.
    """
    pure = load_pure_database()
    binary = load_binary_database()
    n_polar = sum(1 for p in pure if p.get("mu", 0.0) > 0)
    n_assoc_full = sum(
        1 for p in pure if p.get("association_sites")
        and any("kappa_ab" in s for s in p["association_sites"]))
    n_assoc_induced = sum(
        1 for p in pure if p.get("association_sites")
        and not any("kappa_ab" in s for s in p["association_sites"]))
    n_kij = sum(1 for b in binary if b.get("k_ij") is not None)
    n_xassoc = sum(1 for b in binary if b.get("association_sites"))
    return {
        "n_pure_components": len(pure),
        "n_pure_polar": n_polar,
        "n_pure_assoc_full": n_assoc_full,
        "n_pure_assoc_induced": n_assoc_induced,
        "n_binary_pairs": len(binary),
        "n_binary_with_kij": n_kij,
        "n_binary_with_cross_assoc": n_xassoc,
    }
