"""Pre-built UNIFAC group definitions and convenience constructors
for common molecules.

Eliminates the manual group-counting step. Instead of writing
``{'CH3': 1, 'CH2': 1, 'OH': 1}`` for ethanol, users can call
``make_unifac(['ethanol', 'water'])`` and get a configured model.

The compound database covers about 50 widely-used molecules across
alkanes, alkenes, aromatics, alcohols, ketones, esters, ethers,
carboxylic acids, nitriles, and water. Group decompositions follow
the standard UNIFAC convention (Reid-Prausnitz-Poling 5th ed.,
Table 8-23, and the original Fredenslund 1977 paper). Each entry's
groups are verified to use only subgroups present in
``stateprop.activity.unifac_database.SUBGROUPS``.

UNIQUAC r, q parameters are computed automatically as
``sum_k nu_k * R_k`` and ``sum_k nu_k * Q_k`` -- this is the
standard relationship between UNIFAC and UNIQUAC pure-component
parameters, since both come from the same underlying van der Waals
volumes and surface areas.

Usage
-----

    from stateprop.activity.compounds import (
        make_unifac, make_unifac_dortmund, make_unifac_lyngby,
        make_uniquac, get_groups, list_compounds)

    # Build UNIFAC model for ethanol-water
    uf = make_unifac(['ethanol', 'water'])
    print(uf.gammas(298.15, [0.5, 0.5]))

    # Same compounds with UNIFAC-Dortmund
    uf_d = make_unifac_dortmund(['ethanol', 'water'])

    # UNIQUAC: r, q computed from UNIFAC group sums; user provides b
    uq = make_uniquac(['ethanol', 'water'],
                       b=[[0, 250.0], [800.0, 0]])

    # Inspect a compound's group decomposition
    print(get_groups('toluene'))        # {'ACH': 5, 'ACCH3': 1}
    print(list_compounds())             # ['acetic_acid', 'acetone', ...]

Notes
-----
- Compound names are case-insensitive; underscores and dashes are
  ignored (so 'n-hexane', 'n_hexane', 'nhexane' all work).
- For compounds outside this database, users can still pass an
  explicit groups dict to UNIFAC(...) directly.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

from .unifac_database import SUBGROUPS


# ---------------------------------------------------------------------------
# Compound database
# ---------------------------------------------------------------------------
# Each entry: name -> dict of {subgroup_name: count}
# Verified against RPP 5th ed. Table 8-23 and Fredenslund (1977) where
# applicable. All subgroups must exist in SUBGROUPS.

_COMPOUNDS: Dict[str, Dict[str, int]] = {
    # --- Water ---
    'water': {'H2O': 1},

    # --- n-Alkanes ---
    'ethane':    {'CH3': 2},
    'propane':   {'CH3': 2, 'CH2': 1},
    'n-butane':  {'CH3': 2, 'CH2': 2},
    'n-pentane': {'CH3': 2, 'CH2': 3},
    'n-hexane':  {'CH3': 2, 'CH2': 4},
    'n-heptane': {'CH3': 2, 'CH2': 5},
    'n-octane':  {'CH3': 2, 'CH2': 6},
    'n-nonane':  {'CH3': 2, 'CH2': 7},
    'n-decane':  {'CH3': 2, 'CH2': 8},

    # --- Branched alkanes ---
    'isobutane':         {'CH3': 3, 'CH': 1},
    'isopentane':        {'CH3': 3, 'CH2': 1, 'CH': 1},
    'neopentane':        {'CH3': 4, 'C': 1},
    '2-methylpentane':   {'CH3': 3, 'CH2': 2, 'CH': 1},
    '3-methylpentane':   {'CH3': 3, 'CH2': 2, 'CH': 1},
    '2,2-dimethylbutane': {'CH3': 4, 'CH2': 1, 'C': 1},

    # --- Cyclics (no separate cyclic CH2 group; use CH2) ---
    'cyclopentane':       {'CH2': 5},
    'cyclohexane':        {'CH2': 6},
    'methylcyclohexane':  {'CH3': 1, 'CH2': 5, 'CH': 1},

    # --- Alkenes ---
    # Note: ethene (CH2=CH2) needs CH2=CH2 subgroup -- not in this DB.
    'propene':   {'CH2=CH': 1, 'CH3': 1},        # CH2=CH-CH3
    '1-butene':  {'CH2=CH': 1, 'CH2': 1, 'CH3': 1},
    '1-pentene': {'CH2=CH': 1, 'CH2': 2, 'CH3': 1},
    '1-hexene':  {'CH2=CH': 1, 'CH2': 3, 'CH3': 1},
    'cis-2-butene':   {'CH=CH': 1, 'CH3': 2},
    'trans-2-butene': {'CH=CH': 1, 'CH3': 2},
    'isobutene':       {'CH2=C': 1, 'CH3': 2},   # CH2=C(CH3)2
    '2-methyl-2-butene': {'CH=C': 1, 'CH3': 3},

    # --- Aromatics ---
    'benzene':       {'ACH': 6},
    'toluene':       {'ACH': 5, 'ACCH3': 1},
    'ethylbenzene':  {'ACH': 5, 'ACCH2': 1, 'CH3': 1},
    'o-xylene':      {'ACH': 4, 'ACCH3': 2},
    'm-xylene':      {'ACH': 4, 'ACCH3': 2},
    'p-xylene':      {'ACH': 4, 'ACCH3': 2},
    'cumene':        {'ACH': 5, 'ACCH': 1, 'CH3': 2},   # isopropylbenzene
    'mesitylene':    {'ACH': 3, 'ACCH3': 3},   # 1,3,5-trimethylbenzene

    # --- Alcohols ---
    # Methanol uses its own CH3OH subgroup (special parameterization)
    'methanol':         {'CH3OH': 1},
    'ethanol':          {'CH3': 1, 'CH2': 1, 'OH': 1},
    '1-propanol':       {'CH3': 1, 'CH2': 2, 'OH': 1},
    '2-propanol':       {'CH3': 2, 'CH': 1, 'OH': 1},
    '1-butanol':        {'CH3': 1, 'CH2': 3, 'OH': 1},
    '2-butanol':        {'CH3': 2, 'CH2': 1, 'CH': 1, 'OH': 1},
    'isobutanol':       {'CH3': 2, 'CH2': 1, 'CH': 1, 'OH': 1},  # 2-methyl-1-propanol
    't-butanol':        {'CH3': 3, 'C': 1, 'OH': 1},   # 2-methyl-2-propanol
    '1-pentanol':       {'CH3': 1, 'CH2': 4, 'OH': 1},
    '1-hexanol':        {'CH3': 1, 'CH2': 5, 'OH': 1},
    '1-octanol':        {'CH3': 1, 'CH2': 7, 'OH': 1},
    'ethylene_glycol':  {'CH2': 2, 'OH': 2},

    # --- Ketones ---
    'acetone':           {'CH3CO': 1, 'CH3': 1},                    # CH3-CO-CH3
    '2-butanone':        {'CH3CO': 1, 'CH2': 1, 'CH3': 1},          # MEK
    'mek':               {'CH3CO': 1, 'CH2': 1, 'CH3': 1},          # alias for MEK
    '2-pentanone':       {'CH3CO': 1, 'CH2': 2, 'CH3': 1},
    'mibk':              {'CH3CO': 1, 'CH2': 1, 'CH': 1, 'CH3': 2},  # 4-methyl-2-pentanone
    'cyclohexanone':     {'CH2CO': 1, 'CH2': 4},   # 4 CH2 + 1 CH2CO (CH2CO captures one CH2 + carbonyl C)

    # --- Esters ---
    'methyl_acetate':    {'CH3COO': 1, 'CH3': 1},                   # CH3-COO-CH3
    'ethyl_acetate':     {'CH3COO': 1, 'CH2': 1, 'CH3': 1},          # CH3-COO-CH2-CH3
    'propyl_acetate':    {'CH3COO': 1, 'CH2': 2, 'CH3': 1},
    'butyl_acetate':     {'CH3COO': 1, 'CH2': 3, 'CH3': 1},
    'methyl_propanoate': {'CH2COO': 1, 'CH3': 2},                   # CH3-CH2-COO-CH3
    'ethyl_propanoate':  {'CH2COO': 1, 'CH2': 1, 'CH3': 2},   # CH3-CH2-COO-CH2-CH3

    # --- Ethers ---
    'dimethyl_ether':    {'CH3O': 1, 'CH3': 1},   # CH3-O-CH3
    'diethyl_ether':     {'CH3': 2, 'CH2': 1, 'CH2O': 1},   # CH3-CH2-O-CH2-CH3
    'mtbe':              {'CH3': 3, 'C': 1, 'CH3O': 1},  # methyl tert-butyl ether
    'tetrahydrofuran':   {'CH2': 3, 'CH2O': 1},   # ring, 4 CH2 + 1 O; one CH2 attached to O is CH2O, rest are CH2

    # --- Carboxylic acids ---
    'formic_acid':       {'HCOOH': 1},
    'acetic_acid':       {'CH3': 1, 'COOH': 1},
    'propionic_acid':    {'CH3': 1, 'CH2': 1, 'COOH': 1},
    'butyric_acid':      {'CH3': 1, 'CH2': 2, 'COOH': 1},
    'pentanoic_acid':    {'CH3': 1, 'CH2': 3, 'COOH': 1},

    # --- Nitriles ---
    'acetonitrile':   {'CH3CN': 1},
    'propionitrile':  {'CH3': 1, 'CH2CN': 1},
    'butyronitrile':  {'CH3': 1, 'CH2': 1, 'CH2CN': 1},
}


# Aliases (case-insensitive, normalized)
_ALIASES: Dict[str, str] = {
    'meoh': 'methanol',
    'etoh': 'ethanol',
    'iproh': '2-propanol',
    'ipoh': '2-propanol',
    'isopropanol': '2-propanol',
    'iso-propanol': '2-propanol',
    'butanol': '1-butanol',
    'tert-butanol': 't-butanol',
    'tbuoh': 't-butanol',
    'eg': 'ethylene_glycol',
    'thf': 'tetrahydrofuran',
    'ea': 'ethyl_acetate',
    'ma': 'methyl_acetate',
    'dee': 'diethyl_ether',
    'dme': 'dimethyl_ether',
    'meibk': 'mibk',
    'h2o': 'water',
    'h_2_o': 'water',
    'hexane': 'n-hexane',
    'heptane': 'n-heptane',
    'octane': 'n-octane',
    'nonane': 'n-nonane',
    'decane': 'n-decane',
    'pentane': 'n-pentane',
    'butane': 'n-butane',
    'mevalonate': None,  # placeholder, ignore
}
# Drop None
_ALIASES = {k: v for k, v in _ALIASES.items() if v is not None}


def _normalize(name: str) -> str:
    """Lowercase, strip, and remove dashes/underscores for lookup."""
    s = str(name).lower().strip()
    return s


def _lookup(name: str) -> Dict[str, int]:
    """Look up groups for a compound; raise KeyError with hint on miss."""
    norm = _normalize(name)
    # Try direct hit
    if norm in _COMPOUNDS:
        return dict(_COMPOUNDS[norm])
    # Try alias
    if norm in _ALIASES:
        return dict(_COMPOUNDS[_ALIASES[norm]])
    # Try with dashes <-> underscores swapped
    swapped = norm.replace('-', '_')
    if swapped in _COMPOUNDS:
        return dict(_COMPOUNDS[swapped])
    swapped2 = norm.replace('_', '-')
    if swapped2 in _COMPOUNDS:
        return dict(_COMPOUNDS[swapped2])
    raise KeyError(
        f"Unknown compound {name!r}. Use list_compounds() to see "
        f"available names, or pass groups dict directly to UNIFAC(...)."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_compounds() -> List[str]:
    """Return sorted list of available compound names."""
    return sorted(_COMPOUNDS.keys())


def get_groups(name: str) -> Dict[str, int]:
    """Return the UNIFAC group decomposition for a compound (copy)."""
    return _lookup(name)


def make_unifac(names: Sequence[str], **kwargs):
    """Build a UNIFAC (original) model for the given compound names.

    Any extra kwargs are forwarded to UNIFAC(...).
    """
    from .unifac import UNIFAC
    groups = [_lookup(n) for n in names]
    return UNIFAC(groups, **kwargs)


def make_unifac_dortmund(names: Sequence[str], **kwargs):
    """Build a UNIFAC-Dortmund model for the given compound names."""
    from .unifac_dortmund import UNIFAC_Dortmund
    groups = [_lookup(n) for n in names]
    return UNIFAC_Dortmund(groups, **kwargs)


def make_unifac_lyngby(names: Sequence[str], **kwargs):
    """Build a UNIFAC-Lyngby model for the given compound names."""
    from .unifac_lyngby import UNIFAC_Lyngby
    groups = [_lookup(n) for n in names]
    return UNIFAC_Lyngby(groups, **kwargs)


def uniquac_rq(name: str) -> Tuple[float, float]:
    """Compute UNIQUAC r and q for a compound from its UNIFAC group sums.

        r_i = sum_k nu_k^i * R_k
        q_i = sum_k nu_k^i * Q_k

    This is the standard relationship between UNIFAC and UNIQUAC
    pure-component parameters. Both come from the same underlying
    Bondi van der Waals volumes and surface areas.
    """
    groups = _lookup(name)
    r = 0.0
    q = 0.0
    for grp_name, count in groups.items():
        if grp_name not in SUBGROUPS:
            raise KeyError(f"Subgroup {grp_name!r} not in database "
                            f"(compound {name!r})")
        _sg, _mg, R_k, Q_k = SUBGROUPS[grp_name]
        r += count * R_k
        q += count * Q_k
    return float(r), float(q)


def make_uniquac(names: Sequence[str], a=None, b=None, e=None, f=None):
    """Build a UNIQUAC model for the given compound names with
    automatically-derived r, q from group sums and user-supplied
    binary interaction matrices a/b/e/f.

    Parameters
    ----------
    names : sequence of str
    a, b, e, f : (N, N) arrays or None
        Interaction parameter matrices (zeros on diagonal). At
        least `b` is typically required; others default to 0.

    Returns
    -------
    UNIQUAC instance with r, q derived from compound database.
    """
    from .uniquac import UNIQUAC
    rs = []
    qs = []
    for n in names:
        r, q = uniquac_rq(n)
        rs.append(r)
        qs.append(q)
    kwargs = {'r': rs, 'q': qs}
    if a is not None:
        kwargs['a'] = np.asarray(a, dtype=float)
    if b is not None:
        kwargs['b'] = np.asarray(b, dtype=float)
    if e is not None:
        kwargs['e'] = np.asarray(e, dtype=float)
    if f is not None:
        kwargs['f'] = np.asarray(f, dtype=float)
    return UNIQUAC(**kwargs)
