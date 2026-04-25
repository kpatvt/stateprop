"""LLE-fitted UNIFAC parameter database.

Standard UNIFAC parameters (from Hansen et al. 1991, in
`stateprop.activity.unifac_database`) are fitted predominantly to
vapor-liquid equilibrium (VLE) data and give qualitatively wrong
liquid-liquid equilibrium (LLE) predictions for many systems.

The LLE-UNIFAC parameter set was published by:

    Magnussen, T., Rasmussen, P., Fredenslund, A. (1981)
    "UNIFAC parameter table for prediction of liquid-liquid
    equilibria", Ind. Eng. Chem. Process Des. Dev. 20, 331-339.

This module provides:

1. A mechanism for OVERRIDING specific main-group interaction
   parameters with their LLE-fitted values, while keeping the
   rest of the standard UNIFAC parameter table.
2. A starter set of widely-cited LLE-fitted overrides for the
   most common interactions in aqueous-organic systems
   (CH2/OH/H2O/ACH).
3. A `make_lle_database()` factory that builds a custom database
   module satisfying the structure expected by `UNIFAC(database=...)`.
4. A `UNIFAC_LLE` class wrapping the standard UNIFAC engine but
   pointed at the LLE database.

Scope and caveats
=================

**The bundled override set is intentionally narrow.** Building a
complete LLE-UNIFAC parameter table requires the full Magnussen
(1981) paper which is paywalled and not reproducible here without
risk of typographical errors. The provided 4 main-group pairs
cover the most common aqueous-organic interactions (alcohol-water,
alkane-water, aromatic-water, alkane-alcohol) and demonstrate the
override mechanism. Users with access to the original paper or to
DDBST data can extend the database via `LLE_OVERRIDES_USER` or by
constructing a custom database module.

Parameters not in `LLE_OVERRIDES` fall back to the standard UNIFAC
values from Hansen 1991. This is a pragmatic choice -- the
alternative (raising on missing pairs) would make the model
unusable for any system not entirely covered by the override set.
Users should be aware that mixing LLE-fitted and VLE-fitted
interactions in one calculation is methodologically imperfect
and should validate against published LLE data for their system.

Validity range
==============

Magnussen et al. fitted their parameters using LLE data in the
narrow range 283-313 K. Predictions outside this range degrade.
For T well above 313 K, consider modified UNIFAC (Dortmund) or
fitting your own parameters with `regression.regress_lle()`.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple
from . import unifac_database as _vle_db

# ---------------------------------------------------------------------------
# LLE-fitted main-group interaction parameters from Magnussen 1981.
#
# Format: (main_m, main_n) -> (a_mn, a_nm) in K, with main group indices
# matching `unifac_database.SUBGROUPS`:
#   1 = CH2, 3 = ACH, 5 = OH, 7 = H2O.
#
# These four pairs cover the most common aqueous-organic interactions
# and are widely cross-referenced in chemical engineering textbooks
# (Walas 1985 Appendix E; Sandler 4th ed. Section 9.5; SciELO and
# multiple subsequent LLE papers citing Magnussen).
# ---------------------------------------------------------------------------

LLE_OVERRIDES: Dict[Tuple[int, int], Tuple[float, float]] = {
    # CH2 (1) <-> OH (5)
    (1, 5): (644.6, 328.2),
    # CH2 (1) <-> H2O (7)
    (1, 7): (1300.0, 342.4),
    # ACH (3) <-> H2O (7)
    (3, 7): (859.4, 362.3),
    # OH (5) <-> H2O (7) -- the most consequential for alcohol-water LLE.
    # Standard VLE: 353.5 / -229.1; LLE: 155.6 / -49.29.
    (5, 7): (155.6, -49.29),
}


def make_lle_database(extra_overrides: Optional[Dict[Tuple[int, int],
                                                      Tuple[float, float]]] = None):
    """Build a UNIFAC database object with LLE-fitted main-group overrides.

    The returned object exposes `SUBGROUPS` (identical to the standard
    UNIFAC database) and `A_MAIN` (with LLE overrides applied to the
    standard table). It can be passed to `UNIFAC(database=...)`.

    Parameters
    ----------
    extra_overrides : dict, optional
        Additional (m, n) -> (a_mn, a_nm) overrides to apply on top
        of `LLE_OVERRIDES`. Useful for users with access to the full
        Magnussen 1981 table or other LLE-fitted parameter sets.

    Returns
    -------
    object with .SUBGROUPS and .A_MAIN attributes
    """

    class _LLEDatabase:
        SUBGROUPS = _vle_db.SUBGROUPS
        # Deep-copy the standard A_MAIN so we don't mutate the original
        A_MAIN = copy.deepcopy(_vle_db.A_MAIN)

    db = _LLEDatabase
    # Apply the bundled LLE overrides
    for (m, n), (a_mn, a_nm) in LLE_OVERRIDES.items():
        if m in db.A_MAIN:
            db.A_MAIN[m][n] = float(a_mn)
        if n in db.A_MAIN:
            db.A_MAIN[n][m] = float(a_nm)
    # Apply user-provided extras
    if extra_overrides:
        for (m, n), (a_mn, a_nm) in extra_overrides.items():
            if m in db.A_MAIN:
                db.A_MAIN[m][n] = float(a_mn)
            if n in db.A_MAIN:
                db.A_MAIN[n][m] = float(a_nm)
    return db


# Default LLE database (bundled overrides only)
LLE_DATABASE = make_lle_database()


# ---------------------------------------------------------------------------
# UNIFAC_LLE class
# ---------------------------------------------------------------------------

# Imported lazily inside the class to avoid circular imports at module load.


class UNIFAC_LLE:
    """UNIFAC activity coefficient model with LLE-fitted parameters.

    Drop-in replacement for `UNIFAC` that uses the LLE-fitted
    main-group interaction parameters from Magnussen et al. (1981)
    where available, falling back to VLE-fitted parameters from
    Hansen et al. (1991) for pairs not in the LLE override set.

    Parameters
    ----------
    subgroups_per_component : list of dict
        Same as `UNIFAC`: each dict maps subgroup name to count.
    extra_overrides : dict, optional
        Additional (m, n) -> (a_mn, a_nm) overrides to apply.

    Examples
    --------
    >>> from stateprop.activity.unifac_lle import UNIFAC_LLE
    >>> uf = UNIFAC_LLE([{'CH3': 1, 'CH2': 3, 'OH': 1}, {'H2O': 1}])
    >>> uf.gammas(298.15, [0.5, 0.5])
    """

    def __init__(self, subgroups_per_component, extra_overrides=None,
                  **kwargs):
        # Lazy import to avoid circular dependency
        from .unifac import UNIFAC
        if extra_overrides:
            db = make_lle_database(extra_overrides=extra_overrides)
        else:
            db = LLE_DATABASE
        # Build the inner UNIFAC with the LLE database
        self._unifac = UNIFAC(subgroups_per_component, database=db, **kwargs)
        # Expose key attributes for compatibility
        self.N = self._unifac.N
        self.subgroups_per_component = list(subgroups_per_component)

    # Delegate the activity-coefficient interface
    def gammas(self, T, x):
        return self._unifac.gammas(T, x)

    def dlngammas_dT(self, T, x):
        return self._unifac.dlngammas_dT(T, x)

    def hE(self, T, x):
        return self._unifac.hE(T, x)

    def gE(self, T, x):
        return self._unifac.gE(T, x)

    def sE(self, T, x):
        return self._unifac.sE(T, x)

    def cpE(self, T, x):
        return self._unifac.cpE(T, x)

    def __repr__(self):
        return (f"UNIFAC_LLE(N={self.N}, "
                f"groups={self.subgroups_per_component})")
