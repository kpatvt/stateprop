"""Coverage reporting and validation tools for LLE-UNIFAC.

The bundled `LLE_OVERRIDES` set in `unifac_lle.py` covers a small
fraction of the full Magnussen (1981) parameter table -- only the 4
most-cited aqueous-organic main-group pairs. For systems whose
relevant main-group pairs are NOT in `LLE_OVERRIDES`, calculations
fall back to standard VLE-fitted values (Hansen 1991), which is
methodologically imperfect.

This module helps users understand and work with that limitation:

- `lle_coverage()` reports which main-group pairs in a user's system
  are LLE-fitted vs falling back to VLE.

- `lle_coverage_summary()` returns a human-readable diagnostic string.

- `load_overrides_from_json()` / `save_overrides_to_json()` provide a
  clean import/export path so users can ship/share custom parameter
  sets (e.g., from full Magnussen table, DDBST consortium data, or
  per-publication fits).

- `validate_against_benchmarks()` runs LLE flash on a small set of
  canonical aqueous-organic LLE systems with published mutual
  solubilities, returning a side-by-side comparison.

The benchmarks let users see WHICH systems the bundled set predicts
well and WHICH need user-supplied extensions or per-system fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from . import unifac_database as _vle_db
from .unifac_lle import LLE_OVERRIDES


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------

# Map main-group id to a human-friendly name (taken from a representative
# subgroup name where possible).
_MAIN_NAMES = {
    1: 'CH2 (alkane)',
    2: 'C=C (alkene)',
    3: 'ACH (aromatic)',
    4: 'ACCH2 (aromatic-alkyl)',
    5: 'OH (alcohol)',
    6: 'CH3OH (methanol)',
    7: 'H2O (water)',
    9: 'CH2CO (ketone)',
    11: 'CCOO (ester)',
    13: 'CH2O (ether)',
    19: 'CCN (nitrile)',
    20: 'COOH (carboxylic acid)',
}


@dataclass
class CoverageReport:
    """Coverage of LLE_OVERRIDES against a system's main-group pairs."""
    main_groups: List[int]
    pairs: List[Tuple[int, int]]
    fitted_pairs: List[Tuple[int, int]]
    unfitted_pairs: List[Tuple[int, int]]
    fraction_fitted: float

    def __str__(self):
        return lle_coverage_summary(self)


def _main_groups_in_system(subgroups_per_component) -> List[int]:
    """Return sorted list of main-group ids appearing in the system."""
    seen = set()
    for groups in subgroups_per_component:
        for sg_name in groups.keys():
            if sg_name not in _vle_db.SUBGROUPS:
                continue  # Will raise later in UNIFAC; ignore here
            _sg, mg, _R, _Q = _vle_db.SUBGROUPS[sg_name]
            seen.add(int(mg))
    return sorted(seen)


def lle_coverage(subgroups_per_component,
                  overrides: Optional[Dict] = None) -> CoverageReport:
    """Report LLE-coverage of a system's main-group pair interactions.

    For an N-component system with main groups {m_1, ..., m_M}, returns
    which of the M*(M-1)/2 unordered pairs are LLE-fitted (in `overrides`)
    and which fall back to VLE values.

    Parameters
    ----------
    subgroups_per_component : list of dict
        Same as input to UNIFAC: {subgroup_name: count} per component.
    overrides : dict, optional
        Override mapping (m, n) -> (a_mn, a_nm). Defaults to
        `LLE_OVERRIDES`.

    Returns
    -------
    CoverageReport
        - main_groups: sorted list of main-group ids in the system
        - pairs: all unordered (m, n) pairs (m < n)
        - fitted_pairs: pairs present in `overrides`
        - unfitted_pairs: pairs falling back to VLE
        - fraction_fitted: len(fitted) / len(pairs), or 1.0 if no pairs
    """
    if overrides is None:
        overrides = LLE_OVERRIDES
    mgs = _main_groups_in_system(subgroups_per_component)
    pairs: List[Tuple[int, int]] = []
    for i, m in enumerate(mgs):
        for n in mgs[i + 1:]:
            pairs.append((m, n))
    # An overrides entry (m,n) covers the pair (m,n); we accept either
    # ordering as evidence of LLE fitting.
    fitted = []
    unfitted = []
    for pair in pairs:
        m, n = pair
        if (m, n) in overrides or (n, m) in overrides:
            fitted.append(pair)
        else:
            unfitted.append(pair)
    frac = len(fitted) / len(pairs) if pairs else 1.0
    return CoverageReport(
        main_groups=mgs, pairs=pairs,
        fitted_pairs=fitted, unfitted_pairs=unfitted,
        fraction_fitted=frac,
    )


def lle_coverage_summary(report: CoverageReport) -> str:
    """Human-readable diagnostic string for a CoverageReport."""
    lines = []
    lines.append(f"LLE coverage: {len(report.fitted_pairs)}/"
                  f"{len(report.pairs)} pairs LLE-fitted "
                  f"({100 * report.fraction_fitted:.0f}%)")
    if report.fitted_pairs:
        lines.append("  LLE-fitted (Magnussen 1981):")
        for m, n in report.fitted_pairs:
            mn = _MAIN_NAMES.get(m, f'main {m}')
            nn = _MAIN_NAMES.get(n, f'main {n}')
            lines.append(f"    ({m},{n})  {mn} <-> {nn}")
    if report.unfitted_pairs:
        lines.append("  Falling back to VLE values (Hansen 1991):")
        for m, n in report.unfitted_pairs:
            mn = _MAIN_NAMES.get(m, f'main {m}')
            nn = _MAIN_NAMES.get(n, f'main {n}')
            lines.append(f"    ({m},{n})  {mn} <-> {nn}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON import/export
# ---------------------------------------------------------------------------


def save_overrides_to_json(overrides: Dict[Tuple[int, int],
                                              Tuple[float, float]],
                              path: str) -> None:
    """Save an overrides dict to a JSON file.

    JSON keys are stored as "m,n" strings since JSON keys must be
    strings.
    """
    out = {f"{m},{n}": [float(a), float(b)]
           for (m, n), (a, b) in overrides.items()}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)


def load_overrides_from_json(path: str) -> Dict[Tuple[int, int],
                                                  Tuple[float, float]]:
    """Load an overrides dict from a JSON file.

    JSON keys "m,n" are converted back to (int, int) tuples.
    """
    with open(path, 'r') as f:
        raw = json.load(f)
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for k, v in raw.items():
        parts = k.split(',')
        if len(parts) != 2:
            raise ValueError(f"Bad key in JSON: {k!r}")
        m, n = int(parts[0]), int(parts[1])
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError(f"Bad value for {k!r}: {v!r}")
        out[(m, n)] = (float(v[0]), float(v[1]))
    return out


# ---------------------------------------------------------------------------
# Validation harness: canonical LLE benchmarks
# ---------------------------------------------------------------------------

# A small, curated set of binary aqueous-organic LLE systems with
# published mutual solubilities at 298.15 K. Sources for the published
# values are textbook compilations (DDBST online, Sorensen-Arlt
# DECHEMA Vol. V Part 1) which are widely cross-referenced; the
# numbers cited below are commonly accepted benchmarks within ~5-10%
# tolerance.
#
# Format: name -> {
#     'components': (group_dict_1, group_dict_2),
#     'T': K,
#     'x_org_in_water': mole fraction of organic in water-rich phase,
#     'x_org_in_org':   mole fraction of organic in organic-rich phase,
# }
#
# The component ordering is (organic, water).

LLE_BENCHMARKS = {
    '1-butanol_water': {
        'components': (
            {'CH3': 1, 'CH2': 3, 'OH': 1},
            {'H2O': 1},
        ),
        'T': 298.15,
        'x_org_in_water': 0.0185,    # water-rich x_BuOH
        'x_org_in_org': 0.485,       # BuOH-rich x_BuOH
        'description': 'n-butanol + water',
    },
    '1-pentanol_water': {
        'components': (
            {'CH3': 1, 'CH2': 4, 'OH': 1},
            {'H2O': 1},
        ),
        'T': 298.15,
        'x_org_in_water': 0.0050,
        'x_org_in_org': 0.620,
        'description': 'n-pentanol + water',
    },
    '1-hexanol_water': {
        'components': (
            {'CH3': 1, 'CH2': 5, 'OH': 1},
            {'H2O': 1},
        ),
        'T': 298.15,
        'x_org_in_water': 0.00115,
        'x_org_in_org': 0.690,
        'description': 'n-hexanol + water',
    },
    'benzene_water': {
        'components': (
            {'ACH': 6},
            {'H2O': 1},
        ),
        'T': 298.15,
        'x_org_in_water': 0.00041,
        'x_org_in_org': 0.9974,
        'description': 'benzene + water (very low mutual solubility)',
    },
}


@dataclass
class BenchmarkResult:
    """Result of running one LLE benchmark."""
    name: str
    description: str
    T: float
    published_x_org_in_water: float
    published_x_org_in_org: float
    predicted_x_org_in_water: Optional[float]
    predicted_x_org_in_org: Optional[float]
    abs_error_water_phase: Optional[float]
    abs_error_org_phase: Optional[float]
    converged: bool
    notes: str = ''


def validate_against_benchmarks(
    benchmarks: Optional[Dict] = None,
    overrides: Optional[Dict] = None,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """Run LLE flash on canonical aqueous-organic systems and compare
    predictions against published mutual solubilities.

    Parameters
    ----------
    benchmarks : dict, optional
        Benchmark dict (default: `LLE_BENCHMARKS`).
    overrides : dict, optional
        LLE overrides to use (default: bundled `LLE_OVERRIDES`).
    verbose : bool
        If True, print per-system results.

    Returns
    -------
    list of BenchmarkResult
    """
    from .unifac_lle import UNIFAC_LLE
    from .lle import LLEFlash

    if benchmarks is None:
        benchmarks = LLE_BENCHMARKS

    results: List[BenchmarkResult] = []

    for name, bench in benchmarks.items():
        groups_org, groups_wat = bench['components']
        T = bench['T']
        x_org_water_pub = bench['x_org_in_water']
        x_org_org_pub = bench['x_org_in_org']
        description = bench.get('description', name)

        if overrides is None:
            uf = UNIFAC_LLE([groups_org, groups_wat])
        else:
            uf = UNIFAC_LLE([groups_org, groups_wat], extra_overrides=overrides)

        # Initial guesses from published values
        x1_g = [x_org_water_pub, 1.0 - x_org_water_pub]   # water-rich
        x2_g = [x_org_org_pub, 1.0 - x_org_org_pub]       # organic-rich
        # Equimolar feed
        z = [0.5, 0.5]

        try:
            lle = LLEFlash(uf)
            r = lle.solve(T, z, x1_guess=x1_g, x2_guess=x2_g)
            # Identify which phase is which (component 0 is the organic)
            if r.x1[0] < r.x2[0]:
                # x1 is water-rich, x2 is organic-rich
                pred_water = float(r.x1[0])
                pred_org = float(r.x2[0])
            else:
                pred_water = float(r.x2[0])
                pred_org = float(r.x1[0])
            # Reject trivial (single-phase) solutions
            if abs(pred_water - pred_org) < 1e-3:
                results.append(BenchmarkResult(
                    name=name, description=description, T=T,
                    published_x_org_in_water=x_org_water_pub,
                    published_x_org_in_org=x_org_org_pub,
                    predicted_x_org_in_water=None,
                    predicted_x_org_in_org=None,
                    abs_error_water_phase=None,
                    abs_error_org_phase=None,
                    converged=False,
                    notes='collapsed to single phase',
                ))
                continue
            err_water = abs(pred_water - x_org_water_pub)
            err_org = abs(pred_org - x_org_org_pub)
            results.append(BenchmarkResult(
                name=name, description=description, T=T,
                published_x_org_in_water=x_org_water_pub,
                published_x_org_in_org=x_org_org_pub,
                predicted_x_org_in_water=pred_water,
                predicted_x_org_in_org=pred_org,
                abs_error_water_phase=err_water,
                abs_error_org_phase=err_org,
                converged=True,
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                name=name, description=description, T=T,
                published_x_org_in_water=x_org_water_pub,
                published_x_org_in_org=x_org_org_pub,
                predicted_x_org_in_water=None,
                predicted_x_org_in_org=None,
                abs_error_water_phase=None,
                abs_error_org_phase=None,
                converged=False,
                notes=f'flash failed: {str(e)[:60]}',
            ))

    if verbose:
        print(format_benchmark_results(results))

    return results


def format_benchmark_results(results: List[BenchmarkResult]) -> str:
    """Format a list of BenchmarkResult into a readable table."""
    lines = []
    lines.append(f"{'System':<32s} {'T':>5s}  {'pub_water':>10s} "
                  f"{'pred_water':>11s}  {'pub_org':>10s} {'pred_org':>10s}  "
                  f"{'status':<24s}")
    lines.append("-" * 112)
    for r in results:
        if r.converged:
            pred_w = f"{r.predicted_x_org_in_water:.4f}"
            pred_o = f"{r.predicted_x_org_in_org:.4f}"
            status = (f"err_water={r.abs_error_water_phase:.3f}, "
                       f"err_org={r.abs_error_org_phase:.3f}")
        else:
            pred_w = '   -   '
            pred_o = '   -   '
            status = r.notes
        lines.append(
            f"{r.description[:32]:<32s} "
            f"{r.T:>5.1f}  "
            f"{r.published_x_org_in_water:>10.4f} {pred_w:>11s}  "
            f"{r.published_x_org_in_org:>10.4f} {pred_o:>10s}  "
            f"{status:<24s}"
        )
    return "\n".join(lines)
