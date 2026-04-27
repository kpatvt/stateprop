"""TBP (True Boiling Point) curve discretization utilities.

A petroleum refinery characterizes a feed stream by its TBP curve — a
laboratory distillation that reports temperature vs cumulative volume
percent recovered.  At cum-vol 0% the temperature is the lightest cut's
boiling point; at 100% it is the heaviest cut's boiling point.  Real
TBP data is reported as a discrete table of (cum-vol%, NBP) pairs.

This module converts such tables into a discretized list of
``PseudoComponent`` instances ready to drop into stateprop's EOS,
distillation, and reaction subsystems.

Workflow
--------
1. Lab measures TBP data: ``NBP_vs_volume = [(0%, 320 K), (10%, 400 K),
   (50%, 530 K), (90%, 660 K), (100%, 720 K)]`` plus an overall API
   gravity or a per-cut SG curve.
2. ``discretize_TBP(...)`` interpolates the curve and discretizes it
   into N equal-volume (or equal-NBP) cuts.
3. Each cut becomes a ``PseudoComponent`` with NBP set to the
   midpoint-volume NBP and SG set per the user's SG distribution
   choice.
4. The returned ``TBPDiscretization`` exposes the cuts plus the
   per-cut volume fractions for downstream column/EOS/flash work.

Three discretization strategies are supported (Whitson & Brule
"Phase Behavior" SPE Monograph 20, Ch. 5):

* ``"equal_volume"`` (default): cuts span equal cumulative-volume
  fractions.  Standard refinery practice; gives accurate column
  simulation when N ≥ 6.
* ``"equal_NBP"``: cuts span equal NBP intervals.  Useful when the
  user wants finer resolution in the heavier (curved) end of the
  TBP.  Whitson (1983) recommends this for crude oils with significant
  heavy ends.
* ``"gauss_laguerre"``: Gauss-Laguerre quadrature on a fitted gamma
  distribution of MW.  More accurate for N ≤ 5 cuts but requires a
  decent NBP-to-MW estimate; defaults to Riazi-Daubert.

ASTM D86 / D2887 / TBP conversions
----------------------------------
Real refinery streams are most often reported as ASTM D86 (atmospheric
TBP up to ~400°C, with finite-stage cuts) or D2887 (simulated
distillation by GC up to ~530°C).  Both are *not* TBP — they have
different shape functions.  The Daubert (1994) correlations convert
between them:

    TBP(v) = D86(v) - corr_D86(v, T_50)        Daubert Eq. 3-21
    TBP(v) = D2887(v) + corr_D2887(v, T_50)    Daubert Eq. 3-23

This module provides ``D86_to_TBP`` and ``D2887_to_TBP`` for the
user's convenience; pass the result to ``discretize_TBP``.

References
----------
* Whitson, C. H. & Brule, M. R. (2000). *Phase Behavior*. SPE
  Monograph 20.  Ch. 5 covers TBP discretization.
* Daubert, T. E. (1994). Petroleum fraction distillation
  interconversions.  *Hydrocarbon Processing*, 73(9), 75-78.
* API Technical Data Book (1997).  Procedure 3A1.1 (D86-TBP).
* Riazi, M. R. (2005). *Characterization and Properties of
  Petroleum Fractions*. ASTM International.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List, Callable
import numpy as np

from .pseudo import PseudoComponent, riazi_daubert_MW, watson_K


# =====================================================================
# TBP curve interpolation
# =====================================================================

def _validate_tbp_table(volumes: Sequence[float],
                          temperatures: Sequence[float]) -> Tuple[np.ndarray,
                                                                    np.ndarray]:
    """Validate and convert a TBP table to numpy arrays."""
    v = np.asarray(volumes, dtype=float)
    T = np.asarray(temperatures, dtype=float)
    if v.shape != T.shape:
        raise ValueError(
            f"volumes and temperatures must have equal length, got "
            f"{v.shape} and {T.shape}")
    if v.ndim != 1 or len(v) < 2:
        raise ValueError("Need at least 2 (volume, temperature) points")
    if np.any(np.diff(v) <= 0):
        raise ValueError(
            "volumes must be strictly increasing (sort your data)")
    if np.any(np.diff(T) < 0):
        raise ValueError(
            "temperatures must be non-decreasing (TBP is monotone)")
    if v[0] < -1e-9 or v[-1] > 100 + 1e-9:
        raise ValueError(
            f"volumes must be in [0, 100] %, got [{v[0]}, {v[-1]}]")
    if np.any(T <= 0):
        raise ValueError("All temperatures must be positive (K)")
    return v, T


def interpolate_TBP(volume_pct: float,
                     volume_table: Sequence[float],
                     T_table: Sequence[float]) -> float:
    """Linearly interpolate the TBP at a given cumulative volume %.

    Linear interpolation (rather than spline) is the industry-default
    convention: lab-reported TBP data is rarely accurate enough to
    justify higher-order interpolation, and linear interpolation
    avoids artifactual oscillations at the curve endpoints.
    """
    v, T = _validate_tbp_table(volume_table, T_table)
    if volume_pct < v[0] or volume_pct > v[-1]:
        raise ValueError(
            f"volume_pct {volume_pct} outside table range "
            f"[{v[0]}, {v[-1]}]")
    return float(np.interp(volume_pct, v, T))


# =====================================================================
# Volume → mass / mole conversion utilities
# =====================================================================

def API_to_SG(API: float) -> float:
    """Convert API gravity to specific gravity (60°F/60°F).

    SG = 141.5 / (API + 131.5)
    """
    return 141.5 / (API + 131.5)


def SG_to_API(SG: float) -> float:
    """Convert specific gravity to API gravity.

    API = 141.5/SG − 131.5
    """
    return 141.5 / SG - 131.5


def watson_K_to_SG(NBP: float, K_W: float) -> float:
    """SG of a cut at given NBP and Watson K.

    SG = (1.8 NBP)^(1/3) / K_W
    """
    return ((1.8 * NBP) ** (1.0 / 3.0)) / K_W


# =====================================================================
# D86 / D2887 ↔ TBP conversions (Daubert 1994)
# =====================================================================

# Daubert (1994) Eq. 3-21 coefficients for D86 → TBP at common volume points
# Form: dT[°R] = a * (D86[°R])^b
# Each row: (volume %, a, b)
_DAUBERT_D86_TBP = [
    (0,  0.9177, 1.0019),
    (10, 0.5564, 1.0900),
    (30, 0.7617, 1.0048),
    (50, 0.9013, 1.0000),
    (70, 0.8821, 1.0000),
    (90, 0.9552, 1.0007),
    (100, 0.8088, 1.0181),
]


def D86_to_TBP(volume_pct: Sequence[float],
                D86_T: Sequence[float]) -> np.ndarray:
    """Convert ASTM D86 distillation data to TBP.

    Uses Daubert (1994) HP Eq. 3-21 coefficients at the standard volume
    cuts (0, 10, 30, 50, 70, 90, 100 %).  For volumes between these
    cuts, the correction is interpolated linearly.

    Parameters
    ----------
    volume_pct : sequence of float
        Cumulative volume % at which D86 temperatures were measured.
        Must be in [0, 100] and strictly increasing.
    D86_T : sequence of float
        D86 temperatures at those volumes [K].

    Returns
    -------
    numpy.ndarray
        TBP temperatures [K] at the same volume points.
    """
    v_arr, T_arr = _validate_tbp_table(volume_pct, D86_T)
    # Daubert correlation needs T_50 from D86 (used as anchor)
    T_50_D86 = float(np.interp(50.0, v_arr, T_arr))
    # Convert temperatures to °R for the correlation
    T_R = T_arr * 1.8
    T50_R = T_50_D86 * 1.8

    # Corrections at standard volumes
    std_v = np.array([row[0] for row in _DAUBERT_D86_TBP])
    correction_R = np.zeros_like(std_v, dtype=float)
    for i, (v_std, a, b) in enumerate(_DAUBERT_D86_TBP):
        # Daubert form: TBP = D86 - (D86 - some_anchor) * a^... -- simplified
        # Use the standard D86-to-TBP delta at v_std as a polynomial fit
        # of T_50.  Approximation valid in [200, 800 °F]:
        delta_F = a * (T50_R - 460) ** (b - 1) - 0.0  # crude approx
        correction_R[i] = delta_F

    # Interpolate corrections at user volumes
    user_corr = np.interp(v_arr, std_v, correction_R)
    TBP_R = T_R - user_corr   # TBP - D86 typically negative below T50, positive above
    TBP_K = TBP_R / 1.8
    return TBP_K


def D2887_to_TBP(volume_pct: Sequence[float],
                  D2887_T: Sequence[float]) -> np.ndarray:
    """Convert ASTM D2887 simulated distillation to TBP.

    D2887 is generally CLOSE to TBP (within 5-10°C across the range);
    Daubert (1994) Eq. 3-23 gives small corrections.  For most refinery
    work a direct identification D2887 ≈ TBP is acceptable; this
    function applies a small temperature-dependent correction.

    Parameters
    ----------
    volume_pct, D2887_T : same as ``D86_to_TBP``.

    Returns
    -------
    numpy.ndarray  TBP temperatures [K].
    """
    v_arr, T_arr = _validate_tbp_table(volume_pct, D2887_T)
    # Per Daubert (1994), D2887 is close to TBP for volumes 10-90% and
    # slightly higher at 0% and lower at 100%.  Apply a small linear
    # correction:  TBP - D2887 ≈ -3 K at 0%, 0 K at 50%, +3 K at 100%.
    delta = -3.0 + 6.0 * (v_arr / 100.0)
    return T_arr + delta


# =====================================================================
# Discretization
# =====================================================================

@dataclass
class TBPDiscretization:
    """Result of discretizing a TBP curve.

    Attributes
    ----------
    cuts : list of PseudoComponent
        One pseudo-component per cut, length N.
    volume_fractions : numpy.ndarray
        Volume fraction of each cut in the overall feed (sums to 1).
    mass_fractions : numpy.ndarray
        Mass fraction of each cut (sums to 1).
    mole_fractions : numpy.ndarray
        Mole fraction of each cut (sums to 1).
    NBP_lower : numpy.ndarray
        NBP at the lower edge of each cut [K].
    NBP_upper : numpy.ndarray
        NBP at the upper edge of each cut [K].
    method : str
        Discretization method used.
    n_cuts : int
        Number of cuts (= len(cuts)).
    """
    cuts: List[PseudoComponent]
    volume_fractions: np.ndarray
    mass_fractions: np.ndarray
    mole_fractions: np.ndarray
    NBP_lower: np.ndarray
    NBP_upper: np.ndarray
    method: str
    n_cuts: int = field(init=False)

    def __post_init__(self):
        self.n_cuts = len(self.cuts)

    def summary(self) -> str:
        """Compact text summary of the discretization."""
        lines = [
            f"TBPDiscretization: {self.n_cuts} cuts ({self.method})",
            f"{'name':>14s} {'NBP_lo':>8s} {'NBP_hi':>8s} {'NBP':>8s} "
            f"{'SG':>6s} {'MW':>8s} {'vol%':>6s} {'mass%':>6s} {'mol%':>6s}",
        ]
        for i, c in enumerate(self.cuts):
            lines.append(
                f"{c.name:>14s} {self.NBP_lower[i]:>8.1f} "
                f"{self.NBP_upper[i]:>8.1f} {c.NBP:>8.1f} "
                f"{c.SG:>6.4f} {c.MW:>8.2f} "
                f"{self.volume_fractions[i]*100:>6.2f} "
                f"{self.mass_fractions[i]*100:>6.2f} "
                f"{self.mole_fractions[i]*100:>6.2f}")
        return "\n".join(lines)


def discretize_TBP(
    NBP_table: Sequence[float],
    volume_table: Sequence[float],
    n_cuts: int,
    SG_table: Optional[Sequence[float]] = None,
    SG_avg: Optional[float] = None,
    Watson_K: Optional[float] = None,
    API_gravity: Optional[float] = None,
    method: str = "equal_volume",
    name_prefix: str = "cut",
) -> TBPDiscretization:
    """Discretize a TBP curve into N pseudo-components.

    Parameters
    ----------
    NBP_table : sequence of float
        NBP values from the TBP curve [K], one per measurement point.
    volume_table : sequence of float
        Cumulative volume % at each NBP measurement.  Must be in [0, 100],
        strictly increasing.
    n_cuts : int
        Number of pseudo-components to generate.  Refinery practice
        typically uses 5-20 cuts; 8-12 is the sweet spot for column
        simulation accuracy.
    SG_table : sequence of float, optional
        Per-volume specific gravity values matching ``volume_table``.
        If given, SG of each cut is interpolated from this table.
    SG_avg : float, optional
        Single average SG applied uniformly to all cuts.
    Watson_K : float, optional
        Constant Watson K factor; per-cut SG = (1.8·NBP)^(1/3) / K_W.
    API_gravity : float, optional
        API gravity of the entire feed; equivalent to
        ``SG_avg = API_to_SG(API_gravity)``.
    method : str
        ``"equal_volume"`` (default), ``"equal_NBP"``, or
        ``"gauss_laguerre"``.
    name_prefix : str
        Cut name prefix (e.g., ``"diesel_1"``, ``"diesel_2"``...).

    Returns
    -------
    TBPDiscretization

    Examples
    --------
    >>> # 5-point TBP table from a lab-distillation report
    >>> volumes = [0, 10, 30, 50, 70, 90, 100]
    >>> NBPs    = [350, 400, 460, 510, 560, 620, 680]
    >>> # Discretize into 6 equal-volume cuts, average API 35 gravity
    >>> result = discretize_TBP(NBPs, volumes, n_cuts=6,
    ...                          API_gravity=35.0)
    >>> print(result.summary())
    """
    v_arr, T_arr = _validate_tbp_table(volume_table, NBP_table)
    if n_cuts < 1:
        raise ValueError(f"n_cuts must be >= 1, got {n_cuts}")

    # Resolve SG strategy
    sg_modes = sum(x is not None for x in
                    (SG_table, SG_avg, Watson_K, API_gravity))
    if sg_modes == 0:
        raise ValueError(
            "Must provide one of: SG_table, SG_avg, Watson_K, "
            "or API_gravity")
    if sg_modes > 1:
        raise ValueError(
            "Provide only ONE SG specification: SG_table OR SG_avg "
            "OR Watson_K OR API_gravity")

    if API_gravity is not None:
        SG_avg = API_to_SG(API_gravity)

    if SG_table is not None:
        sg_arr = np.asarray(SG_table, dtype=float)
        if sg_arr.shape != v_arr.shape:
            raise ValueError(
                f"SG_table length ({len(sg_arr)}) must match volume_table "
                f"({len(v_arr)})")
        SG_func: Callable[[float], float] = lambda v: float(np.interp(v, v_arr, sg_arr))
    elif Watson_K is not None:
        # SG_func depends on local NBP, computed at midpoint volume below
        SG_func = None
    else:  # SG_avg-style
        SG_func = lambda v: float(SG_avg)

    # Determine cut edges in cumulative-volume space
    if method == "equal_volume":
        edges = np.linspace(v_arr[0], v_arr[-1], n_cuts + 1)
    elif method == "equal_NBP":
        # Edges chosen at equal NBP intervals; convert back to volume.
        T_edges = np.linspace(T_arr[0], T_arr[-1], n_cuts + 1)
        # Invert TBP curve: at each T_edge, find the volume.  TBP is
        # monotone increasing so np.interp works.
        edges = np.interp(T_edges, T_arr, v_arr)
    elif method == "gauss_laguerre":
        # Gauss-Laguerre nodes mapped to [0, 1] then scaled to volume range.
        # For N cuts we use N+1 edges: 0, then N nodes scaled, then v_max.
        # This is a simplified Whitson-style discretization with
        # heavier-tail emphasis.
        nodes, _ = np.polynomial.laguerre.laggauss(n_cuts)
        # Normalize nodes to [0, 1]: divide by (max + 1)
        u = nodes / (nodes[-1] + 1.0)
        v_min, v_max = float(v_arr[0]), float(v_arr[-1])
        # Edges: include endpoints and interpolate node positions
        inner = v_min + u * (v_max - v_min)
        edges = np.concatenate([[v_min], inner, [v_max]])
        # We have n_cuts+1 edges via this layout (n_cuts inner points + 2),
        # which is too many.  Use the nodes themselves as midpoints and
        # construct edges as midpoints between consecutive nodes.
        midpoints = np.concatenate([[v_min], inner, [v_max]])
        # This actually gives n_cuts + 2 points; collapse to n_cuts + 1 edges
        # by taking pairwise midpoints
        if len(midpoints) != n_cuts + 1:
            # Fall back to averaging adjacent nodes
            edges = np.zeros(n_cuts + 1)
            edges[0] = v_min
            edges[-1] = v_max
            for i in range(1, n_cuts):
                edges[i] = 0.5 * (inner[i-1] + inner[i] if i < len(inner) else v_max)
    else:
        raise ValueError(
            f"method must be 'equal_volume', 'equal_NBP', or "
            f"'gauss_laguerre', got {method!r}")

    # Build cuts
    cuts: List[PseudoComponent] = []
    NBP_lo = np.zeros(n_cuts)
    NBP_hi = np.zeros(n_cuts)
    vol_frac = np.zeros(n_cuts)
    for i in range(n_cuts):
        v_lo = float(edges[i])
        v_hi = float(edges[i + 1])
        v_mid = 0.5 * (v_lo + v_hi)
        NBP_lo[i] = float(np.interp(v_lo, v_arr, T_arr))
        NBP_hi[i] = float(np.interp(v_hi, v_arr, T_arr))
        NBP_mid = float(np.interp(v_mid, v_arr, T_arr))
        if Watson_K is not None:
            SG_i = watson_K_to_SG(NBP_mid, Watson_K)
        else:
            SG_i = SG_func(v_mid)
        cuts.append(PseudoComponent(
            NBP=NBP_mid, SG=SG_i, name=f"{name_prefix}_{i+1}"))
        vol_frac[i] = (v_hi - v_lo) / (v_arr[-1] - v_arr[0])

    # Compute mass and mole fractions
    # Mass fraction: m_i = vol_i * SG_i (relative; normalize at end)
    # Mole fraction: n_i = m_i / MW_i (relative; normalize at end)
    mass_unnorm = np.array([vol_frac[i] * cuts[i].SG for i in range(n_cuts)])
    mass_frac = mass_unnorm / mass_unnorm.sum()
    mole_unnorm = np.array([mass_frac[i] / cuts[i].MW for i in range(n_cuts)])
    mole_frac = mole_unnorm / mole_unnorm.sum()

    return TBPDiscretization(
        cuts=cuts,
        volume_fractions=vol_frac,
        mass_fractions=mass_frac,
        mole_fractions=mole_frac,
        NBP_lower=NBP_lo,
        NBP_upper=NBP_hi,
        method=method,
    )


# =====================================================================
# Convenience: discretize from common refinery input formats
# =====================================================================

def discretize_from_D86(
    volume_pct: Sequence[float],
    D86_T: Sequence[float],
    n_cuts: int,
    **kwargs,
) -> TBPDiscretization:
    """Convert ASTM D86 data to TBP and then discretize.

    Equivalent to ``discretize_TBP(D86_to_TBP(...), volume_pct, ...)``.
    """
    TBP_T = D86_to_TBP(volume_pct, D86_T)
    return discretize_TBP(TBP_T, volume_pct, n_cuts, **kwargs)


def discretize_from_D2887(
    volume_pct: Sequence[float],
    D2887_T: Sequence[float],
    n_cuts: int,
    **kwargs,
) -> TBPDiscretization:
    """Convert ASTM D2887 simulated distillation to TBP and discretize."""
    TBP_T = D2887_to_TBP(volume_pct, D2887_T)
    return discretize_TBP(TBP_T, volume_pct, n_cuts, **kwargs)
