"""Batch grid-generation utilities for VLE flashes.

Workflows like Pxy / Txy / ternary VLE-LLE diagrams, phase envelope
plotting, and parameter regression require running the same flash
calculation across hundreds-to-thousands of conditions. This module
provides batch helpers that:

1. Iterate over conditions and collect results into arrays.
2. Use **warm starts**: each call seeds the next call's iteration with
   the previous result's T (for bubble_t/dew_t) or K-values (for
   isothermal). For smooth grids, this typically halves the SS
   iteration count.
3. Optionally run in parallel via `multiprocessing.Pool`.
4. Handle individual failures gracefully (return None for that point
   rather than aborting the whole grid).

All functions are thin wrappers around the single-call flash methods
on `GammaPhiFlash` and `GammaPhiEOSFlash`. They return either lists
of result objects (sequential) or numpy arrays of the relevant fields
(vectorized accessors).

Example: generate a Pxy diagram at fixed T:

    from stateprop.activity import UNIFAC, GammaPhiFlash, AntoinePsat
    from stateprop.activity.batch import batch_bubble_p

    flash = GammaPhiFlash(...)
    x_grid = np.linspace(0.001, 0.999, 200)
    x_array = np.column_stack([x_grid, 1 - x_grid])  # binary
    results = batch_bubble_p(flash, T=350.0, x_list=x_array)
    p_array = np.array([r.p if r else np.nan for r in results])
    y_array = np.array([r.y[0] if r else np.nan for r in results])
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np


def batch_bubble_t(flash, p: float, x_list,
                   T_guess: Optional[float] = None,
                   warm_start: bool = True,
                   n_workers: int = 1) -> List[Any]:
    """Compute bubble T for many liquid compositions at fixed p.

    Parameters
    ----------
    flash : GammaPhiFlash or GammaPhiEOSFlash
        Flash object with `bubble_t(p, x, T_guess=...)` method.
    p : float
        System pressure [Pa].
    x_list : (M, N) array or sequence of length-N arrays
        M liquid compositions to flash.
    T_guess : float, optional
        Initial guess for the first point's bubble T. Subsequent
        points use the previous result's T.
    warm_start : bool, default True
        If True, use previous result's T as next call's guess.
    n_workers : int, default 1
        If > 1, use multiprocessing.Pool with this many workers.
        Disables warm_start (each worker starts cold).

    Returns
    -------
    list of BubbleResult or None
        One entry per composition; None if that flash failed.
    """
    x_array = np.asarray(x_list, dtype=float)
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)

    if n_workers > 1:
        return _parallel_bubble_t(flash, p, x_array, n_workers)

    results: List[Any] = []
    T_prev = T_guess
    for i in range(x_array.shape[0]):
        x = x_array[i]
        try:
            r = flash.bubble_t(p=p, x=x, T_guess=T_prev)
            results.append(r)
            if warm_start:
                T_prev = r.T
        except Exception:
            results.append(None)
            # Don't propagate failed warm-start to next point
    return results


def batch_bubble_p(flash, T: float, x_list,
                   warm_start: bool = True,
                   n_workers: int = 1) -> List[Any]:
    """Compute bubble pressure for many liquid compositions at fixed T.

    Note: for `GammaPhiFlash` (ideal-gas vapor), bubble_p is direct
    evaluation with no iteration -- warm_start has no effect there.
    For `GammaPhiEOSFlash`, the SS iteration on K via `_K_values`
    benefits from warm-starting via `p_guess`.
    """
    x_array = np.asarray(x_list, dtype=float)
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)

    if n_workers > 1:
        return _parallel_bubble_p(flash, T, x_array, n_workers)

    results: List[Any] = []
    p_prev: Optional[float] = None
    for i in range(x_array.shape[0]):
        x = x_array[i]
        try:
            # GammaPhiEOSFlash.bubble_p accepts p_guess; GammaPhiFlash ignores it
            kwargs = {}
            if warm_start and p_prev is not None and hasattr(flash, '_K_values'):
                kwargs['p_guess'] = p_prev
            r = flash.bubble_p(T=T, x=x, **kwargs)
            results.append(r)
            if warm_start:
                p_prev = r.p
        except Exception:
            results.append(None)
    return results


def batch_isothermal(flash, conditions: Sequence[Tuple[float, float, Sequence[float]]],
                     warm_start: bool = True,
                     n_workers: int = 1) -> List[Any]:
    """Run isothermal PT flash for many (T, p, z) conditions.

    Parameters
    ----------
    flash : GammaPhiFlash or GammaPhiEOSFlash
        Flash object with `isothermal(T, p, z, K_guess=...)` method.
    conditions : list of (T, p, z) tuples
        Points to flash.
    warm_start : bool, default True
        Pass previous result's K to next call. Reliable for smooth
        grids; can hurt convergence if conditions jump across phase
        boundaries (warm K may be ill-conditioned).
    n_workers : int, default 1
        If > 1, use multiprocessing.

    Returns
    -------
    list of FlashResult or None
    """
    conditions = list(conditions)

    if n_workers > 1:
        return _parallel_isothermal(flash, conditions, n_workers)

    results: List[Any] = []
    K_prev: Optional[np.ndarray] = None
    for T, p, z in conditions:
        try:
            r = flash.isothermal(T=T, p=p, z=z, K_guess=K_prev)
            results.append(r)
            if warm_start:
                K_prev = r.K
        except Exception:
            results.append(None)
            # Reset warm-start on failure to avoid propagating bad K
            K_prev = None
    return results


# ---------------------------------------------------------------------------
# Vectorized accessors -- pull arrays out of a list of FlashResults
# ---------------------------------------------------------------------------


def stack_T(results) -> np.ndarray:
    """Extract T values; np.nan for failed flashes."""
    return np.array([r.T if r is not None else np.nan for r in results])


def stack_p(results) -> np.ndarray:
    return np.array([r.p if r is not None else np.nan for r in results])


def stack_V(results) -> np.ndarray:
    """Vapor fraction; nan for failed or non-flash results."""
    return np.array([getattr(r, 'V', np.nan) if r is not None else np.nan
                      for r in results])


def stack_x(results) -> np.ndarray:
    """(M, N) array of liquid compositions; nan rows for failures."""
    out = []
    for r in results:
        if r is None:
            out.append(None)
        else:
            out.append(r.x)
    # Determine N from first valid
    N = next((len(r.x) for r in results if r is not None), 0)
    arr = np.full((len(results), N), np.nan)
    for i, x in enumerate(out):
        if x is not None:
            arr[i, :] = x
    return arr


def stack_y(results) -> np.ndarray:
    out = []
    for r in results:
        out.append(None if r is None else r.y)
    N = next((len(r.y) for r in results if r is not None), 0)
    arr = np.full((len(results), N), np.nan)
    for i, y in enumerate(out):
        if y is not None:
            arr[i, :] = y
    return arr


def stack_K(results) -> np.ndarray:
    out = []
    for r in results:
        out.append(None if r is None else r.K)
    N = next((len(r.K) for r in results if r is not None), 0)
    arr = np.full((len(results), N), np.nan)
    for i, K in enumerate(out):
        if K is not None:
            arr[i, :] = K
    return arr


# ---------------------------------------------------------------------------
# Parallel implementations (multiprocessing)
# ---------------------------------------------------------------------------
# These are kept simple: each worker process gets the flash object via
# pickle, then runs the single-call method. No warm-start across workers
# (each chunk starts cold). Activity models built from numpy arrays
# pickle fine; cubic/SAFT EOS objects also pickle since they're plain
# Python classes with numpy/float members.


def _do_bubble_t(args):
    flash, p, x = args
    try:
        return flash.bubble_t(p=p, x=x)
    except Exception:
        return None


def _do_bubble_p(args):
    flash, T, x = args
    try:
        return flash.bubble_p(T=T, x=x)
    except Exception:
        return None


def _do_isothermal(args):
    flash, T, p, z = args
    try:
        return flash.isothermal(T=T, p=p, z=z)
    except Exception:
        return None


def _parallel_bubble_t(flash, p, x_array, n_workers):
    import multiprocessing as mp
    args = [(flash, p, x_array[i]) for i in range(x_array.shape[0])]
    with mp.Pool(n_workers) as pool:
        return pool.map(_do_bubble_t, args)


def _parallel_bubble_p(flash, T, x_array, n_workers):
    import multiprocessing as mp
    args = [(flash, T, x_array[i]) for i in range(x_array.shape[0])]
    with mp.Pool(n_workers) as pool:
        return pool.map(_do_bubble_p, args)


def _parallel_isothermal(flash, conditions, n_workers):
    import multiprocessing as mp
    args = [(flash, T, p, z) for (T, p, z) in conditions]
    with mp.Pool(n_workers) as pool:
        return pool.map(_do_isothermal, args)
