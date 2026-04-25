"""Parameter regression tools for activity coefficient models.

Fit binary interaction parameters of NRTL, UNIQUAC, or any activity
model to user-supplied VLE or LLE experimental data via nonlinear
least squares. Enables quantitative VLLE flash with custom-fitted
parameters where the bundled UNIFAC predictions are insufficient.

Two main entry points:

- `regress_lle(model_factory, tie_lines, x0, ...)` -- fit to LLE
  tie-line data: triples (T, x1_exp, x2_exp).

- `regress_vle(model_factory, vle_points, x0, ...)` -- fit to VLE
  data: tuples (T, p, x_exp, y_exp).

Both use `scipy.optimize.least_squares` (Levenberg-Marquardt or
Trust-Region-Reflective). The user supplies a `model_factory` that
maps a parameter vector to a configured activity-model instance.
This decouples the regression from any particular model
parameterization -- you choose what to fit (b matrix only,
b + alpha, full a/b/e/f matrices, etc.).

Example: fit NRTL b parameters to water-organic LLE data:

    from stateprop.activity import NRTL
    from stateprop.activity.regression import regress_lle

    def nrtl_factory(p):
        # p = [b_12, b_21], alpha fixed at 0.3
        alpha = np.array([[0, 0.3], [0.3, 0]])
        b = np.array([[0, p[0]], [p[1], 0]])
        return NRTL(alpha=alpha, b=b)

    tie_lines = [
        (298.15, [0.05, 0.95], [0.55, 0.45]),
        (313.15, [0.07, 0.93], [0.51, 0.49]),
        # ...
    ]
    fit = regress_lle(nrtl_factory, tie_lines, x0=[200.0, 800.0])
    print("Optimal b:", fit.x)
    print("Residual norm:", fit.cost)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

from .lle import LLEFlash, LLEResult


def regress_lle(model_factory: Callable[[np.ndarray], object],
                  tie_lines: Sequence[Tuple[float, Sequence[float], Sequence[float]]],
                  x0: Sequence[float],
                  bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                  weights: Optional[Sequence[float]] = None,
                  objective: str = 'activity',
                  penalty: float = 10.0,
                  flash_maxiter: int = 100,
                  verbose: int = 0,
                  **least_squares_kwargs):
    """Fit activity-model parameters to LLE tie-line data.

    Two objective functions are available:

    - **'activity'** (default, recommended): minimize the equal-activity
      residuals at the EXPERIMENTAL compositions. For each tie line
      (T, x1_exp, x2_exp) and each component i, the residual is
            r_i = ln(x1_i gamma_i(T, x1_exp)) - ln(x2_i gamma_i(T, x2_exp)).
      At LLE equilibrium r_i = 0, so minimizing the squared residuals
      drives the model toward equilibrium at the experimental
      compositions. This is **smooth** in the parameters (no flash is
      run during regression), so Levenberg-Marquardt converges reliably
      even from poor initial guesses.

    - **'flash'**: run an LLE flash with the candidate parameters and
      compare the predicted phase compositions to experimental.
      Residuals are (x1_pred - x1_exp, x2_pred - x2_exp). This more
      directly minimizes composition error, but is non-smooth (fails
      when the candidate parameters don't give LLE) and can stall on
      bad initial guesses. Use only after 'activity' has produced a
      reasonable starting point.

    Parameters
    ----------
    model_factory : callable(params) -> activity_model
    tie_lines : list of (T, x1_exp, x2_exp)
    x0 : sequence
        Initial parameter vector.
    bounds : (lower, upper), optional
        Box bounds. If supplied, uses trust-region reflective; else LM.
    weights : sequence of length len(tie_lines), optional
        Per-tie-line weights. Default 1.
    objective : {'activity', 'flash'}, default 'activity'
        Choice of residual function (see above).
    penalty : float
        Used only with objective='flash' to penalize flash failures.
    flash_maxiter : int
        Used only with objective='flash'.
    verbose : int
        scipy verbosity level.
    **least_squares_kwargs
        Forwarded to scipy.optimize.least_squares.

    Returns
    -------
    OptimizeResult from scipy.optimize.least_squares.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("scipy required for regression tools") from e

    if objective not in ('activity', 'flash'):
        raise ValueError(f"objective must be 'activity' or 'flash', got {objective!r}")

    tie_lines_arr = [(float(T),
                       np.asarray(x1, dtype=float),
                       np.asarray(x2, dtype=float))
                      for T, x1, x2 in tie_lines]
    if weights is None:
        weights = np.ones(len(tie_lines_arr))
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.size != len(tie_lines_arr):
            raise ValueError(
                f"weights length {weights.size} != "
                f"number of tie lines {len(tie_lines_arr)}"
            )

    N = tie_lines_arr[0][1].size

    if objective == 'activity':
        n_resid_per_tie = N
        # Floor used to safely take log when x or gamma underflow
        x_floor = 1e-15

        def residuals(params):
            try:
                model = model_factory(np.asarray(params, dtype=float))
            except Exception:
                return np.full(n_resid_per_tie * len(tie_lines_arr),
                               penalty)
            out = []
            for (T, x1_exp, x2_exp), w in zip(tie_lines_arr, weights):
                try:
                    g1 = np.asarray(model.gammas(T, x1_exp), dtype=float)
                    g2 = np.asarray(model.gammas(T, x2_exp), dtype=float)
                    # Equal-activity residual on log scale
                    r = (np.log(np.maximum(x1_exp, x_floor) * np.maximum(g1, x_floor))
                         - np.log(np.maximum(x2_exp, x_floor) * np.maximum(g2, x_floor)))
                    out.extend(np.sqrt(w) * r)
                except Exception:
                    out.extend([penalty] * n_resid_per_tie)
            return np.array(out)
    else:
        # flash-based residuals
        n_resid_per_tie = 2 * N

        def residuals(params):
            try:
                model = model_factory(np.asarray(params, dtype=float))
            except Exception:
                return np.full(n_resid_per_tie * len(tie_lines_arr),
                               penalty)
            flash = LLEFlash(model)
            out = []
            for (T, x1_exp, x2_exp), w in zip(tie_lines_arr, weights):
                z = 0.5 * (x1_exp + x2_exp)
                try:
                    r = flash.solve(T, z, x1_exp, x2_exp,
                                      maxiter=flash_maxiter)
                    out.extend(np.sqrt(w) * (r.x1 - x1_exp))
                    out.extend(np.sqrt(w) * (r.x2 - x2_exp))
                except Exception:
                    out.extend([penalty] * n_resid_per_tie)
            return np.array(out)

    if bounds is None:
        result = least_squares(residuals, x0, method='lm',
                                  verbose=verbose,
                                  **least_squares_kwargs)
    else:
        result = least_squares(residuals, x0, bounds=bounds,
                                  method='trf',
                                  verbose=verbose,
                                  **least_squares_kwargs)
    return result


def regress_vle(model_factory: Callable[[np.ndarray], object],
                  vle_points: Sequence,
                  x0: Sequence[float],
                  psat_funcs: Sequence,
                  mode: str = 'isobaric',
                  bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                  weights: Optional[Sequence[float]] = None,
                  penalty: float = 10.0,
                  verbose: int = 0,
                  **least_squares_kwargs):
    """Fit activity-model parameters to VLE data.

    Two modes:
    - 'isobaric': data is (T_exp, p, x_exp, y_exp). At each x_exp, p,
      compute predicted T_pred and y_pred via bubble-T flash; fit by
      minimizing (T_err, y_err).
    - 'isothermal': data is (T, p_exp, x_exp, y_exp). At each x_exp, T,
      compute predicted p_pred and y_pred; fit (p_err, y_err).

    Parameters
    ----------
    model_factory : callable(params) -> activity_model
    vle_points : list
        For 'isobaric': (T_exp, p, x_exp, y_exp) tuples
        For 'isothermal': (T, p_exp, x_exp, y_exp) tuples
    x0 : sequence
        Initial parameter vector.
    psat_funcs : sequence of callables
        Pure-component saturation pressure functions (Antoine or
        equivalent). Required for the gamma-phi flash.
    mode : 'isobaric' or 'isothermal'
    bounds, weights, penalty, verbose, **kwargs : as in regress_lle.

    Returns
    -------
    OptimizeResult from scipy.optimize.least_squares.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError("scipy required for regression tools") from e
    from .gamma_phi import GammaPhiFlash

    if mode not in ('isobaric', 'isothermal'):
        raise ValueError(f"mode must be 'isobaric' or 'isothermal', got {mode!r}")

    points = list(vle_points)
    if weights is None:
        weights = np.ones(len(points))
    else:
        weights = np.asarray(weights, dtype=float)

    N = np.asarray(points[0][2], dtype=float).size
    # Residual layout per point: T_err (or p_err) + y_err (length N)
    n_resid_per_point = 1 + N

    def residuals(params):
        try:
            model = model_factory(np.asarray(params, dtype=float))
        except Exception:
            return np.full(n_resid_per_point * len(points), penalty)
        flash = GammaPhiFlash(activity_model=model, psat_funcs=psat_funcs)
        out = []
        for pt, w in zip(points, weights):
            T_exp, p_exp_or_p, x_exp, y_exp = pt
            x_exp = np.asarray(x_exp, dtype=float)
            y_exp = np.asarray(y_exp, dtype=float)
            try:
                if mode == 'isobaric':
                    r = flash.bubble_t(p=p_exp_or_p, x=x_exp)
                    err_main = (r.T - T_exp) / max(abs(T_exp), 1.0)
                else:
                    r = flash.bubble_p(T=T_exp, x=x_exp)
                    err_main = (r.p - p_exp_or_p) / max(abs(p_exp_or_p), 1.0)
                y_err = r.y - y_exp
                out.append(np.sqrt(w) * err_main)
                out.extend(np.sqrt(w) * y_err)
            except Exception:
                out.extend([penalty] * n_resid_per_point)
        return np.array(out)

    if bounds is None:
        result = least_squares(residuals, x0, method='lm',
                                  verbose=verbose,
                                  **least_squares_kwargs)
    else:
        result = least_squares(residuals, x0, bounds=bounds,
                                  method='trf',
                                  verbose=verbose,
                                  **least_squares_kwargs)
    return result


# Convenience factories for common parameterizations
# ----------------------------------------------------


def make_nrtl_factory(N: int, alpha_value: float = 0.3,
                       fit_a: bool = False) -> Callable:
    """Build a model factory for NRTL with binary 'b' parameters
    (and optionally 'a' parameters).

    For N=2, the parameter vector is [b_12, b_21] (or [a_12, a_21,
    b_12, b_21] if fit_a=True). For N=3, it's [b_12, b_13, b_21,
    b_23, b_31, b_32] (etc.).

    Parameters
    ----------
    N : int
        Number of components.
    alpha_value : float
        Fixed non-randomness parameter (typically 0.2-0.47).
    fit_a : bool
        If True, fit 'a' (T-independent term) AND 'b'. The parameter
        vector is then [a_offdiag..., b_offdiag...] in the same order.

    Returns
    -------
    callable(params) -> NRTL
    """
    from .nrtl import NRTL
    n_off = N * (N - 1)  # off-diagonal count
    expected_n = (2 * n_off) if fit_a else n_off
    alpha_mat = np.full((N, N), alpha_value)
    np.fill_diagonal(alpha_mat, 0.0)

    def factory(params):
        if len(params) != expected_n:
            raise ValueError(
                f"NRTL factory expected {expected_n} params, got {len(params)}"
            )
        a = np.zeros((N, N))
        b = np.zeros((N, N))
        if fit_a:
            a_vals = params[:n_off]
            b_vals = params[n_off:]
        else:
            b_vals = params
            a_vals = None
        # Fill off-diagonal in row-major order
        idx = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    if a_vals is not None:
                        a[i, j] = a_vals[idx]
                    b[i, j] = b_vals[idx]
                    idx += 1
        return NRTL(alpha=alpha_mat, a=a, b=b)
    return factory


def make_uniquac_factory(r: Sequence[float], q: Sequence[float],
                           fit_a: bool = False) -> Callable:
    """Build a model factory for UNIQUAC with binary 'b' parameters.

    r, q are fixed (from molecular structure). The parameter vector
    is [b_12, b_21, ...] (off-diagonal entries in row-major order).

    Parameters
    ----------
    r, q : sequences of length N
        UNIQUAC volume and surface area parameters.
    fit_a : bool
        If True, also fit 'a' (T-independent log-tau offset).

    Returns
    -------
    callable(params) -> UNIQUAC
    """
    from .uniquac import UNIQUAC
    N = len(r)
    n_off = N * (N - 1)
    expected_n = (2 * n_off) if fit_a else n_off

    def factory(params):
        if len(params) != expected_n:
            raise ValueError(
                f"UNIQUAC factory expected {expected_n} params, got {len(params)}"
            )
        a = np.zeros((N, N))
        b = np.zeros((N, N))
        if fit_a:
            a_vals = params[:n_off]
            b_vals = params[n_off:]
        else:
            b_vals = params
            a_vals = None
        idx = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    if a_vals is not None:
                        a[i, j] = a_vals[idx]
                    b[i, j] = b_vals[idx]
                    idx += 1
        return UNIQUAC(r=list(r), q=list(q), a=a, b=b)
    return factory
