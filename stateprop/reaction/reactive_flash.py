"""Reactive flash: VLE coupled with liquid-phase chemical equilibrium.

For an isothermal-isobaric reactive flash with feed composition z at
(T, p), three sets of equations must be satisfied simultaneously:

  1. Material balances per species:
       n_i^V + n_i^L = n_i^F + Sum_r nu[i,r] * xi_r

  2. Vapor-liquid equilibrium (modified Raoult, ideal vapor):
       y_i = gamma_i(T, x) * p_i^sat(T) * x_i / p
       i.e., K_i = gamma_i p_i^sat / p  and  y_i = K_i x_i

  3. Chemical equilibrium in the liquid phase (R reactions):
       K_eq,r(T) = Prod_i (gamma_i * x_i)^nu[i,r]

The combined system is solved by alternating:
  - Inner: gamma-phi flash for given post-reaction composition (xi held
    fixed). Produces x, y, beta = V/F_eff via successive substitution
    on K-values and Rachford-Rice.
  - Outer: Newton step on xi to satisfy liquid-phase chemical
    equilibrium.

This formulation handles the canonical reactive distillation cases
(esterification with simultaneous water removal, MTBE synthesis,
acetalization, etc.) at moderate pressures where modified Raoult is
a reasonable VLE model. For higher pressures, swap the inner flash
for `GammaPhiEOSFlash.isothermal()` which couples a vapor EOS.

References
----------
Taylor, R. and Krishna, R. (2000), "Modelling reactive distillation",
Chemical Engineering Science, 55(22): 5183-5229.

Doherty, M. F. and Malone, M. F. (2001), Conceptual Design of Distillation
Systems, McGraw-Hill, Chapters 9-10.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Callable, List
import math
import numpy as np

from .liquid_phase import LiquidPhaseReaction


@dataclass
class ReactiveFlashResult:
    """Outcome of a reactive flash calculation."""
    converged: bool
    T: float                    # [K]
    p: float                    # [Pa]
    F: float                    # initial feed moles
    z: np.ndarray               # initial feed mole fractions
    F_eff: float                # post-reaction total moles
    V: float                    # vapor moles
    L: float                    # liquid moles
    y: np.ndarray               # vapor mole fractions
    x: np.ndarray               # liquid mole fractions
    xi: np.ndarray              # reaction extents
    gamma: np.ndarray           # activity coefficients in liquid
    K: np.ndarray               # K_i = y_i / x_i at solution
    K_a: np.ndarray             # K_a per reaction = Prod (gamma_i x_i)^nu[r,i]
    K_eq: np.ndarray            # K_eq(T) per reaction (target)
    iterations: int = 0
    message: str = ""

    @property
    def vapor_fraction(self) -> float:
        """V / (V + L) in mole fractions of post-reaction system."""
        return self.V / max(self.V + self.L, 1e-30)


# =========================================================================
# Inner: modified-Raoult isothermal flash
# =========================================================================

def _solve_rachford_rice(z: np.ndarray, K: np.ndarray,
                           tol: float = 1e-12, maxiter: int = 100) -> float:
    """Solve Rachford-Rice for vapor fraction beta given feed z and K-values.

    Sum_i z_i (K_i - 1) / (1 + beta * (K_i - 1)) = 0

    Bracketed Newton with bisection fallback. Returns beta clipped to [0, 1].
    """
    K = np.asarray(K, dtype=float)
    z = np.asarray(z, dtype=float)
    # Existence checks
    if np.all(K >= 1.0):
        # No condensation possible: pure vapor (or supersaturated)
        rr_at_one = float((z * (K - 1.0)).sum())
        if rr_at_one >= 0:
            return 1.0
    if np.all(K <= 1.0):
        rr_at_zero = float((z * (K - 1.0)).sum())
        if rr_at_zero <= 0:
            return 0.0
    # Bracket: [eps, 1-eps]; Newton on beta
    beta_lo = 1.0 / (1.0 - K.max()) + 1e-10 if K.max() > 1 else 0.0
    beta_hi = 1.0 / (1.0 - K.min()) - 1e-10 if K.min() < 1 else 1.0
    beta_lo = max(beta_lo, 1e-10)
    beta_hi = min(beta_hi, 1.0 - 1e-10)
    if beta_lo >= beta_hi:
        return 0.5
    beta = 0.5 * (beta_lo + beta_hi)

    def f(b):
        return float((z * (K - 1.0) / (1.0 + b * (K - 1.0))).sum())
    def df(b):
        return float((-z * (K - 1.0) ** 2 / (1.0 + b * (K - 1.0)) ** 2).sum())

    for _ in range(maxiter):
        fb = f(beta)
        if abs(fb) < tol:
            return float(np.clip(beta, 0.0, 1.0))
        dfb = df(beta)
        if dfb >= 0 or not np.isfinite(dfb):
            beta = 0.5 * (beta_lo + beta_hi)
        else:
            step = -fb / dfb
            beta_new = beta + step
            if beta_new <= beta_lo or beta_new >= beta_hi:
                # Bisect
                if fb > 0:
                    beta_lo = beta
                else:
                    beta_hi = beta
                beta = 0.5 * (beta_lo + beta_hi)
            else:
                # Update bracket
                if fb > 0:
                    beta_lo = beta
                else:
                    beta_hi = beta
                beta = beta_new
    return float(np.clip(beta, 0.0, 1.0))


def _gamma_phi_eos_inner_flash(
    T: float, p: float, z: np.ndarray,
    activity_model, psat_funcs,
    vapor_eos,
    pure_liquid_volumes=None,
    phi_sat_funcs=None,
    K_init: Optional[np.ndarray] = None,
    tol: float = 1e-7, maxiter: int = 200,
):
    """γ-φ-EOS inner isothermal flash for use inside the reactive
    Newton loop.  Wraps ``GammaPhiEOSFlash.isothermal`` and adapts
    the return signature to match ``_modified_raoult_flash``:
    ``(beta, x, y, gamma, K, iterations)``.

    The activity-coefficient array ``gamma`` is recomputed at the
    converged liquid composition from ``activity_model``; the K-value
    array ``K`` already incorporates ``φ_i^V`` and (if given) the
    saturation-fugacity coefficient and Poynting correction.
    """
    # Lazy import to avoid a circular dependency at module load
    from ..activity.gamma_phi_eos import GammaPhiEOSFlash

    flash = GammaPhiEOSFlash(
        activity_model=activity_model,
        psat_funcs=psat_funcs,
        vapor_eos=vapor_eos,
        pure_liquid_volumes=pure_liquid_volumes,
        phi_sat_funcs=phi_sat_funcs,
    )
    res = flash.isothermal(T=T, p=p, z=z,
                            K_guess=(np.asarray(K_init, dtype=float)
                                     if K_init is not None else None),
                            tol=tol, maxiter=maxiter)
    gamma_at_x = np.asarray(activity_model.gammas(T, res.x))
    return float(res.V), np.asarray(res.x), np.asarray(res.y), \
           gamma_at_x, np.asarray(res.K), int(res.iterations)


def _modified_raoult_flash(
    T: float, p: float, z: np.ndarray,
    activity_model, psat_funcs,
    K_init: Optional[np.ndarray] = None,
    tol: float = 1e-7, maxiter: int = 200,
):
    """Modified-Raoult isothermal flash: gamma-only, ideal vapor.

    K_i = gamma_i(T, x) * p_i^sat(T) / p

    Returns: (beta, x, y, gamma, K, iterations).
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    N = z.size
    psat = np.array([f(T) for f in psat_funcs])

    if K_init is not None:
        K = np.asarray(K_init, dtype=float)
    else:
        # Raoult initial guess: gamma=1
        K = psat / p

    x = z.copy()
    y = z.copy()
    for it in range(maxiter):
        beta = _solve_rachford_rice(z, K)
        x_new = z / (1.0 + beta * (K - 1.0))
        s = x_new.sum()
        if s > 0:
            x_new = x_new / s
        y_new = K * x_new
        s = y_new.sum()
        if s > 0:
            y_new = y_new / s
        gammas = np.asarray(activity_model.gammas(T, x_new))
        K_new = gammas * psat / p
        err = float(np.max(np.abs(K_new - K) / np.maximum(K, 1e-12)))
        x, y, K = x_new, y_new, K_new
        if err < tol:
            return beta, x, y, gammas, K, it + 1
    raise RuntimeError(
        f"Modified-Raoult flash did not converge in {maxiter} iter "
        f"(max K err {err:.2e})")


# =========================================================================
# Outer: reactive flash
# =========================================================================

def reactive_flash_TP(
    T: float,
    p: float,
    F: float,
    z: Sequence[float],
    activity_model,
    psat_funcs: Sequence[Callable[[float], float]],
    reactions: Sequence[LiquidPhaseReaction],
    species_names: Optional[Sequence[str]] = None,
    xi_init: Optional[Sequence[float]] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
    damping: float = 0.5,
    verbose: bool = False,
    # ---- γ-φ-EOS coupling for high-pressure flash (v0.9.77+) ----
    vapor_eos = None,
    pure_liquid_volumes: Optional[Sequence[float]] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
) -> ReactiveFlashResult:
    """Isothermal-isobaric reactive flash.

    Solves simultaneous vapor-liquid and chemical equilibrium at fixed
    (T, p).  By default the vapor phase is treated as an ideal gas
    (modified Raoult's law: ``K_i = γ_i p_sat,i / p``); when
    ``vapor_eos`` is given, the full **γ-φ-EOS** formulation is used:

        K_i = γ_i p_sat,i Φ_sat,i exp[V_L,i (p - p_sat,i)/RT] / (p φ_V,i)

    valid up through ~100 bar for typical mixtures.  Reactions occur
    in the liquid phase only and reference liquid-phase activities
    a_i = γ_i x_i, so the chemistry residual is unchanged between the
    two formulations — only the partition between phases differs.

    Parameters
    ----------
    T : float
        Flash temperature [K].
    p : float
        Flash pressure [Pa].
    F : float
        Total feed moles (any consistent unit).
    z : sequence of float, length N
        Feed composition (mole fractions, sums to 1).
    activity_model : object with .gammas(T, x) returning length-N array
        e.g. NRTL, UNIQUAC, UNIFAC. Species ordering must match
        `species_names` (and reactions).
    psat_funcs : sequence of N callables T -> p_sat(T) [Pa]
        Pure-component saturation pressures. Antoine equations, DIPPR
        correlations, or any T -> Pa callable.
    reactions : sequence of `LiquidPhaseReaction`
        Liquid-phase reactions; their `species_names` must be a subset
        of the canonical species ordering. Stoichiometry is mapped
        automatically.
    species_names : sequence of str, optional
        Canonical species ordering. If `activity_model` has a
        `species_names` attribute, that is used; otherwise required.
    xi_init : sequence of float, length R, optional
        Initial reaction extents. Default zeros.
    tol : float
        Convergence tolerance on the chemical-equilibrium residual.
    maxiter : int
        Maximum outer iterations.
    damping : float
        Newton step damping (0 < damping <= 1). Default 0.5.
    verbose : bool
        Print iteration progress.
    vapor_eos : EOS mixture, optional
        If given, switches the inner VLE solve from modified Raoult
        (ideal vapor, ``φ_V = 1``) to a γ-φ-EOS coupling that draws
        vapor fugacity coefficients from the EOS.  Must implement
        ``density_from_pressure(p, T, x, phase_hint='vapor') -> rho``
        and ``ln_phi(rho, T, x) -> ndarray``.  Both ``CubicMixture``
        and ``SAFTMixture`` from this package qualify.  Recommended
        for ``p > 5 bar`` and required for accuracy above ``~30 bar``.
    pure_liquid_volumes : sequence of float, optional
        ``V_L,i`` in m³/mol per component.  If supplied, the Poynting
        correction is included in K-values.  Default: omitted (a ~5%
        approximation at 100 bar).  Only used with ``vapor_eos``.
    phi_sat_funcs : sequence of callables, optional
        ``φ_sat,i(T) -> float`` saturation fugacity coefficients.  If
        omitted, ``φ_sat = 1`` (a low-pressure approximation, accurate
        below ~10 bar but a 5-10% error near 50 bar).  Only used with
        ``vapor_eos``.

    Returns
    -------
    ReactiveFlashResult

    Notes
    -----
    Algorithm:
      1. xi = xi_init (zeros by default).
      2. Compute post-reaction total moles per species:
         n_i = z_i * F + Sum_r nu[i,r] * xi_r
         Effective feed F_eff = Sum n_i; z_eff = n / F_eff.
      3. Inner: modified-Raoult flash at (T, p, z_eff) -> beta, x, y, gamma.
      4. Compute chemical-equilibrium residual in liquid:
         f_r = Sum_i nu[i,r] (ln gamma_i + ln x_i) - ln K_eq,r(T)
      5. Newton step on xi using ideal-mixture Jacobian on liquid moles
         n_L_i = L * x_i, where L = (1 - beta) * F_eff:
         J[r,s] = Sum_i nu[i,r] nu[i,s] / n_L_i - dn_r dn_s / L
      6. Update xi = xi + damping * dxi (with feasibility constraint).
      7. Repeat until |f| < tol or maxiter exhausted.
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    N = z.size
    if len(psat_funcs) != N:
        raise ValueError(f"psat_funcs length {len(psat_funcs)} != z length {N}")

    # Inner-flash dispatch: γ-φ-EOS if vapor_eos given, else modified Raoult
    use_eos = vapor_eos is not None
    if use_eos and pure_liquid_volumes is not None and \
            len(pure_liquid_volumes) != N:
        raise ValueError(
            f"pure_liquid_volumes length {len(pure_liquid_volumes)} "
            f"!= N={N}")
    if use_eos and phi_sat_funcs is not None and \
            len(phi_sat_funcs) != N:
        raise ValueError(
            f"phi_sat_funcs length {len(phi_sat_funcs)} != N={N}")

    def _inner_flash(T_loc, p_loc, z_loc, K_init=None, tol_in=1e-9,
                     maxiter_in=300):
        if use_eos:
            return _gamma_phi_eos_inner_flash(
                T_loc, p_loc, z_loc, activity_model, psat_funcs,
                vapor_eos=vapor_eos,
                pure_liquid_volumes=pure_liquid_volumes,
                phi_sat_funcs=phi_sat_funcs,
                K_init=K_init, tol=tol_in, maxiter=maxiter_in)
        return _modified_raoult_flash(
            T_loc, p_loc, z_loc, activity_model, psat_funcs,
            K_init=K_init, tol=tol_in, maxiter=maxiter_in)

    # Resolve species order
    if species_names is None:
        if hasattr(activity_model, 'species_names'):
            species_names = list(activity_model.species_names)
        else:
            raise ValueError(
                "species_names must be provided (or activity_model must "
                "have a .species_names attribute)")
    species_names = list(species_names)
    if len(species_names) != N:
        raise ValueError(f"species_names length {len(species_names)} != N={N}")

    # Build R x N stoichiometry matrix from reactions
    reactions = list(reactions)
    R = len(reactions)
    nu = np.zeros((R, N))
    for r, rxn in enumerate(reactions):
        for sp_name, nu_local in zip(rxn.species_names, rxn.nu):
            if sp_name not in species_names:
                raise KeyError(f"Reaction {r} species '{sp_name}' not in "
                               f"flash species list {species_names}")
            j = species_names.index(sp_name)
            nu[r, j] = nu_local
    dn = nu.sum(axis=1)   # net mole change per reaction

    # Feasibility check on stoichiometry rank
    if R > 0:
        rk = np.linalg.matrix_rank(nu)
        if rk < R:
            raise ValueError(f"Stoichiometry rank {rk} < R={R}; "
                             "reactions are linearly dependent")

    # Initial xi: warm-start from pure-liquid equilibrium (no VLE)
    # to avoid the Newton singularity that occurs when product species
    # have zero initial moles (1/n_L diverges in the Jacobian).
    if xi_init is None:
        if R == 0:
            xi = np.zeros(0)
        else:
            from .liquid_phase import (LiquidPhaseReaction as _LPR,
                                          MultiLiquidPhaseReaction as _MLR)
            try:
                # Build the liquid-only system in the canonical species ordering
                if R == 1:
                    rxn = reactions[0]
                    # Map full feed to reaction species ordering
                    n_init_rxn = []
                    for sp in rxn.species_names:
                        idx = species_names.index(sp)
                        n_init_rxn.append(z[idx] * F)
                    r_l = rxn.equilibrium_extent(
                        T=T, n_initial=n_init_rxn,
                        activity_model=activity_model
                            if all(s in species_names for s in
                                   getattr(activity_model, 'species_names', species_names))
                            else activity_model,
                        tol=1e-7, maxiter=200)
                    xi = np.array([r_l.xi if r_l.converged else 0.0])
                else:
                    sys = _MLR(reactions)
                    n_init_dict = {sp: z[species_names.index(sp)] * F
                                    for sp in sys.species_names}
                    r_l = sys.equilibrium(
                        T=T, n_initial=n_init_dict,
                        activity_model=activity_model,
                        tol=1e-7, maxiter=200, damping=0.7)
                    if r_l.converged:
                        xi = r_l.xi.copy()
                    else:
                        xi = np.zeros(R)
            except Exception:
                # Fall back: small non-zero xi to escape the n_product=0 singularity
                xi = np.full(R, 0.01 * F)
    else:
        xi = np.asarray(xi_init, dtype=float)
        if xi.size != R:
            raise ValueError(f"xi_init length {xi.size} != R={R}")

    K_init = None  # warm-start for inner flash from prev iteration
    last_err = math.inf

    def _evaluate_at_xi(xi_arr):
        """Evaluate the reactive-flash residual at a candidate xi.

        Returns (n_total, F_eff, beta, x, y, gammas, K, f, K_a, K_eq_T)
        or raises if infeasible.
        """
        n_total_loc = z * F + nu.T @ xi_arr
        if (n_total_loc < 0).any():
            raise ValueError("infeasible xi (n_i < 0)")
        F_eff_loc = float(n_total_loc.sum())
        if F_eff_loc <= 0:
            raise ValueError("F_eff <= 0")
        z_eff_loc = n_total_loc / F_eff_loc
        beta_loc, x_loc, y_loc, gammas_loc, K_loc, _ = \
            _inner_flash(T, p, z_eff_loc,
                         K_init=K_init, tol_in=1e-9, maxiter_in=300)
        ln_K_eq_loc = np.array([rxn.ln_K_eq(T) for rxn in reactions])
        x_safe_loc = np.maximum(x_loc, 1e-30)
        gamma_safe = np.maximum(gammas_loc, 1e-30)
        ln_gx = np.log(gamma_safe) + np.log(x_safe_loc)
        f_loc = nu @ ln_gx - ln_K_eq_loc
        K_a_loc = np.array([np.prod((gammas_loc * x_safe_loc) ** nu[r])
                              for r in range(R)])
        K_eq_T_loc = np.exp(ln_K_eq_loc)
        return (n_total_loc, F_eff_loc, beta_loc, x_loc, y_loc,
                gammas_loc, K_loc, f_loc, K_a_loc, K_eq_T_loc)

    def _make_result(converged_, n_total_, F_eff_, beta_, x_, y_,
                       gammas_, K_, K_a_, K_eq_T_, iters_, msg_):
        V_ = beta_ * F_eff_
        L_ = (1.0 - beta_) * F_eff_
        return ReactiveFlashResult(
            converged=converged_, T=T, p=p, F=F, z=z,
            F_eff=F_eff_, V=V_, L=L_, y=y_, x=x_,
            xi=xi.copy(), gamma=gammas_, K=K_,
            K_a=K_a_, K_eq=K_eq_T_,
            iterations=iters_, message=msg_)

    # ---------------------------------------------------------------
    # Single reaction: bisection on extent (robust)
    # ---------------------------------------------------------------
    if R == 1:
        # Determine feasible range for xi
        xi_lo, xi_hi = -np.inf, np.inf
        for i in range(N):
            if nu[0, i] > 0:
                xi_lo = max(xi_lo, -z[i] * F / nu[0, i])
            elif nu[0, i] < 0:
                xi_hi = min(xi_hi, -z[i] * F / nu[0, i])
        eps_xi = 1e-9 * max(1.0, F)
        xi_lo = (xi_lo + eps_xi) if xi_lo > -np.inf else -1e6
        xi_hi = (xi_hi - eps_xi) if xi_hi <  np.inf else  1e6
        if xi_lo >= xi_hi:
            return _make_result(False, z*F, F, 0.0, np.full(N, np.nan),
                                  np.full(N, np.nan), np.full(N, np.nan),
                                  np.full(N, np.nan), np.full(R, np.nan),
                                  np.full(R, np.nan), 0,
                                  "Empty feasible xi range")

        # Evaluate at endpoints
        try:
            tup_lo = _evaluate_at_xi(np.array([xi_lo]))
            f_lo = float(tup_lo[7][0])
        except Exception:
            f_lo = math.nan
        try:
            tup_hi = _evaluate_at_xi(np.array([xi_hi]))
            f_hi = float(tup_hi[7][0])
        except Exception:
            f_hi = math.nan

        if math.isnan(f_lo) or math.isnan(f_hi) or f_lo * f_hi > 0:
            # No bracket; fall back to damped Newton from xi_init
            pass
        else:
            xi_a, xi_b = xi_lo, xi_hi
            f_a, f_b = f_lo, f_hi
            best = None
            for it in range(maxiter):
                xi_m = 0.5 * (xi_a + xi_b)
                xi = np.array([xi_m])
                try:
                    tup = _evaluate_at_xi(xi)
                except Exception:
                    xi_b = xi_m
                    continue
                (n_total, F_eff, beta, x, y, gammas, K, f, K_a, K_eq_T) = tup
                f_m = float(f[0])
                K_init = K.copy()
                # Relative convergence on K_a vs K_eq (more meaningful than ln-residual
                # for a reaction where K_eq spans many orders of magnitude)
                rel_err = abs(K_a[0] - K_eq_T[0]) / max(K_eq_T[0], 1e-30)
                if verbose:
                    print(f"  bisect it {it}: xi={xi_m:.5f}, f={f_m:.3e}, "
                          f"K_a/K_eq={K_a[0]/K_eq_T[0]:.6f}, beta={beta:.4f}")
                best = (n_total, F_eff, beta, x, y, gammas, K, K_a, K_eq_T)
                if abs(f_m) < tol or rel_err < tol or (xi_b - xi_a) < tol * max(1.0, abs(xi_b)):
                    return _make_result(True, *best, it + 1,
                                          f"bisection converged in {it+1} iters "
                                          f"(rel_err={rel_err:.2e})")
                if f_a * f_m < 0:
                    xi_b, f_b = xi_m, f_m
                else:
                    xi_a, f_a = xi_m, f_m
            # Did not hit tol in maxiter; return current best
            if best is not None:
                return _make_result(False, *best, maxiter,
                                      f"bisection: {maxiter} iters, "
                                      f"residual {abs(f_m):.2e}")

    # ---------------------------------------------------------------
    # Multi-reaction (or fallback): damped Newton
    # ---------------------------------------------------------------

    for outer_it in range(maxiter):
        # 1. Post-reaction species moles and effective feed
        n_total = z * F + nu.T @ xi    # length N
        if (n_total < 0).any():
            # Infeasible -- back off xi
            xi = 0.5 * xi
            continue
        F_eff = float(n_total.sum())
        if F_eff <= 0:
            return ReactiveFlashResult(
                converged=False, T=T, p=p, F=F, z=z, F_eff=0.0,
                V=0.0, L=0.0, y=np.full(N, np.nan), x=np.full(N, np.nan),
                xi=xi.copy(), gamma=np.full(N, np.nan),
                K=np.full(N, np.nan),
                K_a=np.full(R, np.nan), K_eq=np.full(R, np.nan),
                iterations=outer_it,
                message=f"F_eff <= 0 at outer iteration {outer_it}")
        z_eff = n_total / F_eff

        # 2. Inner flash
        try:
            beta, x, y, gammas, K, n_inner = _inner_flash(
                T, p, z_eff,
                K_init=K_init, tol_in=1e-9, maxiter_in=300)
            K_init = K.copy()
        except RuntimeError as e:
            return ReactiveFlashResult(
                converged=False, T=T, p=p, F=F, z=z, F_eff=F_eff,
                V=0.0, L=0.0, y=np.full(N, np.nan), x=np.full(N, np.nan),
                xi=xi.copy(), gamma=np.full(N, np.nan),
                K=np.full(N, np.nan),
                K_a=np.full(R, np.nan), K_eq=np.full(R, np.nan),
                iterations=outer_it,
                message=f"Inner flash failed at outer iter {outer_it}: {e}")

        V = beta * F_eff
        L = (1.0 - beta) * F_eff

        # 3. Chemical equilibrium residual (in liquid phase)
        if R == 0:
            # Pure VLE flash, no reactions
            return ReactiveFlashResult(
                converged=True, T=T, p=p, F=F, z=z, F_eff=F_eff,
                V=V, L=L, y=y, x=x, xi=xi.copy(), gamma=gammas, K=K,
                K_a=np.zeros(0), K_eq=np.zeros(0),
                iterations=outer_it + 1,
                message="No reactions; pure VLE flash converged")

        ln_K_eq = np.array([rxn.ln_K_eq(T) for rxn in reactions])
        # Cap x at minimum to avoid log(0) for trace species
        x_safe = np.maximum(x, 1e-30)
        gamma_safe = np.maximum(gammas, 1e-30)
        ln_gx = np.log(gamma_safe) + np.log(x_safe)
        f = nu @ ln_gx - ln_K_eq
        err = float(np.max(np.abs(f)))

        if verbose:
            print(f"  outer it {outer_it}: ||f||={err:.3e}, beta={beta:.4f}, "
                  f"xi={xi}")

        if err < tol:
            K_a = np.array([np.prod((gammas * x) ** nu[r])
                              for r in range(R)])
            K_eq_T = np.exp(ln_K_eq)
            return ReactiveFlashResult(
                converged=True, T=T, p=p, F=F, z=z, F_eff=F_eff,
                V=V, L=L, y=y, x=x, xi=xi.copy(), gamma=gammas, K=K,
                K_a=K_a, K_eq=K_eq_T,
                iterations=outer_it + 1,
                message=f"converged in {outer_it + 1} outer iters "
                        f"(||f||={err:.2e})")

        # 4. Newton step on xi using LIQUID-mole Jacobian
        # n_L_i = L * x_i; dn_r,liq is just the net stoichiometry per reaction
        n_L = L * x_safe   # liquid moles per species
        if L < 1e-12:
            # All-vapor case, no liquid for reactions to occur in
            # In this case xi must be zero (no reaction); break.
            return ReactiveFlashResult(
                converged=False, T=T, p=p, F=F, z=z, F_eff=F_eff,
                V=V, L=L, y=y, x=x, xi=xi.copy(), gamma=gammas, K=K,
                K_a=np.zeros(R), K_eq=np.exp(ln_K_eq),
                iterations=outer_it + 1,
                message="No liquid phase; reactions cannot occur. "
                        "Check feed composition and (T, p).")

        inv_n = 1.0 / n_L
        J = np.einsum('ri,si,i->rs', nu, nu, inv_n)
        J -= np.outer(dn, dn) / max(L, 1e-30)

        try:
            dxi = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dxi, *_ = np.linalg.lstsq(J, -f, rcond=None)

        # Damped step with feasibility limiter
        dxi = damping * dxi
        # Don't let n_total go negative
        d_n = nu.T @ dxi
        alpha = 1.0
        for i in range(N):
            if d_n[i] < 0 and n_total[i] > 1e-30:
                a_lim = -0.5 * n_total[i] / d_n[i]
                if a_lim < alpha:
                    alpha = max(a_lim, 1e-6)
        xi = xi + alpha * dxi
        last_err = err

    # Did not converge
    n_total = z * F + nu.T @ xi
    F_eff = float(n_total.sum()) if (n_total > 0).all() else F
    z_eff = n_total / F_eff if F_eff > 0 else z
    try:
        beta, x, y, gammas, K, _ = _inner_flash(
            T, p, z_eff,
            K_init=K_init, tol_in=1e-7, maxiter_in=200)
        V = beta * F_eff
        L = (1.0 - beta) * F_eff
        ln_K_eq = np.array([rxn.ln_K_eq(T) for rxn in reactions])
        K_a = np.array([np.prod((gammas * np.maximum(x, 1e-30)) ** nu[r])
                          for r in range(R)])
        K_eq_T = np.exp(ln_K_eq)
    except Exception:
        x = np.full(N, np.nan); y = np.full(N, np.nan)
        gammas = np.full(N, np.nan); K = np.full(N, np.nan)
        V = L = 0.0
        K_a = np.full(R, np.nan); K_eq_T = np.full(R, np.nan)
    return ReactiveFlashResult(
        converged=False, T=T, p=p, F=F, z=z, F_eff=F_eff,
        V=V, L=L, y=y, x=x, xi=xi.copy(), gamma=gammas, K=K,
        K_a=K_a, K_eq=K_eq_T,
        iterations=maxiter,
        message=f"did not converge in {maxiter} outer iters "
                f"(||f||={last_err:.2e})")
