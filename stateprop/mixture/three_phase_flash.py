"""Three-phase (VLLE) flash for Helmholtz/GERG mixtures (v0.9.19).

Extends the two-phase PT flash to handle vapor-liquid-liquid equilibria,
which arise in systems like water + hydrocarbons, CO2 + heavy hydrocarbons,
and alcohols + hydrocarbons where two liquid phases coexist with a vapor.

Algorithm (Michelsen 1982):

1. Run standard 2-phase flash_pt at (p, T, z).
2. If 2-phase returned single phase: check if any trial split exists via
   TPD stability. Currently limited to pure VLE -> 1 phase cases; this
   implementation focuses on the 2-phase -> 3-phase case which is the
   practically common VLLE regime.
3. If 2-phase returned two phases (liquid + vapor): run Michelsen TPD
   stability on EACH of the two phases. If either phase is unstable,
   split it into two sub-phases and initialize a 3-phase flash.
4. 3-phase flash: iterate on two K-ratios (K^V/L1 = y/x1, K^L2/L1 = x2/x1)
   via successive substitution, with 2D Rachford-Rice at each K update
   to solve for phase fractions (beta_V, beta_L1, beta_L2) satisfying
   material balance.
5. Fugacity-equality convergence: at the solution,
      ln phi^V_i = ln phi^L1_i = ln phi^L2_i   for all i
   equivalently K^V/L1 = exp(ln phi^L1 - ln phi^V) and
                K^L2/L1 = exp(ln phi^L1 - ln phi^L2).

Returns `ThreePhaseFlashResult` with x1 (liquid-1 composition), x2
(liquid-2 composition), y (vapor), phase fractions, and caloric props.
Falls back to 2-phase MixtureFlashResult if the feed doesn't split
three-phase.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .properties import (
    ln_phi, density_from_pressure, enthalpy, entropy, _pure_caloric,
)
from .stability import stability_test_TPD, wilson_K
from .flash import flash_pt, MixtureFlashResult


@dataclass
class ThreePhaseFlashResult:
    """Outcome of a three-phase PT flash.

    Conventions: L1 is the denser liquid (arbitrary labeling otherwise),
    L2 is the lighter/second liquid, V is the vapor phase. When the
    problem is actually two-phase or single-phase, the 2-phase result
    is returned instead via the `two_phase_result` field.
    """
    phase: str                     # "VLLE" | "VLE" | "LLE" | <single-phase labels>
    T: float
    p: float
    beta_V: float                  # vapor fraction (0 for LLE)
    beta_L1: float                 # liquid-1 fraction
    beta_L2: float                 # liquid-2 fraction (0 for VLE)
    x1: np.ndarray                 # liquid-1 composition
    x2: np.ndarray                 # liquid-2 composition
    y: np.ndarray                  # vapor composition
    z: np.ndarray                  # feed composition
    rho_V: Optional[float]         # vapor density [mol/m^3]
    rho_L1: Optional[float]        # liquid-1 density
    rho_L2: Optional[float]        # liquid-2 density
    h: float                       # mixture-average molar enthalpy [J/mol]
    s: float                       # mixture-average molar entropy [J/(mol K)]
    iterations: int
    K_VL1: Optional[np.ndarray] = None   # K_V/L1 = y / x1
    K_L2L1: Optional[np.ndarray] = None  # K_L2/L1 = x2 / x1
    two_phase_result: Optional[MixtureFlashResult] = None   # set when the
                                   # problem is actually 2-phase or 1-phase


def _rachford_rice_3p(z, K_VL1, K_L2L1, tol=1e-10, maxiter=100):
    """Solve 3-phase Rachford-Rice for (beta_V, beta_L2) at fixed K's.

    v0.9.21 note: as of v0.9.21, the Helmholtz three-phase SS
    (`_three_phase_ss`) no longer uses this solver directly -- it has
    been replaced by a composition-based SS (ported from cubic v0.9.20)
    that solves for betas by projected linear least-squares on the
    material balance, which avoids the rank-1 Jacobian pathology when
    K_VL1 - 1 and K_L2L1 - 1 become proportional. This function remains
    as a reference utility for unit-testing the RR kernel directly with
    exact analytical inputs, and for external users who want to solve
    the RR problem in isolation.

    Material balance: z_i = beta_V y_i + beta_L1 x1_i + beta_L2 x2_i.
    Using y_i = K_VL1_i x1_i, x2_i = K_L2L1_i x1_i, beta_L1 = 1 - beta_V - beta_L2:

        x1_i = z_i / D_i,   D_i = 1 + beta_V (K_VL1_i - 1) + beta_L2 (K_L2L1_i - 1)

    Two residuals that must both be zero at the solution:

        R1(beta_V, beta_L2) = sum_i z_i (K_VL1_i - 1) / D_i    (= sum(y - x1))
        R2(beta_V, beta_L2) = sum_i z_i (K_L2L1_i - 1) / D_i   (= sum(x2 - x1))

    Solved via 2D Newton, starting from a small-interior guess and
    step-capping. The Jacobian is negative-definite, so R1 and R2 are
    monotonically decreasing in beta_V and beta_L2 respectively,
    guaranteeing a unique solution if one exists in the feasibility
    region.

    Feasibility: beta_V >= 0, beta_L2 >= 0, beta_V + beta_L2 <= 1, and
    D_i > 0 for all i. The solver returns (beta_V, beta_L1, beta_L2,
    x1, x2, y) even if convergence was partial; caller should verify.

    Raises RuntimeError if the 2D Newton stalls or leaves the feasible
    region permanently, which generally means the provided K-factors do
    not admit a valid 3-phase split at composition z.
    """
    z = np.asarray(z, dtype=np.float64)
    # Starting point: small interior -- 10% each for beta_V, beta_L2 to avoid clipping
    beta_V = 0.1
    beta_L2 = 0.1

    aV = K_VL1 - 1.0
    aL2 = K_L2L1 - 1.0

    last_res = float('inf')
    stall_count = 0
    for it in range(maxiter):
        D = 1.0 + beta_V * aV + beta_L2 * aL2
        if np.any(D <= 0):
            # Step into infeasible region; back off both fractions
            beta_V *= 0.5; beta_L2 *= 0.5
            continue
        R1 = float(np.sum(z * aV / D))
        R2 = float(np.sum(z * aL2 / D))
        res = max(abs(R1), abs(R2))
        if res < tol:
            break
        D2 = D * D
        J11 = -float(np.sum(z * aV * aV / D2))
        J12 = -float(np.sum(z * aV * aL2 / D2))
        J22 = -float(np.sum(z * aL2 * aL2 / D2))
        det = J11 * J22 - J12 * J12
        if abs(det) < 1e-18:
            raise RuntimeError("3-phase RR: Jacobian singular")
        dbV = -(J22 * R1 - J12 * R2) / det
        dbL2 = -(-J12 * R1 + J11 * R2) / det
        # Step-capping: no more than 25% of current step in each variable
        step = max(abs(dbV), abs(dbL2))
        if step > 0.25:
            dbV *= 0.25 / step
            dbL2 *= 0.25 / step
        beta_V_new = beta_V + dbV
        beta_L2_new = beta_L2 + dbL2
        # Stay strictly inside feasible region; if step pushes negative,
        # backtrack the step instead of clipping.
        if beta_V_new < 0 or beta_L2_new < 0 or beta_V_new + beta_L2_new > 1:
            # Line search: shrink step until feasible
            alpha = 1.0
            for _ in range(10):
                alpha *= 0.5
                bV_try = beta_V + alpha * dbV
                bL2_try = beta_L2 + alpha * dbL2
                if bV_try >= 0 and bL2_try >= 0 and bV_try + bL2_try <= 1.0:
                    beta_V_new = bV_try
                    beta_L2_new = bL2_try
                    break
            else:
                # Couldn't find feasible step; give up
                raise RuntimeError(
                    f"3-phase RR: no feasible step found (R=({R1:.2e}, {R2:.2e}))"
                )
        beta_V = beta_V_new
        beta_L2 = beta_L2_new
        # Stall detection: residual not decreasing meaningfully
        if abs(res - last_res) / max(res, 1e-30) < 1e-12:
            stall_count += 1
            if stall_count > 3:
                raise RuntimeError(
                    f"3-phase RR: stalled at R=({R1:.2e}, {R2:.2e})"
                )
        else:
            stall_count = 0
        last_res = res

    # Ensure convergence
    if res > 10 * tol:
        raise RuntimeError(
            f"3-phase RR: did not converge, final R=({R1:.2e}, {R2:.2e})"
        )

    D = 1.0 + beta_V * aV + beta_L2 * aL2
    x1 = z / D
    x1 = x1 / x1.sum()
    y = K_VL1 * x1; y = y / y.sum()
    x2 = K_L2L1 * x1; x2 = x2 / x2.sum()
    beta_L1 = 1.0 - beta_V - beta_L2
    return beta_V, beta_L1, beta_L2, x1, x2, y


def _three_phase_ss(z, T, p, mixture, K_VL1_init, K_L2L1_init,
                    tol=1e-8, maxiter=150):
    """Successive-substitution three-phase flash at (p, T, z).

    v0.9.21: ported from the cubic composition-based SS (v0.9.20). Iterates
    on (x1, x2, y) directly rather than on K-factors, avoiding the rank-1
    Rachford-Rice pathology that arises when one phase becomes near-pure
    (e.g., water in water-hydrocarbon LLE, where x1 -> [1.0, ~0] makes
    K_L2L1 = x2/x1 overflow and the 2D RR Jacobian rank-deficient).

    At each iteration:
      1. Compute ln phi of each phase, build current K_VL1, K_L2L1.
      2. Solve for (beta_V, beta_L1, beta_L2) by projected linear
         least-squares on material balance
             z = beta_V * y + beta_L1 * x1 + beta_L2 * x2
         with simplex constraint. No Rachford-Rice Jacobian inversion.
      3. Compute new compositions from x1 = z / D with
             D = 1 + beta_V (K_VL1 - 1) + beta_L2 (K_L2L1 - 1),
         then x2 = K_L2L1 x1, y = K_VL1 x1, all normalized.

    This approach is more robust than the K-based Rachford-Rice used in
    v0.9.19 -- same algorithm now in both cubic and Helmholtz paths.

    Raises RuntimeError on non-convergence or trivial collapse.
    Returns (beta_V, beta_L1, beta_L2, x1, x2, y, K_VL1, K_L2L1,
             rho_V, rho_L1, rho_L2, niter).
    """
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    K_VL1 = K_VL1_init.copy()
    K_L2L1 = K_L2L1_init.copy()

    # Seed (x1, x2, y) from the K's at a small-beta interior point
    aV = K_VL1 - 1.0; aL2 = K_L2L1 - 1.0
    beta_V_seed = min(0.1, 0.5 / max(float(np.max(np.abs(aV))), 1.0))
    beta_L2_seed = min(0.1, 0.5 / max(float(np.max(np.abs(aL2))), 1.0))
    D = 1.0 + beta_V_seed * aV + beta_L2_seed * aL2
    if np.any(D <= 0):
        raise RuntimeError("3-phase SS seed: D has non-positive entries")
    x1 = z / D; x1 = x1 / x1.sum()
    x2 = K_L2L1 * x1; x2 = x2 / x2.sum()
    y = K_VL1 * x1; y = y / y.sum()

    for it in range(maxiter):
        # Trivial collapse check
        if np.max(np.abs(x1 - x2)) < 1e-4 or np.max(np.abs(x1 - y)) < 1e-4:
            raise RuntimeError("Two phases collapsed to identical composition")

        # Compute ln phi for each phase
        try:
            rho_L1 = density_from_pressure(p, T, x1, mixture, phase_hint='liquid')
        except RuntimeError:
            rho_L1 = density_from_pressure(p, T, x1, mixture, phase_hint='vapor')
        try:
            rho_L2 = density_from_pressure(p, T, x2, mixture, phase_hint='liquid')
        except RuntimeError:
            rho_L2 = density_from_pressure(p, T, x2, mixture, phase_hint='vapor')
        try:
            rho_V = density_from_pressure(p, T, y, mixture, phase_hint='vapor')
        except RuntimeError:
            rho_V = density_from_pressure(p, T, y, mixture, phase_hint='liquid')

        lnphi_L1 = ln_phi(rho_L1, T, x1, mixture)
        lnphi_L2 = ln_phi(rho_L2, T, x2, mixture)
        lnphi_V = ln_phi(rho_V, T, y, mixture)
        K_VL1 = np.exp(lnphi_L1 - lnphi_V)
        K_L2L1 = np.exp(lnphi_L1 - lnphi_L2)

        # Solve betas by projected linear least-squares on
        #   z - x1 = beta_V (y - x1) + beta_L2 (x2 - x1)
        A = np.column_stack([y - x1, x2 - x1])
        b = z - x1
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        beta_V, beta_L2 = float(beta[0]), float(beta[1])
        # Simplex projection
        beta_V = max(1e-8, min(1.0 - 1e-8, beta_V))
        beta_L2 = max(1e-8, min(1.0 - 1e-8, beta_L2))
        if beta_V + beta_L2 > 1.0 - 1e-8:
            scale = (1.0 - 1e-8) / (beta_V + beta_L2)
            beta_V *= scale; beta_L2 *= scale
        beta_L1 = 1.0 - beta_V - beta_L2

        # Composition update from K's + betas
        aV = K_VL1 - 1.0; aL2 = K_L2L1 - 1.0
        D = 1.0 + beta_V * aV + beta_L2 * aL2
        if np.any(D <= 0):
            raise RuntimeError(f"3-phase SS iter {it}: D <= 0, betas infeasible")
        x1_new = z / D; x1_new = x1_new / x1_new.sum()
        x2_new = K_L2L1 * x1_new; x2_new = x2_new / x2_new.sum()
        y_new = K_VL1 * x1_new; y_new = y_new / y_new.sum()

        delta = max(float(np.max(np.abs(x1_new - x1))),
                    float(np.max(np.abs(x2_new - x2))),
                    float(np.max(np.abs(y_new - y))))
        x1, x2, y = x1_new, x2_new, y_new

        if delta < tol:
            return (beta_V, beta_L1, beta_L2, x1, x2, y, K_VL1, K_L2L1,
                    rho_V, rho_L1, rho_L2, it + 1)

    raise RuntimeError(
        f"Helmholtz 3-phase SS did not converge: delta={delta:.2e} after {maxiter} iters"
    )


def flash_pt_three_phase(p, T, z, mixture, tol=1e-8, maxiter=100,
                         check_single_phase=True):
    """PT flash allowing for up to three equilibrium phases.

    Wraps the standard 2-phase flash_pt. If the 2-phase flash returns a
    two-phase split, runs TPD stability on each resulting phase; if
    either is unstable, initializes a 3-phase SS iteration.

    Algorithm:
    1. Call flash_pt(p, T, z, mixture).
    2. If single-phase: return as-is (wrapped in two_phase_result).
    3. If two-phase (liquid + vapor at compositions x, y, fractions
       beta_L=1-beta, beta_V=beta): run TPD stability on x and y.
       a. If both stable: return two-phase result as-is.
       b. If one is unstable (typically the liquid), split it into
          x1, x2 using the unstable trial K-factor. Initialize
          K_VL1 = y/x1_init, K_L2L1 = x2_init/x1_init.
       c. Run _three_phase_ss to convergence.
       d. Post-check: if one phase fraction becomes negligible (<1e-5),
          collapse to 2-phase.

    Parameters
    ----------
    p, T : floats
    z : array (N,)          feed composition
    mixture : Mixture
    tol : float             convergence tolerance on ln K step
    maxiter : int           max 3-phase SS iterations
    check_single_phase : bool   whether to run stability on the feed

    Returns
    -------
    ThreePhaseFlashResult
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    N = len(z)

    # Step 1: run 2-phase flash
    result_2p = flash_pt(p, T, z, mixture, check_stability=check_single_phase,
                          tol=1e-9, maxiter=80)

    # Single-phase result: wrap and return
    if result_2p.beta is None:
        return ThreePhaseFlashResult(
            phase=result_2p.phase, T=T, p=p,
            beta_V=0.0 if 'liquid' in result_2p.phase else 1.0,
            beta_L1=1.0 if 'liquid' in result_2p.phase else 0.0,
            beta_L2=0.0,
            x1=z.copy(), x2=z.copy(), y=z.copy(), z=z,
            rho_V=None, rho_L1=None, rho_L2=None,
            h=result_2p.h, s=result_2p.s,
            iterations=result_2p.iterations,
            two_phase_result=result_2p,
        )

    # 2-phase: result_2p.x (liquid), result_2p.y (vapor), result_2p.beta (V fraction)
    x_2p = result_2p.x
    y_2p = result_2p.y
    beta_2p = result_2p.beta

    # Step 2: TPD stability on each phase
    # The liquid phase is most often the one that splits (LLE in hydrocarbons
    # + water, etc.). Check it first.
    try:
        L_stable, K_stab_L, S_L = stability_test_TPD(x_2p, T, p, mixture)
    except RuntimeError:
        L_stable = True; S_L = 0.0; K_stab_L = None
    try:
        V_stable, K_stab_V, S_V = stability_test_TPD(y_2p, T, p, mixture)
    except RuntimeError:
        V_stable = True; S_V = 0.0; K_stab_V = None

    # If both phases stable, 2-phase is the answer
    if L_stable and V_stable:
        return ThreePhaseFlashResult(
            phase="VLE", T=T, p=p,
            beta_V=beta_2p, beta_L1=1.0 - beta_2p, beta_L2=0.0,
            x1=x_2p.copy(), x2=x_2p.copy(), y=y_2p.copy(), z=z,
            rho_V=result_2p.rho_V, rho_L1=result_2p.rho_L, rho_L2=None,
            h=result_2p.h, s=result_2p.s,
            iterations=result_2p.iterations,
            K_VL1=result_2p.K, K_L2L1=None,
            two_phase_result=result_2p,
        )

    # Step 3: initialize 3-phase K's via multi-seed strategy (matches
    # cubic v0.9.20). When the 2-phase flash converges to a degenerate
    # split (e.g., x_2p[i] ~ 1e-12 for one component in water-hydrocarbon
    # LLE), building K_L2L1 = x2/x1 would overflow. We regularize x1
    # and use the stability-trial composition for x2.
    DEGEN_EPS = 1e-4

    def _build_seed(x_base, K_trial, other_phase):
        x1 = np.maximum(x_base, DEGEN_EPS)
        x1 = x1 / x1.sum()
        x2 = K_trial * x_base
        x2 = x2 / x2.sum()
        K_VL1 = other_phase / x1
        K_L2L1 = x2 / x1
        return x1, x2, K_VL1, K_L2L1

    seeds = []
    if not L_stable:
        seeds.append(_build_seed(x_2p, K_stab_L, y_2p))
    if not V_stable:
        _, x2_V, _, _ = _build_seed(y_2p, K_stab_V, x_2p)
        x1v = np.maximum(x_2p, DEGEN_EPS); x1v = x1v / x1v.sum()
        K_VL1_v = y_2p / x1v
        K_L2L1_v = x2_V / x1v
        seeds.append((x1v, x2_V, K_VL1_v, K_L2L1_v))

    # Fallback: near-uniform x2 direction off the L1-V tie-line
    x1_fb = np.maximum(x_2p, DEGEN_EPS); x1_fb = x1_fb / x1_fb.sum()
    x2_fb = np.ones(N) / N
    K_VL1_fb = y_2p / x1_fb
    K_L2L1_fb = x2_fb / x1_fb
    seeds.append((x1_fb, x2_fb, K_VL1_fb, K_L2L1_fb))

    # Step 4: run 3-phase SS with seed retry
    ss_result = None
    for seed in seeds:
        _, _, K_VL1_init, K_L2L1_init = seed
        try:
            ss_result = _three_phase_ss(
                z, T, p, mixture, K_VL1_init, K_L2L1_init,
                tol=tol, maxiter=maxiter,
            )
            break
        except RuntimeError:
            continue

    if ss_result is None:
        # All seeds failed -- fall back to 2-phase
        return ThreePhaseFlashResult(
            phase="VLE", T=T, p=p,
            beta_V=beta_2p, beta_L1=1.0 - beta_2p, beta_L2=0.0,
            x1=x_2p.copy(), x2=x_2p.copy(), y=y_2p.copy(), z=z,
            rho_V=result_2p.rho_V, rho_L1=result_2p.rho_L, rho_L2=None,
            h=result_2p.h, s=result_2p.s,
            iterations=result_2p.iterations,
            K_VL1=result_2p.K, K_L2L1=None,
            two_phase_result=result_2p,
        )

    (beta_V, beta_L1, beta_L2, x1, x2, y, K_VL1, K_L2L1,
     rho_V, rho_L1, rho_L2, niter) = ss_result

    # Step 5: check if any phase fraction collapsed (< 1e-5) -- if so, 2-phase
    if beta_V < 1e-5 or beta_L1 < 1e-5 or beta_L2 < 1e-5:
        # Collapse to 2-phase, use 2-phase result
        return ThreePhaseFlashResult(
            phase="VLE", T=T, p=p,
            beta_V=beta_2p, beta_L1=1.0 - beta_2p, beta_L2=0.0,
            x1=x_2p.copy(), x2=x_2p.copy(), y=y_2p.copy(), z=z,
            rho_V=result_2p.rho_V, rho_L1=result_2p.rho_L, rho_L2=None,
            h=result_2p.h, s=result_2p.s,
            iterations=result_2p.iterations,
            K_VL1=result_2p.K, K_L2L1=None,
            two_phase_result=result_2p,
        )

    # Compute mixture-average enthalpy and entropy
    h_V = enthalpy(rho_V, T, y, mixture)
    s_V = entropy(rho_V, T, y, mixture)
    h_L1 = enthalpy(rho_L1, T, x1, mixture)
    s_L1 = entropy(rho_L1, T, x1, mixture)
    h_L2 = enthalpy(rho_L2, T, x2, mixture)
    s_L2 = entropy(rho_L2, T, x2, mixture)
    h_total = beta_V * h_V + beta_L1 * h_L1 + beta_L2 * h_L2
    s_total = beta_V * s_V + beta_L1 * s_L1 + beta_L2 * s_L2

    return ThreePhaseFlashResult(
        phase="VLLE", T=T, p=p,
        beta_V=beta_V, beta_L1=beta_L1, beta_L2=beta_L2,
        x1=x1, x2=x2, y=y, z=z,
        rho_V=rho_V, rho_L1=rho_L1, rho_L2=rho_L2,
        h=h_total, s=s_total,
        iterations=niter,
        K_VL1=K_VL1, K_L2L1=K_L2L1,
    )
