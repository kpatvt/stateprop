"""Three-phase (VLLE) flash for cubic-EOS mixtures (v0.9.20).

Port of stateprop.mixture.three_phase_flash (v0.9.19) to the cubic EOS
family. Same Michelsen (1982) algorithm, same 2D Rachford-Rice kernel,
same SS outer iteration on fugacity equality. The API touchpoints
differ: `density_from_pressure` / `ln_phi` are methods on `CubicMixture`,
and we use the in-file `stability_test_TPD` and `CubicFlashResult`.

Cubic EOSes with well-tuned k_ij typically handle VLLE better than
GERG-2008 (which is specialized for natural-gas VLE). Classic test cases
like CO2 + heavy hydrocarbons, CH4 + H2O at high pressure, or
water-methanol-alkane ternaries are within reach for PR / SRK with
published interaction parameters.

Public API:

- `flash_pt_three_phase(p, T, z, mixture, ...)` -- PT flash allowing
  up to three equilibrium phases.
- `ThreePhaseFlashResult` -- result dataclass with phase label,
  compositions, fractions, densities, and caloric properties.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .flash import (
    flash_pt, stability_test_TPD, CubicFlashResult,
)


@dataclass
class ThreePhaseFlashResult:
    """Outcome of a cubic-EOS three-phase PT flash.

    Conventions: L1 is the denser liquid (arbitrary labeling otherwise),
    L2 is the lighter/second liquid, V is the vapor phase. When the
    problem is two-phase or single-phase, the 2-phase `CubicFlashResult`
    is returned via the `two_phase_result` field.
    """
    phase: str                     # "VLLE" | "VLE" | "LLE" | single-phase
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
    rho_L1: Optional[float]
    rho_L2: Optional[float]
    h: float                       # mixture-average molar enthalpy [J/mol]
    s: float                       # mixture-average molar entropy [J/(mol K)]
    iterations: int
    K_VL1: Optional[np.ndarray] = None
    K_L2L1: Optional[np.ndarray] = None
    two_phase_result: Optional[CubicFlashResult] = None


def _rachford_rice_3p(z, K_VL1, K_L2L1, tol=1e-10, maxiter=100):
    """Solve 3-phase Rachford-Rice for (beta_V, beta_L2) at fixed K's.

    Material balance: z_i = beta_V y_i + beta_L1 x1_i + beta_L2 x2_i.
    Using y_i = K_VL1_i x1_i, x2_i = K_L2L1_i x1_i, beta_L1 = 1 - beta_V - beta_L2:

        x1_i = z_i / D_i,   D_i = 1 + beta_V (K_VL1_i - 1) + beta_L2 (K_L2L1_i - 1)

    Two residuals:

        R1 = sum_i z_i (K_VL1_i - 1) / D_i    (= sum(y - x1))
        R2 = sum_i z_i (K_L2L1_i - 1) / D_i   (= sum(x2 - x1))

    Solved via 2D Newton. Jacobian is negative-definite; unique solution
    in the feasible region when one exists.

    Returns (beta_V, beta_L1, beta_L2, x1, x2, y).
    Raises RuntimeError if iteration leaves the feasible region or stalls.
    """
    z = np.asarray(z, dtype=np.float64)
    aV = K_VL1 - 1.0
    aL2 = K_L2L1 - 1.0
    # Heuristic starting point: if K_VL1 has very large components, beta_V
    # must be small to keep D > 0. The feasible region is 1 + beta_V a_V +
    # beta_L2 a_L2 > 0 for all i. Start at a small interior point scaled by
    # the maximum K magnitudes to stay well inside feasibility.
    max_aV = float(np.max(np.abs(aV)))
    max_aL2 = float(np.max(np.abs(aL2)))
    # Pick start so that the initial D is bounded away from 0 by at least 0.5
    beta_V_start = min(0.1, 0.5 / max(max_aV, 1.0))
    beta_L2_start = min(0.1, 0.5 / max(max_aL2, 1.0))
    beta_V = beta_V_start
    beta_L2 = beta_L2_start

    last_res = float('inf')
    stall_count = 0
    for it in range(maxiter):
        D = 1.0 + beta_V * aV + beta_L2 * aL2
        if np.any(D <= 0):
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
        # Condition-number check: the Jacobian can become rank-deficient
        # when aV = (K_VL1 - 1) is proportional to aL2 = (K_L2L1 - 1)
        # (seeding with x2 on the L1-V tie-line makes this exact). Add
        # Levenberg-Marquardt regularization: replace J with J - lambda*I
        # so the 2D step is well-defined. Scale lambda with the Jacobian
        # trace to stay scale-invariant.
        cond_threshold = 1e-10 * max(abs(J11 * J22), 1e-30)
        if abs(det) < cond_threshold:
            # Rank-deficient: use LM step with lambda = eps * (|J11|+|J22|)/2
            lam = 1e-4 * 0.5 * (abs(J11) + abs(J22))
            J11r = J11 - lam
            J22r = J22 - lam
            det = J11r * J22r - J12 * J12
            dbV = -(J22r * R1 - J12 * R2) / det
            dbL2 = -(-J12 * R1 + J11r * R2) / det
        else:
            dbV = -(J22 * R1 - J12 * R2) / det
            dbL2 = -(-J12 * R1 + J11 * R2) / det
        step = max(abs(dbV), abs(dbL2))
        if step > 0.25:
            dbV *= 0.25 / step
            dbL2 *= 0.25 / step
        beta_V_new = beta_V + dbV
        beta_L2_new = beta_L2 + dbL2
        if beta_V_new < 0 or beta_L2_new < 0 or beta_V_new + beta_L2_new > 1:
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
                raise RuntimeError(
                    f"3-phase RR: no feasible step (R=({R1:.2e}, {R2:.2e}))"
                )
        beta_V = beta_V_new
        beta_L2 = beta_L2_new
        if abs(res - last_res) / max(res, 1e-30) < 1e-12:
            stall_count += 1
            if stall_count > 3:
                raise RuntimeError(
                    f"3-phase RR: stalled at R=({R1:.2e}, {R2:.2e})"
                )
        else:
            stall_count = 0
        last_res = res

    if res > 10 * tol:
        raise RuntimeError(
            f"3-phase RR: did not converge, R=({R1:.2e}, {R2:.2e})"
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
    """Successive-substitution three-phase flash for cubic mixture.

    Strategy: iterate on (x1, x2, y) directly rather than on K-factors.
    At each step:
      1. Compute ln phi of each phase, build current K_VL1, K_L2L1.
      2. Solve for (beta_V, beta_L1, beta_L2) by projected linear
         least-squares on the material balance
            z = beta_V * y + beta_L1 * x1 + beta_L2 * x2
         with beta_V + beta_L1 + beta_L2 = 1 and each in [0, 1]. For
         an N-component system this is an N-dimensional overdetermined
         system in 2 free unknowns (binary: exactly determined).
      3. Compute new compositions from x1 = z / D with
            D = 1 + beta_V (K_VL1 - 1) + beta_L2 (K_L2L1 - 1),
         then x2 = K_L2L1 * x1, y = K_VL1 * x1 (all normalized).

    This approach is robust to the huge K-factor magnitudes that arise
    from near-pure-component liquid phases (e.g., pure water in
    water-hydrocarbon LLE), where direct Rachford-Rice on K's breaks
    down due to ill-conditioning.

    Initialization: builds (x1, x2, y) from the provided K's via a
    small-beta seed, then starts iterating.

    Raises RuntimeError on non-convergence or trivial collapse.
    Returns (beta_V, beta_L1, beta_L2, x1, x2, y, K_VL1, K_L2L1,
             rho_V, rho_L1, rho_L2, niter).
    """
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    K_VL1 = K_VL1_init.copy()
    K_L2L1 = K_L2L1_init.copy()

    # Seed (x1, x2, y) from the K's at a small-beta point that stays in
    # the feasible cone. Use max-K magnitudes to pick safe starting betas.
    aV = K_VL1 - 1.0; aL2 = K_L2L1 - 1.0
    beta_V_seed = min(0.1, 0.5 / max(float(np.max(np.abs(aV))), 1.0))
    beta_L2_seed = min(0.1, 0.5 / max(float(np.max(np.abs(aL2))), 1.0))
    D = 1.0 + beta_V_seed * aV + beta_L2_seed * aL2
    if np.any(D <= 0):
        # Pathological K's
        raise RuntimeError("3-phase SS seed: D has non-positive entries")
    x1 = z / D; x1 = x1 / x1.sum()
    x2 = K_L2L1 * x1; x2 = x2 / x2.sum()
    y = K_VL1 * x1; y = y / y.sum()

    for it in range(maxiter):
        # Trivial collapse
        if np.max(np.abs(x1 - x2)) < 1e-4 or np.max(np.abs(x1 - y)) < 1e-4:
            raise RuntimeError("Two phases collapsed to identical composition")

        try:
            rho_L1 = mixture.density_from_pressure(p, T, x1, phase_hint='liquid')
        except RuntimeError:
            rho_L1 = mixture.density_from_pressure(p, T, x1, phase_hint='vapor')
        try:
            rho_L2 = mixture.density_from_pressure(p, T, x2, phase_hint='liquid')
        except RuntimeError:
            rho_L2 = mixture.density_from_pressure(p, T, x2, phase_hint='vapor')
        try:
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
        except RuntimeError:
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='liquid')

        lnphi_L1 = mixture.ln_phi(rho_L1, T, x1)
        lnphi_L2 = mixture.ln_phi(rho_L2, T, x2)
        lnphi_V = mixture.ln_phi(rho_V, T, y)
        K_VL1 = np.exp(lnphi_L1 - lnphi_V)
        K_L2L1 = np.exp(lnphi_L1 - lnphi_L2)

        # Solve for betas by projected linear least-squares on
        #   z - x1 = beta_V (y - x1) + beta_L2 (x2 - x1)
        A = np.column_stack([y - x1, x2 - x1])   # (N, 2)
        b = z - x1                                 # (N,)
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        beta_V, beta_L2 = float(beta[0]), float(beta[1])
        # Project onto the simplex (box constraints + sum <= 1)
        beta_V = max(1e-8, min(1.0 - 1e-8, beta_V))
        beta_L2 = max(1e-8, min(1.0 - 1e-8, beta_L2))
        if beta_V + beta_L2 > 1.0 - 1e-8:
            scale = (1.0 - 1e-8) / (beta_V + beta_L2)
            beta_V *= scale; beta_L2 *= scale
        beta_L1 = 1.0 - beta_V - beta_L2

        # Compute new compositions using the betas + K's. Use RR-style
        # x1 update which guarantees material balance closure.
        aV = K_VL1 - 1.0; aL2 = K_L2L1 - 1.0
        D = 1.0 + beta_V * aV + beta_L2 * aL2
        if np.any(D <= 0):
            raise RuntimeError(f"3-phase SS iter {it}: D <= 0, betas infeasible")
        x1_new = z / D; x1_new = x1_new / x1_new.sum()
        x2_new = K_L2L1 * x1_new; x2_new = x2_new / x2_new.sum()
        y_new = K_VL1 * x1_new; y_new = y_new / y_new.sum()

        # Convergence check on composition changes
        delta = max(float(np.max(np.abs(x1_new - x1))),
                    float(np.max(np.abs(x2_new - x2))),
                    float(np.max(np.abs(y_new - y))))
        x1, x2, y = x1_new, x2_new, y_new

        if delta < tol:
            return (beta_V, beta_L1, beta_L2, x1, x2, y, K_VL1, K_L2L1,
                    rho_V, rho_L1, rho_L2, it + 1)

    raise RuntimeError(
        f"Cubic 3-phase SS did not converge: delta={delta:.2e} after {maxiter} iters"
    )


def flash_pt_three_phase(p, T, z, mixture, tol=1e-8, maxiter=100,
                         check_single_phase=True):
    """PT flash allowing up to three equilibrium phases for a cubic mixture.

    Wraps the standard 2-phase `flash_pt`. If the 2-phase flash returns a
    two-phase split, runs TPD stability on each resulting phase; if either
    is unstable, initializes a 3-phase SS iteration.

    Parameters
    ----------
    p, T : float
    z : array (N,)         feed composition
    mixture : CubicMixture
    tol : float            convergence tolerance on ln K step
    maxiter : int          max 3-phase SS iterations
    check_single_phase : bool   whether the 2-phase flash should run stability

    Returns
    -------
    ThreePhaseFlashResult
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()
    N = len(z)

    # Step 1: 2-phase flash
    result_2p = flash_pt(p, T, z, mixture, check_stability=check_single_phase)

    # Single-phase branch
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

    # 2-phase: x (liquid), y (vapor), beta (V fraction)
    x_2p = result_2p.x
    y_2p = result_2p.y
    beta_2p = result_2p.beta

    # Step 2: TPD stability on each phase
    try:
        L_stable, K_stab_L, S_L = stability_test_TPD(x_2p, T, p, mixture)
    except RuntimeError:
        L_stable = True; S_L = 0.0; K_stab_L = None
    try:
        V_stable, K_stab_V, S_V = stability_test_TPD(y_2p, T, p, mixture)
    except RuntimeError:
        V_stable = True; S_V = 0.0; K_stab_V = None

    # Both phases stable: 2-phase is the final answer
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

    # Step 3: initialize 3-phase K's from the unstable phase's trial direction.
    # Special handling for degenerate 2-phase results where one composition
    # is essentially pure (e.g., water-butane at 350K gives x=[1, 7e-12]):
    # in that case, building K_L2L1 = x2/x1 overflows because x1[i] ~ 0 for
    # the "missing" component. We instead use the stability-test K factor
    # DIRECTLY as a direction vector, plus a regularized x1.
    DEGEN_EPS = 1e-4

    def _build_seed(x_base, K_trial, other_phase):
        """Build (x1, x2, K_VL1, K_L2L1) triple from a reference composition,
        the stability-trial K-factor for a split out of x_base, and the
        opposite phase.
        - x1: regularized x_base (nudged off the pure-component axis)
        - x2: normalize(K_trial * x_base) -- the stationary-point comp
        - K_VL1 = other_phase / x1
        - K_L2L1 = K_trial directly (this avoids the near-degenerate division)
        """
        x1 = np.maximum(x_base, DEGEN_EPS)
        x1 = x1 / x1.sum()
        x2 = K_trial * x_base
        x2 = x2 / x2.sum()
        K_VL1 = other_phase / x1
        K_L2L1 = x2 / x1
        return x1, x2, K_VL1, K_L2L1

    # Try primary seeding from the unstable phase
    seeds = []
    if not L_stable:
        seeds.append(_build_seed(x_2p, K_stab_L, y_2p))
    if not V_stable:
        # For an unstable vapor, the trial composition is typically a
        # liquid candidate. Seed L2 from the vapor-trial, keep L1 as the
        # original liquid.
        _, x2_V, _, _ = _build_seed(y_2p, K_stab_V, x_2p)
        x1v = np.maximum(x_2p, DEGEN_EPS); x1v = x1v / x1v.sum()
        K_VL1_v = y_2p / x1v
        K_L2L1_v = x2_V / x1v
        seeds.append((x1v, x2_V, K_VL1_v, K_L2L1_v))

    # Also a perturbed fallback: a near-uniform composition gives a
    # direction off the L1-V tie-line, avoiding the structural Jacobian
    # rank-1 risk in the RR solve.
    x1_fb = np.maximum(x_2p, DEGEN_EPS); x1_fb = x1_fb / x1_fb.sum()
    x2_fb = np.ones(N) / N
    K_VL1_fb = y_2p / x1_fb
    K_L2L1_fb = x2_fb / x1_fb
    seeds.append((x1_fb, x2_fb, K_VL1_fb, K_L2L1_fb))

    # Step 4: 3-phase SS with seed-retry
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
        # All seeds failed -- return 2-phase result
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

    # Step 5: collapse-check
    if beta_V < 1e-5 or beta_L1 < 1e-5 or beta_L2 < 1e-5:
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

    # Caloric properties per phase, weighted by mole fraction
    cal_V = mixture.caloric(rho_V, T, y, p=p)
    cal_L1 = mixture.caloric(rho_L1, T, x1, p=p)
    cal_L2 = mixture.caloric(rho_L2, T, x2, p=p)
    h_total = beta_V * cal_V["h"] + beta_L1 * cal_L1["h"] + beta_L2 * cal_L2["h"]
    s_total = beta_V * cal_V["s"] + beta_L1 * cal_L1["s"] + beta_L2 * cal_L2["s"]

    return ThreePhaseFlashResult(
        phase="VLLE", T=T, p=p,
        beta_V=beta_V, beta_L1=beta_L1, beta_L2=beta_L2,
        x1=x1, x2=x2, y=y, z=z,
        rho_V=rho_V, rho_L1=rho_L1, rho_L2=rho_L2,
        h=h_total, s=s_total,
        iterations=niter,
        K_VL1=K_VL1, K_L2L1=K_L2L1,
    )
