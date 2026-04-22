"""Mixture critical-point solver for cubic EOS.

Heidemann-Khalil (1980) / Michelsen (1982) formulation. At the critical point,
the N x N scaled Hessian of Helmholtz free energy becomes singular AND the
third-order directional derivative along the null eigenvector vanishes.

Algorithm:
- Work in Helmholtz (T, V, n) variables with n_tot = 1 (so x = n).
- Compute A_ij^{res} = d^2 F^r / dn_i dn_j |_{T,V,n_{k != i,j}} analytically.
- Form Michelsen's scaled B = I + sqrt(n_i n_j) A^{res}_{ij} / (RT).
- Critical conditions:
    R1 = lambda_min(B) = 0          (spinodal at fixed T, V)
    R2 = u^T (dB/ds)|_{s=0} u = 0   (third-order condition)
  where u is the eigenvector with smallest eigenvalue.
- Newton's method in (T, V) with FD Jacobian on the residuals.
- Pressure at critical follows from the EOS.

Restrictions:
- Requires sigma != epsilon (so applies to PR, SRK, RK; not vdW).
- Ignores volume translation -- c_i = 0 assumed. Vapor pressures and
  critical loci are invariant under volume translation anyway, but
  V_c reported will be the untranslated value.

References:
- R.A. Heidemann and A.M. Khalil, AIChE J. 26, 769 (1980).
- M.L. Michelsen and R.A. Heidemann, AIChE J. 27, 521 (1981).
- Michelsen & Mollerup, "Thermodynamic Models: Fundamentals and
  Computational Aspects", 2007, chapter 12.
"""
import numpy as np

from .mixture import CubicMixture


# ---------------------------------------------------------------------------
# Analytic A^{res} matrix
# ---------------------------------------------------------------------------

def _A_residual_matrix(T, V, n, mixture):
    """Compute A_ij^{res} = d^2 F^r/dn_i dn_j |_{T,V,n_{k != i,j}}.

    Analytic expressions. Requires sigma != epsilon (i.e. not pure vdW).

    Parameters
    ----------
    T : float
        Temperature [K].
    V : float
        Total volume [m^3]. For unit total moles (n_tot = 1), this equals
        the molar volume in m^3/mol.
    n : array
        Mole numbers (length N). If n_tot = 1, these are mole fractions.
    mixture : CubicMixture
    """
    eps_ = mixture.epsilon
    sig = mixture.sigma
    if abs(sig - eps_) < 1e-14:
        raise NotImplementedError(
            "Critical-point solver requires sigma != epsilon. "
            "van der Waals EOS not supported."
        )
    R = mixture.R

    # Per-component a(T); b is composition-independent and cached
    a_vec = np.array([c.a_T(T)[0] for c in mixture.components])
    sqrt_a = np.sqrt(a_vec)
    b_vec = mixture.b_vec

    # Extensive mixture quantities
    B_n = float(n @ b_vec)                   # = Sigma n_k b_k
    M = np.outer(sqrt_a, sqrt_a) * mixture.one_minus_kij  # M_ij
    D_n = float(n @ M @ n)                    # = Sigma_ij n_i n_j M_ij
    S = M @ n                                 # S_i = Sigma_j n_j M_ij
    n_tot = float(n.sum())

    # Cubic-form auxiliaries
    beta = B_n / V
    one_m_beta = 1.0 - beta
    sp = 1.0 + sig * beta
    ep = 1.0 + eps_ * beta
    L = np.log(sp / ep)
    # dL/dB_n = (sig - eps) / (V * sp * ep)
    L_B = (sig - eps_) / (V * sp * ep)
    # d^2L/dB_n^2 = -(sig-eps)(sig+eps + 2 sig eps beta) / (V^2 * sp^2 * ep^2)
    L_BB = -(sig - eps_) * (sig + eps_ + 2.0 * sig * eps_ * beta) \
           / (V * V * sp * sp * ep * ep)

    # Repulsive term F^r_1 = -n RT ln(1 - beta), d^2/dn_i dn_j:
    #   = RT (b_i + b_j) / (V (1-beta))
    #   + RT n b_i b_j / (V^2 (1-beta)^2)
    A1 = (R * T / V / one_m_beta) * (b_vec[:, None] + b_vec[None, :]) \
       + (R * T * n_tot / (V * V * one_m_beta * one_m_beta)) \
         * np.outer(b_vec, b_vec)

    # Attractive term F^r_2 = -D_n L / ((sig-eps) B_n), d^2/dn_i dn_j:
    #   = -alpha * [ 2 M_ij L/B_n
    #               + 2 (S_i b_j + S_j b_i)(L_B/B_n - L/B_n^2)
    #               + D_n b_i b_j (L_BB/B_n - 2 L_B/B_n^2 + 2 L/B_n^3) ]
    alpha_ = 1.0 / (sig - eps_)
    inv_Bn = 1.0 / B_n
    inv_Bn2 = inv_Bn * inv_Bn
    inv_Bn3 = inv_Bn2 * inv_Bn

    term_a = (2.0 * L * inv_Bn) * M
    fac_b = L_B * inv_Bn - L * inv_Bn2
    # (S_i b_j + S_j b_i) is symmetric
    term_b = 2.0 * fac_b * (np.outer(S, b_vec) + np.outer(b_vec, S))
    fac_c = L_BB * inv_Bn - 2.0 * L_B * inv_Bn2 + 2.0 * L * inv_Bn3
    term_c = D_n * fac_c * np.outer(b_vec, b_vec)

    A2 = -alpha_ * (term_a + term_b + term_c)

    return A1 + A2


def _B_matrix(T, V, n, mixture):
    """Michelsen's scaled B matrix = I + sqrt(n_i n_j) A^{res}_{ij} / (RT).

    At the critical point, one eigenvalue of B is zero.
    """
    A_res = _A_residual_matrix(T, V, n, mixture)
    sqrt_n = np.sqrt(n)
    return np.eye(len(n)) + np.outer(sqrt_n, sqrt_n) * A_res / (mixture.R * T)


# ---------------------------------------------------------------------------
# Critical-point solver
# ---------------------------------------------------------------------------

def _pressure_from_TV(T, V, x, mixture):
    """Evaluate p = RT/(V - b_mix) - a_mix/((V + eps b)(V + sig b)) at given
    (T, V, x). Here V is molar volume (for n_tot = 1)."""
    a_mix, b_mix, *_ = mixture.a_b_mix(T, x)
    eps_ = mixture.epsilon; sig = mixture.sigma
    if abs(sig - eps_) > 1e-14:
        attr = a_mix / ((V + eps_ * b_mix) * (V + sig * b_mix))
    else:
        attr = a_mix / (V * V)
    return mixture.R * T / (V - b_mix) - attr


def _critical_residuals(T, V, z, mixture, eps_perturb=None):
    """Return (R1, R2, eigvec_u) at (T, V) and composition z (n_tot = 1).

    R1 = lambda_min(B)           -- spinodal
    R2 = (u^T B(n + eps u) u - u^T B(n - eps u) u) / (2 eps)
                                  -- third-order condition via directional FD
    """
    n = np.asarray(z, dtype=float)
    B = _B_matrix(T, V, n, mixture)
    # Symmetric eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = int(np.argmin(eigvals))
    lam = float(eigvals[idx])
    u = eigvecs[:, idx]

    # Compute directional derivative of u^T B(n + s u) u at s=0 by central FD.
    # eps_perturb chosen small enough that n + eps*u stays positive.
    if eps_perturb is None:
        u_max = float(np.max(np.abs(u)))
        # Keep n + eps*u strictly positive, and not too close to zero
        eps_perturb = min(0.1 * float(np.min(n)) / max(u_max, 1e-30), 1e-4)
    n_p = n + eps_perturb * u
    n_m = n - eps_perturb * u
    # Safety: if perturbation pushes a component to near zero, shrink
    while np.any(n_p <= 1e-12) or np.any(n_m <= 1e-12):
        eps_perturb *= 0.5
        n_p = n + eps_perturb * u
        n_m = n - eps_perturb * u
        if eps_perturb < 1e-12:
            raise RuntimeError("Cannot perturb along null eigenvector safely.")

    B_p = _B_matrix(T, V, n_p, mixture)
    B_m = _B_matrix(T, V, n_m, mixture)
    Q_p = float(u @ B_p @ u)
    Q_m = float(u @ B_m @ u)
    dQ = (Q_p - Q_m) / (2.0 * eps_perturb)
    return lam, dQ, u


def critical_point(z, mixture, T_init=None, V_init=None,
                   tol=1e-8, maxiter=80, step_cap=0.15, verbose=False):
    """Find the mixture critical point at composition z.

    Heidemann-Khalil conditions solved via Newton's method on (T, V) with
    residuals (lambda_min(B), dQ/ds|_{s=0}).

    Parameters
    ----------
    z : array
        Overall composition (mole fractions; will be renormalized).
    mixture : CubicMixture
        Must have sigma != epsilon (PR, SRK, RK families; vdW not supported).
    T_init, V_init : float, optional
        Initial guesses. Defaults: T_init = x-averaged Tc,
        V_init = V_c_pseudo from mixture.reduce().
    tol : float
        Convergence tolerance on max(|R1|, |R2|).
    maxiter : int
    step_cap : float
        Maximum fractional step on T or V per iteration.

    Returns
    -------
    dict with keys:
        T_c, p_c, V_c, rho_c, u (null eigenvector at convergence),
        iterations, residual
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))
    T = float(T_init)

    if V_init is None:
        # Use a pseudo-critical molar volume from the mixture's own rho_c
        T_c_pseudo, rho_c_pseudo = mixture.reduce(z)
        V = 1.0 / rho_c_pseudo
    else:
        V = float(V_init)

    for it in range(maxiter):
        R1, R2, u = _critical_residuals(T, V, z, mixture)
        if verbose:
            print(f"  it {it}: T={T:.4f}, V={V:.4e}, R1={R1:.3e}, R2={R2:.3e}")
        if max(abs(R1), abs(R2)) < tol:
            break

        # FD Jacobian of residuals w.r.t. (T, V)
        dT = max(0.01, 1e-4 * T)
        dV = max(1e-10, 1e-4 * V)
        R1_Tp, R2_Tp, _ = _critical_residuals(T + dT, V, z, mixture)
        R1_Tm, R2_Tm, _ = _critical_residuals(T - dT, V, z, mixture)
        R1_Vp, R2_Vp, _ = _critical_residuals(T, V + dV, z, mixture)
        R1_Vm, R2_Vm, _ = _critical_residuals(T, V - dV, z, mixture)

        J = np.array([
            [(R1_Tp - R1_Tm) / (2.0 * dT), (R1_Vp - R1_Vm) / (2.0 * dV)],
            [(R2_Tp - R2_Tm) / (2.0 * dT), (R2_Vp - R2_Vm) / (2.0 * dV)],
        ])

        try:
            delta = -np.linalg.solve(J, np.array([R1, R2]))
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"Singular Jacobian at T={T}, V={V}; cannot proceed."
            )

        # Cap step to step_cap fraction of current T, V
        dT_step = delta[0]
        dV_step = delta[1]
        if abs(dT_step) > step_cap * abs(T):
            dT_step = step_cap * abs(T) * np.sign(dT_step)
        if abs(dV_step) > step_cap * abs(V):
            dV_step = step_cap * abs(V) * np.sign(dV_step)

        T += dT_step
        V += dV_step
        if T <= 0 or V <= mixture.a_b_mix(T, z)[1]:
            raise RuntimeError(
                f"Critical Newton iterate left valid region: T={T}, V={V}."
            )
    else:
        raise RuntimeError(
            f"Critical-point solver did not converge in {maxiter} iters. "
            f"Final residuals: R1={R1}, R2={R2}"
        )

    # Final pressure from EOS
    p_c = _pressure_from_TV(T, V, z, mixture)
    return {
        "T_c": T,
        "p_c": p_c,
        "V_c": V,
        "rho_c": 1.0 / V,
        "u": u,
        "iterations": it + 1,
        "residual": max(abs(R1), abs(R2)),
    }
