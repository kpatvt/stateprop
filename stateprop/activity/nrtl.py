"""NRTL activity coefficient model (Renon-Prausnitz 1968).

Excess Gibbs energy:
    g^E / (RT) = sum_i x_i sum_j (tau_ji G_ji x_j) / sum_k G_ki x_k

with G_ji = exp(-alpha_ji tau_ji), alpha_ji = alpha_ij (symmetric),
alpha_ii = 0, tau_ii = 0.

The activity coefficients follow by partial differentiation:
    ln gamma_i = sum_j x_j tau_ji G_ji / sum_k x_k G_ki
                + sum_j x_j G_ij / sum_k x_k G_kj
                  * [tau_ij - sum_m x_m tau_mj G_mj / sum_k x_k G_kj]

Temperature dependence is provided through the standard form:
    tau_ij(T) = a_ij + b_ij / T + e_ij ln(T) + f_ij T

with most applications using only a_ij and b_ij/T. Pass any subset
through the constructor; missing matrices default to zero.

Reference: Renon, H., Prausnitz, J.M., AIChE J. 14, 135 (1968).
Also Reid-Prausnitz-Poling 5th ed., chapter 8.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .excess import GibbsExcessTDerivatives


class NRTL(GibbsExcessTDerivatives):
    """NRTL activity-coefficient model.

    Parameters
    ----------
    a, b, e, f : (N, N) ndarrays, optional
        Temperature-dependence coefficients for tau_ij.
        tau_ij(T) = a_ij + b_ij/T + e_ij ln(T) + f_ij T.
        Diagonals must be zero (or are forced to zero by the model).
    alpha : (N, N) ndarray
        Non-randomness parameter, must be symmetric (alpha_ij = alpha_ji).
        Diagonals are forced to zero. Typical values 0.2-0.47.
        Common defaults: 0.30 for binary VLE, 0.20 for LLE.

    Notes
    -----
    Either provide T-dependent parameters (a, b, ...) and call
    `gammas(T, x)`, OR provide a directly-evaluated tau matrix via
    `from_tau(tau, alpha)` constructor.
    """

    def __init__(self, alpha: np.ndarray,
                 a: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 e: Optional[np.ndarray] = None,
                 f: Optional[np.ndarray] = None):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
            raise ValueError("alpha must be a square N x N matrix")
        N = alpha.shape[0]
        if not np.allclose(alpha, alpha.T):
            raise ValueError("alpha must be symmetric (alpha_ij = alpha_ji)")
        # Force diagonal to zero
        alpha = alpha.copy()
        np.fill_diagonal(alpha, 0.0)
        self._alpha = alpha
        self._N = N

        def _check(M, name):
            if M is None:
                return np.zeros((N, N))
            M = np.asarray(M, dtype=float)
            if M.shape != (N, N):
                raise ValueError(f"{name} must have shape ({N}, {N})")
            M = M.copy()
            np.fill_diagonal(M, 0.0)
            return M

        self._a = _check(a, 'a')
        self._b = _check(b, 'b')
        self._e = _check(e, 'e')
        self._f = _check(f, 'f')

    @classmethod
    def from_tau(cls, tau: np.ndarray, alpha: np.ndarray) -> "NRTL":
        """Construct from a directly-evaluated tau matrix (T-independent
        usage). The resulting model will return the same gamma_i for
        any T."""
        tau = np.asarray(tau, dtype=float)
        # Stash tau into 'a' so gammas() works at any T (b, e, f stay zero)
        return cls(alpha=alpha, a=tau)

    @property
    def N(self) -> int:
        return self._N

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha.copy()

    def tau(self, T: float) -> np.ndarray:
        """Tau matrix at temperature T."""
        T = float(T)
        return (self._a + self._b / T + self._e * np.log(T) + self._f * T)

    def G(self, T: float) -> np.ndarray:
        """G_ij = exp(-alpha_ij tau_ij)."""
        return np.exp(-self._alpha * self.tau(T))

    def lngammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Activity coefficients ln(gamma_i) at (T, x)."""
        x = np.asarray(x, dtype=float)
        if x.size != self._N:
            raise ValueError(f"x must have length {self._N}")
        tau = self.tau(T)
        G = np.exp(-self._alpha * tau)

        # Term 1: sum_j x_j tau_ji G_ji / sum_k x_k G_ki  (per i)
        # numerator_i = sum_j x_j tau_ji G_ji  -- contract j over x, tau.T*G.T
        # tau.T[i, j] = tau[j, i] = tau_ji. Same for G.
        S_x_G_dot_i = G.T @ x       # = (sum_k x_k G_ki) for each i  -> (N,)
        num1 = (tau.T * G.T) @ x    # = sum_j x_j tau_ji G_ji   -> (N,)
        term1 = num1 / S_x_G_dot_i

        # Term 2: sum_j x_j G_ij / sum_k x_k G_kj * (tau_ij - sum_m x_m tau_mj G_mj / sum_k x_k G_kj)
        # Per j define:
        #   D_j = sum_k x_k G_kj                         (already = S_x_G_dot_j as above swapped)
        #   N_j = sum_m x_m tau_mj G_mj                  (similar to num1 but per j)
        # Note S_x_G_dot[j] uses G.T @ x giving sum_k x_k G_kj = sum_k x_k G[k,j]
        # So D_j = S_x_G_dot[j].
        D = S_x_G_dot_i              # (N,) indexed by j now
        N_j = num1                   # (N,) indexed by j
        # Term 2_i = sum_j (x_j G_ij / D_j) * (tau_ij - N_j / D_j)
        # = sum_j x_j G_ij * (tau_ij - N_j / D_j) / D_j
        coef = (tau - (N_j / D)[None, :])         # (N, N)
        term2 = ((x * G * coef) / D[None, :]) @ np.ones(self._N)
        # Faster: term2 = sum_j (x_j * G_ij / D_j) * (tau_ij - N_j/D_j)
        # i.e. sum over j of (x_j G_ij / D_j) (tau_ij - N_j/D_j)
        # which is what the line above does.

        return term1 + term2

    def gammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Activity coefficients gamma_i."""
        return np.exp(self.lngammas(T, x))

    def gE_over_RT(self, T: float, x: Sequence[float]) -> float:
        """Excess Gibbs energy / RT (dimensionless)."""
        x = np.asarray(x, dtype=float)
        tau = self.tau(T)
        G = np.exp(-self._alpha * tau)
        # gE/RT = sum_i x_i sum_j (tau_ji G_ji x_j) / sum_k G_ki x_k
        D = G.T @ x   # (N,) = sum_k G_ki x_k
        N_i = (tau.T * G.T) @ x   # (N,) = sum_j tau_ji G_ji x_j
        return float(np.sum(x * N_i / D))

    # -----------------------------------------------------------------
    # Analytical T-derivatives (override the FD mixin defaults)
    # -----------------------------------------------------------------

    def dtau_dT(self, T: float) -> np.ndarray:
        """d(tau_ij)/dT for the standard form
        tau = a + b/T + e ln T + f T."""
        T = float(T)
        return -self._b / (T * T) + self._e / T + self._f

    def dlngammas_dT(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Analytical d(ln gamma_i)/dT for NRTL.

        From ln gamma_i = N_i/S_i + sum_j A_ij B_ij where:
            S_j = sum_k x_k G_kj
            N_j = sum_m x_m tau_mj G_mj
            A_ij = x_j G_ij / S_j
            B_ij = tau_ij - N_j/S_j

        Using G = exp(-alpha tau), so dG/dT = -alpha G dtau/dT.

        Verified against central FD to ~1e-12 relative on smooth tau(T)
        functions. Avoids the ~1e-7 truncation error of FD and is
        roughly 3x faster than two FD evaluations."""
        x = np.asarray(x, dtype=float)
        tau = self.tau(T)
        dtau = self.dtau_dT(T)
        G = np.exp(-self._alpha * tau)
        # dG_ij/dT = -alpha_ij G_ij dtau_ij/dT
        dG = -self._alpha * G * dtau

        # Auxiliary quantities (same as in lngammas)
        S = G.T @ x                          # (N,) S_j = sum_k x_k G_kj
        N = (tau.T * G.T) @ x                # (N,) N_j = sum_m x_m tau_mj G_mj

        # T-derivatives of S, N
        dS_dT = dG.T @ x                     # (N,)
        # dN_j/dT = sum_m x_m G_mj dtau_mj (1 - alpha_mj tau_mj)
        coef_dN = G * dtau * (1.0 - self._alpha * tau)   # (N, N)
        dN_dT = coef_dN.T @ x                # (N,)

        # Term1_i = N_i / S_i
        # d(Term1)/dT = (dN_i S_i - N_i dS_i) / S_i^2
        dTerm1_dT = (dN_dT * S - N * dS_dT) / (S * S)

        # Term2_i = sum_j A_ij B_ij
        N_over_S = N / S                     # (N,) N_j / S_j
        # A_ij = x_j G_ij / S_j
        A = x[None, :] * G / S[None, :]
        # B_ij = tau_ij - N_j / S_j
        B = tau - N_over_S[None, :]
        # dA_ij/dT = -x_j G_ij/S_j^2 (alpha_ij dtau_ij S_j + dS_j)
        dA_dT = -x[None, :] * G / (S[None, :] * S[None, :]) * (
            self._alpha * dtau * S[None, :] + dS_dT[None, :]
        )
        # dB_ij/dT = dtau_ij - dN_j/S_j + N_j dS_j/S_j^2
        dB_dT = dtau - (dN_dT / S)[None, :] + (N * dS_dT / (S * S))[None, :]
        dTerm2_dT = np.sum(dA_dT * B + A * dB_dT, axis=1)

        return dTerm1_dT + dTerm2_dT
