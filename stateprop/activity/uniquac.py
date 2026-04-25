"""UNIQUAC activity coefficient model (Abrams-Prausnitz 1975).

Two-contribution model: combinatorial (entropic, depends on r_i, q_i)
plus residual (energetic, depends on tau_ij interaction parameters):

    ln gamma_i = ln gamma_i^C + ln gamma_i^R

Combinatorial part (Staverman-Guggenheim form):
    phi_i = x_i r_i / sum_j x_j r_j         (segment fraction)
    theta_i = x_i q_i / sum_j x_j q_j       (surface area fraction)
    l_i = (z/2)(r_i - q_i) - (r_i - 1)      (z = 10 by convention)
    ln gamma_i^C = ln(phi_i / x_i) + (z/2) q_i ln(theta_i / phi_i)
                   + l_i - (phi_i / x_i) sum_j x_j l_j

Residual part:
    tau_ij = exp(-(u_ij - u_jj) / RT) -- typically written as a_ij / T
    ln gamma_i^R = q_i [ 1 - ln(sum_j theta_j tau_ji)
                          - sum_j (theta_j tau_ij / sum_k theta_k tau_kj) ]

References:
    Abrams, D.S., Prausnitz, J.M., AIChE J. 21, 116 (1975).
    Reid-Prausnitz-Poling 5th ed., chapter 8.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .excess import GibbsExcessTDerivatives


class UNIQUAC(GibbsExcessTDerivatives):
    """UNIQUAC activity-coefficient model.

    Parameters
    ----------
    r, q : sequences of length N
        Pure-component volume (r) and surface area (q) parameters.
        These are typically tabulated or computed from group
        contributions (UNIFAC).
    a, b, e, f : (N, N) ndarrays, optional
        Temperature-dependence coefficients for tau_ij = -Delta_u_ij / R T.
        tau_ij(T) = exp( a_ij + b_ij / T + e_ij ln(T) + f_ij T ).
        Common practice: only b_ij/T term used (a, e, f = 0), with
        b_ij in units of K.
    """

    Z_COORD = 10.0   # coordination number, standard UNIQUAC convention

    def __init__(self, r: Sequence[float], q: Sequence[float],
                 a: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 e: Optional[np.ndarray] = None,
                 f: Optional[np.ndarray] = None):
        r = np.asarray(r, dtype=float)
        q = np.asarray(q, dtype=float)
        if r.ndim != 1 or q.ndim != 1 or r.size != q.size:
            raise ValueError("r and q must be 1-D arrays of equal length")
        N = r.size
        self._r = r
        self._q = q
        self._N = N
        # Pre-compute combinatorial l_i
        self._l = (self.Z_COORD / 2.0) * (r - q) - (r - 1.0)

        def _check(M, name):
            if M is None:
                return np.zeros((N, N))
            M = np.asarray(M, dtype=float)
            if M.shape != (N, N):
                raise ValueError(f"{name} must have shape ({N}, {N})")
            return M.copy()

        self._a = _check(a, 'a')
        self._b = _check(b, 'b')
        self._e = _check(e, 'e')
        self._f = _check(f, 'f')

    @classmethod
    def from_tau(cls, r: Sequence[float], q: Sequence[float],
                 tau: np.ndarray) -> "UNIQUAC":
        """Construct with directly-evaluated tau matrix. log(tau)
        becomes the 'a' parameter, so result is T-independent."""
        tau = np.asarray(tau, dtype=float)
        return cls(r=r, q=q, a=np.log(tau))

    @property
    def N(self) -> int:
        return self._N

    @property
    def r(self) -> np.ndarray:
        return self._r.copy()

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    def tau(self, T: float) -> np.ndarray:
        """tau_ij = exp(a_ij + b_ij/T + ...)."""
        T = float(T)
        log_tau = self._a + self._b / T + self._e * np.log(T) + self._f * T
        return np.exp(log_tau)

    def lngammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size != self._N:
            raise ValueError(f"x must have length {self._N}")
        r = self._r
        q = self._q
        # Combinatorial part
        sum_xr = float(np.sum(x * r))
        sum_xq = float(np.sum(x * q))
        # phi_i / x_i = r_i / (sum_xr / 1) but normalized: r_i / sum_xr * sum_x = r_i / sum_xr
        # phi_i = x_i r_i / sum_xr; phi_i/x_i = r_i / sum_xr
        phi_over_x = r / sum_xr      # (N,)
        theta = x * q / sum_xq        # (N,)
        # theta_i / phi_i: in terms of formula, log(theta_i / phi_i)
        # theta_i = x_i q_i / sum_xq, phi_i = x_i r_i / sum_xr
        # theta_i / phi_i = (q_i / r_i) (sum_xr / sum_xq)
        theta_over_phi = (q / r) * (sum_xr / sum_xq)   # (N,)
        # ln gamma^C
        sum_xl = float(np.sum(x * self._l))
        lngamma_C = (np.log(phi_over_x)
                     + (self.Z_COORD / 2.0) * q * np.log(theta_over_phi)
                     + self._l
                     - phi_over_x * sum_xl)

        # Residual part
        tau = self.tau(T)             # (N, N)
        # S_j := sum_k theta_k tau_kj
        S = theta @ tau               # (N,)  [theta @ tau gives sum_k theta_k tau_kj]
        # Term 1: 1 - ln(sum_j theta_j tau_ji)
        # We need sum_j theta_j tau_ji = sum over j of theta_j * tau[j,i]
        # = (theta @ tau.T)[i] -- wait that's the same as theta @ tau.T which gives sum_j theta_j tau[j,i]
        # Hmm but in matrix theta @ tau, [i] = sum_k theta_k tau[k,i] which is same as sum_j theta_j tau_ji. OK.
        # Actually I used theta @ tau which gives a row vector [j] = sum_k theta_k tau[k,j] = sum_k theta_k tau_kj
        # That's the S_j we wanted.
        # Now we need a different sum: sum_j theta_j tau_ji (sum over j with i fixed). That's sum_j theta_j tau[j,i]
        # = (tau.T @ theta)[i] = (theta @ tau)[i] only if tau is symmetric, which it isn't in general.
        # Actually theta @ tau gives row vector where [j] = sum_k theta_k tau[k,j]. If we want sum_j theta_j tau[j,i] for i,
        # that's the same sum just renamed: sum_k theta_k tau[k,i] (k=j, target i). It's S evaluated at i.
        # So sum_j theta_j tau_ji = S[i] -- same as S_j. Good.
        # Wait no -- the formula for ln gamma^R uses tau_ji (with i fixed, sum over j), which is tau[j,i].
        # sum_j theta_j tau[j,i] = (tau.T @ theta)[i]. But tau.T @ theta = theta @ tau when... no, only if
        # tau is symmetric.
        # In general, theta @ tau gives [j] = sum_k theta_k tau[k,j]. Substituting j → i gives the wanted quantity.
        # So if I index "S_at_position[i]" using S[i], that's the right one.
        # Yes -- S as computed (theta @ tau) is a vector, and S[i] = sum_k theta_k tau[k,i] = sum_j theta_j tau_ji ✓

        # Term 2: sum_j (theta_j tau_ij / sum_k theta_k tau_kj)
        # = sum_j (theta_j tau[i,j] / S[j])
        # = sum_j theta_j tau[i,j] / S[j]
        # = (tau / S[None, :]) @ theta * ... wait need to think
        # Per i: sum_j (theta_j * tau[i,j] / S[j])  
        # = sum_j (tau[i,j] * theta[j] / S[j])
        # In matrix: (tau * (theta / S)[None, :]) @ ones = tau @ (theta / S)
        term2 = tau @ (theta / S)     # (N,)

        lngamma_R = q * (1.0 - np.log(S) - term2)

        return lngamma_C + lngamma_R

    def gammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        return np.exp(self.lngammas(T, x))

    def gE_over_RT(self, T: float, x: Sequence[float]) -> float:
        """Excess Gibbs / RT."""
        x = np.asarray(x, dtype=float)
        # gE = RT [combinatorial + residual sums]
        # gE/RT = sum_i x_i ln(phi_i/x_i) + (z/2) sum_i x_i q_i ln(theta_i/phi_i)
        #          - sum_i x_i q_i ln(sum_j theta_j tau_ji)
        sum_xr = float(np.sum(x * self._r))
        sum_xq = float(np.sum(x * self._q))
        phi_over_x = self._r / sum_xr
        theta = x * self._q / sum_xq
        theta_over_phi = (self._q / self._r) * (sum_xr / sum_xq)
        gE_C = float(np.sum(x * np.log(phi_over_x))
                     + (self.Z_COORD / 2.0) * np.sum(x * self._q * np.log(theta_over_phi)))
        tau = self.tau(T)
        S = theta @ tau   # sum_k theta_k tau_kj  for each j
        # gE_R / RT = -sum_i theta_i ln(S_i) * (sum_xq) ... actually
        # The published formula is: gE^R/RT = -sum_i x_i q_i ln(sum_j theta_j tau_ji)
        # sum_j theta_j tau_ji at i = (theta @ tau)[i]
        gE_R = -float(np.sum(x * self._q * np.log(S)))
        return gE_C + gE_R

    # -----------------------------------------------------------------
    # Analytical T-derivatives (override FD mixin defaults)
    # -----------------------------------------------------------------

    def dtau_dT(self, T):
        """d(tau_ij)/dT. Since tau = exp(a + b/T + e ln T + f T),
        dtau/dT = tau * (-b/T^2 + e/T + f)."""
        T = float(T)
        tau = self.tau(T)
        d_log_tau = -self._b / (T * T) + self._e / T + self._f
        return tau * d_log_tau

    def dlngammas_dT(self, T, x):
        """Analytical d(ln gamma_i)/dT for UNIQUAC.

        Combinatorial part is T-independent at fixed x, so contributes
        zero. Residual:

            ln gamma_i^R = q_i [1 - ln(S_i) - sum_j theta_j tau_ij/S_j]
            S_j = sum_k theta_k tau_kj

        d(ln gamma_i^R)/dT = q_i [-dS_i/(S_i)
                                   - sum_j theta_j (dtau_ij/S_j - tau_ij dS_j/S_j^2)]
        """
        x = np.asarray(x, dtype=float)
        sum_xq = float(np.sum(x * self._q))
        theta = x * self._q / sum_xq
        tau = self.tau(T)
        dtau = self.dtau_dT(T)

        S = theta @ tau                    # (N,)
        dS_dT = theta @ dtau               # (N,)

        term_A = -dS_dT / S                # -dS_i/S_i
        first = dtau @ (theta / S)         # sum_j theta_j dtau_ij / S_j
        second = tau @ (theta * dS_dT / (S * S))   # sum_j theta_j tau_ij dS_j/S_j^2
        return self._q * (term_A - first + second)
