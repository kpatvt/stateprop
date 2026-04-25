"""
Kunz-Wagner reducing functions for mixture-reduced temperature and density,
and their composition derivatives.

Formulas (Kunz & Wagner 2012, eq. 14-15 and Appendix A):

    T_r(x)     = sum_i sum_j  x_i x_j * beta_T_ij * gamma_T_ij
                             * (x_i + x_j) / (beta_T_ij^2 * x_i + x_j)
                             * sqrt(T_ci T_cj)

    1/rho_r(x) = sum_i sum_j  x_i x_j * beta_v_ij * gamma_v_ij
                             * (x_i + x_j) / (beta_v_ij^2 * x_i + x_j)
                             * (1/8) * (1/rho_ci^(1/3) + 1/rho_cj^(1/3))^3

The binary parameters (beta, gamma) are symmetric in i,j in the sense that
beta_ij * beta_ji = 1 and gamma_ij = gamma_ji; so we only store one value
per unordered pair. The convention used here: for i <= j, beta_T_ij is
stored in the table; the reverse beta_T_ji is computed as 1/beta_T_ij.

For self-pairs (i = j), beta = gamma = 1.

Derivatives needed for the mixture model (Kunz-Wagner Appendix A):
  d T_r / d x_i          -- needed for composition derivatives of alpha
  d (1/rho_r) / d x_i
  d^2 T_r / (d x_i d x_j)    -- needed for the Hessian in some flash methods
  d^2 (1/rho_r) / (d x_i d x_j)

For clarity we implement these directly from the definitions rather than
precomputing.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

try:
    from numba import njit
except ImportError:
    # Numba is optional; fall back to a pass-through decorator. The hot
    # _reducing_derivatives_kernel below will still work but at Python speed.
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn): return fn
        return deco


@njit(cache=True)
def _reducing_derivatives_kernel(x, T_c, vc, sqrtTcTc, vr_factor,
                                 beta_T_mat, gamma_T_mat,
                                 beta_v_mat, gamma_v_mat):
    """Numba-JIT inner of KunzWagnerReducing.derivatives.

    Mathematically identical to the original Python double loop in
    derivatives() (which still exists below as a reference / fallback);
    factored out so that the inner loops compile to optimized machine code
    via Numba. Accepts pre-computed parameter matrices (built once in the
    KunzWagnerReducing constructor) so all dict lookups are eliminated.

    Returns (Tr, inv_rho_r, dTr_dxi, d_invrho_dxi).
    """
    N = x.shape[0]
    Tr = 0.0
    inv_rho_r = 0.0
    dTr = np.zeros(N)
    d_invrho = np.zeros(N)

    for i in range(N):
        xi = x[i]
        Tci = T_c[i]
        vci = vc[i]
        Tr += xi * xi * Tci
        inv_rho_r += xi * xi * vci
        dTr[i] += 2.0 * xi * Tci
        d_invrho[i] += 2.0 * xi * vci

        for j in range(N):
            if j == i:
                continue
            xj = x[j]

            # Temperature contribution from ordered pair (i, j)
            betaT = beta_T_mat[i, j]
            gammaT = gamma_T_mat[i, j]
            sqrtTcij = sqrtTcTc[i, j]
            denom = betaT * betaT * xi + xj
            if denom > 0:
                f = xi * xj * (xi + xj) / denom
                Tr += betaT * gammaT * sqrtTcij * f
                # df/dxi
                dfdxi = (xj * (xi + xj) + xi * xj) * denom - xi * xj * (xi + xj) * betaT * betaT
                dfdxi /= denom * denom
                dTr[i] += betaT * gammaT * sqrtTcij * dfdxi
                # df/dxj
                dfdxj = (xi * (xi + xj) + xi * xj) * denom - xi * xj * (xi + xj) * 1.0
                dfdxj /= denom * denom
                dTr[j] += betaT * gammaT * sqrtTcij * dfdxj

            # Volume contribution from ordered pair (i, j)
            betav = beta_v_mat[i, j]
            gammav = gamma_v_mat[i, j]
            vfac = vr_factor[i, j]
            denomv = betav * betav * xi + xj
            if denomv > 0:
                fv = xi * xj * (xi + xj) / denomv
                inv_rho_r += betav * gammav * vfac * fv
                dfvdxi = (xj * (xi + xj) + xi * xj) * denomv - xi * xj * (xi + xj) * betav * betav
                dfvdxi /= denomv * denomv
                d_invrho[i] += betav * gammav * vfac * dfvdxi
                dfvdxj = (xi * (xi + xj) + xi * xj) * denomv - xi * xj * (xi + xj) * 1.0
                dfvdxj /= denomv * denomv
                d_invrho[j] += betav * gammav * vfac * dfvdxj

    return Tr, inv_rho_r, dTr, d_invrho


def _cube_root(x):
    return x ** (1.0 / 3.0)


@dataclass
class BinaryParams:
    """Kunz-Wagner binary interaction parameters for one component pair.

    The four reducing-function parameters (beta_T, gamma_T, beta_v, gamma_v)
    are always present. The optional departure-function slot carries the
    scaling factor F_ij plus the DepartureFunction alpha_r_ij; if `departure`
    is None, no departure term is contributed (simplified-GERG approximation).

    Stored for i <= j; for i > j, parameters are obtained via the symmetry
    relations:
        beta_T_ji = 1 / beta_T_ij,    gamma_T_ji = gamma_T_ij
        beta_v_ji = 1 / beta_v_ij,    gamma_v_ji = gamma_v_ij
        F_ji = F_ij                   (symmetric scalar)
    """
    beta_T: float = 1.0
    gamma_T: float = 1.0
    beta_v: float = 1.0
    gamma_v: float = 1.0
    F: float = 0.0
    # Optional: departure function object (evaluator for alpha_r_ij)
    departure: object = None    # DepartureFunction or None


class KunzWagnerReducing:
    """Reducing functions T_r(x), rho_r(x) for an N-component mixture.

    Uses the Kunz-Wagner combining rule. Binary parameters are stored in a
    dict keyed by (i, j) with i < j.

    Parameters
    ----------
    T_c : array-like, shape (N,)
        Critical temperatures of each component [K].
    rho_c : array-like, shape (N,)
        Critical densities of each component [mol/m^3].
    binary : dict[(int, int), BinaryParams] or None
        Binary interaction parameters, keyed by (i, j) with i < j.
        Missing pairs default to ideal mixing (beta=gamma=1, F=0).
    """
    def __init__(self, T_c, rho_c, binary=None):
        self.T_c = np.asarray(T_c, dtype=np.float64)
        self.rho_c = np.asarray(rho_c, dtype=np.float64)
        self.N = len(self.T_c)
        self.binary = binary if binary is not None else {}
        # Precompute fixed quantities
        self._sqrtTcTc = np.sqrt(np.outer(self.T_c, self.T_c))   # sqrt(Tc_i * Tc_j)
        self._vc = 1.0 / self.rho_c                              # 1/rho_c_i
        cr = _cube_root(self._vc)                                # vc_i^(1/3)
        self._vr_factor = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                s = cr[i] + cr[j]
                self._vr_factor[i, j] = 0.125 * s * s * s       # (1/8) * (vc_i^(1/3) + vc_j^(1/3))^3

        # Pre-compute (N, N) parameter matrices so the hot Numba kernel
        # in `derivatives()` can do array indexing instead of going through
        # the Python dict-lookup `beta_gamma_T(i, j)` call. The diagonals
        # are 1.0 (i==j returns 1.0 by the helper convention); off-diagonals
        # are looked up with the asymmetric Kunz-Wagner rule (beta_ji = 1/beta_ij).
        self._beta_T_mat = np.ones((self.N, self.N))
        self._gamma_T_mat = np.ones((self.N, self.N))
        self._beta_v_mat = np.ones((self.N, self.N))
        self._gamma_v_mat = np.ones((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                bT, gT = self.beta_gamma_T(i, j)
                bv, gv = self.beta_gamma_v(i, j)
                self._beta_T_mat[i, j] = bT
                self._gamma_T_mat[i, j] = gT
                self._beta_v_mat[i, j] = bv
                self._gamma_v_mat[i, j] = gv

    # ---- parameter lookup (honoring i<j storage) ----
    def beta_gamma_T(self, i, j):
        """Return (beta_T_ij, gamma_T_ij) with symmetry handling."""
        if i == j:
            return 1.0, 1.0
        if i < j:
            bp = self.binary.get((i, j), BinaryParams())
            return bp.beta_T, bp.gamma_T
        else:
            bp = self.binary.get((j, i), BinaryParams())
            return 1.0 / bp.beta_T, bp.gamma_T

    def beta_gamma_v(self, i, j):
        if i == j:
            return 1.0, 1.0
        if i < j:
            bp = self.binary.get((i, j), BinaryParams())
            return bp.beta_v, bp.gamma_v
        else:
            bp = self.binary.get((j, i), BinaryParams())
            return 1.0 / bp.beta_v, bp.gamma_v

    def F_ij(self, i, j):
        if i == j:
            return 0.0
        if i < j:
            return self.binary.get((i, j), BinaryParams()).F
        else:
            return self.binary.get((j, i), BinaryParams()).F

    # ---- reducing functions ----
    def evaluate(self, x):
        """Compute (T_r, rho_r) at composition x.

        Uses the Kunz-Wagner double sum with the asymmetric combining rule.
        """
        x = np.asarray(x, dtype=np.float64)
        Tr = 0.0
        inv_rho_r = 0.0
        for i in range(self.N):
            if x[i] == 0.0:
                continue
            # Diagonal term (i = i): beta=gamma=1, factor simplifies to x_i^2 * Tc_i
            Tr += x[i] * x[i] * self.T_c[i]
            inv_rho_r += x[i] * x[i] / self.rho_c[i]
            # Off-diagonal: combine contributions from (i,j) and (j,i)
            for j in range(i + 1, self.N):
                if x[j] == 0.0:
                    continue
                betaT, gammaT = self.beta_gamma_T(i, j)
                # Standard KW combining rule (symmetric combined form):
                #   contribution = 2 * x_i x_j * (x_i+x_j)/(beta^2 x_i + x_j) * beta*gamma * sqrt(Tci Tcj)
                # (because of the symmetry, we use 2 * the (i,j) term)
                denom_T = betaT * betaT * x[i] + x[j]
                Tr += 2.0 * x[i] * x[j] * (x[i] + x[j]) / denom_T * betaT * gammaT * self._sqrtTcTc[i, j]

                betav, gammav = self.beta_gamma_v(i, j)
                denom_v = betav * betav * x[i] + x[j]
                inv_rho_r += 2.0 * x[i] * x[j] * (x[i] + x[j]) / denom_v * betav * gammav * self._vr_factor[i, j]

        rho_r = 1.0 / inv_rho_r if inv_rho_r > 0 else 0.0
        return Tr, rho_r

    def derivatives(self, x):
        """Compute (T_r, rho_r, dTr_dxi, drho_r_dxi) at composition x.

        The first derivatives are taken "constrained" in the Kunz-Wagner
        convention: we treat all x_k as independent variables and do NOT
        enforce sum(x)=1 during differentiation. The mole-fraction constraint
        is applied by the caller via the projection formula.

        IMPLEMENTATION NOTE:
        T_r and 1/rho_r have the structure of a symmetric sum over all
        unordered pairs (i, j) with i <= j. Using the Kunz-Wagner asymmetric
        beta convention (beta_ji = 1/beta_ij), we verified that both orderings
        of the same unordered pair contribute identical amounts to T_r. So
        Tr = diagonal + 2 * sum_{i<j} g_ij  with g_ij given by the i,j slot's
        parameters.

        For the derivative with respect to x_k, **each term** 2*g_ij contributes:
            d/dx_k (2 g_ij) = 2 [delta_ki * g_ij'_xi(x_i,x_j) + delta_kj * g_ij'_xj(x_i,x_j)]
        So:
            dTr/dx_i gets: 2 * g_ij'_xi summed over j > i, PLUS 2 * g_ji'_xj (treating the
            same unordered pair with different ordering) summed over j < i.
        Since g_ij and g_ji are the same function of the two composition variables (by the
        KW symmetry), this is equivalently:
            dTr/dx_k = diag_term(k) + 2 * sum_{j != k} g_{kj}'_{x_k}(x_k, x_j)
        where g_kj uses the (k,j) slot parameters (i.e. beta_{kj}, gamma_{kj}).

        We implement this by iterating over all ordered (k, j) pairs k != j
        and accumulating into dTr[k] the derivative d g_{kj} / d x_k with
        factor 1 -- because the factor of 2 in Tr above is already absorbed
        by the fact that each unordered pair contributes *twice* to the sum
        over all ordered (k, j).

        Returns
        -------
        Tr : float
        rho_r : float
        dTr_dxi : ndarray, shape (N,)
        d_inv_rhor_dxi : ndarray, shape (N,)   -- derivative of 1/rho_r
        """
        x = np.asarray(x, dtype=np.float64)
        # Hot path: dispatch to the JIT-compiled kernel using pre-computed
        # parameter matrices. The Python double-loop reference implementation
        # below this line is dead code under normal use, kept as a clear
        # algorithmic reference (the kernel is mathematically identical but
        # harder to read).
        Tr, inv_rho_r, dTr, d_invrho = _reducing_derivatives_kernel(
            x, self.T_c, self._vc, self._sqrtTcTc, self._vr_factor,
            self._beta_T_mat, self._gamma_T_mat,
            self._beta_v_mat, self._gamma_v_mat,
        )
        rho_r = 1.0 / inv_rho_r if inv_rho_r > 0 else 0.0
        return Tr, rho_r, dTr, d_invrho

    def hessian(self, x):
        """Compute (d^2 T_r / dx_k dx_l, d^2 (1/rho_r) / dx_k dx_l) at x.

        Both are N x N symmetric matrices. The same KW pair-sum structure
        is used as in derivatives(); the algebra of differentiating the
        per-pair contribution u(a, b; beta) = a*b*(a+b)/(beta^2*a + b)
        twice gives:

            N(a, b)   = a*b*(a+b)
            D(a, b; beta) = beta^2 * a + b
            N_a = b*(2a + b),   N_b = a*(a + 2b),
            N_aa = 2b,  N_bb = 2a,  N_ab = 2*(a + b) = 2*s
            d2u/da^2  = (N_aa * D^2 - 2*beta^2*N_a*D + 2*beta^4*N) / D^3
            d2u/db^2  = (N_bb * D^2 - 2*N_b*D + 2*N) / D^3
            d2u/da db = (2*s * D^2 - (beta^2*N_b + N_a) * D + 2*beta^2*N) / D^3

        Each ordered pair (i, j) with i != j contributes c_ij * d2u where
        c_ij = beta_ij * gamma_ij * sqrt(Tc_i * Tc_j) (or for volume,
        c_ij = beta_v_ij * gamma_v_ij * vr_factor[i, j]).

        Hessian entries:
          (k, l) with k != l : contributions from ordered pairs (k, l) and
                               (l, k), via cross-derivative d2u/da db
          (k, k) diagonal    : 2*Tc_k (or 2*vc_k) from x_k^2 term, plus
                               for each j != k contributions from pair
                               (k, j) [d2u/da^2 with a=x_k] and (j, k)
                               [d2u/db^2 with b=x_k]
        """
        x = np.asarray(x, dtype=np.float64)
        N = self.N
        H_T = np.zeros((N, N))
        H_invrho = np.zeros((N, N))
        # Diagonal contributions from x_i^2 * Tc_i (or x_i^2 * vc_i)
        for k in range(N):
            H_T[k, k] += 2.0 * self.T_c[k]
            H_invrho[k, k] += 2.0 * self._vc[k]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                a = x[i]; b = x[j]
                s = a + b
                Nij = a * b * s
                N_a = b * (2.0 * a + b)
                N_b = a * (a + 2.0 * b)
                N_aa = 2.0 * b
                N_bb = 2.0 * a

                # Temperature pair contribution
                bT = self._beta_T_mat[i, j]; gT = self._gamma_T_mat[i, j]
                D = bT * bT * a + b
                if D > 0:
                    cij_T = bT * gT * self._sqrtTcTc[i, j]
                    D2 = D * D; D3 = D2 * D
                    bT2 = bT * bT; bT4 = bT2 * bT2
                    d2u_aa = (N_aa * D2 - 2.0 * bT2 * N_a * D + 2.0 * bT4 * Nij) / D3
                    d2u_bb = (N_bb * D2 - 2.0 * N_b * D + 2.0 * Nij) / D3
                    d2u_ab = (2.0 * s * D2 - (bT2 * N_b + N_a) * D + 2.0 * bT2 * Nij) / D3
                    # Pair (i, j) contributes c_ij * d2u with a=x_i, b=x_j:
                    #   d^2/dx_i^2  -> H[i,i] += c * d2u_aa
                    #   d^2/dx_j^2  -> H[j,j] += c * d2u_bb
                    #   d^2/dx_i dx_j -> H[i,j] += c * d2u_ab; H[j,i] += c * d2u_ab
                    H_T[i, i] += cij_T * d2u_aa
                    H_T[j, j] += cij_T * d2u_bb
                    H_T[i, j] += cij_T * d2u_ab
                    H_T[j, i] += cij_T * d2u_ab

                # Volume pair contribution (same structure with vol params)
                bv = self._beta_v_mat[i, j]; gv = self._gamma_v_mat[i, j]
                Dv = bv * bv * a + b
                if Dv > 0:
                    cij_v = bv * gv * self._vr_factor[i, j]
                    Dv2 = Dv * Dv; Dv3 = Dv2 * Dv
                    bv2 = bv * bv; bv4 = bv2 * bv2
                    d2v_aa = (N_aa * Dv2 - 2.0 * bv2 * N_a * Dv + 2.0 * bv4 * Nij) / Dv3
                    d2v_bb = (N_bb * Dv2 - 2.0 * N_b * Dv + 2.0 * Nij) / Dv3
                    d2v_ab = (2.0 * s * Dv2 - (bv2 * N_b + N_a) * Dv + 2.0 * bv2 * Nij) / Dv3
                    H_invrho[i, i] += cij_v * d2v_aa
                    H_invrho[j, j] += cij_v * d2v_bb
                    H_invrho[i, j] += cij_v * d2v_ab
                    H_invrho[j, i] += cij_v * d2v_ab

        return H_T, H_invrho

    def _derivatives_python_reference(self, x):
        """Reference Python implementation of derivatives(). Kept for
        debugging and as a check on the JIT kernel; not used in hot paths."""
        x = np.asarray(x, dtype=np.float64)
        Tr = 0.0
        inv_rho_r = 0.0
        dTr = np.zeros(self.N)
        d_invrho = np.zeros(self.N)

        for i in range(self.N):
            # Diagonal
            xi = x[i]
            Tci = self.T_c[i]
            vci = 1.0 / self.rho_c[i]
            Tr += xi * xi * Tci
            inv_rho_r += xi * xi * vci
            dTr[i] += 2.0 * xi * Tci
            d_invrho[i] += 2.0 * xi * vci

            for j in range(self.N):
                if j == i:
                    continue
                xj = x[j]
                # Temperature contribution from ordered pair (i, j)
                betaT, gammaT = self.beta_gamma_T(i, j)
                sqrtTcij = np.sqrt(Tci * self.T_c[j])
                # g_ij = beta*gamma*sqrt(Tc_i Tc_j) * x_i x_j (x_i+x_j) / (beta^2 x_i + x_j)
                denom = betaT * betaT * xi + xj
                if denom > 0:
                    f = xi * xj * (xi + xj) / denom
                    Tr += betaT * gammaT * sqrtTcij * f
                    # d f / d x_i (the "first slot" derivative at ordered pair (i,j))
                    # f = (xi*xj*(xi+xj)) / D, D = beta^2 xi + xj
                    # df/dxi = [ (xj*(xi+xj) + xi*xj) * D - xi*xj*(xi+xj) * beta^2 ] / D^2
                    dfdxi = (xj * (xi + xj) + xi * xj) * denom - xi * xj * (xi + xj) * betaT * betaT
                    dfdxi /= denom * denom
                    dTr[i] += betaT * gammaT * sqrtTcij * dfdxi
                    # d f / d x_j (the "second slot" derivative at ordered pair (i,j))
                    # df/dxj = [ (xi*(xi+xj) + xi*xj) * D - xi*xj*(xi+xj) * 1 ] / D^2
                    dfdxj = (xi * (xi + xj) + xi * xj) * denom - xi * xj * (xi + xj) * 1.0
                    dfdxj /= denom * denom
                    dTr[j] += betaT * gammaT * sqrtTcij * dfdxj

                # Volume contribution from ordered pair (i, j)
                betav, gammav = self.beta_gamma_v(i, j)
                vfac = self._vr_factor[i, j]
                denomv = betav * betav * xi + xj
                if denomv > 0:
                    fv = xi * xj * (xi + xj) / denomv
                    inv_rho_r += betav * gammav * vfac * fv
                    dfvdxi = (xj * (xi + xj) + xi * xj) * denomv - xi * xj * (xi + xj) * betav * betav
                    dfvdxi /= denomv * denomv
                    d_invrho[i] += betav * gammav * vfac * dfvdxi
                    dfvdxj = (xi * (xi + xj) + xi * xj) * denomv - xi * xj * (xi + xj) * 1.0
                    dfvdxj /= denomv * denomv
                    d_invrho[j] += betav * gammav * vfac * dfvdxj

        rho_r = 1.0 / inv_rho_r if inv_rho_r > 0 else 0.0
        return Tr, rho_r, dTr, d_invrho


def make_reducing_from_components(components, binary=None):
    """Build a KunzWagnerReducing from a list of Component objects and
    an optional binary parameter table.
    """
    Tc = np.array([c.T_c for c in components])
    rc = np.array([c.rho_c for c in components])
    return KunzWagnerReducing(Tc, rc, binary)
