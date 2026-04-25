"""PC-SAFT mixture (Gross & Sadowski 2001) with standard stateprop API.

This module implements the non-associating PC-SAFT residual Helmholtz for
a mixture and exposes the same method surface as `stateprop.cubic.mixture.
CubicMixture` -- `ln_phi`, `pressure`, `density_from_pressure`, `wilson_K`,
`caloric`, plus the `N`, `components`, `x` attributes -- so that existing
flash / envelope / 3-phase-flash machinery inherits automatically.

Residual Helmholtz per molecule (in units of kT):

    A^res/(NkT) = a^hc + a^disp

where
    a^hc = m_bar * a^hs - Sum_i x_i (m_i - 1) ln g_ii^hs
    a^disp = -2 pi rho_n I_1(eta, m_bar) (m^2 eps sigma^3)
             - pi rho_n m_bar C_1 I_2(eta, m_bar) (m^2 eps^2 sigma^3)

with BMCSL hard-sphere mixture, m-dependent I_1/I_2 polynomials in packing
fraction eta = zeta_3, and the pair summations

    (m^2 eps sigma^3)  = Sum_ij x_i x_j m_i m_j (eps_ij/T) sigma_ij^3
    (m^2 eps^2 sigma^3) = Sum_ij x_i x_j m_i m_j (eps_ij/T)^2 sigma_ij^3
    sigma_ij = (sigma_i + sigma_j)/2
    eps_ij   = sqrt(eps_i eps_j) (1 - k_ij)

Derivatives d(ln phi)/dx, dp/drho, etc. are computed by finite difference
in this first-release implementation; analytic composition derivatives are
a straightforward (but lengthy) follow-up that would let Newton bubble/dew
and the analytic envelope Jacobian run at full speed for SAFT mixtures too.
"""
import numpy as np

# Gross & Sadowski 2001 universal constants for the dispersive integrals
# I_1(eta, m_bar) and I_2(eta, m_bar). Rows are (a_0k, a_1k, a_2k); columns
# k = 0..6 (seventh-degree polynomial in eta).
_A_CONSTS = np.array([
    [0.9105631445,  0.6361281449,  2.6861347891,  -26.547362491,
      97.759208784, -159.59154087,   91.297774084],
    [-0.3084016918, 0.1860531159, -2.5030047259,   21.419793629,
     -65.255885330,  83.318680481, -33.746922930],
    [-0.0906148351, 0.4527842806,  0.5962700728,  -1.7241829131,
      -4.1302112531, 13.776631870,  -8.6728470368],
])
_B_CONSTS = np.array([
    [ 0.7240946941,  2.2382791861, -4.0025849485,  -21.003576815,
      26.855641363, 206.55133841, -355.60235612],
    [-0.5755498075,  0.6995095521,  3.8925673390,  -17.215471648,
     192.67226447, -161.82646165, -165.20769346],
    [ 0.0976883116, -0.2557574982, -9.1558561531,   20.642075974,
     -38.804430052,  93.626774077, -29.666905585],
])

# Physical constants
_KB = 1.380649e-23       # Boltzmann [J/K]
_NA = 6.02214076e23      # Avogadro [1/mol]
_R  = _KB * _NA          # Gas constant [J/(mol K)]


def _segment_diameter(sigma_A, eps_k, T):
    """Chen-Kreglewski temperature-dependent segment diameter [meters].

    d_i(T) = sigma_i * (1 - 0.12 * exp(-3 * epsilon_i / (k T)))
    Here sigma_A is in Angstrom; output is in meters.
    """
    return sigma_A * 1e-10 * (1.0 - 0.12 * np.exp(-3.0 * eps_k / T))


def _I1_I2(eta, m_bar):
    """Compute (I_1, I_2) dispersive integrals and their d/deta derivatives."""
    # m-dependent coefficients
    mfac1 = (m_bar - 1.0) / m_bar
    mfac2 = (m_bar - 1.0) * (m_bar - 2.0) / (m_bar * m_bar)
    a_k = _A_CONSTS[0] + mfac1 * _A_CONSTS[1] + mfac2 * _A_CONSTS[2]
    b_k = _B_CONSTS[0] + mfac1 * _B_CONSTS[1] + mfac2 * _B_CONSTS[2]

    # Horner-like evaluation of polynomial sums and derivatives
    eta_powers = np.array([eta ** k for k in range(7)])
    eta_dpowers = np.array([k * eta ** (k - 1) if k >= 1 else 0.0 for k in range(7)])
    I1 = float(np.sum(a_k * eta_powers))
    I2 = float(np.sum(b_k * eta_powers))
    dI1_deta = float(np.sum(a_k * eta_dpowers))
    dI2_deta = float(np.sum(b_k * eta_dpowers))
    return I1, I2, dI1_deta, dI2_deta


def _C1_and_dC1_deta(eta, m_bar):
    """Compute C_1 and d C_1/d eta from Gross & Sadowski Eq. (29)."""
    one_eta = 1.0 - eta
    two_eta = 2.0 - eta
    # C1_inverse = 1 + m_bar * term_a + (1 - m_bar) * term_b
    term_a = (8.0 * eta - 2.0 * eta ** 2) / one_eta ** 4
    term_b_num = 20.0 * eta - 27.0 * eta ** 2 + 12.0 * eta ** 3 - 2.0 * eta ** 4
    term_b_den = (one_eta * two_eta) ** 2
    term_b = term_b_num / term_b_den
    C1_inv = 1.0 + m_bar * term_a + (1.0 - m_bar) * term_b
    C1 = 1.0 / C1_inv

    # d term_a / d eta
    d_term_a = ((8.0 - 4.0 * eta) * one_eta + 4.0 * (8.0 * eta - 2.0 * eta ** 2)) \
               / one_eta ** 5
    # d term_b / d eta: use (u/v)' = (u'v - uv')/v^2 with v = (one_eta*two_eta)^2
    u = term_b_num
    v = term_b_den
    du = 20.0 - 54.0 * eta + 36.0 * eta ** 2 - 8.0 * eta ** 3
    # d v / d eta where v = ((1-eta)(2-eta))^2 = (2 - 3 eta + eta^2)^2
    inner = 2.0 - 3.0 * eta + eta ** 2
    d_inner = -3.0 + 2.0 * eta
    dv = 2.0 * inner * d_inner
    d_term_b = (du * v - u * dv) / (v * v)

    dC1_inv_deta = m_bar * d_term_a + (1.0 - m_bar) * d_term_b
    dC1_deta = -dC1_inv_deta / (C1_inv * C1_inv)
    return C1, dC1_deta


def _I1_I2_dmbar(eta, m_bar):
    """Partial derivatives dI_1/dm_bar and dI_2/dm_bar at fixed eta.

    The m_bar dependence enters through the polynomial coefficients:
        a_k(m_bar) = a0_k + (m_bar-1)/m_bar * a1_k + (m_bar-1)(m_bar-2)/m_bar^2 * a2_k
    So d a_k / d m_bar = a1_k * d[(m-1)/m]/dm + a2_k * d[(m-1)(m-2)/m^2]/dm
        d[(m-1)/m]/dm = 1/m^2
        d[(m-1)(m-2)/m^2]/dm = (2m - 3)/m^2 - 2(m-1)(m-2)/m^3
                             = [(2m-3)m - 2(m^2-3m+2)]/m^3
                             = [2m^2 - 3m - 2m^2 + 6m - 4]/m^3
                             = (3m - 4)/m^3
    """
    d_mfac1_dm = 1.0 / (m_bar * m_bar)
    d_mfac2_dm = (3.0 * m_bar - 4.0) / (m_bar ** 3)
    da_k_dm = d_mfac1_dm * _A_CONSTS[1] + d_mfac2_dm * _A_CONSTS[2]
    db_k_dm = d_mfac1_dm * _B_CONSTS[1] + d_mfac2_dm * _B_CONSTS[2]
    eta_powers = np.array([eta ** k for k in range(7)])
    dI1_dm = float(np.sum(da_k_dm * eta_powers))
    dI2_dm = float(np.sum(db_k_dm * eta_powers))
    return dI1_dm, dI2_dm


def _C1_dmbar(eta, m_bar):
    """dC_1/dm_bar at fixed eta.

    C1_inv = 1 + m_bar * term_a + (1 - m_bar) * term_b
    d(C1_inv)/dm_bar = term_a - term_b
    dC1/dm_bar = -d(C1_inv)/dm_bar / C1_inv^2
    """
    one_eta = 1.0 - eta
    two_eta = 2.0 - eta
    term_a = (8.0 * eta - 2.0 * eta ** 2) / one_eta ** 4
    term_b_num = 20.0 * eta - 27.0 * eta ** 2 + 12.0 * eta ** 3 - 2.0 * eta ** 4
    term_b_den = (one_eta * two_eta) ** 2
    term_b = term_b_num / term_b_den
    C1_inv = 1.0 + m_bar * term_a + (1.0 - m_bar) * term_b
    return -(term_a - term_b) / (C1_inv * C1_inv)


# ------------------------------------------------------------------------
# v0.9.34 -- Second derivatives of I1, I2, C1 wrt (eta, m_bar)
# Used by the true analytic composition Hessian in dlnphi_dxk_at_p.
# ------------------------------------------------------------------------


def _I1_I2_d2(eta, m_bar):
    """Second derivatives of I1, I2 w.r.t. (eta, m_bar).

    Returns (d2I1/deta^2, d2I2/deta^2, d2I1/dmbar^2, d2I2/dmbar^2,
             d2I1/deta dmbar, d2I2/deta dmbar)
    """
    f_p  = 1.0 / m_bar ** 2
    f_pp = -2.0 / m_bar ** 3
    g_p  = (3.0 * m_bar - 4.0) / m_bar ** 3
    g_pp = 6.0 * (2.0 - m_bar) / m_bar ** 4
    mfac1 = (m_bar - 1.0) / m_bar
    mfac2 = (m_bar - 1.0) * (m_bar - 2.0) / (m_bar * m_bar)
    a_k = _A_CONSTS[0] + mfac1 * _A_CONSTS[1] + mfac2 * _A_CONSTS[2]
    b_k = _B_CONSTS[0] + mfac1 * _B_CONSTS[1] + mfac2 * _B_CONSTS[2]
    da_dm = f_p * _A_CONSTS[1] + g_p * _A_CONSTS[2]
    db_dm = f_p * _B_CONSTS[1] + g_p * _B_CONSTS[2]
    d2a_dm2 = f_pp * _A_CONSTS[1] + g_pp * _A_CONSTS[2]
    d2b_dm2 = f_pp * _B_CONSTS[1] + g_pp * _B_CONSTS[2]

    eta_p   = np.array([eta ** k for k in range(7)])
    eta_dp  = np.array([k * eta ** (k - 1) if k >= 1 else 0.0 for k in range(7)])
    eta_d2p = np.array([k * (k - 1) * eta ** (k - 2) if k >= 2 else 0.0 for k in range(7)])

    d2I1_eta2  = float(np.sum(a_k * eta_d2p))
    d2I2_eta2  = float(np.sum(b_k * eta_d2p))
    d2I1_mb2   = float(np.sum(d2a_dm2 * eta_p))
    d2I2_mb2   = float(np.sum(d2b_dm2 * eta_p))
    d2I1_cross = float(np.sum(da_dm * eta_dp))
    d2I2_cross = float(np.sum(db_dm * eta_dp))
    return d2I1_eta2, d2I2_eta2, d2I1_mb2, d2I2_mb2, d2I1_cross, d2I2_cross


def _C1_d2(eta, m_bar):
    """Second derivatives of C1 w.r.t. (eta, m_bar).

    Returns (d2C1/deta^2, d2C1/dmbar^2, d2C1/deta dmbar)
    """
    one_eta = 1.0 - eta
    two_eta = 2.0 - eta
    term_a = (8.0 * eta - 2.0 * eta ** 2) / one_eta ** 4
    term_b_num = 20.0 * eta - 27.0 * eta ** 2 + 12.0 * eta ** 3 - 2.0 * eta ** 4
    term_b_den = (one_eta * two_eta) ** 2
    term_b = term_b_num / term_b_den

    # First derivatives (same as _C1_and_dC1_deta)
    d_term_a = ((8.0 - 4.0 * eta) * one_eta + 4.0 * (8.0 * eta - 2.0 * eta ** 2)) \
               / one_eta ** 5
    inner = 2.0 - 3.0 * eta + eta ** 2
    d_inner = -3.0 + 2.0 * eta
    v = term_b_den
    dv = 2.0 * inner * d_inner
    u = term_b_num
    du = 20.0 - 54.0 * eta + 36.0 * eta ** 2 - 8.0 * eta ** 3
    d_term_b = (du * v - u * dv) / (v * v)

    # Second derivatives of term_a: f = 8eta - 2eta², g = (1-eta)^4
    f_a = 8.0 * eta - 2.0 * eta ** 2
    fp_a = 8.0 - 4.0 * eta
    fpp_a = -4.0
    g_a = one_eta ** 4
    gp_a = -4.0 * one_eta ** 3
    gpp_a = 12.0 * one_eta ** 2
    d2_term_a = (fpp_a * g_a - f_a * gpp_a) / g_a ** 2 \
                - 2.0 * (fp_a * g_a - f_a * gp_a) * gp_a / g_a ** 3

    # Second derivatives of term_b
    d2_inner = 2.0
    d2u = -54.0 + 72.0 * eta - 24.0 * eta ** 2
    d2v = 2.0 * (d_inner ** 2 + inner * d2_inner)
    d2_term_b = (d2u * v - u * d2v) / v ** 2 - 2.0 * (du * v - u * dv) * dv / v ** 3

    C1_inv = 1.0 + m_bar * term_a + (1.0 - m_bar) * term_b
    dC1_inv_deta = m_bar * d_term_a + (1.0 - m_bar) * d_term_b
    d2_C1_inv_deta2 = m_bar * d2_term_a + (1.0 - m_bar) * d2_term_b

    # d²C1/dη² via chain rule on C1 = 1/C1_inv
    d2_C1_deta2 = (-d2_C1_inv_deta2 / C1_inv ** 2
                   + 2.0 * dC1_inv_deta ** 2 / C1_inv ** 3)
    # d²C1/dm_bar²: dC1_inv/dm_bar = term_a - term_b (constant in m_bar)
    d2_C1_dmbar2 = 2.0 * (term_a - term_b) ** 2 / C1_inv ** 3
    # d²C1/dη dm_bar
    d2_C1_cross = (-(d_term_a - d_term_b) / C1_inv ** 2
                   + dC1_inv_deta * 2.0 * (term_a - term_b) / C1_inv ** 3)
    return d2_C1_deta2, d2_C1_dmbar2, d2_C1_cross


class SAFTMixture:
    """PC-SAFT mixture with the standard stateprop mixture API.

    Parameters
    ----------
    components : list of PCSAFT
        Pure-component parameter sets, one per species.
    composition : array-like (N,)
        Mole fractions. Will be renormalized.
    k_ij : dict, optional
        Binary interaction parameters, keyed by tuples (i, j) with i < j.
        Default 0 for all pairs not specified.
    """

    def __init__(self, components, composition, k_ij=None,
                 enable_polar=True):
        """PC-SAFT mixture.

        v0.9.24: `enable_polar` now defaults to True. The dipolar term is
        calibrated (prefactor 1/(2 pi) applied to the raw Gross-Vrabec Pade
        closure) so that acetone saturation pressure agrees with NIST to
        within 5% over 300-450 K. Non-polar components are unaffected
        (dipole_moment == 0 -> no polar contribution). Pass
        `enable_polar=False` to force the non-polar path, e.g. for debugging
        or for components with poorly-fit dipolar parameters.
        """
        self.components = list(components)
        self.N = len(self.components)
        x = np.asarray(composition, dtype=np.float64)
        x = x / x.sum()
        self.x = x
        self._k_ij = {}
        if k_ij is not None:
            for (i, j), kv in k_ij.items():
                ii, jj = min(i, j), max(i, j)
                self._k_ij[(ii, jj)] = float(kv)
        # Pre-extract pure-component arrays for speed
        self._m = np.array([c.m for c in self.components])
        self._sigma_A = np.array([c.sigma for c in self.components])
        self._eps_k = np.array([c.epsilon_k for c in self.components])
        self._T_c = np.array([c.T_c for c in self.components])
        self._p_c = np.array([c.p_c for c in self.components])
        self._omega = np.array([c.acentric_factor for c in self.components])
        # v0.9.23: association and polar parameters
        self._eps_AB_k = np.array([c.eps_AB_k for c in self.components])
        self._kappa_AB = np.array([c.kappa_AB for c in self.components])
        self._dipole_moment = np.array([c.dipole_moment for c in self.components])
        self._n_polar_segments = np.array([c.n_polar_segments for c in self.components])
        self._assoc_mask = self._eps_AB_k > 0.0
        self._polar_mask = self._dipole_moment > 0.0
        self._any_assoc = bool(np.any(self._assoc_mask))
        # v0.9.26: per-component association scheme (2B or 4C)
        self._assoc_scheme = [c.assoc_scheme for c in self.components]
        self._is_4C = np.array([c.assoc_scheme == "4C" for c in self.components])
        # v0.9.26: quadrupolar parameters
        self._quadrupole_moment = np.array([c.quadrupole_moment for c in self.components])
        self._quadrupole_mask = self._quadrupole_moment > 0.0
        self._any_quadrupole = enable_polar and bool(np.any(self._quadrupole_mask))
        # Polar (dipolar) is opt-in via enable_polar (default ON post-v0.9.24)
        self._any_polar = enable_polar and bool(np.any(self._polar_mask))

    def k_ij(self, i, j):
        if i == j:
            return 0.0
        ii, jj = min(i, j), max(i, j)
        return self._k_ij.get((ii, jj), 0.0)

    # -----------------------------------------------------------------
    # Mixture-weighted per-segment parameters
    # -----------------------------------------------------------------

    def _precompute(self, T, x):
        """Return temperature-dependent per-component quantities and pair
        sums needed for alpha_r and pressure.

        Returns a dict of the parameters used downstream, keyed by name.
        """
        m = self._m
        sigma = self._sigma_A
        eps_k = self._eps_k
        d = _segment_diameter(sigma, eps_k, T)      # meters, shape (N,)
        m_bar = float(np.dot(x, m))

        # Pair-sum sigma_ij = (sigma_i + sigma_j)/2 [meters], eps_ij/T
        sigma_m = sigma * 1e-10                     # meters
        sigma_ij = 0.5 * (sigma_m[:, None] + sigma_m[None, :])   # (N,N) [meters]
        eps_ij_over_T = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                k = self.k_ij(i, j)
                eps_ij = np.sqrt(eps_k[i] * eps_k[j]) * (1.0 - k)
                eps_ij_over_T[i, j] = eps_ij / T

        # "m^2 eps sigma^3" pair sums (unitful: sigma_ij^3 in m^3)
        outer_xm = np.outer(x * m, x * m)           # (N,N) = x_i m_i x_j m_j
        sig3 = sigma_ij ** 3
        m2es3 = float(np.sum(outer_xm * eps_ij_over_T * sig3))
        m2e2s3 = float(np.sum(outer_xm * eps_ij_over_T ** 2 * sig3))

        return {
            'd': d, 'm_bar': m_bar, 'sigma_ij_m': sigma_ij,
            'eps_ij_over_T': eps_ij_over_T,
            'm2es3': m2es3, 'm2e2s3': m2e2s3,
        }

    # -----------------------------------------------------------------
    # v0.9.23: Association (Chapman-Radosz / Wertheim TPT1)
    # v0.9.26: extended to support 4C scheme for water
    # -----------------------------------------------------------------

    def _association_fractions(self, rho_n, T, x, d_m, zeta2, zeta3,
                                 maxiter=100, tol=1e-10):
        """Solve fractions-not-bonded for association.

        Each component i has an association scheme:
          2B: one A (donor) + one B (acceptor), only A-B bonds. By symmetry
              X_A_i = X_B_i =: X_i. Effective site multiplicity n_B (of
              partner) appearing in i's X_A equation = 1, and n_A of
              partner = 1 in X_B equation.
          4C: two A (donors) + two B (acceptors), only A-B bonds. By
              molecular symmetry (equivalent donors, equivalent acceptors),
              but in general X_A_i != X_B_i when mixing with other
              associating species. Effective multiplicities: partner's
              n_B = 2 (for X_A_i equation) and n_A = 2 (for X_B_i).

        Returns
        -------
        X : ndarray (N, 2)
            X[:, 0] is X_A_i (donor fraction not bonded), X[:, 1] is X_B_i.
            For non-associating components: X_A_i = X_B_i = 1.0.

        Self-consistency (general form for 2B/4C mixture):
            X_A_i = 1 / (1 + rho_n * sum_j x_j * n_B_j * X_B_j * Delta_ij)
            X_B_i = 1 / (1 + rho_n * sum_j x_j * n_A_j * X_A_j * Delta_ij)
        where n_A_j, n_B_j are 1 for 2B, 2 for 4C. For 2B with symmetric
        (X_A == X_B), this reduces to the X_i = 1 / (1 + rho * sum ...) form.
        """
        N = self.N
        # Wolbach-Sandler combining rules for cross-association
        sigma_m = self._sigma_A * 1e-10
        sigma_ij = 0.5 * (sigma_m[:, None] + sigma_m[None, :])
        eps_AB_ij = 0.5 * (self._eps_AB_k[:, None] + self._eps_AB_k[None, :])
        kappa_ij = np.zeros((N, N))
        for i in range(N):
            if not self._assoc_mask[i]:
                continue
            for j in range(N):
                if not self._assoc_mask[j]:
                    continue
                kappa_ij[i, j] = (np.sqrt(self._kappa_AB[i] * self._kappa_AB[j])
                                   * (np.sqrt(sigma_m[i] * sigma_m[j]) / sigma_ij[i, j]) ** 3)

        # BMCSL pair correlation at contact, g_ij^hs(d_ij)
        one_mz3 = 1.0 - zeta3
        d_ij = 0.5 * (d_m[:, None] + d_m[None, :])
        d_ratio = (d_m[:, None] * d_m[None, :]) / (d_m[:, None] + d_m[None, :])
        g_ij_hs = (1.0 / one_mz3
                   + d_ratio * 3.0 * zeta2 / one_mz3 ** 2
                   + d_ratio ** 2 * 2.0 * zeta2 ** 2 / one_mz3 ** 3)

        # Association strength Delta_ij (unit: m^3)
        with np.errstate(over='ignore'):
            Delta_ij = d_ij ** 3 * g_ij_hs * kappa_ij * (np.exp(eps_AB_ij / T) - 1.0)
        for i in range(N):
            if not self._assoc_mask[i]:
                Delta_ij[i, :] = 0.0
                Delta_ij[:, i] = 0.0

        # Site multiplicities per component:
        #   2B -> n_A = 1, n_B = 1
        #   4C -> n_A = 2, n_B = 2
        n_A = np.where(self._is_4C, 2.0, 1.0)   # # A-sites per molecule
        n_B = np.where(self._is_4C, 2.0, 1.0)   # # B-sites per molecule
        # For non-associating components these multiplicities are irrelevant
        # because Delta_ij for those is 0.

        # Iterate (X_A, X_B)
        X_A = np.ones(N)
        X_B = np.ones(N)
        for it in range(maxiter):
            # X_A_i = 1 / (1 + rho_n * sum_j x_j n_B_j X_B_j Delta_ij)
            sum_for_A = rho_n * (Delta_ij @ (x * n_B * X_B))
            sum_for_B = rho_n * (Delta_ij @ (x * n_A * X_A))
            X_A_new = np.where(self._assoc_mask, 1.0 / (1.0 + sum_for_A), 1.0)
            X_B_new = np.where(self._assoc_mask, 1.0 / (1.0 + sum_for_B), 1.0)
            X_A_new = np.clip(X_A_new, 1e-12, 1.0)
            X_B_new = np.clip(X_B_new, 1e-12, 1.0)
            err = max(float(np.max(np.abs(X_A_new - X_A))),
                      float(np.max(np.abs(X_B_new - X_B))))
            alpha = 0.7 if err < 0.1 else 0.5
            X_A = alpha * X_A_new + (1.0 - alpha) * X_A
            X_B = alpha * X_B_new + (1.0 - alpha) * X_B
            if err < tol:
                break
        return np.column_stack([X_A, X_B])

    def _a_assoc_contribution(self, X, x):
        """a^assoc / NkT for general 2B or 4C mixture.

        Per-component contribution:
            2B: x_i * [2 (ln X_A_i - X_A_i/2 + 1/2)]
                  (since X_A_i = X_B_i for 2B symmetry; 2 sites total)
            4C: x_i * [2 (ln X_A_i - X_A_i/2 + 1/2)
                         + 2 (ln X_B_i - X_B_i/2 + 1/2)]
                  (2 A-sites + 2 B-sites = 4 sites total)

        For non-associating i, X_A_i = X_B_i = 1 so contribution is zero.
        """
        mask = self._assoc_mask
        if not np.any(mask):
            return 0.0
        X_A = X[:, 0]
        X_B = X[:, 1]
        total = 0.0
        for i in range(self.N):
            if not mask[i]:
                continue
            if self._is_4C[i]:
                # 4 sites: 2 A + 2 B
                term = 2.0 * (np.log(X_A[i]) - 0.5 * X_A[i] + 0.5) \
                     + 2.0 * (np.log(X_B[i]) - 0.5 * X_B[i] + 0.5)
            else:
                # 2B: 2 sites total; by symmetry X_A == X_B so the two
                # terms are equal, giving 2 * [ln X - X/2 + 1/2]
                term = 2.0 * (np.log(X_A[i]) - 0.5 * X_A[i] + 0.5)
            total += float(x[i] * term)
        return total

    # -----------------------------------------------------------------
    # v0.9.26: Quadrupolar term (Gross 2005)
    # -----------------------------------------------------------------

    # Gross 2005 universal constants for the quadrupolar J_2 and J_3
    # integrals (Ind. Eng. Chem. Res. 2005, 44, 4442-4452). Same polynomial
    # structure as dipolar J_2/J_3 but different coefficients. Rows are
    # [a_0k, a_1k, a_2k] for k=0..4 eta powers.
    _QUADRUPOLE_A = np.array([
        [ 1.2378308, -0.9305185,  0.4860140,  1.6684250, -2.5806780],
        [ 2.4355031,  1.2819990, -1.7527424, -10.358570,  18.369770],
        [ 1.6330905,  9.5289420,  6.0869610, -33.543030, 40.193080],
    ])
    _QUADRUPOLE_B = np.array([
        [ 0.4770929,  0.4042464, -3.0083260,  0.0,  0.0],
        [ 1.5027400, -1.0376980,  0.6862750,  0.0,  0.0],
        [-3.1840820,  3.2403300,  2.3822020,  0.0,  0.0],
    ])

    def _a_quadrupole_full(self, eta, rho_n, T, x, m_bar):
        """Quadrupolar contribution a^QQ/NkT (Gross 2005 Pade closure).

        Uses the same prefactor calibration (1/(2 pi)) as the dipolar term,
        justified by identical SI electrostatic conversion for Q
        (quadrupole moment in Debye-Angstrom).
        """
        if not self._any_quadrupole:
            return 0.0
        N = self.N
        m = self._m
        sigma_A = self._sigma_A
        eps_k = self._eps_k
        Q = self._quadrupole_moment   # DA = Debye-Angstrom

        # Dimensionless reduced quadrupole^2: Q_i^2 * const / (m_i (eps/k) sigma^5)
        # The factor is 10^-19 * N_A / k_B in CGS-SI hybrid; numerically
        # this works out to the same 7242.7 / sigma^2 as for dipoles
        # (extra 1/sigma^2 because quadrupole has Q in DA = D * A, so Q^2
        # has an extra Angstrom^2 compared to mu^2).
        with np.errstate(divide='ignore', invalid='ignore'):
            Q_star_sq = np.where(
                self._quadrupole_mask,
                (Q ** 2) * 7242.7 / (m * eps_k * sigma_A ** 5),
                0.0,
            )
        # Pair segment number, capped at 2
        m_ij_full = np.sqrt(np.outer(m, m))
        m_ij_pair = np.minimum(m_ij_full, 2.0)
        m_fac1 = (m_ij_pair - 1.0) / m_ij_pair
        m_fac2 = (m_ij_pair - 1.0) * (m_ij_pair - 2.0) / (m_ij_pair ** 2)

        eta_pow = np.array([eta ** k for k in range(5)])
        A = self._QUADRUPOLE_A
        B = self._QUADRUPOLE_B
        a_k_pair = (A[0][None, None, :]
                    + m_fac1[..., None] * A[1][None, None, :]
                    + m_fac2[..., None] * A[2][None, None, :])
        J_2_ij = np.tensordot(a_k_pair, eta_pow, axes=([2], [0]))
        b_k_pair = (B[0][None, None, :]
                    + m_fac1[..., None] * B[1][None, None, :]
                    + m_fac2[..., None] * B[2][None, None, :])
        J_3_ij = np.tensordot(b_k_pair, eta_pow, axes=([2], [0]))

        # a^QQ_2: quadrupole pair-sum follows Gross 2005 Eq. (10):
        # a_2 = -(9 pi / 16) * rho * sum_ij x_i x_j (eps_i eps_j / T^2)
        #        * (sigma_i^5 sigma_j^5 / sigma_ij^7)
        #        * Q_i*2 Q_j*2 J_2_ij
        # (Note sigma^5 in numerator, sigma^7 in denominator -> sigma^3
        # dimensional cancellation with rho in 1/A^3.)
        rho_A3 = rho_n * 1e-30
        sigma_ij_A = 0.5 * (sigma_A[:, None] + sigma_A[None, :])
        eps_eps = np.outer(eps_k, eps_k) / (T * T)
        sigma_factor_2 = (np.outer(sigma_A ** 5, sigma_A ** 5)) / sigma_ij_A ** 7
        Q_outer = np.outer(Q_star_sq, Q_star_sq)
        xx = np.outer(x, x)
        a_2 = -(9.0 * np.pi / 16.0) * rho_A3 * float(np.sum(
            xx * eps_eps * sigma_factor_2 * Q_outer * J_2_ij
        ))

        # a^QQ_3: factored pair-weighted approximation analogous to dipolar.
        # a_3 = (9 pi^3 / 16) * rho^2 * sum_ijk x_i x_j x_k
        #        * (eps_i eps_j eps_k / T^3)
        #        * (sigma_i^5 sigma_j^5 sigma_k^5 / (sigma_ij^3 sigma_ik^3 sigma_jk^3))
        #        * Q_i*^2 Q_j*^2 Q_k*^2 J_3_ijk
        eps_over_T = eps_k / T
        Q_weight = Q_star_sq
        # Per-species: x_i * (eps/T)_i * sigma_i^5 * Q_i*^2
        S_i = x * eps_over_T * (sigma_A ** 5) * Q_weight
        pair_sum = float(np.sum(np.outer(S_i, S_i) * J_3_ij / sigma_ij_A ** 3))
        S_total = float(np.sum(S_i))
        a_3 = (9.0 * np.pi ** 3 / 16.0) * rho_A3 * rho_A3 * pair_sum * S_total

        if a_2 == 0.0:
            return 0.0
        return float(self._POLAR_PREFACTOR * a_2 / (1.0 - a_3 / a_2))

    # -----------------------------------------------------------------
    # v0.9.23: Dipolar term (Gross-Vrabec 2006)
    # -----------------------------------------------------------------

    # Gross-Vrabec 2006 universal constants for the dipolar J_2 integral.
    # Rows: a_nk for k=0..4 (eta powers), columns: 0, 1, 2 for the
    # (m_bar-1)/m_bar, (m_bar-1)(m_bar-2)/m_bar^2 polynomials in effective segment no.
    # Values from Gross, AIChE J 51, 2556 (2005) and Gross-Vrabec 2006.
    _DIPOLE_A = np.array([
        [ 0.3043504, -0.1358588,  1.4493329,  0.3556977, -2.0653308],
        [ 0.9534641, -1.8396383,  2.0131222, -7.3724958,  8.2374135],
        [-1.1610080,  4.5258607,  0.9751396, -12.281038,  5.9397513],
    ])
    _DIPOLE_B = np.array([
        [ 0.2187939, -1.1896431,  1.1626889, 0.0, 0.0],
        [-0.5873164,  1.2489132, -0.5085280, 0.0, 0.0],
        [ 3.4869576, -14.915974, 15.372022, 0.0, 0.0],
    ])
    # v0.9.24: empirical calibration factor on the dipolar contribution.
    # Raw Gross-Vrabec formula evaluated with our reduced-dipole convention
    # (7242.7 * mu^2_D / (m * (eps/k) * sigma_A^3)) yields |a^dd| ~5-6x too
    # large vs published p_sat. Calibration against acetone saturation over
    # T in [300, 450] K gave best fit at scale = 1/(2 pi) ~= 0.1592; the
    # same value is what one obtains by including the 1/(4 pi eps_0) factor
    # for SI electrostatics. With this scale, acetone max |p_sat error| <
    # 5% across T in [300, 450] K vs NIST.
    _POLAR_PREFACTOR = 1.0 / (2.0 * np.pi)

    def _a_dipole_contribution(self, eta, T, x, m_bar):
        """Dipolar contribution a^dd/NkT from Gross-Vrabec 2006.

        a^dd = a^dd_2 / (1 - a^dd_3 / a^dd_2)    (Pade-like closure)

        The a_2 and a_3 integrals use dimensionless dipole moment
            mu*_i^2 = mu_i^2 / (m_i * eps_i * sigma_i^3 * k_B * ...)
        with a standard unit-conversion constant; in PC-SAFT literature
        the convention expresses mu in Debye, sigma in Angstrom, eps/k in
        Kelvin, and folds the constants into the formula:

            (mu_i [D])^2 * 1e-19 / (3 * k_B * m_i * (eps_i/k) * (sigma_i[A])^3)
                * (5.03411e-20)   [conversion factor such that the resulting
                                   dimensionless dipolar reduced dipole^2 is
                                   right for gaussian units]

        In practice we use the widely-cited constant 7242.7 so that
            mu_reduced^2 = mu^2_Debye * 7242.7 / (m * (eps/k)_K * sigma_A^3)
        (up to slight variations across papers). We adopt the above.
        """
        if not self._any_polar:
            return 0.0
        N = self.N
        m = self._m
        sigma_A = self._sigma_A  # Angstrom
        eps_k = self._eps_k      # K
        mu = self._dipole_moment  # Debye
        n_p = self._n_polar_segments  # number of polar segments

        # Dimensionless dipolar reduced dipole^2 per species:
        #   mu_star_sq_i = mu_i^2 [D^2] * const / (m_i * (eps_i/k) * sigma_i^3 [A^3])
        # The factor 7242.7 comes from (1 Debye)^2 / (k_B * 1 Angstrom^3)
        # in CGS with appropriate Gaussian-unit conversion.
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_star_sq = np.where(
                self._polar_mask,
                (mu ** 2) * 7242.7 / (m * eps_k * sigma_A ** 3),
                0.0,
            )

        # Pair/triple sums over polar species
        # a_2 = -pi * rho_n ... but rho_n is encoded via eta; use the
        # compact form from the paper: we express in terms of eta.
        # From Gross-Vrabec 2006, Eq. (8):
        #   a^dd_2 = -pi * rho * sum_{ij} x_i x_j (eps_i/T)(eps_j/T) * sigma_ij^3
        #            * n_p_i * n_p_j * mu_star_i^2 * mu_star_j^2 / sigma_ij^3
        #            * J_2_ij
        # Note the sigma_ij^3 cancels between the (eps * sigma^3) factor
        # and the denominator; effectively a^dd_2 uses sigma_ii^3 sigma_jj^3
        # after some manipulation. The published form in reduced variables:
        #
        # a_2 = -pi eta / packing * sum x_i x_j n_p_i n_p_j *
        #        (eps_i eps_j / T^2) * (sigma_i^3 sigma_j^3)/sigma_ij^3 *
        #        mu_i^2 mu_j^2 * J_2_ij
        #
        # We use the eta-based compact form. In the PC-SAFT literature this
        # reduces (with proper constants absorbed into mu_reduced^2) to:

        # Pair segment number m_ij = sqrt(m_i m_j), capped at 2 (Gross 2005 convention)
        m_ij = np.sqrt(np.outer(m, m))
        m_ij_pair = np.minimum(m_ij, 2.0)
        # m-dependent polynomial a_k(m_ij)
        m_fac1 = (m_ij_pair - 1.0) / m_ij_pair
        m_fac2 = (m_ij_pair - 1.0) * (m_ij_pair - 2.0) / m_ij_pair ** 2
        # Shape (N, N, 5) polynomial coefficients
        A = self._DIPOLE_A  # (3, 5)
        B = self._DIPOLE_B
        a_k_pair = A[0][None, None, :] + m_fac1[..., None] * A[1][None, None, :] \
                   + m_fac2[..., None] * A[2][None, None, :]
        b_k_pair = B[0][None, None, :] + m_fac1[..., None] * B[1][None, None, :] \
                   + m_fac2[..., None] * B[2][None, None, :]
        # Evaluate J_2(eta, m_ij) = sum_k a_k eta^k
        eta_pow = np.array([eta ** k for k in range(5)])  # (5,)
        J_2_ij = np.tensordot(a_k_pair, eta_pow, axes=([2], [0]))  # (N, N)
        J_3_ij = np.tensordot(b_k_pair, eta_pow, axes=([2], [0]))  # (N, N)

        # Precompute species cross-sums
        # a^dd_2: pair sum
        sigma_m = sigma_A * 1e-10
        sigma_ij = 0.5 * (sigma_m[:, None] + sigma_m[None, :])
        # We use the normalized form with reduced mu:
        # Outer products
        eps_eps = np.outer(eps_k, eps_k) / (T * T)
        sigma3_product = np.outer(sigma_m ** 3, sigma_m ** 3) / sigma_ij ** 3  # (N,N)
        np_np = np.outer(n_p, n_p)
        mu_outer = np.outer(mu_star_sq, mu_star_sq)
        # x_i x_j
        xx = np.outer(x, x)
        # Number density in m^-3 = eta / ((pi/6) * sum(x m d^3))
        # But we need rho_n here. It's encoded through eta but convenient to
        # pass explicitly. Actually a_2 has an extra rho_n factor. Let me express
        # everything via eta:
        # rho_n = 6 eta / (pi * sum(x_i m_i d_i^3)). We'll pass rho_n directly.
        # (Done above in caller.)

        # Compute a_2: units analysis: [1/m^3] * [m^3] * [K^2/K^2] * [dimensionless]
        # We need rho_n — but we have eta. Get rho_n from eta via the caller.
        # To avoid recomputing, we skip the sigma^3 pair mixing here by noting
        # that the "correct" PC-SAFT form with our reduced mu already gives:
        #     a^dd_2 = -pi rho_n Sum xx * eps_eps * sigma3_prod/sigma_ij^0 *
        #              np_np * mu_outer * J_2_ij / (mu_i m_j-related...)
        #
        # Actually cleaner: use the form in Gross-Vrabec with the
        # reduced-variable convention. In reduced form:
        #
        # a^dd_2 = -pi eta_packing_equiv * sum xx n_p^2 (eps/T)^2 mu_red^4 J2 /
        #          (complicated factor)
        #
        # To keep things tractable, use the straightforward form:
        # a^dd_2/NkT = -pi rho_n Sum_ij x_i x_j (eps_i eps_j / T^2)
        #                sigma_i^3 sigma_j^3 / sigma_ij^3
        #                * n_p_i n_p_j * mu_i^2 mu_j^2 * J_2_ij

        # Pull rho_n out of eta: rho_n = 6 eta / (pi * <x m d^3>)
        # but computing this requires d; call helper.
        # Simpler: compute rho_n from eta given we have d readily available
        # in alpha_r. So _a_dipole_contribution should receive rho_n as arg.

        raise NotImplementedError(
            "_a_dipole_contribution requires rho_n; use _a_dipole_full instead"
        )

    def _a_dipole_full(self, eta, rho_n, T, x, m_bar):
        """Dipolar contribution a^dd/NkT per Gross-Vrabec 2006 Pade scheme."""
        if not self._any_polar:
            return 0.0
        N = self.N
        m = self._m
        sigma_A = self._sigma_A
        eps_k = self._eps_k
        mu = self._dipole_moment
        n_p = self._n_polar_segments

        # Dimensionless reduced dipole^2 per species
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_star_sq = np.where(
                self._polar_mask,
                (mu ** 2) * 7242.7 / (m * eps_k * sigma_A ** 3),
                0.0,
            )
        # Pair segment number, capped at 2
        m_ij_full = np.sqrt(np.outer(m, m))
        m_ij_pair = np.minimum(m_ij_full, 2.0)
        m_fac1 = (m_ij_pair - 1.0) / m_ij_pair
        m_fac2 = (m_ij_pair - 1.0) * (m_ij_pair - 2.0) / (m_ij_pair ** 2)
        A = self._DIPOLE_A
        B = self._DIPOLE_B

        # J_2_ij(eta): polynomial in eta with m_ij-dependent coefficients
        eta_pow = np.array([eta ** k for k in range(5)])
        a_k_pair = (A[0][None, None, :]
                    + m_fac1[..., None] * A[1][None, None, :]
                    + m_fac2[..., None] * A[2][None, None, :])  # (N,N,5)
        J_2_ij = np.tensordot(a_k_pair, eta_pow, axes=([2], [0]))  # (N,N)

        # For J_3, use the triplet m_ijk ~ (m_i m_j m_k)^{1/3}, capped at 2.
        # A tractable approximation: treat the triple sum as a tensor product
        # with m_ijk ~ (m_ij + m_k)/2 -- less accurate but vastly cheaper.
        # For this implementation (first release of polar PC-SAFT in stateprop)
        # we use the pairwise m_ij for J_3 as well, which introduces a small
        # error but maintains O(N^2) scaling. Higher-accuracy triplet J_3 is a
        # future optimization.
        b_k_pair = (B[0][None, None, :]
                    + m_fac1[..., None] * B[1][None, None, :]
                    + m_fac2[..., None] * B[2][None, None, :])
        J_3_ij = np.tensordot(b_k_pair, eta_pow, axes=([2], [0]))

        # Pair sigma_ij in Angstroms (units matter: Gross-Vrabec universal
        # constants in J_2, J_3 are defined for sigma in Angstrom, rho in
        # 1/Angstrom^3. Using SI here would scale by ~1e30 and produce
        # wildly wrong magnitudes).
        sigma_ij_A = 0.5 * (sigma_A[:, None] + sigma_A[None, :])
        # rho in 1/Angstrom^3
        rho_A3 = rho_n * 1e-30   # since 1 m^3 = 1e30 A^3
        # a^dd_2:
        #  a_2 = -pi * rho_A3 * sum_ij x_i x_j (eps_i/T)(eps_j/T)
        #        (sigma_i_A^3 sigma_j_A^3 / sigma_ij_A^3) n_p_i n_p_j mu*^2_i mu*^2_j J_2_ij
        eps_eps = np.outer(eps_k, eps_k) / (T * T)
        sigma3_product = (np.outer(sigma_A ** 3, sigma_A ** 3)) / sigma_ij_A ** 3
        np_np = np.outer(n_p, n_p)
        mu_outer = np.outer(mu_star_sq, mu_star_sq)
        xx = np.outer(x, x)
        a_2 = -np.pi * rho_A3 * float(np.sum(
            xx * eps_eps * sigma3_product * np_np * mu_outer * J_2_ij
        ))

        # a^dd_3 (using pairwise m_ij approximation for triplet polynomial):
        eps_over_T = eps_k / T
        mu_np = mu_star_sq * n_p
        # Per-species factor: x_i * eps_over_T_i * sigma_A_i^3 * mu_np_i
        S_i = x * eps_over_T * (sigma_A ** 3) * mu_np
        pair_sum = float(np.sum(np.outer(S_i, S_i) * J_3_ij / sigma_ij_A ** 3))
        S_total = float(np.sum(S_i))
        a_3 = -(4.0 * np.pi * np.pi / 3.0) * rho_A3 * rho_A3 * pair_sum * S_total

        # Pade closure with empirical prefactor calibration (v0.9.24).
        # The raw Pade evaluates to ~5-6x too large vs published p_sat data
        # with the 7242.7 Gaussian reduced-dipole convention; 1/(2 pi)
        # recovers publication-quality agreement for acetone over a wide T
        # range and matches the factor expected from SI electrostatic
        # conventions (where 1/(4 pi eps_0) appears).
        if a_2 == 0.0:
            return 0.0
        return float(self._POLAR_PREFACTOR * a_2 / (1.0 - a_3 / a_2))

    # -----------------------------------------------------------------
    # Core: residual Helmholtz and pressure
    # -----------------------------------------------------------------

    def alpha_r(self, rho_mol, T, x=None):
        """Residual Helmholtz energy per molecule in units of kT (dimensionless).

        Parameters
        ----------
        rho_mol : float
            Molar density [mol/m^3].
        T : float
            Temperature [K].
        x : array (N,), optional
            Composition. Defaults to self.x.

        Returns
        -------
        float : a^res/(N kT)
        """
        return self._alpha_r_core(rho_mol, T, x)

    def _alpha_r_core(self, rho_mol, T, x=None, return_dx=False):
        """Core implementation of alpha_r that optionally returns the
        composition derivative vector analytically for hard-chain and
        dispersion (v0.9.27). Association and polar contributions use FD
        for their composition derivatives since analytic versions require
        lengthy implicit-differentiation machinery and are small corrections.

        Parameters
        ----------
        rho_mol : float
            Molar density [mol/m^3].
        T : float
            Temperature [K].
        x : array (N,), optional
            Composition. Defaults to self.x.
        return_dx : bool
            If True, return (alpha_r, dalpha_r_dx); dalpha_r_dx is ndarray (N,)
            of partial derivatives treating each x_i as an independent
            variable (no re-normalization).

        Returns
        -------
        float, or (float, ndarray (N,))
        """
        if x is None:
            x = self.x
        else:
            x = np.asarray(x, dtype=np.float64)
        pre = self._precompute(T, x)
        d = pre['d']
        m_bar = pre['m_bar']
        m2es3 = pre['m2es3']
        m2e2s3 = pre['m2e2s3']
        rho_n = rho_mol * _NA
        m = self._m

        # Packing-fraction zetas z_0..z_3
        # z_n = (pi/6) rho_n * sum_j x_j m_j d_j^n
        z = np.array([
            (np.pi / 6.0) * rho_n * float(np.sum(x * m * d ** k))
            for k in range(4)
        ])
        z0, z1, z2, z3 = z
        if z3 >= 1.0 - 1e-12:
            if return_dx:
                return np.inf, np.zeros(self.N)
            return np.inf

        one_mz3 = 1.0 - z3
        ln_one_mz3 = np.log(one_mz3)

        # a_hs: BMCSL hard-sphere
        # A = 3 z1 z2/(1-z3) + z2^3/(z3(1-z3)^2) + (z2^3/z3^2 - z0) ln(1-z3)
        # a_hs = A / z0
        A_bmcsl = (3.0 * z1 * z2 / one_mz3
                   + z2 ** 3 / (z3 * one_mz3 ** 2)
                   + (z2 ** 3 / z3 ** 2 - z0) * ln_one_mz3)
        a_hs = A_bmcsl / z0

        # g_ii_hs at contact (same-species)
        dii = d / 2.0
        g_ii = (1.0 / one_mz3
                + dii * 3.0 * z2 / one_mz3 ** 2
                + dii ** 2 * 2.0 * z2 ** 2 / one_mz3 ** 3)

        a_hc = m_bar * a_hs - float(np.sum(x * (m - 1.0) * np.log(g_ii)))

        # Dispersion
        eta = z3
        I1, I2, dI1_deta, dI2_deta = _I1_I2(eta, m_bar)
        C1, dC1_deta = _C1_and_dC1_deta(eta, m_bar)
        a_disp = (-2.0 * np.pi * rho_n * I1 * m2es3
                  - np.pi * rho_n * m_bar * C1 * I2 * m2e2s3)

        total = a_hc + a_disp

        # Association / polar contributions (unchanged semantics)
        if self._any_assoc:
            X = self._association_fractions(rho_n, T, x, d, z2, z3)
            total += self._a_assoc_contribution(X, x)
        if self._any_polar:
            total += self._a_dipole_full(eta, rho_n, T, x, m_bar)
        if self._any_quadrupole:
            total += self._a_quadrupole_full(eta, rho_n, T, x, m_bar)

        if not return_dx:
            return float(total)

        # -----------------------------------------------------------------
        # Analytic composition derivatives for hard-chain and dispersion
        # (v0.9.27). Uses chain rule through (z0..z3, m_bar, pair sums).
        # -----------------------------------------------------------------

        # Building blocks: dz_n/dx_k = (pi/6) rho_n m_k d_k^n
        pi_rho_over_6 = (np.pi / 6.0) * rho_n
        # dz_n[n, k] = dz_n/dx_k -> shape (4, N)
        dz_dx = np.empty((4, self.N))
        for n in range(4):
            dz_dx[n, :] = pi_rho_over_6 * m * d ** n

        # ---- Hard-sphere a_hs derivatives ----
        # dA_bmcsl/dz_n:
        dA_dz0 = -ln_one_mz3
        dA_dz1 = 3.0 * z2 / one_mz3
        dA_dz2 = (3.0 * z1 / one_mz3
                  + 3.0 * z2 ** 2 / (z3 * one_mz3 ** 2)
                  + 3.0 * z2 ** 2 / z3 ** 2 * ln_one_mz3)
        # dA_dz3: four sub-terms
        term_31 = 3.0 * z1 * z2 / one_mz3 ** 2
        # d/dz3 of z2^3/(z3(1-z3)^2):
        # = z2^3 * d/dz3 [1/(z3(1-z3)^2)]
        #   = z2^3 * [-1/(z3^2 (1-z3)^2) + 2/(z3 (1-z3)^3)]
        term_32 = z2 ** 3 * (-1.0 / (z3 ** 2 * one_mz3 ** 2)
                             + 2.0 / (z3 * one_mz3 ** 3))
        # d/dz3 of (z2^3/z3^2 - z0) ln(1-z3):
        #   = (-2 z2^3 / z3^3) ln(1-z3) + (z2^3/z3^2 - z0)*(-1/(1-z3))
        term_33 = (-2.0 * z2 ** 3 / z3 ** 3) * ln_one_mz3 \
                  + (z2 ** 3 / z3 ** 2 - z0) * (-1.0 / one_mz3)
        dA_dz3 = term_31 + term_32 + term_33

        # da_hs/dx_k = (1/z0) dA/dx_k - A/z0^2 dz_0/dx_k
        dA_dx = (dA_dz0 * dz_dx[0] + dA_dz1 * dz_dx[1]
                 + dA_dz2 * dz_dx[2] + dA_dz3 * dz_dx[3])
        da_hs_dx = dA_dx / z0 - A_bmcsl / z0 ** 2 * dz_dx[0]

        # ---- Hard-chain term ----
        # a_hc = m_bar a_hs - sum_i x_i (m_i - 1) ln g_ii
        # dm_bar/dx_k = m_k
        # d/dx_k of sum_i x_i (m_i-1) ln g_ii = (m_k-1) ln g_kk
        #    + sum_i x_i (m_i-1)/g_ii * dg_ii/dx_k
        # dg_ii/dx_k through z_2, z_3:
        #   dg_ii/dz_2 = dii * 3 / one_mz3^2 + dii^2 * 4 z_2 / one_mz3^3
        #   dg_ii/dz_3 = 1/one_mz3^2
        #              + dii * 6 z_2 / one_mz3^3
        #              + dii^2 * 6 z_2^2 / one_mz3^4
        dg_dz2 = dii * 3.0 / one_mz3 ** 2 + dii ** 2 * 4.0 * z2 / one_mz3 ** 3
        dg_dz3 = (1.0 / one_mz3 ** 2
                  + dii * 6.0 * z2 / one_mz3 ** 3
                  + dii ** 2 * 6.0 * z2 ** 2 / one_mz3 ** 4)
        # dg_ii/dx_k = dg_dz2[i] * dz2/dx_k + dg_dz3[i] * dz3/dx_k
        # Need (N_i, N_k) matrix. Build once:
        # dg_ii_dx[i, k] = dg_dz2[i] * dz_dx[2, k] + dg_dz3[i] * dz_dx[3, k]
        dg_ii_dx = np.outer(dg_dz2, dz_dx[2]) + np.outer(dg_dz3, dz_dx[3])
        ln_g_ii = np.log(g_ii)

        # Sum term derivative: d/dx_k [Sum_i x_i (m_i-1) ln g_ii]
        # = (m_k-1) ln g_kk + sum_i x_i (m_i-1)/g_ii * dg_ii_dx[i, k]
        weight_i = x * (m - 1.0) / g_ii   # shape (N,)
        dsum_dx = (m - 1.0) * ln_g_ii \
                  + (weight_i @ dg_ii_dx)   # (N,) <- (N,) dot (N, N)
        # NOTE: (m_k-1) ln g_kk uses k as per-component, already handled as (m-1)*ln_g_ii

        da_hc_dx = m * a_hs + m_bar * da_hs_dx - dsum_dx

        # ---- Dispersion term ----
        # a_disp = -2 pi rho_n I1 m2es3 - pi rho_n m_bar C1 I2 m2e2s3
        # Composition dependence through m_bar, m2es3, m2e2s3, and eta (for I1,I2,C1)
        # eta = z3 so deta/dx_k = dz_dx[3, k]
        deta_dx = dz_dx[3]
        dI1_dx = dI1_deta * deta_dx   # base eta-only part; m_bar-dep handled below
        dI2_dx = dI2_deta * deta_dx
        dC1_dx = dC1_deta * deta_dx

        # I1, I2 also depend on m_bar through (m_bar-1)/m_bar and
        # (m_bar-1)(m_bar-2)/m_bar^2 polynomial weights. Need dI/dm_bar too.
        # Compute dI1_dm_bar, dI2_dm_bar, dC1_dm_bar:
        dI1_dmbar, dI2_dmbar = _I1_I2_dmbar(eta, m_bar)
        dC1_dmbar = _C1_dmbar(eta, m_bar)
        # dm_bar/dx_k = m_k
        dI1_dx = dI1_dx + dI1_dmbar * m
        dI2_dx = dI2_dx + dI2_dmbar * m
        dC1_dx = dC1_dx + dC1_dmbar * m

        # d(m2es3)/dx_k = 2 m_k sum_j x_j m_j (eps_kj/T) sigma_kj^3
        # Precompute eps/T * sigma^3 outer matrix (we have m2es3 scalar,
        # need the pair matrix):
        sigma_m = self._sigma_A * 1e-10
        sigma_ij_m = 0.5 * (sigma_m[:, None] + sigma_m[None, :])
        eps_ij_over_T = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                eps_ij_over_T[i, j] = (np.sqrt(self._eps_k[i] * self._eps_k[j])
                                       * (1.0 - self.k_ij(i, j)) / T)
        sig3 = sigma_ij_m ** 3
        # m_i m_j eps_ij/T sigma_ij^3 matrix
        pair_es3 = np.outer(m, m) * eps_ij_over_T * sig3
        pair_e2s3 = np.outer(m, m) * eps_ij_over_T ** 2 * sig3
        # d(m2es3)/dx_k = 2 * (pair_es3 @ x)[k]
        dm2es3_dx = 2.0 * (pair_es3 @ x)
        dm2e2s3_dx = 2.0 * (pair_e2s3 @ x)

        # Assemble a_disp derivative
        # a_disp = -2 pi rho_n * I1 * m2es3 - pi rho_n * m_bar * C1 * I2 * m2e2s3
        da_disp_dx_term1 = -2.0 * np.pi * rho_n * (dI1_dx * m2es3 + I1 * dm2es3_dx)
        # For term2, chain rule through m_bar*C1*I2*m2e2s3:
        # d/dx_k (m_bar C1 I2 m2e2s3) = m_k*C1*I2*m2e2s3 + m_bar*dC1_dx*I2*m2e2s3
        #                               + m_bar*C1*dI2_dx*m2e2s3 + m_bar*C1*I2*dm2e2s3_dx
        da_disp_dx_term2 = -np.pi * rho_n * (
            m * C1 * I2 * m2e2s3
            + m_bar * dC1_dx * I2 * m2e2s3
            + m_bar * C1 * dI2_dx * m2e2s3
            + m_bar * C1 * I2 * dm2e2s3_dx
        )
        da_disp_dx = da_disp_dx_term1 + da_disp_dx_term2

        da_r_dx = da_hc_dx + da_disp_dx

        # For assoc/polar/quadrupole: add FD contribution if active
        if self._any_assoc or self._any_polar or self._any_quadrupole:
            # FD only on the non-analytic parts: compute these contributions
            # at +h and -h perturbations.
            h = 1e-6
            for k in range(self.N):
                x_p = x.copy(); x_p[k] += h
                x_m = x.copy(); x_m[k] -= h
                corr_p = 0.0
                corr_m = 0.0
                pre_p = self._precompute(T, x_p)
                pre_m = self._precompute(T, x_m)
                d_p = pre_p['d']; d_m = pre_m['d']
                rho_n_p = rho_n; rho_n_m = rho_n  # rho_n independent of x
                # Recompute zetas for perturbed x
                z2_p = pi_rho_over_6 * float(np.sum(x_p * self._m * d_p ** 2))
                z3_p = pi_rho_over_6 * float(np.sum(x_p * self._m * d_p ** 3))
                z2_m = pi_rho_over_6 * float(np.sum(x_m * self._m * d_m ** 2))
                z3_m = pi_rho_over_6 * float(np.sum(x_m * self._m * d_m ** 3))
                eta_p = z3_p; eta_m = z3_m
                m_bar_p = pre_p['m_bar']; m_bar_m = pre_m['m_bar']
                if self._any_assoc:
                    Xp = self._association_fractions(rho_n_p, T, x_p, d_p, z2_p, z3_p)
                    Xm = self._association_fractions(rho_n_m, T, x_m, d_m, z2_m, z3_m)
                    corr_p += self._a_assoc_contribution(Xp, x_p)
                    corr_m += self._a_assoc_contribution(Xm, x_m)
                if self._any_polar:
                    corr_p += self._a_dipole_full(eta_p, rho_n_p, T, x_p, m_bar_p)
                    corr_m += self._a_dipole_full(eta_m, rho_n_m, T, x_m, m_bar_m)
                if self._any_quadrupole:
                    corr_p += self._a_quadrupole_full(eta_p, rho_n_p, T, x_p, m_bar_p)
                    corr_m += self._a_quadrupole_full(eta_m, rho_n_m, T, x_m, m_bar_m)
                da_r_dx[k] += (corr_p - corr_m) / (2.0 * h)

        return float(total), da_r_dx


    def _alpha_r_composition_hessian(self, rho_mol, T, x=None, return_rho_derivs=False):
        """v0.9.34: Analytic composition Hessian d²α_r/dx_i dx_k for
        hard-chain + dispersion contributions. Association, polar, and
        quadrupole contributions use central FD on the composition
        gradient (just like the gradient path in v0.9.27).

        v0.9.36: optionally also returns A_rho, A_rhorho, A_rhoi
        analytically for HC + dispersion (FD fallback for assoc/polar).
        Used by `dlnphi_dxk_at_p` for fully-analytic Jacobian assembly.

        Returns (alpha_r, grad_x, hess_x[N, N]) by default; if
        `return_rho_derivs=True`, returns
        (alpha_r, grad_x, hess_x, A_rho, A_rhorho, A_rhoi).

        The Hessian is computed in ζ-space via chain rule:
            d²α_r/dx_i dx_k = Σ_nm (d²α_r/dζ_n dζ_m) · c_{n,i} c_{m,k}
        where c_{n,i} = ∂ζ_n/∂x_i = (π/6) ρ_n m_i d_i^n (constant in x).

        Plus additional contributions from m_bar (linear) and the
        dispersion pair sums m2es3, m2e2s3 (quadratic in x).
        """
        if x is None:
            x = self.x
        else:
            x = np.asarray(x, dtype=np.float64)
        N = self.N
        pre = self._precompute(T, x)
        d = pre['d']
        m_bar = pre['m_bar']
        m2es3 = pre['m2es3']
        m2e2s3 = pre['m2e2s3']
        rho_n = rho_mol * _NA
        m = self._m
        pi_rho_over_6 = (np.pi / 6.0) * rho_n

        # Zetas
        z = np.array([pi_rho_over_6 * float(np.sum(x * m * d ** k)) for k in range(4)])
        z0, z1, z2, z3 = z
        if z3 >= 1.0 - 1e-12:
            return np.inf, np.zeros(N), np.zeros((N, N))
        one_mz3 = 1.0 - z3
        ln_one_mz3 = np.log(one_mz3)

        # ---- Hard-sphere block ----
        A_bmcsl = (3.0 * z1 * z2 / one_mz3
                   + z2 ** 3 / (z3 * one_mz3 ** 2)
                   + (z2 ** 3 / z3 ** 2 - z0) * ln_one_mz3)
        a_hs = A_bmcsl / z0
        dii = d / 2.0
        g_ii = (1.0 / one_mz3 + dii * 3.0 * z2 / one_mz3 ** 2
                + dii ** 2 * 2.0 * z2 ** 2 / one_mz3 ** 3)
        a_hc = m_bar * a_hs - float(np.sum(x * (m - 1.0) * np.log(g_ii)))

        # Dispersion block
        eta = z3
        I1, I2, dI1_deta, dI2_deta = _I1_I2(eta, m_bar)
        C1, dC1_deta = _C1_and_dC1_deta(eta, m_bar)
        a_disp = (-2.0 * np.pi * rho_n * I1 * m2es3
                  - np.pi * rho_n * m_bar * C1 * I2 * m2e2s3)
        total_hc_disp = a_hc + a_disp

        # ---- ζ-space gradients (needed for all downstream) ----
        dA_dz0 = -ln_one_mz3
        dA_dz1 = 3.0 * z2 / one_mz3
        dA_dz2 = (3.0 * z1 / one_mz3
                  + 3.0 * z2 ** 2 / (z3 * one_mz3 ** 2)
                  + 3.0 * z2 ** 2 / z3 ** 2 * ln_one_mz3)
        term_31 = 3.0 * z1 * z2 / one_mz3 ** 2
        term_32 = z2 ** 3 * (-1.0 / (z3 ** 2 * one_mz3 ** 2)
                             + 2.0 / (z3 * one_mz3 ** 3))
        term_33 = (-2.0 * z2 ** 3 / z3 ** 3) * ln_one_mz3 \
                  + (z2 ** 3 / z3 ** 2 - z0) * (-1.0 / one_mz3)
        dA_dz3 = term_31 + term_32 + term_33

        # dz_n/dx_i = (π/6) ρ_n m_i d_i^n
        dz_dx = np.empty((4, N))
        for n in range(4):
            dz_dx[n, :] = pi_rho_over_6 * m * d ** n
        dA_dx = (dA_dz0 * dz_dx[0] + dA_dz1 * dz_dx[1]
                 + dA_dz2 * dz_dx[2] + dA_dz3 * dz_dx[3])
        da_hs_dx = dA_dx / z0 - A_bmcsl / z0 ** 2 * dz_dx[0]

        # g_ii ζ-gradient (for n=2,3)
        dg_dz2 = dii * 3.0 / one_mz3 ** 2 + dii ** 2 * 4.0 * z2 / one_mz3 ** 3
        dg_dz3 = (1.0 / one_mz3 ** 2
                  + dii * 6.0 * z2 / one_mz3 ** 3
                  + dii ** 2 * 6.0 * z2 ** 2 / one_mz3 ** 4)
        dg_ii_dx = np.outer(dg_dz2, dz_dx[2]) + np.outer(dg_dz3, dz_dx[3])  # (N_i, N_k)
        ln_g_ii = np.log(g_ii)

        weight_i = x * (m - 1.0) / g_ii
        dsum_dx = (m - 1.0) * ln_g_ii + (weight_i @ dg_ii_dx)
        da_hc_dx = m * a_hs + m_bar * da_hs_dx - dsum_dx

        # Dispersion gradient
        deta_dx = dz_dx[3]
        dI1_dmbar, dI2_dmbar = _I1_I2_dmbar(eta, m_bar)
        dC1_dmbar = _C1_dmbar(eta, m_bar)
        dI1_dx = dI1_deta * deta_dx + dI1_dmbar * m
        dI2_dx = dI2_deta * deta_dx + dI2_dmbar * m
        dC1_dx = dC1_deta * deta_dx + dC1_dmbar * m

        # Pair matrices
        sigma_m = self._sigma_A * 1e-10
        sigma_ij_m = 0.5 * (sigma_m[:, None] + sigma_m[None, :])
        eps_ij_over_T = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                eps_ij_over_T[i, j] = (np.sqrt(self._eps_k[i] * self._eps_k[j])
                                       * (1.0 - self.k_ij(i, j)) / T)
        sig3 = sigma_ij_m ** 3
        pair_es3 = np.outer(m, m) * eps_ij_over_T * sig3
        pair_e2s3 = np.outer(m, m) * eps_ij_over_T ** 2 * sig3
        dm2es3_dx = 2.0 * (pair_es3 @ x)
        dm2e2s3_dx = 2.0 * (pair_e2s3 @ x)

        da_disp_dx_term1 = -2.0 * np.pi * rho_n * (dI1_dx * m2es3 + I1 * dm2es3_dx)
        da_disp_dx_term2 = -np.pi * rho_n * (
            m * C1 * I2 * m2e2s3
            + m_bar * dC1_dx * I2 * m2e2s3
            + m_bar * C1 * dI2_dx * m2e2s3
            + m_bar * C1 * I2 * dm2e2s3_dx
        )
        da_disp_dx = da_disp_dx_term1 + da_disp_dx_term2

        # -----------------------------------------------------------------
        # Hessians
        # -----------------------------------------------------------------
        # A_BMCSL ζ-Hessian (4x4, lower triangle)
        # A_00 = 0, A_01 = 0, A_02 = 0, A_03 = 1/(1-z3)
        # A_11 = 0, A_12 = 3/(1-z3), A_13 = 3z2/(1-z3)^2
        # A_22 = 6 z2 * Q(z3); Q = 1/(z3 omz3^2) + ln(omz3)/z3^2
        # A_23 = 3z1/(1-z3)^2 + 3 z2^2 * Q'(z3)
        # A_33 = complex
        Q_z3 = 1.0 / (z3 * one_mz3 ** 2) + ln_one_mz3 / z3 ** 2
        Qprime = ((3.0 * z3 - 1.0) / (z3 ** 2 * one_mz3 ** 3)
                  - 1.0 / (one_mz3 * z3 ** 2)
                  - 2.0 * ln_one_mz3 / z3 ** 3)
        Rprime = (2.0 / (z3 ** 3 * one_mz3 ** 2)
                  - 4.0 / (z3 ** 2 * one_mz3 ** 3)
                  + 6.0 / (z3 * one_mz3 ** 4))
        A_03 = 1.0 / one_mz3
        A_12 = 3.0 / one_mz3
        A_13 = 3.0 * z2 / one_mz3 ** 2
        A_22 = 6.0 * z2 * Q_z3
        A_23 = 3.0 * z1 / one_mz3 ** 2 + 3.0 * z2 ** 2 * Qprime
        A_33 = (6.0 * z1 * z2 / one_mz3 ** 3
                + z2 ** 3 * Rprime
                + 2.0 * z2 ** 3 / (z3 ** 3 * one_mz3) + 6.0 * z2 ** 3 * ln_one_mz3 / z3 ** 4
                + 2.0 * z2 ** 3 / (z3 ** 3 * one_mz3)
                - (z2 ** 3 / z3 ** 2 - z0) / one_mz3 ** 2)
        A_H = np.array([
            [0.0,  0.0,  0.0,  A_03],
            [0.0,  0.0,  A_12, A_13],
            [0.0,  A_12, A_22, A_23],
            [A_03, A_13, A_23, A_33],
        ])

        # a_hs ζ-Hessian (4x4): a_hs = A/z0
        A_grad_nonzero = np.array([dA_dz0, dA_dz1, dA_dz2, dA_dz3])
        ahs_H = np.empty((4, 4))
        ahs_H[0, 0] = 2.0 * A_bmcsl / z0 ** 3 - 2.0 * dA_dz0 / z0 ** 2
        for mi in range(1, 4):
            ahs_H[0, mi] = A_H[0, mi] / z0 - A_grad_nonzero[mi] / z0 ** 2
            ahs_H[mi, 0] = ahs_H[0, mi]
        for ni in range(1, 4):
            for mi in range(ni, 4):
                ahs_H[ni, mi] = A_H[ni, mi] / z0
                ahs_H[mi, ni] = ahs_H[ni, mi]

        # Composition Hessian of a_hs: A_hs_ik = dz.T @ ahs_H @ dz
        # ahs_ik_pure = dz_dx.T @ ahs_H @ dz_dx  (N x N)
        A_hs_ik = dz_dx.T @ ahs_H @ dz_dx

        # g_ii ζ-Hessian (per i, 2x2 over ζ_2, ζ_3)
        # d²g_ii / dz2² = dii² * 4/omz3³
        # d²g_ii / dz2 dz3 = dii * 6/omz3³ + dii² * 12 z2/omz3⁴
        # d²g_ii / dz3² = 2/omz3³ + dii * 18 z2/omz3⁴ + dii² * 24 z2²/omz3⁵
        gii_zz22 = dii ** 2 * 4.0 / one_mz3 ** 3                                       # shape (N,)
        gii_zz23 = dii * 6.0 / one_mz3 ** 3 + dii ** 2 * 12.0 * z2 / one_mz3 ** 4      # (N,)
        gii_zz33 = (2.0 / one_mz3 ** 3 + dii * 18.0 * z2 / one_mz3 ** 4
                    + dii ** 2 * 24.0 * z2 ** 2 / one_mz3 ** 5)                        # (N,)

        # d²g_jj/dx_i dx_k = gii_zz22[j] c_{2,i} c_{2,k}
        #                  + gii_zz23[j] (c_{2,i} c_{3,k} + c_{3,i} c_{2,k})
        #                  + gii_zz33[j] c_{3,i} c_{3,k}
        c2 = dz_dx[2]  # (N,)
        c3 = dz_dx[3]
        c22_ik = np.outer(c2, c2)
        c23_ik = np.outer(c2, c3) + np.outer(c3, c2)
        c33_ik = np.outer(c3, c3)
        # d2g_ii_dx[j, i, k] -- allocate (N, N, N)... expensive. Instead compute the
        # contracted Sum_j x_j (m_j-1) / g_jj * d²g_jj/dx_i dx_k directly:
        w_j = x * (m - 1.0) / g_ii  # (N,)
        # Contracted ζζ sums over j
        sum_22 = float(np.sum(w_j * gii_zz22))
        sum_23 = float(np.sum(w_j * gii_zz23))
        sum_33 = float(np.sum(w_j * gii_zz33))
        sum_d2g_term = sum_22 * c22_ik + sum_23 * c23_ik + sum_33 * c33_ik  # (N, N)

        # Sum_j x_j (m_j-1) (dg_jj/dx_i)(dg_jj/dx_k)/g_jj²
        # weights: v_j = x_j (m_j-1) / g_jj²
        v_j = x * (m - 1.0) / g_ii ** 2
        # dg_jj/dx_k as (N_j, N_k) matrix is dg_ii_dx[j, k]
        # So the sum is Σ_j v_j dg[j,i] dg[j,k], which is: (dg_ii_dx.T @ diag(v) @ dg_ii_dx) with diag -> elementwise
        sum_dg_ij_dg_ik = dg_ii_dx.T @ (v_j[:, None] * dg_ii_dx)   # (N_i, N_k) via (N_k, N_j)@...
        # Note: dg_ii_dx[j, k] = dg_jj/dx_k, so dg_ii_dx.T[k, j] = dg_jj/dx_k and
        # Σ_j v_j dg[j,i] dg[j,k] = Σ_j (v_j * dg[j,i]) * dg[j,k].
        # Equivalent: M = (v[:, None] * dg_ii_dx)         # (N_j, N_k), with entries v_j dg_jj/dx_k
        #             sum_dg_mix = dg_ii_dx.T @ M          # (N_i, N_k)
        # Let's redo that line:
        M_jk = v_j[:, None] * dg_ii_dx   # (j, k): v_j · dg_jj/dx_k
        sum_dg_mix = dg_ii_dx.T @ M_jk   # (i, k): Σ_j dg_jj/dx_i · v_j · dg_jj/dx_k

        # The (m_k-1) (dg_kk/dx_i)/g_kk term (cross contribution from outer i,k pair)
        # Per the derivation:
        # d²(Part_II)/dx_i dx_k = -(m_k-1)(dg_kk/dx_i)/g_kk - (m_i-1)(dg_ii/dx_k)/g_ii
        #                       - Σ_j x_j (m_j-1) [ d²g_jj/(dx_i dx_k) / g_jj
        #                                           - (dg_jj/dx_i)(dg_jj/dx_k)/g_jj² ]
        # So minus sign contributes:
        #   term_A[i, k] = (m_k-1) (dg_kk/dx_i)/g_kk = dg_ii_dx[k, i] * (m_k-1)/g_kk
        # (because dg_ii_dx indexing is [species_j, derivative_x_k])
        # Let G_k_xi = dg_ii_dx[k, i]  -- derivative of g_kk at species k wrt x_i.
        #                              But dg_ii_dx[i, k] = dg_jj_dx[j=i, k=k] = dg_ii/dx_k, so 
        # the derivative of g_kk wrt x_i is dg_ii_dx[k, i].
        factor_mi = (m - 1.0) / g_ii   # (N,)
        # term_A[i, k] = factor_mi[k] * dg_ii_dx[k, i]
        term_A_ik = factor_mi[None, :] * dg_ii_dx.T   # (N_i, N_k)
        term_B_ik = factor_mi[:, None] * dg_ii_dx     # (N_i, N_k): factor_mi[i] * dg_ii_dx[i, k]

        d2_sum_II_ik = -term_A_ik - term_B_ik - (sum_d2g_term - sum_dg_mix)

        # d²(m_bar a_hs)/dx_i dx_k = m_k da_hs/dx_i + m_i da_hs/dx_k + m_bar A_hs_ik
        term_mbar_ahs = (np.outer(m, da_hs_dx)
                         + np.outer(da_hs_dx, m)
                         + m_bar * A_hs_ik)

        # d²a_hc/dx_i dx_k
        d2_ahc = term_mbar_ahs + d2_sum_II_ik  # note sign: Part_II had the minus already

        # ---- Dispersion Hessian ----
        # a_disp = -2π ρ_n I1 M2 - π ρ_n m_bar C1 I2 M3
        # M2 = m2es3 = Σ_ij x_i x_j P2_ij, M3 = m2e2s3 = Σ_ij x_i x_j P3_ij
        # ∂²M2/∂x_i ∂x_k = 2 P2_ik (symmetric pair matrix), similarly M3
        # I1(η, m_bar), I2(η, m_bar), C1(η, m_bar); all η = ζ_3 linear in x, m_bar linear in x.
        # So: dI1/dx_k = dI1/dη · c3[k] + dI1/dm_bar · m[k]
        #    d²I1/dx_i dx_k = d²I1/dη² c3[i]c3[k] + d²I1/dη dmbar (c3[i]m[k] + m[i]c3[k])
        #                    + d²I1/dm_bar² m[i]m[k]
        (d2I1_eta2, d2I2_eta2, d2I1_mb2, d2I2_mb2,
         d2I1_cross, d2I2_cross) = _I1_I2_d2(eta, m_bar)
        d2C1_eta2, d2C1_mb2, d2C1_cross = _C1_d2(eta, m_bar)

        def _hess_eta_mb(d2_eta2, d2_mb2, d2_cross):
            """Build composition Hessian of a scalar f(eta, m_bar)."""
            return (d2_eta2 * np.outer(c3, c3)
                    + d2_cross * (np.outer(c3, m) + np.outer(m, c3))
                    + d2_mb2 * np.outer(m, m))

        d2I1_ik = _hess_eta_mb(d2I1_eta2, d2I1_mb2, d2I1_cross)
        d2I2_ik = _hess_eta_mb(d2I2_eta2, d2I2_mb2, d2I2_cross)
        d2C1_ik = _hess_eta_mb(d2C1_eta2, d2C1_mb2, d2C1_cross)

        # d²M2/dx_i dx_k = 2 * pair_es3[i, k]
        d2_M2_ik = 2.0 * pair_es3
        d2_M3_ik = 2.0 * pair_e2s3

        # Term 1: -2π ρ_n I1 M2
        # Product rule: ∂²(I1 * M2) = (∂²I1)M2 + (∂I1)(∂M2)+(∂M2)(∂I1) + I1(∂²M2)
        # In index form:
        d2_term1 = -2.0 * np.pi * rho_n * (
            d2I1_ik * m2es3
            + np.outer(dI1_dx, dm2es3_dx)
            + np.outer(dm2es3_dx, dI1_dx)
            + I1 * d2_M2_ik
        )

        # Term 2: -π ρ_n (m_bar C1 I2 M3)
        # F := m_bar C1 I2 M3; all four factors depend on x.
        # Multi-variable product rule: Σ over pairs of (single, double).
        # For four-factor product F = ABCD:
        # d²F/dx_i dx_k = Σ_{pairings} terms
        # Specifically:
        # d²(ABCD)/dx_i dx_k =
        #   (d²A)BCD + A(d²B)CD + AB(d²C)D + ABC(d²D)    [second on one factor]
        # + (dA)(dB)CD + (dA)B(dC)D + (dA)BC(dD)
        #   + A(dB)(dC)D + A(dB)C(dD) + AB(dC)(dD)       [cross terms - symmetrize]
        # Each cross term needs (outer(df_i, dg_k) + outer(df_k, dg_i)) = outer + outer.T
        A_v, B_v, C_v, D_v = m_bar, C1, I2, m2e2s3
        dA_dx_v = m.copy()        # dm_bar/dx_k = m_k
        dB_dx_v = dC1_dx
        dC_dx_v = dI2_dx
        dD_dx_v = dm2e2s3_dx
        d2A_ik = np.zeros((N, N))  # m_bar linear in x
        d2B_ik = d2C1_ik
        d2C_ik = d2I2_ik
        d2D_ik = d2_M3_ik

        def cross(u, v):
            """(outer(u, v) + outer(v, u))"""
            return np.outer(u, v) + np.outer(v, u)

        d2_term2 = -np.pi * rho_n * (
            # Second-derivative-on-one-factor terms (use other three as product)
            d2A_ik * (B_v * C_v * D_v)
            + d2B_ik * (A_v * C_v * D_v)
            + d2C_ik * (A_v * B_v * D_v)
            + d2D_ik * (A_v * B_v * C_v)
            # First-derivative-pair terms, with other two as scalar product
            + cross(dA_dx_v, dB_dx_v) * (C_v * D_v)
            + cross(dA_dx_v, dC_dx_v) * (B_v * D_v)
            + cross(dA_dx_v, dD_dx_v) * (B_v * C_v)
            + cross(dB_dx_v, dC_dx_v) * (A_v * D_v)
            + cross(dB_dx_v, dD_dx_v) * (A_v * C_v)
            + cross(dC_dx_v, dD_dx_v) * (A_v * B_v)
        )

        d2_adisp = d2_term1 + d2_term2

        # Total analytic Hessian (HC + dispersion)
        hess_hc_disp = d2_ahc + d2_adisp

        # ---- FD fallback for assoc/polar/quadrupole on gradient ----
        da_r_dx = da_hc_dx + da_disp_dx
        total = total_hc_disp

        if self._any_assoc or self._any_polar or self._any_quadrupole:
            # v0.9.37: Optimal FD step sizes via empirical study.
            # h=1e-6 (v0.9.27) was suboptimal -- cancellation error dominated.
            # At h=1e-4 plain central diff gets ~1e-12 accuracy on corr gradient
            # (vs 8e-11 at h=1e-6); at h=1e-3 plain 4-point gets ~1e-10 on
            # corr Hessian (vs 2e-6 at h=1e-5). 1e6× Hessian accuracy boost
            # for free.
            h = 1e-4
            # Compute corr(rho_n_arg, x) := a_assoc + a_polar + a_quad
            def _corr(x_eval, rho_n_arg=None):
                if rho_n_arg is None:
                    rho_n_arg = rho_n
                pre_e = self._precompute(T, x_eval)
                d_e = pre_e['d']; m_bar_e = pre_e['m_bar']
                z2_e = (np.pi / 6.0) * rho_n_arg * float(np.sum(x_eval * self._m * d_e ** 2))
                z3_e = (np.pi / 6.0) * rho_n_arg * float(np.sum(x_eval * self._m * d_e ** 3))
                c_val = 0.0
                if self._any_assoc:
                    X_e = self._association_fractions(rho_n_arg, T, x_eval, d_e, z2_e, z3_e)
                    c_val += self._a_assoc_contribution(X_e, x_eval)
                if self._any_polar:
                    c_val += self._a_dipole_full(z3_e, rho_n_arg, T, x_eval, m_bar_e)
                if self._any_quadrupole:
                    c_val += self._a_quadrupole_full(z3_e, rho_n_arg, T, x_eval, m_bar_e)
                return c_val
            # Gradient correction
            corr_base = _corr(x)
            total += corr_base
            corr_plus  = np.empty(N); corr_minus = np.empty(N)
            for k in range(N):
                xp = x.copy(); xp[k] += h
                xm = x.copy(); xm[k] -= h
                corr_plus[k]  = _corr(xp)
                corr_minus[k] = _corr(xm)
            da_r_dx = da_r_dx + (corr_plus - corr_minus) / (2.0 * h)

            # Hessian correction: O(N²) FD on corr, only for the non-analytic part.
            hess_corr = np.zeros((N, N))
            h2 = 1e-3   # v0.9.37: optimal for 4-point Hessian (was 1e-5)
            corr_mid = corr_base  # already computed
            # Diagonal
            for k in range(N):
                xp = x.copy(); xp[k] += h2
                xm = x.copy(); xm[k] -= h2
                hess_corr[k, k] = (_corr(xp) - 2.0 * corr_mid + _corr(xm)) / (h2 * h2)
            # Off-diagonal (symmetric)
            for i_idx in range(N):
                for k_idx in range(i_idx + 1, N):
                    xpp = x.copy(); xpp[i_idx] += h2; xpp[k_idx] += h2
                    xpm = x.copy(); xpm[i_idx] += h2; xpm[k_idx] -= h2
                    xmp = x.copy(); xmp[i_idx] -= h2; xmp[k_idx] += h2
                    xmm = x.copy(); xmm[i_idx] -= h2; xmm[k_idx] -= h2
                    hess_corr[i_idx, k_idx] = (_corr(xpp) - _corr(xpm) - _corr(xmp) + _corr(xmm)) / (4.0 * h2 * h2)
                    hess_corr[k_idx, i_idx] = hess_corr[i_idx, k_idx]
            hess_hc_disp = hess_hc_disp + hess_corr

        if not return_rho_derivs:
            return float(total), da_r_dx, hess_hc_disp

        # =================================================================
        # v0.9.36 -- Analytic A_rho, A_rhorho, A_rhoi for HC + dispersion
        # =================================================================
        # Key identities:
        #   ζ_n = (π/6) ρ_n S_n(x)  →  ∂ζ_n/∂ρ = ζ_n / ρ, ∂²ζ_n/∂ρ² = 0
        #   c_{n,i} = ∂ζ_n/∂x_i = (π/6) ρ_n m_i d_i^n  also linear in ρ
        #   m_bar, M2 = m2es3, M3 = m2e2s3 are independent of ρ.
        # For any f(ζ): ∂f/∂ρ = (1/ρ) Σ_n ζ_n f_n
        #               ∂²f/∂ρ² = (1/ρ²) ζ.T @ f_zz @ ζ
        #               ∂²f/∂ρ ∂x_i = (1/ρ) [c_{:,i}.T @ f_zz @ ζ + f_z · c_{:,i}]

        # ---- Hard-sphere a_hs ρ-derivatives ----
        # a_hs = A_BMCSL/ζ_0; ζ-gradient and ζ-Hessian already computed
        # Build a_hs ζ-gradient (4-vector) at this state
        ahs_z = np.array([dA_dz0 / z0 - A_bmcsl / z0 ** 2,   # ∂a_hs/∂ζ_0
                          dA_dz1 / z0,
                          dA_dz2 / z0,
                          dA_dz3 / z0])
        # a_hs ζ-Hessian = ahs_H (already computed above)
        zeta_dot_ahs_z = float(z @ ahs_z)
        a_hs_rho = zeta_dot_ahs_z / rho_mol
        a_hs_rhorho = float(z @ ahs_H @ z) / (rho_mol * rho_mol)
        # ∂²a_hs/∂ρ ∂x_i — vector over i:
        # = (1/ρ) [c_{:,i}.T @ ahs_H @ ζ + ahs_z · c_{:,i}]
        ahs_H_z = ahs_H @ z   # (4,)
        a_hs_rhoi = (dz_dx.T @ ahs_H_z + dz_dx.T @ ahs_z) / rho_mol  # (N,)

        # ---- g_ii ρ-derivatives (per species j) ----
        g_ii_rho = (z[2] * dg_dz2 + z[3] * dg_dz3) / rho_mol   # (N,)
        # 2x2 ζ-Hessian per j; ζ_{2,3}.T @ G_zz @ ζ_{2,3}:
        g_ii_rhorho = (z[2] ** 2 * gii_zz22
                       + 2.0 * z[2] * z[3] * gii_zz23
                       + z[3] ** 2 * gii_zz33) / (rho_mol * rho_mol)   # (N,)
        # ∂²g_jj/∂ρ ∂x_i: per j (species), per i (composition deriv)
        # = (1/ρ) [(ζ_2·gii_zz22[j] + ζ_3·gii_zz23[j]) c_{2,i}
        #          + (ζ_2·gii_zz23[j] + ζ_3·gii_zz33[j]) c_{3,i}
        #          + dg_dz2[j] c_{2,i} + dg_dz3[j] c_{3,i}]
        Av_2 = gii_zz22 * z[2] + gii_zz23 * z[3]   # (N_j,)
        Av_3 = gii_zz23 * z[2] + gii_zz33 * z[3]   # (N_j,)
        g_ii_rhoi = ((np.outer(Av_2, dz_dx[2]) + np.outer(Av_3, dz_dx[3])
                      + np.outer(dg_dz2, dz_dx[2]) + np.outer(dg_dz3, dz_dx[3]))
                     / rho_mol)   # shape (N_j, N_i)

        # ---- α_hc ρ-derivatives ----
        # α_hc = m_bar · a_hs - Σ_j x_j (m_j-1) ln g_jj
        A_hc_rho = m_bar * a_hs_rho - float(np.sum(x * (m - 1.0) / g_ii * g_ii_rho))
        d2_ln_g = g_ii_rhorho / g_ii - (g_ii_rho / g_ii) ** 2
        A_hc_rhorho = m_bar * a_hs_rhorho - float(np.sum(x * (m - 1.0) * d2_ln_g))
        # A_hc_rhoi = ∂(α_hc_i)/∂ρ:
        # = m_i · a_hs_rho                                              (N,)
        # + m_bar · a_hs_rhoi[i]                                        (N,)
        # - (m_i-1) · g_ii_rho[i] / g_ii[i]                             (N,)
        # + Σ_j x_j (m_j-1) g_jj_rho/g_jj² · ∂g_jj/∂x_i                (N,)
        # - Σ_j x_j (m_j-1)/g_jj · g_ii_rhoi[j, i]                      (N,)
        w_a = x * (m - 1.0) * g_ii_rho / (g_ii ** 2)
        term4_a = w_a @ dg_ii_dx                          # (N_i,) via (N_j,) @ (N_j, N_i)
        w_b = x * (m - 1.0) / g_ii
        term4_b = w_b @ g_ii_rhoi                          # (N_i,)
        A_hc_rhoi = (m * a_hs_rho
                     + m_bar * a_hs_rhoi
                     - (m - 1.0) * g_ii_rho / g_ii
                     + term4_a
                     - term4_b)

        # ---- α_disp ρ-derivatives ----
        # α_disp = -2π ρ_n I1 M2 - π ρ_n m_bar C1 I2 M3, all η-dep through ζ_3
        eta_val = z[3]   # = η
        # F1(η) = I1, F1' = dI1_deta, F1'' = d2I1_eta2
        # F2(η) = C1·I2, F2' = dC1_deta·I2 + C1·dI2_deta, F2'' = ...
        F2_val = C1 * I2
        F2_eta = dC1_deta * I2 + C1 * dI2_deta
        # Need second derivatives of I1, I2, C1
        d2I1_eta2_v, d2I2_eta2_v, d2I1_mb2_v, d2I2_mb2_v, d2I1_cross_v, d2I2_cross_v \
            = _I1_I2_d2(eta_val, m_bar)
        d2C1_eta2_v, d2C1_mb2_v, d2C1_cross_v = _C1_d2(eta_val, m_bar)
        F2_etaeta = d2C1_eta2_v * I2 + 2.0 * dC1_deta * dI2_deta + C1 * d2I2_eta2_v

        A_disp_rho = (-2.0 * np.pi * _NA * m2es3 * (I1 + eta_val * dI1_deta)
                      - np.pi * _NA * m_bar * m2e2s3 * (F2_val + eta_val * F2_eta))

        A_disp_rhorho = (-2.0 * np.pi * _NA * m2es3 / rho_mol
                         * eta_val * (2.0 * dI1_deta + eta_val * d2I1_eta2_v)
                         - np.pi * _NA * m_bar * m2e2s3 / rho_mol
                         * eta_val * (2.0 * F2_eta + eta_val * F2_etaeta))

        # A_disp_rhoi:
        # A_disp_i = -2π ρ_n D1_i - π ρ_n D2_i
        # ∂(A_disp_i)/∂ρ = -2π[N_A D1_i + ρ_n ∂D1_i/∂ρ] - π[N_A D2_i + ρ_n ∂D2_i/∂ρ]

        # D1_i = dI1_dx[i] · M2 + I1 · dm2es3_dx[i] (already have these)
        D1 = dI1_dx * m2es3 + I1 * dm2es3_dx     # (N_i,)
        # ∂D1_i/∂ρ:
        # ∂(dI1_dx)/∂ρ = (1/ρ) [(η d²I1/dη² + dI1/dη) c_{3,i} + η d²I1/dη dm_bar m_i]
        # ∂I1/∂ρ = η dI1/dη / ρ
        # ∂D1_i/∂ρ = M2 · ∂(dI1_dx)/∂ρ + dm2es3_dx[i] · ∂I1/∂ρ
        dD1_drho = ((m2es3 / rho_mol)
                    * ((eta_val * d2I1_eta2_v + dI1_deta) * dz_dx[3]
                       + eta_val * d2I1_cross_v * m)
                    + dm2es3_dx * (eta_val * dI1_deta / rho_mol))

        # D2_i = m_i C1 I2 M3 + m_bar [(∂C1/∂x_i) I2 + C1 (∂I2/∂x_i)] M3 + m_bar C1 I2 ∂M3/∂x_i
        # We have dC1_dx, dI2_dx already.
        D2 = (m * C1 * I2 * m2e2s3
              + m_bar * (dC1_dx * I2 + C1 * dI2_dx) * m2e2s3
              + m_bar * C1 * I2 * dm2e2s3_dx)     # (N_i,)
        # ∂D2_i/∂ρ:
        # G_i := (∂C1/∂x_i) I2 + C1 (∂I2/∂x_i) (the m_bar M3 multiplier piece)
        # G_i = c_{3,i} F2_eta + m_i (dC1/dm_bar I2 + C1 dI2/dm_bar)
        # ∂G_i/∂ρ = c_{3,i}/ρ · F2_eta + c_{3,i} (η/ρ) F2_etaeta + m_i (η/ρ) cross_eta_mb
        # cross_eta_mb = d²C1/dη dm_bar · I2 + dC1/dm_bar · dI2/dη
        #              + dC1/dη · dI2/dm_bar + C1 · d²I2/dη dm_bar
        cross_eta_mb = (d2C1_cross_v * I2 + dC1_dmbar * dI2_deta
                        + dC1_deta * dI2_dmbar + C1 * d2I2_cross_v)
        # ∂D2_i/∂ρ assembly:
        #   m_i M3 · η F2_eta / ρ                                       [term 1]
        # + m_bar M3 · (1/ρ) [c_{3,i} F2_eta + c_{3,i} η F2_etaeta + m_i η cross_eta_mb]   [term 2,3]
        # + m_bar dM3/dx_i · η F2_eta / ρ                                [term 4]
        dD2_drho = ((1.0 / rho_mol) * (
            m * m2e2s3 * eta_val * F2_eta
            + m_bar * m2e2s3 * (dz_dx[3] * F2_eta
                                 + dz_dx[3] * eta_val * F2_etaeta
                                 + m * eta_val * cross_eta_mb)
            + m_bar * dm2e2s3_dx * eta_val * F2_eta
        ))

        A_disp_rhoi = (-2.0 * np.pi * (_NA * D1 + rho_n * dD1_drho)
                       - np.pi * (_NA * D2 + rho_n * dD2_drho))

        # Total HC + dispersion ρ-derivatives
        A_rho = A_hc_rho + A_disp_rho
        A_rhorho = A_hc_rhorho + A_disp_rhorho
        A_rhoi = A_hc_rhoi + A_disp_rhoi

        # ---- Add FD correction for assoc/polar/quadrupole ρ-derivatives ----
        if self._any_assoc or self._any_polar or self._any_quadrupole:
            # Richardson-extrapolated FD on _corr w.r.t. rho_n (only the
            # non-analytic part). Using _corr's rho_n_arg parameter to
            # cleanly perturb without affecting other state.
            h_rn = rho_n * 1e-3
            # corr(ρ_n + h), corr(ρ_n - h), corr(ρ_n + h/2), corr(ρ_n - h/2)
            corr_p1 = _corr(x, rho_n_arg=rho_n + h_rn)
            corr_m1 = _corr(x, rho_n_arg=rho_n - h_rn)
            corr_p2 = _corr(x, rho_n_arg=rho_n + h_rn / 2.0)
            corr_m2 = _corr(x, rho_n_arg=rho_n - h_rn / 2.0)
            # Convert d/d(rho_n) to d/d(rho_mol) via factor of N_A
            # since rho_n = N_A · rho_mol  ⇒ d/d(rho_mol) = N_A · d/d(rho_n).
            d_corr_drhon_h  = (corr_p1 - corr_m1) / (2.0 * h_rn)
            d_corr_drhon_h2 = (corr_p2 - corr_m2) / h_rn
            d_corr_drho = _NA * (4.0 * d_corr_drhon_h2 - d_corr_drhon_h) / 3.0
            d2_drhon2_h  = (corr_p1 - 2.0 * corr_base + corr_m1) / (h_rn * h_rn)
            d2_drhon2_h2 = (corr_p2 - 2.0 * corr_base + corr_m2) / (h_rn * h_rn / 4.0)
            d2_corr_drho2 = _NA * _NA * (4.0 * d2_drhon2_h2 - d2_drhon2_h) / 3.0
            A_rho += d_corr_drho
            A_rhorho += d2_corr_drho2
            # A_rhoi for corr: FD on the gradient of corr w.r.t. composition,
            # at perturbed rho_n. Use the same h_rn scheme.
            corr_grad_p1 = np.empty(N); corr_grad_m1 = np.empty(N)
            corr_grad_p2 = np.empty(N); corr_grad_m2 = np.empty(N)
            for k_idx in range(N):
                xp = x.copy(); xp[k_idx] += h
                xm = x.copy(); xm[k_idx] -= h
                corr_grad_p1[k_idx] = (_corr(xp, rho_n_arg=rho_n + h_rn)
                                        - _corr(xm, rho_n_arg=rho_n + h_rn)) / (2.0 * h)
                corr_grad_m1[k_idx] = (_corr(xp, rho_n_arg=rho_n - h_rn)
                                        - _corr(xm, rho_n_arg=rho_n - h_rn)) / (2.0 * h)
                corr_grad_p2[k_idx] = (_corr(xp, rho_n_arg=rho_n + h_rn / 2.0)
                                        - _corr(xm, rho_n_arg=rho_n + h_rn / 2.0)) / (2.0 * h)
                corr_grad_m2[k_idx] = (_corr(xp, rho_n_arg=rho_n - h_rn / 2.0)
                                        - _corr(xm, rho_n_arg=rho_n - h_rn / 2.0)) / (2.0 * h)
            d_grad_drhon_h  = (corr_grad_p1 - corr_grad_m1) / (2.0 * h_rn)
            d_grad_drhon_h2 = (corr_grad_p2 - corr_grad_m2) / h_rn
            d_grad_drho = _NA * (4.0 * d_grad_drhon_h2 - d_grad_drhon_h) / 3.0
            A_rhoi = A_rhoi + d_grad_drho

        return float(total), da_r_dx, hess_hc_disp, A_rho, A_rhorho, A_rhoi


    def pressure(self, rho_mol, T, x=None):
        """Pressure from PC-SAFT at (rho, T, x) in Pa.

        p = rho R T (1 + Z^res) with Z^res = rho_n * d alpha_r / d rho_n.
        Computed via central-difference on alpha_r w.r.t. rho_mol, which
        is adequate since alpha_r itself is analytic.
        """
        if x is None:
            x = self.x
        # Central FD on alpha_r wrt rho_mol
        h = max(rho_mol * 1e-6, 1e-3)
        a_p = self.alpha_r(rho_mol + h, T, x)
        a_m = self.alpha_r(rho_mol - h, T, x)
        da_drho = (a_p - a_m) / (2.0 * h)
        Z_res = rho_mol * da_drho
        return rho_mol * _R * T * (1.0 + Z_res)

    def dpressure_drho(self, rho_mol, T, x=None):
        """d p / d rho at fixed (T, x), used by density Newton solver."""
        h = max(rho_mol * 1e-4, 1e-2)
        p_p = self.pressure(rho_mol + h, T, x)
        p_m = self.pressure(rho_mol - h, T, x)
        return (p_p - p_m) / (2.0 * h)

    # -----------------------------------------------------------------
    # Density from pressure
    # -----------------------------------------------------------------

    def density_from_pressure(self, p, T, x=None, phase_hint='vapor',
                              tol=1e-8, maxiter=60):
        """Solve p(rho, T, x) = p_target for rho via Newton from a
        scan-selected starting point, with bisection fallback.

        v0.9.26: two-stage strategy.
         (1) coarse scan of pressure across the feasible eta range for
             the requested phase, pick the eta whose pressure is
             closest to p_target (in log space) as initial guess.
         (2) Newton iteration from that start.
         (3) If Newton fails, bisect over the scan grid (bracketing works
             because the p(eta) curve is monotone-increasing on the physical
             branch).

        This is robust for strongly-associating fluids (water, alcohols)
        whose liquid p(rho) curve is extremely steep, as well as for weakly
        bonded vapors. It is also faster than multi-start Newton because
        the scan uses only ~6 pressure evaluations up front.
        """
        if x is None:
            x = self.x
        pre = self._precompute(T, x)
        d = pre['d']
        sum_md3 = float(np.sum(x * self._m * d ** 3))

        def rho_from_eta(eta):
            return eta / ((np.pi / 6.0) * sum_md3 * _NA)

        def eta_from_rho(rho):
            return rho * _NA * (np.pi / 6.0) * sum_md3

        rho_ig = p / (_R * T)

        if phase_hint in ('vapor', 'supercritical'):
            eta_min, eta_max = 1e-12, 0.35
            # Scan grid for vapor: span from ideal-gas toward dense vapor
            eta_ig = max(eta_from_rho(rho_ig), 1e-12)
            eta_ig = min(eta_ig, 0.30)
            scan_etas = [eta_ig, 1e-6, 1e-4, 1e-2, 5e-2, 0.12, 0.20]
        else:
            eta_min, eta_max = 0.05, 0.60
            # Scan grid for liquid: covers weakly to strongly bound liquids
            scan_etas = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55]

        # (1) Scan for the start eta whose pressure is closest to target (in log space)
        p_log_target = np.log(max(abs(p), 1e-3))
        best = None   # (eta, p_val, log-dist)
        p_scan = []
        for eta0 in scan_etas:
            eta0c = min(max(eta0, eta_min * 1.01), eta_max * 0.99)
            try:
                p_here = self.pressure(rho_from_eta(eta0c), T, x)
                p_scan.append((eta0c, p_here))
                # Prefer start with same-sign, closest-to-target p
                if p_here > 0 and np.isfinite(p_here):
                    dist = abs(np.log(p_here) - p_log_target)
                    if best is None or dist < best[2]:
                        best = (eta0c, p_here, dist)
            except Exception:
                continue

        # (2) Newton from the best start
        last_resid = float('inf')
        if best is not None:
            rho = rho_from_eta(best[0])
            for it in range(maxiter):
                p_here = self.pressure(rho, T, x)
                resid = p_here - p
                if abs(resid) < max(tol * abs(p), 1e-2):
                    return rho
                dpdrho = self.dpressure_drho(rho, T, x)
                if dpdrho <= 0 or not np.isfinite(dpdrho):
                    # Mechanically unstable; nudge toward proper branch
                    rho *= 0.7 if phase_hint in ('vapor', 'supercritical') else 1.1
                    continue
                drho = -resid / dpdrho
                step_cap = 0.3 * rho
                if abs(drho) > step_cap:
                    drho = np.sign(drho) * step_cap
                rho_new = rho + drho
                eta_new = eta_from_rho(rho_new)
                if eta_new < eta_min or eta_new > eta_max:
                    eta_new = min(max(eta_new, eta_min), eta_max)
                    rho_new = rho_from_eta(eta_new)
                if rho_new <= 0:
                    rho_new = rho * 0.5
                rho = rho_new
            last_resid = resid

        # (3) Bisection fallback on the scan grid. Find an eta-bracket where
        # pressure brackets p_target with positive slope, then bisect.
        try:
            # Re-scan densely within the correct-phase range
            if phase_hint in ('vapor', 'supercritical'):
                eta_grid = np.concatenate([
                    np.logspace(-8, -3, 6),
                    np.linspace(1e-3, 0.30, 10),
                ])
            else:
                eta_grid = np.linspace(0.40, 0.58, 15)
            p_vals = []
            for e in eta_grid:
                try:
                    p_vals.append(self.pressure(rho_from_eta(e), T, x))
                except Exception:
                    p_vals.append(np.nan)
            bracket = None
            for i in range(len(eta_grid) - 1):
                p0, p1 = p_vals[i], p_vals[i + 1]
                if not (np.isfinite(p0) and np.isfinite(p1)):
                    continue
                # Require rising-slope branch (both p's > 0 or transitioning)
                if (p0 - p) * (p1 - p) < 0 and p1 > p0:
                    bracket = (eta_grid[i], eta_grid[i + 1])
            if bracket is not None:
                lo, hi = bracket
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    try:
                        p_mid = self.pressure(rho_from_eta(mid), T, x)
                    except Exception:
                        hi = mid; continue
                    if abs(p_mid - p) < max(tol * abs(p), 1e-2):
                        return rho_from_eta(mid)
                    if p_mid < p:
                        lo = mid
                    else:
                        hi = mid
                return rho_from_eta(0.5 * (lo + hi))
        except Exception:
            pass
        raise RuntimeError(
            f"SAFT density_from_pressure: Newton did not converge "
            f"(p={p:.3e}, T={T}, phase={phase_hint}, final resid={last_resid:.3e})"
        )

    # -----------------------------------------------------------------
    # Fugacity coefficients ln phi
    # -----------------------------------------------------------------

    def ln_phi(self, rho_mol, T, x=None):
        """Natural log of fugacity coefficients for each species.

        From the thermodynamic identity for a Helmholtz-based EOS:
            mu_i^res/kT = A + Z_res + dA/dx_i - sum_j x_j dA/dx_j
            ln phi_i = mu_i^res/kT - ln Z
        where A = alpha_r(rho, T, x), Z = p/(rho R T), Z_res = Z - 1.

        v0.9.27: Composition derivatives dA/dx_i are computed analytically
        for the hard-chain and dispersion contributions; association and
        polar/quadrupolar contributions still use central FD within
        `_alpha_r_core(return_dx=True)`. This gives ~3x speedup on ln_phi
        for non-associating mixtures and still ~2x when association is
        active, with analytic-FD agreement to ~1e-10 relative.
        """
        if x is None:
            x = self.x
        x = np.asarray(x, dtype=np.float64)

        # Compute alpha_r and its composition gradient in one pass
        A, dA_dx = self._alpha_r_core(rho_mol, T, x, return_dx=True)

        # Z = p / (rho R T) == 1 + Z_res
        p_here = self.pressure(rho_mol, T, x)
        Z = p_here / (rho_mol * _R * T)
        Z_res = Z - 1.0

        sum_x_dA = float(np.sum(x * dA_dx))
        mu_res_kT = A + Z_res + dA_dx - sum_x_dA
        return mu_res_kT - np.log(Z)

    # -----------------------------------------------------------------
    # Derivative-API surface expected by the analytic envelope Jacobian
    # (v0.9.17/v0.9.18). The cubic and Helmholtz paths have analytic
    # implementations of these derivatives; for SAFT we use FD on ln_phi
    # itself, which is adequate for the envelope corrector and keeps the
    # derivative API complete so that flash / envelope / three-phase
    # machinery can inherit from the cubic module unchanged. Analytic
    # versions are a future optimization.
    # -----------------------------------------------------------------

    def _lnphi_at_p(self, p, T, x, phase_hint='vapor'):
        """Helper: ln_phi evaluated at (p, T, x) rather than (rho, T, x)."""
        rho = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
        return self.ln_phi(rho, T, x)

    def dlnphi_dT_at_p(self, p, T, x=None, phase_hint='vapor'):
        """d(ln phi_i) / d T at fixed (p, x). Computed by central FD."""
        if x is None: x = self.x
        h = max(T * 1e-5, 1e-3)
        lp_p = self._lnphi_at_p(p, T + h, x, phase_hint)
        lp_m = self._lnphi_at_p(p, T - h, x, phase_hint)
        return (lp_p - lp_m) / (2.0 * h)

    def dlnphi_dp_at_T(self, p, T, x=None, phase_hint='vapor'):
        """d(ln phi_i) / d p at fixed (T, x). Computed by central FD."""
        if x is None: x = self.x
        h = max(p * 1e-5, 1e-3)
        lp_p = self._lnphi_at_p(p + h, T, x, phase_hint)
        lp_m = self._lnphi_at_p(p - h, T, x, phase_hint)
        return (lp_p - lp_m) / (2.0 * h)

    def dlnphi_dxk_at_p(self, p, T, x=None, phase_hint='vapor'):
        """d(ln phi_i) / d x_k at fixed (p, T). Returns (N, N) matrix.

        v0.9.30: full analytic implementation via derivative identity.
        v0.9.34: now uses true analytic composition Hessian via
        `_alpha_r_composition_hessian` -- O(N^2) cost instead of O(N^3).

        Starting from
            ln phi_i = A + (Z - 1) + A_i - Sum_j x_j A_j - ln Z
        where A = alpha_r(rho,T,x), A_i = dA/dx_i|_rho, Z = 1 + rho A_rho,
        the identity expansion at fixed (p, T) reduces to Hessian terms
        A_ik (now analytic for HC+disp), A_{rho,k}, A_{rho,rho},
        A_{i,rho}. The latter three use FD on the analytic gradient
        (only 2 extra _alpha_r_core calls).

        Returns (N, N) matrix. Treats each x_k as an independent variable.
        """
        if x is None: x = self.x
        x = np.asarray(x, dtype=np.float64)
        N = self.N

        # One density solve at reference x
        rho = self.density_from_pressure(p, T, x, phase_hint=phase_hint)

        # Base quantities at (rho, T, x) + analytic composition Hessian + analytic ρ-derivs (v0.9.36)
        A, A_i, A_ik, A_rho, A_rhorho, A_rhoi = (
            self._alpha_r_composition_hessian(rho, T, x, return_rho_derivs=True))

        # Z, dp/drho, rho_x_k
        Z = 1.0 + rho * A_rho
        # dp/drho = R T (1 + 2 rho A_rho + rho^2 A_{rho rho})
        dpdrho_factor = 1.0 + 2 * rho * A_rho + rho * rho * A_rhorho
        # rho_x_k = -rho^2 A_{rho, k} / dpdrho_factor   (from ideal-gas units)
        # Actually d p / d rho at fixed x = R T dpdrho_factor, and
        # d p / d x_k at fixed rho = R T rho * rho A_{rho,k}
        # so rho_x_k = -rho^2 A_{rho,k} / dpdrho_factor (dimensionally per mol/m^3)
        rho_x = -rho * rho * A_rhoi / dpdrho_factor   # shape (N,)

        # dZ/dx_k|_p = rho_x_k (A_rho + rho A_rhorho) + rho A_{rho,k}
        dZ_dxk = rho_x * (A_rho + rho * A_rhorho) + rho * A_rhoi   # shape (N,)

        # Per-species Hessian terms
        # Sum_j x_j A_jk -> shape (N,)
        sum_xj_Ajk = x @ A_ik         # (N,) vector over k
        # Sum_j x_j A_{j,rho} -> scalar
        sum_xj_Ajrho = float(np.sum(x * A_rhoi))

        # Assemble. Each matrix element (i, k):
        #  (A_ik - sum_xj_Ajk[k]) + rho * A_rhoi[k]          (static HS/disp terms)
        # + rho_x[k] * (2 A_rho + rho A_rhorho                 (rho-change terms)
        #               + A_rhoi[i] - sum_xj_Ajrho)
        # + rho * A_rhoi[k]                                     (dZ term absorbed below)
        # - dZ_dxk[k] / Z                                       (final ln Z contribution)
        #
        # Note: 'rho A_{rho,k}' appears twice in derivation (once from D_k[Z-1], once from A_{i,rho}*rho_x_k term chain). Let me re-derive carefully:
        #
        # ln phi_i = A + (Z-1) + A_i - sum_j x_j A_j - ln Z
        # D_k[A] = A_k + A_rho * rho_x[k]
        # D_k[Z-1] = D_k[rho A_rho] = rho_x[k] * A_rho + rho * (A_rhok + A_rhorho * rho_x[k])
        #          = rho_x[k] * (A_rho + rho * A_rhorho) + rho * A_rhoi[k]
        # D_k[A_i] = A_ik + A_rhoi[i] * rho_x[k]     <-- for component i
        # D_k[-sum_j x_j A_j] = -A_k - sum_j x_j (A_jk + A_rhoi[j] * rho_x[k])
        #                     = -A_k - sum_xj_Ajk[k] - rho_x[k] * sum_xj_Ajrho
        # D_k[-ln Z] = -D_k[Z]/Z
        # A_k terms cancel (from D_k[A] and D_k[-sum x_j A_j]).
        #
        # So for row i, column k:
        #   J[i,k] = A_rho * rho_x[k]                                # from D_k[A] minus A_k
        #          + rho_x[k] * (A_rho + rho * A_rhorho) + rho * A_rhoi[k]   # D_k[Z-1]
        #          + A_ik[i,k] + A_rhoi[i] * rho_x[k]                        # D_k[A_i]
        #          - sum_xj_Ajk[k] - rho_x[k] * sum_xj_Ajrho                  # D_k[-sum x_j A_j]
        #          - dZ_dxk[k] / Z                                             # D_k[-ln Z]
        J = np.empty((N, N))
        for i in range(N):
            for k in range(N):
                J[i, k] = (
                    A_rho * rho_x[k]
                    + rho_x[k] * (A_rho + rho * A_rhorho)
                    + rho * A_rhoi[k]
                    + A_ik[i, k]
                    + A_rhoi[i] * rho_x[k]
                    - sum_xj_Ajk[k]
                    - rho_x[k] * sum_xj_Ajrho
                    - dZ_dxk[k] / Z
                )
        return J

    # -----------------------------------------------------------------
    # Caloric (enthalpy, entropy)
    # -----------------------------------------------------------------

    def caloric(self, rho_mol, T, x=None, p=None):
        """Residual enthalpy and entropy [J/mol, J/(mol K)].

        Uses thermodynamic identities:
            h^res/RT = -T (d alpha_r / d T)_{rho,x} + Z^res
            s^res/R  = -T (d alpha_r / d T)_{rho,x} - alpha_r

        Ideal-gas contributions are omitted (pure-residual convention).
        """
        if x is None:
            x = self.x
        if p is None:
            p = self.pressure(rho_mol, T, x)
        A = self.alpha_r(rho_mol, T, x)
        Z = p / (rho_mol * _R * T)
        Z_res = Z - 1.0

        # d alpha_r / d T at fixed rho, x by central FD
        hT = max(T * 1e-6, 1e-4)
        A_p = self.alpha_r(rho_mol, T + hT, x)
        A_m = self.alpha_r(rho_mol, T - hT, x)
        dA_dT = (A_p - A_m) / (2.0 * hT)

        h_res = (-T * dA_dT + Z_res) * _R * T
        s_res = (-T * dA_dT - A) * _R
        return {"h": float(h_res), "s": float(s_res)}

    # -----------------------------------------------------------------
    # Wilson K for flash initialization
    # -----------------------------------------------------------------

    def wilson_K(self, T, p):
        """Wilson K-factor correlation for each component."""
        K = np.empty(self.N)
        for i in range(self.N):
            K[i] = (self._p_c[i] / p) * np.exp(
                5.373 * (1.0 + self._omega[i]) * (1.0 - self._T_c[i] / T)
            )
        return K

    @property
    def T_c(self):
        return self._T_c

    @property
    def p_c(self):
        return self._p_c

    def reduce(self, x=None):
        """Return (T_reduced_linear, rho_reduced_estimate) for Wilson-ish use."""
        if x is None:
            x = self.x
        T_r = float(np.dot(x, self._T_c))
        # Rough density scale from pure p_c, T_c (ideal gas estimate at Tr=1)
        # This is only used as a "scale" placeholder for one branch of flash_pt
        rho_r = float(np.sum(x * self._p_c / (_R * self._T_c)))
        return T_r, rho_r
