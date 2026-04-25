"""
Multicomponent cubic EOS via van der Waals one-fluid mixing.

For a mixture with composition x = (x_1, ..., x_N), the mixture parameters are:

    a_mix(T, x) = sum_i sum_j x_i x_j (1 - k_ij) sqrt(a_i(T) a_j(T))
    b_mix(x)    = sum_i x_i b_i

where k_ij (=k_ji, k_ii = 0) are binary interaction parameters,
commonly small (|k_ij| < 0.2), tabulated from experimental data.

With (a_mix, b_mix) plugged into the same cubic functional form as pure
components, the mixture residual Helmholtz and ln_phi have closed-form
expressions.

Classical mixture ln_phi for a cubic EOS (Michelsen-Mollerup, Eq. 4.64 etc):

    ln phi_i(T, p, x) = b_i/b_mix * (Z - 1)
                         - ln(Z - B)
                         + A/(B*(eps - sig))
                           * [2 sum_j(x_j sqrt(a_i a_j)(1-k_ij))/a_mix - b_i/b_mix]
                           * ln((Z + sig B)/(Z + eps B))

where A = a_mix p/(RT)^2 and B = b_mix p/(RT), and Z = pv/(RT).

Density solving: given (p, T, x), solve the cubic in Z and select the root
per phase hint (liquid = largest Z root with Z > B; vapor = smallest root).

Reference: Michelsen & Mollerup, "Thermodynamic Models: Fundamentals and
Computational Aspects" (2nd ed., 2007), Chapter 4-5.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .eos import CubicEOS, PR, SRK, RK, VDW, _cubic_real_roots


class CubicMixture:
    """An N-component mixture using van der Waals one-fluid mixing of a cubic EOS.

    All components must use the same EOS family (all PR, or all SRK, etc.),
    since the (epsilon, sigma, Omega_a, Omega_b) are tied to the family.

    Parameters
    ----------
    components : list[CubicEOS]
        Each a pure-fluid CubicEOS. Must all share the same family.
    composition : array-like, shape (N,)
        Mole fractions; normalized to sum=1.
    k_ij : dict[(int, int), float] or (N, N) ndarray
        Binary interaction parameters. Diagonal ignored (k_ii = 0). Missing
        entries default to 0.
    """
    def __init__(self, components, composition=None, k_ij=None):
        self.components = list(components)
        self.N = len(self.components)
        if self.N < 1:
            raise ValueError("CubicMixture must have at least 1 component")
        # Verify EOS-family compatibility.
        # Components may mix within an equivalence class that shares
        # (epsilon, sigma, Omega_a, Omega_b). For PR, that means 'pr' and
        # 'pr78' may be freely mixed (they differ only in the m(omega)
        # polynomial, applied per-component before mixing). SRK cannot mix
        # with PR because their (eps, sigma) differ.
        _pr_like = {"pr", "pr78"}

        def _family_class(fam_name):
            fn = fam_name.lower()
            return "pr" if fn in _pr_like else fn

        fam0 = _family_class(self.components[0].family)
        for c in self.components[1:]:
            if _family_class(c.family) != fam0:
                raise ValueError(
                    f"All components must share the same EOS family class; "
                    f"got {self.components[0].family!r} and {c.family!r}."
                )
        # For the mixture's bookkeeping, use the family string of the first
        # component (user-visible). The actual (epsilon, sigma) come from the
        # effective family (same for pr and pr78).
        self.family = self.components[0].family
        self.epsilon = self.components[0].epsilon
        self.sigma   = self.components[0].sigma
        self.R = self.components[0].R

        # k_ij matrix
        K = np.zeros((self.N, self.N))
        if isinstance(k_ij, dict):
            for (i, j), val in k_ij.items():
                K[i, j] = val
                K[j, i] = val
        elif k_ij is not None:
            K = np.asarray(k_ij, dtype=np.float64)
            if K.shape != (self.N, self.N):
                raise ValueError(f"k_ij must be {self.N}x{self.N}")
        np.fill_diagonal(K, 0.0)
        self.k_ij = K

        # Aggregates
        self.T_c = np.array([c.T_c for c in self.components])
        self.p_c = np.array([c.p_c for c in self.components])
        self.names = [c.name for c in self.components]
        self.molar_masses = np.array([c.molar_mass for c in self.components])
        # Per-component volume shift parameters (for Peneloux-style translation).
        # Zero for untranslated components.
        self.c_shifts = np.array([getattr(c, "c", 0.0) for c in self.components])
        # Cache b_i values (constant, not T-dependent) for fast mixing
        self.b_vec = np.array([c.b for c in self.components])
        # Cache (1 - k_ij) matrix so hot paths avoid the subtraction each call
        self.one_minus_kij = 1.0 - self.k_ij
        # Volume-shift fast-path flag: skip all translation arithmetic when
        # no component has a nonzero shift (the common case).
        self._has_volume_shift = bool(np.any(self.c_shifts != 0.0))

        # Composition
        if composition is None:
            composition = np.full(self.N, 1.0 / self.N)
        self.set_composition(composition)

    def c_mix(self, x=None):
        """Linear mole-average mixture volume shift c_m = sum_i x_i c_i [m^3/mol]."""
        if x is None:
            x = self.x
        return float(np.dot(x, self.c_shifts))

    def set_composition(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.N,):
            raise ValueError(f"Composition must have length N={self.N}")
        s = x.sum()
        if s <= 0:
            raise ValueError("Composition must be positive")
        self.x = x / s

    def molar_mass(self, x=None):
        xv = self.x if x is None else np.asarray(x)
        return float(np.dot(xv, self.molar_masses))

    # ------------------------------------------------------------------
    # Mixture parameters a_mix(T, x), b_mix(x)
    # ------------------------------------------------------------------
    def a_b_mix(self, T, x=None):
        """Return (a_mix, b_mix, sqrt_a, SI, a_vec, da_mix_dT).

        Performance-critical: called in every flash / density / ln_phi
        evaluation. Uses fully vectorized numpy operations for the N x N
        double sum rather than Python loops.

        a_mix = sum_i sum_j x_i x_j (1 - k_ij) sqrt(a_i a_j)
              = x.T @ [sqrt_a outer sqrt_a * (1-k_ij)] @ x
        SI_i  = 2 * sum_j x_j (1 - k_ij) sqrt(a_i a_j)   (used by ln_phi)
        """
        if x is None:
            x = self.x

        # Per-component a(T) and da/dT. These scalar calls are the only
        # non-vectorized cost left (each component dispatches on its own
        # alpha function kind). For typical N < 10 the overhead is small.
        N = self.N
        a_vec = np.empty(N)
        da_dT = np.empty(N)
        for i, c in enumerate(self.components):
            a_i, da_i, _ = c.a_T(T)
            a_vec[i] = a_i
            da_dT[i] = da_i
        sqrt_a = np.sqrt(a_vec)

        # Build the symmetric matrix A_ij = sqrt(a_i) sqrt(a_j) (1-k_ij)
        # then a_mix = x.T A x, SI = 2 * A @ x (using sqrt_a factored out).
        #
        # Vectorized: A_ij = (sqrt_a outer sqrt_a) * one_minus_kij
        # This is O(N^2) in numpy C loops, not Python loops.
        A_ij = np.outer(sqrt_a, sqrt_a) * self.one_minus_kij

        # a_mix = x . (A_ij . x) = quadratic form in x
        Ax = A_ij.dot(x)
        a_mix = float(x.dot(Ax))
        SI = 2.0 * Ax       # SI_i = 2 * sum_j x_j A_ij

        # b_mix = sum_i x_i b_i
        b_mix = float(x.dot(self.b_vec))

        return a_mix, b_mix, sqrt_a, SI, a_vec, da_dT

    def b_i(self, i):
        return self.components[i].b

    def reduce(self, x=None):
        """Return pseudo-critical (T_c, rho_c) for the mixture.

        For cubics without a real mixture reducing function, we just use
        the mole-averaged critical temperature and a mole-averaged
        pseudo-critical density (from the EOS-predicted rho_c of each
        pure component).
        """
        if x is None:
            x = self.x
        Tc_mix = float(np.dot(x, self.T_c))
        rho_c_mix = 1.0 / float(sum(x[i] / self.components[i].rho_c for i in range(self.N)))
        return Tc_mix, rho_c_mix

    def wilson_K(self, T, p):
        """Wilson K-factor estimate for each component at (T, p).

            K_i = (p_c_i / p) * exp(5.373 * (1 + omega_i) * (1 - T_c_i/T))
        """
        K = np.zeros(self.N)
        for i, c in enumerate(self.components):
            K[i] = (c.p_c / p) * np.exp(5.373 * (1.0 + c.acentric_factor) * (1.0 - c.T_c / T))
        return K

    # ------------------------------------------------------------------
    # Residual enthalpy and entropy
    # ------------------------------------------------------------------
    # For a cubic EOS with mixture parameters (a_mix, b_mix), the residual
    # (departure) enthalpy and entropy are:
    #
    #   h_res/(RT) = (a_mix - T*da_mix/dT) / (RT*b_mix*(sig-eps))
    #                 * ln((1 + sig*B)/(1 + eps*B))
    #                + (Z - 1)
    #
    #   s_res/R    = (-da_mix/dT) / (R*b_mix*(sig-eps)) * ln((1 + sig*B)/(1 + eps*B))
    #                + ln(Z - B)
    #
    # where B = b_mix*p/(RT), Z = pv/(RT).
    #
    # In a complete EOS, total h = h_ideal(T, x) + h_res. The cubic module
    # does not ship ideal-gas correlations for h and s (one would need Cp^ig
    # correlations per component). Users can combine h_res from here with
    # their own ideal-gas reference to get absolute h, s. For two-phase flash
    # work, only DIFFERENCES in h and s matter, and those are fully captured
    # by the residual parts (when ideal-gas contributions are composition-only).

    def residual_h_s(self, rho, T, x=None):
        """DEPRECATED: use `caloric(rho, T, x)` instead, which returns a dict
        with pressure, Z, h, s, u (both ideal-gas and residual parts).

        This method is kept as a thin wrapper for backwards compatibility.
        Returns (h_res_RT, s_res_R, Z).
        """
        cal = self.caloric(rho, T, x)
        h_res_RT = cal["h_res"] / (self.R * T)
        s_res_R = cal["s_res"] / self.R
        return h_res_RT, s_res_R, cal["Z"]

    # ------------------------------------------------------------------
    # Density root solving
    # ------------------------------------------------------------------
    def cubic_Z_roots(self, T, p, x=None):
        """Solve the cubic in Z at (T, p, x). Return sorted real roots with Z > B.

        The cubic (derived from p = RT/(v-b) - a/((v+eps*b)(v+sig*b))):
            Z^3 - (1 + B - uB) Z^2 + (A + (w^2 - u) B^2 - u B) Z - (w^2 B^3 + w^2 B^2 + A B) = 0
        where u = eps + sig, w^2 = eps * sig, A = a*p/(RT)^2, B = b*p/(RT).

        Returns (Zs, A, B).  Zs is sorted smallest first (vapor-like).
        """
        a_mix, b_mix, *_ = self.a_b_mix(T, x)
        A = a_mix * p / (self.R * T) ** 2
        B = b_mix * p / (self.R * T)
        eps_ = self.epsilon; sig = self.sigma
        u = eps_ + sig; w2 = eps_ * sig
        c2 = -(1.0 + B - u * B)
        c1 = A + (w2 - u) * B * B - u * B
        c0 = -(w2 * B ** 3 + w2 * B ** 2 + A * B)
        roots = _cubic_real_roots(c2, c1, c0)
        real = [r for r in roots if r > B + 1e-14]
        return sorted(real), A, B

    def density_from_pressure(self, p, T, x=None, phase_hint="vapor"):
        """Solve rho from (p, T, x) on requested phase branch.

        The cubic can have 1 or 3 real Z roots with Z > B. Convention:
          - smallest Z  <=>  smallest v  <=>  largest rho  = liquid
          - largest Z   <=>  largest v   <=>  smallest rho = vapor
          - middle Z (only when 3 roots): thermodynamically unstable, discarded.

        When volume translation is active (any component has c != 0), the
        cubic's Z solves for v_cubic = Z*R*T/p; the real molar volume is
        v_real = v_cubic - c_mix(x), and rho = 1/v_real.

        Returns rho (mol/m^3).
        """
        Zs, A, B = self.cubic_Z_roots(T, p, x)
        if len(Zs) == 0:
            raise RuntimeError(
                f"Cubic has no real roots with Z > B at T={T}, p={p}, x={x}."
            )
        if len(Zs) == 1:
            Z = Zs[0]
        else:
            if len(Zs) == 3:
                Zs = [Zs[0], Zs[-1]]
            Z = Zs[-1] if phase_hint == "vapor" else Zs[0]
        if not self._has_volume_shift:
            return p / (Z * self.R * T)
        v_cubic = Z * self.R * T / p
        c_m = self.c_mix(x)
        v_real = v_cubic - c_m
        if v_real <= 0:
            raise RuntimeError(
                f"volume-translated v_real={v_real} <= 0 at T={T}, p={p}, "
                f"x={x}; c_mix={c_m} may be too large for this state."
            )
        return 1.0 / v_real

    # ------------------------------------------------------------------
    # Caloric properties: h, s, u
    # ------------------------------------------------------------------
    def caloric(self, rho, T, x=None, p=None):
        """Return a dict with pressure, enthalpy, entropy, internal energy
        of the mixture at (rho, T, x).

        All properties in SI units (J/mol or J/(mol K)). The ideal-gas part
        uses each pure component's Cp(T) polynomial (if specified) or the
        default constant 3.5R. The residual part is computed from the
        cubic alpha^r and its T-derivative at fixed rho.

        Parameters
        ----------
        rho : float
            molar density [mol/m^3]
        T : float
            temperature [K]
        x : array-like, optional
            composition; default: self.x
        p : float, optional
            If provided, used directly (caller may have computed p more
            accurately); otherwise p is computed from the cubic.

        Returns a dict:
          {"p": p, "Z": Z, "h": h, "s": s, "u": u, "h_res": h_res,
           "s_res": s_res, "u_res": u_res, "h_ig": h_ig, "s_ig": s_ig}
        """
        if x is None:
            x = self.x
        N = self.N
        a_mix, b_mix, sqrt_a, _, a_vec, da_dT_vec = self.a_b_mix(T, x)

        # Mixture da_mix/dT: derivative of a_mix = sum_ij x_i x_j (1-k_ij) sqrt(a_i a_j).
        # With u_i = sqrt(a_i), du_i/dT = 0.5 da_i/dT / u_i, and
        #   d/dT(u_i u_j) = 0.5*(u_i * da_j/dT / u_j + u_j * da_i/dT / u_i).
        # Fully vectorized using outer products -- O(N^2) in numpy C, no Python loops:
        #   r_i = da_i/dT / u_i   (length N)
        #   M_ij = 0.5 * (u_i * r_j + u_j * r_i) * (1 - k_ij)
        #   da_mix/dT = x.T M x
        r = da_dT_vec / sqrt_a
        M = 0.5 * (np.outer(sqrt_a, r) + np.outer(r, sqrt_a)) * self.one_minus_kij
        da_mix_dT = float(x @ M @ x)

        # Volume translation fast path
        if self._has_volume_shift:
            c_mix_val = self.c_mix(x)
            v_real = 1.0 / rho
            v_cubic = v_real + c_mix_val
            rho_cubic = 1.0 / v_cubic
        else:
            rho_cubic = rho
            v_cubic = 1.0 / rho
            v_real = v_cubic   # same when c=0

        B = b_mix * rho_cubic
        eps_ = self.epsilon
        sig = self.sigma
        R = self.R

        # Pressure from the cubic at v_cubic
        if p is None:
            if abs(sig - eps_) > 1e-14:
                p = R * T / (v_cubic - b_mix) - a_mix / (
                    (v_cubic + eps_ * b_mix) * (v_cubic + sig * b_mix)
                )
            else:
                p = R * T / (v_cubic - b_mix) - a_mix / (v_cubic * v_cubic)
        # User-visible compressibility: Z = p * v_real / (RT)
        Z = p * v_real / (R * T)

        # Residual properties from alpha_r = -ln(1-B) - (a_m/(RT b_m(sig-eps))) * L
        if abs(sig - eps_) > 1e-14:
            L = np.log((1.0 + sig * B) / (1.0 + eps_ * B))
            alpha_r = -np.log(1.0 - B) - (a_mix / (R * T * b_mix * (sig - eps_))) * L
            u_res_RT = L * (T * da_mix_dT - a_mix) / (R * T * b_mix * (sig - eps_))
        else:
            # vdW: alpha_r = -ln(1-B) - a_m rho_cubic / (RT)
            alpha_r = -np.log(1.0 - B) - a_mix * rho_cubic / (R * T)
            u_res_RT = (da_mix_dT - a_mix / T) * rho_cubic / R

        u_res = R * T * u_res_RT
        # h_res at the real (translated) state: h_res = u_res + RT*(Z_real - 1)
        # where Z_real = p*v_real/(RT) -- using the user's real volume.
        h_res = u_res + R * T * (Z - 1.0)
        # s_res/R = u_res/(RT) - alpha_r  (unchanged by translation at fixed (T, rho_real),
        # since both sides use rho_cubic consistently)
        s_res = R * (u_res_RT - alpha_r)

        # Ideal-gas contributions.
        # h_ig(T, x) = sum x_i h_ig_i(T)
        # s_ig(T, rho, x) = sum x_i [s_ig_i(T, p=x_i*p_total) ... ] -- actually cleaner via partial densities
        #
        # For each component, s_ig_i at partial pressure p_i = x_i * p:
        #   s_ig_i(T, p_i) = s_ig_i(T, p_ref) - R*ln(p_i/p_ref)
        #                  = s_ig_i(T, p_ref) - R*ln(x_i) - R*ln(p/p_ref)
        # Total:
        #   s_ig = sum x_i [s_ig_i(T, p_ref) - R*ln(x_i) - R*ln(p/p_ref)]
        #        = [sum x_i s_ig_i(T, p_ref)] - R sum x_i ln(x_i) - R*ln(p/p_ref)
        # Each component's s_ig_i(T, p_ref) is computed via its own reference state.
        # For h_ig there's no composition correction (h_ig depends only on T).
        h_ig = 0.0
        s_ig = 0.0
        for i in range(N):
            if x[i] <= 0.0:
                continue
            h_ig += x[i] * self.components[i].ideal_gas_h(T)
            # Per-component s_ig at mixture's total pressure (not partial);
            # the ideal-mixing entropy is added explicitly below.
            s_ig += x[i] * self.components[i].ideal_gas_s(T, p)
            # Ideal mixing entropy contribution (always, when x_i > 0):
            s_ig += -R * x[i] * np.log(x[i])

        h = h_ig + h_res
        s = s_ig + s_res
        u = h - p / rho

        return {
            "p": p, "Z": Z,
            "h": h, "s": s, "u": u,
            "h_res": h_res, "s_res": s_res, "u_res": u_res,
            "h_ig": h_ig, "s_ig": s_ig,
        }
    def ln_phi(self, rho, T, x=None):
        """Log fugacity coefficients of each component at (rho, T, x).

        Derived from first principles via  ln phi_i = d(n alpha_r)/dn_i - ln Z
        at fixed T, V, n_{j != i}, using:

            alpha_r = -ln(1 - b_m rho) - [a_m / (RT b_m (sig-eps))] ln[(1 + sig b_m rho)/(1 + eps b_m rho)]

        with extensive quantities D_n = a_m n^2 = sum_{ij} n_i n_j (1-k_ij) sqrt(a_i a_j)
        and B_n = b_m n = sum_i n_i b_i. The derivatives d D_n / d n_i = 2 S_i (where
        S_i = sum_j n_j (1-k_ij) sqrt(a_i a_j)) and d B_n / d n_i = b_i are both
        simple linear-in-n expressions.

        This formula has been FD-verified to ~1e-9 across multiple compositions and
        (T, p) states. The algebra yields the final expression:

            ln phi_i = -ln(1 - B) + (rho b_i)/(1 - B)
                       - L/(RT (sig-eps)) * [SI_i / b_m - (a_m b_i)/b_m^2]
                       - (a_m rho b_i)/(RT b_m (sig-eps)) * [sig/(1+sig B) - eps/(1+eps B)]
                       - ln Z

        where SI_i = 2 * sum_j x_j (1 - k_ij) sqrt(a_i a_j),
              L    = ln[(1 + sig B) / (1 + eps B)],
              B    = b_m rho,  Z = p/(rho R T).
        """
        if x is None:
            x = self.x
        a_mix, b_mix, sqrt_a, SI, a_vec, _ = self.a_b_mix(T, x)

        # Volume translation: when active, rho_cubic = 1/(1/rho + c_mix);
        # otherwise rho_cubic = rho (fast path).
        if self._has_volume_shift:
            c_mix_val = self.c_mix(x)
            v_real = 1.0 / rho
            v_cubic = v_real + c_mix_val
            rho_cubic = 1.0 / v_cubic
        else:
            rho_cubic = rho
            v_cubic = 1.0 / rho

        B = b_mix * rho_cubic
        eps_ = self.epsilon
        sig = self.sigma
        R = self.R
        RT = R * T

        # Reconstruct p from the cubic at v_cubic
        if abs(sig - eps_) > 1e-14:
            p = RT / (v_cubic - b_mix) - a_mix / (
                (v_cubic + eps_ * b_mix) * (v_cubic + sig * b_mix)
            )
        else:
            p = RT / (v_cubic - b_mix) - a_mix / (v_cubic * v_cubic)
        Z_cubic = p * v_cubic / RT

        one_minus_B = 1.0 - B
        log_one_minus_B = np.log(one_minus_B)
        log_Z = np.log(Z_cubic)
        b_vec = self.b_vec

        if abs(sig - eps_) > 1e-14:
            L = np.log((1.0 + sig * B) / (1.0 + eps_ * B))
            bracket_dL = sig / (1.0 + sig * B) - eps_ / (1.0 + eps_ * B)
            term1 = -log_one_minus_B + (rho_cubic / one_minus_B) * b_vec
            term2a = -L / (RT * (sig - eps_)) * (
                SI / b_mix - a_mix * b_vec / (b_mix * b_mix)
            )
            term2b = (-a_mix * rho_cubic / (RT * b_mix * (sig - eps_))) * bracket_dL * b_vec
            lnphi = term1 + term2a + term2b - log_Z
        else:
            term1 = -log_one_minus_B + (rho_cubic / one_minus_B) * b_vec
            term2 = -SI * rho_cubic / RT
            lnphi = term1 + term2 - log_Z

        # Peneloux correction (only when active)
        if self._has_volume_shift:
            lnphi = lnphi + self.c_shifts * (p / RT)

        return lnphi

    # ------------------------------------------------------------------
    # Analytic composition derivatives of ln phi
    # ------------------------------------------------------------------
    #
    # The derivation below produces the N x N Jacobian
    #
    #     J[i, k] = d(ln phi_i) / d x_k    at fixed (T, p)
    #
    # which is the central piece needed for second-order methods on the
    # cubic-EOS flash: full Newton-Raphson on ln K (vs the rank-1 Broyden
    # secant approximation used in v0.9.7), Hessian-based stability tests
    # for global instability detection, arclength continuation along phase
    # envelopes, and direct critical-point solvers.
    #
    # Strategy: derive at fixed (T, rho) first (where everything is
    # algebraic in x), then convert to fixed (T, p) via the chain rule
    #
    #     dlnphi_i/dx_k|_{T,p} = dlnphi_i/dx_k|_{T,rho}
    #                          + dlnphi_i/drho|_{T,x} * drho/dx_k|_{T,p}
    #
    # where drho/dx_k|_{T,p} comes from implicit differentiation of the
    # state equation p(rho, T, x) = p_target:
    #
    #     drho/dx_k|_{T,p} = -dp/dx_k|_{T,rho} / dp/drho|_{T,x}
    #
    # The four scalar/vector/matrix building blocks are:
    #   (1) dlnphi_dx_at_rho   :  N x N
    #   (2) dlnphi_drho        :  N
    #   (3) dp_dx_at_rho       :  N
    #   (4) dp_drho_at_x       :  scalar  (already exists as dp_drho_T)
    #
    # All four are FD-verified to ~1e-7 absolute by the test suite. The
    # combined J is FD-verified to ~1e-6 (slightly larger due to compounded
    # rounding in the chain-rule combination).

    def _dp_dx_at_rho(self, rho, T, x):
        """Composition derivative of pressure at fixed (T, rho).

        From p = RT*rho/(1-B) - a_m * rho^2 / [(1+eps*B)(1+sig*B)]:
            dp/dx_k = RT * rho * (b_k * rho) / (1-B)^2
                    - SI_k * rho^2 / [(1+eps*B)(1+sig*B)]
                    + a_m * rho^2 * b_k * rho * (eps + sig + 2*eps*sig*B)
                                              / [(1+eps*B)(1+sig*B)]^2

        For vdW degenerate case (eps == sig == 0), the second term reduces
        to -SI_k * rho^2 and the third term vanishes.

        Returns ndarray of shape (N,).
        """
        a_mix, b_mix, _, SI, _, _ = self.a_b_mix(T, x)
        eps_ = self.epsilon
        sig = self.sigma
        RT = self.R * T
        b_vec = self.b_vec
        B = b_mix * rho
        one_minus_B = 1.0 - B
        # Term 1: derivative of RT*rho/(1-B)
        # = RT*rho * d(1/(1-B))/dx_k = RT*rho * (b_k*rho)/(1-B)^2
        t1 = RT * rho * (b_vec * rho) / (one_minus_B * one_minus_B)
        if abs(sig - eps_) > 1e-14:
            E = 1.0 + eps_ * B
            S = 1.0 + sig * B
            ES = E * S
            # Term 2: -SI_k * rho^2 / ES
            t2 = -SI * rho * rho / ES
            # Term 3: -a_m * rho^2 * d(1/ES)/dx_k
            #       = +a_m * rho^2 * (eps*S + sig*E) * b_k * rho / ES^2
            # Note: d(ES)/dx_k = eps*b_k*rho * S + sig*b_k*rho * E
            #                  = b_k*rho * (eps*S + sig*E)
            t3 = a_mix * rho * rho * b_vec * rho * (eps_ * S + sig * E) / (ES * ES)
            return t1 + t2 + t3
        # vdW degenerate: p = RT*rho/(1-B) - a_m * rho^2
        return t1 - SI * rho * rho

    def _dlnphi_drho_at_x(self, rho, T, x):
        """Density derivative of ln phi at fixed (T, x), N-vector.

        Differentiate the ln_phi formula w.r.t. rho. Each term is treated
        independently. With B = b_m*rho, dB/drho = b_m. Returns d/drho of
        each ln phi_i.
        """
        a_mix, b_mix, _, SI, _, _ = self.a_b_mix(T, x)
        eps_ = self.epsilon
        sig = self.sigma
        RT = self.R * T
        b_vec = self.b_vec
        B = b_mix * rho
        one_minus_B = 1.0 - B
        # Z = p / (rho * RT). d(ln Z)/drho = d ln p / drho - 1/rho
        # d ln p / drho = (1/p) * dp/drho
        # We need dp/drho at fixed (T, x), already implemented as dp_drho_T.
        # Use it directly.
        dp_drho = self._dp_drho_T_value(rho, T, x)
        # And p itself
        v = 1.0 / rho
        if abs(sig - eps_) > 1e-14:
            p = RT / (v - b_mix) - a_mix / ((v + eps_ * b_mix) * (v + sig * b_mix))
        else:
            p = RT / (v - b_mix) - a_mix / (v * v)
        # Term 1: -ln(1-B). d/drho = b_m/(1-B)  (component-independent scalar)
        d1 = b_mix / one_minus_B
        # Term 2: rho*b_i/(1-B). d/drho = b_i/(1-B) + rho*b_i*b_m/(1-B)^2
        d2 = b_vec / one_minus_B + rho * b_vec * b_mix / (one_minus_B * one_minus_B)
        # Term 5: -ln Z = -ln p + ln rho + ln(RT)
        # d/drho = -dp/drho / p + 1/rho
        d5 = -dp_drho / p + 1.0 / rho
        if abs(sig - eps_) > 1e-14:
            E = 1.0 + eps_ * B
            S = 1.0 + sig * B
            L = np.log(S / E)
            # dL/drho = d/drho [ln(1+sig*B) - ln(1+eps*B)]
            #         = sig*b_m/S - eps*b_m/E
            #         = b_m * D_func   where D_func = sig/S - eps/E
            D_func = sig / S - eps_ / E
            dL_drho = b_mix * D_func
            U = SI / b_mix - a_mix * b_vec / (b_mix * b_mix)  # bracket from term 3
            # Term 3: -L/[RT(sig-eps)] * U  ... U doesn't depend on rho, only L does
            d3 = -dL_drho / (RT * (sig - eps_)) * U
            # Term 4: -(a_m * rho * b_i)/(RT * b_m * (sig-eps)) * D_func
            # Both rho and D_func depend on rho. Use product rule:
            #   prefactor = -(a_m * b_vec)/(RT * b_m * (sig-eps))
            #   d/drho [prefactor * rho * D_func] = prefactor * (D_func + rho * dD_func/drho)
            # dD_func/drho = -sig^2*b_m/S^2 + eps^2*b_m/E^2 = -b_m * Dprime
            # where Dprime = sig^2/S^2 - eps^2/E^2
            Dprime = sig * sig / (S * S) - eps_ * eps_ / (E * E)
            dD_drho = -b_mix * Dprime
            prefac = -(a_mix * b_vec) / (RT * b_mix * (sig - eps_))
            d4 = prefac * (D_func + rho * dD_drho)
            return d1 + d2 + d3 + d4 + d5
        # vdW degenerate: ln_phi reduces to
        #   lnphi_i = -ln(1-B) + rho*b_i/(1-B) - SI_i*rho/RT - ln Z
        # d4 absent in this branch.
        d3_vdw = -SI / RT
        return d1 + d2 + d3_vdw + d5

    def _dp_drho_T_value(self, rho, T, x):
        """Compute dp/drho at fixed (T, x). Helper used by _dlnphi_drho_at_x.

        Same math as the existing dp_drho method but returns just the scalar
        without re-fetching mixture params.
        """
        a_mix, b_mix, _, _, _, _ = self.a_b_mix(T, x)
        eps_ = self.epsilon
        sig = self.sigma
        RT = self.R * T
        v = 1.0 / rho
        # p = RT/(v-b) - a/((v+eps*b)(v+sig*b))
        # dp/drho at fixed (T, x) = -v^2 * dp/dv|_{T,x}
        # dp/dv = -RT/(v-b)^2 + a*[(v+eps*b)+(v+sig*b)]/[(v+eps*b)(v+sig*b)]^2
        # so dp/drho = v^2 * RT/(v-b)^2 - v^2 * a*[(v+eps*b)+(v+sig*b)]/[...]^2
        if abs(sig - eps_) > 1e-14:
            ve = v + eps_ * b_mix
            vs = v + sig * b_mix
            dp_drho = (v * v) * RT / ((v - b_mix) * (v - b_mix)) \
                    - (v * v) * a_mix * (ve + vs) / ((ve * vs) * (ve * vs))
        else:
            dp_drho = (v * v) * RT / ((v - b_mix) * (v - b_mix)) \
                    - (v * v) * a_mix * 2.0 / (v * v * v * v)
        return dp_drho

    def _dlnphi_dx_at_rho(self, rho, T, x):
        """N x N Jacobian d(ln phi_i) / d x_k at fixed (T, rho).

        From the ln_phi formula. Each of the four (or three, for vdW)
        terms in the formula is differentiated component-wise. The result
        is NOT symmetric in i and k because ln_phi_i has explicit i-dependence
        through b_i and SI_i.

        Returns ndarray of shape (N, N). J[i, k] = dlnphi_i/dx_k.
        """
        a_mix, b_mix, sqrt_a, SI, a_vec, _ = self.a_b_mix(T, x)
        eps_ = self.epsilon
        sig = self.sigma
        RT = self.R * T
        N = self.N
        b_vec = self.b_vec
        # A_ij = sqrt_a * sqrt_a * (1-k_ij), used for dSI_i/dx_k = 2*A_ik
        A_ij = np.outer(sqrt_a, sqrt_a) * self.one_minus_kij
        B = b_mix * rho
        one_minus_B = 1.0 - B

        # Build J term by term as N x N matrices.
        # All "k" indices are the differentiation variable; "i" indices
        # are the component whose ln phi we're differentiating.
        # b_vec[k] gives column-broadcast b_k; b_vec[i] gives row-broadcast b_i.

        # Term 1: -ln(1-B). dT1/dx_k = b_k * rho / (1-B). Same for all i.
        # As an N x N matrix: J1[i, k] = b_k * rho / (1-B).
        T1_row = b_vec * rho / one_minus_B          # shape (N,)
        J1 = np.broadcast_to(T1_row, (N, N))

        # Term 2: rho * b_i / (1-B). dT2/dx_k = rho * b_i * b_k * rho / (1-B)^2
        # J2[i, k] = b_i * b_k * rho^2 / (1-B)^2
        J2 = (rho * rho / (one_minus_B * one_minus_B)) * np.outer(b_vec, b_vec)

        if abs(sig - eps_) > 1e-14:
            E = 1.0 + eps_ * B
            S = 1.0 + sig * B
            L = np.log(S / E)
            D_func = sig / S - eps_ / E
            Dprime = sig * sig / (S * S) - eps_ * eps_ / (E * E)
            # dL/dx_k = D_func * b_k * rho
            dL_dx = D_func * b_vec * rho   # shape (N,)
            # dD_func/dx_k = -Dprime * b_k * rho
            dD_dx = -Dprime * b_vec * rho

            # U_i = SI_i/b_m - a_m b_i / b_m^2
            U = SI / b_mix - a_mix * b_vec / (b_mix * b_mix)   # shape (N,)
            # dU_i/dx_k:
            # d(SI_i/b_m)/dx_k = (2*A_ik * b_m - SI_i * b_k) / b_m^2
            # d(a_m b_i/b_m^2)/dx_k = b_i * d(a_m/b_m^2)/dx_k
            #   d(a_m/b_m^2)/dx_k = (SI_k * b_m^2 - a_m * 2*b_m*b_k) / b_m^4
            #                     = SI_k/b_m^2 - 2*a_m*b_k/b_m^3
            # So d(a_m b_i/b_m^2)/dx_k = b_i*(SI_k/b_m^2 - 2*a_m*b_k/b_m^3)
            dU_dx = (2.0 * A_ij * b_mix - np.outer(SI, b_vec)) / (b_mix * b_mix) \
                  - np.outer(b_vec, SI / (b_mix * b_mix) - 2.0 * a_mix * b_vec / (b_mix ** 3))
            # J3[i, k] = -1/(RT*(sig-eps)) * (dL_dx[k] * U[i] + L * dU_dx[i, k])
            J3 = -1.0 / (RT * (sig - eps_)) * (np.outer(U, dL_dx) + L * dU_dx)

            # Term 4: -(a_m * rho * b_i) / (RT * b_m * (sig-eps)) * D_func
            # Let V_i = a_m * rho * b_i / (RT * b_m * (sig-eps))
            # Then T4_i = -V_i * D_func, so dT4_i/dx_k = -dV_i/dx_k * D_func - V_i * dD_func/dx_k
            # V_i = (a_m / b_m) * (rho * b_i / (RT * (sig-eps)))
            # dV_i/dx_k = d(a_m/b_m)/dx_k * (rho * b_i / (RT * (sig-eps)))
            # d(a_m/b_m)/dx_k = (SI_k*b_m - a_m*b_k)/b_m^2
            d_amb_dx = (SI * b_mix - a_mix * b_vec) / (b_mix * b_mix)   # shape (N,) — this is over k
            V_i = (a_mix / b_mix) * (rho * b_vec / (RT * (sig - eps_)))    # shape (N,)
            # J4[i, k] = -(rho*b_i)/(RT*(sig-eps)) * (d_amb_dx[k] * D_func + (a_m/b_m) * dD_dx[k])
            prefac_i = -(rho * b_vec) / (RT * (sig - eps_))                # shape (N,)
            inner_k = d_amb_dx * D_func + (a_mix / b_mix) * dD_dx          # shape (N,)
            J4 = np.outer(prefac_i, inner_k)
        else:
            # vdW degenerate: term 3 collapses, term 4 absent.
            # ln_phi has the term -SI_i * rho / RT instead.
            # d(-SI_i * rho / RT)/dx_k = -2*A_ik * rho / RT
            J3 = -2.0 * A_ij * rho / RT
            J4 = np.zeros((N, N))

        # Term 5: -ln Z. Z = p/(rho*RT). At fixed (T, rho), dZ/dx_k = (1/(rho*RT)) * dp/dx_k.
        # d(ln Z)/dx_k = (1/Z) * (1/(rho*RT)) * dp/dx_k = dp/dx_k / p
        v = 1.0 / rho
        if abs(sig - eps_) > 1e-14:
            p = RT / (v - b_mix) - a_mix / ((v + eps_ * b_mix) * (v + sig * b_mix))
        else:
            p = RT / (v - b_mix) - a_mix / (v * v)
        dp_dx = self._dp_dx_at_rho(rho, T, x)            # shape (N,)
        T5_row = -dp_dx / p                              # same for all i
        J5 = np.broadcast_to(T5_row, (N, N))

        return J1 + J2 + J3 + J4 + J5

    def dlnphi_dxk_at_p(self, p, T, x, phase_hint='vapor'):
        """N x N Jacobian d(ln phi_i) / d x_k at fixed (T, p).

        This is the analytic derivative used to construct the Newton
        Jacobian for cubic-EOS flash. Combines:
          (1) d(ln phi_i)/dx_k at fixed (T, rho), with
          (2) d(ln phi_i)/drho at fixed (T, x), and
          (3) drho/dx_k at fixed (T, p) from implicit differentiation
              of the state equation p(rho, T, x) = p_target.

            J[i, k] = J_x[i, k]_{T,rho} + J_rho[i] * (drho/dx_k)_{T,p}
            (drho/dx_k)_{T,p} = -dp/dx_k|_{T,rho} / dp/drho|_{T,x}

        FD-verified to ~1e-6 across multiple compositions and phases.

        Parameters
        ----------
        p : pressure [Pa]
        T : temperature [K]
        x : composition (length N)
        phase_hint : 'vapor' | 'liquid' selecting which cubic root to use
            for rho. The Jacobian is phase-specific because the cubic
            generally has multiple physical roots.

        Returns
        -------
        J : ndarray, shape (N, N), J[i, k] = d(ln phi_i)/d x_k.
        """
        x = np.asarray(x, dtype=np.float64)
        # Volume-translation handling (v0.9.16):
        # The Peneloux shift adds `c_i * p/RT` to ln phi_i, which has no x
        # dependence at fixed (T, p) since c_i is a pure-component constant.
        # So dlnphi_real/dx_k = dlnphi_cubic/dx_k at fixed (T, p), and the
        # cubic derivative chain rule runs unchanged -- EXCEPT the helper
        # functions (_dlnphi_dx_at_rho, etc.) internally treat their rho
        # argument as rho_cubic. For Peneloux, density_from_pressure returns
        # rho_real = 1/v_real = 1/(v_cubic - c_mix), so we must convert to
        # rho_cubic = 1/(1/rho_real + c_mix) before calling the helpers.
        if self._has_volume_shift:
            rho_real = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
            c_m = self.c_mix(x)
            rho = 1.0 / (1.0 / rho_real + c_m)         # rho_cubic
        else:
            rho = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
        J_x_at_rho = self._dlnphi_dx_at_rho(rho, T, x)            # (N, N)
        dlnphi_drho = self._dlnphi_drho_at_x(rho, T, x)           # (N,)
        dp_dx = self._dp_dx_at_rho(rho, T, x)                     # (N,)
        dp_drho = self._dp_drho_T_value(rho, T, x)                # scalar
        drho_dx = -dp_dx / dp_drho                                 # (N,)
        # Chain rule: J[i, k] = J_x[i, k] + dlnphi_drho[i] * drho_dx[k]
        # Note: this is dlnphi_i/dx_k |_{T, p}, where rho (=rho_cubic) is
        # the implicit function of x at fixed (T, p) via the cubic's p
        # equation. No Peneloux correction term needed -- the c_i*p/RT
        # shift in ln phi is x-independent.
        return J_x_at_rho + np.outer(dlnphi_drho, drho_dx)

    # ------------------------------------------------------------------
    # Temperature and pressure derivatives of ln phi (v0.9.10)
    # ------------------------------------------------------------------
    #
    # The four building blocks below complete the primitive set of
    # analytic derivatives of ln phi needed for second-order methods on
    # the bubble-point, dew-point, and phase-envelope solvers:
    #
    #   d(ln phi)/dp at fixed (T, x)  -- N-vector
    #   d(ln phi)/dT at fixed (p, x)  -- N-vector
    #
    # Both are FD-verified to ~1e-8 across multiple phases/states by the
    # regression tests. Combined with dlnphi_dxk_at_p (v0.9.8), they give
    # the full set of first-order sensitivities of ln phi to its
    # independent variables, which is all that's needed for any Newton
    # solver in the (x, T, p) space.

    def _dp_dT_at_rho(self, rho, T, x):
        """Temperature derivative of pressure at fixed (rho, x).

        p = RT/(v - b_m) - a_m/((v+eps b_m)(v+sig b_m))
        Only a_m depends on T (through each pure component's a_i(T)).
        b_m does not depend on T, so nor does B = b_m * rho.

            dp/dT|_{rho, x} = R/(v - b_m) - (da_m/dT) / ((v+eps b_m)(v+sig b_m))
                            = R * rho/(1-B) - (da_m/dT) * rho^2 / ((1+eps B)(1+sig B))

        For vdW degenerate (eps == sig == 0):
            dp/dT|_{rho, x} = R * rho/(1-B) - (da_m/dT) * rho^2
        """
        a_mix, b_mix, sqrt_a, _, a_vec, da_dT_vec = self.a_b_mix(T, x)
        # da_m/dT = x^T M x  where M[i,j] = d A_ij/dT with A_ij = (1-k_ij)sqrt(a_i a_j)
        #   M[i, j] = (1 - k_ij)/(2 sqrt(a_i a_j)) * (r_i a_j + r_j a_i),  r_i = da_i/dT
        # Vectorized as in _dp_dT (if it existed); do it inline here.
        r = da_dT_vec / sqrt_a
        M = 0.5 * (np.outer(sqrt_a, r) + np.outer(r, sqrt_a)) * self.one_minus_kij
        da_mix_dT = float(x.dot(M.dot(x)))

        eps_ = self.epsilon
        sig = self.sigma
        B = b_mix * rho
        one_minus_B = 1.0 - B
        # dp/dT at fixed (rho, x)
        if abs(sig - eps_) > 1e-14:
            E = 1.0 + eps_ * B
            S = 1.0 + sig * B
            return self.R * rho / one_minus_B - da_mix_dT * rho * rho / (E * S)
        return self.R * rho / one_minus_B - da_mix_dT * rho * rho

    def _dlnphi_dT_at_rho(self, rho, T, x):
        """Temperature derivative of ln phi at fixed (rho, x), N-vector.

        Differentiate the ln_phi formula w.r.t. T at fixed (rho, x).
        Only a_m, SI_i carry T-dependence (through each pure a_i(T));
        L, B, bracket_dL are all pure functions of (rho, x).

        Writing U_i = SI_i/b_m - a_m b_i/b_m^2, term-by-term:

            term1 = -ln(1-B) + rho b_i/(1-B)                 no T-dep
            term2 = -L U_i/(RT (sig-eps))
                d/dT = -L/(R(sig-eps)) * (dU/dT / T - U/T^2)
                      = -L/(R T (sig-eps)) * (dU/dT - U/T)
            term3 = -(a_m rho b_i)/(RT b_m (sig-eps)) * bracket_dL
                d/dT = -(rho b_i bracket_dL)/(R b_m (sig-eps)) * (da_m/dT / T - a_m/T^2)
                     = -(rho b_i bracket_dL)/(R T b_m (sig-eps)) * (da_m/dT - a_m/T)
            term4 = -ln Z = -ln(p/(rho R T)) = -ln p + ln(rho R T)
                d/dT = -(1/p) dp/dT + 1/T

        For vdW degenerate: term2 and term3 collapse into -SI_i rho/(RT):
            term23 = -SI_i rho/(RT)
            d/dT = -rho/R * (dSI/dT/T - SI/T^2) = -rho/(RT) * (dSI/dT - SI/T)
        """
        a_mix, b_mix, sqrt_a, SI, a_vec, da_dT_vec = self.a_b_mix(T, x)
        # M[i, j] = d A_ij/dT, where A_ij = (1 - k_ij) sqrt(a_i a_j)
        r = da_dT_vec / sqrt_a
        M = 0.5 * (np.outer(sqrt_a, r) + np.outer(r, sqrt_a)) * self.one_minus_kij
        # dSI_i/dT = 2 sum_j x_j M[i, j]
        dSI_dT = 2.0 * M.dot(x)                    # (N,)
        da_mix_dT = float(x.dot(M.dot(x)))

        eps_ = self.epsilon
        sig = self.sigma
        RT = self.R * T
        b_vec = self.b_vec
        B = b_mix * rho
        one_minus_B = 1.0 - B
        v = 1.0 / rho
        if abs(sig - eps_) > 1e-14:
            E = 1.0 + eps_ * B
            S = 1.0 + sig * B
            L = np.log(S / E)
            bracket_dL = sig / S - eps_ / E
            U = SI / b_mix - a_mix * b_vec / (b_mix * b_mix)                   # (N,)
            # dU_i/dT
            dU_dT = dSI_dT / b_mix - da_mix_dT * b_vec / (b_mix * b_mix)       # (N,)
            t2 = -L / (RT * (sig - eps_)) * (dU_dT - U / T)
            t3 = -(rho * b_vec * bracket_dL) / (RT * b_mix * (sig - eps_)) * (da_mix_dT - a_mix / T)
            # Recompute p at fixed (rho, x, T)
            p = RT / (v - b_mix) - a_mix / ((v + eps_ * b_mix) * (v + sig * b_mix))
            dp_dT = self._dp_dT_at_rho(rho, T, x)
            t4 = -(1.0 / p) * dp_dT + 1.0 / T
            return t2 + t3 + t4
        # vdW degenerate
        p = RT / (v - b_mix) - a_mix / (v * v)
        dp_dT = self._dp_dT_at_rho(rho, T, x)
        t23 = -rho / (RT) * (dSI_dT - SI / T)
        t4 = -(1.0 / p) * dp_dT + 1.0 / T
        return t23 + t4

    def dlnphi_dp_at_T(self, p, T, x, phase_hint='vapor'):
        """N-vector d(ln phi)/dp at fixed (T, x).

        Only rho depends on p. So d(ln phi_cubic)/dp = d(ln phi_cubic)/drho_cubic
        / (dp/drho_cubic)_{T,x}. For Peneloux, add the explicit shift
        derivative d(c_i * p/RT)/dp = c_i / RT.

        FD-verified to ~1e-8 across phases.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._has_volume_shift:
            rho_real = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
            c_m = self.c_mix(x)
            rho = 1.0 / (1.0 / rho_real + c_m)     # rho_cubic
        else:
            rho = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
        dlnphi_drho = self._dlnphi_drho_at_x(rho, T, x)
        dp_drho = self._dp_drho_T_value(rho, T, x)
        dlnphi_dp_cubic = dlnphi_drho / dp_drho
        if self._has_volume_shift:
            # Peneloux correction: ln phi_real = ln phi_cubic + c_i * p/(RT)
            # d/dp at fixed T: + c_i / RT
            return dlnphi_dp_cubic + self.c_shifts / (self.R * T)
        return dlnphi_dp_cubic

    def dlnphi_dT_at_p(self, p, T, x, phase_hint='vapor'):
        """N-vector d(ln phi)/dT at fixed (p, x).

        Combines the partial at fixed rho with the implicit density
        response to temperature:

            d(ln phi_cubic)/dT|_{p,x} = d(ln phi_cubic)/dT|_{rho,x}
                                + d(ln phi_cubic)/drho|_{T,x} * drho/dT|_{p,x}
            drho/dT|_{p,x} = -dp/dT|_{rho,x} / dp/drho|_{T,x}

        For Peneloux, add the explicit shift derivative
        d(c_i * p/RT)/dT = -c_i * p/(RT^2).

        FD-verified to ~1e-8 across phases.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._has_volume_shift:
            rho_real = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
            c_m = self.c_mix(x)
            rho = 1.0 / (1.0 / rho_real + c_m)     # rho_cubic
        else:
            rho = self.density_from_pressure(p, T, x, phase_hint=phase_hint)
        dlnphi_dT_rho = self._dlnphi_dT_at_rho(rho, T, x)
        dlnphi_drho = self._dlnphi_drho_at_x(rho, T, x)
        dp_dT = self._dp_dT_at_rho(rho, T, x)
        dp_drho = self._dp_drho_T_value(rho, T, x)
        drho_dT = -dp_dT / dp_drho
        dlnphi_dT_cubic = dlnphi_dT_rho + dlnphi_drho * drho_dT
        if self._has_volume_shift:
            # Peneloux correction: ln phi_real = ln phi_cubic + c_i * p/(RT)
            # d/dT at fixed p: -c_i * p / (RT^2)
            return dlnphi_dT_cubic - self.c_shifts * p / (self.R * T * T)
        return dlnphi_dT_cubic


def p_Z(rho, T, a_mix, b_mix, eps_, sig, R):
    """Return Z = p*v/(RT) from (rho, T) and mixture parameters, via the cubic.

    Z = 1 + B_cub/(1 - B_cub) - A_cub/Z ... no wait, Z comes from
      Z = pv/RT with p = RT/(v-b) - a/((v+eps b)(v+sig b))
    So for given rho, T, a_mix, b_mix, compute p and hence Z.
    """
    v = 1.0 / rho
    rep = R * T / (v - b_mix)
    if abs(sig - eps_) > 1e-14:
        attr = a_mix / ((v + eps_ * b_mix) * (v + sig * b_mix))
    else:
        # vdW degenerate
        attr = a_mix / (v * v)
    p = rep - attr
    Z = p * v / (R * T)
    return Z


# ---------------------------------------------------------------------------
# Module-level convenience functions matching the Helmholtz-mixture API
# ---------------------------------------------------------------------------

def ln_phi(rho, T, x, mixture):
    """Module-level ln_phi matching the Helmholtz-mixture signature."""
    return mixture.ln_phi(rho, T, x)


def density_from_pressure(p, T, x, mixture, phase_hint="vapor"):
    return mixture.density_from_pressure(p, T, x, phase_hint=phase_hint)
