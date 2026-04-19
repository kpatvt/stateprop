"""
Generalized two-parameter cubic equations of state.

A two-parameter cubic EOS takes the general form:

    p = R T / (v - b) - a(T) / [(v + epsilon b)(v + sigma b)]

where v = 1/rho is molar volume, (epsilon, sigma) are EOS-family constants
that select the particular cubic (see table below), b is a temperature-
independent size parameter, and a(T) is a temperature-dependent attractive
parameter.

The residual Helmholtz energy has the closed form (for epsilon != sigma):

    alpha_r(T, rho) = A_res/(RT)
                    = -ln(1 - b rho)
                      - [a(T) / (R T b (sigma - epsilon))]
                        * ln[(1 + sigma b rho) / (1 + epsilon b rho)]

(For vdW with epsilon = sigma = 0, use the alternative form
    alpha_r = -ln(1 - b rho) - a(T) rho / (R T)
which follows from taking the limit.)

| EOS                           | epsilon  | sigma    | Omega_a   | Omega_b    | alpha_f(T_r, omega) |
|-------------------------------|----------|----------|-----------|------------|---------------------|
| van der Waals (vdW)           | 0        | 0        | 0.421875  | 0.125      | 1                    |
| Redlich-Kwong (RK)            | 0        | 1        | 0.42748   | 0.08664    | 1/sqrt(T_r)          |
| Soave-Redlich-Kwong (SRK)     | 0        | 1        | 0.42748   | 0.08664    | [1+m(1-sqrt(T_r))]^2 |
| Peng-Robinson (PR)            | 1-sqrt2  | 1+sqrt2  | 0.45724   | 0.07780    | [1+m(1-sqrt(T_r))]^2 |

Size parameters scaled to the pure fluid's critical point:
    b   = Omega_b  * R T_c / p_c
    a_c = Omega_a  * R^2 T_c^2 / p_c       (so a(T) = a_c * alpha_f(T_r, omega))

Critical compressibility predicted by the EOS (what pc_c rho_c / T_c is):
    Z_c_EOS = 0.375 for vdW, 1/3 for RK/SRK, 0.30740 for PR.

This library uses the EOS-predicted critical density as the reducing density:
    rho_c_EOS = p_c / (Z_c_EOS * R * T_c)
so the EOS is exactly at its own (pc, Tc, rho_c_EOS) critical point. This is
internally consistent but does NOT match experimental rho_c.

Reference:
  Smith, Van Ness & Abbott, "Introduction to Chemical Engineering Thermodynamics"
  Michelsen & Mollerup, "Thermodynamic Models: Fundamentals and Computational Aspects"
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# EOS family constants
_EOS_FAMILIES = {
    "vdw": {
        "epsilon": 0.0, "sigma": 0.0,
        "Omega_a": 27.0/64.0,        # 0.421875
        "Omega_b": 1.0/8.0,           # 0.125
        "Z_c": 3.0/8.0,               # 0.375
        "alpha_type": "constant",
    },
    "rk": {
        "epsilon": 0.0, "sigma": 1.0,
        "Omega_a": 0.42748,
        "Omega_b": 0.08664,
        "Z_c": 1.0/3.0,
        "alpha_type": "rk",           # 1/sqrt(T_r)
    },
    "srk": {
        "epsilon": 0.0, "sigma": 1.0,
        "Omega_a": 0.42748,
        "Omega_b": 0.08664,
        "Z_c": 1.0/3.0,
        "alpha_type": "soave",
        "m_coeffs": (0.480, 1.574, -0.176),     # m = a0 + a1*omega + a2*omega^2
    },
    "pr": {
        "epsilon": 1.0 - np.sqrt(2.0),
        "sigma":   1.0 + np.sqrt(2.0),
        "Omega_a": 0.45724,
        "Omega_b": 0.07780,
        "Z_c": 0.30740,
        "alpha_type": "soave",
        "m_coeffs": (0.37464, 1.54226, -0.26992),
    },
}


def _soave_alpha(T_r, m):
    """Soave-type alpha function and its two temperature derivatives.

    alpha_f(T_r) = [1 + m(1 - sqrt(T_r))]^2

    Returns (alpha, alpha', alpha'') where primes are d/dT_r.
    """
    sqrt_Tr = np.sqrt(T_r)
    u = 1.0 + m * (1.0 - sqrt_Tr)         # = 1 + m - m*sqrt(Tr)
    alpha = u * u
    du_dTr = -m / (2.0 * sqrt_Tr)
    alpha_p = 2.0 * u * du_dTr            # -m*u/sqrt(Tr)
    d2u_dTr2 = m / (4.0 * T_r * sqrt_Tr)  # d/dTr[-m/(2 sqrt(Tr))] = m/(4 Tr^1.5)
    alpha_pp = 2.0 * du_dTr * du_dTr + 2.0 * u * d2u_dTr2
    return alpha, alpha_p, alpha_pp


def _rk_alpha(T_r):
    """Original Redlich-Kwong: alpha_f = 1/sqrt(T_r)."""
    alpha = 1.0 / np.sqrt(T_r)
    alpha_p = -0.5 / (T_r * np.sqrt(T_r))            # -0.5 * T_r^{-3/2}
    alpha_pp = 0.75 / (T_r * T_r * np.sqrt(T_r))     # 0.75 * T_r^{-5/2}
    return alpha, alpha_p, alpha_pp


def _constant_alpha(T_r):
    """vdW: alpha_f = 1 (no temperature dependence)."""
    return 1.0, 0.0, 0.0


@dataclass
class CubicEOS:
    """A two-parameter cubic equation of state for a pure fluid.

    Parameters
    ----------
    T_c : float             critical temperature [K]
    p_c : float             critical pressure [Pa]
    acentric_factor : float Pitzer's omega (for Soave-type alpha functions)
    family : str            'pr', 'srk', 'rk', 'vdw'
    R : float               gas constant [J/(mol K)]. Default 8.314462618.
    molar_mass : float      [kg/mol]. Optional, for density-unit conversions.
    name : str              display name

    Optional ideal-gas model (for caloric properties h, s, u):
    ---------------------------------------------------------
    ideal_gas_cp_poly : tuple of floats, optional
        Coefficients (a0, a1, a2, ...) for the ideal-gas Cp(T) polynomial
        in J/(mol K): Cp^ig(T) = a0 + a1*T + a2*T^2 + ...
        If None, defaults to Cp^ig = 3.5R (diatomic-like constant; gives
        consistent residual-dominated answers but not absolute accuracy).
    T_ref : float
        Reference temperature [K] at which h_ref and s_ref are defined.
        Default 298.15 K.
    p_ref : float
        Reference pressure [Pa] for s_ref (entropy is density-dependent).
        Default 101325 Pa.
    h_ref : float
        Enthalpy at the reference state (T_ref, p_ref) [J/mol]. Default 0.
    s_ref : float
        Entropy at the reference state (T_ref, p_ref) [J/(mol K)]. Default 0.
    """
    T_c: float
    p_c: float
    acentric_factor: float = 0.0
    family: str = "pr"
    R: float = 8.314462618
    molar_mass: float = 0.0
    name: str = ""
    # Ideal-gas model (optional, for caloric properties)
    ideal_gas_cp_poly: Optional[tuple] = None
    T_ref: float = 298.15
    p_ref: float = 101325.0
    h_ref: float = 0.0
    s_ref: float = 0.0

    def __post_init__(self):
        fam = _EOS_FAMILIES.get(self.family.lower())
        if fam is None:
            raise ValueError(
                f"Unknown cubic EOS family {self.family!r}. "
                f"Choose from {list(_EOS_FAMILIES)}."
            )
        self._fam = fam
        self.epsilon = fam["epsilon"]
        self.sigma = fam["sigma"]
        # Pre-compute size parameters
        self.b = fam["Omega_b"] * self.R * self.T_c / self.p_c
        self.a_c = fam["Omega_a"] * (self.R * self.T_c) ** 2 / self.p_c
        # EOS-predicted critical density
        self.Z_c_EOS = fam["Z_c"]
        self.rho_c = self.p_c / (self.Z_c_EOS * self.R * self.T_c)
        # m coefficient for Soave-type alpha functions
        if fam["alpha_type"] == "soave":
            a0, a1, a2 = fam["m_coeffs"]
            self._m = a0 + a1 * self.acentric_factor + a2 * self.acentric_factor ** 2
        else:
            self._m = None
        # Use the EOS rho_c as the reducing density:
        #   delta = rho / rho_c,     tau = T_c / T
        # With this convention, b * rho_c = Omega_b / Z_c_EOS  -- a constant per family.
        self._b_rhoc = self.b * self.rho_c    # = Omega_b / Z_c_EOS

    # ------------------------------------------------------------------
    # alpha-function
    # ------------------------------------------------------------------
    def alpha_func(self, T_r):
        """Return (alpha, d_alpha/d_Tr, d^2_alpha/d_Tr^2) at reduced temperature T_r."""
        fam = self._fam
        if fam["alpha_type"] == "constant":
            return _constant_alpha(T_r)
        elif fam["alpha_type"] == "rk":
            return _rk_alpha(T_r)
        elif fam["alpha_type"] == "soave":
            return _soave_alpha(T_r, self._m)
        raise ValueError(f"unknown alpha_type {fam['alpha_type']}")

    # ------------------------------------------------------------------
    # a(T) and its tau-derivatives
    # ------------------------------------------------------------------
    def a_T(self, T):
        """Return (a, da/dT, d^2a/dT^2) at temperature T."""
        T_r = T / self.T_c
        alpha, ap, app = self.alpha_func(T_r)
        a = self.a_c * alpha
        # d/dT = (1/Tc) * d/dT_r
        da_dT = self.a_c * ap / self.T_c
        d2a_dT2 = self.a_c * app / (self.T_c * self.T_c)
        return a, da_dT, d2a_dT2

    # ------------------------------------------------------------------
    # Residual Helmholtz alpha_r(delta, tau) and its derivatives
    # ------------------------------------------------------------------
    def alpha_r_derivs(self, delta, tau):
        """Return (alpha_r, a_d, a_t, a_dd, a_tt, a_dt) as functions of (delta, tau).

        Here delta = rho / rho_c_EOS, tau = T_c / T.
        Derivatives are w.r.t. delta and tau (NOT the scaled "log-derivative" form).
        """
        eps = self.epsilon
        sig = self.sigma
        # B = b * rho = (b * rho_c) * delta
        B0 = self._b_rhoc
        B = B0 * delta

        # Log-terms and their B-derivatives
        # psi(B)   = -ln(1 - B)
        # dpsi/dB  = 1/(1 - B)
        # d2psi/dB2= 1/(1 - B)^2
        one_minus_B = 1.0 - B
        psi = -np.log(one_minus_B)
        dpsi_dB = 1.0 / one_minus_B
        d2psi_dB2 = dpsi_dB * dpsi_dB

        if abs(sig - eps) > 1e-14:
            # Normal case
            one_plus_sig_B = 1.0 + sig * B
            one_plus_eps_B = 1.0 + eps * B
            I_B = np.log(one_plus_sig_B / one_plus_eps_B)
            # dI/dB = sigma/(1+sig B) - epsilon/(1+eps B)
            dI_dB = sig / one_plus_sig_B - eps / one_plus_eps_B
            # d2I/dB2 = -sigma^2/(1+sig B)^2 + epsilon^2/(1+eps B)^2
            d2I_dB2 = -sig * sig / (one_plus_sig_B * one_plus_sig_B) + \
                       eps * eps / (one_plus_eps_B * one_plus_eps_B)

            # Temperature part
            T = self.T_c / tau
            a, da_dT, d2a_dT2 = self.a_T(T)
            # q(T) = a(T) / (R T b (sigma - eps))
            denom = self.R * T * self.b * (sig - eps)
            q = a / denom
            # dq/dT: product rule
            #   q = a(T) / [R b (sig-eps) * T]
            #   Let C = R b (sig-eps). Then q = a/(C*T)
            #   dq/dT = da_dT/(C T) - a/(C T^2) = (da_dT * T - a) / (C T^2) = da_dT/(C T) - q/T
            C = self.R * self.b * (sig - eps)
            dq_dT = da_dT / (C * T) - q / T
            d2q_dT2 = d2a_dT2 / (C * T) - 2.0 * da_dT / (C * T * T) + 2.0 * q / (T * T)
            # Convert to tau-derivatives: T = Tc/tau, dT/dtau = -Tc/tau^2 = -T/tau
            #   dq/dtau = dq/dT * dT/dtau = -(T/tau) * dq/dT
            #   d2q/dtau2 = d/dtau[dq/dtau] = d/dtau[-(T/tau) dq/dT]
            #     T depends on tau: d/dtau(-T/tau) = -(-T/tau)/tau - (1/tau)*(dT/dtau)
            #                     = T/tau^2 - (1/tau)*(-T/tau) = 2 T / tau^2
            #     and d(dq/dT)/dtau = d2q/dT2 * dT/dtau = -(T/tau) * d2q/dT2
            #     so d2q/dtau2 = (2 T / tau^2) * dq/dT + (-T/tau) * (-T/tau) * d2q/dT2
            #                  = (2 T / tau^2) * dq/dT + (T/tau)^2 * d2q/dT2
            dq_dtau = -(T / tau) * dq_dT
            d2q_dtau2 = (2.0 * T / (tau * tau)) * dq_dT + (T / tau) * (T / tau) * d2q_dT2

            # alpha_r = psi(B) - q * I(B)
            alpha_r = psi - q * I_B
            # d/d(delta): B = B0 * delta, so d/d(delta) = B0 * d/dB
            a_d = B0 * (dpsi_dB - q * dI_dB)
            a_dd = B0 * B0 * (d2psi_dB2 - q * d2I_dB2)
            # d/d(tau): only q depends on tau, B independent
            a_t = -dq_dtau * I_B
            a_tt = -d2q_dtau2 * I_B
            # Mixed: d/d(tau) of a_d = -B0 * dq_dtau * dI_dB
            a_dt = -B0 * dq_dtau * dI_dB

            return alpha_r, a_d, a_t, a_dd, a_tt, a_dt

        else:
            # Degenerate: epsilon = sigma (vdW). Use direct form.
            # alpha_r = -ln(1 - B) - a rho / (R T)
            # Here "a rho / (R T)" has no log ratio.
            T = self.T_c / tau
            a, da_dT, d2a_dT2 = self.a_T(T)
            # Let's write the attractive part as -a(T) * rho / (R T)
            # In delta: -a(T) * (rho_c * delta) / (R T) = -[a(T) rho_c / (R T)] * delta
            # Compact form: define eta(T) = a(T) * rho_c / (R T)
            eta = a * self.rho_c / (self.R * T)
            # deta/dT: d/dT [a/(RT)] * rho_c = [(da_dT * T - a)/(R T^2)] * rho_c = (da_dT/(R T) - a/(R T^2)) * rho_c
            deta_dT = (da_dT / (self.R * T) - a / (self.R * T * T)) * self.rho_c
            d2eta_dT2 = (d2a_dT2 / (self.R * T)
                        - 2.0 * da_dT / (self.R * T * T)
                        + 2.0 * a / (self.R * T * T * T)) * self.rho_c
            # Convert to tau: dT/dtau = -T/tau
            deta_dtau = -(T / tau) * deta_dT
            d2eta_dtau2 = (2.0 * T / (tau * tau)) * deta_dT + (T / tau) ** 2 * d2eta_dT2

            alpha_r = psi - eta * delta
            a_d = B0 * dpsi_dB - eta
            a_dd = B0 * B0 * d2psi_dB2
            a_t = -deta_dtau * delta
            a_tt = -d2eta_dtau2 * delta
            a_dt = -deta_dtau

            return alpha_r, a_d, a_t, a_dd, a_tt, a_dt

    # ------------------------------------------------------------------
    # Convenience: pressure at (rho, T) via the direct cubic form
    # ------------------------------------------------------------------
    def pressure(self, rho, T):
        """Pressure at (rho, T) from the direct cubic expression.

        Equivalent to p = rho * R * T * (1 + delta * a_d), but useful as an
        independent sanity check.
        """
        a, _, _ = self.a_T(T)
        v = 1.0 / rho
        rep = self.R * T / (v - self.b)
        if abs(self.sigma - self.epsilon) > 1e-14:
            attr = a / ((v + self.epsilon * self.b) * (v + self.sigma * self.b))
        else:
            # vdW degenerate
            attr = a / (v * v)
        return rep - attr

    # ------------------------------------------------------------------
    # Ideal-gas caloric properties (per pure component)
    # ------------------------------------------------------------------
    def _cp_poly_coeffs(self):
        """Return the Cp polynomial coefficients (default: 3.5R = constant)."""
        if self.ideal_gas_cp_poly is None:
            return (3.5 * self.R,)     # ~diatomic default
        return tuple(self.ideal_gas_cp_poly)

    def ideal_gas_h(self, T):
        """Ideal-gas molar enthalpy [J/mol] at temperature T.

        Computed from the Cp(T) polynomial by analytic integration:
            h(T) = h_ref + integral_{T_ref}^{T} Cp(T') dT'
        """
        coeffs = self._cp_poly_coeffs()
        dh = 0.0
        for k, a_k in enumerate(coeffs):
            # integral of a_k T^k dT from T_ref to T = a_k * (T^(k+1) - T_ref^(k+1)) / (k+1)
            dh += a_k * (T ** (k + 1) - self.T_ref ** (k + 1)) / (k + 1)
        return self.h_ref + dh

    def ideal_gas_s(self, T, p):
        """Ideal-gas molar entropy [J/(mol K)] at (T, p).

        s^ig(T, p) = s_ref + integral_{T_ref}^{T} Cp(T')/T' dT'  -  R*ln(p/p_ref)

        The integral for a polynomial Cp:
          a_0 ln(T/T_ref) + sum_{k>=1} a_k/k * (T^k - T_ref^k)
        """
        coeffs = self._cp_poly_coeffs()
        ds_T = 0.0
        for k, a_k in enumerate(coeffs):
            if k == 0:
                ds_T += a_k * np.log(T / self.T_ref)
            else:
                ds_T += a_k * (T ** k - self.T_ref ** k) / k
        ds_p = -self.R * np.log(p / self.p_ref)
        return self.s_ref + ds_T + ds_p
        """Pressure from the cubic at (rho, T), using the direct form.

        This is redundant with p = rho R T (1 + delta * a_d), but is useful
        as an independent check.
        """
        a, _, _ = self.a_T(T)
        v = 1.0 / rho
        rep = self.R * T / (v - self.b)
        attr = a / ((v + self.epsilon * self.b) * (v + self.sigma * self.b))
        return rep - attr

    # ------------------------------------------------------------------
    # Pure-fluid saturation
    # ------------------------------------------------------------------
    def saturation_p(self, T, p_init=None, tol=1e-9, maxiter=80):
        """Pure-fluid saturation pressure at temperature T < T_c.

        Uses the classic two-phase fugacity-equality iteration. At each trial
        pressure, solve the cubic in Z for liquid and vapor roots; equating
        f_L = f_V (i.e., ln phi_L = ln phi_V for a pure fluid) gives the
        update

            p_new = p * exp(ln phi_L - ln phi_V)

        Converges quadratically near the answer. Falls back to bisection-like
        nudging if the cubic produces only one real root.
        """
        if T >= self.T_c:
            raise ValueError(f"T={T} is not below T_c={self.T_c}; no saturation.")

        eps_ = self.epsilon
        sig = self.sigma
        u = eps_ + sig
        w2 = eps_ * sig

        # Initial guess from Wilson K-factor form (empirically good)
        if p_init is None:
            p_init = self.p_c * np.exp(
                5.373 * (1.0 + self.acentric_factor) * (1.0 - self.T_c / T)
            )
        p = max(p_init, 1e3)
        a_T, _, _ = self.a_T(T)

        for _ in range(maxiter):
            A = a_T * p / (self.R * T) ** 2
            B = self.b * p / (self.R * T)
            c2 = -(1.0 + B - u * B)
            c1 = A + (w2 - u) * B * B - u * B
            c0 = -(w2 * B ** 3 + w2 * B ** 2 + A * B)
            roots = np.roots([1.0, c2, c1, c0])
            real = sorted(
                r.real for r in roots
                if abs(r.imag) < 1e-10 and r.real > B + 1e-14
            )
            if len(real) < 2:
                # Single-root regime: nudge p inward. If we're above sat, single
                # root is vapor-like at low p or liquid-like at high p. Use a
                # simple heuristic: if only root has Z >> B, we're above vapor
                # pressure -> decrease p; if Z ~ B, increase p.
                Z_single = real[0] if real else 1.0
                if Z_single > 0.5:
                    p *= 0.9    # vapor-like single root at high p: lower p
                else:
                    p *= 1.1
                continue

            Z_L = real[0]
            Z_V = real[-1]
            # Pure-fluid ln phi:
            #   ln phi = Z - 1 - ln(Z - B) + A/(B*(sig-eps)) * ln[(Z+eps B)/(Z+sig B)]
            if abs(sig - eps_) > 1e-14:
                lnphi_L = (
                    Z_L - 1.0 - np.log(Z_L - B)
                    + A / (B * (sig - eps_))
                    * np.log((Z_L + eps_ * B) / (Z_L + sig * B))
                )
                lnphi_V = (
                    Z_V - 1.0 - np.log(Z_V - B)
                    + A / (B * (sig - eps_))
                    * np.log((Z_V + eps_ * B) / (Z_V + sig * B))
                )
            else:
                # vdW degenerate
                lnphi_L = Z_L - 1.0 - np.log(Z_L - B) - A / Z_L
                lnphi_V = Z_V - 1.0 - np.log(Z_V - B) - A / Z_V

            resid = lnphi_L - lnphi_V
            if abs(resid) < tol:
                return p

            # Update: p_new = p * exp(resid) with damping if change is large
            step = resid
            if abs(step) > 0.5:
                step = 0.5 * np.sign(step)
            p = p * np.exp(step)

        raise RuntimeError(
            f"saturation_p did not converge: T={T}, final p={p*1e-6:.4f} MPa"
        )

    # ------------------------------------------------------------------
    # Pure-fluid density from (p, T) -- solve the cubic in Z
    # ------------------------------------------------------------------
    def density_from_pressure(self, p, T, phase_hint="vapor"):
        """Solve the cubic for rho at (p, T) on the requested phase branch.

        Returns rho (mol/m^3).

        Convention: smallest Z => densest phase (liquid), largest Z => vapor.
        If 3 real roots exist, the middle one is thermodynamically unstable
        and is discarded.
        """
        a, _, _ = self.a_T(T)
        A = a * p / (self.R * T) ** 2
        B = self.b * p / (self.R * T)
        eps_ = self.epsilon; sig = self.sigma
        u = eps_ + sig; w2 = eps_ * sig
        c2 = -(1.0 + B - u * B)
        c1 = A + (w2 - u) * B * B - u * B
        c0 = -(w2 * B ** 3 + w2 * B ** 2 + A * B)
        roots = np.roots([1.0, c2, c1, c0])
        real = sorted([r.real for r in roots if abs(r.imag) < 1e-10 and r.real > B + 1e-14])
        if not real:
            raise RuntimeError(f"No Z > B roots at T={T}, p={p}")
        if len(real) == 3:
            real = [real[0], real[-1]]
        if len(real) == 1:
            Z = real[0]
        else:
            Z = real[-1] if phase_hint == "vapor" else real[0]
        return p / (Z * self.R * T)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def PR(T_c, p_c, acentric_factor=0.0, **kw):
    """Peng-Robinson cubic EOS for a pure fluid."""
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor, family="pr", **kw)


def SRK(T_c, p_c, acentric_factor=0.0, **kw):
    """Soave-Redlich-Kwong cubic EOS."""
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor, family="srk", **kw)


def RK(T_c, p_c, **kw):
    """Original Redlich-Kwong cubic EOS (no acentric factor needed)."""
    return CubicEOS(T_c=T_c, p_c=p_c, family="rk", **kw)


def VDW(T_c, p_c, **kw):
    """Van der Waals cubic EOS."""
    return CubicEOS(T_c=T_c, p_c=p_c, family="vdw", **kw)
