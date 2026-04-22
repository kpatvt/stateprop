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
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Fast analytic cubic solver (Cardano + trigonometric form).
# Used in every density_from_pressure call -- about 28x faster than
# numpy.roots for depressed cubics of the form Z^3 + bZ^2 + cZ + d = 0.
# Pure-Python arithmetic, no external dependencies.
# --------------------------------------------------------------------------

def _cubic_real_roots(b, c, d):
    """Return a sorted list of all real roots of Z^3 + b Z^2 + c Z + d = 0.

    Uses the substitution Z = t - b/3 to reduce to a depressed cubic
    t^3 + p t + q = 0, then Cardano (one real root) or the trigonometric
    form (three real roots). Matches numpy.roots to machine precision
    while running ~30x faster.
    """
    # Depressed cubic coefficients
    p = c - b * b / 3.0
    q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d
    shift = -b / 3.0
    # Discriminant-related quantity: D2 = (q/2)^2 + (p/3)^3
    # D2 > 0 : one real root.  D2 < 0 : three distinct real roots.
    # D2 = 0 : a multiple root.
    D2 = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if D2 > 0:
        sqrtD2 = math.sqrt(D2)
        u_cubed = -q / 2.0 + sqrtD2
        v_cubed = -q / 2.0 - sqrtD2
        u = math.copysign(abs(u_cubed) ** (1.0 / 3.0), u_cubed)
        v = math.copysign(abs(v_cubed) ** (1.0 / 3.0), v_cubed)
        return [u + v + shift]
    if D2 < 0:
        r = math.sqrt(-p / 3.0)
        cos_arg = -q / (2.0 * r * r * r)
        # Clamp for numerical safety (cos_arg should be in [-1, 1])
        if cos_arg > 1.0:
            cos_arg = 1.0
        elif cos_arg < -1.0:
            cos_arg = -1.0
        theta = math.acos(cos_arg) / 3.0
        TWO_PI_3 = 2.0 * math.pi / 3.0
        roots = [
            2.0 * r * math.cos(theta) + shift,
            2.0 * r * math.cos(theta - TWO_PI_3) + shift,
            2.0 * r * math.cos(theta - 2.0 * TWO_PI_3) + shift,
        ]
        return sorted(roots)
    # D2 == 0: multiple roots
    if abs(q) < 1e-30:
        return [shift]     # triple root
    u = math.copysign(abs(q / 2.0) ** (1.0 / 3.0), -q / 2.0)
    return sorted([2.0 * u + shift, -u + shift, -u + shift])


# EOS family constants
#
# The `m_coeffs` entry lists the polynomial coefficients (a0, a1, a2, ...) such
# that  m(omega) = a0 + a1*omega + a2*omega^2 + ... (any length). SRK uses a
# 3-term polynomial; PR-1976 uses a 3-term polynomial; PR-1978 uses a 4-term
# polynomial tuned for heavy components (omega > 0.49).
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
        # PR-1976 (original Peng-Robinson, 1976). Accurate for omega < ~0.49.
        "epsilon": 1.0 - np.sqrt(2.0),
        "sigma":   1.0 + np.sqrt(2.0),
        "Omega_a": 0.45724,
        "Omega_b": 0.07780,
        "Z_c": 0.30740,
        "alpha_type": "soave",
        "m_coeffs": (0.37464, 1.54226, -0.26992),
    },
    "pr78": {
        # PR-1978 (Peng & Robinson's 1978 revision). Tuned for heavy fractions
        # (omega > 0.49). Adds a cubic term to the m(omega) polynomial:
        #   m = 0.379642 + 1.48503 * omega - 0.164423 * omega^2 + 0.016666 * omega^3
        # Reference: Peng & Robinson, in "The Characterization of the Heptanes
        # and Heavier Fractions for the GPA Peng-Robinson Programs" (GPA
        # Research Report RR-28, 1978).
        "epsilon": 1.0 - np.sqrt(2.0),
        "sigma":   1.0 + np.sqrt(2.0),
        "Omega_a": 0.45724,
        "Omega_b": 0.07780,
        "Z_c": 0.30740,
        "alpha_type": "soave",
        "m_coeffs": (0.379642, 1.48503, -0.164423, 0.016666),
    },
}


# Threshold above which PR-1976's m(omega) polynomial becomes unreliable.
# PR-1978 was fit specifically to improve accuracy above this value.
PR78_OMEGA_THRESHOLD = 0.49


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


def _mathias_copeman_alpha(T_r, c1, c2, c3):
    """Mathias-Copeman (1983) alpha function with three component-specific
    coefficients. Piecewise: full cubic below T_r = 1, reduces to Soave form
    above critical.

    Below T_r = 1:
        alpha = [1 + c1*(1-sqrt(Tr)) + c2*(1-sqrt(Tr))^2 + c3*(1-sqrt(Tr))^3]^2
    Above T_r = 1:
        alpha = [1 + c1*(1-sqrt(Tr))]^2     (Soave form, c2 and c3 dropped)

    The piecewise definition is the standard way to avoid pathological
    behavior where the cubic extrapolation can go negative or blow up at
    high T_r. Both branches agree at T_r = 1 (where 1-sqrt(Tr)=0).

    Reference: Mathias & Copeman, "Extension of the Peng-Robinson equation
    of state to complex mixtures: Evaluation of the various forms of the
    local composition concept", Fluid Phase Equil. 13 (1983) 91.

    Returns (alpha, alpha_p, alpha_pp) with primes = d/dT_r.
    """
    sqrt_Tr = np.sqrt(T_r)
    s = 1.0 - sqrt_Tr
    ds_dTr = -0.5 / sqrt_Tr                      # d/dT_r (1-sqrt(Tr))
    d2s_dTr2 = 0.25 / (T_r * sqrt_Tr)            # = 1/(4 T_r^{3/2})

    if T_r <= 1.0:
        # Full cubic in s
        u = 1.0 + c1 * s + c2 * s * s + c3 * s * s * s
        du_ds = c1 + 2.0 * c2 * s + 3.0 * c3 * s * s
        d2u_ds2 = 2.0 * c2 + 6.0 * c3 * s
    else:
        # Soave form: drop c2, c3
        u = 1.0 + c1 * s
        du_ds = c1
        d2u_ds2 = 0.0

    du_dTr = du_ds * ds_dTr
    d2u_dTr2 = d2u_ds2 * ds_dTr * ds_dTr + du_ds * d2s_dTr2

    alpha = u * u
    alpha_p = 2.0 * u * du_dTr
    alpha_pp = 2.0 * du_dTr * du_dTr + 2.0 * u * d2u_dTr2
    return alpha, alpha_p, alpha_pp


def _twu_alpha(T_r, L, M, N):
    """Twu-Coon-Cunningham (1995) alpha function.

        alpha(T_r) = T_r^{N(M-1)} * exp[L * (1 - T_r^{N*M})]

    Three component-specific parameters (L, M, N). Known to handle polar
    and associating fluids well, and extrapolates sensibly at both low and
    high T_r. Parameters differ between PR and SRK (they're fit to each
    base EOS separately).

    Reference: Twu, Coon & Cunningham, "A new generalized alpha function
    for a cubic equation of state. Part 1. Peng-Robinson equation",
    Fluid Phase Equil. 105 (1995) 49.

    Returns (alpha, alpha_p, alpha_pp) with primes = d/dT_r.
    """
    # Use log form for derivative identities:
    #   ln alpha = N(M-1) ln(T_r) + L*(1 - T_r^{NM})
    # and alpha'/alpha = (ln alpha)', alpha''/alpha = (ln alpha)'^2 + (ln alpha)''.
    NM = N * M
    Nm1 = N * (M - 1.0)
    Tr_NM = T_r ** NM           # T_r^{NM}
    alpha = (T_r ** Nm1) * np.exp(L * (1.0 - Tr_NM))

    # (ln alpha)' = Nm1 / T_r - L*NM*T_r^{NM-1}
    dlna = Nm1 / T_r - L * NM * (T_r ** (NM - 1.0))
    # (ln alpha)'' = -Nm1 / T_r^2 - L*NM*(NM-1)*T_r^{NM-2}
    d2lna = -Nm1 / (T_r * T_r) - L * NM * (NM - 1.0) * (T_r ** (NM - 2.0))

    alpha_p = alpha * dlna
    alpha_pp = alpha * (dlna * dlna + d2lna)
    return alpha, alpha_p, alpha_pp


# PRSV kappa_0 coefficients (Stryjek-Vera 1986; refinement of PR's m(omega)):
# kappa_0(omega) = 0.378893 + 1.4897153 omega - 0.17131848 omega^2 + 0.0196554 omega^3
_PRSV_KAPPA0_COEFFS = (0.378893, 1.4897153, -0.17131848, 0.0196554)


def _prsv_kappa_and_derivs(T_r, kappa0, kappa1):
    """Return kappa(T_r), dkappa/dT_r, d2kappa/dT_r^2 for PRSV (Stryjek-Vera).

    kappa(T_r) = kappa0 + kappa1 * (1 + sqrt(T_r)) * (0.7 - T_r)

    kappa0 depends only on omega (evaluated once at construction time);
    kappa1 is the component-specific tuning parameter (often from PRSV
    tables; defaults to 0 when not provided, which recovers a modified-PR
    without the polar correction).
    """
    sqrt_Tr = np.sqrt(T_r)
    # kappa = kappa0 + kappa1 * (1 + sqrt_Tr) * (0.7 - T_r)
    # Let g(T_r) = (1 + sqrt_Tr) * (0.7 - T_r). Expand derivatives.
    g = (1.0 + sqrt_Tr) * (0.7 - T_r)
    # dg/dT_r = d[(1+sqrt_Tr)]/dT_r * (0.7 - T_r) + (1+sqrt_Tr) * (-1)
    #         = (1/(2 sqrt_Tr))(0.7 - T_r) - (1 + sqrt_Tr)
    dg_dTr = (0.7 - T_r) / (2.0 * sqrt_Tr) - (1.0 + sqrt_Tr)
    # d2g/dT_r^2:
    # first term: d/dT_r [(0.7 - T_r)/(2 sqrt_Tr)]
    #   = [-1/(2 sqrt_Tr) * 2 sqrt_Tr - (0.7 - T_r) * (1/sqrt_Tr)] / (4 T_r)
    # Actually simpler: write  A(T_r) = (0.7 - T_r) / (2 sqrt_Tr)
    # = 0.5 * (0.7 * T_r^{-1/2} - T_r^{1/2})
    # so dA/dT_r = 0.5 * (-0.35 T_r^{-3/2} - 0.5 T_r^{-1/2})
    #            = -0.175 / (T_r * sqrt_Tr) - 0.25 / sqrt_Tr
    dA_dTr = -0.175 / (T_r * sqrt_Tr) - 0.25 / sqrt_Tr
    # second term: d/dT_r [-(1 + sqrt_Tr)] = -1/(2 sqrt_Tr)
    d2g_dTr2 = dA_dTr - 0.5 / sqrt_Tr

    kappa = kappa0 + kappa1 * g
    dkappa_dTr = kappa1 * dg_dTr
    d2kappa_dTr2 = kappa1 * d2g_dTr2
    return kappa, dkappa_dTr, d2kappa_dTr2


def _prsv_alpha(T_r, kappa0, kappa1):
    """PRSV (Stryjek-Vera 1986) alpha function.

        alpha(T_r) = [1 + kappa(T_r) * (1 - sqrt(T_r))]^2
        kappa(T_r) = kappa0(omega) + kappa1 * (1 + sqrt(T_r)) * (0.7 - T_r)

    Needs kappa0 (polynomial in omega; computed from the standard PRSV
    formula at construction time) and kappa1 (component-specific, from
    PRSV tables; commonly 0 for nonpolar fluids).

    Reference: Stryjek & Vera, "PRSV: An improved Peng-Robinson equation
    of state for pure compounds and mixtures", Can. J. Chem. Eng. 64
    (1986) 323.

    Returns (alpha, alpha_p, alpha_pp) with primes = d/dT_r.
    """
    sqrt_Tr = np.sqrt(T_r)
    s = 1.0 - sqrt_Tr
    ds_dTr = -0.5 / sqrt_Tr
    d2s_dTr2 = 0.25 / (T_r * sqrt_Tr)

    kappa, dkappa_dTr, d2kappa_dTr2 = _prsv_kappa_and_derivs(T_r, kappa0, kappa1)

    # u = 1 + kappa * s
    u = 1.0 + kappa * s
    # du/dT_r = dkappa/dT_r * s + kappa * ds/dT_r
    du_dTr = dkappa_dTr * s + kappa * ds_dTr
    # d2u/dT_r^2 = d2kappa/dT_r^2 * s + 2 dkappa/dT_r * ds/dT_r + kappa * d2s/dT_r^2
    d2u_dTr2 = (d2kappa_dTr2 * s
                + 2.0 * dkappa_dTr * ds_dTr
                + kappa * d2s_dTr2)

    alpha = u * u
    alpha_p = 2.0 * u * du_dTr
    alpha_pp = 2.0 * du_dTr * du_dTr + 2.0 * u * d2u_dTr2
    return alpha, alpha_p, alpha_pp


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

    PR-specific:
    ------------
    use_pr78 : str
        Controls the PR-1976 vs PR-1978 switch when family='pr'. One of:
        - 'auto'   (default): use PR-1978 if acentric_factor > 0.49, else
                   PR-1976. This matches the industrial convention.
        - 'always': always use PR-1978, regardless of acentric factor.
        - 'never':  always use PR-1976, regardless of acentric factor.
        Has no effect on non-PR families. To select PR-1978 explicitly
        via the family string, pass family='pr78' instead.

    Alpha-function override (advanced):
    -----------------------------------
    alpha_override : tuple, optional
        Replace the family's default alpha function with a variant. Format:
            ('mathias_copeman', c1, c2, c3)
                Mathias-Copeman (1983). c1 may be None to default to the
                family's m(omega). Commonly c1 = m(omega) with c2, c3 being
                small component-specific tuning parameters.
            ('twu', L, M, N)
                Twu-Coon-Cunningham (1995). Note: acentric_factor is NOT
                used -- (L, M, N) are the component-specific parameters.
            ('prsv', kappa1)
                Stryjek-Vera PRSV (1986). Uses the PRSV kappa0(omega)
                polynomial plus the user-supplied kappa1 tuning parameter.
                Pass kappa1=0 for PRSV without the polar correction.
        When alpha_override is used, the family's (epsilon, sigma, Omega_a,
        Omega_b) are still used for the size / attractive constants; only
        the T-dependence of a(T) is changed.

    Volume translation (Peneloux-style):
    ------------------------------------
    volume_shift_c : float, str, or None
        Peneloux-style volume shift parameter [m^3/mol]. When non-zero,
        the molar volume reported externally is v = v_cubic - c (so the
        cubic internally operates on v_cubic = v_real + c). This does NOT
        affect phase equilibria (vapor pressures, fugacity coefficient
        RATIOS, K-factors) but substantially improves liquid-phase density
        predictions.
        - None (default): no shift, pure cubic behavior.
        - float: user-supplied c value in m^3/mol.
        - 'peneloux': auto-compute for SRK only, from the original
          Peneloux et al. (1982) formula with Yamada-Gunn's estimate
          Z_RA ~ 0.29056 - 0.08775*omega. NOT available for PR (no
          single simple one-parameter correlation exists for PR; use
          published per-compound c values or Jhaveri-Youngren 1988).
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
    # PR-specific: controls PR-1976 vs PR-1978 dispatch (see class docstring)
    use_pr78: str = "auto"
    # Alpha-function override: replace the family default with a variant.
    alpha_override: Optional[tuple] = None
    # Volume translation (Peneloux-style)
    volume_shift_c: object = None     # None | float | 'peneloux'

    def __post_init__(self):
        # Resolve 'auto' PR / PR78 dispatch: if family=='pr' and omega > threshold,
        # transparently switch to 'pr78'. This matches industrial convention and
        # is invisible to the user. Explicit family='pr78' always uses PR-1978;
        # explicit family='pr' with use_pr78=='never' keeps PR-1976.
        fam_name = self.family.lower()
        use_pr78 = getattr(self, "use_pr78", "auto")
        if fam_name == "pr" and use_pr78 != "never":
            if use_pr78 == "always" or (
                use_pr78 == "auto"
                and self.acentric_factor > PR78_OMEGA_THRESHOLD
            ):
                fam_name = "pr78"
        self._effective_family = fam_name

        fam = _EOS_FAMILIES.get(fam_name)
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
        # m coefficient for Soave-type alpha functions: polynomial in omega
        # of arbitrary length, evaluated as m = sum_k a_k * omega^k.
        if fam["alpha_type"] == "soave":
            coeffs = fam["m_coeffs"]
            m = 0.0
            omega_k = 1.0
            for a_k in coeffs:
                m += a_k * omega_k
                omega_k *= self.acentric_factor
            self._m = m
        else:
            self._m = None
        # Use the EOS rho_c as the reducing density:
        #   delta = rho / rho_c,     tau = T_c / T
        # With this convention, b * rho_c = Omega_b / Z_c_EOS  -- a constant per family.
        self._b_rhoc = self.b * self.rho_c    # = Omega_b / Z_c_EOS

        # Parse alpha_override (if any) -- stash the resolved parameters so
        # alpha_func can dispatch on them.
        self._alpha_kind = None
        self._alpha_params = None
        if self.alpha_override is not None:
            ov = tuple(self.alpha_override)
            if len(ov) < 1 or not isinstance(ov[0], str):
                raise ValueError(
                    "alpha_override must be a tuple starting with a string "
                    "type name, e.g. ('mathias_copeman', c1, c2, c3)."
                )
            kind = ov[0].lower()
            rest = ov[1:]
            if kind == "mathias_copeman":
                if len(rest) != 3:
                    raise ValueError(
                        "mathias_copeman alpha_override needs (c1, c2, c3); "
                        "pass c1=None to default to the family's m(omega)."
                    )
                c1, c2, c3 = rest
                if c1 is None:
                    if self._m is None:
                        raise ValueError(
                            "c1=None defaults to m(omega), but this EOS family "
                            "is not Soave-type; pass c1 explicitly."
                        )
                    c1 = self._m
                self._alpha_kind = "mathias_copeman"
                self._alpha_params = (float(c1), float(c2), float(c3))
            elif kind == "twu":
                if len(rest) != 3:
                    raise ValueError("twu alpha_override needs (L, M, N)")
                L, M, N = rest
                self._alpha_kind = "twu"
                self._alpha_params = (float(L), float(M), float(N))
            elif kind == "prsv":
                if len(rest) != 1:
                    raise ValueError("prsv alpha_override needs (kappa1,)")
                (kappa1,) = rest
                # PRSV kappa0 polynomial in omega
                kappa0 = 0.0
                w = 1.0
                for c in _PRSV_KAPPA0_COEFFS:
                    kappa0 += c * w
                    w *= self.acentric_factor
                self._alpha_kind = "prsv"
                self._alpha_params = (float(kappa0), float(kappa1))
            else:
                raise ValueError(
                    f"Unknown alpha_override type {kind!r}. Choose from "
                    f"'mathias_copeman', 'twu', 'prsv'."
                )

        # -----------------------------------------------------------
        # Resolve volume_shift_c -> self.c (a numeric m^3/mol value)
        # -----------------------------------------------------------
        vs = self.volume_shift_c
        if vs is None:
            self.c = 0.0
        elif isinstance(vs, str):
            if vs.lower() != "peneloux":
                raise ValueError(
                    f"Unknown volume_shift_c string {vs!r}. Use 'peneloux' or "
                    f"pass a numeric c value."
                )
            fam_key = self._effective_family
            if fam_key in ("srk", "rk"):
                # Peneloux 1982 for SRK: c = 0.40768 * R*Tc/Pc * (0.29441 - Z_RA)
                # with Z_RA estimated from Yamada-Gunn: Z_RA ~ 0.29056 - 0.08775*omega
                Z_RA = 0.29056 - 0.08775 * self.acentric_factor
                self.c = (0.40768 * self.R * self.T_c / self.p_c
                          * (0.29441 - Z_RA))
            else:
                raise ValueError(
                    f"volume_shift_c='peneloux' is only defined for SRK "
                    f"(Peneloux et al. 1982). For PR no simple one-parameter "
                    f"auto-correlation exists; published correlations (e.g. "
                    f"Jhaveri-Youngren 1988) require additional inputs like "
                    f"molecular weight and paraffinicity. Pass a numeric c "
                    f"value (in m^3/mol) instead. Got family={fam_key!r}."
                )
        else:
            self.c = float(vs)

    # ------------------------------------------------------------------
    # alpha-function
    # ------------------------------------------------------------------
    def alpha_func(self, T_r):
        """Return (alpha, d_alpha/d_Tr, d^2_alpha/d_Tr^2) at reduced temperature T_r."""
        # If an alpha_override is active, dispatch on it first
        if self._alpha_kind is not None:
            if self._alpha_kind == "mathias_copeman":
                c1, c2, c3 = self._alpha_params
                return _mathias_copeman_alpha(T_r, c1, c2, c3)
            elif self._alpha_kind == "twu":
                L, M, N = self._alpha_params
                return _twu_alpha(T_r, L, M, N)
            elif self._alpha_kind == "prsv":
                k0, k1 = self._alpha_params
                return _prsv_alpha(T_r, k0, k1)
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

        Equivalent to p = rho * R * T * (1 + delta * a_d) for an untranslated
        cubic. When self.c != 0 (Peneloux-style volume translation active),
        the user-visible density rho corresponds to the "real" molar volume
        v = 1/rho, and the cubic internally operates on v_cubic = v + c.
        The pressure returned is p(v, T) = p^cubic(v + c, T).
        """
        a, _, _ = self.a_T(T)
        v_real = 1.0 / rho
        v = v_real + self.c         # volume used inside the cubic
        rep = self.R * T / (v - self.b)
        if abs(self.sigma - self.epsilon) > 1e-14:
            attr = a / ((v + self.epsilon * self.b) * (v + self.sigma * self.b))
        else:
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
            roots = _cubic_real_roots(c2, c1, c0)
            real = sorted(r for r in roots if r > B + 1e-14)
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
        roots = _cubic_real_roots(c2, c1, c0)
        real = sorted(r for r in roots if r > B + 1e-14)
        if not real:
            raise RuntimeError(f"No Z > B roots at T={T}, p={p}")
        if len(real) == 3:
            real = [real[0], real[-1]]
        if len(real) == 1:
            Z = real[0]
        else:
            Z = real[-1] if phase_hint == "vapor" else real[0]
        if self.c == 0.0:
            return p / (Z * self.R * T)
        # Volume translation active
        v_cubic = Z * self.R * T / p
        v_real = v_cubic - self.c
        if v_real <= 0:
            raise RuntimeError(
                f"volume-translated v_real={v_real} <= 0 at T={T}, p={p}; "
                f"c={self.c} may be too large for this state."
            )
        return 1.0 / v_real


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def PR(T_c, p_c, acentric_factor=0.0, **kw):
    """Peng-Robinson cubic EOS for a pure fluid.

    By default (use_pr78='auto'), switches to PR-1978 for heavy components
    (acentric_factor > 0.49). Pass use_pr78='never' to force PR-1976 or
    use_pr78='always' to force PR-1978.
    """
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor, family="pr", **kw)


def PR78(T_c, p_c, acentric_factor=0.0, **kw):
    """Peng-Robinson-1978 cubic EOS for a pure fluid.

    Uses the 1978 m(omega) polynomial unconditionally, even for light
    components where PR-1976 would also work. Preferred when mixing with
    heavy components and wanting a single consistent alpha function.
    """
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor, family="pr78", **kw)


def SRK(T_c, p_c, acentric_factor=0.0, **kw):
    """Soave-Redlich-Kwong cubic EOS."""
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor, family="srk", **kw)


def RK(T_c, p_c, **kw):
    """Original Redlich-Kwong cubic EOS (no acentric factor needed)."""
    return CubicEOS(T_c=T_c, p_c=p_c, family="rk", **kw)


def VDW(T_c, p_c, **kw):
    """Van der Waals cubic EOS."""
    return CubicEOS(T_c=T_c, p_c=p_c, family="vdw", **kw)


def PR_MC(T_c, p_c, c1, c2, c3, acentric_factor=0.0, **kw):
    """Peng-Robinson with Mathias-Copeman alpha function.

    c1, c2, c3 are the three Mathias-Copeman parameters fit to vapor
    pressure data for the specific compound. Pass c1=None to default to
    m(omega) from the classical PR correlation (so c2, c3 act as small
    corrections).

    Use for polar or highly non-ideal fluids (water, methanol, etc.)
    where classical PR under- or over-predicts vapor pressures.
    """
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor,
                    family="pr", use_pr78="never",
                    alpha_override=("mathias_copeman", c1, c2, c3), **kw)


def SRK_MC(T_c, p_c, c1, c2, c3, acentric_factor=0.0, **kw):
    """Soave-Redlich-Kwong with Mathias-Copeman alpha function."""
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor,
                    family="srk",
                    alpha_override=("mathias_copeman", c1, c2, c3), **kw)


def PR_Twu(T_c, p_c, L, M, N, **kw):
    """Peng-Robinson with Twu-Coon-Cunningham alpha function.

    Twu alpha: alpha(T_r) = T_r^{N(M-1)} * exp[L * (1 - T_r^{NM})]

    The (L, M, N) parameters are component-specific and fitted to vapor
    pressure data. Note that acentric_factor is NOT used with this alpha
    function, so it defaults to 0.

    For a given compound, PR-Twu and SRK-Twu parameters are different
    (they're fit to each base EOS separately).
    """
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=0.0,
                    family="pr", use_pr78="never",
                    alpha_override=("twu", L, M, N), **kw)


def SRK_Twu(T_c, p_c, L, M, N, **kw):
    """Soave-Redlich-Kwong with Twu-Coon-Cunningham alpha function."""
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=0.0,
                    family="srk",
                    alpha_override=("twu", L, M, N), **kw)


def PRSV(T_c, p_c, acentric_factor, kappa1=0.0, **kw):
    """Stryjek-Vera PRSV equation of state (modified Peng-Robinson).

    Uses a refined kappa0(omega) polynomial plus a component-specific
    kappa1 polar correction. For nonpolar components kappa1 = 0 recovers
    a slightly improved version of PR. For polar fluids (water, alcohols)
    nonzero kappa1 substantially improves vapor pressure accuracy.

    kappa1 values are tabulated in Stryjek & Vera (1986) for 90 fluids
    (water: kappa1 = -0.06635, methanol: 0.1629, etc.). Omitting the
    argument (kappa1 = 0) gives the non-polar PRSV form.
    """
    return CubicEOS(T_c=T_c, p_c=p_c, acentric_factor=acentric_factor,
                    family="pr", use_pr78="never",
                    alpha_override=("prsv", kappa1), **kw)
