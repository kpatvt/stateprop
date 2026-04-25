"""Chung-Lee-Starling transport-property correlations.

Chung, T.-H., M. Ajlan, L.L. Lee, K.E. Starling, "Generalized Multiparameter
Correlation for Nonpolar and Polar Fluid Transport Properties", Ind. Eng.
Chem. Res. 1988, 27, 671-679.

Also referenced: Reid, R.C., J.M. Prausnitz, B.E. Poling, "The Properties
of Gases and Liquids", 4th ed. (1987), chapters 9 (viscosity) and 10
(thermal conductivity).

The Chung correlation is a corresponding-states method that uses the
critical constants (T_c, V_c), acentric factor, dipole moment, and a
kappa polar-correction factor (for strong-associating fluids) plus the
molar mass. Pure-fluid accuracy is typically 5-10%. Mixtures use
Chung's own mixing rules for the pseudo-component properties.

Units throughout:
    T [K], rho_mol [mol/m^3], M [kg/mol]
    dipole_moment [Debye]  (1 D = 3.33564e-30 C.m)
    viscosity eta [Pa.s]
    thermal conductivity lambda [W/(m.K)]
"""

from __future__ import annotations

from typing import Sequence, Optional
import numpy as np


# -------------------------------------------------------------------------
# Collision integral Omega^{(2,2)*} (Neufeld-Janzen-Aziz 1972)
# -------------------------------------------------------------------------

def _omega_22(T_star: float) -> float:
    """Collision integral for viscosity / thermal conductivity,
    Neufeld-Janzen-Aziz 1972 correlation.

    T_star = k T / eps (reduced temperature).
    Valid for T_star in [0.3, 100].
    """
    A = 1.16145
    B = 0.14874
    C = 0.52487
    D = 0.77320
    E = 2.16178
    F = 2.43787
    return (A * T_star ** (-B)
            + C * np.exp(-D * T_star)
            + E * np.exp(-F * T_star))


# -------------------------------------------------------------------------
# Dense-fluid E_k coefficients (Chung 1988 Table I)
# -------------------------------------------------------------------------

# Viscosity-correction polynomial coefficients (eta / eta0 terms).
# Each row: [a0, a1, a2, a3] for E_k = a0 + a1*omega + a2*mu_r^4 + a3*kappa
_E_COEF_VISC = np.array([
    [ 6.324,    50.412,  -51.680,  1189.0  ],
    [ 1.210e-3, -1.154e-3, -6.257e-3, 0.03728],
    [ 5.283,    254.209, -168.48,  3898.0  ],
    [ 6.623,    38.096,  -8.464,   31.42   ],
    [19.745,   7.630,   -14.354,  31.53   ],
    [-1.900,   -12.537,   4.985,  -18.15   ],
    [24.275,    3.450,   -11.291,  69.35   ],
    [ 0.7972,   1.117,    0.01235, -4.117  ],
    [-0.2382,   0.06770,  -0.8163, 4.025   ],
    [ 0.06863,  0.3479,   0.5926, -0.727  ],
], dtype=np.float64)

# Thermal conductivity B_k coefficients (Chung 1988 Table IV)
_E_COEF_TCOND = np.array([
    [ 2.4166,    0.74824,  -0.91858,   121.72],
    [-0.50924, -1.5094,    -49.991,    69.983],
    [ 6.6107,   5.6207,     64.760,    27.039],
    [14.543,   -8.9139,    -5.6379,    74.464],
    [ 0.79274,  0.82019,   -0.69369,    6.3173],
    [-5.8634,   12.801,     9.5893,    65.529],
    [91.089,    128.11,    -54.217,   523.81],
], dtype=np.float64)


# -------------------------------------------------------------------------
# Reduced dipole moment (dimensionless)
# -------------------------------------------------------------------------

def _mu_r(dipole_D: float, V_c_m3_per_mol: float, T_c: float) -> float:
    """Dimensionless dipole moment for Chung correlation.

    mu_r = 131.3 * dipole[D] / sqrt(V_c[cm^3/mol] * T_c[K])

    (Chung 1988 Eq. 2; units work out so mu_r is dimensionless.)
    """
    if dipole_D <= 0.0:
        return 0.0
    V_c_cm3 = V_c_m3_per_mol * 1e6
    return 131.3 * dipole_D / np.sqrt(V_c_cm3 * T_c)


def _V_c_estimate(T_c: float, p_c: float, omega: float) -> float:
    """Estimate critical molar volume [m^3/mol] via modified Rackett
    (Reid-Prausnitz-Poling chapter 3):
        Z_c ~= 0.291 - 0.08 omega,  V_c = Z_c R T_c / p_c
    """
    R = 8.314462618
    Z_c = 0.291 - 0.080 * omega
    return Z_c * R * T_c / p_c


def _component_properties(comp):
    """Extract (T_c, V_c_m3, omega, dipole_D, kappa, M) from a component.

    Recognises:
      - CubicMixture component: T_c, p_c, acentric_factor, molar_mass,
        and optional dipole_moment
      - PCSAFT component: same + optional V_c (we use estimated)
      - Any duck-type with the above attrs
    """
    T_c = float(comp.T_c)
    p_c = float(comp.p_c)
    omega = float(comp.acentric_factor)
    # V_c from explicit attribute or critical estimate
    V_c = getattr(comp, 'V_c', None)
    if V_c is None or V_c == 0.0:
        V_c = _V_c_estimate(T_c, p_c, omega)
    dipole = float(getattr(comp, 'dipole_moment', 0.0) or 0.0)
    # kappa: association correction factor (Chung 1988 Table III).
    # Default 0 for non-associating. Specific values for water, alcohols
    # available from the literature; we read from comp.chung_kappa if set.
    kappa = float(getattr(comp, 'chung_kappa', 0.0) or 0.0)
    M = float(getattr(comp, 'molar_mass', 0.0) or 0.0)
    if M <= 0.0:
        raise ValueError(
            f"component {getattr(comp, 'name', '?')!r} has molar_mass={M}; "
            "Chung transport correlation requires molar mass [kg/mol].")
    return T_c, float(V_c), omega, dipole, kappa, M


# -------------------------------------------------------------------------
# Low-pressure (dilute-gas) viscosity
# -------------------------------------------------------------------------

def _eta0(T: float, T_c: float, V_c: float, omega: float,
          dipole_D: float, kappa: float, M: float) -> float:
    """Low-pressure (dilute-gas) viscosity from Chung 1988 Eq. 11.

    eta0 = 26.692 * F_c * sqrt(M[g/mol] * T) / (V_c[cm^3/mol]^(2/3) * Omega^(2,2))
    result in micropoise (1 uP = 1e-7 Pa.s).

    We return Pa.s.

    F_c = 1 - 0.2756 omega + 0.059035 mu_r^4 + kappa
    """
    # T* = T / (eps/k) with eps/k = T_c / 1.2593 (Chung 1988 Eq. 10)
    T_star = T * 1.2593 / T_c
    omega_val = _omega_22(T_star)
    V_c_cm3 = V_c * 1e6
    M_g = M * 1000.0   # g/mol
    mu_r = _mu_r(dipole_D, V_c, T_c)
    F_c = 1.0 - 0.2756 * omega + 0.059035 * mu_r ** 4 + kappa
    eta_uP = 26.692 * F_c * np.sqrt(M_g * T) / (V_c_cm3 ** (2.0/3.0) * omega_val)
    return float(eta_uP * 1e-7)


# -------------------------------------------------------------------------
# Pure-fluid viscosity: low-pressure + Chung dense correction
# -------------------------------------------------------------------------

def viscosity_chung(comp, T: float, rho_mol: Optional[float] = None) -> float:
    """Viscosity of a pure fluid via Chung-Lee-Starling [Pa.s].

    Parameters
    ----------
    comp : component-like
        Must have T_c, p_c, acentric_factor, molar_mass; optional
        dipole_moment, chung_kappa, V_c.
    T : float
        Temperature [K].
    rho_mol : float or None
        Molar density [mol/m^3]. If None, returns low-pressure limit
        (dilute-gas viscosity).

    Returns
    -------
    float : viscosity [Pa.s]
    """
    T_c, V_c, omega, dipole, kappa, M = _component_properties(comp)
    # Low-pressure (zero-density) value
    eta0_val = _eta0(T, T_c, V_c, omega, dipole, kappa, M)
    if rho_mol is None or rho_mol <= 0.0:
        return eta0_val

    mu_r = _mu_r(dipole, V_c, T_c)
    # Chung dense-fluid factor G_1, G_2 (Chung 1988 Eq. 12-15).
    y = V_c * rho_mol / 6.0   # reduced density parameter
    # Polynomial coefficients E_1..E_10
    E = (_E_COEF_VISC[:, 0]
         + _E_COEF_VISC[:, 1] * omega
         + _E_COEF_VISC[:, 2] * mu_r ** 4
         + _E_COEF_VISC[:, 3] * kappa)
    E1, E2, E3, E4, E5, E6, E7, E8, E9, E10 = E

    G_1 = (1.0 - 0.5 * y) / (1.0 - y) ** 3
    G_2 = ((E1 * (1 - np.exp(-E4 * y)) / y
             + E2 * G_1 * np.exp(E5 * y)
             + E3 * G_1)
           / (E1 * E4 + E2 + E3))

    # Dense-fluid viscosity (Chung 1988 Eq. 13)
    eta_star_star = E7 * y * y * G_2 * np.exp(
        E8 + E9 / (T * 1.2593 / T_c) + E10 / (T * 1.2593 / T_c) ** 2
    )
    # eta0 in micropoise here; convert eta** via the Chung relation:
    # eta = eta0 (1/G_2 + E6 * y) + eta**
    eta_uP_0 = eta0_val * 1e7
    eta_uP = eta_uP_0 * (1.0 / G_2 + E6 * y) + eta_star_star * 1e2  # eta** is in micropoise*10^-1? Let's check units
    # Correction: eta_star_star is the dimensionless dense excess; Chung's
    # formula gives it in micropoise directly when eta0 is in micropoise.
    # The literature form:
    #   eta = eta_K + eta_P
    #   eta_K = eta_0 * (1/G_2 + E_6 * y)
    #   eta_P = 36.344e-6 * (M*T_c)^0.5 / V_c^(2/3) * E_7 * y^2 * G_2 * exp(...)
    # Re-derive:
    M_g = M * 1000.0
    V_c_cm3 = V_c * 1e6
    eta_P_uP = (36.344 * np.sqrt(M_g * T_c) / V_c_cm3 ** (2.0/3.0)
                * E7 * y * y * G_2 * np.exp(
                    E8 + E9 * (T_c / T / 1.2593) + E10 * (T_c / T / 1.2593) ** 2
                ))
    eta_uP = eta_uP_0 * (1.0 / G_2 + E6 * y) + eta_P_uP
    return float(eta_uP * 1e-7)


# -------------------------------------------------------------------------
# Pure-fluid thermal conductivity
# -------------------------------------------------------------------------

def thermal_conductivity_chung(comp, T: float, rho_mol: Optional[float] = None,
                                cv_ideal: Optional[float] = None) -> float:
    """Thermal conductivity of a pure fluid via Chung-Lee-Starling [W/(m.K)].

    Parameters
    ----------
    comp : component-like
    T : float
        Temperature [K].
    rho_mol : float or None
        Molar density. None -> low-pressure limit.
    cv_ideal : float or None
        Total ideal-gas isochoric heat capacity c_v^ig [J/(mol.K)].
        This is crucial for polyatomic fluids since thermal conductivity
        picks up a large contribution from internal modes (rotation,
        vibration). If None, a rough polyatomic estimate
        (3R/2 + R * (N_atoms_approx - 1)) is used based on molar mass,
        which gives typical errors of 20-40% vs experiment.
        For accurate polyatomic predictions, pass the actual cv_ideal
        (available from NIST Webbook, CoolProp, or a polynomial fit).

    Returns
    -------
    float : thermal conductivity [W/(m.K)]
    """
    T_c, V_c, omega, dipole, kappa, M = _component_properties(comp)
    R = 8.314462618
    eta0_val = _eta0(T, T_c, V_c, omega, dipole, kappa, M)

    # Ideal-gas cv: if not given, estimate from molar mass (rough).
    # For monatomic gases (He, Ar): cv = 3R/2.
    # For diatomic (N2, O2): cv ~ 5R/2.
    # For polyatomic: cv ~ 3R + vibrational.
    if cv_ideal is None:
        # Estimate based on mass: heavier molecule -> more modes.
        # Rough: 3R/2 (trans) + R (rot, approximated linear) + R * min(max((M*1000-4)/15, 0), 8)
        M_g = M * 1000.0
        dof_internal = 2.0 if M_g < 6 else min(2.0 + (M_g - 4) / 6.0, 12.0)
        cv_ideal = (1.5 + dof_internal / 2.0) * R
    alpha = cv_ideal / R - 1.5
    beta = 0.7862 - 0.7109 * omega + 1.3168 * omega ** 2
    T_r = T / T_c
    Z = 2.0 + 10.5 * T_r ** 2
    psi = 1.0 + alpha * (
        (0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z)
        / (0.6366 + beta * Z + 1.061 * alpha * beta)
    )
    # Chung low-p thermal conductivity (SI form, Reid-Prausnitz-Poling Eq. 10-3.14):
    #   lambda_0 [W/(m.K)] = 3.75 * R / M[kg/mol] * eta0 [Pa.s] * psi
    lambda_0 = 3.75 * R / M * eta0_val * psi

    if rho_mol is None or rho_mol <= 0.0:
        return float(lambda_0)

    # Dense-fluid correction (Chung 1988 Eq. 22-24).
    mu_r = _mu_r(dipole, V_c, T_c)
    y = V_c * rho_mol / 6.0
    B = (_E_COEF_TCOND[:, 0]
         + _E_COEF_TCOND[:, 1] * omega
         + _E_COEF_TCOND[:, 2] * mu_r ** 4
         + _E_COEF_TCOND[:, 3] * kappa)
    B1, B2, B3, B4, B5, B6, B7 = B

    H_2 = ((B1 * (1 - np.exp(-B4 * y)) / y
            + B2 * ((1 - 0.5 * y) / (1 - y) ** 3) * np.exp(B5 * y)
            + B3 * ((1 - 0.5 * y) / (1 - y) ** 3))
           / (B1 * B4 + B2 + B3))

    M_g = M * 1000.0
    V_c_cm3 = V_c * 1e6
    # Chung 1988 Eq. 21 second term: lambda_P in cal/(cm.s.K)
    lambda_P_cal = (3.586e-3 * np.sqrt(T_c / M_g) / V_c_cm3 ** (2.0/3.0)
                    * B7 * y * y * np.sqrt(T_r) * H_2)
    # 1 cal/(cm.s.K) = 418.4 W/(m.K)
    lambda_P = lambda_P_cal * 418.4
    return float(lambda_0 * (1.0 / H_2 + B6 * y) + lambda_P)


# -------------------------------------------------------------------------
# Mixture viscosity via Chung's mixing rules (Chung 1988 Sec. V)
# -------------------------------------------------------------------------

def viscosity_mixture_chung(comps: Sequence, x: Sequence[float],
                            T: float, rho_mol: Optional[float] = None) -> float:
    """Mixture viscosity via Chung 1988 Sec. V mixing rules.

    The mixing rules define pseudo-component properties (T_c_m, V_c_m,
    M_m, omega_m, mu_r_m, kappa_m) via composition-weighted combining
    rules, then uses the pure-fluid Chung correlation with those
    pseudo-component values.

    Parameters
    ----------
    comps : list of components (length N)
    x : array-like (N,)
        Mole fractions.
    T, rho_mol : as for viscosity_chung
    """
    x = np.asarray(x, dtype=np.float64)
    x = x / x.sum()
    N = len(comps)
    # Extract per-component props
    Tci, Vci, wi, dipi, kapi, Mi = zip(*[_component_properties(c) for c in comps])
    Tci = np.array(Tci); Vci = np.array(Vci); wi = np.array(wi)
    dipi = np.array(dipi); kapi = np.array(kapi); Mi = np.array(Mi)
    sigma_i = 0.809 * Vci ** (1.0/3.0) * 1e10  # Angstrom; V_c in m^3/mol
    eps_over_k_i = Tci / 1.2593

    # Chung mixing rules (their Eqs. 27-34)
    sigma_ij = 0.5 * (sigma_i[:, None] + sigma_i[None, :])
    eps_ij = np.sqrt(np.outer(eps_over_k_i, eps_over_k_i))
    omega_ij = 0.5 * (wi[:, None] + wi[None, :])
    M_ij = 2.0 * np.outer(Mi, Mi) / (Mi[:, None] + Mi[None, :])
    kappa_ij = np.sqrt(np.outer(kapi, kapi))

    xx = np.outer(x, x)
    sigma_m_cubed = float(np.sum(xx * sigma_ij ** 3))
    sigma_m = sigma_m_cubed ** (1.0/3.0)
    eps_m = float(np.sum(xx * eps_ij * sigma_ij ** 3)) / sigma_m_cubed
    T_cm = 1.2593 * eps_m
    V_cm = (sigma_m / 0.809) ** 3 * 1e-30  # back to m^3/mol
    # M and omega use square-root / linear mixing in sigma^2
    M_m = (float(np.sum(xx * eps_ij * sigma_ij ** 2 * np.sqrt(M_ij)))
           / (eps_m * sigma_m ** 2)) ** 2
    omega_m = (float(np.sum(xx * omega_ij * sigma_ij ** 3))
               / sigma_m_cubed)
    # dipole: use mu_r_m ** 4 as composition-weighted sum (Chung Eq. 34)
    mu_r_i4 = np.array([_mu_r(d, Vci[i], Tci[i]) ** 4 for i, d in enumerate(dipi)])
    # mu_m^4 * V_cm * T_cm = sum_i sum_j x_i x_j mu_i^2 mu_j^2 sigma_ij^-3 ... (complex)
    # Simplified approximation (Eq. 35 of Chung):
    mu_m4 = float(np.sum(xx * np.outer(mu_r_i4, np.ones(N))
                         * sigma_ij ** 3) / sigma_m_cubed)
    # kappa mixing (Eq. 36):
    kappa_m = float(np.sum(xx * kappa_ij * sigma_ij ** 3) / sigma_m_cubed)
    # Recover pseudo-dipole in Debye from mu_r_m^4 = 131.3^4 * mu^4 / (V_cm_cm3 * T_cm)^2
    V_cm_cm3 = V_cm * 1e6
    dipole_m = (mu_m4 * (V_cm_cm3 * T_cm) ** 2 / 131.3 ** 4) ** 0.25

    # Use pure-Chung formula with pseudo-component
    class _Pseudo:
        def __init__(self, T_c, p_c, omega, M, dipole, kappa, V_c):
            self.T_c = T_c; self.p_c = p_c
            self.acentric_factor = omega; self.molar_mass = M
            self.dipole_moment = dipole; self.chung_kappa = kappa
            self.V_c = V_c
    # Dummy p_c (not used by Chung)
    pseudo = _Pseudo(T_cm, 1e7, omega_m, M_m, dipole_m, kappa_m, V_cm)
    return viscosity_chung(pseudo, T, rho_mol)


def thermal_conductivity_mixture_chung(comps: Sequence, x: Sequence[float],
                                        T: float, rho_mol: Optional[float] = None,
                                        cv_ideal: Optional[float] = None) -> float:
    """Mixture thermal conductivity via Chung 1988 mixing rules (same
    pseudo-component approach as viscosity)."""
    x = np.asarray(x, dtype=np.float64); x = x / x.sum()
    N = len(comps)
    Tci, Vci, wi, dipi, kapi, Mi = zip(*[_component_properties(c) for c in comps])
    Tci = np.array(Tci); Vci = np.array(Vci); wi = np.array(wi)
    dipi = np.array(dipi); kapi = np.array(kapi); Mi = np.array(Mi)
    sigma_i = 0.809 * Vci ** (1.0/3.0) * 1e10
    eps_over_k_i = Tci / 1.2593
    sigma_ij = 0.5 * (sigma_i[:, None] + sigma_i[None, :])
    eps_ij = np.sqrt(np.outer(eps_over_k_i, eps_over_k_i))
    omega_ij = 0.5 * (wi[:, None] + wi[None, :])
    M_ij = 2.0 * np.outer(Mi, Mi) / (Mi[:, None] + Mi[None, :])
    kappa_ij = np.sqrt(np.outer(kapi, kapi))
    xx = np.outer(x, x)
    sigma_m_cubed = float(np.sum(xx * sigma_ij ** 3))
    sigma_m = sigma_m_cubed ** (1.0/3.0)
    eps_m = float(np.sum(xx * eps_ij * sigma_ij ** 3)) / sigma_m_cubed
    T_cm = 1.2593 * eps_m
    V_cm = (sigma_m / 0.809) ** 3 * 1e-30
    M_m = (float(np.sum(xx * eps_ij * sigma_ij ** 2 * np.sqrt(M_ij)))
           / (eps_m * sigma_m ** 2)) ** 2
    omega_m = float(np.sum(xx * omega_ij * sigma_ij ** 3)) / sigma_m_cubed
    mu_r_i4 = np.array([_mu_r(d, Vci[i], Tci[i]) ** 4 for i, d in enumerate(dipi)])
    mu_m4 = float(np.sum(xx * np.outer(mu_r_i4, np.ones(N)) * sigma_ij ** 3)
                  / sigma_m_cubed)
    kappa_m = float(np.sum(xx * kappa_ij * sigma_ij ** 3) / sigma_m_cubed)
    V_cm_cm3 = V_cm * 1e6
    dipole_m = (mu_m4 * (V_cm_cm3 * T_cm) ** 2 / 131.3 ** 4) ** 0.25

    class _Pseudo:
        def __init__(self, T_c, p_c, omega, M, dipole, kappa, V_c):
            self.T_c = T_c; self.p_c = p_c
            self.acentric_factor = omega; self.molar_mass = M
            self.dipole_moment = dipole; self.chung_kappa = kappa
            self.V_c = V_c
    pseudo = _Pseudo(T_cm, 1e7, omega_m, M_m, dipole_m, kappa_m, V_cm)
    return thermal_conductivity_chung(pseudo, T, rho_mol, cv_ideal=cv_ideal)
