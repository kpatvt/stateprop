"""
Thermodynamic properties from reduced Helmholtz-energy derivatives.

Given the six derivatives of alpha = alpha_0 + alpha_r w.r.t. (delta, tau)
evaluated at a state point (rho, T), every thermodynamic property follows
from standard identities.

Packed-argument convention (from Fluid.pack()):
    (R, rho_c, T_c,                                       # 3 scalars
     pn, pd, pt,                                          # 3 polynomial arrays
     en, ed, et, ec,                                      # 4 exponential arrays
     gn, gd, gt, g_eta, g_eps, g_beta, g_gamma,           # 7 gaussian arrays
     n_a, n_aa, n_b, n_B, n_C, n_D, n_A, n_beta,          # 8 non-analytic arrays
     ideal_codes, ideal_a, ideal_b)                       # 3 ideal arrays
  = 3 scalars + 14 residual arrays + 8 non-analytic arrays + 3 ideal arrays
  = 28 total.

Residual-only kernels take the first 25 (3 scalars + 22 arrays).
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(fn):
            return fn
        return deco

from .core import alpha_r_derivs, alpha_0_derivs


# Number of leading args in pack() for the residual-only kernels
N_RES_ARGS = 25   # 3 scalars + 22 residual arrays


# ---------------------------------------------------------------------------
# Pressure and its derivatives
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pressure_kernel(rho, T, R, rho_c, T_c,
                     pn, pd, pt, en, ed, et, ec,
                     gn, gd, gt, ge, geps, gb, ggam,
                     na, naa, nb, nB, nC, nD, nA, nbeta):
    delta = rho / rho_c
    tau = T_c / T
    _, A_d, _, _, _, _ = alpha_r_derivs(delta, tau,
                                         pn, pd, pt,
                                         en, ed, et, ec,
                                         gn, gd, gt, ge, geps, gb, ggam,
                                         na, naa, nb, nB, nC, nD, nA, nbeta)
    return rho * R * T * (1.0 + delta * A_d)


@njit(cache=True)
def _dp_drho_T_kernel(rho, T, R, rho_c, T_c,
                      pn, pd, pt, en, ed, et, ec,
                      gn, gd, gt, ge, geps, gb, ggam,
                      na, naa, nb, nB, nC, nD, nA, nbeta):
    delta = rho / rho_c
    tau = T_c / T
    _, A_d, _, A_dd, _, _ = alpha_r_derivs(delta, tau,
                                            pn, pd, pt,
                                            en, ed, et, ec,
                                            gn, gd, gt, ge, geps, gb, ggam,
                                            na, naa, nb, nB, nC, nD, nA, nbeta)
    return R * T * (1.0 + 2.0 * delta * A_d + delta * delta * A_dd)


@njit(cache=True)
def _dp_dT_rho_kernel(rho, T, R, rho_c, T_c,
                      pn, pd, pt, en, ed, et, ec,
                      gn, gd, gt, ge, geps, gb, ggam,
                      na, naa, nb, nB, nC, nD, nA, nbeta):
    delta = rho / rho_c
    tau = T_c / T
    _, A_d, _, _, _, A_dt = alpha_r_derivs(delta, tau,
                                            pn, pd, pt,
                                            en, ed, et, ec,
                                            gn, gd, gt, ge, geps, gb, ggam,
                                            na, naa, nb, nB, nC, nD, nA, nbeta)
    return rho * R * (1.0 + delta * A_d - delta * tau * A_dt)


# ---------------------------------------------------------------------------
# Full property kernel (one pass)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _all_props_kernel(rho, T, R, rho_c, T_c,
                      pn, pd, pt, en, ed, et, ec,
                      gn, gd, gt, ge, geps, gb, ggam,
                      na, naa, nb, nB, nC, nD, nA, nbeta,
                      codes, a_arr, b_arr):
    """Compute properties in a single pass.  Returns a 20-element array."""
    delta = rho / rho_c
    tau = T_c / T

    A0, A0_d, A0_t, A0_dd, A0_tt, A0_dt = alpha_0_derivs(delta, tau, codes, a_arr, b_arr)
    Ar, Ar_d, Ar_t, Ar_dd, Ar_tt, Ar_dt = alpha_r_derivs(delta, tau,
                                                          pn, pd, pt,
                                                          en, ed, et, ec,
                                                          gn, gd, gt, ge, geps, gb, ggam,
                                                          na, naa, nb, nB, nC, nD, nA, nbeta)

    one_plus = 1.0 + delta * Ar_d
    dp_drho_group = 1.0 + 2.0 * delta * Ar_d + delta * delta * Ar_dd
    dp_dT_group = 1.0 + delta * Ar_d - delta * tau * Ar_dt
    tau2_alphatt = tau * tau * (A0_tt + Ar_tt)

    Z = one_plus
    p = rho * R * T * Z
    u_over_RT = tau * (A0_t + Ar_t)
    u = R * T * u_over_RT
    h = R * T * (1.0 + u_over_RT + delta * Ar_d)
    s = R * (u_over_RT - A0 - Ar)

    cv = -R * tau2_alphatt
    if dp_drho_group > 0.0:
        cp = cv + R * (dp_dT_group * dp_dT_group) / dp_drho_group
    else:
        cp = np.nan

    if tau2_alphatt != 0.0:
        W = dp_drho_group - (dp_dT_group * dp_dT_group) / tau2_alphatt
    else:
        W = np.nan

    g = R * T * (1.0 + A0 + Ar + delta * Ar_d)
    dp_drho_T = R * T * dp_drho_group
    dp_dT_rho = rho * R * dp_dT_group

    out = np.empty(20)
    out[0] = p
    out[1] = Z
    out[2] = u
    out[3] = h
    out[4] = s
    out[5] = cv
    out[6] = cp
    out[7] = W
    out[8] = g
    out[9] = dp_drho_T
    out[10] = dp_dT_rho
    out[11] = A0
    out[12] = A0_t
    out[13] = A0_tt
    out[14] = Ar
    out[15] = Ar_d
    out[16] = Ar_t
    out[17] = Ar_dd
    out[18] = Ar_tt
    out[19] = Ar_dt
    return out


# ---------------------------------------------------------------------------
# Python-side scalar/vector wrapper
# ---------------------------------------------------------------------------

def _scalar_or_vec(func, n_args=None):
    def wrapper(rho, T, fluid):
        rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
        if rho_arr.shape != T_arr.shape:
            rho_arr, T_arr = np.broadcast_arrays(rho_arr, T_arr)
            rho_arr = np.ascontiguousarray(rho_arr, dtype=np.float64)
            T_arr = np.ascontiguousarray(T_arr, dtype=np.float64)
        out = np.empty_like(rho_arr)
        full_args = fluid.pack()
        args = full_args if n_args is None else full_args[:n_args]
        for i in range(rho_arr.size):
            out.flat[i] = func(rho_arr.flat[i], T_arr.flat[i], *args)
        if np.isscalar(rho) and np.isscalar(T):
            return float(out.flat[0])
        return out
    return wrapper


def pressure(rho, T, fluid):
    """Pressure [Pa]."""
    return _scalar_or_vec(_pressure_kernel, n_args=N_RES_ARGS)(rho, T, fluid)


def compressibility_factor(rho, T, fluid):
    """Compressibility factor Z = p/(rho R T)."""
    def k(rho, T, R, rho_c, T_c,
          pn, pd, pt, en, ed, et, ec,
          gn, gd, gt, ge, geps, gb, ggam,
          na, naa, nb, nB, nC, nD, nA, nbeta):
        p = _pressure_kernel(rho, T, R, rho_c, T_c,
                             pn, pd, pt, en, ed, et, ec,
                             gn, gd, gt, ge, geps, gb, ggam,
                             na, naa, nb, nB, nC, nD, nA, nbeta)
        return p / (rho * R * T)
    return _scalar_or_vec(k, n_args=N_RES_ARGS)(rho, T, fluid)


def dp_drho_T(rho, T, fluid):
    """(dp/drho)_T  [Pa m^3/mol]."""
    return _scalar_or_vec(_dp_drho_T_kernel, n_args=N_RES_ARGS)(rho, T, fluid)


def dp_dT_rho(rho, T, fluid):
    """(dp/dT)_rho  [Pa/K]."""
    return _scalar_or_vec(_dp_dT_rho_kernel, n_args=N_RES_ARGS)(rho, T, fluid)


def _prop_factory(index):
    def kernel(rho, T, R, rho_c, T_c,
               pn, pd, pt, en, ed, et, ec,
               gn, gd, gt, ge, geps, gb, ggam,
               na, naa, nb, nB, nC, nD, nA, nbeta,
               codes, a_arr, b_arr):
        out = _all_props_kernel(rho, T, R, rho_c, T_c,
                                pn, pd, pt, en, ed, et, ec,
                                gn, gd, gt, ge, geps, gb, ggam,
                                na, naa, nb, nB, nC, nD, nA, nbeta,
                                codes, a_arr, b_arr)
        return out[index]
    return njit(cache=True)(kernel)


_k_internal_energy = _prop_factory(2)
_k_enthalpy        = _prop_factory(3)
_k_entropy         = _prop_factory(4)
_k_cv              = _prop_factory(5)
_k_cp              = _prop_factory(6)
_k_W               = _prop_factory(7)
_k_gibbs           = _prop_factory(8)


def internal_energy(rho, T, fluid):
    """Molar internal energy [J/mol]."""
    return _scalar_or_vec(_k_internal_energy, n_args=None)(rho, T, fluid)

def enthalpy(rho, T, fluid):
    """Molar enthalpy [J/mol]."""
    return _scalar_or_vec(_k_enthalpy, n_args=None)(rho, T, fluid)

def entropy(rho, T, fluid):
    """Molar entropy [J/(mol K)]."""
    return _scalar_or_vec(_k_entropy, n_args=None)(rho, T, fluid)

def cv(rho, T, fluid):
    """Isochoric molar heat capacity [J/(mol K)]."""
    return _scalar_or_vec(_k_cv, n_args=None)(rho, T, fluid)

def cp(rho, T, fluid):
    """Isobaric molar heat capacity [J/(mol K)]."""
    return _scalar_or_vec(_k_cp, n_args=None)(rho, T, fluid)

def gibbs_energy(rho, T, fluid):
    """Molar Gibbs energy [J/mol]."""
    return _scalar_or_vec(_k_gibbs, n_args=None)(rho, T, fluid)


def speed_of_sound(rho, T, fluid):
    """Thermodynamic speed of sound [m/s]. Needs molar_mass."""
    if fluid.molar_mass is None:
        raise ValueError(f"molar_mass required for speed of sound (fluid '{fluid.name}')")
    M = fluid.molar_mass
    W = _scalar_or_vec(_k_W, n_args=None)(rho, T, fluid)
    if np.isscalar(W):
        return float(np.sqrt(W * fluid.R * T / M))
    return np.sqrt(W * fluid.R * np.asarray(T) / M)


def fugacity_coefficient(rho, T, fluid):
    """Fugacity coefficient phi = f/p."""
    @njit(cache=True)
    def k(rho, T, R, rho_c, T_c,
          pn, pd, pt, en, ed, et, ec,
          gn, gd, gt, ge, geps, gb, ggam,
          na, naa, nb, nB, nC, nD, nA, nbeta):
        delta = rho / rho_c
        tau = T_c / T
        Ar, Ar_d, _, _, _, _ = alpha_r_derivs(delta, tau,
                                               pn, pd, pt,
                                               en, ed, et, ec,
                                               gn, gd, gt, ge, geps, gb, ggam,
                                               na, naa, nb, nB, nC, nD, nA, nbeta)
        Z = 1.0 + delta * Ar_d
        return np.exp(Ar + delta * Ar_d - np.log(Z))
    return _scalar_or_vec(k, n_args=N_RES_ARGS)(rho, T, fluid)


def joule_thomson_coefficient(rho, T, fluid):
    """Joule-Thomson coefficient mu = (dT/dp)_H  [K/Pa]."""
    @njit(cache=True)
    def k(rho, T, R, rho_c, T_c,
          pn, pd, pt, en, ed, et, ec,
          gn, gd, gt, ge, geps, gb, ggam,
          na, naa, nb, nB, nC, nD, nA, nbeta,
          codes, a_arr, b_arr):
        out = _all_props_kernel(rho, T, R, rho_c, T_c,
                                pn, pd, pt, en, ed, et, ec,
                                gn, gd, gt, ge, geps, gb, ggam,
                                na, naa, nb, nB, nC, nD, nA, nbeta,
                                codes, a_arr, b_arr)
        cp_val = out[6]
        Ar_d = out[15]
        Ar_dd = out[17]
        Ar_dt = out[19]
        delta = rho / rho_c
        tau = T_c / T
        num = delta * Ar_d + delta * delta * Ar_dd + delta * tau * Ar_dt
        den = 1.0 + 2.0 * delta * Ar_d + delta * delta * Ar_dd
        return -(num / den) / (rho * cp_val)
    return _scalar_or_vec(k, n_args=None)(rho, T, fluid)
