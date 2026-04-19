"""
stateprop.cubic: Generalized two-parameter cubic equations of state.

Supports:
  - Peng-Robinson (PR)
  - Soave-Redlich-Kwong (SRK)
  - Original Redlich-Kwong (RK)
  - Van der Waals (vdW)

Pure-fluid usage:
    from stateprop.cubic import PR
    eos = PR(T_c=304.13, p_c=7.377e6, acentric_factor=0.224)
    # Get residual Helmholtz and derivatives
    A, A_d, A_t, A_dd, A_tt, A_dt = eos.alpha_r_derivs(delta=0.5, tau=1.2)
    # Pressure at (rho, T)
    p = eos.pressure(rho=5000.0, T=280.0)
    # a(T) for the cubic
    a, da_dT, d2a_dT2 = eos.a_T(T=280.0)

Mixture usage:
    from stateprop.cubic import CubicMixture, PR
    c1 = PR(T_c=190.56, p_c=4.599e6, acentric_factor=0.011, name="methane")
    c2 = PR(T_c=126.19, p_c=3.396e6, acentric_factor=0.039, name="nitrogen")
    mx = CubicMixture([c1, c2], composition=[0.9, 0.1],
                      k_ij={(0, 1): 0.025})   # or None for ideal mixing
    ln_phi = mx.ln_phi(rho, T)
    rho = mx.density_from_pressure(p, T, phase_hint='liquid')

Note: the cubic module is self-contained and independent of the multiparameter
Helmholtz stack. Cubic fluids are defined directly by critical parameters,
not by JSON EOS tables. This keeps cubics simple and general: any fluid can
be used as long as Tc, Pc, and omega are known.
"""
from .eos import CubicEOS, PR, SRK, RK, VDW
from .mixture import CubicMixture, ln_phi, density_from_pressure
from .flash import (
    CubicFlashResult,
    flash_pt,
    flash_ph, flash_ps, flash_th, flash_ts,
    stability_test_TPD,
    bubble_point_p, bubble_point_T,
    dew_point_p, dew_point_T,
)

__all__ = [
    "CubicEOS", "PR", "SRK", "RK", "VDW",
    "CubicMixture",
    "ln_phi", "density_from_pressure",
    "CubicFlashResult",
    "flash_pt",
    "flash_ph", "flash_ps", "flash_th", "flash_ts",
    "stability_test_TPD",
    "bubble_point_p", "bubble_point_T",
    "dew_point_p", "dew_point_T",
]
