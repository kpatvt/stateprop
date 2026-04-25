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
from .eos import (
    CubicEOS, PR, PR78, SRK, RK, VDW,
    PR_MC, SRK_MC, PR_Twu, SRK_Twu, PRSV,
)
from .mixture import CubicMixture, ln_phi, density_from_pressure
from .flash import (
    CubicFlashResult,
    flash_pt,
    flash_ph, flash_ps, flash_th, flash_ts,
    flash_tv, flash_uv, flash_pv,
    flash_p_alpha, flash_t_alpha,
    stability_test_TPD,
    bubble_point_p, bubble_point_T,
    dew_point_p, dew_point_T,
    newton_bubble_point_p, newton_bubble_point_T,
    newton_dew_point_p, newton_dew_point_T,
)
from .critical import critical_point
from .envelope import envelope_point, trace_envelope
from .three_phase_flash import (
    ThreePhaseFlashResult, flash_pt_three_phase,
)
from .from_chemicals import (
    lookup_pure_component,
    cubic_from_name,
    PR_from_name, PR78_from_name, SRK_from_name, RK_from_name, VDW_from_name,
    cubic_mixture_from_names,
    chemicals_available,
)

__all__ = [
    "CubicEOS",
    "PR", "PR78", "SRK", "RK", "VDW",
    "PR_MC", "SRK_MC", "PR_Twu", "SRK_Twu", "PRSV",
    "CubicMixture",
    "ln_phi", "density_from_pressure",
    "CubicFlashResult",
    "flash_pt",
    "flash_ph", "flash_ps", "flash_th", "flash_ts",
    "stability_test_TPD",
    "bubble_point_p", "bubble_point_T",
    "dew_point_p", "dew_point_T",
    "newton_bubble_point_p", "newton_bubble_point_T",
    "newton_dew_point_p", "newton_dew_point_T",
    "critical_point",
    "envelope_point",
    "trace_envelope",
    "ThreePhaseFlashResult", "flash_pt_three_phase",
    # chemicals-library interface (v0.8.0)
    "lookup_pure_component", "cubic_from_name",
    "PR_from_name", "PR78_from_name", "SRK_from_name",
    "RK_from_name", "VDW_from_name",
    "cubic_mixture_from_names", "chemicals_available",
]
