"""
stateprop: Numba-accelerated evaluation of thermodynamic properties via
multiparameter Helmholtz equations of state.

Public API
----------
Core model (pure components):
    Fluid, load_fluid
    alpha, alpha_r, alpha_0
    alpha_derivs, alpha_r_derivs, alpha_0_derivs

Property evaluation (rho [mol/m^3], T [K]):
    pressure, compressibility_factor,
    internal_energy, enthalpy, entropy,
    cv, cp, speed_of_sound, gibbs_energy,
    fugacity_coefficient, joule_thomson_coefficient,
    dp_drho_T, dp_dT_rho

Phase equilibrium / flashing (pure components):
    saturation_pT, density_from_pressure
    flash_pt, flash_ph, flash_ps, flash_th, flash_ts, flash_tv, flash_uv
    FlashResult
    trace_phase_envelope, PhaseEnvelope

Mixture support (multicomponent, GERG-style multi-fluid):
    stateprop.mixture submodule -- see stateprop.mixture docstring.
    Mixture, Component, load_mixture
    KunzWagnerReducing, BinaryParams, DepartureFunction
    flash_pt, flash_tbeta, flash_pbeta, flash_ph, flash_ps, flash_th, flash_ts
    bubble_point_p, bubble_point_T, dew_point_p, dew_point_T
    stability_test_TPD, wilson_K, rachford_rice

Cubic EOS (PR, SRK, RK, vdW):
    stateprop.cubic submodule -- see stateprop.cubic docstring.
    CubicEOS, PR, PR78, SRK, RK, VDW
    PR_MC, SRK_MC, PR_Twu, SRK_Twu, PRSV  (alpha-function variants)
    CubicMixture (van der Waals one-fluid mixing with k_ij)
    Peneloux-style volume translation via CubicEOS(..., volume_shift_c=...)
    flash_pt, flash_ph, flash_ps, flash_th, flash_ts
    stability_test_TPD
    bubble_point_p, bubble_point_T, dew_point_p, dew_point_T
    critical_point (Heidemann-Khalil / Michelsen, analytic derivatives)
    envelope_point, trace_envelope (phase envelope, critical-seeded)
"""
from .fluid import Fluid, load_fluid
from .core import (
    alpha, alpha_r, alpha_0,
    alpha_derivs, alpha_r_derivs, alpha_0_derivs,
)
from .properties import (
    pressure, compressibility_factor,
    internal_energy, enthalpy, entropy,
    cv, cp, speed_of_sound, gibbs_energy,
    fugacity_coefficient, joule_thomson_coefficient,
    dp_drho_T, dp_dT_rho,
)
from .saturation import saturation_pT, density_from_pressure
from .flash import (
    flash_pt, flash_ph, flash_ps, flash_th, flash_ts, flash_tv, flash_uv,
    FlashResult,
)
from .phase_envelope import trace_phase_envelope, PhaseEnvelope
from . import mixture
from . import cubic

__version__ = "0.9.57"

__all__ = [
    # Core
    "Fluid", "load_fluid",
    "alpha", "alpha_r", "alpha_0",
    "alpha_derivs", "alpha_r_derivs", "alpha_0_derivs",
    # Properties
    "pressure", "compressibility_factor",
    "internal_energy", "enthalpy", "entropy",
    "cv", "cp", "speed_of_sound", "gibbs_energy",
    "fugacity_coefficient", "joule_thomson_coefficient",
    "dp_drho_T", "dp_dT_rho",
    # Saturation / density
    "saturation_pT", "density_from_pressure",
    # Flash (pure)
    "flash_pt", "flash_ph", "flash_ps", "flash_th", "flash_ts",
    "flash_tv", "flash_uv",
    "FlashResult",
    # Phase envelope
    "trace_phase_envelope", "PhaseEnvelope",
    # Submodules
    "mixture", "cubic",
]
