"""
stateprop: Numba-accelerated evaluation of single-component thermodynamic
properties via multiparameter Helmholtz equations of state.

Public API
----------
Core model:
    Fluid, load_fluid
    alpha, alpha_r, alpha_0
    alpha_derivs, alpha_r_derivs, alpha_0_derivs

Property evaluation (rho [mol/m^3], T [K]):
    pressure, compressibility_factor,
    internal_energy, enthalpy, entropy,
    cv, cp, speed_of_sound, gibbs_energy,
    fugacity_coefficient, joule_thomson_coefficient,
    dp_drho_T, dp_dT_rho

Phase equilibrium / flashing:
    saturation_pT      -- solve vapor-liquid equilibrium at given T
    density_from_pressure -- solve rho from (p, T) on a chosen branch
    flash_pt, flash_ph, flash_ps, flash_th, flash_ts, flash_tv, flash_uv
    FlashResult
    trace_phase_envelope, PhaseEnvelope
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

__version__ = "0.2.0"

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
    # Flash
    "flash_pt", "flash_ph", "flash_ps", "flash_th", "flash_ts",
    "flash_tv", "flash_uv",
    "FlashResult",
    # Phase envelope
    "trace_phase_envelope", "PhaseEnvelope",
]
