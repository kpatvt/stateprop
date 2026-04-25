"""
stateprop.mixture: Multicomponent thermodynamics via the GERG-style multi-fluid
approximation.

This subpackage extends the pure-component framework to N-component mixtures
using the following model structure (following Kunz & Wagner 2012):

  alpha(delta, tau, x) = alpha_0(rho, T, x) + alpha_r(delta, tau, x)

  alpha_0(rho, T, x) = sum_i x_i * [alpha_0_oi(rho, T) + ln(x_i)]

  alpha_r(delta, tau, x) = sum_i x_i * alpha_r_oi(delta, tau)
                         + Delta_alpha_r(delta, tau, x)     (departure function; optional)

where delta = rho / rho_r(x) and tau = T_r(x) / T, with the reducing functions:

  T_r(x)     = sum_i sum_j  x_i x_j * beta_T_ij * gamma_T_ij
                             * (x_i + x_j) / (beta_T_ij^2 * x_i + x_j)
                             * sqrt(T_ci T_cj)

  1/rho_r(x) = sum_i sum_j  x_i x_j * beta_v_ij * gamma_v_ij
                             * (x_i + x_j) / (beta_v_ij^2 * x_i + x_j)
                             * (1/8) * (1/rho_ci^(1/3) + 1/rho_cj^(1/3))^3

The departure function Delta_alpha_r is a sum over binary pairs of residual-form
terms (polynomial + generalized exponential) weighted by F_ij. GERG-2008
publishes specific coefficients for pairs of natural-gas components. In
this library, departure functions are loaded from the binary JSON files; any
pair without an explicit `departure` block defaults to Delta = 0 (simplified
multi-fluid).

Public API:

    Component          -- a pure component with GERG-specific data
    Mixture            -- mixture of components with composition x[]

    KunzWagnerReducing -- reducing functions T_r(x), rho_r(x) and derivatives
    DepartureFunction  -- one binary departure alpha_r_ij(delta, tau) evaluator

    alpha_r_mix, alpha_0_mix   -- full mixture Helmholtz derivatives
    ln_phi                      -- fugacity coefficients of each component

    flash_pt, flash_tbeta, flash_pbeta     -- core flashes
    flash_ph, flash_ps, flash_th, flash_ts -- state-function flashes via outer Newton
    bubble_point_p, bubble_point_T         -- beta=0 specializations
    dew_point_p, dew_point_T               -- beta=1 specializations

    stability_test_TPD  -- Michelsen tangent-plane stability check
    rachford_rice       -- solve 1-D Rachford-Rice equation for beta

    MixtureFlashResult  -- result dataclass
"""
from .mixture import Mixture, load_mixture
from .component import Component, load_component
from .reducing import KunzWagnerReducing, BinaryParams
from .departure import DepartureFunction, DepartureTerm
from .properties import (
    pressure, enthalpy, entropy, ln_phi, density_from_pressure,
    alpha_r_mix_derivs,
)
from .stability import stability_test_TPD, wilson_K
from .flash import (
    MixtureFlashResult, rachford_rice,
    flash_pt, flash_tbeta, flash_pbeta,
    flash_ph, flash_ps, flash_th, flash_ts,
    bubble_point_p, bubble_point_T, dew_point_p, dew_point_T,
    newton_bubble_point_p, newton_bubble_point_T,
    newton_dew_point_p, newton_dew_point_T,
)
from .critical import critical_point, critical_point_multistart
from .envelope import trace_envelope, envelope_point
from .three_phase_flash import (
    ThreePhaseFlashResult, flash_pt_three_phase,
)

__all__ = [
    "Mixture", "load_mixture",
    "Component", "load_component",
    "KunzWagnerReducing", "BinaryParams",
    "DepartureFunction", "DepartureTerm",
    "pressure", "enthalpy", "entropy", "ln_phi", "density_from_pressure",
    "stability_test_TPD", "wilson_K",
    "MixtureFlashResult", "rachford_rice",
    "flash_pt", "flash_tbeta", "flash_pbeta",
    "flash_ph", "flash_ps", "flash_th", "flash_ts",
    "bubble_point_p", "bubble_point_T", "dew_point_p", "dew_point_T",
    "newton_bubble_point_p", "newton_bubble_point_T",
    "newton_dew_point_p", "newton_dew_point_T",
    "critical_point",
    "critical_point_multistart",
    "trace_envelope",
    "envelope_point",
    "ThreePhaseFlashResult",
    "flash_pt_three_phase",
]
