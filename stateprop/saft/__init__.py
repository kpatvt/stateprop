"""
stateprop.saft: PC-SAFT equation of state (non-associating, Gross & Sadowski 2001).

PC-SAFT is a molecular-based EOS that works well for hydrocarbons, light
gases, and their mixtures. This module implements the non-associating
variant, suitable for non-polar and weakly polar fluids. Association
(for water, alcohols, acids) is a planned extension.

Usage mirrors `stateprop.cubic`:

    from stateprop.saft import PCSAFT, SAFTMixture
    c1 = PCSAFT(m=1.0, sigma=3.7039, epsilon_k=150.03,
                T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142,
                name="methane")
    c2 = PCSAFT(m=1.6069, sigma=3.5206, epsilon_k=191.42,
                T_c=305.322, p_c=4.8722e6, acentric_factor=0.0995,
                name="ethane")
    mx = SAFTMixture([c1, c2], composition=[0.5, 0.5], k_ij={(0, 1): 0.0})
    p = mx.pressure(rho_mol=1000.0, T=220.0)
    ln_phi = mx.ln_phi(rho_mol=1000.0, T=220.0)
    rho = mx.density_from_pressure(p=1e6, T=220.0, phase_hint='liquid')

A `SAFTMixture` exposes the same method surface as `stateprop.cubic.mixture.
CubicMixture`, so the cubic flash / envelope / three-phase-flash machinery
works directly:

    from stateprop.cubic.flash import flash_pt
    result = flash_pt(p=1e6, T=220.0, z=[0.5, 0.5], mixture=mx)

Pre-packaged parameter sets for common substances are available as module
constants (METHANE, ETHANE, CO2, N_BUTANE, ...) from `stateprop.saft.eos`.

Reference
---------
Gross, J. and Sadowski, G. (2001). Ind. Eng. Chem. Res. 40, 1244-1260.
"""
from .eos import (
    PCSAFT,
    METHANE, NITROGEN, CO2, ETHANE, PROPANE, N_BUTANE,
    N_PENTANE, N_HEXANE, N_HEPTANE, N_OCTANE,
    # v0.9.23: associating and polar
    WATER, METHANOL, ETHANOL, N_PROPANOL,
    ACETONE, DME,
)
from .mixture import SAFTMixture

__all__ = [
    "PCSAFT", "SAFTMixture",
    "METHANE", "NITROGEN", "CO2", "ETHANE", "PROPANE", "N_BUTANE",
    "N_PENTANE", "N_HEXANE", "N_HEPTANE", "N_OCTANE",
    # v0.9.23
    "WATER", "METHANOL", "ETHANOL", "N_PROPANOL",
    "ACETONE", "DME",
]
