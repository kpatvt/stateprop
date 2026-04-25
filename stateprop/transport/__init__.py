"""Transport property correlations (v0.9.32+).

Chung-Lee-Starling method for viscosity and thermal conductivity (pure
and mixture). Brock-Bird corresponding-states correlation for surface
tension. Macleod-Sugden parachor surface tension (v0.9.33). Stiel-
Thodos / Jossi high-pressure viscosity correction (v0.9.38).
Wassiljewa-Mason-Saxena mixture thermal conductivity (v0.9.38).

All functions take component objects (any object with fields T_c, p_c,
acentric_factor, optionally dipole_moment, molar_mass) and return SI
quantities:

    viscosity                   [Pa.s]
    thermal_conductivity        [W/(m.K)]
    surface_tension             [N/m]

The transport module is independent of EOS choice -- it works with
CoolPropFluid, cubic Component, and PCSAFT interchangeably, as long as
the component has the required fields.
"""

from .chung import (
    viscosity_chung,
    thermal_conductivity_chung,
    viscosity_mixture_chung,
    thermal_conductivity_mixture_chung,
)
from .surface_tension import (
    surface_tension_brock_bird,
    surface_tension_macleod_sugden,
    surface_tension_mixture_macleod_sugden,
)
from .stiel_thodos import (
    viscosity_stiel_thodos,
    viscosity_mixture_stiel_thodos,
)
from .wassiljewa import (
    thermal_conductivity_mixture_wassiljewa,
    viscosity_mixture_wilke,
)

__all__ = [
    "viscosity_chung",
    "thermal_conductivity_chung",
    "viscosity_mixture_chung",
    "thermal_conductivity_mixture_chung",
    "surface_tension_brock_bird",
    "surface_tension_macleod_sugden",
    "surface_tension_mixture_macleod_sugden",
    "viscosity_stiel_thodos",
    "viscosity_mixture_stiel_thodos",
    "thermal_conductivity_mixture_wassiljewa",
    "viscosity_mixture_wilke",
]
