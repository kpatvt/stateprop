"""
Component: a pure fluid with the data needed to serve as a mixture constituent.

Stores the pure-component Helmholtz EOS (reusing the existing Fluid class from
stateprop) plus any GERG-specific overrides (e.g. the GERG ideal-gas form with
Cooper hyperbolic terms).

Physically, a component encapsulates:
  - Its critical parameters (T_c, rho_c, M) used in the reducing functions
  - Its pure-component residual Helmholtz alpha^r_oi(delta, tau), evaluated in
    mixture-reduced coordinates
  - Its pure-component ideal-gas alpha^0_oi(rho, T), evaluated in natural
    (rho, T) coordinates (not mixture-reduced)

GERG-2008 uses slightly different pure-component EOS than the stand-alone
Setzmann-Wagner etc. formulations. For the framework-first implementation we
allow either: use the existing Fluid JSON (sufficient for architecture testing
and flash demonstration), or supply GERG-specific coefficients (for strict
GERG-2008 compliance).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from ..fluid import Fluid, load_fluid


# Directory where component JSON files live (inside the package as of v0.6.2)
COMPONENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fluids", "components",
)


@dataclass
class Component:
    """A pure fluid for use as a mixture constituent.

    Holds a reference to a `Fluid` object for the pure-component EOS, plus
    any metadata needed for the mixture model (e.g. short name used to index
    into binary-parameter tables).

    Attributes
    ----------
    name         : canonical name (e.g. "methane")
    short_name   : short key (e.g. "CH4") used for binary-parameter lookup
    fluid        : the pure-component Fluid object (provides alpha_r, alpha_0)
    T_c, rho_c   : critical parameters used in reducing functions [K, mol/m^3]
    molar_mass   : [kg/mol]
    """
    name: str
    short_name: str
    fluid: Fluid
    T_c: float
    rho_c: float
    molar_mass: float

    @classmethod
    def from_fluid(cls, fluid: Fluid, short_name: Optional[str] = None):
        """Create a Component by wrapping an existing Fluid.

        Useful for quick testing: any pure fluid in the library can serve as
        a mixture component via its own Helmholtz EOS as alpha^r_oi.
        """
        return cls(
            name=fluid.name,
            short_name=short_name or fluid.name[:4].upper(),
            fluid=fluid,
            T_c=fluid.T_c,
            rho_c=fluid.rho_c,
            molar_mass=fluid.molar_mass,
        )

    def __repr__(self):
        return (f"Component({self.name!r} [{self.short_name}], "
                f"T_c={self.T_c:.2f}, rho_c={self.rho_c:.2f}, M={self.molar_mass*1000:.3f} g/mol)")


def load_component(name: str) -> Component:
    """Load a component by name.

    First tries the components/ directory for a GERG-specific JSON.
    Falls back to loading as a plain Fluid and wrapping it.
    """
    # Try components/ first
    path = os.path.join(COMPONENTS_DIR, f"{name.lower()}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # A component JSON may reference a pure-fluid JSON by name
        if "pure_fluid" in data:
            fluid = load_fluid(data["pure_fluid"])
        else:
            # Or it may embed the full fluid definition
            raise NotImplementedError(
                f"Embedded fluid definition in component '{name}' not yet supported. "
                f"Use 'pure_fluid': <name> to reference an existing fluid."
            )
        return Component(
            name=data.get("name", name),
            short_name=data.get("short_name", name[:4].upper()),
            fluid=fluid,
            T_c=data.get("T_c", fluid.T_c),
            rho_c=data.get("rho_c", fluid.rho_c),
            molar_mass=data.get("molar_mass", fluid.molar_mass),
        )

    # Fall back: treat name as a pure fluid
    fluid = load_fluid(name)
    return Component.from_fluid(fluid)
