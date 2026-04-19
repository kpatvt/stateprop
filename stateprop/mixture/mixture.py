"""
Mixture: a container of components + composition + reducing rules.

The Mixture is the user-facing object for mixture calculations. It holds:
  - The list of Component objects (pure fluids)
  - The composition vector x (mole fractions, length N)
  - The reducing-function calculator (KunzWagnerReducing)
  - An optional departure-function table (unused in this framework-first build)

Thermodynamic methods delegate to the `properties` module which does the
heavy lifting.
"""
import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .component import Component, load_component
from .reducing import KunzWagnerReducing, BinaryParams, make_reducing_from_components
from .departure import DepartureFunction


# Directory where mixture (binary) JSON files live
BINARIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "fluids", "binaries",
)


class Mixture:
    """An N-component fluid mixture.

    Parameters
    ----------
    components : list[Component]
        The N pure-component fluids.
    composition : array-like or None
        Initial composition (mole fractions). If None, defaults to equal
        composition 1/N for all components. Must sum to 1.
    binary : dict[(int, int), BinaryParams] or None
        Binary interaction parameters keyed by (i, j) with i < j and i,j
        indexing into `components`.
    """

    def __init__(self, components, composition=None, binary=None):
        self.components = list(components)
        self.N = len(self.components)
        if self.N < 1:
            raise ValueError("Mixture must have at least 1 component")

        self.names = [c.name for c in self.components]
        self.short_names = [c.short_name for c in self.components]
        self.T_c = np.array([c.T_c for c in self.components])
        self.rho_c = np.array([c.rho_c for c in self.components])
        self.molar_masses = np.array([c.molar_mass for c in self.components])

        self.binary = binary if binary is not None else {}
        self.reducing = KunzWagnerReducing(self.T_c, self.rho_c, self.binary)

        # Build departure-function table: {(i,j): (F_ij, DepartureFunction)}
        # Skip pairs without a departure block or with F=0.
        self.departures = {}
        for (i, j), bp in self.binary.items():
            if bp.departure is not None and bp.F != 0.0:
                self.departures[(i, j)] = (bp.F, bp.departure)

        # Composition
        if composition is None:
            composition = np.full(self.N, 1.0 / self.N)
        self.set_composition(composition)

    def set_composition(self, x):
        """Set the composition. Values are normalized to sum to 1."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.N,):
            raise ValueError(f"Composition must have length N={self.N}, got {x.shape}")
        s = x.sum()
        if s <= 0:
            raise ValueError("Composition must have positive sum")
        self.x = x / s

    def molar_mass(self, x=None):
        """Average molar mass at composition x (or self.x if None)."""
        xv = self.x if x is None else np.asarray(x, dtype=np.float64)
        return float(np.dot(xv, self.molar_masses))

    def reduce(self, x=None):
        """Reducing-function values (T_r, rho_r) at composition x."""
        xv = self.x if x is None else np.asarray(x, dtype=np.float64)
        return self.reducing.evaluate(xv)

    def delta_tau(self, rho, T, x=None):
        """Reduced density delta = rho/rho_r(x) and inverse temperature tau = T_r(x)/T."""
        Tr, rho_r = self.reduce(x)
        return rho / rho_r, Tr / T

    def __repr__(self):
        comp_str = ", ".join(f"{c.short_name}={self.x[i]:.4f}" for i, c in enumerate(self.components))
        return f"Mixture(N={self.N}: {comp_str})"


def load_mixture(component_names, composition=None, binary_set=None):
    """Convenience builder: load components by name and set up binary parameters.

    Parameters
    ----------
    component_names : list[str]
        Component names to load (e.g. ["methane", "nitrogen", "carbondioxide"]).
    composition : array-like or None
        Initial mole fractions. Defaults to 1/N each.
    binary_set : str or None
        If given, load a named binary-parameter set (by name of JSON file
        under fluids/binaries/). Missing pairs default to ideal mixing.

    Returns
    -------
    Mixture
    """
    components = [load_component(name) for name in component_names]
    binary = _load_binary_parameters(components, binary_set) if binary_set else {}
    return Mixture(components, composition, binary)


def _load_binary_parameters(components, binary_set: str) -> Dict[Tuple[int, int], BinaryParams]:
    """Load binary interaction parameters from a JSON file.

    The JSON file lists pairs by component short_name or full name.
    Each entry may include:
      - Reducing-function parameters: beta_T, gamma_T, beta_v, gamma_v
      - Departure-function: F and a "departure" block with polynomial/exponential
        term tables (see DepartureFunction.from_dict).

    The departure function may be defined inline per-pair, or referenced by
    a name if the JSON carries a shared "departure_functions" table.
    """
    path = os.path.join(BINARIES_DIR, f"{binary_set}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)

    # Shared departure-function library (by name)
    dep_library = {}
    for name, dep_data in data.get("departure_functions", {}).items():
        dep_library[name] = DepartureFunction.from_dict(dep_data, pair=("shared", name))

    # Build a lookup by short_name -> index (case-insensitive)
    idx = {c.short_name.lower(): i for i, c in enumerate(components)}
    name_idx = {c.name.lower(): i for i, c in enumerate(components)}

    binary = {}
    for entry in data.get("pairs", []):
        a, b = entry["pair"]
        ia = idx.get(a.lower(), name_idx.get(a.lower()))
        ib = idx.get(b.lower(), name_idx.get(b.lower()))
        if ia is None or ib is None:
            continue
        i, j = (ia, ib) if ia < ib else (ib, ia)

        # Build departure function if present
        departure = None
        if "departure" in entry:
            if isinstance(entry["departure"], str):
                # Reference to shared library
                departure = dep_library.get(entry["departure"])
            else:
                # Inline definition
                departure = DepartureFunction.from_dict(
                    entry["departure"], pair=(a, b)
                )

        binary[(i, j)] = BinaryParams(
            beta_T=entry.get("beta_T", 1.0),
            gamma_T=entry.get("gamma_T", 1.0),
            beta_v=entry.get("beta_v", 1.0),
            gamma_v=entry.get("gamma_v", 1.0),
            F=entry.get("F", 0.0),
            departure=departure,
        )
    return binary
