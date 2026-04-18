"""
Phase envelope generation for single-component fluids.

For a pure fluid the "phase envelope" in (p, T) is the 1-D vapor-liquid
coexistence curve running from triple point to critical point. In other
coordinates (T-rho, p-h, T-s, p-v) it becomes a 2-D "dome" enclosing the
two-phase region.

This module traces the curve with an adaptive scheme that concentrates
points near the critical point, where most properties change rapidly.

Main function
-------------
    trace_phase_envelope(fluid, n_points=100) -> PhaseEnvelope

The ``PhaseEnvelope`` object holds parallel arrays for every common
thermodynamic property on both the saturated-liquid and saturated-vapor
branches, plus the vapor pressure curve.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from . import properties as props
from .saturation import saturation_pT


@dataclass
class PhaseEnvelope:
    """Vapor-liquid coexistence curve sampled at a series of temperatures.

    All per-point arrays have the same length.  Liquid-branch (``_L``) and
    vapor-branch (``_V``) arrays are aligned with ``T`` and ``p``.

    Units
    -----
    T     : K
    p     : Pa
    rho_* : mol/m^3
    u_*   : J/mol
    h_*   : J/mol
    s_*   : J/(mol K)

    If the fluid provides ``molar_mass``, mass-based versions are also
    populated (suffix ``_kg`` for density, or converted fields for the
    intensive ones via the helper methods).
    """
    fluid_name: str
    T_c: float
    p_c: float
    rho_c: float
    T_triple: float
    p_triple: float
    molar_mass: Optional[float]

    T:     np.ndarray = field(default_factory=lambda: np.zeros(0))
    p:     np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_L: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_V: np.ndarray = field(default_factory=lambda: np.zeros(0))
    u_L:   np.ndarray = field(default_factory=lambda: np.zeros(0))
    u_V:   np.ndarray = field(default_factory=lambda: np.zeros(0))
    h_L:   np.ndarray = field(default_factory=lambda: np.zeros(0))
    h_V:   np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_L:   np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_V:   np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ---- mass-based helpers ----
    def as_mass_based(self):
        """Return a dict of arrays in mass-based engineering units.
        Requires ``molar_mass`` to be set.
        """
        if self.molar_mass is None:
            raise ValueError("molar_mass is required for mass-based conversion")
        M = self.molar_mass
        return {
            "T":       self.T,                         # K
            "p":       self.p,                         # Pa
            "p_MPa":   self.p * 1e-6,
            "rho_L":   self.rho_L * M,                 # kg/m^3
            "rho_V":   self.rho_V * M,                 # kg/m^3
            "u_L":     self.u_L / M * 1e-3,            # kJ/kg
            "u_V":     self.u_V / M * 1e-3,
            "h_L":     self.h_L / M * 1e-3,            # kJ/kg
            "h_V":     self.h_V / M * 1e-3,
            "s_L":     self.s_L / M * 1e-3,            # kJ/(kg K)
            "s_V":     self.s_V / M * 1e-3,
            "h_vap":   (self.h_V - self.h_L) / M * 1e-3,  # kJ/kg
        }

    # ---- coordinate extraction for plotting ----
    def dome_coordinates(self, x_kind, y_kind):
        """Return (x, y) arrays for the coexistence dome in requested coordinates.

        ``x_kind`` and ``y_kind`` each one of:
          'T', 'p', 'p_MPa', 'rho', 'rho_kg', 'u', 'h', 's',
          'u_kg', 'h_kg', 's_kg'    (mass-based variants require molar_mass).

        For coordinate axes that differ on the two branches ('rho', 'h', 's',
        etc.), the dome is a closed loop: left side = liquid branch, right
        side = vapor branch, joined at the critical point.  The returned (x, y)
        walks liquid from triple to critical, then vapor from critical back to
        triple.
        """
        xL, xV = self._arrays_for(x_kind)
        yL, yV = self._arrays_for(y_kind)
        # Walk: liquid (triple -> critical), then vapor (critical -> triple).
        # We assume T is ordered increasing; check and reverse if not.
        if not np.all(np.diff(self.T) >= 0):
            order = np.argsort(self.T)
            xL, xV = xL[order], xV[order]
            yL, yV = yL[order], yV[order]
        x = np.concatenate([xL, xV[::-1]])
        y = np.concatenate([yL, yV[::-1]])
        return x, y

    def _arrays_for(self, kind):
        """Return (liquid_array, vapor_array) for the requested coordinate."""
        k = kind.lower()
        if k == "t":
            return self.T, self.T
        if k == "p":
            return self.p, self.p
        if k == "p_mpa":
            return self.p * 1e-6, self.p * 1e-6
        if k == "rho":
            return self.rho_L, self.rho_V
        if k == "rho_kg":
            if self.molar_mass is None:
                raise ValueError("molar_mass required")
            return self.rho_L * self.molar_mass, self.rho_V * self.molar_mass
        if k == "u":
            return self.u_L, self.u_V
        if k == "h":
            return self.h_L, self.h_V
        if k == "s":
            return self.s_L, self.s_V
        if k == "u_kg":
            if self.molar_mass is None:
                raise ValueError("molar_mass required")
            return self.u_L / self.molar_mass * 1e-3, self.u_V / self.molar_mass * 1e-3
        if k == "h_kg":
            if self.molar_mass is None:
                raise ValueError("molar_mass required")
            return self.h_L / self.molar_mass * 1e-3, self.h_V / self.molar_mass * 1e-3
        if k == "s_kg":
            if self.molar_mass is None:
                raise ValueError("molar_mass required")
            return self.s_L / self.molar_mass * 1e-3, self.s_V / self.molar_mass * 1e-3
        raise ValueError(f"Unknown coordinate kind: {kind!r}")


def _temperature_grid(T_min, T_c, n_points, critical_density=0.7):
    """Generate a non-uniform temperature grid from T_min to ~T_c.

    Points are concentrated toward T_c to resolve the critical region.
    Concretely we use a convex combination of:
      - a uniform grid in T
      - a grid uniform in  log(T_c - T)  (which clusters near T_c)
    with weight `critical_density` on the log-spaced half.

    We stop just short of T_c to keep the saturation solver robust.
    """
    T_end = T_c - 1e-3 * T_c       # ~0.1% below T_c
    T_lin = np.linspace(T_min, T_end, n_points)

    # log-spaced in (T_c - T)
    gap_max = T_c - T_min
    gap_min = T_c - T_end
    log_gaps = np.logspace(np.log10(gap_min), np.log10(gap_max), n_points)
    T_log = (T_c - log_gaps)[::-1]  # reverse so it's increasing in T

    # Blend
    w = float(critical_density)
    T_mix = (1.0 - w) * T_lin + w * T_log
    # Deduplicate and sort
    T_mix = np.unique(np.sort(T_mix))
    return T_mix


def trace_phase_envelope(fluid, n_points=80, T_min=None, critical_density=0.7):
    """Trace the full vapor-liquid coexistence curve for a single-component fluid.

    Parameters
    ----------
    fluid : Fluid
    n_points : int
        Approximate number of temperatures to sample. Actual count may be
        slightly less after deduplication.
    T_min : float or None
        Lowest temperature in K.  Defaults to ``fluid.T_triple + 0.1``
        if a triple-point temperature is available, else ``fluid.T_min + 0.1``.
    critical_density : float in [0, 1]
        How strongly to concentrate points near the critical point.
        0 => uniform in T. 1 => uniform in log(T_c - T). Default 0.7.

    Returns
    -------
    PhaseEnvelope
    """
    if T_min is None:
        T_min = fluid.T_triple + 0.1 if fluid.T_triple > 0 else fluid.T_min + 0.1

    T_grid = _temperature_grid(T_min, fluid.T_c, n_points, critical_density)

    T_out = []
    p_out = []
    rhoL_out, rhoV_out = [], []
    uL_out, uV_out = [], []
    hL_out, hV_out = [], []
    sL_out, sV_out = [], []

    for T in T_grid:
        try:
            rho_L, rho_V, p_sat = saturation_pT(T, fluid)
        except Exception:
            # Skip points where the saturation solver fails (close to critical)
            continue

        # Spurious-solution filter: enforce physical ordering rho_V < rho_c < rho_L
        # and enforce monotonic progression versus previous point (p, rho_L, rho_V
        # all change in a known direction as T rises).
        if rho_L <= fluid.rho_c or rho_V >= fluid.rho_c or rho_V <= 0:
            continue
        if T_out:
            # If a previous point exists, require p to increase with T and
            # rho_L to decrease, rho_V to increase. Very close to T_c, small
            # numerical errors in near-critical Newton can cause non-monotone
            # jumps; skip those points.
            if p_sat < p_out[-1] * 0.999:
                continue
            if rho_L > rhoL_out[-1] * 1.001:
                continue
            if rho_V < rhoV_out[-1] * 0.999:
                continue

        u_L = props.internal_energy(rho_L, T, fluid)
        u_V = props.internal_energy(rho_V, T, fluid)
        h_L = props.enthalpy(rho_L, T, fluid)
        h_V = props.enthalpy(rho_V, T, fluid)
        s_L = props.entropy(rho_L, T, fluid)
        s_V = props.entropy(rho_V, T, fluid)

        T_out.append(T)
        p_out.append(p_sat)
        rhoL_out.append(rho_L); rhoV_out.append(rho_V)
        uL_out.append(u_L); uV_out.append(u_V)
        hL_out.append(h_L); hV_out.append(h_V)
        sL_out.append(s_L); sV_out.append(s_V)

    # Optionally anchor the top of the dome at the critical point itself:
    # all branches merge there. We add (T_c, p_c, rho_c, ...) as the apex.
    if fluid.T_c not in T_out:
        T_out.append(fluid.T_c)
        p_out.append(fluid.p_c)
        rhoL_out.append(fluid.rho_c)
        rhoV_out.append(fluid.rho_c)
        # Caloric values at the critical point: compute directly
        try:
            uc = props.internal_energy(fluid.rho_c, fluid.T_c, fluid)
            hc = props.enthalpy(fluid.rho_c, fluid.T_c, fluid)
            sc = props.entropy(fluid.rho_c, fluid.T_c, fluid)
            uL_out.append(uc); uV_out.append(uc)
            hL_out.append(hc); hV_out.append(hc)
            sL_out.append(sc); sV_out.append(sc)
        except Exception:
            # If properties diverge here, pop the point back out
            T_out.pop(); p_out.pop()
            rhoL_out.pop(); rhoV_out.pop()

    return PhaseEnvelope(
        fluid_name=fluid.name,
        T_c=fluid.T_c, p_c=fluid.p_c, rho_c=fluid.rho_c,
        T_triple=fluid.T_triple, p_triple=fluid.p_triple,
        molar_mass=fluid.molar_mass,
        T=np.asarray(T_out),
        p=np.asarray(p_out),
        rho_L=np.asarray(rhoL_out),
        rho_V=np.asarray(rhoV_out),
        u_L=np.asarray(uL_out),
        u_V=np.asarray(uV_out),
        h_L=np.asarray(hL_out),
        h_V=np.asarray(hV_out),
        s_L=np.asarray(sL_out),
        s_V=np.asarray(sV_out),
    )
