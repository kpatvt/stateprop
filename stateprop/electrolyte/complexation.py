"""Aqueous complexation framework (v0.9.102).

Solves for the speciation of an aqueous solution given total component
molalities, accounting for ion-pair complexes (CaSO4°, MgCO3°,
NaSO4⁻, etc.).  This addresses the two main limitations of v0.9.101:

  1. CaSO4-NaCl salting-in (real systems have CaSO4° ion pair)
  2. Seawater carbonate SI (real systems have CaCO3°, MgCO3° complexes
     that reduce free CO3²⁻ by ~10×)

Theory
------
For a complex ML formed from components M and L:

    M^z_M + L^z_L  ⇌  ML^(z_M+z_L)

the dissociation constant is:

    K_diss = a_M · a_L / a_ML
           = (γ_M · m_M_free) · (γ_L · m_L_free) / (γ_ML · m_ML)

We use the *dissociation* form (K_diss < 1 for stable complexes) for
consistency with the geochemistry literature (PHREEQC, EQ3/6).
The reciprocal K_assoc = 1/K_diss is sometimes used.

Mass balance for component j:

    m_j_total = m_j_free + Σ_i (ν_ji · m_complex_i)

where ν_ji is the stoichiometric coefficient of component j in
complex i.

The solver uses Newton iteration in log-space with x_j = log10(m_j_free).
Complex concentrations follow directly from mass action; mass balance
residuals form N nonlinear equations in N unknowns (where N is the
number of components participating in any complex).

Activity coefficients
---------------------
* Main ions (those in the supplied MultiPitzerSystem) use Pitzer γ
* Charged complexes (NaSO4⁻, CaHCO3⁺, MgOH⁺, etc.) use Davies:
    log γ = -A · z² · (√I / (1 + √I) - 0.3·I)
* Neutral complexes (CaSO4°, CaCO3°, MgCO3°, etc.) have γ = 1

The Davies equation is accurate to ~5% for I < 0.5 mol/kg.  For
high-I systems the dominant γ effect is on main ions which use
Pitzer (calibrated to high I), so the overall accuracy is good.

K_sp consistency
----------------
The v0.9.101 Mineral database uses *apparent* (stoichiometric) K_sp
values calibrated against total-concentration solubility measurements.
When using explicit complexation, *thermodynamic* (free-ion) K_sp
values must be used to avoid double-counting.  The Mineral dataclass
now carries an optional `log_K_sp_25_thermo` field; if set, it is
used by SpeciationResult.saturation_index().

References
----------
* Truesdell, A. H., Jones, B. F. (1974). WATEQ — a computer program
  for calculating chemical equilibria of natural waters. J. Res. USGS
  2, 233.
* Plummer, L. N., Parkhurst, D. L., et al. (1990).  PHREEQE — a
  computer program for geochemical calculations.  USGS WRI 80-96.
* Stumm, W., Morgan, J. J. (1996).  Aquatic Chemistry, 3rd ed.
* Millero, F. J. (1995).  Thermodynamics of the carbon dioxide system
  in the oceans.  Geochim. Cosmochim. Acta 59, 661.
* Plummer, L. N., Busenberg, E. (1982).  K_assoc for CaCO3°, MgCO3°,
  CaHCO3⁺, MgHCO3⁺.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union, TYPE_CHECKING
import numpy as np

from .multi_pitzer import MultiPitzerSystem

if TYPE_CHECKING:
    from .minerals import Mineral

_R = 8.314462618    # J/(mol·K)
_LN10 = np.log(10.0)
_T_REF = 298.15


# =====================================================================
# Complex dataclass
# =====================================================================

@dataclass
class Complex:
    """Aqueous ion-pair complex.

    Parameters
    ----------
    name : str
        Identifier for the complex (e.g. ``"CaSO4°"``, ``"NaSO4-"``).
        Use ``°`` for neutrals, ``+``/``-`` for charged.
    components : dict[str, int]
        Component stoichiometry, e.g. {"Ca++": 1, "SO4--": 1} for
        CaSO4°.  Keys must match ion names in the MultiPitzerSystem.
    charge : int
        Net charge of the complex (0, ±1, ±2 typical).
    log_K_diss_25 : float
        log10(K_diss) at 298.15 K, where K_diss is defined such that
        K_diss = a_components / a_complex.  Smaller (more negative)
        means stronger ion-pair association.
    delta_H_rxn : float, default 0.0
        Enthalpy of dissociation [J/mol].  Used for van't Hoff
        T-dependence: log_K(T) = log_K_25 - (ΔH / R·ln10) · (1/T - 1/Tref).
        Most ion-pair dissociation reactions are endothermic
        (ΔH > 0), so K_diss increases with T (complex is less stable
        at higher T).
    """
    name: str
    components: Dict[str, int]
    charge: int
    log_K_diss_25: float
    delta_H_rxn: float = 0.0

    @property
    def log_K_assoc_25(self) -> float:
        """log_K_assoc = -log_K_diss (K_assoc = 1/K_diss)."""
        return -self.log_K_diss_25

    def log_K_diss(self, T: float) -> float:
        """log10(K_diss) at T via van't Hoff."""
        if self.delta_H_rxn == 0.0:
            return self.log_K_diss_25
        return (self.log_K_diss_25
                 - (self.delta_H_rxn / (_R * _LN10))
                   * (1.0 / T - 1.0 / _T_REF))

    def K_diss(self, T: float) -> float:
        return 10.0 ** self.log_K_diss(T)

    def log_K_assoc(self, T: float) -> float:
        return -self.log_K_diss(T)

    def K_assoc(self, T: float) -> float:
        return 10.0 ** self.log_K_assoc(T)


# =====================================================================
# Bundled complex database
# =====================================================================
# Sources:
#   PHREEQC llnl.dat / minteq.v4.dat (USGS, Parkhurst-Appelo 2013)
#   Plummer-Busenberg 1982 (carbonate ion pairs)
#   Reardon 1990 (NaCO3⁻)
#   Cooke 1989 (CaSO4° ΔH)

_COMPLEX_DB: Dict[str, Complex] = {
    # ---------- Sulfate ion pairs ----------
    "CaSO4o": Complex(
        name="CaSO4°", components={"Ca++": 1, "SO4--": 1}, charge=0,
        log_K_diss_25=-2.30, delta_H_rxn=6900.0),
    "MgSO4o": Complex(
        name="MgSO4°", components={"Mg++": 1, "SO4--": 1}, charge=0,
        log_K_diss_25=-2.26, delta_H_rxn=5400.0),
    "NaSO4-": Complex(
        name="NaSO4-", components={"Na+": 1, "SO4--": 1}, charge=-1,
        log_K_diss_25=-0.70, delta_H_rxn=1100.0),
    "KSO4-": Complex(
        name="KSO4-", components={"K+": 1, "SO4--": 1}, charge=-1,
        log_K_diss_25=-0.85),
    # ---------- Carbonate ion pairs ----------
    "CaCO3o": Complex(
        name="CaCO3°", components={"Ca++": 1, "CO3--": 1}, charge=0,
        log_K_diss_25=-3.22, delta_H_rxn=14600.0),
    "MgCO3o": Complex(
        name="MgCO3°", components={"Mg++": 1, "CO3--": 1}, charge=0,
        log_K_diss_25=-2.98, delta_H_rxn=11500.0),
    "NaCO3-": Complex(
        name="NaCO3-", components={"Na+": 1, "CO3--": 1}, charge=-1,
        log_K_diss_25=-1.27),
    # ---------- Bicarbonate complexes ----------
    "CaHCO3+": Complex(
        name="CaHCO3+", components={"Ca++": 1, "HCO3-": 1}, charge=+1,
        log_K_diss_25=-1.11, delta_H_rxn=12300.0),
    "MgHCO3+": Complex(
        name="MgHCO3+", components={"Mg++": 1, "HCO3-": 1}, charge=+1,
        log_K_diss_25=-1.07, delta_H_rxn=10000.0),
    # ---------- Hydroxide complexes ----------
    "CaOH+": Complex(
        name="CaOH+", components={"Ca++": 1, "OH-": 1}, charge=+1,
        log_K_diss_25=-1.30),
    "MgOH+": Complex(
        name="MgOH+", components={"Mg++": 1, "OH-": 1}, charge=+1,
        log_K_diss_25=-2.58),
}


def lookup_complex(name: str) -> Complex:
    """Look up a Complex by name (case-insensitive, accepts variants
    like 'CaSO4°', 'CaSO4o')."""
    key = name.replace("°", "o").replace(" ", "")
    if key in _COMPLEX_DB:
        return _COMPLEX_DB[key]
    # Try lowercase
    for k, v in _COMPLEX_DB.items():
        if k.lower() == key.lower() or v.name.lower() == name.lower():
            return v
    raise KeyError(f"Unknown complex {name!r}. "
                    f"Available: {sorted(_COMPLEX_DB.keys())}")


def list_complexes() -> List[str]:
    """Return alphabetical list of bundled complex names."""
    return sorted(c.name for c in _COMPLEX_DB.values())


# =====================================================================
# Davies equation for charged complex γ
# =====================================================================

def _davies_log_gamma(z: int, I: float, T: float = 298.15) -> float:
    """Davies-equation log10 γ for a charged species.

    log γ = -A(T) · z² · (√I / (1 + √I) - 0.3·I)

    Accurate to ~5% for I < 0.5 mol/kg.  The Debye-Hückel A constant
    has slight T-dependence: A(25 °C) = 0.5093, A(50 °C) ≈ 0.532,
    A(100 °C) ≈ 0.598.  Linear-in-T approximation:
    A ≈ 0.509 + 0.001·(T - 298.15) [rough].
    """
    if z == 0 or I <= 0:
        return 0.0
    A = 0.509 + 0.001 * (T - _T_REF)
    sqrt_I = np.sqrt(I)
    return -A * z * z * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I)


# =====================================================================
# Speciation result
# =====================================================================

@dataclass
class SpeciationResult:
    """Result of a Speciation.solve() call.

    Attributes
    ----------
    free : dict[str, float]
        Free-ion molalities (post-complexation) [mol/kg]. Includes both
        component ions (Ca²⁺, SO4²⁻, etc.) and pass-through ions
        (Cl⁻ when not complexed).
    complexes : dict[str, float]
        Aqueous-complex molalities [mol/kg].
    gammas : dict[str, float]
        Activity coefficients for free ions and charged complexes.
        Neutral complexes have γ = 1 (not in this dict).
    a_w : float
        Water activity at the converged speciation.
    I : float
        Ionic strength [mol/kg], including charged-complex contribution.
    converged : bool
    iterations : int
    T : float
        Temperature [K] at which speciation was solved.
    """
    free: Dict[str, float]
    complexes: Dict[str, float]
    gammas: Dict[str, float]
    a_w: float
    I: float
    converged: bool
    iterations: int
    T: float = 298.15

    def saturation_index(self,
                          mineral: Union[str, "Mineral"]) -> float:
        """Compute SI = log10(IAP / K_sp) using free-ion concentrations.

        Uses the mineral's `log_K_sp_25_thermo` if set (preferred for
        complexation work); otherwise falls back to `log_K_sp_25`
        (apparent K_sp; double-counts complexation, will give
        systematically negative-biased SI).
        """
        from .minerals import lookup_mineral, saturation_index
        if isinstance(mineral, str):
            mineral = lookup_mineral(mineral)

        # Use thermodynamic K_sp if available
        log_K_thermo = getattr(mineral, "log_K_sp_25_thermo", None)
        if log_K_thermo is not None:
            from .minerals import Mineral
            shifted = Mineral(
                name=mineral.name,
                formula=mineral.formula,
                cations=mineral.cations,
                anions=mineral.anions,
                n_H2O=mineral.n_H2O,
                log_K_sp_25=log_K_thermo,
                delta_H_rxn=mineral.delta_H_rxn,
                binary_salt=mineral.binary_salt)
            return saturation_index(shifted, self.free, self.gammas,
                                       T=self.T, a_w=self.a_w)
        return saturation_index(mineral, self.free, self.gammas,
                                   T=self.T, a_w=self.a_w)


# =====================================================================
# Speciation solver
# =====================================================================

class Speciation:
    """Aqueous speciation calculator with mass-action ion-pairing.

    Given total component molalities (e.g., total Ca = m_Ca²⁺ +
    m_CaSO4° + m_CaCO3° + m_CaHCO3⁺ + m_CaOH⁺), this class solves the
    coupled nonlinear system for free-ion and aqueous-complex
    concentrations.

    The Newton solver iterates in log-space (x_j = log10(m_j_free)) and
    uses the analytical Jacobian.  Activity coefficients are updated
    once per outer iteration; the inner Newton converges quickly.

    Parameters
    ----------
    pitzer : MultiPitzerSystem
        Provides activity coefficients for the main (Pitzer-modelled)
        ions and water activity.
    complexes : sequence of str or Complex
        Aqueous complexes to include in the speciation.  Strings are
        looked up in the bundled database; Complex instances are
        used directly (including user-supplied ones).

    Examples
    --------
    >>> from stateprop.electrolyte import (
    ...     MultiPitzerSystem, Speciation,
    ... )
    >>> pitzer = MultiPitzerSystem.from_salts(["NaCl", "Na2SO4"])
    >>> spec = Speciation(pitzer, ["NaSO4-"])
    >>> result = spec.solve({"Na+": 1.0, "Cl-": 0.5, "SO4--": 0.25})
    >>> result.free["SO4--"]
    0.235     # ~5% complexed as NaSO4-
    >>> result.complexes["NaSO4-"]
    0.015
    """

    def __init__(self,
                  pitzer: MultiPitzerSystem,
                  complexes: Sequence[Union[str, Complex]]):
        self.pitzer = pitzer
        self.complexes: List[Complex] = []
        for c in complexes:
            self.complexes.append(
                lookup_complex(c) if isinstance(c, str) else c)
        # Components that participate in any complex
        self._components: List[str] = sorted(set(
            comp for c in self.complexes for comp in c.components))

    def solve(self,
                totals: Dict[str, float],
                T: float = 298.15,
                max_iter: int = 100,
                tol: float = 1e-8,
                verbose: bool = False) -> SpeciationResult:
        """Solve for free-ion and complex concentrations.

        Parameters
        ----------
        totals : dict
            Total component molalities {ion: mol/kg}.  Ions not
            participating in any complex pass through unchanged.
        T : float, default 298.15
            Temperature [K].
        max_iter : int, default 100
        tol : float, default 1e-8
            Relative convergence tolerance on mass-balance residuals.
        verbose : bool, default False
            Print iteration progress.

        Returns
        -------
        SpeciationResult
        """
        # Separate components (in complexes) from passthroughs (not)
        component_totals = {k: v for k, v in totals.items()
                              if k in self._components}
        passthrough = {k: v for k, v in totals.items()
                          if k not in self._components}

        # Trivial case: no complexation needed
        if not component_totals:
            full_m = dict(passthrough)
            gammas = self.pitzer.gammas(full_m, T)
            a_w = self.pitzer.water_activity(full_m, T)
            I = self.pitzer.ionic_strength(full_m)
            return SpeciationResult(
                free=full_m, complexes={}, gammas=gammas, a_w=a_w,
                I=I, converged=True, iterations=0, T=T)

        # Initial guess: free = total (zero complexation)
        N = len(self._components)
        x = np.array([np.log10(max(component_totals[c], 1e-15))
                       for c in self._components])

        complex_m: Dict[str, float] = {}
        gammas: Dict[str, float] = {}
        a_w = 1.0
        I = 0.0
        converged = False
        last_resid = np.inf

        for outer in range(max_iter):
            # 1. Build current free-ion dict
            free = {c: 10.0 ** x[i] for i, c in enumerate(self._components)}
            full_m = {**passthrough, **free}

            # 2. Compute γ (Pitzer) and a_w using current free + passthroughs
            #    Charged complexes contribute to I but not the Pitzer dict.
            gammas = self.pitzer.gammas(full_m, T)
            a_w = self.pitzer.water_activity(full_m, T)

            # 3. Compute ionic strength including charged complexes from
            #    the previous iteration (zero on iter 0).
            I = self.pitzer.ionic_strength(full_m)
            for c in self.complexes:
                if c.charge != 0 and c.name in complex_m:
                    I += 0.5 * complex_m[c.name] * c.charge ** 2

            # 4. Compute complex molalities from mass action.
            #    m_complex = K_assoc · Π (γ_comp · m_comp)^ν / γ_complex
            complex_m = {}
            for c in self.complexes:
                K_assoc = c.K_assoc(T)
                if c.charge == 0:
                    gamma_c = 1.0
                else:
                    gamma_c = 10.0 ** _davies_log_gamma(c.charge, I, T)
                num = 1.0
                for comp, nu in c.components.items():
                    g = gammas.get(comp, 1.0)
                    m = free.get(comp, full_m.get(comp, 0.0))
                    num *= (g * m) ** nu
                complex_m[c.name] = K_assoc * num / gamma_c

            # 5. Mass-balance residuals (component j):
            #    F_j = m_total_j - m_free_j - Σ ν_ji · m_complex_i
            F = np.zeros(N)
            for j, comp_j in enumerate(self._components):
                m_in_complexes = sum(
                    c.components[comp_j] * complex_m[c.name]
                    for c in self.complexes if comp_j in c.components)
                F[j] = component_totals[comp_j] - free[comp_j] - m_in_complexes

            # 6. Convergence check (relative residual)
            max_rel = max(
                abs(F[j]) / max(component_totals[c], 1e-15)
                for j, c in enumerate(self._components))

            if verbose:
                print(f"  iter {outer:3d}: max_rel_resid = {max_rel:.3e}, "
                       f"I = {I:.4f}")

            if max_rel < tol:
                converged = True
                break

            # 7. Build Jacobian (γ treated constant within Newton step):
            #    ∂F_j/∂x_l = -ln10 · (δ_jl · m_free_l + Σ_i ν_ji · ν_li · m_complex_i)
            J = np.zeros((N, N))
            for j, cj in enumerate(self._components):
                J[j, j] -= _LN10 * free[cj]
                for c in self.complexes:
                    if cj not in c.components:
                        continue
                    nu_j = c.components[cj]
                    for l, cl in enumerate(self._components):
                        if cl not in c.components:
                            continue
                        nu_l = c.components[cl]
                        J[j, l] -= _LN10 * nu_j * nu_l * complex_m[c.name]

            # 8. Newton step  J · dx = -F
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # Singular Jacobian (shouldn't happen for well-posed cases) —
                # fall back to safe Picard step.
                dx = np.zeros(N)
                for j, cj in enumerate(self._components):
                    target = max(component_totals[cj] - sum(
                        c.components[cj] * complex_m[c.name]
                        for c in self.complexes
                        if cj in c.components), 1e-15)
                    dx[j] = (np.log10(target) - x[j]) * 0.5

            # 9. Damp the step to prevent overshoot in log space (cap |dx|≤1)
            step_max = float(np.max(np.abs(dx)))
            damp = min(1.0, 1.0 / step_max) if step_max > 1.0 else 1.0
            x = x + damp * dx

            # Detect divergence (residual increasing significantly)
            if outer > 5 and max_rel > 10 * last_resid:
                # Reset to safer starting point
                x = np.array([np.log10(max(component_totals[c] * 0.5, 1e-15))
                                for c in self._components])
            last_resid = max_rel

        # Final assembly
        free = {c: 10.0 ** x[i] for i, c in enumerate(self._components)}
        full_m = {**passthrough, **free}
        gammas = self.pitzer.gammas(full_m, T)
        a_w = self.pitzer.water_activity(full_m, T)
        # Add Davies γ for charged complexes
        for c in self.complexes:
            if c.charge != 0:
                gammas[c.name] = 10.0 ** _davies_log_gamma(c.charge, I, T)

        return SpeciationResult(
            free=free, complexes=complex_m, gammas=gammas, a_w=a_w,
            I=I, converged=converged, iterations=outer + 1, T=T)


# =====================================================================
# Convenience: solve_speciation
# =====================================================================

def solve_speciation(totals: Dict[str, float],
                       pitzer: MultiPitzerSystem,
                       complexes: Sequence[Union[str, Complex]],
                       T: float = 298.15,
                       **kwargs) -> SpeciationResult:
    """Convenience wrapper: build a Speciation and solve it in one call.

    Useful for one-off calculations.  For repeated calls with the same
    pitzer/complexes config, build a Speciation instance directly.
    """
    return Speciation(pitzer, complexes).solve(totals, T=T, **kwargs)
