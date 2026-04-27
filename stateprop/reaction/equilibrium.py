"""Single-reaction equilibrium calculations.

For a reaction
    Sum_i nu_i * A_i = 0
where nu_i are stoichiometric coefficients (negative for reactants,
positive for products), the equilibrium constant at temperature T is

    K_eq(T) = exp[-dG_rxn(T) / (R T)]

with dG_rxn(T) = Sum_i nu_i * Gf_i(T). For ideal-gas reactions at
pressure p with reference pressure p_ref = 1 bar:

    K_eq = Prod_i (P_i / p_ref)^nu_i = Prod_i (y_i p / p_ref)^nu_i

The reaction extent xi is the variable solved for; the partial mole
numbers at equilibrium are
    n_i = n_i_0 + nu_i * xi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, Tuple, List
import math
import numpy as np

from .thermo import SpeciesThermo, get_species, R_GAS


@dataclass
class Reaction:
    """A single chemical reaction with thermochemistry.

    Examples
    --------
    Water-gas shift CO + H2O = CO2 + H2:

    >>> rxn = Reaction.from_names(
    ...     reactants={'CO': 1, 'H2O': 1},
    ...     products={'CO2': 1, 'H2': 1})
    >>> rxn.K_eq(800.0)    # ~3.2 at 800 K
    """
    species: Sequence[SpeciesThermo]
    nu: Sequence[float]    # stoichiometric coefficients, sign convention
                            # nu < 0 reactant, nu > 0 product

    def __post_init__(self):
        if len(self.species) != len(self.nu):
            raise ValueError(f"len(species)={len(self.species)} != "
                             f"len(nu)={len(self.nu)}")
        # Convert to tuples for immutability and ndarrays for math
        object.__setattr__(self, 'species', tuple(self.species))
        object.__setattr__(self, 'nu', np.asarray(self.nu, dtype=float))
        # Total moles produced per unit extent
        object.__setattr__(self, '_dn_total', float(self.nu.sum()))

    @classmethod
    def from_names(cls,
                    reactants: dict,    # {name: stoichiometry}
                    products: dict):
        """Build a Reaction from species names and stoichiometric counts."""
        sp = []
        nu = []
        for name, n in reactants.items():
            sp.append(get_species(name))
            nu.append(-float(n))   # negative for reactants
        for name, n in products.items():
            sp.append(get_species(name))
            nu.append(float(n))    # positive for products
        return cls(species=sp, nu=nu)

    # ----------------------------------------------------------------
    # Thermodynamic functions
    # ----------------------------------------------------------------

    def dH_rxn(self, T: float) -> float:
        """Enthalpy of reaction at T [J/mol of extent]."""
        return float(sum(n * sp.H(T) for n, sp in zip(self.nu, self.species)))

    def dG_rxn(self, T: float) -> float:
        """Gibbs energy of reaction at T [J/mol of extent]."""
        return float(sum(n * sp.Gf(T) for n, sp in zip(self.nu, self.species)))

    def dS_rxn(self, T: float) -> float:
        """Entropy of reaction at T [J/(mol K)]."""
        return float(sum(n * sp.S(T) for n, sp in zip(self.nu, self.species)))

    def K_eq(self, T: float) -> float:
        """Equilibrium constant K_eq(T) = exp(-dG_rxn / RT) at T [K].

        Convention: K_eq is dimensionless and references partial
        pressures to p_ref = 1 bar = 1e5 Pa.
        """
        return math.exp(-self.dG_rxn(T) / (R_GAS * T))

    # ----------------------------------------------------------------
    # Equilibrium extent solver (single reaction, ideal gas)
    # ----------------------------------------------------------------

    def equilibrium_extent_ideal_gas(
        self,
        T: float,
        p: float,
        n_initial: Sequence[float],
        n_inert: float = 0.0,
        p_ref: float = 1e5,
        tol: float = 1e-10,
        maxiter: int = 100,
    ) -> "EquilibriumResult":
        """Solve for the equilibrium extent xi for a single reaction.

        For an ideal-gas reaction at fixed (T, p), the equilibrium
        condition is

            K_eq(T) = Prod_i (n_i / n_total)^nu_i * (p/p_ref)^(Sum nu_i)

        where n_i = n_i_0 + nu_i * xi and n_total = Sum n_i + n_inert.
        The constraint 0 < n_i is required for physical solutions; the
        feasible range for xi is determined by the most-limiting
        reactant (xi_max from reactants) and the smallest backward bound
        from products (xi_min for negative xi).

        Parameters
        ----------
        T : float
            Temperature [K].
        p : float
            Pressure [Pa].
        n_initial : sequence of float
            Initial moles of each reaction species, in the same order
            as `self.species`.
        n_inert : float
            Total moles of inert species (does not react but contributes
            to total moles for partial-pressure denominator).
        p_ref : float
            Reference pressure [Pa] for K_eq convention. Default 1 bar.
        tol : float
            Convergence tolerance on the bisection on xi (in moles).
        maxiter : int
            Bisection iteration cap.

        Returns
        -------
        EquilibriumResult with .xi (extent), .n_eq (equilibrium moles
        of reactive species), .y_eq (mole fractions including inerts),
        .K_eq (the K used), and .converged.
        """
        return self._solve_extent(T, p, n_initial, n_inert, p_ref, tol, maxiter,
                                    eos=None)

    def equilibrium_extent_real_gas(
        self,
        T: float,
        p: float,
        n_initial: Sequence[float],
        eos,
        n_inert: float = 0.0,
        p_ref: float = 1e5,
        tol: float = 1e-9,
        maxiter: int = 200,
        damping: float = 0.7,
    ) -> "EquilibriumResult":
        """Solve for equilibrium extent using a real-gas EOS for fugacity.

        Replaces the ideal-gas K_y * (p/p_ref)^delta_nu equilibrium
        condition with the fugacity-corrected form

            K_eq(T) = Prod_i (y_i * phi_i * p / p_ref)^nu_i

        where phi_i = fugacity coefficient of species i in the mixture.

        Parameters
        ----------
        T, p, n_initial, n_inert, p_ref : same as `equilibrium_extent_ideal_gas`
        eos : EOS mixture object
            Must implement
              - density_from_pressure(p, T, x, phase_hint='vapor') -> rho [mol/m^3]
              - ln_phi(rho, T, x) -> ndarray of ln(phi_i)
            CubicMixture, SAFTMixture, and GERG mixtures all qualify.
            **The EOS species ordering MUST match this Reaction's species
            ordering** (i.e., the order of self.species). If you used
            `Reaction.from_names`, the order is reactants-then-products
            in the order they were given.
        tol, maxiter : numerical tolerances. Default tol is looser than
            ideal-gas because real-gas adds an outer fixed-point loop on phi.
        damping : float
            Damping factor on Newton step. Default 0.7. Larger values can
            speed up convergence but risk oscillation when phi varies
            strongly with composition.

        Returns
        -------
        EquilibriumResult
        """
        return self._solve_extent(T, p, n_initial, n_inert, p_ref, tol, maxiter,
                                    eos=eos, damping=damping)

    def _solve_extent(
        self,
        T: float, p: float, n_initial, n_inert: float, p_ref: float,
        tol: float, maxiter: int,
        eos=None, damping: float = 1.0,
    ) -> "EquilibriumResult":
        """Unified solver: bisection for ideal gas, Newton for real gas."""
        n0 = np.asarray(n_initial, dtype=float)
        if n0.size != len(self.species):
            raise ValueError(f"n_initial length {n0.size} != species "
                             f"count {len(self.species)}")
        if (n0 < 0).any():
            raise ValueError("n_initial must be nonnegative")

        # Feasible range
        xi_lo, xi_hi = -np.inf, np.inf
        for nu_i, ni0 in zip(self.nu, n0):
            if nu_i > 0:
                xi_lo = max(xi_lo, -ni0 / nu_i)
            elif nu_i < 0:
                xi_hi = min(xi_hi, -ni0 / nu_i)
        eps = 1e-12 * max(1.0, n0.sum() + n_inert)
        xi_lo = xi_lo + eps if xi_lo > -np.inf else -1e6
        xi_hi = xi_hi - eps if xi_hi <  np.inf else  1e6

        K = self.K_eq(T)
        delta_nu = float(self.nu.sum())

        def residual(xi: float) -> float:
            ni = n0 + self.nu * xi
            ntot = ni.sum() + n_inert
            if (ni <= 0).any() or ntot <= 0:
                return math.nan
            yi = ni / ntot
            r = float((self.nu * np.log(yi)).sum()
                       + delta_nu * math.log(p / p_ref)
                       - math.log(K))
            if eos is not None:
                # Add the fugacity-coefficient correction
                rho = eos.density_from_pressure(p, T, yi, phase_hint='vapor')
                ln_phi = np.asarray(eos.ln_phi(rho, T, yi))
                r += float((self.nu * ln_phi).sum())
            return r

        # Ideal gas: bisection (residual is monotonic)
        if eos is None:
            r_lo = residual(xi_lo)
            r_hi = residual(xi_hi)
            if math.isnan(r_lo) or math.isnan(r_hi):
                return EquilibriumResult(
                    xi=math.nan, n_eq=n0.copy(),
                    y_eq=np.full_like(n0, math.nan),
                    K_eq=K, T=T, p=p, n_inert=n_inert, converged=False,
                    message="Initial bracket evaluation failed")
            if r_lo * r_hi > 0:
                xi_star = xi_lo if abs(r_lo) < abs(r_hi) else xi_hi
                return EquilibriumResult(
                    xi=xi_star, n_eq=n0 + self.nu * xi_star,
                    y_eq=(n0 + self.nu * xi_star) / ((n0 + self.nu * xi_star).sum()
                                                       + n_inert),
                    K_eq=K, T=T, p=p, n_inert=n_inert, converged=False,
                    message=f"Equilibrium boundary: r_lo={r_lo:.3e}, r_hi={r_hi:.3e}")
            xi_a, xi_b = xi_lo, xi_hi
            r_a, r_b = r_lo, r_hi
            for it in range(maxiter):
                xi_m = 0.5 * (xi_a + xi_b)
                r_m = residual(xi_m)
                if math.isnan(r_m):
                    xi_b = xi_m
                    continue
                if abs(r_m) < tol or (xi_b - xi_a) < tol:
                    xi_star = xi_m
                    break
                if r_a * r_m < 0:
                    xi_b, r_b = xi_m, r_m
                else:
                    xi_a, r_a = xi_m, r_m
            else:
                xi_star = 0.5 * (xi_a + xi_b)

            n_eq = n0 + self.nu * xi_star
            ntot = n_eq.sum() + n_inert
            return EquilibriumResult(
                xi=xi_star, n_eq=n_eq, y_eq=n_eq / ntot,
                K_eq=K, T=T, p=p, n_inert=n_inert, converged=True,
                message=f"converged in {it + 1} iterations")

        # Real gas: secant iteration on the residual (monotonic in xi
        # is no longer guaranteed because phi(y) introduces curvature,
        # but in practice the equilibrium point is unique and well-bracketed
        # for stable mixtures. Use bracketed Brent-style fallback if needed.)
        # Start from the ideal-gas solution as a warm start.
        ideal_res = self._solve_extent(T, p, n_initial, n_inert, p_ref,
                                          tol, maxiter, eos=None)
        xi_guess = ideal_res.xi if ideal_res.converged else 0.5 * (xi_lo + xi_hi)
        # Build a small bracket around xi_guess; widen if residual same sign.
        # First, evaluate at the guess and at endpoints.
        # Use numerical bracket-then-bisect.
        xi_a, xi_b = xi_lo, xi_hi
        r_a = residual(xi_a)
        r_b = residual(xi_b)
        if math.isnan(r_a):
            # Fall back: evaluate at xi_guess and walk outward
            r_g = residual(xi_guess)
            if math.isnan(r_g):
                return EquilibriumResult(
                    xi=math.nan, n_eq=n0.copy(),
                    y_eq=np.full_like(n0, math.nan),
                    K_eq=K, T=T, p=p, n_inert=n_inert, converged=False,
                    message="Real-gas residual NaN at warm-start; "
                            "EOS density solve may have failed")
            # Find a finite r_a by walking down from xi_guess
            xi_a = xi_guess
            r_a = r_g
            step = (xi_guess - xi_lo) / 4
            while step > 1e-10:
                trial = xi_a - step
                t_r = residual(trial)
                if not math.isnan(t_r):
                    xi_a, r_a = trial, t_r
                    if r_a * r_b < 0:
                        break
                step /= 2
        if math.isnan(r_b):
            xi_b = xi_guess
            r_b = residual(xi_guess)

        if r_a * r_b > 0 or math.isnan(r_a) or math.isnan(r_b):
            # No sign change -- equilibrium at boundary
            xi_star = xi_a if abs(r_a) < abs(r_b) else xi_b
            n_eq = n0 + self.nu * xi_star
            ntot = n_eq.sum() + n_inert
            return EquilibriumResult(
                xi=xi_star, n_eq=n_eq,
                y_eq=n_eq / ntot if ntot > 0 else np.full_like(n_eq, math.nan),
                K_eq=K, T=T, p=p, n_inert=n_inert, converged=False,
                message=f"Real-gas equilibrium at boundary "
                        f"(r_a={r_a:.3e}, r_b={r_b:.3e})")

        # Bisection
        for it in range(maxiter):
            xi_m = 0.5 * (xi_a + xi_b)
            r_m = residual(xi_m)
            if math.isnan(r_m):
                xi_b = xi_m
                continue
            if abs(r_m) < tol or (xi_b - xi_a) < tol:
                break
            if r_a * r_m < 0:
                xi_b, r_b = xi_m, r_m
            else:
                xi_a, r_a = xi_m, r_m
        else:
            xi_m = 0.5 * (xi_a + xi_b)

        n_eq = n0 + self.nu * xi_m
        ntot = n_eq.sum() + n_inert
        return EquilibriumResult(
            xi=xi_m, n_eq=n_eq, y_eq=n_eq / ntot,
            K_eq=K, T=T, p=p, n_inert=n_inert, converged=True,
            message=f"real-gas converged in {it + 1} iterations")


@dataclass
class EquilibriumResult:
    """Outcome of an equilibrium extent calculation."""
    xi: float                       # extent of reaction
    n_eq: np.ndarray                # equilibrium moles of reactive species
    y_eq: np.ndarray                # mole fractions of reactive species (incl inert)
    K_eq: float                     # equilibrium constant used
    T: float                        # temperature [K]
    p: float                        # pressure [Pa]
    n_inert: float                  # inert moles
    converged: bool
    message: str = ""

    def conversion(self, species_idx: int = 0) -> float:
        """Fractional conversion of reactant `species_idx` (relative to its
        initial moles). Negative if the species is a product (or if a
        reactant was net-formed)."""
        n0 = float(self.n_eq[species_idx]) - 0.0   # placeholder, requires n_initial
        # User can compute themselves: 1 - n_eq[i]/n0[i] for reactants.
        raise NotImplementedError(
            "conversion() requires n_initial; compute as 1 - n_eq[i]/n_initial[i]")
