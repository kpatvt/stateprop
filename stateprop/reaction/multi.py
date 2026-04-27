"""Multi-reaction equilibrium for ideal-gas systems.

For R simultaneous reactions and N species, the equilibrium-condition
residuals are

    f_r(xi) = Sum_i nu[i,r] * ln(y_i) + dn_r * ln(p/p_ref) - ln(K_r(T))

where xi = (xi_1, ..., xi_R) are reaction extents,
      n_i = n_i_0 + Sum_s nu[i,s] * xi_s
      y_i = n_i / N_tot
      dn_r = Sum_i nu[i,r]
      K_r(T) = exp(-dG_rxn,r(T) / RT)

At equilibrium f_r = 0 for all r. Newton's method on this residual
system converges quadratically with the analytic Jacobian

    J[r,s] = Sum_i nu[i,r] nu[i,s] / n_i - dn_r dn_s / N_tot.

The classical example is methane steam reforming, where the two
linearly-independent reactions CH4 + H2O = CO + 3 H2 and
CO + H2O = CO2 + H2 fully span the C-H-O reaction space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, List
import math
import numpy as np

from .thermo import SpeciesThermo, get_species, R_GAS
from .equilibrium import Reaction


@dataclass
class MultiEquilibriumResult:
    """Outcome of a multi-reaction equilibrium calculation."""
    xi: np.ndarray              # extent vector (length R)
    species: List[str]          # species names in order
    n_eq: np.ndarray            # equilibrium moles per species (length N)
    y_eq: np.ndarray            # mole fractions (length N), incl inerts
    K_eq: np.ndarray            # K_r(T) per reaction (length R)
    T: float
    p: float
    n_inert: float
    converged: bool
    iterations: int
    message: str = ""


class MultiReaction:
    """A coupled system of reactions, sharing a unified species list.

    The first time you build a MultiReaction, the union of all
    species across the supplied reactions becomes the canonical order;
    each reaction's stoichiometry is then expressed in that order.

    Parameters
    ----------
    reactions : sequence of `Reaction`
        Should be linearly independent. Linear dependence makes the
        Jacobian singular and the equilibrium solver will fail.

    Examples
    --------
    Methane steam reforming with water-gas shift:

    >>> from stateprop.reaction import Reaction, MultiReaction
    >>> r1 = Reaction.from_names(reactants={'CH4': 1, 'H2O': 1},
    ...                           products={'CO': 1, 'H2': 3})
    >>> r2 = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
    ...                           products={'CO2': 1, 'H2': 1})
    >>> system = MultiReaction([r1, r2])
    >>> result = system.equilibrium_ideal_gas(
    ...     T=1100.0, p=1e5,
    ...     n_initial={'CH4': 1.0, 'H2O': 3.0})
    """

    def __init__(self, reactions: Sequence[Reaction]):
        if len(reactions) == 0:
            raise ValueError("MultiReaction requires at least one reaction")
        self.reactions = tuple(reactions)
        self.R = len(reactions)

        # Build canonical species ordering: union across all reactions
        # in order of first appearance.
        species_list = []
        species_idx = {}
        for rxn in reactions:
            for sp in rxn.species:
                if sp.name not in species_idx:
                    species_idx[sp.name] = len(species_list)
                    species_list.append(sp)
        self.species = tuple(species_list)
        self.species_idx = species_idx
        self.species_names = tuple(sp.name for sp in species_list)
        self.N = len(species_list)

        # Build R x N stoichiometry matrix
        nu = np.zeros((self.R, self.N))
        for r, rxn in enumerate(reactions):
            for sp_local, nu_local in zip(rxn.species, rxn.nu):
                idx = species_idx[sp_local.name]
                nu[r, idx] = nu_local
        self.nu = nu             # shape (R, N)
        self.dn = nu.sum(axis=1) # net mole change per reaction, shape (R,)

        # Check linear independence: rank of nu should equal R
        rk = np.linalg.matrix_rank(nu)
        if rk < self.R:
            raise ValueError(
                f"MultiReaction stoichiometry matrix has rank {rk} "
                f"< {self.R} reactions. Reactions must be linearly "
                "independent (otherwise the Jacobian is singular).")

    @classmethod
    def from_specs(cls, specs: Sequence[dict]) -> "MultiReaction":
        """Build from a list of {'reactants': {...}, 'products': {...}} specs.

        Examples
        --------
        >>> system = MultiReaction.from_specs([
        ...     {'reactants': {'CH4': 1, 'H2O': 1},
        ...      'products':  {'CO': 1, 'H2': 3}},
        ...     {'reactants': {'CO': 1, 'H2O': 1},
        ...      'products':  {'CO2': 1, 'H2': 1}},
        ... ])
        """
        reactions = [Reaction.from_names(reactants=s['reactants'],
                                            products=s['products'])
                     for s in specs]
        return cls(reactions)

    # ----------------------------------------------------------------
    # Thermodynamic functions
    # ----------------------------------------------------------------

    def K_eq(self, T: float) -> np.ndarray:
        """K_eq for each reaction at T [K]; returns ndarray of length R."""
        return np.array([rxn.K_eq(T) for rxn in self.reactions])

    def dG_rxn(self, T: float) -> np.ndarray:
        """Gibbs of reaction for each reaction at T."""
        return np.array([rxn.dG_rxn(T) for rxn in self.reactions])

    def dH_rxn(self, T: float) -> np.ndarray:
        """Enthalpy of reaction for each reaction at T."""
        return np.array([rxn.dH_rxn(T) for rxn in self.reactions])

    # ----------------------------------------------------------------
    # Equilibrium solver
    # ----------------------------------------------------------------

    def equilibrium_ideal_gas(
        self,
        T: float,
        p: float,
        n_initial: Dict[str, float],
        n_inert: float = 0.0,
        p_ref: float = 1e5,
        xi_init: Optional[Sequence[float]] = None,
        tol: float = 1e-9,
        maxiter: int = 100,
        damping: float = 1.0,
        min_n: float = 1e-30,
    ) -> MultiEquilibriumResult:
        """Solve simultaneous equilibrium for all reactions, ideal gas."""
        return self._solve(T, p, n_initial, n_inert, p_ref, xi_init, tol,
                            maxiter, damping, min_n, eos=None)

    def equilibrium_real_gas(
        self,
        T: float,
        p: float,
        n_initial: Dict[str, float],
        eos,
        n_inert: float = 0.0,
        p_ref: float = 1e5,
        xi_init: Optional[Sequence[float]] = None,
        tol: float = 1e-8,
        maxiter: int = 200,
        damping: float = 0.7,
        min_n: float = 1e-30,
    ) -> MultiEquilibriumResult:
        """Solve simultaneous equilibrium for all reactions with EOS-based
        fugacity coefficients.

        Replaces the ideal-gas K_y * (p/p_ref)^Δν condition with the
        fugacity-corrected form:

            K_eq,r(T) = Prod_i (y_i * phi_i * p / p_ref)^nu[r,i]

        Equivalently, residual r is:

            f_r = Σ nu[r,i] (ln y_i + ln φ_i) + Δν_r ln(p/p_ref) − ln K_r

        The Newton Jacobian is computed using the ideal-gas formula
        (∂φ_i/∂y_j contributions are neglected). This makes Newton steps
        slightly less aggressive but the iteration still converges
        because φ_i depends only weakly on y_j compared to the ln(y_i)
        term. Damping is applied to keep iteration stable.

        Parameters
        ----------
        T, p, n_initial, n_inert, p_ref, xi_init, tol, maxiter, damping,
        min_n: same as `equilibrium_ideal_gas`.
        eos : EOS mixture object
            Must implement
              - density_from_pressure(p, T, x, phase_hint='vapor') -> rho
              - ln_phi(rho, T, x) -> ndarray of ln(phi_i)
            CubicMixture, SAFTMixture, GERG mixtures all qualify.
            **The EOS species ordering MUST match this MultiReaction's
            canonical species ordering** (`self.species_names`).

        Returns
        -------
        MultiEquilibriumResult
        """
        return self._solve(T, p, n_initial, n_inert, p_ref, xi_init, tol,
                            maxiter, damping, min_n, eos=eos)

    def _solve(
        self, T: float, p: float, n_initial: Dict[str, float],
        n_inert: float, p_ref: float, xi_init, tol: float, maxiter: int,
        damping: float, min_n: float, eos,
    ) -> MultiEquilibriumResult:
        """Unified Newton solver: ideal-gas if eos is None, else real-gas."""
        # Initial mole numbers on canonical species ordering
        n0 = np.zeros(self.N)
        for name, n in n_initial.items():
            if name not in self.species_idx:
                raise KeyError(f"Species '{name}' not in this MultiReaction. "
                               f"Available: {self.species_names}")
            if n < 0:
                raise ValueError(f"Initial moles for {name} must be >= 0")
            n0[self.species_idx[name]] = float(n)

        K = self.K_eq(T)
        ln_K = np.log(K)

        if xi_init is None:
            xi = np.zeros(self.R)
        else:
            xi = np.asarray(xi_init, dtype=float)
            if xi.size != self.R:
                raise ValueError(f"xi_init length {xi.size} != R={self.R}")

        delta_p = math.log(p / p_ref)

        for it in range(maxiter):
            n = n0 + self.nu.T @ xi
            n_safe = np.maximum(n, min_n)
            N_tot = n_safe.sum() + n_inert
            y = n_safe / N_tot

            # Residual
            ln_y = np.log(y)
            f = self.nu @ ln_y + self.dn * delta_p - ln_K
            if eos is not None:
                rho = eos.density_from_pressure(p, T, y, phase_hint='vapor')
                ln_phi = np.asarray(eos.ln_phi(rho, T, y))
                f = f + self.nu @ ln_phi

            err = float(np.max(np.abs(f)))
            if err < tol:
                return MultiEquilibriumResult(
                    xi=xi.copy(), species=list(self.species_names),
                    n_eq=n.copy(), y_eq=(n / (n.sum() + n_inert)),
                    K_eq=K, T=T, p=p, n_inert=n_inert,
                    converged=True, iterations=it,
                    message=("real-gas " if eos is not None else "")
                            + f"converged in {it} iterations (||f||={err:.2e})")

            # Ideal-gas Jacobian (used for both ideal and real cases;
            # for real gas this neglects ∂phi/∂y but Newton step still
            # converges with sufficient damping).
            inv_n = 1.0 / n_safe
            J = np.einsum('ri,si,i->rs', self.nu, self.nu, inv_n)
            J -= np.outer(self.dn, self.dn) / N_tot

            try:
                dxi = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                dxi, *_ = np.linalg.lstsq(J, -f, rcond=None)

            dxi = damping * dxi

            # Step-size limiter to keep n_i >= min_n
            d_n = self.nu.T @ dxi
            alpha = 1.0
            for i in range(self.N):
                if d_n[i] < 0 and n[i] > min_n:
                    a_lim = (min_n - n[i]) / d_n[i]
                    if a_lim < alpha:
                        alpha = max(0.5 * a_lim, 1e-6)
            xi = xi + alpha * dxi

        # No convergence
        n = n0 + self.nu.T @ xi
        return MultiEquilibriumResult(
            xi=xi.copy(), species=list(self.species_names),
            n_eq=n.copy(),
            y_eq=(np.maximum(n, 0) / (np.maximum(n, 0).sum() + n_inert)),
            K_eq=K, T=T, p=p, n_inert=n_inert,
            converged=False, iterations=maxiter,
            message=("real-gas " if eos is not None else "")
                    + f"did not converge in {maxiter} iterations "
                      f"(||f||_last = {err:.2e})")
