"""Liquid-phase reaction equilibrium with activity coefficients.

For a liquid-phase reaction
    Sum_i nu_i * A_i = 0
the equilibrium condition is

    K_eq(T) = Prod_i (gamma_i x_i)^nu_i

where gamma_i is the activity coefficient (typically from NRTL,
UNIQUAC, or UNIFAC) and x_i is the liquid mole fraction. The
reference state is implicit in K_eq(T): for the rational convention
gamma_i -> 1 as x_i -> 1 (pure-component liquid reference at the
system temperature and pressure).

This module differs from `equilibrium.py` (gas-phase) in two ways:
  - K_eq(T) must be supplied externally (typically from literature
    tabulation, since liquid-phase formation data is rarely available
    in Shomate form).
  - The fugacity correction phi_i is replaced by the activity
    coefficient gamma_i, computed from the user's activity model.

For systems where liquid-phase K_eq can be derived from gas-phase
formation data plus vapor pressures, use `from_gas_reaction()`.

Examples
--------
Esterification: AcOH + EtOH = EtOAc + H2O at 333 K with UNIFAC

>>> from stateprop.reaction import LiquidPhaseReaction
>>> from stateprop.activity.compounds import make_unifac
>>>
>>> rxn = LiquidPhaseReaction(
...     species_names=['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'],
...     nu=[-1, -1, +1, +1],
...     K_eq_298=4.0,           # well-known mild-positive K
...     dH_rxn=-2.3e3,          # nearly thermoneutral
... )
>>> uf = make_unifac(['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'])
>>> result = rxn.equilibrium_extent(
...     T=333.15, activity_model=uf,
...     n_initial=[1.0, 1.0, 0.0, 0.0])
>>> result.xi      # 0.5-0.7 typically
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Callable, Dict, List
import math
import numpy as np

from .thermo import R_GAS


@dataclass
class LiquidEquilibriumResult:
    """Outcome of a liquid-phase reaction equilibrium calculation."""
    xi: float                           # extent of reaction (single)
    species: List[str]                  # species names in canonical order
    n_eq: np.ndarray                    # equilibrium moles
    x_eq: np.ndarray                    # liquid mole fractions (incl inerts)
    gamma_eq: np.ndarray                # activity coefficients at solution
    K_eq: float                         # K_eq(T) used
    K_a: float                          # K_a from solution = Prod(gamma*x)^nu
    T: float
    n_inert: float
    converged: bool
    iterations: int = 0
    message: str = ""


@dataclass
class LiquidMultiEquilibriumResult:
    """Outcome of a multi-reaction liquid-phase equilibrium calculation."""
    xi: np.ndarray                      # extent vector (length R)
    species: List[str]                  # species names in canonical order
    n_eq: np.ndarray                    # equilibrium moles per species
    x_eq: np.ndarray                    # liquid mole fractions (incl inerts)
    gamma_eq: np.ndarray                # activity coefficients at solution
    K_eq: np.ndarray                    # K_r(T) per reaction
    T: float
    n_inert: float
    converged: bool
    iterations: int
    message: str = ""


# =========================================================================
# Single liquid-phase reaction
# =========================================================================

class LiquidPhaseReaction:
    """Single chemical reaction in the liquid phase.

    Parameters
    ----------
    species_names : sequence of str
        Names of species in canonical order. Used by the activity model
        for coordination -- for UNIFAC/UNIQUAC built from the compound
        database, this order must match the activity_model species
        ordering.
    nu : sequence of float
        Stoichiometric coefficients in the same order. Negative for
        reactants, positive for products.
    K_eq_298 : float, optional
        Equilibrium constant at 298.15 K. Used together with `dH_rxn`
        and the van't Hoff equation to evaluate K_eq(T).
    dH_rxn : float, optional
        Enthalpy of reaction in liquid phase [J/mol]. Defaults to 0
        (T-independent K_eq); pass a non-zero value for van't Hoff
        T-dependence.
    ln_K_eq_T : callable, optional
        Function T -> ln K_eq(T). Overrides `K_eq_298` + `dH_rxn`
        if provided. Use this when you have a published correlation
        like ln K = A + B/T + C ln T.

    Notes
    -----
    The reference state convention for K_eq must match the activity
    model: for gamma_i -> 1 as x_i -> 1 (pure-component liquid
    reference at system T), K_eq comes from pure-liquid reference
    Gibbs energies. The user is responsible for ensuring this
    consistency.
    """

    def __init__(self,
                 species_names: Sequence[str],
                 nu: Sequence[float],
                 K_eq_298: Optional[float] = None,
                 dH_rxn: float = 0.0,
                 ln_K_eq_T: Optional[Callable[[float], float]] = None):
        if len(species_names) != len(nu):
            raise ValueError(f"len(species_names)={len(species_names)} "
                             f"!= len(nu)={len(nu)}")
        self.species_names = tuple(species_names)
        self.nu = np.asarray(nu, dtype=float)
        self.N = len(species_names)
        self._dn_total = float(self.nu.sum())

        if ln_K_eq_T is not None:
            self._ln_K_func = ln_K_eq_T
            # public dH_rxn accessor: only set if van't Hoff form was used
            self.dH_rxn = 0.0
        elif K_eq_298 is not None:
            if K_eq_298 <= 0:
                raise ValueError("K_eq_298 must be positive")
            self._K_298 = float(K_eq_298)
            self._dH = float(dH_rxn)
            self.dH_rxn = float(dH_rxn)
            self._ln_K_func = self._vant_hoff
        else:
            raise ValueError("Provide either ln_K_eq_T or K_eq_298")

    def _vant_hoff(self, T: float) -> float:
        """ln K(T) from van't Hoff equation, integrated form."""
        # ln K(T) = ln K(298) - dH/R * (1/T - 1/298.15)
        return math.log(self._K_298) - self._dH / R_GAS * (1.0/T - 1.0/298.15)

    def K_eq(self, T: float) -> float:
        """Equilibrium constant at temperature T."""
        return math.exp(self._ln_K_func(T))

    def ln_K_eq(self, T: float) -> float:
        """ln K_eq(T)."""
        return self._ln_K_func(T)

    def equilibrium_extent(
        self,
        T: float,
        n_initial: Sequence[float],
        activity_model,
        n_inert: float = 0.0,
        tol: float = 1e-9,
        maxiter: int = 200,
    ) -> LiquidEquilibriumResult:
        """Solve for the reaction extent xi at given T.

        Equilibrium condition:
            ln K_eq(T) = Sum_i nu_i [ln gamma_i(T, x) + ln x_i]

        Solved by bisection on xi over the feasible range, with
        gamma_i evaluated at each candidate x.

        Parameters
        ----------
        T : float
            Temperature [K].
        n_initial : sequence of float
            Initial moles of each reaction species, in canonical order.
        activity_model : object with .gammas(T, x) -> ndarray of length N
            NRTL, UNIQUAC, UNIFAC instance with the same species order.
        n_inert : float
            Total moles of inert species (does not react but counts
            toward N_tot for x_i denominator).
        tol : float
            Convergence tolerance on the residual.
        maxiter : int
            Bisection iteration cap.

        Returns
        -------
        LiquidEquilibriumResult
        """
        n0 = np.asarray(n_initial, dtype=float)
        if n0.size != self.N:
            raise ValueError(f"n_initial length {n0.size} != "
                             f"species count {self.N}")
        if (n0 < 0).any():
            raise ValueError("n_initial must be nonnegative")

        # Feasible xi range
        xi_lo, xi_hi = -np.inf, np.inf
        for nu_i, ni0 in zip(self.nu, n0):
            if nu_i > 0:
                xi_lo = max(xi_lo, -ni0 / nu_i)
            elif nu_i < 0:
                xi_hi = min(xi_hi, -ni0 / nu_i)
        eps = 1e-12 * max(1.0, n0.sum() + n_inert)
        xi_lo = xi_lo + eps if xi_lo > -np.inf else -1e6
        xi_hi = xi_hi - eps if xi_hi <  np.inf else  1e6

        ln_K = self.ln_K_eq(T)

        def residual(xi: float) -> float:
            ni = n0 + self.nu * xi
            ntot = ni.sum() + n_inert
            if (ni <= 0).any() or ntot <= 0:
                return math.nan
            xi_arr = ni / ntot
            # Note: activity model gets the full liquid composition (excluding inerts)
            # but here we assume inerts are zero or the activity model handles them.
            # For this implementation, we pass only the reactive composition.
            # If inerts are present, the user should include them as a "species"
            # with nu_i = 0. (We'll relax this later if needed.)
            gammas = np.asarray(activity_model.gammas(T, xi_arr))
            # ln K_a = Sum_i nu_i [ln gamma_i + ln x_i]
            return float((self.nu * (np.log(gammas) + np.log(xi_arr))).sum()
                          - ln_K)

        # Bisection
        r_lo = residual(xi_lo)
        r_hi = residual(xi_hi)
        if math.isnan(r_lo) or math.isnan(r_hi):
            return LiquidEquilibriumResult(
                xi=math.nan, species=list(self.species_names),
                n_eq=n0.copy(),
                x_eq=np.full_like(n0, math.nan),
                gamma_eq=np.full_like(n0, math.nan),
                K_eq=math.exp(ln_K), K_a=math.nan, T=T, n_inert=n_inert,
                converged=False, iterations=0,
                message="Initial bracket evaluation failed")
        if r_lo * r_hi > 0:
            xi_star = xi_lo if abs(r_lo) < abs(r_hi) else xi_hi
            n_eq = n0 + self.nu * xi_star
            x_eq = n_eq / (n_eq.sum() + n_inert)
            gammas = np.asarray(activity_model.gammas(T, x_eq))
            K_a = float(np.prod((gammas * x_eq) ** self.nu))
            return LiquidEquilibriumResult(
                xi=xi_star, species=list(self.species_names),
                n_eq=n_eq, x_eq=x_eq, gamma_eq=gammas,
                K_eq=math.exp(ln_K), K_a=K_a, T=T, n_inert=n_inert,
                converged=False, iterations=0,
                message=f"Equilibrium at boundary "
                        f"(r_lo={r_lo:.3e}, r_hi={r_hi:.3e})")

        xi_a, xi_b = xi_lo, xi_hi
        r_a, r_b = r_lo, r_hi
        for it in range(maxiter):
            xi_m = 0.5 * (xi_a + xi_b)
            r_m = residual(xi_m)
            if math.isnan(r_m):
                xi_b = xi_m
                continue
            if abs(r_m) < tol or (xi_b - xi_a) < tol * max(1.0, abs(xi_b)):
                break
            if r_a * r_m < 0:
                xi_b, r_b = xi_m, r_m
            else:
                xi_a, r_a = xi_m, r_m
        else:
            xi_m = 0.5 * (xi_a + xi_b)

        n_eq = n0 + self.nu * xi_m
        ntot = n_eq.sum() + n_inert
        x_eq = n_eq / ntot
        gammas = np.asarray(activity_model.gammas(T, x_eq))
        K_a = float(np.prod((gammas * x_eq) ** self.nu))
        return LiquidEquilibriumResult(
            xi=xi_m, species=list(self.species_names),
            n_eq=n_eq, x_eq=x_eq, gamma_eq=gammas,
            K_eq=math.exp(ln_K), K_a=K_a, T=T, n_inert=n_inert,
            converged=True, iterations=it + 1,
            message=f"converged in {it + 1} iterations")


# =========================================================================
# Multi-reaction liquid-phase
# =========================================================================

class MultiLiquidPhaseReaction:
    """Coupled multi-reaction equilibrium in the liquid phase.

    Parameters
    ----------
    reactions : sequence of LiquidPhaseReaction
        Must be linearly independent. Stoichiometry matrix rank check
        is performed at construction.

    Examples
    --------
    Coupled esterification + transesterification (illustrative, not
    physically rigorous):

    >>> r1 = LiquidPhaseReaction(['AcOH', 'EtOH', 'EtOAc', 'H2O'],
    ...                            [-1,-1,+1,+1], K_eq_298=4.0)
    >>> r2 = LiquidPhaseReaction(['AcOH', 'MeOH', 'MeOAc', 'H2O'],
    ...                            [-1,-1,+1,+1], K_eq_298=5.0)
    >>> system = MultiLiquidPhaseReaction([r1, r2])
    """

    def __init__(self, reactions: Sequence[LiquidPhaseReaction]):
        if len(reactions) == 0:
            raise ValueError("MultiLiquidPhaseReaction requires "
                             ">= 1 reaction")
        self.reactions = tuple(reactions)
        self.R = len(reactions)

        # Build canonical species order (union, in order of first appearance)
        species_list = []
        species_idx = {}
        for rxn in reactions:
            for nm in rxn.species_names:
                if nm not in species_idx:
                    species_idx[nm] = len(species_list)
                    species_list.append(nm)
        self.species_names = tuple(species_list)
        self.species_idx = species_idx
        self.N = len(species_list)

        # Stoichiometry matrix R x N
        nu = np.zeros((self.R, self.N))
        for r, rxn in enumerate(reactions):
            for sp_name, nu_local in zip(rxn.species_names, rxn.nu):
                nu[r, species_idx[sp_name]] = nu_local
        self.nu = nu

        # Rank check
        rk = np.linalg.matrix_rank(nu)
        if rk < self.R:
            raise ValueError(
                f"Stoichiometry matrix rank {rk} < {self.R} reactions. "
                "Reactions must be linearly independent.")

    def K_eq(self, T: float) -> np.ndarray:
        return np.array([rxn.K_eq(T) for rxn in self.reactions])

    def equilibrium(
        self,
        T: float,
        n_initial: Dict[str, float],
        activity_model,
        n_inert: float = 0.0,
        xi_init: Optional[Sequence[float]] = None,
        tol: float = 1e-8,
        maxiter: int = 200,
        damping: float = 0.7,
        min_n: float = 1e-30,
    ) -> LiquidMultiEquilibriumResult:
        """Solve for simultaneous equilibrium of all reactions in liquid.

        Newton's method on the residual system:
            f_r = Sum_i nu[r,i] (ln gamma_i + ln x_i) - ln K_r(T)

        Uses the ideal-mixture Jacobian (analogous to the gas-phase
        case): J[r,s] = Sum_i nu[r,i] nu[s,i]/n_i - dn_r dn_s/N_tot.
        The neglected term ∂ln(gamma_i)/∂x_j is small for moderately
        non-ideal mixtures and damping=0.7 keeps iteration stable.

        Parameters
        ----------
        T, n_initial, n_inert, xi_init, tol, maxiter, damping, min_n :
            See `MultiReaction.equilibrium_ideal_gas` (gas-phase analog).
        activity_model : object with `gammas(T, x)` returning length-N array.

        Returns
        -------
        LiquidMultiEquilibriumResult
        """
        # Build n0 in canonical order
        n0 = np.zeros(self.N)
        for name, n in n_initial.items():
            if name not in self.species_idx:
                raise KeyError(f"Species '{name}' not in this MultiLiquid"
                               f"PhaseReaction. Available: {self.species_names}")
            if n < 0:
                raise ValueError(f"Initial moles for {name} must be >= 0")
            n0[self.species_idx[name]] = float(n)

        K = self.K_eq(T)
        ln_K = np.log(K)
        dn = self.nu.sum(axis=1)   # net mole change per reaction

        if xi_init is None:
            xi = np.zeros(self.R)
        else:
            xi = np.asarray(xi_init, dtype=float)
            if xi.size != self.R:
                raise ValueError(f"xi_init length {xi.size} != R={self.R}")

        for it in range(maxiter):
            n = n0 + self.nu.T @ xi
            n_safe = np.maximum(n, min_n)
            N_tot = n_safe.sum() + n_inert
            x = n_safe / N_tot

            gammas = np.asarray(activity_model.gammas(T, x))
            ln_gx = np.log(gammas) + np.log(x)
            f = self.nu @ ln_gx - ln_K

            err = float(np.max(np.abs(f)))
            if err < tol:
                K_a = np.array([np.prod((gammas * x) ** self.nu[r])
                                  for r in range(self.R)])
                return LiquidMultiEquilibriumResult(
                    xi=xi.copy(), species=list(self.species_names),
                    n_eq=n.copy(), x_eq=x, gamma_eq=gammas,
                    K_eq=K, T=T, n_inert=n_inert,
                    converged=True, iterations=it,
                    message=f"converged in {it} iterations (||f||={err:.2e})")

            # Ideal-mixture Jacobian
            inv_n = 1.0 / n_safe
            J = np.einsum('ri,si,i->rs', self.nu, self.nu, inv_n)
            J -= np.outer(dn, dn) / N_tot

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
        n_safe = np.maximum(n, min_n)
        N_tot = n_safe.sum() + n_inert
        x = n_safe / N_tot
        gammas = np.asarray(activity_model.gammas(T, x))
        return LiquidMultiEquilibriumResult(
            xi=xi.copy(), species=list(self.species_names),
            n_eq=n.copy(), x_eq=x, gamma_eq=gammas,
            K_eq=K, T=T, n_inert=n_inert,
            converged=False, iterations=maxiter,
            message=f"did not converge in {maxiter} iterations "
                    f"(||f||={err:.2e})")
