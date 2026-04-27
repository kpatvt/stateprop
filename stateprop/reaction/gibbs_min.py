"""Direct Gibbs minimization with element constraints (RAND / White-
Johnson-Dantzig algorithm).

This is an alternative to the extent-of-reaction formulation used in
``equilibrium.py`` / ``multi.py`` for chemical equilibrium.  Instead
of specifying a list of independent reactions and computing extents,
the user supplies:

- a list of species (with their thermochemistry)
- a list of formulas (atomic composition) per species
- an initial composition (any positive composition that satisfies the
  element balance)
- temperature, pressure

and the algorithm finds the equilibrium composition by directly
minimizing the total Gibbs energy

    G(n) = Σ_i n_i [μ_i°(T) + RT ln(a_i)]

subject to atom-balance constraints

    Σ_i a_ki n_i = b_k    for each element k

via Newton iterations on the Lagrangian.  The advantage over the
extent-of-reaction formulation:

- No need to specify reactions; only formulas matter.  For systems
  with many parallel reactions (combustion, gasification, hydrocarbon
  reforming) this is far less error-prone — you can't accidentally
  forget a side reaction.
- Linearly dependent reactions are not a concern: the algorithm
  iterates in a space whose dimension is N - E (number of species
  minus number of elements), which is the number of *independent*
  reactions automatically.
- The iteration history of G is monotonically decreasing (modulo
  step-size effects), so convergence is auditable.

References
----------
- White, Johnson, Dantzig (1958), "Chemical equilibrium in complex
  mixtures", J. Chem. Phys. 28, 751.
- Smith & Missen (1991), "Chemical Reaction Equilibrium Analysis",
  Wiley, Ch. 5 (RAND algorithm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Mapping
import numpy as np

_R_GAS = 8.31446261815324    # J/(mol K)
_P_REF = 1e5                  # Pa, IUPAC standard state (1 bar)


@dataclass
class GibbsMinResult:
    """Result of a direct Gibbs minimization."""
    converged: bool
    T: float
    p: float
    n: np.ndarray                # (N,) equilibrium mole numbers
    n_init: np.ndarray
    species_names: tuple
    elements: tuple              # element symbols
    G_total: float               # G(n) at convergence [J]
    iterations: int
    pi: np.ndarray               # (E,) Lagrange multipliers (λ_k / RT)
    G_history: tuple             # (iters,) Gibbs energy by iteration
    atom_balance_residual: float
    message: str

    @property
    def y(self) -> np.ndarray:
        """Equilibrium mole fractions y_i = n_i / Σn."""
        N = float(self.n.sum())
        return self.n / N if N > 0 else np.zeros_like(self.n)


def _build_atomic_matrix(
    formulas: Sequence[Mapping[str, float]],
) -> tuple:
    """Build the (E, N) atomic matrix A from per-species formula dicts.

    Returns (A, elements_sorted) where A[k, i] is the number of atoms
    of element k in species i, and elements_sorted is a tuple of
    element symbols ordered alphabetically for reproducibility.
    """
    elements = sorted({el for f in formulas for el in f})
    E = len(elements)
    N = len(formulas)
    A = np.zeros((E, N), dtype=float)
    for i, f in enumerate(formulas):
        for el, count in f.items():
            k = elements.index(el)
            A[k, i] = float(count)
    return A, tuple(elements)


def gibbs_minimize_TP(
    T: float,
    p: float,
    species_names: Sequence[str],
    formulas: Sequence[Mapping[str, float]],
    mu_standard_funcs: Sequence[Callable[[float], float]],
    n_init: Sequence[float],
    *,
    phase: str = "gas",
    phase_per_species: Optional[Sequence[str]] = None,
    activity_model = None,
    p_ref: float = _P_REF,
    tol: float = 1e-9,
    maxiter: int = 100,
    damping_init: float = 1.0,
    n_floor: float = 1e-25,
    verbose: bool = False,
) -> GibbsMinResult:
    """Minimize G(n) at fixed (T, p) subject to atom balance.

    Parameters
    ----------
    T, p : float
        Temperature [K] and pressure [Pa].
    species_names : sequence of str, length N
        Species labels for the result.
    formulas : sequence of dict, length N
        Atomic composition per species, e.g. ``{'C': 1, 'O': 2}``
        for CO2.  Element symbols are arbitrary strings; consistency
        across species is what matters.
    mu_standard_funcs : sequence of callables, length N
        Each ``f_i(T) -> J/mol`` returns the species' standard-state
        chemical potential at T.  For ideal gas this is the Gibbs
        energy at 1 bar reference pressure (e.g. NIST ``Gf(T)`` from
        ``SpeciesThermo.Gf``).  For pure solid species, this is the
        Gibbs energy of formation of the solid; activity is 1.
    n_init : sequence of float, length N
        Initial moles per species.  Must be strictly positive (any
        species with ``n_i = 0`` is forbidden by the log) and must
        satisfy ``A n_init = b`` (i.e., it has the right total moles
        of every element).  An ``n_floor`` is applied internally to
        keep iterations away from zero.
    phase : {"gas", "liquid"}
        Default reference state for all species (used when
        ``phase_per_species`` is None).  ``"gas"``: ``μ_i = μ_i° +
        RT ln(y_i p/p_ref)``.  ``"liquid"``: ``μ_i = μ_i° + RT ln(x_i)``
        (no pressure correction; use ``activity_model`` for non-ideality).
    phase_per_species : sequence of str, length N, optional
        Per-species phase ('gas', 'liquid', or 'solid').  Solids are
        treated as pure phases with activity 1 (so ``μ_i = μ_i°(T)``).
        If given, overrides ``phase``.  All non-solid species must
        share the same reference state ('gas' or 'liquid').
    activity_model : object with ``.gammas(T, x) -> array``, optional
        Liquid-phase activity coefficient model.  Only used when
        the non-solid phase is 'liquid'.  Held constant during the
        linearized step (Picard inside Newton).
    p_ref : float
        Reference pressure for ideal-gas standard state [Pa]. Default
        1 bar = 1e5 Pa.
    tol : float
        Convergence on max ``|μ_i / RT - Σ_k π_k a_ki|`` (= 0 at the
        true minimum).
    maxiter : int
        Maximum Newton iterations.
    damping_init : float
        Initial Newton step size.  Backtracking shrinks it as needed.
    n_floor : float
        Minimum mole number kept during iteration to avoid log(0).
    verbose : bool
        Print iteration progress.

    Returns
    -------
    GibbsMinResult

    Notes
    -----
    Solid-phase handling.  For each solid species, the activity is 1
    (pure phase), so ``μ_i = μ_i°(T)`` and the Hessian is zero in
    that direction.  The RAND linear system is augmented from
    ``(E+1)×(E+1)`` to ``(E+1+S)×(E+1+S)`` where ``S`` is the number
    of solid species, with the extra rows enforcing the
    solid-stationarity constraint ``Σ_k π_k a_ki = μ_i°/RT`` and the
    extra columns being the solid mole-number changes ``Δn_i^solid``.
    The system is symmetric and is solved via ``np.linalg.solve`` with
    a ``lstsq`` fallback in case of numerical singularity (which
    typically signals that more than ``E`` independent solid phases
    are nominated to coexist — a Gibbs-phase-rule violation).
    """
    N = len(species_names)
    if len(formulas) != N:
        raise ValueError(f"formulas length {len(formulas)} != N={N}")
    if len(mu_standard_funcs) != N:
        raise ValueError(
            f"mu_standard_funcs length {len(mu_standard_funcs)} != N={N}")
    if len(n_init) != N:
        raise ValueError(f"n_init length {len(n_init)} != N={N}")

    # Resolve phase_per_species
    if phase_per_species is None:
        phase_per_species_list = [phase] * N
    else:
        if len(phase_per_species) != N:
            raise ValueError(
                f"phase_per_species length {len(phase_per_species)} != N={N}")
        phase_per_species_list = list(phase_per_species)
    for ph in phase_per_species_list:
        if ph not in ("gas", "liquid", "solid"):
            raise ValueError(
                f"phase_per_species entries must be 'gas', 'liquid', or "
                f"'solid'; got {ph!r}")
    # Identify solid vs fluid (gas/liquid) species
    is_solid = np.array([p == "solid" for p in phase_per_species_list],
                          dtype=bool)
    is_fluid = ~is_solid
    fluid_phases = {phase_per_species_list[i] for i in range(N)
                     if not is_solid[i]}
    if len(fluid_phases) > 1:
        raise ValueError(
            "All non-solid species must share the same fluid reference "
            f"state ('gas' or 'liquid'); got {fluid_phases}")
    fluid_phase = fluid_phases.pop() if fluid_phases else "gas"
    if activity_model is not None and fluid_phase != "liquid":
        raise ValueError("activity_model only valid for fluid_phase='liquid'")

    n_init_arr = np.asarray(n_init, dtype=float)
    if (n_init_arr <= 0).any():
        raise ValueError("all n_init values must be > 0")

    A_mat, elements = _build_atomic_matrix(formulas)
    E = len(elements)

    # Element balance: b_k = Σ_i a_ki n_init_i (over ALL species,
    # solid included)
    b_vec = A_mat @ n_init_arr

    # Standard-state chemical potentials at T (constant through iter)
    mu0 = np.array([f(T) for f in mu_standard_funcs])

    # Initialize iterate
    n = np.maximum(n_init_arr.copy(), n_floor)
    G_history = []

    fluid_idx = np.where(is_fluid)[0]
    solid_idx = np.where(is_solid)[0]
    S = int(is_solid.sum())
    A_solid = A_mat[:, solid_idx]  # (E, S)
    A_fluid = A_mat[:, fluid_idx]  # (E, N - S)

    def _compute_mu(n_arr, gammas=None):
        """Chemical potentials μ_i at composition n_arr.

        Solid species: μ_i = μ_i°(T) (activity=1).
        Fluid species: standard ideal-gas or activity-coefficient form,
        but only the FLUID-PHASE mole fractions enter the log.
        """
        mu_out = mu0.copy()
        n_fluid = n_arr[fluid_idx]
        N_fluid = float(n_fluid.sum())
        if N_fluid <= 0:
            # No fluid phase; all moles in solids; chemical potential
            # of fluid is undefined.  Return mu0 for solids only.
            return mu_out
        x_fluid = n_fluid / N_fluid
        x_safe = np.maximum(x_fluid, 1e-300)
        if fluid_phase == "gas":
            mu_fluid = (mu0[fluid_idx]
                         + _R_GAS * T * np.log(x_safe)
                         + _R_GAS * T * np.log(p / p_ref))
        else:   # liquid
            if gammas is None:
                mu_fluid = mu0[fluid_idx] + _R_GAS * T * np.log(x_safe)
            else:
                mu_fluid = (mu0[fluid_idx]
                             + _R_GAS * T * np.log(gammas)
                             + _R_GAS * T * np.log(x_safe))
        mu_out[fluid_idx] = mu_fluid
        return mu_out

    def _compute_G(n_arr, gammas=None):
        mu = _compute_mu(n_arr, gammas)
        return float(np.sum(n_arr * mu))

    converged = False
    iters_done = 0
    pi_last = np.zeros(E)
    msg = "did not start"
    theta_max = float("inf")

    for it in range(maxiter):
        if activity_model is not None and fluid_phase == "liquid":
            n_fluid = n[fluid_idx]
            x_fluid = n_fluid / n_fluid.sum()
            gammas_fluid = np.asarray(activity_model.gammas(T, x_fluid))
            # Pad to full N (gammas[i] = 1 for solids — not used)
            gammas_full = np.ones(N)
            gammas_full[fluid_idx] = gammas_fluid
        else:
            gammas_fluid = None
            gammas_full = None

        mu = _compute_mu(n, gammas_full)
        G_curr = float(np.sum(n * mu))
        G_history.append(G_curr)

        muRT = mu / (_R_GAS * T)

        # Build the augmented RAND linear system (gas + solid).
        # Restrict B and c to fluid (non-solid) species; solids enter
        # as separate variables in the linear system.
        n_safe = np.maximum(n, n_floor)
        n_fluid_safe = n_safe[fluid_idx]

        B = (A_fluid * n_fluid_safe[np.newaxis, :]) @ A_fluid.T   # (E, E)
        b_fluid = A_fluid @ n_fluid_safe                          # (E,)
        c_vec = A_fluid @ (n_fluid_safe * muRT[fluid_idx])        # (E,)
        c_tot = float(np.sum(n_fluid_safe * muRT[fluid_idx]))     # scalar

        if S == 0:
            # Original (E+1) x (E+1) system
            M = np.zeros((E + 1, E + 1))
            M[:E, :E] = B
            M[:E, E] = b_fluid
            M[E, :E] = b_fluid
            rhs = np.concatenate([c_vec, [c_tot]])
        else:
            # Augmented (E + 1 + S) x (E + 1 + S) symmetric system
            sz = E + 1 + S
            M = np.zeros((sz, sz))
            M[:E, :E] = B
            M[:E, E] = b_fluid
            M[E, :E] = b_fluid
            M[:E, E + 1:] = A_solid
            M[E + 1:, :E] = A_solid.T
            mu_solid_RT = muRT[solid_idx]
            rhs = np.concatenate([c_vec, [c_tot], mu_solid_RT])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        pi = sol[:E]
        u_var = sol[E]
        delta_n_solid = sol[E + 1:] if S > 0 else np.array([])
        pi_last = pi

        # Δn_fluid = n (Σ_k π_k a_ki - μ/RT + u)
        theta_full = muRT - A_mat.T @ pi   # residual per species
        delta_n = np.zeros(N)
        delta_n[fluid_idx] = n_fluid_safe * (-theta_full[fluid_idx] + u_var)
        delta_n[solid_idx] = delta_n_solid

        theta_max = float(np.max(np.abs(theta_full)))
        if verbose:
            print(f"  iter {it:3d}: G = {G_curr:.4e}, "
                  f"max|θ| = {theta_max:.2e}, "
                  f"||A n - b|| = {np.linalg.norm(A_mat @ n - b_vec):.2e}, "
                  f"S = {S}")
        if theta_max < tol:
            converged = True
            iters_done = it + 1
            msg = f"converged in {iters_done} iters, max|θ|={theta_max:.2e}"
            break

        # Backtracking line search
        alpha = damping_init
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(delta_n < 0, -n / delta_n, np.inf)
            alpha_max = float(np.min(ratios))
        alpha = min(alpha, 0.95 * alpha_max if alpha_max > 0 else 0.5)
        if alpha <= 0:
            alpha = 1e-6

        n_trial = n + alpha * delta_n
        n_trial = np.maximum(n_trial, n_floor)
        if activity_model is not None and fluid_phase == "liquid":
            x_trial = n_trial[fluid_idx] / n_trial[fluid_idx].sum()
            gammas_trial = np.asarray(activity_model.gammas(T, x_trial))
            gammas_trial_full = np.ones(N)
            gammas_trial_full[fluid_idx] = gammas_trial
        else:
            gammas_trial_full = None
        G_trial = _compute_G(n_trial, gammas_trial_full)
        bt_count = 0
        while G_trial > G_curr + 1e-9 * abs(G_curr) and bt_count < 30:
            alpha *= 0.5
            n_trial = n + alpha * delta_n
            n_trial = np.maximum(n_trial, n_floor)
            if activity_model is not None and fluid_phase == "liquid":
                x_trial = n_trial[fluid_idx] / n_trial[fluid_idx].sum()
                gammas_trial = np.asarray(activity_model.gammas(T, x_trial))
                gammas_trial_full = np.ones(N)
                gammas_trial_full[fluid_idx] = gammas_trial
            G_trial = _compute_G(n_trial, gammas_trial_full)
            bt_count += 1
        n = n_trial

        iters_done = it + 1
    else:
        msg = f"did not converge in {maxiter} iters, max|θ|={theta_max:.2e}"

    if activity_model is not None and fluid_phase == "liquid":
        n_fluid = n[fluid_idx]
        x_fluid = n_fluid / n_fluid.sum()
        gammas_fluid = np.asarray(activity_model.gammas(T, x_fluid))
        gammas_full = np.ones(N)
        gammas_full[fluid_idx] = gammas_fluid
    else:
        gammas_full = None
    G_final = _compute_G(n, gammas_full)
    G_history.append(G_final)
    atom_balance = float(np.linalg.norm(A_mat @ n - b_vec))

    return GibbsMinResult(
        converged=converged,
        T=T, p=p,
        n=n,
        n_init=n_init_arr,
        species_names=tuple(species_names),
        elements=elements,
        G_total=G_final,
        iterations=iters_done,
        pi=pi_last,
        G_history=tuple(G_history),
        atom_balance_residual=atom_balance,
        message=msg,
    )


def gibbs_minimize_from_thermo(
    T: float,
    p: float,
    species: Sequence,
    formulas: Sequence[Mapping[str, float]],
    n_init: Sequence[float],
    **kwargs,
) -> GibbsMinResult:
    """Convenience wrapper: build ``mu_standard_funcs`` from a list
    of ``SpeciesThermo`` objects (or anything with a ``.Gf(T)`` method).

    Parameters
    ----------
    species : sequence of objects with ``.name`` attribute and ``.Gf(T)``
        ``SpeciesThermo`` from ``stateprop.reaction.thermo``, or
        compatible object.  ``species[i].Gf(T)`` is taken as μ_i°(T).
    formulas : sequence of dict
        Atomic composition per species.
    n_init : sequence of float
    **kwargs :
        Forwarded to ``gibbs_minimize_TP`` (e.g. ``phase='gas'``,
        ``tol``, ``maxiter``, ``verbose``).
    """
    species_names = [s.name for s in species]
    mu_funcs = [s.Gf for s in species]
    return gibbs_minimize_TP(
        T=T, p=p,
        species_names=species_names,
        formulas=formulas,
        mu_standard_funcs=mu_funcs,
        n_init=n_init,
        **kwargs,
    )


# =========================================================================
# Phase split: simultaneous chemical and phase equilibrium
# =========================================================================

def gibbs_minimize_TP_phase_split(
    T: float,
    p: float,
    species_names: Sequence[str],
    formulas: Sequence[Mapping[str, float]],
    mu_standard_funcs: Sequence[Callable[[float], float]],
    psat_funcs: Sequence[Callable[[float], float]],
    n_init: Sequence[float],
    *,
    activity_model = None,
    vapor_eos = None,
    pure_liquid_volumes: Optional[Sequence[float]] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    phase_per_species: Optional[Sequence[str]] = None,
    p_ref: float = _P_REF,
    tol: float = 1e-9,
    maxiter: int = 100,
    damping_init: float = 1.0,
    n_floor: float = 1e-25,
    flash_tol: float = 1e-9,
    reactivation_seed: float = 1e-3,
    max_reactivations: int = 3,
    verbose: bool = False,
) -> "GibbsMinPhaseSplitResult":
    """Minimize G with simultaneous chemical, VLE, and pure-solid
    phase equilibrium.

    Three regimes:

    - ``phase_per_species`` is None (default): all species participate
      in the VLE flash.  Same as v0.9.80/v0.9.81 phase-split solver.
    - ``phase_per_species`` lists some entries as ``'solid'``: those
      species are treated as pure solids (activity = 1, ``μ_i = μ_i°(T)``)
      and don't enter the inner flash.  The remaining 'fluid' species
      ('gas', 'liquid', or 'fluid' — all equivalent) participate in
      the VLE flash and split between the V and L phases.  This is
      the v0.9.82 mode for systems with vapor + liquid + multiple
      pure solids (Boudouard with steam, graphite formation in steam
      reforming, oxide chemistry in solvent media, etc.).
    - All entries are ``'solid'``: degenerate; falls back to a
      single-phase Gibbs minimization on solids only (no flash needed).

    Algorithm:

    1. Identify fluid and solid species from ``phase_per_species``.
    2. Run the inner VLE flash on fluid species only at composition
       ``z_fluid = n_fluid / Σ n_fluid``.  Get ``β, x, y, γ, φ_V``.
    3. Evaluate chemical potentials:
       - Fluid: ``μ_i = μ_i°V + RT ln(y_i φ_V,i p / p_ref)``.
       - Solid: ``μ_i = μ_i°(T)`` (activity 1).
    4. Build the augmented ``(E + 1 + S) × (E + 1 + S)`` RAND linear
       system, where ``S`` is the number of solid species.  Fluid
       species contribute to ``B = Σ_{fluid} a_ki a_li n_i``; solid
       species enter as additional variables ``Δn_i^solid`` with
       additional stationarity equations ``Σ_k π_k a_ki = μ_i°/RT``.
    5. Solve, take a Newton step on ``n``.  Backtrack to ensure
       ``n > 0`` and ``G ≤ G_old``.  Iterate to convergence.

    Parameters
    ----------
    T, p, species_names, formulas, mu_standard_funcs, psat_funcs, n_init :
        Same meaning as v0.9.80/v0.9.81.  ``psat_funcs`` is required
        for fluid species; for solid species the value is unused
        (pass any callable, including ``lambda T: 0.0``).
    activity_model, vapor_eos, pure_liquid_volumes, phi_sat_funcs :
        Inner-flash options.  If ``vapor_eos`` is given, the γ-φ-EOS
        path is used; otherwise modified Raoult.  All operate on the
        full N-species arrays — entries for solid species are simply
        not consulted.
    phase_per_species : sequence of str, length N, optional
        Per-species phase: ``'solid'`` for pure-solid phases,
        ``'gas'`` / ``'liquid'`` / ``'fluid'`` for species that
        participate in the inner VLE flash.  Default: all fluid.

    Returns
    -------
    GibbsMinPhaseSplitResult

    Notes
    -----
    The Gibbs phase rule limits how many independent solid phases
    can coexist with the fluid phases at fixed T, p — at most ``E``
    (number of elements) for a fully constrained system, but in
    practice the augmented matrix is rank-deficient if more solid
    phases are nominated than the data supports.  A ``lstsq``
    fallback handles this gracefully.
    """
    from .reactive_flash import (_modified_raoult_flash,
                                   _gamma_phi_eos_inner_flash)

    N = len(species_names)
    if len(formulas) != N:
        raise ValueError(f"formulas length {len(formulas)} != N={N}")
    if len(mu_standard_funcs) != N:
        raise ValueError(
            f"mu_standard_funcs length {len(mu_standard_funcs)} != N={N}")
    if len(psat_funcs) != N:
        raise ValueError(f"psat_funcs length {len(psat_funcs)} != N={N}")
    if len(n_init) != N:
        raise ValueError(f"n_init length {len(n_init)} != N={N}")
    if vapor_eos is not None:
        if pure_liquid_volumes is not None and \
                len(pure_liquid_volumes) != N:
            raise ValueError(
                f"pure_liquid_volumes length {len(pure_liquid_volumes)} "
                f"!= N={N}")
        if phi_sat_funcs is not None and len(phi_sat_funcs) != N:
            raise ValueError(
                f"phi_sat_funcs length {len(phi_sat_funcs)} != N={N}")
    use_eos = vapor_eos is not None

    # Resolve phase_per_species
    if phase_per_species is None:
        phase_list = ["fluid"] * N
    else:
        if len(phase_per_species) != N:
            raise ValueError(
                f"phase_per_species length {len(phase_per_species)} "
                f"!= N={N}")
        phase_list = list(phase_per_species)
    valid = {"gas", "liquid", "fluid", "solid"}
    for ph in phase_list:
        if ph not in valid:
            raise ValueError(
                f"phase_per_species entries must be in {valid}; "
                f"got {ph!r}")
    is_solid = np.array([ph == "solid" for ph in phase_list], dtype=bool)
    is_fluid = ~is_solid
    fluid_idx = np.where(is_fluid)[0]
    solid_idx = np.where(is_solid)[0]
    S = int(is_solid.sum())
    F = int(is_fluid.sum())

    n_init_arr = np.asarray(n_init, dtype=float)
    if (n_init_arr <= 0).any():
        raise ValueError("all n_init values must be > 0")

    A_mat, elements = _build_atomic_matrix(formulas)
    E = len(elements)
    A_fluid = A_mat[:, fluid_idx]   # (E, F)
    A_solid = A_mat[:, solid_idx]   # (E, S)
    b_vec = A_mat @ n_init_arr

    mu0 = np.array([f(T) for f in mu_standard_funcs])

    if activity_model is None:
        class _IdealLiquid:
            def gammas(self, T, x):
                return np.ones(len(x))
        activity_model = _IdealLiquid()

    n = np.maximum(n_init_arr.copy(), n_floor)
    G_history = []
    # Per-solid reactivation counter (parallel to solid_idx); v0.9.83
    reactivation_count = np.zeros(S, dtype=int) if S > 0 else np.array([])

    # Inner flash operates on fluid species only.  Build per-fluid
    # accessors for psat / pure_liquid_volumes / phi_sat_funcs.
    psat_fluid = [psat_funcs[i] for i in fluid_idx]
    if pure_liquid_volumes is not None:
        pV_fluid = [pure_liquid_volumes[i] for i in fluid_idx]
    else:
        pV_fluid = None
    if phi_sat_funcs is not None:
        psat_phi_fluid = [phi_sat_funcs[i] for i in fluid_idx]
    else:
        psat_phi_fluid = None

    # The activity model expects an array of length F (fluid species
    # only) when the user passed one.  Build a wrapper so the inner
    # flash sees a coherent N=F problem.
    user_activity_model = activity_model

    class _FluidActivityWrapper:
        def __init__(self, base):
            self.base = base
            try:
                self.N = F
            except Exception:
                pass
        def gammas(self, T, x):
            # x has length F; pass directly to base if it expects F,
            # otherwise pad to N
            try:
                return np.asarray(self.base.gammas(T, x))
            except Exception:
                # Pad x to length N (zero out solids), call, slice fluid
                x_full = np.zeros(N)
                x_full[fluid_idx] = x
                return np.asarray(self.base.gammas(T, x_full))[fluid_idx]

    if F < N:
        flash_activity_model = _FluidActivityWrapper(user_activity_model)
    else:
        flash_activity_model = user_activity_model

    def _flash_at(n_arr):
        """Inner VLE flash on fluid species at z_fluid = n_fluid / Σ n_fluid.

        Returns flash quantities (length F) plus phi_v (length F).
        """
        if F == 0:
            # No fluid species; degenerate
            return 0.0, np.array([]), np.array([]), np.array([]), \
                   np.array([]), np.array([])
        n_fluid = n_arr[fluid_idx]
        N_fluid = float(n_fluid.sum())
        z_fluid = n_fluid / N_fluid
        if use_eos:
            beta, x, y, gammas, K, _ = _gamma_phi_eos_inner_flash(
                T, p, z_fluid, flash_activity_model, psat_fluid,
                vapor_eos=vapor_eos,
                pure_liquid_volumes=pV_fluid,
                phi_sat_funcs=psat_phi_fluid,
                tol=flash_tol, maxiter=300)
            try:
                rho_v = vapor_eos.density_from_pressure(
                    p, T, y, phase_hint="vapor")
                ln_phi = vapor_eos.ln_phi(rho_v, T, y)
                phi_v = np.exp(ln_phi)
            except Exception:
                phi_v = np.ones(F)
        else:
            beta, x, y, gammas, K, _ = _modified_raoult_flash(
                T, p, z_fluid, flash_activity_model, psat_fluid,
                tol=flash_tol, maxiter=300)
            phi_v = np.ones(F)
        return beta, x, y, gammas, K, phi_v

    def _compute_mu_full(y_fluid, phi_v_fluid):
        """μ for ALL species.  Fluid: gas-phase formula at y, φ_V from
        flash.  Solid: μ_i°(T) (constant, activity 1)."""
        mu_full = mu0.copy()
        if F > 0:
            y_safe = np.maximum(y_fluid, 1e-300)
            phi_safe = np.maximum(phi_v_fluid, 1e-300)
            mu_full[fluid_idx] = (mu0[fluid_idx]
                                    + _R_GAS * T * (np.log(y_safe)
                                                     + np.log(phi_safe)
                                                     + np.log(p / p_ref)))
        # Solids: mu = mu0 already
        return mu_full

    converged = False
    iters_done = 0
    pi_last = np.zeros(E)
    msg = "did not start"
    theta_max = float("inf")
    beta_last = 0.0
    x_full = np.zeros(N)    # liquid composition padded to N (0 for solids)
    y_full = np.zeros(N)
    gammas_full = np.ones(N)
    K_full = np.zeros(N)
    phi_v_full = np.ones(N)

    for it in range(maxiter):
        beta_last, x_fl, y_fl, gam_fl, K_fl, phi_v_fl = _flash_at(n)
        # Pad fluid quantities to length N for the result
        x_full = np.zeros(N); x_full[fluid_idx] = x_fl
        y_full = np.zeros(N); y_full[fluid_idx] = y_fl
        gammas_full = np.ones(N); gammas_full[fluid_idx] = gam_fl
        K_full = np.zeros(N); K_full[fluid_idx] = K_fl
        phi_v_full = np.ones(N); phi_v_full[fluid_idx] = phi_v_fl

        mu = _compute_mu_full(y_fl, phi_v_fl)
        G_curr = float(np.sum(n * mu))
        G_history.append(G_curr)
        muRT = mu / (_R_GAS * T)

        # B is built from fluid species only; solids contribute zero
        # Hessian.  Solid stationarity adds extra rows / columns.
        n_safe = np.maximum(n, n_floor)
        n_fluid_safe = n_safe[fluid_idx]

        if F > 0:
            B = (A_fluid * n_fluid_safe[np.newaxis, :]) @ A_fluid.T
            b_curr = A_fluid @ n_fluid_safe
            c_vec = A_fluid @ (n_fluid_safe * muRT[fluid_idx])
            c_tot = float(np.sum(n_fluid_safe * muRT[fluid_idx]))
        else:
            # All solids: degenerate; the linear system reduces to
            # A_solid^T π = μ°/RT with no atom-balance forcing
            B = np.zeros((E, E))
            b_curr = np.zeros(E)
            c_vec = np.zeros(E)
            c_tot = 0.0

        # Active-set on solids: exclude any solid that has been driven
        # to the floor (i.e., the previous iteration tried to push it
        # negative and the line search clamped it).  This avoids the
        # "stuck at boundary" failure mode where the augmented system
        # would otherwise force π such that the fluid composition is
        # inconsistent with the floored solid.  After convergence we
        # re-test whether any inactive solid is supersaturated; if so,
        # restart with it added back.
        n_solid_curr = n_safe[solid_idx] if S > 0 else np.array([])
        active_solid_local = (n_solid_curr > 10.0 * n_floor) \
                              if S > 0 else np.array([], dtype=bool)
        S_active = int(active_solid_local.sum())
        active_solid_idx = solid_idx[active_solid_local] \
                            if S > 0 else np.array([], dtype=int)
        A_solid_active = A_solid[:, active_solid_local] \
                          if S > 0 else np.zeros((E, 0))

        if S_active == 0:
            M = np.zeros((E + 1, E + 1))
            M[:E, :E] = B
            M[:E, E] = b_curr
            M[E, :E] = b_curr
            rhs = np.concatenate([c_vec, [c_tot]])
        else:
            sz = E + 1 + S_active
            M = np.zeros((sz, sz))
            M[:E, :E] = B
            M[:E, E] = b_curr
            M[E, :E] = b_curr
            M[:E, E + 1:] = A_solid_active
            M[E + 1:, :E] = A_solid_active.T
            mu_solid_RT_active = muRT[active_solid_idx]
            rhs = np.concatenate([c_vec, [c_tot], mu_solid_RT_active])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        pi = sol[:E]
        u_var = sol[E]
        delta_n_solid_active = sol[E + 1:] if S_active > 0 else np.array([])
        pi_last = pi

        theta_full = muRT - A_mat.T @ pi
        delta_n = np.zeros(N)
        if F > 0:
            delta_n[fluid_idx] = n_fluid_safe * (-theta_full[fluid_idx]
                                                  + u_var)
        # Δn for active solids comes from solution; inactive solids
        # stay at floor (Δn = 0).  No ad-hoc reactivation: the user
        # should provide reasonable initial seeds for solids they
        # expect to form (any positive value above ~100*n_floor).  If
        # an inactive solid is found to be supersaturated at
        # convergence (its θ residual indicates a positive driving
        # force to form), this is reported via ``ss_violation`` in
        # the result message — the user can re-run with a positive
        # seed for that species.
        if S > 0:
            for k_loc, i in enumerate(solid_idx):
                if active_solid_local[k_loc]:
                    j_active = int(active_solid_local[:k_loc + 1].sum() - 1)
                    delta_n[i] = delta_n_solid_active[j_active]
                # else: leave Δn[i] = 0 (inactive solid stays at floor)

        # Convergence test:
        # - For fluid + active solids: |θ_i| < tol (stationarity).
        # - For inactive solids: must be undersaturated, i.e. not
        #   supersaturated, i.e. -θ_i ≤ 0 (= μ°/RT - Σπa ≤ 0).
        active_mask = np.ones(N, dtype=bool)
        if S > 0:
            active_mask[solid_idx[~active_solid_local]] = False
        theta_active = theta_full[active_mask]
        theta_max = float(np.max(np.abs(theta_active))) if theta_active.size else 0.0
        # Supersaturation check for inactive solids
        ss_violation = 0.0
        if S > 0 and (~active_solid_local).any():
            inactive_idx = solid_idx[~active_solid_local]
            # If supersaturated, -theta > 0 (positive driving force to form)
            ss_terms = -theta_full[inactive_idx]
            ss_violation = float(max(0.0, np.max(ss_terms)) if ss_terms.size else 0.0)
        if verbose:
            print(f"  iter {it:3d}: G = {G_curr:.4e}, "
                  f"max|θ_active| = {theta_max:.2e}, "
                  f"ss_violation = {ss_violation:.2e}, "
                  f"β = {beta_last:.4f}, S_active={S_active}/{S}, "
                  f"||A n - b|| = {np.linalg.norm(A_mat @ n - b_vec):.2e}")
        if theta_max < tol and ss_violation < tol:
            converged = True
            iters_done = it + 1
            msg = (f"converged in {iters_done} iters, "
                   f"max|θ|={theta_max:.2e}, ss={ss_violation:.2e}")
            break

        alpha = damping_init
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(delta_n < 0, -n / delta_n, np.inf)
            alpha_max = float(np.min(ratios))
        alpha = min(alpha, 0.95 * alpha_max if alpha_max > 0 else 0.5)
        if alpha <= 0:
            alpha = 1e-6

        n_trial = np.maximum(n + alpha * delta_n, n_floor)
        try:
            _, _, y_tr, _, _, phi_v_tr = _flash_at(n_trial)
            G_trial = float(np.sum(n_trial *
                                    _compute_mu_full(y_tr, phi_v_tr)))
        except Exception:
            G_trial = G_curr + 1.0
        bt_count = 0
        while G_trial > G_curr + 1e-9 * abs(G_curr) and bt_count < 30:
            alpha *= 0.5
            n_trial = np.maximum(n + alpha * delta_n, n_floor)
            try:
                _, _, y_tr, _, _, phi_v_tr = _flash_at(n_trial)
                G_trial = float(np.sum(n_trial *
                                        _compute_mu_full(y_tr, phi_v_tr)))
            except Exception:
                G_trial = G_curr + 1.0
            bt_count += 1
        n = n_trial
        iters_done = it + 1

        # Active-set re-activation: check inactive solids for
        # supersaturation at the just-computed π.  An inactive solid i
        # is supersaturated when -θ_i = Σ_k π_k a_ki - μ_i°/RT > 0,
        # meaning chemistry-and-phase equilibrium would prefer to form
        # it.  Re-seed such solids with a small positive value so the
        # next iteration brings them back into the active set.  When a
        # solid is reactivated, the atom-balance baseline ``b_vec`` is
        # updated to reflect the new inventory (the seed adds atoms to
        # the system).  Each solid is only eligible to be reactivated
        # up to ``max_reactivations`` times to prevent infinite
        # oscillation at a phase boundary.
        if S > 0:
            reactivated_any = False
            for k_loc, i in enumerate(solid_idx):
                if not active_solid_local[k_loc]:
                    if -theta_full[i] > tol \
                            and reactivation_count[k_loc] < max_reactivations:
                        n[i] = max(n[i], reactivation_seed)
                        reactivation_count[k_loc] += 1
                        reactivated_any = True
                        if verbose:
                            print(f"    reactivating solid '{species_names[i]}'"
                                  f" (count={reactivation_count[k_loc]})")
            if reactivated_any:
                # Update b_vec to reflect the new atom inventory
                b_vec = A_mat @ n
    else:
        msg = f"did not converge in {maxiter} iters, max|θ|={theta_max:.2e}"

    beta_last, x_fl, y_fl, gam_fl, K_fl, phi_v_fl = _flash_at(n)
    x_full = np.zeros(N); x_full[fluid_idx] = x_fl
    y_full = np.zeros(N); y_full[fluid_idx] = y_fl
    gammas_full = np.ones(N); gammas_full[fluid_idx] = gam_fl
    K_full = np.zeros(N); K_full[fluid_idx] = K_fl
    phi_v_full = np.ones(N); phi_v_full[fluid_idx] = phi_v_fl
    mu_final = _compute_mu_full(y_fl, phi_v_fl)
    G_final = float(np.sum(n * mu_final))
    G_history.append(G_final)
    atom_balance = float(np.linalg.norm(A_mat @ n - b_vec))

    return GibbsMinPhaseSplitResult(
        converged=converged,
        T=T, p=p,
        n=n,
        n_init=n_init_arr,
        species_names=tuple(species_names),
        elements=elements,
        G_total=G_final,
        iterations=iters_done,
        pi=pi_last,
        G_history=tuple(G_history),
        atom_balance_residual=atom_balance,
        message=msg,
        beta=float(beta_last),
        x_liquid=x_full,
        y_vapor=y_full,
        gammas=gammas_full,
        K=K_full,
    )


@dataclass
class GibbsMinPhaseSplitResult(GibbsMinResult):
    """Result of a phase-split Gibbs minimization.

    Adds the converged inner-flash quantities to the base result.
    Note: ``y`` is inherited as a property (overall mole fractions).
    The vapor mole fractions are exposed as ``y_vapor``; liquid as
    ``x_liquid``.
    """
    beta: float = 0.0          # vapor fraction
    x_liquid: np.ndarray = field(default_factory=lambda: np.array([]))
    y_vapor: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas: np.ndarray = field(default_factory=lambda: np.array([]))
    K: np.ndarray = field(default_factory=lambda: np.array([]))


# =========================================================================
# Liquid-liquid phase split (LLE) Gibbs minimization
# =========================================================================

@dataclass
class GibbsMinLLSplitResult(GibbsMinResult):
    """Result of an LLE Gibbs minimization.

    Adds the converged LL-flash quantities to the base result.
    ``beta`` is the mole fraction of feed in phase 2.
    """
    beta: float = 0.0          # mole fraction in phase 2 (per LLEFlash convention)
    x1: np.ndarray = field(default_factory=lambda: np.array([]))   # phase-1 composition
    x2: np.ndarray = field(default_factory=lambda: np.array([]))   # phase-2 composition
    gammas1: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas2: np.ndarray = field(default_factory=lambda: np.array([]))


def gibbs_minimize_TP_LL_split(
    T: float,
    p: float,
    species_names: Sequence[str],
    formulas: Sequence[Mapping[str, float]],
    mu_standard_funcs: Sequence[Callable[[float], float]],
    activity_model,
    n_init: Sequence[float],
    x1_seed: Sequence[float],
    x2_seed: Sequence[float],
    *,
    p_ref: float = _P_REF,
    tol: float = 1e-8,
    maxiter: int = 100,
    damping_init: float = 1.0,
    n_floor: float = 1e-25,
    flash_tol: float = 1e-8,
    verbose: bool = False,
) -> GibbsMinLLSplitResult:
    """Minimize G with simultaneous chemical and LIQUID-LIQUID phase
    equilibrium.

    Identical structure to ``gibbs_minimize_TP_phase_split`` but with
    a two-liquid-phase inner flash from ``stateprop.activity.lle.LLEFlash``
    instead of a vapor-liquid flash.  No saturation pressures or vapor
    EOS are required: the only thermodynamic input besides ``μ_i°L(T)``
    is the activity-coefficient model (typically ``make_unifac_lle``
    for solubility-style problems, or NRTL with binary parameters).

    The chemical potential at the converged inner flash is

        μ_i = μ_i°L(T) + RT ln(γ_i x_i)

    evaluated on EITHER liquid phase — they are equal at LL
    equilibrium because that is the inner flash's convergence
    condition.  The outer RAND step on the TOTAL moles ``n_i`` then
    adjusts chemistry until ``μ_i / RT = Σ_k π_k a_ki`` for all
    species.

    Parameters
    ----------
    T, p : float
        Temperature [K] and pressure [Pa] (pressure used only in
        ``μ_i°L`` if the user has folded a Poynting correction in;
        otherwise unused).
    species_names, formulas, n_init :
        As in ``gibbs_minimize_TP_phase_split``.
    mu_standard_funcs : sequence of callables T -> J/mol
        **Liquid-phase** reference chemical potentials.  Often
        ``μ_i°L(T) = μ_i°V(T) + RT ln(p_sat,i(T) / p_ref)`` derived
        from a SpeciesThermo (gas-phase Gf) and a p_sat correlation.
        For pure-component standard states the simple form
        ``Gf_liquid(T) = Gf_gas(T) - latent_heat_correction`` works.
    activity_model : object with ``.gammas(T, x)`` returning length-N array
        UNIFAC-LLE / NRTL / UNIQUAC.  Must reflect the actual phase
        behavior (regular UNIFAC may not capture LLE correctly for
        some systems; UNIFAC-LLE has special LLE-fit parameters).
    x1_seed, x2_seed : sequence of float, length N
        Initial guesses for the two liquid compositions.  Must be
        sufficiently different (max |x1 - x2| > 1e-5) to seed a
        2-phase flash; otherwise the inner LLEFlash raises.

    Returns
    -------
    GibbsMinLLSplitResult

    Notes
    -----
    Solid-phase support is not provided in this LL solver.  For
    chemistry with both solids and LL split, the right structure is
    a 4-phase Gibbs minimization (V + L1 + L2 + S), not currently
    implemented.

    Solubility-only (no chemistry) usage: pass an empty reaction set
    (i.e., n_init that doesn't violate atom balance and no actual
    reactions in the formulas) and the algorithm reduces to a single
    LL flash with the chemistry residual identically zero.
    """
    from .reactive_flash import _modified_raoult_flash  # for typing only
    from ..activity.lle import LLEFlash

    N = len(species_names)
    if len(formulas) != N:
        raise ValueError(f"formulas length {len(formulas)} != N={N}")
    if len(mu_standard_funcs) != N:
        raise ValueError(
            f"mu_standard_funcs length {len(mu_standard_funcs)} != N={N}")
    if len(n_init) != N:
        raise ValueError(f"n_init length {len(n_init)} != N={N}")
    if len(x1_seed) != N or len(x2_seed) != N:
        raise ValueError(f"x1_seed and x2_seed must have length N={N}")

    n_init_arr = np.asarray(n_init, dtype=float)
    if (n_init_arr <= 0).any():
        raise ValueError("all n_init values must be > 0")

    A_mat, elements = _build_atomic_matrix(formulas)
    E = len(elements)
    b_vec = A_mat @ n_init_arr

    mu0 = np.array([f(T) for f in mu_standard_funcs])

    lle = LLEFlash(activity_model)
    n = np.maximum(n_init_arr.copy(), n_floor)
    G_history = []

    # Persistent x1/x2 between iterations (warm start)
    x1_curr = np.asarray(x1_seed, dtype=float).copy()
    x1_curr = x1_curr / x1_curr.sum()
    x2_curr = np.asarray(x2_seed, dtype=float).copy()
    x2_curr = x2_curr / x2_curr.sum()

    def _flash_at(n_arr):
        """LL flash at composition z = n_arr / Σ n_arr.  Returns
        (beta, x1, x2, gammas1, gammas2)."""
        nonlocal x1_curr, x2_curr
        N_tot = float(n_arr.sum())
        z = n_arr / N_tot
        try:
            res = lle.solve(T, z, x1_curr, x2_curr,
                             tol=flash_tol, maxiter=200)
            x1_curr = np.asarray(res.x1)
            x2_curr = np.asarray(res.x2)
            beta = float(res.beta)
            g1 = np.asarray(activity_model.gammas(T, x1_curr))
            g2 = np.asarray(activity_model.gammas(T, x2_curr))
            return beta, x1_curr, x2_curr, g1, g2
        except Exception:
            # If LL flash collapses or fails, fall back to single-phase.
            # The "x1 = x2 = z" choice corresponds to no phase split.
            x1_curr = z.copy()
            x2_curr = z.copy()
            g = np.asarray(activity_model.gammas(T, z))
            return 0.0, z, z, g, g

    def _compute_mu_global(x_arr, gammas):
        """μ_i = μ_i°L + RT ln(γ_i x_i).  Same in both phases at LL
        equilibrium."""
        x_safe = np.maximum(x_arr, 1e-300)
        g_safe = np.maximum(gammas, 1e-300)
        return mu0 + _R_GAS * T * (np.log(g_safe) + np.log(x_safe))

    converged = False
    iters_done = 0
    pi_last = np.zeros(E)
    msg = "did not start"
    theta_max = float("inf")
    beta_last = 0.0
    x1_last = x1_curr.copy()
    x2_last = x2_curr.copy()
    g1_last = np.ones(N)
    g2_last = np.ones(N)

    for it in range(maxiter):
        beta_last, x1_last, x2_last, g1_last, g2_last = _flash_at(n)

        # Use phase 1 for μ; equally valid would be phase 2.
        mu = _compute_mu_global(x1_last, g1_last)
        G_curr = float(np.sum(n * mu))
        G_history.append(G_curr)
        muRT = mu / (_R_GAS * T)

        # Standard (E+1)x(E+1) RAND on TOTAL n
        n_safe = np.maximum(n, n_floor)
        B = (A_mat * n_safe[np.newaxis, :]) @ A_mat.T
        b_curr = A_mat @ n_safe
        c_vec = A_mat @ (n_safe * muRT)
        c_tot = float(np.sum(n_safe * muRT))

        M = np.zeros((E + 1, E + 1))
        M[:E, :E] = B
        M[:E, E] = b_curr
        M[E, :E] = b_curr
        rhs = np.concatenate([c_vec, [c_tot]])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        pi = sol[:E]
        u_var = sol[E]
        pi_last = pi

        theta = muRT - A_mat.T @ pi
        delta_n = n_safe * (-theta + u_var)

        theta_max = float(np.max(np.abs(theta)))
        if verbose:
            print(f"  iter {it:3d}: G = {G_curr:.4e}, "
                  f"max|θ| = {theta_max:.2e}, β_LL = {beta_last:.4f}, "
                  f"||A n - b|| = {np.linalg.norm(b_curr - b_vec):.2e}")
        if theta_max < tol:
            converged = True
            iters_done = it + 1
            msg = f"converged in {iters_done} iters, max|θ|={theta_max:.2e}"
            break

        alpha = damping_init
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(delta_n < 0, -n / delta_n, np.inf)
            alpha_max = float(np.min(ratios))
        alpha = min(alpha, 0.95 * alpha_max if alpha_max > 0 else 0.5)
        if alpha <= 0:
            alpha = 1e-6

        n_trial = np.maximum(n + alpha * delta_n, n_floor)
        try:
            _, x1_tr, _, g1_tr, _ = _flash_at(n_trial)
            G_trial = float(np.sum(n_trial *
                                    _compute_mu_global(x1_tr, g1_tr)))
        except Exception:
            G_trial = G_curr + 1.0
        bt_count = 0
        while G_trial > G_curr + 1e-9 * abs(G_curr) and bt_count < 30:
            alpha *= 0.5
            n_trial = np.maximum(n + alpha * delta_n, n_floor)
            try:
                _, x1_tr, _, g1_tr, _ = _flash_at(n_trial)
                G_trial = float(np.sum(n_trial *
                                        _compute_mu_global(x1_tr, g1_tr)))
            except Exception:
                G_trial = G_curr + 1.0
            bt_count += 1
        n = n_trial
        iters_done = it + 1
    else:
        msg = f"did not converge in {maxiter} iters, max|θ|={theta_max:.2e}"

    beta_last, x1_last, x2_last, g1_last, g2_last = _flash_at(n)
    mu_final = _compute_mu_global(x1_last, g1_last)
    G_final = float(np.sum(n * mu_final))
    G_history.append(G_final)
    atom_balance = float(np.linalg.norm(A_mat @ n - b_vec))

    return GibbsMinLLSplitResult(
        converged=converged,
        T=T, p=p,
        n=n,
        n_init=n_init_arr,
        species_names=tuple(species_names),
        elements=elements,
        G_total=G_final,
        iterations=iters_done,
        pi=pi_last,
        G_history=tuple(G_history),
        atom_balance_residual=atom_balance,
        message=msg,
        beta=float(beta_last),
        x1=x1_last,
        x2=x2_last,
        gammas1=g1_last,
        gammas2=g2_last,
    )


# =========================================================================
# VLLE Gibbs minimization (vapor + 2 liquids + chemistry)
# =========================================================================

@dataclass
class GibbsMinVLLSplitResult(GibbsMinResult):
    """Result of a 3-phase (V + L1 + L2) Gibbs minimization."""
    beta_V: float = 0.0
    beta_L1: float = 0.0
    beta_L2: float = 0.0
    x1: np.ndarray = field(default_factory=lambda: np.array([]))
    x2: np.ndarray = field(default_factory=lambda: np.array([]))
    y_vapor: np.ndarray = field(default_factory=lambda: np.array([]))
    K_y: np.ndarray = field(default_factory=lambda: np.array([]))
    K_x: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas1: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas2: np.ndarray = field(default_factory=lambda: np.array([]))


def gibbs_minimize_TP_VLL_split(
    T: float,
    p: float,
    species_names: Sequence[str],
    formulas: Sequence[Mapping[str, float]],
    mu_standard_funcs: Sequence[Callable[[float], float]],
    psat_funcs: Sequence[Callable[[float], float]],
    activity_model,
    vapor_eos,
    n_init: Sequence[float],
    x1_seed: Sequence[float],
    x2_seed: Sequence[float],
    *,
    pure_liquid_volumes: Optional[Sequence[float]] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    p_ref: float = _P_REF,
    tol: float = 1e-8,
    maxiter: int = 100,
    damping_init: float = 1.0,
    n_floor: float = 1e-25,
    flash_tol: float = 1e-7,
    beta_V_seed: float = 0.05,
    beta_L2_seed: float = 0.3,
    verbose: bool = False,
) -> GibbsMinVLLSplitResult:
    """Minimize G with simultaneous chemical and 3-phase VLLE
    equilibrium (vapor + two coexisting liquid phases).

    Inner flash uses ``GammaPhiEOSThreePhaseFlash.isothermal_3phase``
    (γ-φ-EOS for vapor, activity model for both liquids, full
    Rachford-Rice on the 3-phase split).  Outer RAND iterates on the
    total moles ``n_i`` per species; atom balance is on the totals.

    Phase equilibrium gives ``μ_i^V = μ_i^L1 = μ_i^L2`` at the
    converged inner flash, so the chemical potential

        μ_i = μ_i°V(T) + RT ln(y_i p / p_ref) + RT ln(φ_V,i)

    is uniquely defined (the vapor-side formula is most direct, but
    using the L1 or L2 formula gives identical values).  The outer
    Newton then drives this μ to satisfy ``μ_i / RT = Σ_k π_k a_ki``
    (chemistry + atom-balance stationarity).

    Parameters
    ----------
    T, p : float
        Temperature [K] and pressure [Pa].
    species_names, formulas, mu_standard_funcs : as ``gibbs_minimize_TP_phase_split``
        Standard chemical potentials are gas-phase Gibbs of formation.
    psat_funcs : sequence of callables T -> Pa
        Pure-component saturation pressures.
    activity_model : object with ``.gammas(T, x)``
        Liquid activity-coefficient model.  **Must reflect LLE** —
        regular UNIFAC may not capture LLE; UNIFAC-LLE or NRTL
        with experimentally fitted binary parameters is recommended.
    vapor_eos : EOS mixture
        Required (unlike the 2-phase phase-split solver where it is
        optional).  ``CubicMixture`` from the cubic module qualifies.
    n_init : sequence of float, length N
        Total initial moles per species (sum across all phases).
    x1_seed, x2_seed : sequence of float, length N
        Initial guesses for the two liquid-phase compositions.  Must
        be sufficiently different to seed the LL split.
    pure_liquid_volumes, phi_sat_funcs : optional
        Same Poynting + Φ_sat enhancements as in the 2-phase
        γ-φ-EOS path.
    beta_V_seed, beta_L2_seed : float
        Initial guesses for the vapor and second-liquid mole fractions
        (passed to the inner 3-phase flash).
    tol, maxiter, damping_init, n_floor, flash_tol :
        Newton/Picard tolerances.

    Returns
    -------
    GibbsMinVLLSplitResult

    Notes
    -----
    Solid-phase support is not provided here.  For VLLE + solid +
    chemistry (e.g., a 4-phase reactive equilibrium), the natural
    extension would be a 4-phase inner flash plus the v0.9.82 active-
    set solid handling — not currently implemented.

    The 3-phase inner flash is more sensitive to initial seeds than
    the 2-phase variants.  If ``x1_seed`` and ``x2_seed`` do not
    bracket a true LL split at the current ``z = n / Σn``, the inner
    flash may collapse to 2-phase (V + single L) — no error is
    raised, the algorithm proceeds with the collapsed result.  The
    user should validate by checking ``res.beta_L2`` (very small
    means no real 2-liquid split was found).
    """
    from ..activity.gamma_phi_eos_3phase import GammaPhiEOSThreePhaseFlash

    N = len(species_names)
    if len(formulas) != N:
        raise ValueError(f"formulas length {len(formulas)} != N={N}")
    if len(mu_standard_funcs) != N:
        raise ValueError(
            f"mu_standard_funcs length {len(mu_standard_funcs)} != N={N}")
    if len(psat_funcs) != N:
        raise ValueError(f"psat_funcs length {len(psat_funcs)} != N={N}")
    if len(n_init) != N:
        raise ValueError(f"n_init length {len(n_init)} != N={N}")
    if len(x1_seed) != N or len(x2_seed) != N:
        raise ValueError(f"x1_seed and x2_seed must have length N={N}")
    if pure_liquid_volumes is not None and len(pure_liquid_volumes) != N:
        raise ValueError(
            f"pure_liquid_volumes length != N={N}")
    if phi_sat_funcs is not None and len(phi_sat_funcs) != N:
        raise ValueError(f"phi_sat_funcs length != N={N}")

    n_init_arr = np.asarray(n_init, dtype=float)
    if (n_init_arr <= 0).any():
        raise ValueError("all n_init values must be > 0")

    A_mat, elements = _build_atomic_matrix(formulas)
    E = len(elements)
    b_vec = A_mat @ n_init_arr

    mu0 = np.array([f(T) for f in mu_standard_funcs])

    # Build the 3-phase flash object once (constant across iterations)
    flash3 = GammaPhiEOSThreePhaseFlash(
        activity_model=activity_model,
        psat_funcs=psat_funcs,
        vapor_eos=vapor_eos,
        pure_liquid_volumes=pure_liquid_volumes,
        phi_sat_funcs=phi_sat_funcs,
    )

    n = np.maximum(n_init_arr.copy(), n_floor)
    G_history = []

    # Persistent seeds (warm-start across outer iterations)
    x1_curr = np.asarray(x1_seed, dtype=float).copy()
    x1_curr = x1_curr / x1_curr.sum()
    x2_curr = np.asarray(x2_seed, dtype=float).copy()
    x2_curr = x2_curr / x2_curr.sum()
    bV_curr = float(beta_V_seed)
    bL2_curr = float(beta_L2_seed)

    def _flash_at(n_arr):
        """Auto-dispatched isothermal flash at z = n_arr/Σn_arr.
        The 3-phase routine ``isothermal_3phase`` strictly REQUIRES a
        3-phase region to converge; we instead use ``auto_isothermal``
        which performs a stability test and dispatches to the right
        phase configuration (1L / 1V / 2VL / 2LL / 3VLL).  Then we
        translate the 1-, 2-, or 3-phase result into the unified
        (β_V, β_L1, β_L2, x1, x2, y, K_y, K_x, γ_1, γ_2, φ_V) tuple.
        """
        nonlocal x1_curr, x2_curr, bV_curr, bL2_curr
        N_tot = float(n_arr.sum())
        z = n_arr / N_tot

        # Try auto_isothermal first
        result = None
        try:
            r_auto = flash3.auto_isothermal(T=T, p=p, z=z,
                                              tol=flash_tol, maxiter=300)
            result = r_auto
        except Exception:
            result = None

        beta_V = 0.0
        beta_L1 = 0.0
        beta_L2 = 0.0
        x1 = z.copy()
        x2 = z.copy()
        y = z.copy()
        K_y = np.ones(N)
        K_x = np.ones(N)

        if result is not None:
            r = result.result
            ptype = result.phase_type
            if ptype == "3VLL":
                beta_V = float(r.beta_V)
                beta_L2 = float(r.beta_L2)
                beta_L1 = 1.0 - beta_V - beta_L2
                x1 = np.asarray(r.x1)
                x2 = np.asarray(r.x2)
                y = np.asarray(r.y)
                K_y = np.asarray(r.K_y)
                K_x = np.asarray(r.K_x)
                x1_curr = x1
                x2_curr = x2
                bV_curr = max(1e-6, min(0.999, beta_V))
                bL2_curr = max(1e-6, min(0.999, beta_L2))
            elif ptype == "2VL":
                # 2-phase VLE: V + L1 (no second liquid)
                beta_V = float(r.V)
                beta_L1 = 1.0 - beta_V
                beta_L2 = 0.0
                x1 = np.asarray(r.x)
                x2 = np.asarray(r.x)   # collapsed
                y = np.asarray(r.y)
                K_y = np.asarray(r.K)
                K_x = np.ones(N)
                x1_curr = x1
                bV_curr = max(1e-6, min(0.999, beta_V))
            elif ptype == "2LL":
                # 2-phase LLE (no vapor)
                # LLEResult: beta = mole fraction in phase 2
                beta_V = 0.0
                beta_L2 = float(r.beta)
                beta_L1 = 1.0 - beta_L2
                x1 = np.asarray(r.x1)
                x2 = np.asarray(r.x2)
                y = z.copy()    # fictitious — used only to define μ
                K_x = np.asarray(r.K)
                K_y = np.ones(N)
                x1_curr = x1
                x2_curr = x2
                bL2_curr = max(1e-6, min(0.999, beta_L2))
            elif ptype == "1L":
                # Single liquid phase
                beta_V = 0.0
                beta_L1 = 1.0
                beta_L2 = 0.0
                x1 = z.copy()
                x2 = z.copy()
                y = z.copy()
            elif ptype == "1V":
                # Single vapor phase
                beta_V = 1.0
                beta_L1 = 0.0
                beta_L2 = 0.0
                y = z.copy()
                x1 = z.copy()
                x2 = z.copy()

        # Compute γ on each liquid composition
        try:
            g1 = np.asarray(activity_model.gammas(T, x1))
        except Exception:
            g1 = np.ones(N)
        try:
            g2 = np.asarray(activity_model.gammas(T, x2))
        except Exception:
            g2 = np.ones(N)

        # For all-liquid cases, define a "fictitious" vapor composition
        # consistent with the chemical potential: at converged phase
        # equilibrium, μ_i = μ_i°V + RT ln(y_i φ_V p / p_ref).
        # If only liquid exists, equate μ_L = μ_L_ideal_gas_at_psat:
        #   μ_i = μ_i°V + RT ln(γ_i x_i p_sat,i / p_ref)
        # so the equivalent y_i for the gas-phase formula is
        #   y_i φ_V,i p = γ_i x_i p_sat,i  (modified Raoult; φ_V=1 for ideal-gas reference)
        if beta_V <= 0:
            # Use the dominant liquid (L1) for y
            psats = np.array([f(T) for f in psat_funcs])
            x_dom = x1
            g_dom = g1
            y = g_dom * x_dom * psats / p
            y = np.maximum(y, 1e-300)
            # No need to renormalize — the formula μ = μ° + RT ln(y φ_V p/p_ref)
            # is self-consistent at phase equilibrium without normalization.

        # φ_V: if there's a vapor phase, compute from EOS at y; else φ_V=1
        if beta_V > 0:
            try:
                rho_v = vapor_eos.density_from_pressure(
                    p, T, y, phase_hint="vapor")
                ln_phi = vapor_eos.ln_phi(rho_v, T, y)
                phi_v = np.exp(ln_phi)
            except Exception:
                phi_v = np.ones(N)
        else:
            phi_v = np.ones(N)

        return (beta_V, beta_L1, beta_L2, x1, x2, y,
                K_y, K_x, g1, g2, phi_v)

    def _compute_mu_global(y_arr, phi_v_arr):
        """μ_i = μ_i°V + RT ln(y_i φ_V,i p / p_ref).  Equal across
        all three phases at converged inner flash (V + L1 + L2 phase
        equilibrium)."""
        y_safe = np.maximum(y_arr, 1e-300)
        phi_safe = np.maximum(phi_v_arr, 1e-300)
        return mu0 + _R_GAS * T * (np.log(y_safe) + np.log(phi_safe)
                                    + np.log(p / p_ref))

    converged = False
    iters_done = 0
    pi_last = np.zeros(E)
    msg = "did not start"
    theta_max = float("inf")
    bV_last = 0.0
    bL1_last = 0.0
    bL2_last = 0.0
    x1_last = x1_curr.copy()
    x2_last = x2_curr.copy()
    y_last = np.zeros(N)
    K_y_last = np.zeros(N)
    K_x_last = np.zeros(N)
    g1_last = np.ones(N)
    g2_last = np.ones(N)
    phi_v_last = np.ones(N)

    for it in range(maxiter):
        (bV_last, bL1_last, bL2_last, x1_last, x2_last, y_last,
         K_y_last, K_x_last, g1_last, g2_last, phi_v_last) = _flash_at(n)

        mu = _compute_mu_global(y_last, phi_v_last)
        G_curr = float(np.sum(n * mu))
        G_history.append(G_curr)
        muRT = mu / (_R_GAS * T)

        n_safe = np.maximum(n, n_floor)
        B = (A_mat * n_safe[np.newaxis, :]) @ A_mat.T
        b_curr = A_mat @ n_safe
        c_vec = A_mat @ (n_safe * muRT)
        c_tot = float(np.sum(n_safe * muRT))

        M = np.zeros((E + 1, E + 1))
        M[:E, :E] = B
        M[:E, E] = b_curr
        M[E, :E] = b_curr
        rhs = np.concatenate([c_vec, [c_tot]])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        pi = sol[:E]
        u_var = sol[E]
        pi_last = pi

        theta = muRT - A_mat.T @ pi
        delta_n = n_safe * (-theta + u_var)

        theta_max = float(np.max(np.abs(theta)))
        if verbose:
            print(f"  iter {it:3d}: G = {G_curr:.4e}, "
                  f"max|θ| = {theta_max:.2e}, "
                  f"β_V = {bV_last:.3f}, β_L2 = {bL2_last:.3f}, "
                  f"||A n - b|| = {np.linalg.norm(b_curr - b_vec):.2e}")
        if theta_max < tol:
            converged = True
            iters_done = it + 1
            msg = f"converged in {iters_done} iters, max|θ|={theta_max:.2e}"
            break

        alpha = damping_init
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(delta_n < 0, -n / delta_n, np.inf)
            alpha_max = float(np.min(ratios))
        alpha = min(alpha, 0.95 * alpha_max if alpha_max > 0 else 0.5)
        if alpha <= 0:
            alpha = 1e-6

        n_trial = np.maximum(n + alpha * delta_n, n_floor)
        try:
            (_, _, _, _, _, y_tr, _, _, _, _, phi_v_tr) = _flash_at(n_trial)
            G_trial = float(np.sum(n_trial *
                                    _compute_mu_global(y_tr, phi_v_tr)))
        except Exception:
            G_trial = G_curr + 1.0
        bt_count = 0
        while G_trial > G_curr + 1e-9 * abs(G_curr) and bt_count < 30:
            alpha *= 0.5
            n_trial = np.maximum(n + alpha * delta_n, n_floor)
            try:
                (_, _, _, _, _, y_tr, _, _, _, _, phi_v_tr) = _flash_at(n_trial)
                G_trial = float(np.sum(n_trial *
                                        _compute_mu_global(y_tr, phi_v_tr)))
            except Exception:
                G_trial = G_curr + 1.0
            bt_count += 1
        n = n_trial
        iters_done = it + 1
    else:
        msg = f"did not converge in {maxiter} iters, max|θ|={theta_max:.2e}"

    (bV_last, bL1_last, bL2_last, x1_last, x2_last, y_last,
     K_y_last, K_x_last, g1_last, g2_last, phi_v_last) = _flash_at(n)
    mu_final = _compute_mu_global(y_last, phi_v_last)
    G_final = float(np.sum(n * mu_final))
    G_history.append(G_final)
    atom_balance = float(np.linalg.norm(A_mat @ n - b_vec))

    return GibbsMinVLLSplitResult(
        converged=converged,
        T=T, p=p,
        n=n,
        n_init=n_init_arr,
        species_names=tuple(species_names),
        elements=elements,
        G_total=G_final,
        iterations=iters_done,
        pi=pi_last,
        G_history=tuple(G_history),
        atom_balance_residual=atom_balance,
        message=msg,
        beta_V=float(bV_last), beta_L1=float(bL1_last), beta_L2=float(bL2_last),
        x1=x1_last, x2=x2_last, y_vapor=y_last,
        K_y=K_y_last, K_x=K_x_last,
        gammas1=g1_last, gammas2=g2_last,
    )


# =========================================================================
# 4-phase reactive equilibrium: V + L1 + L2 + S + chemistry  (v0.9.85)
# =========================================================================

@dataclass
class GibbsMin4PhaseSplitResult(GibbsMinResult):
    """Result of a 4-phase (V + L1 + L2 + multiple solids) Gibbs minimization."""
    beta_V: float = 0.0
    beta_L1: float = 0.0
    beta_L2: float = 0.0
    x1: np.ndarray = field(default_factory=lambda: np.array([]))
    x2: np.ndarray = field(default_factory=lambda: np.array([]))
    y_vapor: np.ndarray = field(default_factory=lambda: np.array([]))
    K_y: np.ndarray = field(default_factory=lambda: np.array([]))
    K_x: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas1: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas2: np.ndarray = field(default_factory=lambda: np.array([]))


def gibbs_minimize_TP_VLLS_split(
    T: float,
    p: float,
    species_names: Sequence[str],
    formulas: Sequence[Mapping[str, float]],
    mu_standard_funcs: Sequence[Callable[[float], float]],
    psat_funcs: Sequence[Callable[[float], float]],
    activity_model,
    vapor_eos,
    n_init: Sequence[float],
    x1_seed: Sequence[float],
    x2_seed: Sequence[float],
    *,
    pure_liquid_volumes: Optional[Sequence[float]] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    phase_per_species: Optional[Sequence[str]] = None,
    p_ref: float = _P_REF,
    tol: float = 1e-7,
    maxiter: int = 100,
    damping_init: float = 1.0,
    n_floor: float = 1e-25,
    flash_tol: float = 1e-7,
    beta_V_seed: float = 0.05,
    beta_L2_seed: float = 0.3,
    reactivation_seed: float = 1e-3,
    max_reactivations: int = 3,
    verbose: bool = False,
) -> GibbsMin4PhaseSplitResult:
    """Minimize G with simultaneous chemical, 3-phase VLLE, and
    pure-solid equilibrium (the most general single-vessel reactive
    equilibrium in the library).

    Combines:
      - The v0.9.84 3-phase inner flash (auto_isothermal dispatch
        across 1L / 1V / 2VL / 2LL / 3VLL).
      - The v0.9.82 augmented RAND linear system for solid species
        (each solid contributes a stationarity equation
        ``A_solid^T π = μ°_solid / RT``).
      - The v0.9.83 active-set re-activation: solids whose mole
        numbers fall to the floor are excluded from the linear
        system; supersaturated inactive solids are re-seeded.
      - Atom-balance baseline ``b_vec`` updated after each
        re-activation so the algorithm tracks the new inventory at
        machine precision.

    Parameters
    ----------
    T, p, species_names, formulas, mu_standard_funcs, psat_funcs,
    activity_model, vapor_eos, n_init, x1_seed, x2_seed,
    pure_liquid_volumes, phi_sat_funcs, p_ref, tol, maxiter,
    damping_init, n_floor, flash_tol, beta_V_seed, beta_L2_seed,
    verbose :
        As in ``gibbs_minimize_TP_VLL_split``.  Standard chemical
        potentials are gas-phase Gibbs of formation; solid-species
        potentials are typically ``μ°_solid(T)`` directly (e.g., 0
        for graphite or other reference-state solids).
    phase_per_species : sequence of str, length N, optional
        Per-species phase: ``'solid'`` for pure-solid phases,
        ``'gas'`` / ``'liquid'`` / ``'fluid'`` for species that
        participate in the inner 3-phase VLLE flash.  Default: all
        fluid (equivalent to ``gibbs_minimize_TP_VLL_split``).
    reactivation_seed, max_reactivations :
        Active-set re-activation parameters (same semantics as
        ``gibbs_minimize_TP_phase_split``).

    Returns
    -------
    GibbsMin4PhaseSplitResult

    Notes
    -----
    The Gibbs phase rule limits how many independent solids can
    coexist with V + L1 + L2 at fixed T, p.  In practice, attempting
    to nominate too many solids leads to a rank-deficient augmented
    matrix; the ``lstsq`` fallback handles this gracefully but the
    user should keep solid lists physically reasonable (typically
    1–3 solids per system).

    For systems with chemistry that produces or consumes water
    (hydrolysis, esterification, hydration), atom-balance preservation
    relies on consistent ``μ_standard_funcs`` between the fluid water
    species and any solid hydrates.  The algorithm does not enforce
    physical constraints beyond atom balance and second-law
    stationarity — it is the user's responsibility to provide
    thermochemistry that makes physical sense.
    """
    from ..activity.gamma_phi_eos_3phase import GammaPhiEOSThreePhaseFlash

    N = len(species_names)
    if len(formulas) != N:
        raise ValueError(f"formulas length {len(formulas)} != N={N}")
    if len(mu_standard_funcs) != N:
        raise ValueError(
            f"mu_standard_funcs length {len(mu_standard_funcs)} != N={N}")
    if len(psat_funcs) != N:
        raise ValueError(f"psat_funcs length {len(psat_funcs)} != N={N}")
    if len(n_init) != N:
        raise ValueError(f"n_init length {len(n_init)} != N={N}")
    if len(x1_seed) != N or len(x2_seed) != N:
        raise ValueError(f"x1_seed and x2_seed must have length N={N}")
    if pure_liquid_volumes is not None and len(pure_liquid_volumes) != N:
        raise ValueError(f"pure_liquid_volumes length != N={N}")
    if phi_sat_funcs is not None and len(phi_sat_funcs) != N:
        raise ValueError(f"phi_sat_funcs length != N={N}")

    # Resolve phase_per_species
    if phase_per_species is None:
        phase_list = ["fluid"] * N
    else:
        if len(phase_per_species) != N:
            raise ValueError(
                f"phase_per_species length {len(phase_per_species)} != N={N}")
        phase_list = list(phase_per_species)
    valid = {"gas", "liquid", "fluid", "solid"}
    for ph in phase_list:
        if ph not in valid:
            raise ValueError(
                f"phase_per_species entries must be in {valid}; got {ph!r}")
    is_solid = np.array([ph == "solid" for ph in phase_list], dtype=bool)
    is_fluid = ~is_solid
    fluid_idx = np.where(is_fluid)[0]
    solid_idx = np.where(is_solid)[0]
    F = int(is_fluid.sum())
    S = int(is_solid.sum())

    n_init_arr = np.asarray(n_init, dtype=float)
    if (n_init_arr <= 0).any():
        raise ValueError("all n_init values must be > 0")

    A_mat, elements = _build_atomic_matrix(formulas)
    E = len(elements)
    A_fluid = A_mat[:, fluid_idx]
    A_solid = A_mat[:, solid_idx] if S > 0 else np.zeros((E, 0))
    b_vec = A_mat @ n_init_arr

    mu0 = np.array([f(T) for f in mu_standard_funcs])

    # Inner flash uses fluid-only psat / pure_liquid_volumes / phi_sat
    psat_fluid = [psat_funcs[i] for i in fluid_idx]
    if pure_liquid_volumes is not None:
        pV_fluid = [pure_liquid_volumes[i] for i in fluid_idx]
    else:
        pV_fluid = None
    if phi_sat_funcs is not None:
        psat_phi_fluid = [phi_sat_funcs[i] for i in fluid_idx]
    else:
        psat_phi_fluid = None

    # Slice vapor_eos (CubicMixture) to fluid-only components.  Required
    # because the user-supplied vapor_eos covers all N species but the
    # inner 3-phase flash operates on only F species; calling a length-N
    # mixture with length-F compositions fails with shape mismatch.
    if F < N:
        try:
            from ..cubic.mixture import CubicMixture
            comps_fluid = [vapor_eos.components[i] for i in fluid_idx]
            kij_full = getattr(vapor_eos, "k_ij", None)
            if kij_full is not None:
                kij_full_arr = np.asarray(kij_full)
                kij_fluid = kij_full_arr[np.ix_(fluid_idx, fluid_idx)]
            else:
                kij_fluid = None
            vapor_eos_fluid = CubicMixture(comps_fluid, k_ij=kij_fluid)
        except Exception:
            vapor_eos_fluid = vapor_eos
    else:
        vapor_eos_fluid = vapor_eos

    # Wrap activity model for fluid-only composition
    user_activity_model = activity_model

    class _FluidActivityWrapper:
        def __init__(self, base):
            self.base = base
            self.N = F
        def gammas(self, T, x):
            x_arr = np.asarray(x)
            # First try: call base directly with length-F x
            try:
                g = np.asarray(self.base.gammas(T, x_arr))
                if g.shape[0] == F:
                    return g
            except Exception:
                pass
            # Fallback: pad x to length N, call, slice to F
            x_full = np.zeros(N)
            x_full[fluid_idx] = x_arr
            g_full = np.asarray(self.base.gammas(T, x_full))
            return g_full[fluid_idx]

    if F < N:
        flash_activity_model = _FluidActivityWrapper(user_activity_model)
    else:
        flash_activity_model = user_activity_model

    flash3 = GammaPhiEOSThreePhaseFlash(
        activity_model=flash_activity_model,
        psat_funcs=psat_fluid,
        vapor_eos=vapor_eos_fluid,
        pure_liquid_volumes=pV_fluid,
        phi_sat_funcs=psat_phi_fluid,
    )

    n = np.maximum(n_init_arr.copy(), n_floor)
    G_history = []
    reactivation_count = np.zeros(S, dtype=int) if S > 0 else np.array([])

    # Persistent inner-flash seeds (warm-start across iterations)
    x1_curr_F = np.asarray(x1_seed, dtype=float)[fluid_idx].copy() \
                if F > 0 else np.array([])
    x2_curr_F = np.asarray(x2_seed, dtype=float)[fluid_idx].copy() \
                if F > 0 else np.array([])
    if F > 0:
        s = x1_curr_F.sum()
        x1_curr_F = x1_curr_F / s if s > 0 else np.full(F, 1.0 / F)
        s = x2_curr_F.sum()
        x2_curr_F = x2_curr_F / s if s > 0 else np.full(F, 1.0 / F)
    bV_curr = float(beta_V_seed)
    bL2_curr = float(beta_L2_seed)

    def _flash_at(n_arr):
        """3-phase auto-dispatched flash on the FLUID species only.
        Returns full-N arrays (zeros at solid positions for x1, x2, y;
        γ=1 for solids; β are over the fluid total)."""
        nonlocal x1_curr_F, x2_curr_F, bV_curr, bL2_curr
        if F == 0:
            return (0.0, 0.0, 0.0,
                    np.zeros(N), np.zeros(N), np.zeros(N),
                    np.ones(N), np.ones(N),
                    np.ones(N), np.ones(N), np.ones(N))
        n_fluid = n_arr[fluid_idx]
        N_fluid = float(n_fluid.sum())
        z = n_fluid / N_fluid

        beta_V = 0.0; beta_L1 = 0.0; beta_L2 = 0.0
        x1_F = z.copy(); x2_F = z.copy(); y_F = z.copy()
        K_y_F = np.ones(F); K_x_F = np.ones(F)

        try:
            r_auto = flash3.auto_isothermal(T=T, p=p, z=z,
                                              tol=flash_tol, maxiter=300)
            r = r_auto.result
            ptype = r_auto.phase_type
            if ptype == "3VLL":
                beta_V = float(r.beta_V); beta_L2 = float(r.beta_L2)
                beta_L1 = 1.0 - beta_V - beta_L2
                x1_F = np.asarray(r.x1); x2_F = np.asarray(r.x2)
                y_F = np.asarray(r.y)
                K_y_F = np.asarray(r.K_y); K_x_F = np.asarray(r.K_x)
                x1_curr_F = x1_F; x2_curr_F = x2_F
                bV_curr = max(1e-6, min(0.999, beta_V))
                bL2_curr = max(1e-6, min(0.999, beta_L2))
            elif ptype == "2VL":
                beta_V = float(r.V); beta_L1 = 1.0 - beta_V; beta_L2 = 0.0
                x1_F = np.asarray(r.x); x2_F = np.asarray(r.x)
                y_F = np.asarray(r.y); K_y_F = np.asarray(r.K)
                K_x_F = np.ones(F)
                x1_curr_F = x1_F
                bV_curr = max(1e-6, min(0.999, beta_V))
            elif ptype == "2LL":
                beta_V = 0.0; beta_L2 = float(r.beta)
                beta_L1 = 1.0 - beta_L2
                x1_F = np.asarray(r.x1); x2_F = np.asarray(r.x2)
                y_F = z.copy()
                K_x_F = np.asarray(r.K); K_y_F = np.ones(F)
                x1_curr_F = x1_F; x2_curr_F = x2_F
                bL2_curr = max(1e-6, min(0.999, beta_L2))
            elif ptype == "1L":
                beta_V = 0.0; beta_L1 = 1.0; beta_L2 = 0.0
                x1_F = z.copy(); x2_F = z.copy(); y_F = z.copy()
            elif ptype == "1V":
                beta_V = 1.0; beta_L1 = 0.0; beta_L2 = 0.0
                y_F = z.copy(); x1_F = z.copy(); x2_F = z.copy()
        except Exception:
            beta_V = 1.0; beta_L1 = 0.0; beta_L2 = 0.0
            y_F = z.copy()

        # Compute gammas
        try:
            g1_F = np.asarray(flash_activity_model.gammas(T, x1_F))
        except Exception:
            g1_F = np.ones(F)
        try:
            g2_F = np.asarray(flash_activity_model.gammas(T, x2_F))
        except Exception:
            g2_F = np.ones(F)

        # All-liquid: define fictitious y from modified Raoult on dominant L
        if beta_V <= 0:
            psats = np.array([f(T) for f in psat_fluid])
            y_F = g1_F * x1_F * psats / p
            y_F = np.maximum(y_F, 1e-300)

        # phi_v from EOS (only if vapor exists)
        if beta_V > 0:
            try:
                rho_v = vapor_eos_fluid.density_from_pressure(
                    p, T, y_F, phase_hint="vapor")
                ln_phi = vapor_eos_fluid.ln_phi(rho_v, T, y_F)
                phi_v_F = np.exp(ln_phi)
            except Exception:
                phi_v_F = np.ones(F)
        else:
            phi_v_F = np.ones(F)

        # Pad to length N (zeros/ones at solid positions)
        x1 = np.zeros(N); x1[fluid_idx] = x1_F
        x2 = np.zeros(N); x2[fluid_idx] = x2_F
        y_full = np.zeros(N); y_full[fluid_idx] = y_F
        K_y = np.zeros(N); K_y[fluid_idx] = K_y_F
        K_x = np.zeros(N); K_x[fluid_idx] = K_x_F
        g1 = np.ones(N); g1[fluid_idx] = g1_F
        g2 = np.ones(N); g2[fluid_idx] = g2_F
        phi_v = np.ones(N); phi_v[fluid_idx] = phi_v_F

        return (beta_V, beta_L1, beta_L2, x1, x2, y_full,
                K_y, K_x, g1, g2, phi_v)

    def _compute_mu_full(y_arr, phi_v_arr):
        """μ for ALL species.  Fluid: gas-phase formula with vapor
        non-ideality.  Solid: μ°(T) (activity 1)."""
        mu = mu0.copy()
        if F > 0:
            y_safe = np.maximum(y_arr[fluid_idx], 1e-300)
            phi_safe = np.maximum(phi_v_arr[fluid_idx], 1e-300)
            mu[fluid_idx] = (mu0[fluid_idx]
                              + _R_GAS * T * (np.log(y_safe)
                                               + np.log(phi_safe)
                                               + np.log(p / p_ref)))
        return mu

    converged = False
    iters_done = 0
    pi_last = np.zeros(E)
    msg = "did not start"
    theta_max = float("inf")
    bV_last = 0.0; bL1_last = 0.0; bL2_last = 0.0
    x1_full = np.zeros(N); x2_full = np.zeros(N); y_full = np.zeros(N)
    K_y_full = np.zeros(N); K_x_full = np.zeros(N)
    g1_full = np.ones(N); g2_full = np.ones(N); phi_v_full = np.ones(N)

    for it in range(maxiter):
        (bV_last, bL1_last, bL2_last, x1_full, x2_full, y_full,
         K_y_full, K_x_full, g1_full, g2_full, phi_v_full) = _flash_at(n)

        mu = _compute_mu_full(y_full, phi_v_full)
        G_curr = float(np.sum(n * mu))
        G_history.append(G_curr)
        muRT = mu / (_R_GAS * T)

        n_safe = np.maximum(n, n_floor)
        n_fluid_safe = n_safe[fluid_idx] if F > 0 else np.array([])

        # Active-set on solids (parallel to v0.9.82/v0.9.83 logic)
        n_solid_curr = n_safe[solid_idx] if S > 0 else np.array([])
        active_solid_local = (n_solid_curr > 10.0 * n_floor) \
                              if S > 0 else np.array([], dtype=bool)
        S_active = int(active_solid_local.sum())
        active_solid_idx = solid_idx[active_solid_local] \
                            if S > 0 else np.array([], dtype=int)
        A_solid_active = A_solid[:, active_solid_local] \
                          if S > 0 else np.zeros((E, 0))

        # Build B and constants from FLUID species only
        if F > 0:
            B = (A_fluid * n_fluid_safe[np.newaxis, :]) @ A_fluid.T
            b_curr = A_fluid @ n_fluid_safe
            c_vec = A_fluid @ (n_fluid_safe * muRT[fluid_idx])
            c_tot = float(np.sum(n_fluid_safe * muRT[fluid_idx]))
        else:
            B = np.zeros((E, E))
            b_curr = np.zeros(E)
            c_vec = np.zeros(E)
            c_tot = 0.0

        # Augmented (E + 1 + S_active) x (E + 1 + S_active) RAND
        if S_active == 0:
            M = np.zeros((E + 1, E + 1))
            M[:E, :E] = B
            M[:E, E] = b_curr
            M[E, :E] = b_curr
            rhs = np.concatenate([c_vec, [c_tot]])
        else:
            sz = E + 1 + S_active
            M = np.zeros((sz, sz))
            M[:E, :E] = B
            M[:E, E] = b_curr
            M[E, :E] = b_curr
            M[:E, E + 1:] = A_solid_active
            M[E + 1:, :E] = A_solid_active.T
            mu_solid_RT_active = muRT[active_solid_idx]
            rhs = np.concatenate([c_vec, [c_tot], mu_solid_RT_active])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        pi = sol[:E]
        u_var = sol[E]
        delta_n_solid_active = sol[E + 1:] if S_active > 0 else np.array([])
        pi_last = pi

        theta_full = muRT - A_mat.T @ pi
        delta_n = np.zeros(N)
        if F > 0:
            delta_n[fluid_idx] = n_fluid_safe * (-theta_full[fluid_idx]
                                                  + u_var)
        if S > 0:
            for k_loc, i in enumerate(solid_idx):
                if active_solid_local[k_loc]:
                    j_active = int(active_solid_local[:k_loc + 1].sum() - 1)
                    delta_n[i] = delta_n_solid_active[j_active]

        # Convergence test
        active_mask = np.ones(N, dtype=bool)
        if S > 0:
            active_mask[solid_idx[~active_solid_local]] = False
        theta_active = theta_full[active_mask]
        theta_max = (float(np.max(np.abs(theta_active)))
                     if theta_active.size else 0.0)
        ss_violation = 0.0
        if S > 0 and (~active_solid_local).any():
            inactive_idx = solid_idx[~active_solid_local]
            ss_terms = -theta_full[inactive_idx]
            ss_violation = float(max(0.0, np.max(ss_terms))
                                  if ss_terms.size else 0.0)
        if verbose:
            print(f"  iter {it:3d}: G = {G_curr:.4e}, "
                  f"max|θ| = {theta_max:.2e}, ss_viol={ss_violation:.2e}, "
                  f"βV={bV_last:.3f}, βL2={bL2_last:.3f}, S_act={S_active}/{S}")
        if theta_max < tol and ss_violation < tol:
            converged = True
            iters_done = it + 1
            msg = (f"converged in {iters_done} iters, max|θ|={theta_max:.2e}, "
                   f"ss={ss_violation:.2e}")
            break

        # Damping + line search
        alpha = damping_init
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(delta_n < 0, -n / delta_n, np.inf)
            alpha_max = float(np.min(ratios))
        alpha = min(alpha, 0.95 * alpha_max if alpha_max > 0 else 0.5)
        if alpha <= 0:
            alpha = 1e-6

        n_trial = np.maximum(n + alpha * delta_n, n_floor)
        try:
            (_, _, _, _, _, y_tr, _, _, _, _, phi_tr) = _flash_at(n_trial)
            G_trial = float(np.sum(n_trial *
                                    _compute_mu_full(y_tr, phi_tr)))
        except Exception:
            G_trial = G_curr + 1.0
        bt_count = 0
        while G_trial > G_curr + 1e-9 * abs(G_curr) and bt_count < 30:
            alpha *= 0.5
            n_trial = np.maximum(n + alpha * delta_n, n_floor)
            try:
                (_, _, _, _, _, y_tr, _, _, _, _, phi_tr) = _flash_at(n_trial)
                G_trial = float(np.sum(n_trial *
                                        _compute_mu_full(y_tr, phi_tr)))
            except Exception:
                G_trial = G_curr + 1.0
            bt_count += 1
        n = n_trial
        iters_done = it + 1

        # Active-set re-activation (v0.9.83 logic)
        if S > 0:
            reactivated_any = False
            for k_loc, i in enumerate(solid_idx):
                if not active_solid_local[k_loc]:
                    if -theta_full[i] > tol \
                            and reactivation_count[k_loc] < max_reactivations:
                        n[i] = max(n[i], reactivation_seed)
                        reactivation_count[k_loc] += 1
                        reactivated_any = True
                        if verbose:
                            print(f"    reactivating '{species_names[i]}'"
                                  f" (count={reactivation_count[k_loc]})")
            if reactivated_any:
                b_vec = A_mat @ n
    else:
        msg = f"did not converge in {maxiter} iters, max|θ|={theta_max:.2e}"

    # Final flash + result
    (bV_last, bL1_last, bL2_last, x1_full, x2_full, y_full,
     K_y_full, K_x_full, g1_full, g2_full, phi_v_full) = _flash_at(n)
    mu_final = _compute_mu_full(y_full, phi_v_full)
    G_final = float(np.sum(n * mu_final))
    G_history.append(G_final)
    atom_balance = float(np.linalg.norm(A_mat @ n - b_vec))

    return GibbsMin4PhaseSplitResult(
        converged=converged,
        T=T, p=p,
        n=n,
        n_init=n_init_arr,
        species_names=tuple(species_names),
        elements=elements,
        G_total=G_final,
        iterations=iters_done,
        pi=pi_last,
        G_history=tuple(G_history),
        atom_balance_residual=atom_balance,
        message=msg,
        beta_V=float(bV_last), beta_L1=float(bL1_last), beta_L2=float(bL2_last),
        x1=x1_full, x2=x2_full, y_vapor=y_full,
        K_y=K_y_full, K_x=K_x_full,
        gammas1=g1_full, gammas2=g2_full,
    )
