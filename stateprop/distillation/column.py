"""Non-reactive multi-stage distillation column.

Wraps :func:`stateprop.reaction.reactive_distillation_column` with
``reactions=()`` to provide a dedicated API for the standard separation
case.  The underlying solvers (Wang-Henke fixed-point and
Naphtali-Sandholm simultaneous Newton, with or without an energy
balance) are unchanged.

Stage indexing convention (matches the reactive solver):
  - Stage 1 = top stage (just below a total condenser)
  - Stage `n_stages` = bottom stage (partial reboiler)
  - Liquid streams flow downward, vapor upward
  - The total condenser sits above stage 1; vapor V_1 leaves stage 1,
    is fully condensed, split into reflux L_0 = R*D returning to stage
    1 and distillate D leaving the column.  In a total-condenser
    column ``x_D = y_1``.

Multi-feed and side-draw extensions (v0.9.71+):
  - Pass ``feeds=[FeedSpec(stage=..., F=..., z=..., T=...), ...]`` to
    introduce more than one feed; or use the single-feed scalars
    ``feed_stage/feed_F/feed_z/feed_T`` for the common case.
  - Liquid side draws via ``liquid_draws={stage: flow, ...}``;
    vapor side draws via ``vapor_draws={stage: flow, ...}``.  Stage
    indices are 1-based and refer to the same stage numbering as
    ``feed_stage``.  Multi-feed and side draws require the
    Naphtali-Sandholm solver (``method="naphtali_sandholm"``, default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np

from ..reaction.reactive_column import (
    reactive_distillation_column,
    ColumnResult as _ReactiveColumnResult,
    FeedSpec,
    PumpAround,
    Spec,
)


# =========================================================================
# Result class
# =========================================================================

@dataclass
class DistillationColumnResult:
    """Steady-state non-reactive distillation column profile.

    Mirrors the reactive ``ColumnResult`` but drops the chemistry
    fields (``xi``, ``reactive_stages``) and adds a ``recovery``
    convenience method appropriate for non-reactive separations.
    """
    converged: bool
    iterations: int
    n_stages: int
    species_names: tuple
    T: np.ndarray                # (N,) stage temperatures [K]
    p: np.ndarray                # per-stage pressure [Pa], shape (n_stages,)
    L: np.ndarray                # (N,) liquid flow rates leaving each stage
    V: np.ndarray                # (N,) vapor flow rates leaving each stage
    x: np.ndarray                # (N, C) liquid mole fractions
    y: np.ndarray                # (N, C) vapor mole fractions
    D: float                     # distillate flow rate
    B: float                     # bottoms flow rate
    feed_stage: int              # 1-indexed; first feed for multi-feed cases
    feed_F: float                # first feed's F for multi-feed cases
    feed_z: np.ndarray           # first feed's z for multi-feed cases
    reflux_ratio: float
    message: str = ""
    # Multi-feed and side draws (v0.9.71+)
    feeds: tuple = ()                    # tuple of FeedSpec for ALL feeds
    liquid_draws: tuple = ()             # tuple of (stage_1idx, flow) pairs
    vapor_draws: tuple = ()              # tuple of (stage_1idx, flow) pairs
    # v0.9.72+: partial condenser
    condenser: str = "total"             # "total" or "partial"
    # v0.9.88+: side strippers attached to the main column
    side_strippers: tuple = ()           # tuple of dicts with SS results

    @property
    def x_D(self) -> np.ndarray:
        """Distillate composition.  Total condenser => x_D = y_1."""
        return self.y[0].copy()

    @property
    def x_B(self) -> np.ndarray:
        """Bottoms composition = x_N."""
        return self.x[-1].copy()

    def recovery(self, species: str, to: str = "distillate") -> float:
        """Fraction of `species` in the (combined) feed that leaves in
        the named outlet.  For non-reactive columns the recoveries to
        all outlets (distillate, bottoms, every side draw) sum to 1
        within numerical roundoff.

        Parameters
        ----------
        species : str
        to : str
            One of ``"distillate"``, ``"bottoms"``, ``"liquid_draw:K"``
            (K is the 1-indexed stage of the draw), ``"vapor_draw:K"``.
        """
        idx = list(self.species_names).index(species)
        # Total moles of species i across all feeds
        if self.feeds:
            n_in = sum(f.F * np.asarray(f.z)[idx] for f in self.feeds)
        else:
            n_in = self.feed_F * self.feed_z[idx]
        if n_in <= 0:
            return float("nan")

        if to == "distillate":
            return float(self.D * self.x_D[idx] / n_in)
        if to == "bottoms":
            return float(self.B * self.x_B[idx] / n_in)
        if to.startswith("liquid_draw:"):
            stage = int(to.split(":")[1])
            for s, flow in self.liquid_draws:
                if int(s) == stage:
                    j = stage - 1
                    return float(flow * self.x[j, idx] / n_in)
            raise ValueError(f"no liquid draw at stage {stage}")
        if to.startswith("vapor_draw:"):
            stage = int(to.split(":")[1])
            for s, flow in self.vapor_draws:
                if int(s) == stage:
                    j = stage - 1
                    return float(flow * self.y[j, idx] / n_in)
            raise ValueError(f"no vapor draw at stage {stage}")
        raise ValueError(
            "`to` must be 'distillate', 'bottoms', 'liquid_draw:<stage>', "
            "or 'vapor_draw:<stage>'")


def _strip_reactive_fields(rr: _ReactiveColumnResult) -> DistillationColumnResult:
    """Convert a reactive ColumnResult to the cleaner non-reactive
    DistillationColumnResult.  Validates that the reactive result is
    actually non-reactive (xi all zeros, no reactive stages)."""
    if rr.reactive_stages:
        raise RuntimeError(
            "Internal error: reactive_distillation_column returned "
            "non-empty reactive_stages from a non-reactive call.")
    if rr.xi.size and not np.allclose(rr.xi, 0.0):
        raise RuntimeError(
            "Internal error: non-zero extents reported for a "
            "non-reactive column.")
    return DistillationColumnResult(
        converged=rr.converged,
        iterations=rr.iterations,
        n_stages=rr.n_stages,
        species_names=rr.species_names,
        T=rr.T, p=rr.p, L=rr.L, V=rr.V, x=rr.x, y=rr.y,
        D=rr.D, B=rr.B,
        feed_stage=rr.feed_stage, feed_F=rr.feed_F, feed_z=rr.feed_z,
        reflux_ratio=rr.reflux_ratio,
        message=rr.message.replace("reactive ", ""),
        feeds=rr.feeds,
        liquid_draws=rr.liquid_draws,
        vapor_draws=rr.vapor_draws,
        condenser=rr.condenser,
        side_strippers=rr.side_strippers,
    )


# =========================================================================
# Public API
# =========================================================================

def distillation_column(
    n_stages: int,
    feed_stage: Optional[int] = None,
    feed_F: Optional[float] = None,
    feed_z: Optional[Sequence[float]] = None,
    feed_T: Optional[float] = None,
    reflux_ratio: float = None,
    distillate_rate: float = None,
    pressure: float = None,
    species_names: Sequence[str] = None,
    activity_model = None,
    psat_funcs: Sequence[Callable[[float], float]] = None,
    *,
    feeds: Optional[Sequence] = None,
    liquid_draws: Optional[dict] = None,
    vapor_draws: Optional[dict] = None,
    feed_q: Optional[float] = None,
    condenser: str = "total",
    pressure_drop: Optional[float] = None,
    stage_efficiency = None,
    pump_arounds: Optional[Sequence] = None,
    side_strippers: Optional[Sequence] = None,
    vapor_eos = None,
    pure_liquid_volumes: Optional[Sequence[float]] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    specs: Optional[Sequence] = None,
    initial_distillate_rate: Optional[float] = None,
    initial_reflux_ratio: Optional[float] = None,
    spec_outer_max_iter: int = 30,
    spec_outer_tol: float = 1e-6,
    T_init: Optional[Sequence[float]] = None,
    x_init: Optional[np.ndarray] = None,
    max_outer_iter: int = 100,
    tol: float = 1e-4,
    damping: float = 0.5,
    verbose: bool = False,
    method: str = "naphtali_sandholm",
    max_newton_iter: int = 30,
    newton_tol: float = 1e-7,
    fd_step: float = 1e-7,
    energy_balance: bool = False,
    h_V_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    h_L_funcs: Optional[Sequence[Callable[[float], float]]] = None,
) -> DistillationColumnResult:
    """Solve a non-reactive multi-stage distillation column at steady state.

    The column has a total condenser above stage 1 and a partial
    reboiler at stage `n_stages`.  A single feed of total flow
    ``feed_F`` and composition ``feed_z`` at temperature ``feed_T``
    enters at ``feed_stage``.  External specifications: reflux ratio
    ``R = L_0/D`` and distillate rate ``D``; the bottoms flow follows
    from the overall mass balance ``B = F - D``.

    Parameters
    ----------
    n_stages : int
        Total equilibrium stages (>= 2).  Stage 1 is the top stage
        (just below a total condenser); stage ``n_stages`` is the
        partial reboiler.
    feed_stage : int
        1-indexed stage where the single feed enters.  Must lie in
        ``[1, n_stages]`` and is most commonly an interior stage.
    feed_F : float
        Total molar feed rate.
    feed_z : sequence of float, length C
        Feed mole fractions.  Must sum to 1.
    feed_T : float
        Feed temperature [K].  Used as a starting point for the T
        profile; with ``energy_balance=True`` it also enters the inlet
        enthalpy.
    reflux_ratio : float
        ``R = L_0 / D`` where L_0 is the reflux returning from the
        total condenser to stage 1.  Must be positive.
    distillate_rate : float
        Distillate molar flow rate ``D``.  Must satisfy
        ``0 < D < feed_F`` so ``B = F - D > 0``.
    pressure : float
        Uniform column pressure [Pa].
    species_names : sequence of str, length C
        Canonical species ordering used by the activity model.
    activity_model : object with ``.gammas(T, x)``
        Liquid activity-coefficient model (e.g. UNIFAC, NRTL, UNIQUAC,
        or :class:`IdealActivity`).
    psat_funcs : sequence of C callables ``T -> p_sat(T)`` [Pa]
        Pure-component vapor pressures.
    T_init, x_init : optional initial profiles
        See :func:`reactive_distillation_column` for details.
    method : {"naphtali_sandholm", "wang_henke"}
        Default ``"naphtali_sandholm"`` solves a simultaneous Newton
        system on (x, T) per stage; ``"wang_henke"`` is the legacy
        fixed-point bubble-point method (slower but historically
        common).  Naphtali-Sandholm is required for energy-balance.
    energy_balance : bool
        If True, drop the constant-molar-overflow assumption and solve
        per-stage energy balances.  Requires ``h_V_funcs`` and
        ``h_L_funcs`` and ``method="naphtali_sandholm"``.
    h_V_funcs, h_L_funcs : sequences of C callables, optional
        Pure-component vapor and liquid molar enthalpies as functions
        of T [K], required when ``energy_balance=True``.

    Returns
    -------
    DistillationColumnResult
        Profile of stage temperatures, flows, and compositions.
        ``recovery(species)`` gives the fraction of feed component i
        that leaves in the distillate (or bottoms).

    Notes
    -----
    Mass balance closure is verified post-solve; with the simultaneous
    Newton method the per-species mass balance closes to better than
    1e-10 for well-conditioned cases.

    Examples
    --------
    Benzene-toluene at 1 atm with NRTL:

    >>> from stateprop.distillation import distillation_column
    >>> # ... (set up activity model, psat_funcs)
    >>> res = distillation_column(
    ...     n_stages=12, feed_stage=6,
    ...     feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
    ...     reflux_ratio=2.0, distillate_rate=50.0,
    ...     pressure=101325.0,
    ...     species_names=["benzene", "toluene"],
    ...     activity_model=nrtl, psat_funcs=psats,
    ... )
    >>> res.recovery("benzene", to="distillate")    # > 0.99 typically
    """
    if n_stages < 2:
        raise ValueError("n_stages must be at least 2 (one stage + reboiler)")
    in_design_mode = specs is not None and len(specs) > 0
    if not in_design_mode:
        if reflux_ratio is None or reflux_ratio <= 0:
            raise ValueError("reflux_ratio must be positive")
        if distillate_rate is None or distillate_rate <= 0:
            raise ValueError("distillate_rate must be positive")
    if condenser not in ("total", "partial"):
        raise ValueError(
            f"condenser must be 'total' or 'partial', got {condenser!r}")

    # Compute total feed flow (single-feed scalar OR multi-feed list)
    if feeds is not None:
        if (feed_stage is not None or feed_F is not None
                or feed_z is not None):
            raise ValueError(
                "`feeds` cannot be combined with single-feed scalars "
                "(feed_stage, feed_F, feed_z, feed_T); pick one form.")
        total_F = float(sum(_get_F(f) for f in feeds))
    else:
        if feed_F is None or feed_F <= 0:
            raise ValueError("feed_F must be positive")
        total_F = float(feed_F)

    # Distillate must be strictly less than total feed minus all draws
    # (only validate in classic mode; design-mode varies D and R)
    total_U = sum(float(v) for v in (liquid_draws or {}).values())
    total_W = sum(float(v) for v in (vapor_draws or {}).values())
    max_D = total_F - total_U - total_W
    if not in_design_mode:
        if not (0 < distillate_rate <= max_D + 1e-10):
            raise ValueError(
                f"distillate_rate must lie in (0, total_F - draws] = "
                f"(0, {max_D:.6g}]; got distillate_rate={distillate_rate}.")

    rr = reactive_distillation_column(
        n_stages=n_stages,
        feed_stage=feed_stage,
        feed_F=feed_F,
        feed_z=feed_z,
        feed_T=feed_T,
        reflux_ratio=reflux_ratio,
        distillate_rate=distillate_rate,
        pressure=pressure,
        species_names=species_names,
        activity_model=activity_model,
        psat_funcs=psat_funcs,
        reactions=(),
        reactive_stages=(),
        feeds=feeds,
        liquid_draws=liquid_draws,
        vapor_draws=vapor_draws,
        feed_q=feed_q,
        condenser=condenser,
        pressure_drop=pressure_drop,
        stage_efficiency=stage_efficiency,
        pump_arounds=pump_arounds,
        side_strippers=side_strippers,
        vapor_eos=vapor_eos,
        pure_liquid_volumes=pure_liquid_volumes,
        phi_sat_funcs=phi_sat_funcs,
        specs=specs,
        initial_distillate_rate=initial_distillate_rate,
        initial_reflux_ratio=initial_reflux_ratio,
        spec_outer_max_iter=spec_outer_max_iter,
        spec_outer_tol=spec_outer_tol,
        T_init=T_init,
        x_init=x_init,
        max_outer_iter=max_outer_iter,
        tol=tol,
        damping=damping,
        verbose=verbose,
        method=method,
        max_newton_iter=max_newton_iter,
        newton_tol=newton_tol,
        fd_step=fd_step,
        energy_balance=energy_balance,
        h_V_funcs=h_V_funcs,
        h_L_funcs=h_L_funcs,
    )
    return _strip_reactive_fields(rr)


def _get_F(feed) -> float:
    """Extract F flow from a feed entry (FeedSpec, dict, or tuple)."""
    if hasattr(feed, "F"):
        return float(feed.F)
    if isinstance(feed, dict):
        return float(feed["F"])
    return float(feed[1])
