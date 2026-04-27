"""Multi-stage reactive distillation column at steady state.

Solves an equilibrium-stage column where chemical reactions occur in
the liquid phase on a designated subset of stages ("reactive stages").
The remaining stages are pure-VLE (rectifying / stripping zones).

Stage numbering convention
--------------------------
        ----+-----+----        condenser (total) -> distillate D
            |     |
       +----+-----+----+        stage 1   (top of column)
       |               |
       +----+-----+----+        stage 2
       |               |
            ...                 ...
       |               |
       +----+-----+----+        stage f   (feed enters here)
       |               |
            ...                 ...
       |               |
       +----+-----+----+        stage N-1
       |               |
       +----+-----+----+        stage N   (partial reboiler) -> bottoms B

Stage 1 sits just below the total condenser. The reflux entering stage 1
has composition x_0 = y_1 (total condenser preserves composition).
Stage N is the partial reboiler; its liquid effluent IS the bottoms,
its vapor effluent V_N rises to stage N-1.

Algorithm
---------
Wang-Henke bubble-point method (Henley & Seader, Sec 10.4):

  1. Initialize T_j on each stage (linear between feed bubble-T and
     a higher reboiler T).
  2. Compute K_{j,i} = gamma_{j,i} p_i^sat(T_j) / p at current (T_j, x_j).
  3. For each component i, solve the tridiagonal mass-balance system
     across all stages (the reaction source term is included as RHS).
  4. Normalize x so Sum_i x_{j,i} = 1.
  5. On each REACTIVE stage j, update xi_j to satisfy
     K_eq,r(T_j) = Prod_i (gamma_{j,i} x_{j,i})^nu[r,i].
     For R=1 use bisection (bulletproof); for R>1, damped Newton.
  6. Update T_j by bubble-point: find T_j such that
     Sum_i K_{j,i}(T_j) x_{j,i} = 1.
  7. Update flows V_j and L_j accounting for reaction-induced mole
     changes (constant molar overflow with reaction adjustment).
  8. Iterate until ||T_new - T_old||_inf < tol and ||x_new - x_old||_inf < tol.

Limitations
-----------
- Constant pressure across stages (no Delta_p modeling).
- No energy balance: T_j set by bubble point at each stage, not H-balance.
  This is reasonable for mildly exo/endothermic reactions and adiabatic
  columns with similar feed/distillate enthalpies. For strongly
  exothermic reactions, externally couple to an energy balance.
- Constant molar overflow is assumed in each section, with reaction
  mole-change correction. For wide-boiling mixtures, an energy balance
  would be more accurate.
- Modified Raoult VLE (gamma for liquid, ideal vapor). Adequate up to
  ~10-30 bar; for higher p, swap inner K-evaluation with gamma-phi-EOS.

References
----------
Henley, E. J., Seader, J. D., Roper, D. K. (2011). Separation Process
Principles, 3rd ed., Wiley. Chapter 10.

Doherty, M. F. and Malone, M. F. (2001). Conceptual Design of
Distillation Systems, McGraw-Hill. Chapter 10.

Taylor, R. and Krishna, R. (2000). "Modelling reactive distillation",
Chemical Engineering Science 55(22): 5183-5229.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, Callable, List, Tuple, Iterable
import math
import numpy as np

from .liquid_phase import LiquidPhaseReaction


@dataclass
class ColumnResult:
    """Steady-state reactive-distillation column profile."""
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
    xi: np.ndarray               # (N, R) extents per stage; non-reactive stages = 0
    D: float                     # distillate flow [mol/s or mol]
    B: float                     # bottoms flow
    feed_stage: int              # 1-indexed; first feed's stage when multi-feed
    feed_F: float                # first feed's flow when multi-feed
    feed_z: np.ndarray           # first feed's composition when multi-feed
    reflux_ratio: float
    reactive_stages: tuple
    message: str = ""
    # Multi-feed / side-draw extensions (v0.9.71+)
    feeds: tuple = ()                    # tuple of FeedSpec for ALL feeds
    liquid_draws: tuple = ()             # tuple of (stage_1idx, flow) pairs
    vapor_draws: tuple = ()              # tuple of (stage_1idx, flow) pairs
    # v0.9.72+: partial condenser support
    condenser: str = "total"             # "total" or "partial"
    # v0.9.88: side-stripper results.  Each entry is a dict with keys:
    #   draw_stage, return_stage, n_stages, flow, bottoms_rate, pressure,
    #   T (n_ss,), x (n_ss, C), y (n_ss, C), x_bottoms (C,)
    side_strippers: tuple = ()

    @property
    def x_D(self) -> np.ndarray:
        """Distillate composition. Total condenser: x_D = y_1."""
        return self.y[0].copy()

    @property
    def x_B(self) -> np.ndarray:
        """Bottoms composition = x_N."""
        return self.x[-1].copy()

    def conversion(self, species: str) -> float:
        """Overall conversion of `species` from feed to (V + L) outlets.

        Aggregates across all feeds and all column outlets (D, B, and
        every side draw).
        """
        idx = list(self.species_names).index(species)
        # Total moles in (across all feeds)
        if self.feeds:
            n_in = sum(f.F * np.asarray(f.z)[idx] for f in self.feeds)
        else:
            n_in = self.feed_F * self.feed_z[idx]
        # Total moles out (D, B, plus side draws)
        n_out = self.D * self.x_D[idx] + self.B * self.x_B[idx]
        for stage_1idx, flow in self.liquid_draws:
            j = stage_1idx - 1
            n_out += float(flow) * float(self.x[j, idx])
        for stage_1idx, flow in self.vapor_draws:
            j = stage_1idx - 1
            n_out += float(flow) * float(self.y[j, idx])
        if n_in <= 0:
            return float('nan')
        return float(1.0 - n_out / n_in)


@dataclass
class FeedSpec:
    """Specification of a single feed entering a column at a specified stage.

    Used by ``reactive_distillation_column`` and ``distillation_column``
    when a column has more than one feed.  ``stage`` is 1-indexed.

    Parameters
    ----------
    stage : int
        1-indexed column stage at which the feed enters.
    F : float
        Total molar feed rate.
    z : sequence of float
        Mole-fraction composition (length C).
    T : float, optional
        Feed temperature [K].  Used only by the energy-balance solver
        (``energy_balance=True``); the CMO solver ignores it.  Default
        298.15 K.
    q : float, optional
        Liquid fraction of the feed.  Default 1.0 (saturated liquid).

        - q = 1: saturated liquid feed (all moles join the liquid stream)
        - q = 0: saturated vapor feed (all moles join the vapor stream)
        - 0 < q < 1: two-phase feed; q is the liquid mole fraction
        - q > 1: subcooled liquid (cools the column locally; under CMO
          this manifests as q*F joining the liquid and (1-q)*F joining
          the vapor where (1-q)*F < 0, i.e. some vapor condenses to
          provide heat to bring the cold feed up to bubble point)
        - q < 0: superheated vapor (analogous; some liquid vaporizes)

        The energy-balance solver uses ``q`` to compute the feed
        enthalpy as ``q*h_L(T) + (1-q)*h_V(T)``.
    """
    stage: int
    F: float
    z: Sequence[float]
    T: float = 298.15
    q: float = 1.0


@dataclass
class PumpAround:
    """Specification of a single pump-around (internal liquid recycle).

    A pump-around draws ``flow`` mol/h of liquid from ``draw_stage``
    (1-indexed), optionally cools it through a temperature drop ``dT``,
    and returns it at ``return_stage`` (1-indexed) above the draw
    stage.  This is a common refinery topology for removing heat from
    a column without removing mass, controlling vapor traffic in the
    upper section of a crude tower.

    Mass conservation is automatic (the same flow that leaves at
    ``draw_stage`` enters at ``return_stage``), but the recycled
    composition is the LIQUID composition at the draw stage, which
    introduces a non-tridiagonal coupling in the Jacobian.  The dense
    Newton solve in this library handles the coupling without code
    changes.

    Under CMO (no energy balance), ``dT`` is ignored -- the pump-around
    only modifies the liquid flow profile between return_stage and
    draw_stage.  Under the energy balance solver, ``dT`` enters the
    H balance: the pump-around removes
    ``Q_PA = flow * sum_i x_draw_i * (h_L_i(T_draw) - h_L_i(T_draw - dT))``
    of heat from the column, returned to ``return_stage`` at
    ``T_draw - dT``.

    Parameters
    ----------
    draw_stage : int
        1-indexed stage where liquid is drawn off.
    return_stage : int
        1-indexed stage where liquid returns; must be strictly less
        than ``draw_stage`` (the loop pumps liquid UP).
    flow : float
        Liquid flow rate of the pump-around stream, mol/h.
    dT : float, optional
        Temperature drop across the pump-around cooler, in K
        (default 0.0, i.e. an isothermal pump-around).  Only used by
        the energy-balance solver.
    """
    draw_stage: int
    return_stage: int
    flow: float
    dT: float = 0.0


@dataclass
class SideStripper:
    """Specification of a side stripper attached to the main column.

    A side stripper draws liquid from ``draw_stage`` (1-indexed) of the
    main column at ``flow`` mol/h, sends it to the top of an auxiliary
    column with ``n_stages`` equilibrium stages, and returns the
    overhead vapor of the side stripper to the main column at
    ``return_stage``.  The side product (the bottoms of the side
    stripper) leaves at ``bottoms_rate`` mol/h.

    Stripping mode (v0.9.89+)
    -------------------------
    Two stripping modes are supported, set via ``stripping_mode``:

    * ``"reboil"`` (default, v0.9.88 behavior) — implicit partial
      reboiler at the SS bottom; under CMO the reboil rate is fixed by
      mass balance: ``V_SS_top = flow - bottoms_rate``.  No explicit
      heat duty; no steam stream.

    * ``"steam"`` — live steam (or another inert stripping medium) is
      injected into the bottom stage of the SS at ``steam_flow`` mol/h,
      composition ``steam_z`` (length C, must sum to 1), and temperature
      ``steam_T``.  The bottom-stage mass balance becomes::

          flow + steam_flow == bottoms_rate + V_SS_top
          ⇒ V_SS_top = flow + steam_flow - bottoms_rate

      Steam injection is the dominant industrial mode for refinery
      side strippers because it avoids a fired reboiler and leverages
      cheap utility steam.

    Stage indexing for the SS internally:
       k = 0  is the top of the SS (receives liquid feed from the main
              column draw and sends vapor back to the main column return).
       k = n_stages - 1  is the bottom of the SS (the side product
              leaves here at composition x_SS[n_stages-1]).

    Mass balance overall:

      reboil mode:   flow             == V_SS_top + bottoms_rate
      steam mode:    flow + steam_flow == V_SS_top + bottoms_rate

    Limitations (v0.9.89):
      * No chemistry on SS stages.
      * No Murphree efficiency on SS stages (E = 1 throughout).
      * Single uniform pressure on the SS.
      * Steam composition is constant (no condensation modeled
        beyond the equilibrium flash on the bottom stage).

    Parameters
    ----------
    draw_stage : int
        1-indexed main-column stage from which liquid is drawn.
    return_stage : int
        1-indexed main-column stage to which the SS overhead vapor
        returns.
    n_stages : int
        Number of equilibrium stages in the side stripper itself
        (typically 3-6 in industrial designs).
    flow : float
        Liquid draw rate from the main column to SS top, mol/h.
    bottoms_rate : float
        Side-product flow at the SS bottom, mol/h.
    pressure : float
        Operating pressure of the side stripper, Pa (uniform).
    stripping_mode : str
        ``"reboil"`` (default) or ``"steam"``.
    steam_flow : float
        Steam injection rate, mol/h.  Only used when
        ``stripping_mode == "steam"``.
    steam_z : sequence of float or None
        Steam composition (length C, sums to 1).  Required when
        ``stripping_mode == "steam"``; typically ``[0, ..., 1, ..., 0]``
        with the 1 at the water/steam species index.
    steam_T : float
        Steam temperature, K.  Required when
        ``stripping_mode == "steam"``.

    Validation rules
    ----------------
    * reboil mode: ``0 < bottoms_rate < flow``.
    * steam mode:  ``0 < bottoms_rate < flow + steam_flow``,
      ``steam_flow > 0``, ``steam_z`` sums to 1, ``steam_T > 0``.
    """
    draw_stage: int
    return_stage: int
    n_stages: int
    flow: float
    bottoms_rate: float
    pressure: float
    stripping_mode: str = "reboil"
    steam_flow: float = 0.0
    steam_z: Optional[Sequence[float]] = None
    steam_T: float = 0.0


@dataclass
class Spec:
    """Design specification for a distillation column.

    Used in design mode: instead of fixing both ``distillate_rate`` (D)
    and ``reflux_ratio`` (R), the user passes 1 or 2 ``Spec`` objects
    plus the corresponding number of "free" unknowns (D = None and/or
    R = None).  An outer Newton loop wraps the column solver and
    iterates D and/or R until the specs are satisfied.

    Supported kinds:

    - ``"x_D"`` (with ``species``): distillate composition.
      Residual = x_D[species] - value.
    - ``"x_B"`` (with ``species``): bottoms composition.
      Residual = x_B[species] - value.
    - ``"recovery_D"`` (with ``species``): fraction of species in distillate.
      Residual = (D * x_D[species]) / (total_F * avg_z[species]) - value.
    - ``"recovery_B"`` (with ``species``): fraction in bottoms (analogous).
    - ``"ratio"`` (with ``species`` numerator, ``species2`` denominator):
      x_D[species] / x_D[species2] - value.
    - ``"Q_C"`` (no species): condenser duty in Watts/J/h (whatever
      ``h_V_funcs`` units imply).  Requires ``h_V_funcs`` and
      ``h_L_funcs`` even in CMO mode (used post-solve only).
      Residual = Q_C_computed - value.
    - ``"Q_R"`` (no species): reboiler duty (analogous).

    Parameters
    ----------
    kind : str
        One of the kinds above.
    value : float
        Target for the spec.
    species : str, optional
        Species name used by composition / recovery / ratio specs.
    species2 : str, optional
        Second species name (required for ``"ratio"``).
    """
    kind: str
    value: float
    species: Optional[str] = None
    species2: Optional[str] = None


def _compute_Q_C(
    res: 'ColumnResult',
    h_V_funcs: Sequence[Callable[[float], float]],
    h_L_funcs: Sequence[Callable[[float], float]],
) -> float:
    """Condenser duty (post-solve), positive for heat removed.

    For a TOTAL condenser the duty is the latent heat of the vapor
    leaving the top tray:

        Q_C = V_top * (h_V(y_top, T_top) - h_L(y_top, T_top))

    where V_top = (R+1) * D and y_top is the actual leaving-vapor
    composition of stage 0 (= the distillate composition in this
    library's convention).

    For a PARTIAL condenser the condenser is itself stage 0 and the
    duty closes the H-balance of that stage:

        Q_C = V[1] * h_V[1] - D * h_V[0] - L[0] * h_L[0]
    """
    C = len(res.species_names)
    T0 = float(res.T[0])
    V_top = (res.reflux_ratio + 1.0) * res.D
    y0 = res.y[0]
    if getattr(res, "condenser", "total") == "partial":
        # Partial: V[1] from below condenses partly, vapor leaves as D
        T1 = float(res.T[1])
        y1 = res.y[1]
        x0 = res.x[0]
        h_V_1 = sum(y1[i] * h_V_funcs[i](T1) for i in range(C))
        h_V_0 = sum(y0[i] * h_V_funcs[i](T0) for i in range(C))
        h_L_0 = sum(x0[i] * h_L_funcs[i](T0) for i in range(C))
        L0 = res.reflux_ratio * res.D
        return float(res.V[1] * h_V_1 - res.D * h_V_0 - L0 * h_L_0)
    # Total condenser: latent heat of V_top at composition y_top, T_top
    h_V_top = sum(y0[i] * h_V_funcs[i](T0) for i in range(C))
    h_L_top = sum(y0[i] * h_L_funcs[i](T0) for i in range(C))
    return float(V_top * (h_V_top - h_L_top))


def _compute_Q_R(
    res: 'ColumnResult',
    h_V_funcs: Sequence[Callable[[float], float]],
    h_L_funcs: Sequence[Callable[[float], float]],
) -> float:
    """Reboiler duty (post-solve), positive for heat added.

        Q_R = V_N * h_V_N + B * h_L_N - L_{N-1} * h_L_{N-1}

    All terms in the H-balance for the reboiler stage; the sum is
    the heat that must be supplied externally to vaporize V_N moles
    out of the descending L_{N-1} liquid stream.
    """
    C = len(res.species_names)
    n = res.n_stages
    Tb = float(res.T[n - 1])
    Tabove = float(res.T[n - 2])
    h_V_N = sum(res.y[n - 1, i] * h_V_funcs[i](Tb) for i in range(C))
    h_L_N = sum(res.x[n - 1, i] * h_L_funcs[i](Tb) for i in range(C))
    h_L_above = sum(res.x[n - 2, i] * h_L_funcs[i](Tabove) for i in range(C))
    return float(res.V[n - 1] * h_V_N
                 + res.B * h_L_N
                 - res.L[n - 2] * h_L_above)


def _evaluate_spec(
    spec: 'Spec',
    res: 'ColumnResult',
    species_names: Sequence[str],
    feed_z_total: np.ndarray,   # total moles of each species fed (F * avg z)
    h_V_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    h_L_funcs: Optional[Sequence[Callable[[float], float]]] = None,
) -> float:
    """Compute the spec residual at the converged column state.

    Returns the dimensionless quantity (current - target) so that
    convergence is ``|residual| < tol``.  Composition / recovery /
    ratio specs are O(1).  Q_C / Q_R specs scale as the column duty
    (in the units of ``h_V_funcs``).  If duty specs are used at very
    different orders of magnitude, normalize the target accordingly
    when setting ``spec_outer_tol``.
    """
    kind = spec.kind
    if kind in ("x_D", "x_B", "recovery_D", "recovery_B"):
        if spec.species is None:
            raise ValueError(f"spec kind {kind!r} requires a species name")
        try:
            i = list(species_names).index(spec.species)
        except ValueError:
            raise ValueError(
                f"spec species {spec.species!r} not in {list(species_names)}")
        if kind == "x_D":
            return float(res.x_D[i] - spec.value)
        if kind == "x_B":
            return float(res.x_B[i] - spec.value)
        if kind == "recovery_D":
            tot = float(feed_z_total[i])
            if tot <= 0:
                raise ValueError(
                    f"recovery_D spec requires species {spec.species!r} "
                    f"to be present in the feed")
            return float((res.D * res.x_D[i]) / tot - spec.value)
        if kind == "recovery_B":
            tot = float(feed_z_total[i])
            if tot <= 0:
                raise ValueError(
                    f"recovery_B spec requires species {spec.species!r} "
                    f"to be present in the feed")
            return float((res.B * res.x_B[i]) / tot - spec.value)
    if kind == "ratio":
        if spec.species is None or spec.species2 is None:
            raise ValueError("ratio spec requires species and species2")
        try:
            i_num = list(species_names).index(spec.species)
            i_den = list(species_names).index(spec.species2)
        except ValueError as e:
            raise ValueError(
                f"ratio spec species not in {list(species_names)}: {e}")
        denom = float(res.x_D[i_den])
        if denom < 1e-30:
            return 1e30 - spec.value   # near-zero denominator: large residual
        return float(res.x_D[i_num] / denom - spec.value)
    if kind in ("Q_C", "Q_R"):
        if h_V_funcs is None or h_L_funcs is None:
            raise ValueError(
                f"spec kind {kind!r} requires h_V_funcs and h_L_funcs "
                f"even in CMO mode (used post-solve to compute duty)")
        if kind == "Q_C":
            return _compute_Q_C(res, h_V_funcs, h_L_funcs) - spec.value
        return _compute_Q_R(res, h_V_funcs, h_L_funcs) - spec.value
    raise ValueError(f"unknown spec kind {kind!r}")


_R_GAS_RC = 8.31446261815324


def _stage_K_gamma_phi_eos(
    T_j: float, p_j: float, x_j: np.ndarray,
    activity_model,
    psat_funcs: Sequence[Callable[[float], float]],
    vapor_eos,
    V_L_arr: Optional[np.ndarray] = None,         # (C,) liquid molar vol
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    K_init: Optional[np.ndarray] = None,
    max_picard: int = 25,
    tol: float = 1e-9,
):
    """One-stage γ-φ-EOS K-value calculation with inner Picard on
    self-consistency between ``y`` and ``φ_V``.

    Returns (K_arr, gamma_arr, picard_iters).  ``K`` is consistent with

        K_i = γ_i * p_sat,i * Φ_sat,i * exp(Poynt) / (p * φ_V,i)

    Falls back to modified-Raoult on EOS failure (e.g. supercritical
    or near-critical pure component) so the outer Newton remains
    well-conditioned.  This is acceptable because the column residual
    only requires *some* converged K; if a stage straddles a
    critical region we accept the modified-Raoult error rather than
    blow up the column solve.
    """
    C = len(x_j)
    x_safe = np.maximum(x_j, 1e-30)
    x_norm = x_safe / x_safe.sum()
    gamma = np.asarray(activity_model.gammas(T_j, x_norm))
    psat = np.array([f(T_j) for f in psat_funcs])

    # Phi_sat (default 1)
    if phi_sat_funcs is not None:
        Phi_sat = np.array([f(T_j) for f in phi_sat_funcs])
    else:
        Phi_sat = np.ones(C)

    # Poynting factor (default 1)
    if V_L_arr is not None:
        Poynt = np.exp(V_L_arr * (p_j - psat) / (_R_GAS_RC * T_j))
    else:
        Poynt = np.ones(C)

    # Initial K-value: warm start or modified Raoult
    if K_init is not None:
        K = np.asarray(K_init, dtype=float).copy()
    else:
        K = gamma * psat / p_j

    last_err = float("inf")
    for it in range(max_picard):
        y = K * x_norm
        y_sum = y.sum()
        if y_sum < 1e-30:
            break
        y = y / y_sum
        try:
            rho_v = vapor_eos.density_from_pressure(p_j, T_j, y,
                                                     phase_hint="vapor")
            ln_phi_v = vapor_eos.ln_phi(rho_v, T_j, y)
            phi_v = np.exp(ln_phi_v)
        except Exception:
            phi_v = np.ones(C)   # Modified-Raoult fallback
        K_new = (gamma * psat * Phi_sat * Poynt) / (p_j * phi_v)
        last_err = float(np.max(np.abs(K_new - K) / np.maximum(K, 1e-12)))
        K = K_new
        if last_err < tol:
            return K, gamma, it + 1
    return K, gamma, max_picard


def _get_F_attr(f) -> float:
    """Extract F flow from a feed entry (FeedSpec, dict, or tuple)."""
    if isinstance(f, FeedSpec):
        return float(f.F)
    if hasattr(f, "F"):
        return float(f.F)
    if isinstance(f, dict):
        return float(f["F"])
    return float(f[1])


def _normalize_pressure(
    pressure,
    pressure_drop: Optional[float],
    n_stages: int,
) -> np.ndarray:
    """Convert ``pressure`` (scalar or sequence) and an optional uniform
    ``pressure_drop`` per stage into a per-stage pressure array of length
    ``n_stages``.

    - ``pressure`` scalar, ``pressure_drop`` None or 0.0:
        uniform array filled with ``pressure``.
    - ``pressure`` scalar, ``pressure_drop`` > 0:
        p[j] = pressure + j * pressure_drop  (top stage at ``pressure``;
        each lower stage adds the drop)
    - ``pressure`` sequence: used directly; ``pressure_drop`` must be
        None or 0.0 (would conflict).
    """
    if hasattr(pressure, "__len__"):
        p_arr = np.asarray(pressure, dtype=float)
        if p_arr.size != n_stages:
            raise ValueError(
                f"pressure array length {p_arr.size} != n_stages={n_stages}")
        if pressure_drop is not None and pressure_drop != 0.0:
            raise ValueError(
                "pressure_drop cannot be combined with a per-stage pressure "
                "array; pass either a scalar pressure + pressure_drop, "
                "or a full pressure array.")
        if (p_arr <= 0).any():
            raise ValueError("all pressures must be positive")
        return p_arr
    p0 = float(pressure)
    if p0 <= 0:
        raise ValueError(f"pressure {p0} must be positive")
    dp = float(pressure_drop) if pressure_drop is not None else 0.0
    p_arr = p0 + dp * np.arange(n_stages, dtype=float)
    if (p_arr <= 0).any():
        raise ValueError(
            f"pressure_drop={dp} drives pressure non-positive over "
            f"{n_stages} stages (top {p0}, would-be bottom {p_arr[-1]})")
    return p_arr


def _normalize_efficiency(
    stage_efficiency,
    n_stages: int,
) -> np.ndarray:
    """Convert ``stage_efficiency`` (None | scalar | sequence) into a
    per-stage Murphree vapor efficiency array of length ``n_stages``.

    - None (default): all 1.0 (full equilibrium on every stage).
    - scalar 0 < E <= 1: applied to all stages except the reboiler
      (which is always treated as a full equilibrium stage; E[N-1] = 1).
    - sequence of length ``n_stages``: used directly.  The reboiler
      entry is forced to 1.0 even if the user passes something else,
      because at a reboiler the vapor IS the equilibrium vapor (there
      is no vapor stream below it for partial-equilibrium mixing).
    """
    if stage_efficiency is None:
        return np.ones(n_stages, dtype=float)
    if hasattr(stage_efficiency, "__len__"):
        E_arr = np.asarray(stage_efficiency, dtype=float)
        if E_arr.size != n_stages:
            raise ValueError(
                f"stage_efficiency length {E_arr.size} != "
                f"n_stages={n_stages}")
    else:
        E_val = float(stage_efficiency)
        E_arr = np.full(n_stages, E_val, dtype=float)
    if (E_arr <= 0).any() or (E_arr > 1).any():
        raise ValueError(
            "stage_efficiency must lie in (0, 1]; "
            f"got min={E_arr.min()}, max={E_arr.max()}")
    # Reboiler is always full equilibrium (no vapor below for partial mixing)
    E_arr[-1] = 1.0
    return E_arr


def _compute_y_actual(
    K_arr: np.ndarray,    # (n_stages, C)
    x_arr: np.ndarray,    # (n_stages, C)
    E_arr: np.ndarray,    # (n_stages,)
) -> np.ndarray:
    """Compute the actual vapor composition leaving each stage given
    a Murphree vapor efficiency profile.

    Definition (recursive, bottom-up):
        y_actual[N-1, i] = K[N-1, i] * x[N-1, i]   (reboiler at full eq)
        y_actual[j, i]   = E[j] * K[j, i] * x[j, i]
                           + (1 - E[j]) * y_actual[j+1, i]   for j < N-1

    For E[j] = 1 on every stage, y_actual reduces to the equilibrium
    vapor K * x bit-identically.

    Note that when ``sum_i K[j,i] x[j,i] = 1`` holds at every stage
    (the bubble-point closure), the recursion automatically preserves
    ``sum_i y_actual[j, i] = 1``: a convex combination of two unit-sum
    vectors is itself unit-sum.
    """
    n_stages, C = K_arr.shape
    y_eq = K_arr * x_arr
    y_actual = np.zeros_like(y_eq)
    y_actual[-1] = y_eq[-1]
    for j in range(n_stages - 2, -1, -1):
        y_actual[j] = E_arr[j] * y_eq[j] + (1.0 - E_arr[j]) * y_actual[j + 1]
    return y_actual


def _normalize_pump_arounds(
    pump_arounds: Optional[Sequence],
    n_stages: int,
):
    """Convert ``pump_arounds`` (None or sequence of PumpAround / dict /
    tuple) into structured arrays for fast residual evaluation.

    Returns
    -------
    pa_draw   : (P,) int array, 1-indexed draw stages
    pa_return : (P,) int array, 1-indexed return stages
    pa_flow   : (P,) float array
    pa_dT     : (P,) float array
    """
    if pump_arounds is None or len(pump_arounds) == 0:
        return (np.zeros(0, dtype=int),
                np.zeros(0, dtype=int),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float))
    P = len(pump_arounds)
    draw  = np.zeros(P, dtype=int)
    ret   = np.zeros(P, dtype=int)
    flow  = np.zeros(P, dtype=float)
    dT    = np.zeros(P, dtype=float)
    for k, pa in enumerate(pump_arounds):
        if isinstance(pa, PumpAround):
            d, r, f, dt = pa.draw_stage, pa.return_stage, pa.flow, pa.dT
        elif isinstance(pa, dict):
            d = pa["draw_stage"]; r = pa["return_stage"]; f = pa["flow"]
            dt = pa.get("dT", 0.0)
        else:
            d = pa[0]; r = pa[1]; f = pa[2]
            dt = pa[3] if len(pa) > 3 else 0.0
        draw[k] = int(d); ret[k] = int(r); flow[k] = float(f); dT[k] = float(dt)
    # Validate
    for k in range(P):
        if not (1 <= ret[k] < draw[k] <= n_stages):
            raise ValueError(
                f"pump_around {k}: must satisfy "
                f"1 <= return_stage ({ret[k]}) < draw_stage ({draw[k]}) "
                f"<= n_stages ({n_stages})")
        if flow[k] <= 0:
            raise ValueError(
                f"pump_around {k}: flow must be positive, got {flow[k]}")
        if dT[k] < 0:
            raise ValueError(
                f"pump_around {k}: dT must be non-negative (a pump-around "
                f"removes heat; pass dT=0 for isothermal)")
    return draw, ret, flow, dT


def _normalize_pump_arounds(
    pump_arounds: Optional[Sequence],
    n_stages: int,
):
    """Normalize a list of pump-arounds into internal arrays.

    Returns
    -------
    pa_draw   : (P,) int array of Python-indexed draw stages
    pa_return : (P,) int array of Python-indexed return stages
    pa_flow   : (P,) float array of pump-around flows
    pa_dT     : (P,) float array of cooling temperature drops [K]
    pa_L_add  : (n_stages,) float array; per-Python-stage extra liquid
                flow contributed by pump-arounds.  ``L[j]`` (liquid
                leaving Python stage j going down) gets +pa.flow added
                for every pump-around with j_return-1 <= j <= j_draw-2.
    """
    if pump_arounds is None or len(pump_arounds) == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0, dtype=float), np.zeros(0, dtype=float),
                np.zeros(n_stages, dtype=float))

    P = len(pump_arounds)
    pa_draw   = np.zeros(P, dtype=int)
    pa_return = np.zeros(P, dtype=int)
    pa_flow   = np.zeros(P, dtype=float)
    pa_dT     = np.zeros(P, dtype=float)
    for k, pa in enumerate(pump_arounds):
        if isinstance(pa, PumpAround):
            d, r, F, dT = pa.draw_stage, pa.return_stage, pa.flow, pa.dT
        elif hasattr(pa, "draw_stage"):
            d = pa.draw_stage; r = pa.return_stage
            F = pa.flow; dT = getattr(pa, "dT", 0.0)
        elif isinstance(pa, dict):
            d = pa["draw_stage"]; r = pa["return_stage"]
            F = pa["flow"]; dT = pa.get("dT", 0.0)
        else:
            d = pa[0]; r = pa[1]; F = pa[2]
            dT = pa[3] if len(pa) > 3 else 0.0
        d, r = int(d), int(r)
        F, dT = float(F), float(dT)
        if not (1 <= r < d <= n_stages):
            raise ValueError(
                f"pump-around {k}: must have 1 <= return_stage ({r}) "
                f"< draw_stage ({d}) <= n_stages ({n_stages})")
        if F <= 0:
            raise ValueError(
                f"pump-around {k}: flow must be positive, got {F}")
        if dT < 0:
            raise ValueError(
                f"pump-around {k}: dT must be >= 0 (cooling), got {dT}")
        pa_draw[k] = d - 1   # Python idx
        pa_return[k] = r - 1  # Python idx
        pa_flow[k] = F
        pa_dT[k] = dT

    pa_L_add = np.zeros(n_stages, dtype=float)
    for k in range(P):
        # Python idx j_r .. j_d - 1 inclusive carry the pump-around flow
        for j in range(int(pa_return[k]), int(pa_draw[k])):
            pa_L_add[j] += pa_flow[k]

    return pa_draw, pa_return, pa_flow, pa_dT, pa_L_add


def _normalize_side_strippers(
    side_strippers: Optional[Sequence],
    n_main_stages: int,
    n_components: int = 0,
):
    """Normalize side strippers into internal arrays.

    Accepts ``SideStripper`` instances, dicts, or 6+-tuples.

    Returns
    -------
    ss_draw      : (S,) int, 1-indexed main-column draw stages
    ss_return    : (S,) int, 1-indexed main-column return stages
    ss_n_stages  : (S,) int, equilibrium stages per side stripper
    ss_flow      : (S,) float, liquid feed to each SS, mol/h
    ss_bottoms   : (S,) float, side-product flow per SS, mol/h
    ss_pressure  : (S,) float, SS pressure, Pa
    ss_mode      : (S,) object array of strings ("reboil" / "steam")
    ss_steam_flow: (S,) float, steam injection rate (0 if reboil mode)
    ss_steam_z   : (S, C) float, steam composition (zeros if reboil)
    ss_steam_T   : (S,) float, steam temperature (0 if reboil)
    """
    if side_strippers is None or len(side_strippers) == 0:
        empty_i = np.zeros(0, dtype=int)
        empty_f = np.zeros(0, dtype=float)
        empty_s = np.zeros(0, dtype=object)
        empty_2d = np.zeros((0, max(n_components, 1)), dtype=float)
        return (empty_i, empty_i, empty_i, empty_f, empty_f, empty_f,
                empty_s, empty_f, empty_2d, empty_f)
    S = len(side_strippers)
    draw = np.zeros(S, dtype=int)
    ret  = np.zeros(S, dtype=int)
    n_ss = np.zeros(S, dtype=int)
    flow = np.zeros(S, dtype=float)
    bottoms = np.zeros(S, dtype=float)
    ss_p = np.zeros(S, dtype=float)
    mode = np.empty(S, dtype=object)
    steam_flow = np.zeros(S, dtype=float)
    steam_z = np.zeros((S, max(n_components, 1)), dtype=float)
    steam_T = np.zeros(S, dtype=float)
    for k, ss in enumerate(side_strippers):
        if isinstance(ss, SideStripper):
            d, r, n, f, b, p = (ss.draw_stage, ss.return_stage,
                                 ss.n_stages, ss.flow, ss.bottoms_rate,
                                 ss.pressure)
            m = ss.stripping_mode
            sf = float(ss.steam_flow)
            sz = ss.steam_z
            sT = float(ss.steam_T)
        elif isinstance(ss, dict):
            d = ss["draw_stage"]; r = ss["return_stage"]
            n = ss["n_stages"]; f = ss["flow"]
            b = ss["bottoms_rate"]; p = ss["pressure"]
            m = ss.get("stripping_mode", "reboil")
            sf = float(ss.get("steam_flow", 0.0))
            sz = ss.get("steam_z", None)
            sT = float(ss.get("steam_T", 0.0))
        else:
            d, r, n, f, b, p = ss[0], ss[1], ss[2], ss[3], ss[4], ss[5]
            m = ss[6] if len(ss) > 6 else "reboil"
            sf = float(ss[7]) if len(ss) > 7 else 0.0
            sz = ss[8] if len(ss) > 8 else None
            sT = float(ss[9]) if len(ss) > 9 else 0.0
        draw[k] = int(d); ret[k] = int(r)
        n_ss[k] = int(n); flow[k] = float(f)
        bottoms[k] = float(b); ss_p[k] = float(p)
        mode[k] = str(m)
        steam_flow[k] = sf
        steam_T[k] = sT
        if sz is not None and n_components > 0:
            steam_z[k] = np.asarray(sz, dtype=float)

    for k in range(S):
        if not (1 <= draw[k] <= n_main_stages):
            raise ValueError(
                f"side_stripper {k}: draw_stage ({draw[k]}) must be in "
                f"[1, n_stages={n_main_stages}]")
        if not (1 <= ret[k] <= n_main_stages):
            raise ValueError(
                f"side_stripper {k}: return_stage ({ret[k]}) must be in "
                f"[1, n_stages={n_main_stages}]")
        if n_ss[k] < 1:
            raise ValueError(
                f"side_stripper {k}: n_stages must be >= 1, got {n_ss[k]}")
        if flow[k] <= 0:
            raise ValueError(
                f"side_stripper {k}: flow must be positive, got {flow[k]}")
        if ss_p[k] <= 0:
            raise ValueError(
                f"side_stripper {k}: pressure must be positive, "
                f"got {ss_p[k]}")
        m = mode[k]
        if m == "reboil":
            if not (0 < bottoms[k] < flow[k]):
                raise ValueError(
                    f"side_stripper {k} (reboil mode): bottoms_rate "
                    f"({bottoms[k]}) must satisfy 0 < bottoms_rate < flow "
                    f"({flow[k]}) so that some vapor is generated for "
                    f"stripping")
        elif m == "steam":
            if steam_flow[k] <= 0:
                raise ValueError(
                    f"side_stripper {k} (steam mode): steam_flow must "
                    f"be positive, got {steam_flow[k]}")
            if not (0 < bottoms[k] < flow[k] + steam_flow[k]):
                raise ValueError(
                    f"side_stripper {k} (steam mode): bottoms_rate "
                    f"({bottoms[k]}) must satisfy 0 < bottoms_rate < "
                    f"flow + steam_flow ({flow[k] + steam_flow[k]}) so "
                    f"that some vapor is generated for stripping")
            if steam_T[k] <= 0:
                raise ValueError(
                    f"side_stripper {k} (steam mode): steam_T must be "
                    f"positive, got {steam_T[k]}")
            if n_components > 0:
                z_sum = float(steam_z[k].sum())
                if abs(z_sum - 1.0) > 1e-6:
                    raise ValueError(
                        f"side_stripper {k} (steam mode): steam_z must "
                        f"sum to 1, got {z_sum:.6f}")
                if (steam_z[k] < -1e-12).any():
                    raise ValueError(
                        f"side_stripper {k}: steam_z has negative "
                        f"components: {steam_z[k]}")
        else:
            raise ValueError(
                f"side_stripper {k}: stripping_mode must be 'reboil' or "
                f"'steam', got {m!r}")
    return (draw, ret, n_ss, flow, bottoms, ss_p,
            mode, steam_flow, steam_z, steam_T)


def _normalize_feeds_and_draws(
    n_stages: int,
    C: int,
    feed_stage: Optional[int],
    feed_F: Optional[float],
    feed_z: Optional[Sequence[float]],
    feed_T: Optional[float],
    feeds: Optional[Sequence],
    liquid_draws: Optional[dict],
    vapor_draws: Optional[dict],
    feed_q: Optional[float] = None,
):
    """Normalize multi-feed and side-draw inputs to internal array form.

    Returns
    -------
    feeds_stage : (K,) int array (1-indexed)
    feeds_F     : (K,) float array
    feeds_z     : (K, C) float array
    feeds_T     : (K,) float array
    feeds_q     : (K,) float array (liquid fraction, 1.0 = sat. liquid)
    liquid_draws_arr : (n_stages,) float array (0-indexed by Python stage)
    vapor_draws_arr  : (n_stages,) float array
    """
    # Feeds: accept either single-feed scalars or a feeds list (mutually exclusive)
    if feeds is not None:
        if (feed_stage is not None or feed_F is not None
                or feed_z is not None):
            raise ValueError(
                "`feeds` cannot be combined with single-feed scalars "
                "(feed_stage, feed_F, feed_z, feed_T); pick one form.")
        feeds_list = list(feeds)
    else:
        if feed_stage is None or feed_F is None or feed_z is None:
            raise ValueError(
                "must provide either `feeds` or "
                "(feed_stage, feed_F, feed_z) for the single-feed shorthand.")
        feeds_list = [FeedSpec(stage=int(feed_stage),
                               F=float(feed_F),
                               z=list(feed_z),
                               T=float(feed_T) if feed_T is not None else 298.15,
                               q=float(feed_q) if feed_q is not None else 1.0)]

    if len(feeds_list) == 0:
        raise ValueError("at least one feed is required")

    K = len(feeds_list)
    feeds_stage = np.zeros(K, dtype=int)
    feeds_F = np.zeros(K, dtype=float)
    feeds_z = np.zeros((K, C), dtype=float)
    feeds_T = np.zeros(K, dtype=float)
    feeds_q = np.zeros(K, dtype=float)
    for k, f in enumerate(feeds_list):
        if isinstance(f, FeedSpec):
            stage_k, F_k, z_k = f.stage, f.F, f.z
            T_k, q_k = f.T, f.q
        elif hasattr(f, "stage") and hasattr(f, "F"):
            stage_k = f.stage; F_k = f.F; z_k = f.z
            T_k = getattr(f, "T", 298.15)
            q_k = getattr(f, "q", 1.0)
        elif isinstance(f, dict):
            stage_k = f["stage"]; F_k = f["F"]; z_k = f["z"]
            T_k = f.get("T", 298.15)
            q_k = f.get("q", 1.0)
        else:
            # Tuple form (stage, F, z) | (stage, F, z, T) | (stage, F, z, T, q)
            stage_k = f[0]; F_k = f[1]; z_k = f[2]
            T_k = f[3] if len(f) > 3 else 298.15
            q_k = f[4] if len(f) > 4 else 1.0
        feeds_stage[k] = int(stage_k)
        feeds_F[k] = float(F_k)
        z_arr = np.asarray(z_k, dtype=float)
        if z_arr.size != C:
            raise ValueError(
                f"feed {k}: z has length {z_arr.size}, expected C={C}")
        feeds_z[k] = z_arr
        feeds_T[k] = float(T_k)
        feeds_q[k] = float(q_k)

    # Validate feeds
    for k in range(K):
        if not (1 <= feeds_stage[k] <= n_stages):
            raise ValueError(
                f"feed {k}: stage {feeds_stage[k]} is outside [1, {n_stages}]")
        if feeds_F[k] <= 0:
            raise ValueError(
                f"feed {k}: F = {feeds_F[k]} must be positive")
        z_sum = feeds_z[k].sum()
        if z_sum <= 0:
            raise ValueError(f"feed {k}: z sums to {z_sum}, must be positive")
        if abs(z_sum - 1.0) > 1e-6:
            # Normalize and warn (silent normalization to avoid surprise)
            feeds_z[k] /= z_sum

    # Side draws
    liquid_draws_arr = np.zeros(n_stages, dtype=float)
    vapor_draws_arr = np.zeros(n_stages, dtype=float)
    for d, arr, label in [(liquid_draws, liquid_draws_arr, "liquid_draws"),
                          (vapor_draws,  vapor_draws_arr,  "vapor_draws")]:
        if d is None:
            continue
        if not hasattr(d, "items"):
            raise TypeError(
                f"{label} must be a mapping {{stage: flow_rate}} or None.")
        for stage, flow in d.items():
            stage = int(stage)
            flow = float(flow)
            if not (1 <= stage <= n_stages):
                raise ValueError(
                    f"{label}: stage {stage} is outside [1, {n_stages}]")
            if flow < 0:
                raise ValueError(
                    f"{label}: stage {stage} has negative flow {flow}")
            arr[stage - 1] = flow

    return (feeds_stage, feeds_F, feeds_z, feeds_T, feeds_q,
            liquid_draws_arr, vapor_draws_arr)


# =========================================================================
# Tridiagonal & bubble-point helpers
# =========================================================================

def _solve_tridiagonal(a: np.ndarray, b: np.ndarray,
                          c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal Ax = d.

    a: subdiagonal of length N (a[0] unused)
    b: main diagonal of length N
    c: superdiagonal of length N (c[N-1] unused)
    d: RHS of length N
    Returns x of length N.
    """
    N = b.size
    cp = np.zeros(N); dp = np.zeros(N); x = np.zeros(N)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for j in range(1, N):
        denom = b[j] - a[j] * cp[j-1]
        if abs(denom) < 1e-30:
            denom = 1e-30 if denom >= 0 else -1e-30
        cp[j] = c[j] / denom if j < N - 1 else 0.0
        dp[j] = (d[j] - a[j] * dp[j-1]) / denom
    x[-1] = dp[-1]
    for j in range(N - 2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j+1]
    return x


def _bubble_point_T(x: np.ndarray, p: float, psat_funcs,
                       activity_model, T_init: float,
                       T_min: float = 200.0, T_max: float = 700.0,
                       tol: float = 1e-5, maxiter: int = 50) -> float:
    """Find T such that Sum_i gamma_i(T,x) p_i^sat(T) x_i / p = 1."""
    x = np.asarray(x, dtype=float)
    x = x / max(x.sum(), 1e-30)

    def residual(T):
        psat = np.array([f(T) for f in psat_funcs])
        gammas = np.asarray(activity_model.gammas(T, x))
        return float((gammas * psat * x).sum() / p - 1.0)

    # Bracket
    T_lo, T_hi = T_min, T_max
    f_lo = residual(T_lo)
    f_hi = residual(T_hi)
    if f_lo * f_hi > 0:
        # Try to expand
        for T_try in [T_init, 0.7 * T_init, 1.3 * T_init]:
            f_try = residual(T_try)
            if f_lo * f_try < 0:
                T_hi, f_hi = T_try, f_try
                break
            if f_try * f_hi < 0:
                T_lo, f_lo = T_try, f_try
                break
        else:
            return T_init  # bracket failed; keep current
    # Bisect (simple, robust)
    T_a, T_b = T_lo, T_hi
    f_a, f_b = f_lo, f_hi
    for _ in range(maxiter):
        T_m = 0.5 * (T_a + T_b)
        f_m = residual(T_m)
        if abs(f_m) < tol or (T_b - T_a) < tol:
            return T_m
        if f_a * f_m < 0:
            T_b, f_b = T_m, f_m
        else:
            T_a, f_a = T_m, f_m
    return 0.5 * (T_a + T_b)


def _project_x_to_equilibrium(x_tent: np.ndarray, T: float,
                                  activity_model,
                                  reactions: Sequence[LiquidPhaseReaction],
                                  species_idx_in_rxn: List[List[int]]):
    """Project a tentative liquid composition onto the K_a = K_eq surface
    along the stoichiometry direction.

    Given a candidate x_tent (sums to 1), find the closest x_eq such that
    K_a(x_eq, T) = K_eq(T) AND x_eq - x_tent is in the column space of the
    stoichiometry matrix.  For R = 1 this is a one-parameter projection
    solved by bisection on a 1-mol-of-stuff basis.  For R > 1, damped
    Newton with the ideal-mixture Jacobian.

    Returns (x_eq, xi_norm) where x_eq is the projected composition
    (sums to 1) and xi_norm is the per-1-mol-of-stuff extent vector
    (length R).  xi_norm has the same sign convention as the reactions
    (positive = forward).
    """
    R = len(reactions)
    n_comp = x_tent.size
    if R == 0:
        return x_tent.copy(), np.zeros(0)

    nu = np.zeros((R, n_comp))
    for r, rxn in enumerate(reactions):
        for sp_idx, nu_local in enumerate(rxn.nu):
            i = species_idx_in_rxn[r][sp_idx]
            nu[r, i] = nu_local

    x_tent = np.maximum(x_tent, 1e-30)
    x_tent = x_tent / x_tent.sum()

    if R == 1:
        nu_vec = nu[0]
        # Feasible xi range (no negative moles)
        xi_lo, xi_hi = -np.inf, np.inf
        for i in range(n_comp):
            if nu_vec[i] > 0:
                xi_lo = max(xi_lo, -x_tent[i] / nu_vec[i])
            elif nu_vec[i] < 0:
                xi_hi = min(xi_hi, -x_tent[i] / nu_vec[i])
        eps = 1e-12
        xi_lo = (xi_lo + eps) if xi_lo > -np.inf else -1e6
        xi_hi = (xi_hi - eps) if xi_hi <  np.inf else  1e6
        if xi_lo >= xi_hi:
            return x_tent.copy(), np.zeros(1)

        ln_K_eq = reactions[0].ln_K_eq(T)
        def res(xi):
            n_new = x_tent + nu_vec * xi
            if (n_new <= 0).any():
                return None
            x_new = n_new / n_new.sum()
            gammas = np.asarray(activity_model.gammas(T, x_new))
            return float((nu_vec * (np.log(gammas) + np.log(x_new))).sum()
                          - ln_K_eq)

        f_lo = res(xi_lo)
        f_hi = res(xi_hi)
        if f_lo is None or f_hi is None or f_lo * f_hi > 0:
            return x_tent.copy(), np.zeros(1)

        a, b = xi_lo, xi_hi
        fa = f_lo
        xi_m = 0.5 * (a + b)
        for _ in range(200):
            xi_m = 0.5 * (a + b)
            fm = res(xi_m)
            if fm is None:
                b = xi_m
                continue
            if abs(fm) < 1e-12 or (b - a) < 1e-14:
                break
            if fa * fm < 0:
                b = xi_m
            else:
                a, fa = xi_m, fm
        n_new = x_tent + nu_vec * xi_m
        x_eq = n_new / n_new.sum()
        return x_eq, np.array([xi_m])

    # Multi-reaction: damped Newton with ideal-mixture Jacobian
    xi = np.zeros(R)
    ln_K_eq = np.array([rxn.ln_K_eq(T) for rxn in reactions])
    dn = nu.sum(axis=1)
    damping = 0.7
    x_new = x_tent.copy()
    for _ in range(150):
        n_new = x_tent + nu.T @ xi
        if (n_new <= 0).any():
            xi = 0.5 * xi
            continue
        x_new = n_new / n_new.sum()
        gammas = np.asarray(activity_model.gammas(T, x_new))
        ln_gx = (np.log(np.maximum(gammas, 1e-30))
                 + np.log(np.maximum(x_new, 1e-30)))
        f = nu @ ln_gx - ln_K_eq
        if np.max(np.abs(f)) < 1e-10:
            return x_new, xi
        inv_n = 1.0 / np.maximum(n_new, 1e-30)
        J = np.einsum('ri,si,i->rs', nu, nu, inv_n)
        N_tot = max(n_new.sum(), 1e-30)
        J -= np.outer(dn, dn) / N_tot
        try:
            dxi = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dxi, *_ = np.linalg.lstsq(J, -f, rcond=None)
        xi = xi + damping * dxi
    return x_new, xi


def _solve_extent_at_stage(x: np.ndarray, T: float,
                              activity_model,
                              reactions: Sequence[LiquidPhaseReaction],
                              species_idx_in_rxn: List[List[int]],
                              n_L: float,
                              tol: float = 1e-9, maxiter: int = 100) -> np.ndarray:
    """Compute reaction extents on a stage to satisfy chemical equilibrium.

    Solves: K_eq,r(T) = Prod_i (gamma_i x_i)^nu[r,i]  for each reaction r

    where x is the CURRENT liquid composition on the stage. The extent
    represents the per-stage extent (so n_i_new = n_i_old + nu[i] * xi).

    For R=1: bisection on xi
    For R>1: damped Newton with ideal-mixture Jacobian

    Returns: array of length R giving xi_r in mol units (consistent with n_L).
    """
    R = len(reactions)
    if R == 0 or n_L < 1e-12:
        return np.zeros(R)

    n_comp = x.size
    # Stoichiometry matrix R x N (full species list)
    nu = np.zeros((R, n_comp))
    for r, rxn in enumerate(reactions):
        for sp_local_idx, nu_local in enumerate(rxn.nu):
            j = species_idx_in_rxn[r][sp_local_idx]
            nu[r, j] = nu_local

    # Liquid moles on stage
    n_init = n_L * x  # length n_comp

    if R == 1:
        # Single reaction: bisection
        # Feasible range
        xi_lo, xi_hi = -np.inf, np.inf
        for i in range(n_comp):
            if nu[0, i] > 0:
                xi_lo = max(xi_lo, -n_init[i] / nu[0, i])
            elif nu[0, i] < 0:
                xi_hi = min(xi_hi, -n_init[i] / nu[0, i])
        eps = 1e-10 * max(1.0, n_L)
        xi_lo = (xi_lo + eps) if xi_lo > -np.inf else -1e6
        xi_hi = (xi_hi - eps) if xi_hi <  np.inf else  1e6
        if xi_lo >= xi_hi:
            return np.zeros(R)

        ln_K_eq = reactions[0].ln_K_eq(T)
        def residual(xi_val):
            n_new = n_init + nu[0] * xi_val
            if (n_new <= 0).any():
                return math.nan
            n_total = n_new.sum()
            x_new = n_new / n_total
            gammas = np.asarray(activity_model.gammas(T, x_new))
            return float((nu[0] * (np.log(gammas) + np.log(x_new))).sum()
                           - ln_K_eq)

        f_lo = residual(xi_lo); f_hi = residual(xi_hi)
        if math.isnan(f_lo) or math.isnan(f_hi) or f_lo * f_hi > 0:
            # No bracket; just return zero (no update)
            return np.zeros(R)
        a, b = xi_lo, xi_hi
        fa, fb = f_lo, f_hi
        xi_m = 0.5 * (a + b)
        for _ in range(maxiter):
            xi_m = 0.5 * (a + b)
            f_m = residual(xi_m)
            if math.isnan(f_m):
                b = xi_m
                continue
            if abs(f_m) < tol or (b - a) < tol * max(1.0, abs(b)):
                break
            if fa * f_m < 0:
                b, fb = xi_m, f_m
            else:
                a, fa = xi_m, f_m
        return np.array([xi_m])

    # Multi-reaction: damped Newton
    xi = np.zeros(R)
    ln_K_eq = np.array([rxn.ln_K_eq(T) for rxn in reactions])
    dn = nu.sum(axis=1)
    damping = 0.7
    for _ in range(maxiter):
        n_new = n_init + nu.T @ xi
        if (n_new <= 0).any():
            xi = 0.5 * xi
            continue
        n_total = n_new.sum()
        x_new = n_new / n_total
        gammas = np.asarray(activity_model.gammas(T, x_new))
        ln_gx = np.log(np.maximum(gammas, 1e-30)) + np.log(np.maximum(x_new, 1e-30))
        f = nu @ ln_gx - ln_K_eq
        err = float(np.max(np.abs(f)))
        if err < tol:
            return xi
        inv_n = 1.0 / np.maximum(n_new, 1e-30)
        J = np.einsum('ri,si,i->rs', nu, nu, inv_n)
        J -= np.outer(dn, dn) / max(n_total, 1e-30)
        try:
            dxi = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dxi, *_ = np.linalg.lstsq(J, -f, rcond=None)
        xi = xi + damping * dxi
    return xi


# =========================================================================
# Naphtali-Sandholm simultaneous Newton solver
# =========================================================================

def _build_block_tridiag_jacobian(
    residuals_func,
    w: np.ndarray,
    F_curr: np.ndarray,
    var_offsets: np.ndarray,
    n_stages_main: int,
    n_total_stages: int,
    fd_step: float,
    has_nonlocal_coupling: bool,
) -> np.ndarray:
    """Build the Newton Jacobian using Curtis-Powell-Reid sparsity
    compression for the block-tridiagonal structure.

    Each main-column stage j has residuals that touch variables only at
    stages j-1, j, j+1 (plus side-stripper / pump-around couplings if
    present).  This block-tridiagonal pattern means that variables in
    stages {0, 3, 6, ...} can be perturbed simultaneously: their
    residual sensitivities don't overlap.  CPR reduces the FD cost from
    ``2 * n_total`` residual calls per Newton iteration to roughly
    ``2 * 3 * max(vars_per_stage)`` calls, regardless of column length.

    For columns with side strippers or pump-arounds (non-local coupling),
    falls back to the dense Jacobian.

    Parameters
    ----------
    residuals_func : callable(w) -> F
    w : current variable vector
    F_curr : current residual vector (unused; kept for signature symmetry)
    var_offsets : (n_total_stages+1,) integer array of variable offsets
    n_stages_main : number of main-column stages
    n_total_stages : total stages including side-stripper stages
    fd_step : finite-difference step size
    has_nonlocal_coupling : if True, use dense fallback

    Returns
    -------
    J : (n_total, n_total) Jacobian
    """
    n_total = int(var_offsets[-1])

    if has_nonlocal_coupling:
        # Dense central-difference fallback — the original code path.
        J = np.zeros((n_total, n_total))
        for k in range(n_total):
            h_k = max(fd_step * abs(w[k]), fd_step)
            w_p = w.copy(); w_p[k] += h_k
            w_m = w.copy(); w_m[k] -= h_k
            J[:, k] = (residuals_func(w_p) - residuals_func(w_m)) / (2.0 * h_k)
        return J

    # CPR-compressed: probe with stride-3 stage groups
    J = np.zeros((n_total, n_total))
    # Group all variables by (stage_idx, local_var_idx).  Two columns
    # k and k' can be probed simultaneously if their residual columns
    # don't overlap — equivalent to perturbing variables in stages
    # whose stage indices differ by ≥ 3 (so their stencils — j-1,j,j+1 —
    # are disjoint).
    stage_groups = [list(range(g, n_total_stages, 3)) for g in range(3)]

    # Determine max vars-per-stage so we can iterate by local-var-index
    n_vars_per_stage = np.diff(var_offsets)
    max_vars = int(n_vars_per_stage.max())

    for group_stages in stage_groups:
        for local_var in range(max_vars):
            # Determine which actual variable indices to perturb
            # in this group at this local-var slot
            probe_idxs = []
            for j in group_stages:
                if local_var < int(n_vars_per_stage[j]):
                    probe_idxs.append(int(var_offsets[j] + local_var))
            if not probe_idxs:
                continue

            # Build perturbed vectors (forward and backward)
            w_p = w.copy()
            w_m = w.copy()
            h_arr = {}
            for k in probe_idxs:
                h = max(fd_step * abs(w[k]), fd_step)
                h_arr[k] = h
                w_p[k] += h
                w_m[k] -= h

            F_p = residuals_func(w_p)
            F_m = residuals_func(w_m)
            dF = (F_p - F_m)   # (n_total,)

            # Distribute the response to each perturbed column.  Within
            # this group, each j touches residuals only at its 3-stage
            # stencil; those stencils don't overlap across the group, so
            # we can read off each column's contribution from the local
            # rows.
            for k in probe_idxs:
                # Find which stage k belongs to
                # var_offsets is sorted, use bisect
                stage_k = int(np.searchsorted(var_offsets, k,
                                                    side="right") - 1)
                # The residual rows touched by this k are stages
                # max(0, stage_k-1) through min(N-1, stage_k+1)
                stencil_lo = max(0, stage_k - 1)
                stencil_hi = min(n_total_stages - 1, stage_k + 1)
                row_lo = int(var_offsets[stencil_lo])
                row_hi = int(var_offsets[stencil_hi + 1])
                J[row_lo:row_hi, k] = (dF[row_lo:row_hi]
                                              / (2.0 * h_arr[k]))

    return J



def _naphtali_sandholm_solve(
    n_stages: int,
    feeds_stage: np.ndarray,        # (K,) 1-indexed
    feeds_F: np.ndarray,            # (K,)
    feeds_z: np.ndarray,            # (K, C)
    feeds_q: np.ndarray,            # (K,) liquid fraction; q=1 saturated liquid
    liquid_draws_arr: np.ndarray,   # (n_stages,) 0-indexed
    vapor_draws_arr: np.ndarray,    # (n_stages,) 0-indexed
    distillate_rate: float,
    reflux_ratio: float,
    p_arr: np.ndarray,              # (n_stages,) per-stage pressure
    species_names: list,
    activity_model,
    psat_funcs: Sequence[Callable[[float], float]],
    reactions: Sequence[LiquidPhaseReaction],
    reactive_set: set,
    species_idx_in_rxn: List[List[int]],
    T_init: np.ndarray,
    x_init: np.ndarray,
    xi_init: Optional[np.ndarray] = None,
    E_arr: Optional[np.ndarray] = None,   # (n_stages,) Murphree eff (default all 1.0)
    pa_draw: Optional[np.ndarray] = None,    # (P,) Python-indexed draw stages
    pa_return: Optional[np.ndarray] = None,  # (P,) Python-indexed return stages
    pa_flow: Optional[np.ndarray] = None,    # (P,) flow rates
    pa_L_add: Optional[np.ndarray] = None,   # (n_stages,) extra L per stage
    ss_draw: Optional[np.ndarray] = None,    # (S,) 1-indexed draw stages
    ss_return: Optional[np.ndarray] = None,  # (S,) 1-indexed return stages
    ss_n_stages: Optional[np.ndarray] = None,# (S,) stages per side stripper
    ss_flow: Optional[np.ndarray] = None,    # (S,) liquid feed to each SS
    ss_bottoms: Optional[np.ndarray] = None, # (S,) side product flow per SS
    ss_pressure: Optional[np.ndarray] = None,# (S,) SS pressure
    ss_mode: Optional[np.ndarray] = None,    # (S,) "reboil" / "steam"
    ss_steam_flow: Optional[np.ndarray] = None,  # (S,) steam mol/h
    ss_steam_z: Optional[np.ndarray] = None,     # (S, C) steam composition
    ss_steam_T: Optional[np.ndarray] = None,     # (S,) steam K
    vapor_eos = None,
    V_L_arr: Optional[np.ndarray] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    max_newton_iter: int = 30,
    tol: float = 1e-7,
    fd_step: float = 1e-7,
    verbose: bool = False,
) -> 'ColumnResult':
    """Solve the column via simultaneous Newton on the full augmented system.

    Per stage j the unknowns are {x_{j,i}, T_j, xi_{j,r}}; the matching
    residuals are component balances (C), the bubble-point closure
    Sum_i K_{j,i} x_{j,i} = 1 (1), and chemistry K_a,r = K_eq,r (R per
    reactive stage).  The Jacobian is block-tridiagonal because each
    stage's residual only touches variables at j-1, j, j+1.

    Convergence quadratic in clean cases; the line-search backtracks
    when a full Newton step doesn't reduce ||F||_inf.

    The Jacobian is built by central-difference finite differences,
    which is robust but O(n_total) residual evaluations per Newton
    iteration.  For the column sizes typical here (<=50 stages, <=10
    species) this is fast enough to avoid the bookkeeping overhead of
    analytical derivatives.
    """
    C = len(species_names)
    R = len(reactions)
    K_feeds = len(feeds_F)

    # Cumulative feed and draw bookkeeping for the L/V profile.
    # cum_qF[k]   = sum of  q  * F over feeds at stages 1..k (1-indexed)  [liquid contrib]
    # cum_vapF[k] = sum of (1-q)*F over feeds at stages 1..k              [vapor contrib]
    # cum_U[k]    = sum of liquid draws at stages 1..k
    # cum_W[k]    = sum of vapor  draws at stages 1..k
    cum_qF = np.zeros(n_stages + 1)
    cum_vapF = np.zeros(n_stages + 1)
    for k_idx in range(K_feeds):
        s = int(feeds_stage[k_idx])
        F_k = float(feeds_F[k_idx])
        q_k = float(feeds_q[k_idx])
        cum_qF[s:]   += q_k * F_k
        cum_vapF[s:] += (1.0 - q_k) * F_k
    cum_U = np.zeros(n_stages + 1)
    cum_W = np.zeros(n_stages + 1)
    for j in range(1, n_stages + 1):
        cum_U[j] = cum_U[j - 1] + float(liquid_draws_arr[j - 1])
        cum_W[j] = cum_W[j - 1] + float(vapor_draws_arr[j - 1])
    total_F = float(feeds_F.sum())
    total_U = float(liquid_draws_arr.sum())
    total_W = float(vapor_draws_arr.sum())

    # Pump-around bookkeeping (v0.9.74).  A pump-around drawing flow F at
    # 1-indexed draw_stage and returning at 1-indexed return_stage adds
    # F mol/h to the liquid stream between return_stage and (draw_stage-1)
    # inclusive (in 1-indexed terms; in Python that's idx return_stage-1
    # through draw_stage-2).  Mass conservation: the same flow enters at
    # return_stage and leaves at draw_stage, so the L profile at and below
    # draw_stage is unaffected.

    # Precompute the feed source per (stage, species) for fast residual eval.
    # The component balance is q-independent: F_k * z_k moles enter the stage
    # regardless of phase split (the equilibration reshuffles them).
    feed_src = np.zeros((n_stages, C))   # 0-indexed stage
    for k_idx in range(K_feeds):
        s = int(feeds_stage[k_idx]) - 1
        feed_src[s] += float(feeds_F[k_idx]) * feeds_z[k_idx]

    # Pump-around setup
    if pa_draw is None or pa_flow is None or len(pa_flow) == 0:
        pa_draw = np.zeros(0, dtype=int)
        pa_return = np.zeros(0, dtype=int)
        pa_flow = np.zeros(0, dtype=float)
        pa_L_add = np.zeros(n_stages, dtype=float)
    elif pa_L_add is None:
        pa_L_add = np.zeros(n_stages, dtype=float)
        for k in range(len(pa_flow)):
            for j in range(int(pa_return[k]), int(pa_draw[k])):
                pa_L_add[j] += float(pa_flow[k])
    n_PA = len(pa_flow)

    # Side-stripper setup (v0.9.88)
    if ss_draw is None or ss_flow is None or len(ss_flow) == 0:
        ss_draw     = np.zeros(0, dtype=int)
        ss_return   = np.zeros(0, dtype=int)
        ss_n_stages = np.zeros(0, dtype=int)
        ss_flow     = np.zeros(0, dtype=float)
        ss_bottoms  = np.zeros(0, dtype=float)
        ss_pressure = np.zeros(0, dtype=float)
    n_SS = len(ss_flow)
    if ss_mode is None:
        ss_mode = np.array(["reboil"] * n_SS, dtype=object)
    if ss_steam_flow is None:
        ss_steam_flow = np.zeros(n_SS, dtype=float)
    if ss_steam_z is None:
        ss_steam_z = np.zeros((n_SS, max(C, 1)), dtype=float)
    if ss_steam_T is None:
        ss_steam_T = np.zeros(n_SS, dtype=float)
    # Total side-stripper stages (across all SS)
    n_ss_total = int(ss_n_stages.sum()) if n_SS > 0 else 0
    # ss_offset[k] = starting Python index of SS k's stages within the
    # extended stage array (offsets by n_stages so SS 0 starts at n_stages).
    ss_offset_stages = np.zeros(n_SS, dtype=int)
    for k in range(n_SS):
        ss_offset_stages[k] = n_stages + (
            int(ss_n_stages[:k].sum()) if k > 0 else 0)
    # Vapor flow at SS top (mass balance, CMO):
    #   reboil:  V_top = flow - bottoms
    #   steam:   V_top = flow + steam_flow - bottoms
    if n_SS > 0:
        ss_V_top = np.where(ss_mode == "steam",
                              ss_flow + ss_steam_flow - ss_bottoms,
                              ss_flow - ss_bottoms)
    else:
        ss_V_top = np.zeros(0)

    # Side-stripper flow modifications to the main-column profile.
    # ss_L_sub[j] is the cumulative liquid removal at Python stage j
    # (== ss_flow for stages at or below each SS's draw_stage).
    # ss_V_sub[j] is the cumulative vapor reduction at Python stage j
    # for stages strictly BELOW the return stage (SS top vapor adds
    # to the upward flow at the return stage; below it the boilup must
    # be lower to satisfy the fixed top-of-column V_top = (R+1)*D).
    # ss_bottoms_total is the total side-product flow leaving the system,
    # which reduces the main column bottoms B.
    ss_L_sub = np.zeros(n_stages)
    ss_V_sub = np.zeros(n_stages)
    for k in range(n_SS):
        j_d = int(ss_draw[k]) - 1
        R_ss = int(ss_return[k])    # 1-indexed return stage
        # Liquid: 60 mol/h leaves at the draw, so L[k] for k >= j_d is
        # reduced by ss_flow (under CMO).
        ss_L_sub[j_d:] += float(ss_flow[k])
        # Vapor: the SS returns ss_V_top at the return stage from below.
        # V[j] = vapor leaving Python stage j going up; for j+1 strictly
        # below the return stage (j+1 > ss_return, i.e., j >= R_ss in
        # 1-indexed terms which is j >= R_ss as Python idx), V is the
        # un-boosted boilup, lower than V_top by ss_V_top.
        ss_V_sub[R_ss:] += float(ss_V_top[k])
    ss_bottoms_total = float(ss_bottoms.sum()) if n_SS > 0 else 0.0
    ss_steam_total = float(ss_steam_flow.sum()) if n_SS > 0 else 0.0

    # Full R x C stoichiometry matrix in column-species order
    nu_full = np.zeros((R, C))
    for r in range(R):
        for sp_idx, nu_local in enumerate(reactions[r].nu):
            nu_full[r, species_idx_in_rxn[r][sp_idx]] = nu_local

    # Variable layout: per main stage, [x_{j,0..C-1}, T_j, xi_{j,0..R-1} (if reactive)]
    # SS stages append [x_SS, T_SS] (no chemistry on SS, v0.9.88).
    n_total_stages = n_stages + n_ss_total   # main + all SS stages flat
    n_vars_per_stage: list[int] = []
    for j in range(n_stages):
        is_reactive = (j + 1) in reactive_set
        n_vars_per_stage.append(C + 1 + (R if is_reactive else 0))
    for j in range(n_ss_total):                  # SS stages: x + T only
        n_vars_per_stage.append(C + 1)
    var_offsets = np.array([0] + list(np.cumsum(n_vars_per_stage)))
    n_total = int(var_offsets[-1])

    def unpack(w: np.ndarray):
        x = np.zeros((n_total_stages, C))
        T = np.zeros(n_total_stages)
        xi = np.zeros((n_stages, R))
        for j in range(n_total_stages):
            off = var_offsets[j]
            x[j] = w[off:off + C]
            T[j] = w[off + C]
            if j < n_stages and (j + 1) in reactive_set and R > 0:
                xi[j] = w[off + C + 1:off + C + 1 + R]
        return x, T, xi

    def pack(x: np.ndarray, T: np.ndarray, xi: np.ndarray) -> np.ndarray:
        w = np.zeros(n_total)
        for j in range(n_total_stages):
            off = var_offsets[j]
            w[off:off + C] = x[j]
            w[off + C] = T[j]
            if j < n_stages and (j + 1) in reactive_set and R > 0:
                w[off + C + 1:off + C + 1 + R] = xi[j]
        return w

    # Flow rates under CMO + reaction-mole-change adjustment
    D = float(distillate_rate)
    L_top = reflux_ratio * D                  # reflux returning to stage 1
    V_top = (reflux_ratio + 1.0) * D          # vapor leaving stage 1 to condenser

    def get_flows(xi_arr: np.ndarray):
        # L_j (Python idx j) = liquid leaving physical stage j+1 going down
        #                    = R*D + sum_{k<=j+1} q_k F_k - sum_{l<=j+1} U_l
        #                      + cum_PA[j+1]   (pump-around contribution)
        #                      - ss_L_sub[j]   (side-stripper draw)
        # V_j (Python idx j) = vapor leaving physical stage j+1 going up
        #                    = (R+1)*D + sum_{l<j+1} W_l - sum_{l<j+1} (1-q_l) F_l
        #                      + ss_V_add[j]   (side-stripper vapor return)
        # Pump-arounds do NOT affect V under CMO (they recycle liquid only).
        L = np.zeros(n_stages)
        V = np.zeros(n_stages)
        for j in range(n_stages):
            phys = j + 1
            L[j] = (L_top + cum_qF[phys] - cum_U[phys]
                    + pa_L_add[j] - ss_L_sub[j])
            V[j] = (V_top + cum_W[phys - 1] - cum_vapF[phys - 1]
                    - ss_V_sub[j])
        if R > 0:
            dn_per_rxn = np.array([float(rxn.nu.sum()) for rxn in reactions])
            dN_rxn = float((xi_arr * dn_per_rxn).sum())
        else:
            dN_rxn = 0.0
        # Overall mass balance: B = total_F - D - total_U - total_W
        #                             - ss_bottoms_total + dN_rxn
        # Pump-arounds are internal (mass-conserving) so they don't enter
        # the overall balance.  Side-stripper bottoms DO leave the system.
        B_local = (total_F + dN_rxn - D - total_U - total_W
                   - ss_bottoms_total + ss_steam_total)
        L[-1] = B_local
        return L, V, B_local

    if E_arr is None:
        E_arr = np.ones(n_stages, dtype=float)

    use_eos = vapor_eos is not None

    def residuals(w: np.ndarray) -> np.ndarray:
        x_arr, T_arr, xi_arr = unpack(w)
        K_arr = np.zeros((n_total_stages, C))
        gamma_arr = np.zeros((n_total_stages, C))
        # Build per-stage pressure: main stages use p_arr; SS stages use ss_pressure[k]
        p_full = np.zeros(n_total_stages)
        p_full[:n_stages] = p_arr
        for k in range(n_SS):
            p_full[ss_offset_stages[k]:ss_offset_stages[k]
                   + int(ss_n_stages[k])] = float(ss_pressure[k])

        for j in range(n_total_stages):
            T_j = T_arr[j]
            if use_eos:
                K_j, gamma_j, _ = _stage_K_gamma_phi_eos(
                    T_j, p_full[j], x_arr[j],
                    activity_model, psat_funcs, vapor_eos,
                    V_L_arr=V_L_arr, phi_sat_funcs=phi_sat_funcs)
                K_arr[j] = K_j
                gamma_arr[j] = gamma_j
            else:
                psat = np.array([f(T_j) for f in psat_funcs])
                x_safe = np.maximum(x_arr[j], 1e-30)
                x_norm = x_safe / x_safe.sum()
                gamma = np.asarray(activity_model.gammas(T_j, x_norm))
                gamma_arr[j] = gamma
                K_arr[j] = gamma * psat / p_full[j]

        # Actual leaving-vapor composition (Murphree-modified).  For
        # E[j] = 1 on every stage this is identical to K * x.
        # E_arr is sized to n_stages (main only); SS stages are E=1 (v0.9.88).
        E_full = np.ones(n_total_stages)
        E_full[:n_stages] = E_arr
        y_actual = _compute_y_actual(K_arr, x_arr, E_full)

        L, V, B_local = get_flows(xi_arr)
        F_vec = np.zeros(n_total)

        # ---- Main column stages ----
        for j in range(n_stages):
            off = var_offsets[j]

            # Component balance per species i
            for i in range(C):
                if j == 0:
                    out_i = D * y_actual[0, i] + L[0] * x_arr[0, i]
                    in_i = (V[1] * y_actual[1, i]
                            if n_stages > 1 else 0.0)
                elif j == n_stages - 1:
                    out_i = V[j] * y_actual[j, i] + B_local * x_arr[j, i]
                    in_i = L[j - 1] * x_arr[j - 1, i]
                else:
                    out_i = V[j] * y_actual[j, i] + L[j] * x_arr[j, i]
                    in_i = (L[j - 1] * x_arr[j - 1, i]
                            + V[j + 1] * y_actual[j + 1, i])
                # Multi-feed contribution
                in_i += feed_src[j, i]
                # Side-draw contribution: liquid draw at composition x[j,i],
                # vapor draw at composition y_actual[j,i] (the actual
                # leaving-vapor composition)
                out_i += (liquid_draws_arr[j] * x_arr[j, i]
                          + vapor_draws_arr[j] * y_actual[j, i])

                # Pump-around mass balance.  For each PA whose return is
                # at this stage (Python idx j == j_return), receive
                # pa_flow * x[j_draw, i] IN.  For each PA whose draw is
                # at this stage (j == j_draw), send pa_flow * x[j, i] OUT.
                for k_pa in range(n_PA):
                    if j == int(pa_return[k_pa]):
                        d_idx = int(pa_draw[k_pa])
                        in_i += float(pa_flow[k_pa]) * x_arr[d_idx, i]
                    if j == int(pa_draw[k_pa]):
                        out_i += float(pa_flow[k_pa]) * x_arr[j, i]

                # Side-stripper coupling: at SS draw stage, F_SS leaves as
                # liquid; at SS return stage, V_top enters as vapor.  Both
                # 1-indexed in spec, so compare to (j + 1).
                for k_ss in range(n_SS):
                    if (j + 1) == int(ss_draw[k_ss]):
                        out_i += float(ss_flow[k_ss]) * x_arr[j, i]
                    if (j + 1) == int(ss_return[k_ss]):
                        ss_top = int(ss_offset_stages[k_ss])
                        in_i += float(ss_V_top[k_ss]) * y_actual[ss_top, i]

                rxn_src = 0.0
                if (j + 1) in reactive_set and R > 0:
                    for r in range(R):
                        rxn_src += nu_full[r, i] * xi_arr[j, r]

                F_vec[off + i] = out_i - in_i - rxn_src

            # Bubble-point closure: Sum_i K x = 1 (the LIQUID leaving the
            # stage is always at its bubble point; Murphree efficiency
            # only changes the actual vapor composition leaving)
            F_vec[off + C] = float((K_arr[j] * x_arr[j]).sum() - 1.0)

            # Chemistry on reactive stages
            if (j + 1) in reactive_set and R > 0:
                log_gx = (np.log(np.maximum(gamma_arr[j], 1e-30))
                          + np.log(np.maximum(x_arr[j], 1e-30)))
                for r in range(R):
                    ln_Ka = float((nu_full[r] * log_gx).sum())
                    F_vec[off + C + 1 + r] = (
                        ln_Ka - reactions[r].ln_K_eq(T_arr[j]))

        # ---- Side-stripper stages ----
        # CMO assumption: liquid flow F_SS down through stages 0..n_ss-1,
        # vapor flow V_top up through stages 1..n_ss-1.  Stage 0 (top)
        # receives liquid from main column; bottom stage discharges
        # bottoms_rate as side product.  In steam mode, live steam is
        # injected at the bottom stage at composition ss_steam_z[k] and
        # rate ss_steam_flow[k].
        for k_ss in range(n_SS):
            ss_n = int(ss_n_stages[k_ss])
            ss_F = float(ss_flow[k_ss])
            ss_B = float(ss_bottoms[k_ss])
            ss_V = float(ss_V_top[k_ss])
            ss_main_draw = int(ss_draw[k_ss]) - 1     # Python idx in main
            base_idx = int(ss_offset_stages[k_ss])    # Python idx of SS stage 0
            is_steam = (ss_mode[k_ss] == "steam")
            steam_F = float(ss_steam_flow[k_ss]) if is_steam else 0.0
            steam_z_k = ss_steam_z[k_ss] if is_steam else None

            for k_local in range(ss_n):
                j_full = base_idx + k_local           # global stage index
                off = var_offsets[j_full]

                for i in range(C):
                    # Liquid in/out (CMO): all SS stages have L = ss_F flowing
                    # in from above.
                    if k_local == 0:
                        liq_in = ss_F * x_arr[ss_main_draw, i]
                    else:
                        liq_in = ss_F * x_arr[j_full - 1, i]
                    if k_local == ss_n - 1:
                        liq_out_internal = 0.0
                        bot_out = ss_B * x_arr[j_full, i]
                    else:
                        liq_out_internal = ss_F * x_arr[j_full, i]
                        bot_out = 0.0

                    # Vapor in/out.  Bottom stage in REBOIL mode has no
                    # vapor entering from below (implicit reboiler creates
                    # ss_V from the liquid pool).  In STEAM mode, the
                    # steam entering AS VAPOR at composition steam_z is
                    # the vapor "from below" with flow steam_F.
                    if k_local == ss_n - 1:
                        if is_steam:
                            vap_in = steam_F * steam_z_k[i]
                        else:
                            vap_in = 0.0
                    else:
                        vap_in = ss_V * y_actual[j_full + 1, i]
                    vap_out = ss_V * y_actual[j_full, i]

                    F_vec[off + i] = (liq_out_internal + bot_out + vap_out
                                       - liq_in - vap_in)

                # Bubble-point closure
                F_vec[off + C] = float(
                    (K_arr[j_full] * x_arr[j_full]).sum() - 1.0)

        return F_vec

    # Initialize state.  Bootstrap SS stages from main column draw stage:
    # each SS stage starts with the same x as its associated draw stage,
    # and a temperature that linearly interpolates between the draw-stage
    # T (top) and ~10 K higher (bottom) — a stripping zone runs slightly
    # hotter than the draw because the lighter components are removed.
    if xi_init is None:
        xi_init = np.zeros((n_stages, R))
    x_init_full = np.zeros((n_total_stages, C))
    T_init_full = np.zeros(n_total_stages)
    x_init_full[:n_stages] = np.asarray(x_init, dtype=float)
    T_init_full[:n_stages] = np.asarray(T_init, dtype=float)
    for k in range(n_SS):
        ss_n = int(ss_n_stages[k])
        d_idx = int(ss_draw[k]) - 1
        base = int(ss_offset_stages[k])
        for k_local in range(ss_n):
            x_init_full[base + k_local] = x_init[d_idx]
            T_init_full[base + k_local] = T_init[d_idx] + 10.0 * (
                k_local / max(ss_n - 1, 1))
    w = pack(x_init_full, T_init_full, np.asarray(xi_init, dtype=float))

    F_curr = residuals(w)
    norm_curr = float(np.max(np.abs(F_curr)))
    converged = norm_curr < tol
    last_iter = 0

    # Detect non-local coupling that breaks block-tridiagonal sparsity:
    # pump-arounds and side-strippers connect non-adjacent stages.
    # Murphree efficiency E < 1 introduces a recursive dependency
    # y_actual[j] = E·K·x + (1-E)·y_actual[j+1] that propagates all the
    # way down the column — equivalent to a full upper-triangular block
    # in the Jacobian.  Detect with E ≠ 1 anywhere.
    has_nonlocal = (n_PA > 0) or (n_SS > 0) or bool(np.any(E_arr < 1.0))

    for newton_iter in range(max_newton_iter):
        last_iter = newton_iter
        if verbose:
            print(f"  N-S iter {newton_iter}: ||F||_inf = {norm_curr:.3e}")
        if norm_curr < tol:
            converged = True
            break

        # Block-tridiagonal CPR-compressed Jacobian (or dense fallback
        # when pump-arounds / side-strippers break locality).
        J = _build_block_tridiag_jacobian(
            residuals, w, F_curr, var_offsets,
            n_stages_main=n_stages,
            n_total_stages=n_total_stages,
            fd_step=fd_step,
            has_nonlocal_coupling=has_nonlocal,
        )

        # Solve J dw = -F
        try:
            dw = np.linalg.solve(J, -F_curr)
        except np.linalg.LinAlgError:
            dw, *_ = np.linalg.lstsq(J, -F_curr, rcond=None)

        # Backtracking line search (Armijo with c=1e-4)
        alpha = 1.0
        w_new = w + alpha * dw
        F_new = residuals(w_new)
        norm_new = float(np.max(np.abs(F_new)))
        for _ in range(20):
            if norm_new < (1.0 - 1e-4 * alpha) * norm_curr or alpha < 1e-8:
                break
            alpha *= 0.5
            w_new = w + alpha * dw
            F_new = residuals(w_new)
            norm_new = float(np.max(np.abs(F_new)))

        w = w_new
        F_curr = F_new
        norm_curr = norm_new

    # Recompute final state for return
    x_arr_full, T_arr_full, xi_arr = unpack(w)
    L, V, B_local = get_flows(xi_arr)
    # Slice out main-column-only arrays for the result fields
    x_arr = x_arr_full[:n_stages].copy()
    T_arr = T_arr_full[:n_stages].copy()
    K_arr = np.zeros((n_stages, C))
    for j in range(n_stages):
        if use_eos:
            K_j, _, _ = _stage_K_gamma_phi_eos(
                T_arr[j], p_arr[j], x_arr[j],
                activity_model, psat_funcs, vapor_eos,
                V_L_arr=V_L_arr, phi_sat_funcs=phi_sat_funcs)
            K_arr[j] = K_j
        else:
            psat = np.array([f(T_arr[j]) for f in psat_funcs])
            x_safe = np.maximum(x_arr[j], 1e-30)
            x_norm = x_safe / x_safe.sum()
            gamma = np.asarray(activity_model.gammas(T_arr[j], x_norm))
            K_arr[j] = gamma * psat / p_arr[j]
    # Report the actual leaving-vapor composition (Murphree-modified).
    # For E=1 everywhere this is the equilibrium K * x.
    y_arr = _compute_y_actual(K_arr, x_arr, E_arr)
    y_sums = y_arr.sum(axis=1, keepdims=True)
    y_sums = np.where(y_sums < 1e-12, 1.0, y_sums)
    y_arr = y_arr / y_sums

    # Build SS result dicts
    ss_results = []
    for k in range(n_SS):
        ss_n = int(ss_n_stages[k])
        base = int(ss_offset_stages[k])
        ss_p = float(ss_pressure[k])
        ss_x = x_arr_full[base:base + ss_n].copy()
        ss_T = T_arr_full[base:base + ss_n].copy()
        # K and y for SS stages
        ss_K = np.zeros((ss_n, C))
        for k_local in range(ss_n):
            T_k = ss_T[k_local]
            if use_eos:
                K_j, _, _ = _stage_K_gamma_phi_eos(
                    T_k, ss_p, ss_x[k_local],
                    activity_model, psat_funcs, vapor_eos,
                    V_L_arr=V_L_arr, phi_sat_funcs=phi_sat_funcs)
                ss_K[k_local] = K_j
            else:
                psat = np.array([f(T_k) for f in psat_funcs])
                x_safe = np.maximum(ss_x[k_local], 1e-30)
                x_norm = x_safe / x_safe.sum()
                gamma = np.asarray(activity_model.gammas(T_k, x_norm))
                ss_K[k_local] = gamma * psat / ss_p
        ss_y = ss_K * ss_x
        ss_y_sums = ss_y.sum(axis=1, keepdims=True)
        ss_y_sums = np.where(ss_y_sums < 1e-12, 1.0, ss_y_sums)
        ss_y = ss_y / ss_y_sums
        ss_results.append({
            "draw_stage": int(ss_draw[k]),
            "return_stage": int(ss_return[k]),
            "n_stages": ss_n,
            "flow": float(ss_flow[k]),
            "bottoms_rate": float(ss_bottoms[k]),
            "pressure": ss_p,
            "stripping_mode": str(ss_mode[k]),
            "steam_flow": float(ss_steam_flow[k]),
            "steam_z": ss_steam_z[k].copy() if n_SS > 0 else None,
            "steam_T": float(ss_steam_T[k]),
            "T": ss_T,
            "x": ss_x,
            "y": ss_y,
            "x_bottoms": ss_x[-1].copy(),
        })

    msg = (f"N-S converged in {last_iter + 1} Newton iters, "
           f"||F||={norm_curr:.2e}"
           if converged else
           f"N-S did not converge in {max_newton_iter} iters, "
           f"||F||={norm_curr:.2e}")

    # Build feeds tuple and draw tuples for the result
    feeds_out = tuple(
        FeedSpec(stage=int(feeds_stage[k]),
                 F=float(feeds_F[k]),
                 z=feeds_z[k].copy().tolist(),
                 T=298.15,
                 q=float(feeds_q[k]))
        for k in range(K_feeds))
    liquid_draws_out = tuple(
        (j + 1, float(liquid_draws_arr[j])) for j in range(n_stages)
        if liquid_draws_arr[j] > 0)
    vapor_draws_out = tuple(
        (j + 1, float(vapor_draws_arr[j])) for j in range(n_stages)
        if vapor_draws_arr[j] > 0)

    return ColumnResult(
        converged=converged, iterations=last_iter + 1,
        n_stages=n_stages, species_names=tuple(species_names),
        T=T_arr.copy(), p=p_arr.copy(), L=L.copy(), V=V.copy(),
        x=x_arr.copy(), y=y_arr.copy(), xi=xi_arr.copy(),
        D=D, B=B_local,
        feed_stage=int(feeds_stage[0]),
        feed_F=float(feeds_F[0]), feed_z=feeds_z[0].copy(),
        reflux_ratio=reflux_ratio,
        reactive_stages=tuple(sorted(reactive_set)),
        message=msg,
        feeds=feeds_out,
        liquid_draws=liquid_draws_out,
        vapor_draws=vapor_draws_out,
        side_strippers=tuple(ss_results),
    )


def _naphtali_sandholm_solve_with_energy(
    n_stages: int,
    feeds_stage: np.ndarray,        # (K,) 1-indexed
    feeds_F: np.ndarray,            # (K,)
    feeds_z: np.ndarray,            # (K, C)
    feeds_T: np.ndarray,            # (K,)
    feeds_q: np.ndarray,            # (K,) liquid fraction
    liquid_draws_arr: np.ndarray,   # (n_stages,)
    vapor_draws_arr: np.ndarray,    # (n_stages,)
    distillate_rate: float,
    reflux_ratio: float,
    p_arr: np.ndarray,              # (n_stages,) per-stage pressure
    species_names: list,
    activity_model,
    psat_funcs: Sequence[Callable[[float], float]],
    reactions: Sequence[LiquidPhaseReaction],
    reactive_set: set,
    species_idx_in_rxn: List[List[int]],
    h_V_funcs: Sequence[Callable[[float], float]],
    h_L_funcs: Sequence[Callable[[float], float]],
    T_init: np.ndarray,
    x_init: np.ndarray,
    V_init: Optional[np.ndarray] = None,
    L_init: Optional[np.ndarray] = None,
    xi_init: Optional[np.ndarray] = None,
    E_arr: Optional[np.ndarray] = None,    # (n_stages,) Murphree eff
    pa_draw: Optional[np.ndarray] = None,    # (P,) Python-indexed
    pa_return: Optional[np.ndarray] = None,  # (P,) Python-indexed
    pa_flow: Optional[np.ndarray] = None,    # (P,)
    pa_dT: Optional[np.ndarray] = None,      # (P,) cooling [K]
    pa_L_add: Optional[np.ndarray] = None,   # (n_stages,) extra L per stage
    ss_draw: Optional[np.ndarray] = None,    # (S,) 1-indexed draw stages
    ss_return: Optional[np.ndarray] = None,  # (S,) 1-indexed return stages
    ss_n_stages: Optional[np.ndarray] = None,# (S,) stages per side stripper
    ss_flow: Optional[np.ndarray] = None,    # (S,) liquid feed to each SS
    ss_bottoms: Optional[np.ndarray] = None, # (S,) side product flow per SS
    ss_pressure: Optional[np.ndarray] = None,# (S,) SS pressure
    ss_mode: Optional[np.ndarray] = None,    # (S,) "reboil" / "steam"
    ss_steam_flow: Optional[np.ndarray] = None,
    ss_steam_z: Optional[np.ndarray] = None,
    ss_steam_T: Optional[np.ndarray] = None,
    vapor_eos = None,
    V_L_arr: Optional[np.ndarray] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    max_newton_iter: int = 30,
    tol: float = 1e-7,
    fd_step: float = 1e-7,
    verbose: bool = False,
) -> 'ColumnResult':
    """Naphtali-Sandholm with per-stage energy balance.

    Drops constant-molar-overflow.  Per stage j the unknowns are:

        Stage 0  (top, V_top fixed):   x (C), T, L,    xi (R)
        Interior stages:                x (C), T, V, L, xi (R)
        Stage N-1 (reboiler, L=B):     x (C), T, V,    xi (R)

    Per stage residuals:

        Stage 0/N-1:   M (C), bubble-pt, Sum_x = 1,            chemistry (R)
        Interior:      M (C), bubble-pt, Sum_x = 1, H_balance, chemistry (R)

    Q_C and Q_R are derived post-solve from boundary enthalpy balances;
    they're not enforced as residuals.  This keeps the per-stage system
    square: the missing H equation at each boundary is exactly balanced
    by the missing flow variable (V_top fixed at top, L_N=B fixed at
    bottom).

    Enthalpies use ideal mixing:  h_L(T,x) = sum_i x_i h_L_i(T),
                                 h_V(T,y) = sum_i y_i h_V_i(T).
    Reaction heat:  rxn_h = sum_r (-dH_rxn[r]) * xi_{j,r}.
    Feed assumed liquid at feed_T.

    The H residual is scaled by 1/(feed_F * 1e4) so its magnitude is
    O(1), matching the other residuals for Jacobian conditioning.
    """
    C = len(species_names)
    R = len(reactions)
    K_feeds = len(feeds_F)

    # Cumulative feeds and draws for L/V profile bookkeeping (q-aware)
    cum_qF = np.zeros(n_stages + 1)
    cum_vapF = np.zeros(n_stages + 1)
    for k_idx in range(K_feeds):
        s = int(feeds_stage[k_idx])
        F_k = float(feeds_F[k_idx])
        q_k = float(feeds_q[k_idx])
        cum_qF[s:]   += q_k * F_k
        cum_vapF[s:] += (1.0 - q_k) * F_k
    cum_U = np.zeros(n_stages + 1)
    cum_W = np.zeros(n_stages + 1)
    for j in range(1, n_stages + 1):
        cum_U[j] = cum_U[j - 1] + float(liquid_draws_arr[j - 1])
        cum_W[j] = cum_W[j - 1] + float(vapor_draws_arr[j - 1])
    total_F = float(feeds_F.sum())
    total_U = float(liquid_draws_arr.sum())
    total_W = float(vapor_draws_arr.sum())

    # Per-stage feed mole-source and enthalpy-source.
    # Feed enthalpy: q*h_L(T_F) + (1-q)*h_V(T_F).  For q outside [0,1]
    # this still gives the right value under CMO (a subcooled liquid has
    # q>1 and the formula gives a smaller h_F than h_L; a superheated
    # vapor has q<0 and gives a larger h_F than h_V, both consistent
    # with q being defined as the liquid mole fraction times unit).
    feed_src = np.zeros((n_stages, C))
    feed_h_src = np.zeros(n_stages)
    for k_idx in range(K_feeds):
        s = int(feeds_stage[k_idx]) - 1
        F_k = float(feeds_F[k_idx]); z_k = feeds_z[k_idx]
        T_k = float(feeds_T[k_idx]); q_k = float(feeds_q[k_idx])
        feed_src[s] += F_k * z_k
        h_L_k = sum(z_k[i] * h_L_funcs[i](T_k) for i in range(C))
        h_V_k = sum(z_k[i] * h_V_funcs[i](T_k) for i in range(C))
        h_F_k = q_k * h_L_k + (1.0 - q_k) * h_V_k
        feed_h_src[s] += F_k * h_F_k

    # Pump-around setup (defaults: empty)
    if pa_draw is None or pa_flow is None or len(pa_flow) == 0:
        pa_draw = np.zeros(0, dtype=int)
        pa_return = np.zeros(0, dtype=int)
        pa_flow = np.zeros(0, dtype=float)
        pa_dT = np.zeros(0, dtype=float)
        pa_L_add = np.zeros(n_stages, dtype=float)
    else:
        if pa_dT is None:
            pa_dT = np.zeros(len(pa_flow), dtype=float)
        if pa_L_add is None:
            pa_L_add = np.zeros(n_stages, dtype=float)
            for k in range(len(pa_flow)):
                for j in range(int(pa_return[k]), int(pa_draw[k])):
                    pa_L_add[j] += float(pa_flow[k])
    n_PA = len(pa_flow)

    # Side-stripper setup (v0.9.89: EB + SS).  SS stages are appended to
    # the variable layout as [x, T, V, L] -- 4 unknowns each.  The
    # residuals are: M (C component balances), bubble-pt, sum_x = 1, H_balance.
    if ss_draw is None or ss_flow is None or len(ss_flow) == 0:
        ss_draw     = np.zeros(0, dtype=int)
        ss_return   = np.zeros(0, dtype=int)
        ss_n_stages = np.zeros(0, dtype=int)
        ss_flow     = np.zeros(0, dtype=float)
        ss_bottoms  = np.zeros(0, dtype=float)
        ss_pressure = np.zeros(0, dtype=float)
    n_SS = len(ss_flow)
    if ss_mode is None:
        ss_mode = np.array(["reboil"] * n_SS, dtype=object)
    if ss_steam_flow is None:
        ss_steam_flow = np.zeros(n_SS, dtype=float)
    if ss_steam_z is None:
        ss_steam_z = np.zeros((n_SS, max(C, 1)), dtype=float)
    if ss_steam_T is None:
        ss_steam_T = np.zeros(n_SS, dtype=float)
    n_ss_total = int(ss_n_stages.sum()) if n_SS > 0 else 0
    ss_offset_stages = np.zeros(n_SS, dtype=int)
    for k in range(n_SS):
        ss_offset_stages[k] = n_stages + (
            int(ss_n_stages[:k].sum()) if k > 0 else 0)
    ss_bottoms_total = float(ss_bottoms.sum()) if n_SS > 0 else 0.0
    ss_steam_total = float(ss_steam_flow.sum()) if n_SS > 0 else 0.0

    nu_full = np.zeros((R, C))
    for r in range(R):
        for sp_idx, nu_local in enumerate(reactions[r].nu):
            nu_full[r, species_idx_in_rxn[r][sp_idx]] = nu_local
    dn_per_rxn = (np.array([float(rxn.nu.sum()) for rxn in reactions])
                  if R > 0 else np.zeros(0))
    dH_rxn_arr = (np.array([float(getattr(rxn, 'dH_rxn', 0.0))
                            for rxn in reactions]) if R > 0 else np.zeros(0))

    D = float(distillate_rate)
    L_top = reflux_ratio * D
    V_top = (reflux_ratio + 1.0) * D
    H_scale = max(total_F, 1.0) * 1e4

    # ---------- variable bookkeeping ----------
    # Per-stage layout of unknowns:
    #   stage 0   : x[0..C-1], T, L, xi[0..R-1]                 (C + 2 + R_j)
    #   interior  : x[0..C-1], T, V, L, xi[0..R-1]              (C + 3 + R_j)
    #   stage N-1 : x[0..C-1], T, V, xi[0..R-1]                 (C + 2 + R_j)
    # SS stages (v0.9.89, no chemistry):
    #   SS top    : x[0..C-1], T, V                             (C + 2)
    #   SS interior: x[0..C-1], T, V, L                          (C + 3)
    #   SS bottom : x[0..C-1], T, L                             (C + 2)
    # where R_j = R if reactive, else 0.
    # For a 1-stage SS the single stage acts as both top and bottom; we
    # use [x, T, V] (only V is needed because L_in = ss_flow is fixed).
    n_total_stages = n_stages + n_ss_total
    n_vars_per_stage: list[int] = []
    for j in range(n_stages):
        is_reactive = (j + 1) in reactive_set
        if j == 0 or j == n_stages - 1:
            base = C + 2
        else:
            base = C + 3
        n_vars_per_stage.append(base + (R if is_reactive else 0))
    # SS stage layouts.  The number of FREE unknowns per SS stage is:
    #   single: [x, T, V]    → C + 2 free vars (L_in fixed = ss_flow,
    #                          L_out fixed = ss_bottoms; V is free)
    #   top:    [x, T, V]    → C + 2 free vars (L_in fixed = ss_flow,
    #                          L_out is the L of the next SS stage = free,
    #                          stored on the next SS stage's L slot)
    #   int:    [x, T, V, L] → C + 3 free vars (V and L_out free)
    #   bot:    [x, T]       → C + 1 free vars (L_in is the L of the
    #                          previous int's L; L_out fixed = ss_bottoms;
    #                          V_out is free... wait, that's 3 free.
    #                          Actually V_out is free for bot too: it
    #                          feeds into the int stage above as V_below.
    #                          So bot is [x, T, V] = C+2.)
    #
    # Re-deriving systematically: degrees of freedom per SS stage:
    #   - x (C-1 due to sum_x = 1 implicit -- but we treat all C as vars
    #     and add sum_x = 1 as a residual)
    #   - T
    #   - V_out (vapor leaving upward) — free unless fixed by config
    #   - L_out (liquid leaving downward) — free unless fixed by config
    # Constraints per SS stage:
    #   single (only stage): L_in fixed = ss_flow, L_out fixed = ss_bottoms.
    #     V_out is free.
    #   top (k=0):    L_in fixed = ss_flow.  V_out and L_out free.
    #   int:          all four free (V_out, L_out are interior flows).
    #   bot (k=n-1):  L_out fixed = ss_bottoms.  V_out and L_in free
    #                 (L_in comes from int above as L_arr[j_full-1]).
    #     But V_out for bot also feeds the int above as V_below.
    #
    # So each kind has free flows:
    #   single: V_out only         → C + 2 vars (x[C], T, V)
    #   top:    V_out, L_out       → C + 3 vars (x[C], T, V, L)
    #   int:    V_out, L_out       → C + 3 vars (x[C], T, V, L)
    #   bot:    V_out only         → C + 2 vars (x[C], T, V)
    #
    # Equations per stage (all stages): M (C) + BP + sum_x = C + 2.
    # Energy balance is added on int and top (where V_out is determined
    # by the energy balance as one of the unknowns interacts with L_out).
    # Wait, let me re-derive equation count to match the var count.
    #
    # Square system requires:
    #   single (C+2 vars): C + 2 eqns → M + BP + sum_x.  EB is implicit
    #     in the single stage's V_out-determined-by-feed structure: with
    #     L_in, L_out, and steam_in fixed, V_out is determined by mass
    #     balance.  So drop EB on single.  ✓
    #   top    (C+3 vars): C + 3 eqns → M + BP + sum_x + EB  ✓
    #   int    (C+3 vars): C + 3 eqns → M + BP + sum_x + EB  ✓
    #   bot    (C+2 vars): C + 2 eqns → M + BP + sum_x.  V_out is set
    #     by mass balance (L_in - L_out + steam_in = V_out) implicitly
    #     within the M residuals.  ✓
    ss_var_kinds: list[str] = []
    for k in range(n_SS):
        n_ss = int(ss_n_stages[k])
        for k_local in range(n_ss):
            if n_ss == 1:
                kind = "single"
                base = C + 2
            elif k_local == 0:
                kind = "top"
                base = C + 3
            elif k_local == n_ss - 1:
                kind = "bot"
                base = C + 2
            else:
                kind = "int"
                base = C + 3
            ss_var_kinds.append(kind)
            n_vars_per_stage.append(base)
    var_offsets = np.array([0] + list(np.cumsum(n_vars_per_stage)))
    n_total = int(var_offsets[-1])

    def _slot_x(j):
        return var_offsets[j], var_offsets[j] + C

    def _slot_T(j):
        return var_offsets[j] + C

    def _slot_V(j):
        # Main column: V is variable on stages 1..N-1, absent on stage 0.
        # SS column: V is variable on every stage (top/int/bot/single).
        if j < n_stages:
            if j == 0:
                return None
            return var_offsets[j] + C + 1
        return var_offsets[j] + C + 1

    def _slot_L(j):
        # Main column: L is variable on stages 0..N-2, absent on stage N-1.
        # SS column: L is variable on top/int (V_out and L_out both free);
        # bot and single have no L slot (L_out is pinned to ss_bottoms).
        if j < n_stages:
            if j == n_stages - 1:
                return None
            if j == 0:
                return var_offsets[j] + C + 1     # no V slot ahead of it
            return var_offsets[j] + C + 2
        # SS stage
        kind = ss_var_kinds[j - n_stages]
        if kind in ("bot", "single"):
            return None
        return var_offsets[j] + C + 2

    def _slot_xi(j):
        # Chemistry only on main column reactive stages
        if j >= n_stages:
            return None
        is_reactive = (j + 1) in reactive_set
        if not is_reactive or R == 0:
            return None
        if j == 0 or j == n_stages - 1:
            start = var_offsets[j] + C + 2
        else:
            start = var_offsets[j] + C + 3
        return start, start + R

    def unpack(w: np.ndarray):
        x = np.zeros((n_total_stages, C))
        T = np.zeros(n_total_stages)
        V = np.zeros(n_total_stages)
        L = np.zeros(n_total_stages)
        xi = np.zeros((n_stages, R))
        V[0] = V_top
        for j in range(n_total_stages):
            xs, xe = _slot_x(j)
            x[j] = w[xs:xe]
            T[j] = w[_slot_T(j)]
            sV = _slot_V(j)
            if sV is not None:
                V[j] = w[sV]
            sL = _slot_L(j)
            if sL is not None:
                L[j] = w[sL]
            if j < n_stages:
                sxi = _slot_xi(j)
                if sxi is not None:
                    xi[j] = w[sxi[0]:sxi[1]]
        # Bottoms by overall mass balance:
        #   B = total_F - D - total_U - total_W + dN_rxn
        #       - ss_bottoms_total + ss_steam_total
        if R > 0:
            dN_rxn = float((xi * dn_per_rxn).sum())
        else:
            dN_rxn = 0.0
        B_local = (total_F + dN_rxn - D - total_U - total_W
                   - ss_bottoms_total + ss_steam_total)
        L[n_stages - 1] = B_local
        # SS L pinning: bot and single have no L slot in w; pin L_out
        # to ss_bottoms.  top/int have L slot which Newton solves for.
        for k in range(n_SS):
            base = int(ss_offset_stages[k])
            n_ss = int(ss_n_stages[k])
            kind_top = ss_var_kinds[ss_offset_stages[k] - n_stages]
            # For single (n_ss==1) or bot (last stage of multi-stage SS),
            # the L_out is pinned to ss_bottoms.
            if n_ss == 1:
                L[base] = float(ss_bottoms[k])
            else:
                L[base + n_ss - 1] = float(ss_bottoms[k])
        return x, T, V, L, xi, B_local

    def pack(x, T, V, L, xi):
        w = np.zeros(n_total)
        for j in range(n_total_stages):
            xs, xe = _slot_x(j)
            w[xs:xe] = x[j]
            w[_slot_T(j)] = T[j]
            sV = _slot_V(j)
            if sV is not None:
                w[sV] = V[j]
            sL = _slot_L(j)
            if sL is not None:
                w[sL] = L[j]
            if j < n_stages:
                sxi = _slot_xi(j)
                if sxi is not None:
                    w[sxi[0]:sxi[1]] = xi[j]
        return w

    if E_arr is None:
        E_arr = np.ones(n_stages, dtype=float)

    use_eos = vapor_eos is not None

    def residuals(w: np.ndarray) -> np.ndarray:
        x_arr, T_arr, V_arr, L_arr, xi_arr, B_local = unpack(w)

        K_arr = np.zeros((n_total_stages, C))
        gamma_arr = np.zeros((n_total_stages, C))
        h_L_arr = np.zeros(n_total_stages)
        h_V_arr = np.zeros(n_total_stages)
        # Per-stage pressure: main + SS
        p_full = np.zeros(n_total_stages)
        p_full[:n_stages] = p_arr
        for k in range(n_SS):
            p_full[ss_offset_stages[k]:ss_offset_stages[k]
                   + int(ss_n_stages[k])] = float(ss_pressure[k])

        for j in range(n_total_stages):
            T_j = T_arr[j]
            if use_eos:
                K_j, gamma_j, _ = _stage_K_gamma_phi_eos(
                    T_j, p_full[j], x_arr[j],
                    activity_model, psat_funcs, vapor_eos,
                    V_L_arr=V_L_arr, phi_sat_funcs=phi_sat_funcs)
                K_arr[j] = K_j
                gamma_arr[j] = gamma_j
            else:
                psat = np.array([f(T_j) for f in psat_funcs])
                x_safe = np.maximum(x_arr[j], 1e-30)
                x_norm = x_safe / x_safe.sum()
                gamma = np.asarray(activity_model.gammas(T_j, x_norm))
                gamma_arr[j] = gamma
                K_arr[j] = gamma * psat / p_full[j]
            h_L_arr[j] = sum(x_arr[j, i] * h_L_funcs[i](T_j) for i in range(C))

        # Actual leaving-vapor composition (Murphree-modified).  E_arr is
        # length n_stages (main only); SS stages get E=1 implicitly.
        E_full = np.ones(n_total_stages)
        E_full[:n_stages] = E_arr
        y_actual = _compute_y_actual(K_arr, x_arr, E_full)
        # Vapor enthalpy at the ACTUAL leaving composition
        for j in range(n_total_stages):
            T_j = T_arr[j]
            h_V_arr[j] = sum(y_actual[j, i] * h_V_funcs[i](T_j)
                             for i in range(C))

        F_vec = np.zeros(n_total)

        for j in range(n_stages):
            off = var_offsets[j]
            is_reactive = (j + 1) in reactive_set
            is_interior = (0 < j < n_stages - 1)

            # ---- Component balance per species i (rows: off..off+C-1)
            for i in range(C):
                if j == 0:
                    out_i = D * y_actual[0, i] + L_arr[0] * x_arr[0, i]
                    in_i = (V_arr[1] * y_actual[1, i]
                            if n_stages > 1 else 0.0)
                elif j == n_stages - 1:
                    out_i = V_arr[j] * y_actual[j, i] + B_local * x_arr[j, i]
                    in_i = L_arr[j - 1] * x_arr[j - 1, i]
                else:
                    out_i = V_arr[j] * y_actual[j, i] + L_arr[j] * x_arr[j, i]
                    in_i = (L_arr[j - 1] * x_arr[j - 1, i]
                            + V_arr[j + 1] * y_actual[j + 1, i])
                # Multi-feed contribution
                in_i += feed_src[j, i]
                # Side draws
                out_i += (liquid_draws_arr[j] * x_arr[j, i]
                          + vapor_draws_arr[j] * y_actual[j, i])

                # Pump-around mass balance: at j_return get +flow * x[j_draw, i]
                # IN; at j_draw get +flow * x[j, i] OUT.
                for k_pa in range(n_PA):
                    if j == int(pa_return[k_pa]):
                        d_idx = int(pa_draw[k_pa])
                        in_i += float(pa_flow[k_pa]) * x_arr[d_idx, i]
                    if j == int(pa_draw[k_pa]):
                        out_i += float(pa_flow[k_pa]) * x_arr[j, i]

                # Side-stripper coupling (v0.9.89): at SS draw stage,
                # ss_flow leaves as liquid; at SS return stage, the SS
                # top vapor (V at SS top stage) enters.  Both 1-indexed.
                for k_ss in range(n_SS):
                    if (j + 1) == int(ss_draw[k_ss]):
                        out_i += float(ss_flow[k_ss]) * x_arr[j, i]
                    if (j + 1) == int(ss_return[k_ss]):
                        ss_top = int(ss_offset_stages[k_ss])
                        # V_arr[ss_top] is the SS top stage's vapor outflow,
                        # which feeds back into the main column at this stage
                        in_i += V_arr[ss_top] * y_actual[ss_top, i]

                rxn_src = 0.0
                if is_reactive and R > 0:
                    for r in range(R):
                        rxn_src += nu_full[r, i] * xi_arr[j, r]

                F_vec[off + i] = out_i - in_i - rxn_src

            # ---- Bubble-point closure  (residual row: off + C)
            F_vec[off + C] = float((K_arr[j] * x_arr[j]).sum() - 1.0)

            # ---- Liquid-side closure Sum_i x = 1   (residual row: off + C + 1)
            F_vec[off + C + 1] = float(x_arr[j].sum() - 1.0)

            # ---- Energy balance (interior stages only; row: off + C + 2)
            if is_interior:
                in_h = (L_arr[j - 1] * h_L_arr[j - 1]
                        + V_arr[j + 1] * h_V_arr[j + 1])
                # Multi-feed enthalpy contribution
                in_h += feed_h_src[j]
                out_h = L_arr[j] * h_L_arr[j] + V_arr[j] * h_V_arr[j]
                # Side-draw enthalpy: liquid draw at h_L_j, vapor draw at h_V_j
                out_h += (liquid_draws_arr[j] * h_L_arr[j]
                          + vapor_draws_arr[j] * h_V_arr[j])

                # Pump-around enthalpy.  At j_return, the recycled
                # stream enters at composition x[j_draw] and temperature
                # T[j_draw] - dT (cooled).  At j_draw, the recycled
                # stream leaves at h_L_arr[j_draw] (bubble-point liquid
                # at draw conditions).
                for k_pa in range(n_PA):
                    if j == int(pa_return[k_pa]):
                        d_idx = int(pa_draw[k_pa])
                        flow = float(pa_flow[k_pa])
                        T_ret = float(T_arr[d_idx]) - float(pa_dT[k_pa])
                        h_L_ret = sum(x_arr[d_idx, ii] * h_L_funcs[ii](T_ret)
                                      for ii in range(C))
                        in_h += flow * h_L_ret
                    if j == int(pa_draw[k_pa]):
                        out_h += float(pa_flow[k_pa]) * h_L_arr[j]

                # Side-stripper enthalpy coupling.  At SS draw stage,
                # ss_flow leaves as bubble-point liquid (h_L_arr[j]).
                # At SS return stage, the SS top stage's vapor enters
                # at h_V_arr[ss_top] (the SS top is colder than the
                # main column due to internal stripping).
                for k_ss in range(n_SS):
                    if (j + 1) == int(ss_draw[k_ss]):
                        out_h += float(ss_flow[k_ss]) * h_L_arr[j]
                    if (j + 1) == int(ss_return[k_ss]):
                        ss_top = int(ss_offset_stages[k_ss])
                        in_h += V_arr[ss_top] * h_V_arr[ss_top]

                rxn_h = 0.0
                if is_reactive and R > 0:
                    for r in range(R):
                        rxn_h += -dH_rxn_arr[r] * xi_arr[j, r]
                F_vec[off + C + 2] = (in_h + rxn_h - out_h) / H_scale

            # ---- Chemistry (rows after closure / energy)
            if is_reactive and R > 0:
                xi_off = (off + C + 3) if is_interior else (off + C + 2)
                log_gx = (np.log(np.maximum(gamma_arr[j], 1e-30))
                          + np.log(np.maximum(x_arr[j], 1e-30)))
                for r in range(R):
                    ln_Ka = float((nu_full[r] * log_gx).sum())
                    F_vec[xi_off + r] = ln_Ka - reactions[r].ln_K_eq(T_arr[j])

        # ---- Side-stripper residuals (v0.9.89, EB version) ----
        # For each SS stage we write:
        #   - C component balances
        #   - bubble-point closure
        #   - sum_x = 1
        #   - energy balance
        # SS stage variable layout:
        #   top:    [x, T, V]            -> 4 residual rows (M[C], BP, sumx, EB)
        #   int:    [x, T, V, L]         -> 4 residual rows
        #   bot:    [x, T, L]            -> 4 residual rows
        #   single: [x, T, V]            -> 4 residual rows (top + bot fused)
        # Note: bot has L pinned (= ss_bottoms) so V_at_bot is determined by
        # the bot energy balance (in reboil mode V_at_bot = 0; in steam
        # mode V_at_bot = steam_flow).  We do NOT add V_at_bot as an
        # unknown; we compute it locally from the configuration:
        #   V from below at bot = steam_flow * steam_z (steam mode)
        #                       = 0                    (reboil mode)
        for k_ss in range(n_SS):
            ss_n = int(ss_n_stages[k_ss])
            ss_F = float(ss_flow[k_ss])
            ss_B = float(ss_bottoms[k_ss])
            ss_main_draw = int(ss_draw[k_ss]) - 1
            base_idx = int(ss_offset_stages[k_ss])
            is_steam = (ss_mode[k_ss] == "steam")
            steam_F = float(ss_steam_flow[k_ss]) if is_steam else 0.0
            steam_z_k = ss_steam_z[k_ss] if is_steam else None
            steam_T_k = float(ss_steam_T[k_ss]) if is_steam else 0.0
            # Vapor inflow to SS bottom (from below): live steam in
            # steam mode, zero in reboil mode (vapor is generated by
            # the implicit reboil within the bottom-stage flash).
            if is_steam:
                steam_h_in = sum(steam_z_k[i] * h_V_funcs[i](steam_T_k)
                                  for i in range(C))
            else:
                steam_h_in = 0.0

            for k_local in range(ss_n):
                j_full = base_idx + k_local
                off = var_offsets[j_full]
                kind = ss_var_kinds[j_full - n_stages]

                # Liquid in/out for this SS stage
                if k_local == 0:
                    L_in = ss_F
                    x_above = x_arr[ss_main_draw]
                    h_L_above = sum(x_arr[ss_main_draw, i] * h_L_funcs[i](T_arr[ss_main_draw])
                                     for i in range(C))
                else:
                    L_in = L_arr[j_full - 1]
                    x_above = x_arr[j_full - 1]
                    h_L_above = h_L_arr[j_full - 1]

                # Vapor in/out for this SS stage
                if k_local == ss_n - 1:
                    # bottom stage: vapor "from below" is steam (or 0 in reboil)
                    V_below = steam_F
                    y_below = steam_z_k if is_steam else None
                    h_V_below = steam_h_in
                else:
                    V_below = V_arr[j_full + 1]
                    y_below = y_actual[j_full + 1]
                    h_V_below = h_V_arr[j_full + 1]

                # Liquid leaving this stage (downward)
                if k_local == ss_n - 1:
                    L_out = ss_B   # pinned to bottoms_rate
                else:
                    L_out = L_arr[j_full]

                # Vapor leaving this stage (upward)
                V_out = V_arr[j_full]

                # ---- Component balance per species i ----
                for i in range(C):
                    in_i = L_in * x_above[i]
                    if y_below is not None:
                        in_i += V_below * y_below[i]
                    out_i = L_out * x_arr[j_full, i] + V_out * y_actual[j_full, i]
                    F_vec[off + i] = out_i - in_i

                # ---- Bubble-point closure (always) ----
                F_vec[off + C] = float(
                    (K_arr[j_full] * x_arr[j_full]).sum() - 1.0)

                # ---- Sum_x = 1 (always) ----
                F_vec[off + C + 1] = float(x_arr[j_full].sum() - 1.0)

                # ---- Energy balance (top and interior SS stages) ----
                # Variable count per kind:
                #   single, bot: [x, T, V]    → C + 2 vars; M+BP+sumx = C+2 ✓
                #   top, int   : [x, T, V, L] → C + 3 vars; M+BP+sumx+EB = C+3 ✓
                if kind in ("top", "int"):
                    in_h = L_in * h_L_above + V_below * h_V_below
                    out_h = L_out * h_L_arr[j_full] + V_out * h_V_arr[j_full]
                    F_vec[off + C + 2] = (in_h - out_h) / H_scale

        return F_vec

    # ---------- initial guess ----------
    if xi_init is None:
        xi_init = np.zeros((n_stages, R))
    if V_init is None or L_init is None:
        # Default: CMO + reaction-mole-change, multi-feed/draw/q aware
        L_guess = np.zeros(n_stages)
        V_guess = np.zeros(n_stages)
        for j in range(n_stages):
            phys = j + 1
            L_guess[j] = L_top + cum_qF[phys] - cum_U[phys] + pa_L_add[j]
            V_guess[j] = V_top + cum_W[phys - 1] - cum_vapF[phys - 1]
        if R > 0:
            dN_rxn0 = float((np.asarray(xi_init) * dn_per_rxn).sum())
        else:
            dN_rxn0 = 0.0
        L_guess[-1] = (total_F + dN_rxn0 - D - total_U - total_W
                        - ss_bottoms_total + ss_steam_total)
        # Apply SS draw effect to main column L profile and SS-return effect to V.
        # Draw at stage j_d removes ss_flow from L at and below j_d.
        # Return at stage j_r adds ss_V_top to V at and below j_r.
        for k in range(n_SS):
            j_d = int(ss_draw[k]) - 1
            j_r = int(ss_return[k])  # 1-indexed; below = j >= j_r as Python idx
            if ss_mode[k] == "steam":
                V_top_k = float(ss_flow[k] + ss_steam_flow[k] - ss_bottoms[k])
            else:
                V_top_k = float(ss_flow[k] - ss_bottoms[k])
            for j in range(j_d, n_stages):
                L_guess[j] -= float(ss_flow[k])
            for j in range(j_r, n_stages):
                V_guess[j] -= V_top_k
        if V_init is None:
            V_init = V_guess
        if L_init is None:
            L_init = L_guess

    # Build SS initial state: x and T from main column draw stage,
    # V_ss top ≈ V_top_per_mode, L_ss interior ≈ ss_flow, L_ss bot = ss_bottoms.
    x_init_full = np.zeros((n_total_stages, C))
    T_init_full = np.zeros(n_total_stages)
    V_init_full = np.zeros(n_total_stages)
    L_init_full = np.zeros(n_total_stages)
    x_init_full[:n_stages] = np.asarray(x_init, dtype=float)
    T_init_full[:n_stages] = np.asarray(T_init, dtype=float)
    V_init_full[:n_stages] = np.asarray(V_init, dtype=float)
    L_init_full[:n_stages] = np.asarray(L_init, dtype=float)
    for k in range(n_SS):
        ss_n = int(ss_n_stages[k])
        d_idx = int(ss_draw[k]) - 1
        base = int(ss_offset_stages[k])
        if ss_mode[k] == "steam":
            V_top_k = float(ss_flow[k] + ss_steam_flow[k] - ss_bottoms[k])
        else:
            V_top_k = float(ss_flow[k] - ss_bottoms[k])
        for k_local in range(ss_n):
            x_init_full[base + k_local] = x_init[d_idx]
            T_init_full[base + k_local] = T_init[d_idx] + 10.0 * (
                k_local / max(ss_n - 1, 1))
            V_init_full[base + k_local] = V_top_k
            # Interior and top L: ss_flow.  Bot L is pinned to ss_bottoms
            # at solve time but we still seed it to ss_bottoms.
            if k_local == ss_n - 1:
                L_init_full[base + k_local] = float(ss_bottoms[k])
            else:
                L_init_full[base + k_local] = float(ss_flow[k])

    w = pack(x_init_full, T_init_full, V_init_full, L_init_full,
              np.asarray(xi_init, dtype=float))

    F_curr = residuals(w)
    norm_curr = float(np.max(np.abs(F_curr)))
    converged = norm_curr < tol
    last_iter = 0

    # Detect non-local coupling that breaks block-tridiagonal sparsity
    has_nonlocal = (n_PA > 0) or (n_SS > 0) or bool(np.any(E_arr < 1.0))

    for newton_iter in range(max_newton_iter):
        last_iter = newton_iter
        if verbose:
            print(f"  N-S/E iter {newton_iter}: ||F||_inf = {norm_curr:.3e}")
        if norm_curr < tol:
            converged = True
            break

        # Block-tridiagonal CPR-compressed Jacobian (or dense fallback)
        J = _build_block_tridiag_jacobian(
            residuals, w, F_curr, var_offsets,
            n_stages_main=n_stages,
            n_total_stages=n_total_stages,
            fd_step=fd_step,
            has_nonlocal_coupling=has_nonlocal,
        )

        try:
            dw = np.linalg.solve(J, -F_curr)
        except np.linalg.LinAlgError:
            dw, *_ = np.linalg.lstsq(J, -F_curr, rcond=None)

        # Armijo backtracking
        alpha = 1.0
        w_new = w + alpha * dw
        F_new = residuals(w_new)
        norm_new = float(np.max(np.abs(F_new)))
        for _ in range(20):
            if norm_new < (1.0 - 1e-4 * alpha) * norm_curr or alpha < 1e-8:
                break
            alpha *= 0.5
            w_new = w + alpha * dw
            F_new = residuals(w_new)
            norm_new = float(np.max(np.abs(F_new)))

        w = w_new
        F_curr = F_new
        norm_curr = norm_new

    # ---------- final state ----------
    x_arr_full, T_arr_full, V_arr_full, L_arr_full, xi_arr, B_local = unpack(w)
    # Slice main-column-only arrays
    x_arr = x_arr_full[:n_stages].copy()
    T_arr = T_arr_full[:n_stages].copy()
    V_arr = V_arr_full[:n_stages].copy()
    L_arr = L_arr_full[:n_stages].copy()
    K_arr = np.zeros((n_stages, C))
    for j in range(n_stages):
        if use_eos:
            K_j, _, _ = _stage_K_gamma_phi_eos(
                T_arr[j], p_arr[j], x_arr[j],
                activity_model, psat_funcs, vapor_eos,
                V_L_arr=V_L_arr, phi_sat_funcs=phi_sat_funcs)
            K_arr[j] = K_j
        else:
            psat = np.array([f(T_arr[j]) for f in psat_funcs])
            x_safe = np.maximum(x_arr[j], 1e-30)
            x_norm = x_safe / x_safe.sum()
            gamma = np.asarray(activity_model.gammas(T_arr[j], x_norm))
            K_arr[j] = gamma * psat / p_arr[j]
    y_arr = _compute_y_actual(K_arr, x_arr, E_arr)
    y_sums = y_arr.sum(axis=1, keepdims=True)
    y_sums = np.where(y_sums < 1e-12, 1.0, y_sums)
    y_arr = y_arr / y_sums

    # Build SS result dicts (v0.9.89)
    ss_results = []
    for k in range(n_SS):
        ss_n = int(ss_n_stages[k])
        base = int(ss_offset_stages[k])
        ss_p = float(ss_pressure[k])
        ss_x = x_arr_full[base:base + ss_n].copy()
        ss_T = T_arr_full[base:base + ss_n].copy()
        ss_V_local = V_arr_full[base:base + ss_n].copy()
        ss_L_local = L_arr_full[base:base + ss_n].copy()
        ss_K = np.zeros((ss_n, C))
        for k_local in range(ss_n):
            T_k = ss_T[k_local]
            if use_eos:
                K_j, _, _ = _stage_K_gamma_phi_eos(
                    T_k, ss_p, ss_x[k_local],
                    activity_model, psat_funcs, vapor_eos,
                    V_L_arr=V_L_arr, phi_sat_funcs=phi_sat_funcs)
                ss_K[k_local] = K_j
            else:
                psat = np.array([f(T_k) for f in psat_funcs])
                x_safe = np.maximum(ss_x[k_local], 1e-30)
                x_norm = x_safe / x_safe.sum()
                gamma = np.asarray(activity_model.gammas(T_k, x_norm))
                ss_K[k_local] = gamma * psat / ss_p
        ss_y = ss_K * ss_x
        ss_y_sums = ss_y.sum(axis=1, keepdims=True)
        ss_y_sums = np.where(ss_y_sums < 1e-12, 1.0, ss_y_sums)
        ss_y = ss_y / ss_y_sums
        ss_results.append({
            "draw_stage": int(ss_draw[k]),
            "return_stage": int(ss_return[k]),
            "n_stages": ss_n,
            "flow": float(ss_flow[k]),
            "bottoms_rate": float(ss_bottoms[k]),
            "pressure": ss_p,
            "stripping_mode": str(ss_mode[k]),
            "steam_flow": float(ss_steam_flow[k]),
            "steam_z": ss_steam_z[k].copy(),
            "steam_T": float(ss_steam_T[k]),
            "T": ss_T,
            "x": ss_x,
            "y": ss_y,
            "V": ss_V_local,
            "L": ss_L_local,
            "x_bottoms": ss_x[-1].copy(),
        })

    msg = (f"N-S/E converged in {last_iter + 1} Newton iters, "
           f"||F||={norm_curr:.2e}"
           if converged else
           f"N-S/E did not converge in {max_newton_iter} iters, "
           f"||F||={norm_curr:.2e}")

    # Build feeds tuple and draw tuples for the result
    feeds_out = tuple(
        FeedSpec(stage=int(feeds_stage[k]),
                 F=float(feeds_F[k]),
                 z=feeds_z[k].copy().tolist(),
                 T=float(feeds_T[k]),
                 q=float(feeds_q[k]))
        for k in range(K_feeds))
    liquid_draws_out = tuple(
        (j + 1, float(liquid_draws_arr[j])) for j in range(n_stages)
        if liquid_draws_arr[j] > 0)
    vapor_draws_out = tuple(
        (j + 1, float(vapor_draws_arr[j])) for j in range(n_stages)
        if vapor_draws_arr[j] > 0)

    return ColumnResult(
        converged=converged, iterations=last_iter + 1,
        n_stages=n_stages, species_names=tuple(species_names),
        T=T_arr.copy(), p=p_arr.copy(), L=L_arr.copy(), V=V_arr.copy(),
        x=x_arr.copy(), y=y_arr.copy(), xi=xi_arr.copy(),
        D=D, B=B_local,
        feed_stage=int(feeds_stage[0]),
        feed_F=float(feeds_F[0]), feed_z=feeds_z[0].copy(),
        reflux_ratio=reflux_ratio,
        reactive_stages=tuple(sorted(reactive_set)),
        message=msg,
        feeds=feeds_out,
        liquid_draws=liquid_draws_out,
        vapor_draws=vapor_draws_out,
        side_strippers=tuple(ss_results),
    )


# =========================================================================
# Main column solver
# =========================================================================

def reactive_distillation_column(
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
    reactions: Sequence[LiquidPhaseReaction] = (),
    reactive_stages: Sequence[int] = (),
    T_init: Optional[Sequence[float]] = None,
    x_init: Optional[np.ndarray] = None,
    max_outer_iter: int = 100,
    tol: float = 1e-4,
    damping: float = 0.5,
    stage_holdup: Optional[float] = None,
    verbose: bool = False,
    method: str = "naphtali_sandholm",
    max_newton_iter: int = 30,
    newton_tol: float = 1e-7,
    fd_step: float = 1e-7,
    energy_balance: bool = False,
    h_V_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    h_L_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    # ---- Multi-feed and side draws (v0.9.71+) ----
    feeds: Optional[Sequence] = None,
    liquid_draws: Optional[dict] = None,
    vapor_draws: Optional[dict] = None,
    # ---- q-fraction feed and partial condenser (v0.9.72+) ----
    feed_q: Optional[float] = None,
    condenser: str = "total",
    # ---- pressure profile and stage efficiency (v0.9.73+) ----
    pressure_drop: Optional[float] = None,
    stage_efficiency = None,
    # ---- pump-arounds (v0.9.74+) ----
    pump_arounds: Optional[Sequence] = None,
    # ---- side strippers (v0.9.88+) ----
    side_strippers: Optional[Sequence] = None,
    # γ-φ-EOS coupling at the column level (v0.9.78+)
    vapor_eos = None,
    pure_liquid_volumes: Optional[Sequence[float]] = None,
    phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    # ---- Design-mode specifications (v0.9.75+) ----
    specs: Optional[Sequence['Spec']] = None,
    initial_distillate_rate: Optional[float] = None,
    initial_reflux_ratio: Optional[float] = None,
    spec_outer_max_iter: int = 30,
    spec_outer_tol: float = 1e-6,
) -> ColumnResult:
    """Solve a reactive-distillation column at steady state.

    Parameters
    ----------
    n_stages : int
        Total equilibrium stages. Stage 1 = top (just below total
        condenser), stage `n_stages` = bottom (partial reboiler).
    feed_stage : int
        1-indexed stage where the feed enters. Typical values are
        in the range 2..n_stages-1.
    feed_F : float
        Total feed molar flow rate.
    feed_z : sequence of float, length C
        Feed composition (mole fractions).
    feed_T : float
        Feed temperature [K]. Currently used only for a heuristic
        starting T-profile; energy balance is not enforced.
    reflux_ratio : float
        R = L_reflux / D where L_reflux is the liquid returned to
        stage 1 from the total condenser.
    distillate_rate : float
        D = distillate molar flow rate. Together with F and the
        reactions' net mole change, fixes B = F + DeltaN_rxn - D.
    pressure : float
        Uniform column pressure [Pa].
    species_names : sequence of str, length C
        Canonical species ordering.
    activity_model : object with .gammas(T, x) returning length-C array
        Liquid activity-coefficient model.
    psat_funcs : sequence of C callables T -> p_sat(T) [Pa]
        Pure-component vapor pressures.
    reactions : sequence of LiquidPhaseReaction, optional
        Liquid-phase reactions. If empty, this is a non-reactive
        distillation column.
    reactive_stages : sequence of int, optional
        1-indexed stage indices where reactions occur. Stages not in
        this list are pure-VLE.
    T_init : sequence of float, length n_stages, optional
        Initial temperature profile. If None, linear interpolation from
        feed bubble-T to feed bubble-T + 30 K.
    x_init : ndarray (n_stages, C), optional
        Initial liquid composition profile. If None, feed composition
        is used on every stage.
    max_outer_iter : int
        Maximum Wang-Henke outer iterations.
    tol : float
        Convergence tolerance on max ||T_new - T||_inf and ||x_new - x||_inf.
    damping : float
        Profile damping (0 < damping <= 1). 0.5 is a reasonable default.
    stage_holdup : float, optional
        Liquid holdup per reactive stage. If None, defaults to L_top
        (per-stage extent then has the same units as L). The chemical
        equilibrium does not depend on holdup magnitude (only on
        composition and T), so this parameter affects only the reported
        xi values.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    ColumnResult

    Notes
    -----
    Two solvers are available via the `method` parameter:

    "naphtali_sandholm" (default, recommended)
        Simultaneous Newton on the full augmented residual system
        per stage [x_{j,i}, T_j, xi_{j,r}].  Block-tridiagonal Jacobian
        built by central-difference finite differences; full Newton
        step with Armijo backtracking line search.  Converges
        quadratically near the solution -- typically 10-20 Newton
        iterations to ||F||_inf < 1e-7.  Tolerances on K_a closure
        (~1e-8 relative) and atom balance (~1e-10 relative) are
        essentially at machine precision.

    "wang_henke"
        Classical Henley-Seader bubble-point method: alternating
        per-species tridiagonal mass balance, per-stage chemistry
        update via increment-based xi accumulation, bubble-point
        T-update.  Robust but linearly convergent; typically 50-200
        outer iterations with under-relaxation (damping ~ 0.3-0.5).
        Useful as a fallback when Newton struggles with poor
        initial guesses.

    Convention for reflux on stage 1 (total condenser):
      x_0 = y_1 (composition preserved across condensation)
      L_0 = R * D (reflux returning to stage 1)
      V_1 = (R + 1) * D (vapor leaving stage 1, condenses to D + L_0)

    Mass balance on stage 1:
      V_2 y_2 + L_0 x_0 = V_1 y_1 + L_1 x_1
      Substituting L_0 x_0 = R*D y_1 and V_1 = (R+1)*D:
      V_2 y_2 = D y_1 + L_1 x_1
      With y_j = K_j x_j:
      V_2 K_2 x_2 = D K_1 x_1 + L_1 x_1 = (D K_1 + L_1) x_1

    Mass balance on stage N (partial reboiler):
      L_{N-1} x_{N-1} = V_N y_N + B x_B
      With y_N = K_N x_N and x_B = x_N:
      L_{N-1} x_{N-1} = (V_N K_N + B) x_N
    """
    # ===== v0.9.75: Design-mode specifications =====
    # If specs are given, wrap the column solver in a Newton outer loop
    # that varies D and/or R until the specs are satisfied.
    if specs is not None and len(specs) > 0:
        # Determine which of (D, R) are free.  A None value means "free".
        free_D = (distillate_rate is None)
        free_R = (reflux_ratio is None)
        n_free = int(free_D) + int(free_R)
        if n_free != len(specs):
            raise ValueError(
                f"design mode requires #specs ({len(specs)}) == #free "
                f"unknowns; pass distillate_rate=None and/or "
                f"reflux_ratio=None to free them.  Got "
                f"distillate_rate={distillate_rate}, "
                f"reflux_ratio={reflux_ratio}.")

        # Resolve initial guesses
        if free_D:
            if initial_distillate_rate is None:
                # Heuristic: half the total feed flow
                if feeds is not None:
                    F_total = sum(_get_F_attr(f) for f in feeds)
                else:
                    F_total = float(feed_F) if feed_F is not None else 100.0
                initial_distillate_rate = 0.5 * F_total
            D_use = float(initial_distillate_rate)
        else:
            D_use = float(distillate_rate)

        if free_R:
            if initial_reflux_ratio is None:
                initial_reflux_ratio = 2.0
            R_use = float(initial_reflux_ratio)
        else:
            R_use = float(reflux_ratio)

        # Compute total moles of each species fed (for recovery specs)
        C_local = len(species_names)
        feed_z_total = np.zeros(C_local)
        if feeds is not None:
            for f in feeds:
                if isinstance(f, FeedSpec):
                    Ff, zf = f.F, np.asarray(f.z)
                elif hasattr(f, "F"):
                    Ff = f.F; zf = np.asarray(f.z)
                elif isinstance(f, dict):
                    Ff = f["F"]; zf = np.asarray(f["z"])
                else:
                    Ff = f[1]; zf = np.asarray(f[2])
                feed_z_total += Ff * zf
        else:
            feed_z_total = float(feed_F) * np.asarray(feed_z, dtype=float)

        # Inner solve helper: fix D, R, call solver in non-design mode
        inner_kwargs = dict(
            n_stages=n_stages,
            feed_stage=feed_stage, feed_F=feed_F, feed_z=feed_z, feed_T=feed_T,
            pressure=pressure, species_names=species_names,
            activity_model=activity_model, psat_funcs=psat_funcs,
            reactions=reactions, reactive_stages=reactive_stages,
            T_init=T_init, x_init=x_init,
            max_outer_iter=max_outer_iter, tol=tol, damping=damping,
            stage_holdup=stage_holdup, verbose=False,
            method=method, max_newton_iter=max_newton_iter,
            newton_tol=newton_tol, fd_step=fd_step,
            energy_balance=energy_balance,
            h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
            feeds=feeds, liquid_draws=liquid_draws, vapor_draws=vapor_draws,
            feed_q=feed_q, condenser=condenser,
            pressure_drop=pressure_drop, stage_efficiency=stage_efficiency,
            pump_arounds=pump_arounds,
            side_strippers=side_strippers,
        )

        def inner_solve(D_val, R_val):
            return reactive_distillation_column(
                **inner_kwargs,
                distillate_rate=D_val, reflux_ratio=R_val,
                specs=None,
            )

        def residual_vec(D_val, R_val):
            r = inner_solve(D_val, R_val)
            if not r.converged:
                # Inner did not converge: signal to outer loop
                return None, r
            vec = np.array([
                _evaluate_spec(s, r, species_names, feed_z_total,
                               h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs)
                for s in specs])
            return vec, r

        # Newton outer loop
        u = []
        if free_D: u.append(D_use)
        if free_R: u.append(R_use)
        u = np.array(u, dtype=float)

        last_res = None
        for outer in range(spec_outer_max_iter):
            D_iter = u[0] if free_D else float(distillate_rate)
            R_iter = (u[1] if (free_D and free_R) else
                      (u[0] if free_R else float(reflux_ratio)))
            r_vec, last_res = residual_vec(D_iter, R_iter)
            if r_vec is None:
                # Inner failed; can't continue Newton
                if last_res is None:
                    raise RuntimeError(
                        "design-mode: inner solve never produced a result")
                last_res.message = (
                    f"design-mode failed: inner solver did not converge "
                    f"at outer iter {outer} (D={D_iter:.4g}, R={R_iter:.4g})")
                last_res.converged = False
                return last_res

            r_norm = float(np.max(np.abs(r_vec)))
            if verbose:
                print(f"  outer iter {outer}: u={u}, r={r_vec}, "
                      f"||r||={r_norm:.2e}")
            if r_norm < spec_outer_tol:
                last_res.message = (
                    f"design-mode converged in {outer + 1} outer iters, "
                    f"specs met to {r_norm:.2e}; "
                    + last_res.message)
                return last_res

            # FD Jacobian (1 column per free unknown)
            J = np.zeros((len(specs), len(u)))
            for j in range(len(u)):
                step = 1e-5 * max(abs(u[j]), 1.0)
                u_pert = u.copy(); u_pert[j] += step
                D_p = u_pert[0] if free_D else float(distillate_rate)
                R_p = (u_pert[1] if (free_D and free_R) else
                       (u_pert[0] if free_R else float(reflux_ratio)))
                r_pert, _ = residual_vec(D_p, R_p)
                if r_pert is None:
                    raise RuntimeError(
                        f"design-mode: FD perturbation failed at iter "
                        f"{outer}, unknown {j}")
                J[:, j] = (r_pert - r_vec) / step

            # Newton step with damping and bounds
            try:
                du = -np.linalg.solve(J, r_vec)
            except np.linalg.LinAlgError:
                du = -np.linalg.lstsq(J, r_vec, rcond=None)[0]

            # Damping: limit the relative step size
            for j in range(len(u)):
                max_step = 0.5 * max(abs(u[j]), 1.0)
                if abs(du[j]) > max_step:
                    du[j] *= max_step / abs(du[j])
            u = u + du

            # Bounds: keep D and R positive
            if free_D:
                idx_D = 0
                u[idx_D] = max(u[idx_D], 1e-4)
                # Also: D < total feed (under no-reaction assumption)
                F_total = (sum(_get_F_attr(f) for f in feeds) if feeds is not None
                           else float(feed_F) if feed_F is not None else 1e30)
                u[idx_D] = min(u[idx_D], 0.999 * F_total)
            if free_R:
                idx_R = (1 if (free_D and free_R) else 0)
                u[idx_R] = max(u[idx_R], 1e-3)

        # Did not converge in outer loop
        if last_res is not None:
            last_res.converged = False
            last_res.message = (
                f"design-mode did not converge in {spec_outer_max_iter} "
                f"outer iterations; final ||spec residual|| ~ {r_norm:.2e}")
            return last_res
        raise RuntimeError(
            "design-mode: outer loop did not produce a result")
    # ===== End design-mode =====

    if reflux_ratio is None or distillate_rate is None or pressure is None:
        raise ValueError(
            "reflux_ratio, distillate_rate, and pressure are required.")
    if species_names is None or activity_model is None or psat_funcs is None:
        raise ValueError(
            "species_names, activity_model, and psat_funcs are required.")
    if condenser not in ("total", "partial"):
        raise ValueError(
            f"condenser must be 'total' or 'partial', got {condenser!r}")

    # v0.9.73: per-stage pressure profile
    p_arr = _normalize_pressure(pressure, pressure_drop, n_stages)
    # v0.9.73: per-stage Murphree efficiency (None -> all 1.0)
    E_arr = _normalize_efficiency(stage_efficiency, n_stages)
    # v0.9.74: pump-arounds
    (pa_draw_arr, pa_return_arr, pa_flow_arr, pa_dT_arr,
     pa_L_add_arr) = _normalize_pump_arounds(pump_arounds, n_stages)
    # Side strippers (v0.9.88+) — solved simultaneously with the main column
    (ss_draw_arr, ss_return_arr, ss_n_stages_arr,
     ss_flow_arr, ss_bottoms_arr, ss_pressure_arr,
     ss_mode_arr, ss_steam_flow_arr, ss_steam_z_arr, ss_steam_T_arr) = (
        _normalize_side_strippers(side_strippers, n_stages,
                                   n_components=len(species_names)))

    C = len(species_names)
    if len(psat_funcs) != C:
        raise ValueError(f"psat_funcs length {len(psat_funcs)} != C={C}")

    # v0.9.78: γ-φ-EOS at the column level (validated after C is known)
    if vapor_eos is not None:
        if pure_liquid_volumes is not None and \
                len(pure_liquid_volumes) != C:
            raise ValueError(
                f"pure_liquid_volumes length {len(pure_liquid_volumes)} "
                f"!= C={C}")
        if phi_sat_funcs is not None and len(phi_sat_funcs) != C:
            raise ValueError(
                f"phi_sat_funcs length {len(phi_sat_funcs)} != C={C}")
        V_L_arr = (np.asarray(pure_liquid_volumes, dtype=float)
                   if pure_liquid_volumes is not None else None)
    else:
        V_L_arr = None

    # Normalize feeds (single-feed scalars OR feeds list) and side draws to
    # internal arrays.  For Wang-Henke, only the single-feed shorthand is
    # supported; if `feeds` is given with K>1 or any draws are provided,
    # the user must use method="naphtali_sandholm".
    (feeds_stage_arr, feeds_F_arr, feeds_z_arr, feeds_T_arr, feeds_q_arr,
     liquid_draws_arr, vapor_draws_arr) = _normalize_feeds_and_draws(
            n_stages, C,
            feed_stage, feed_F, feed_z, feed_T,
            feeds, liquid_draws, vapor_draws,
            feed_q=feed_q)
    K_feeds = len(feeds_F_arr)

    if method == "wang_henke" and (K_feeds > 1 or
                                    liquid_draws_arr.any() or
                                    vapor_draws_arr.any() or
                                    not np.allclose(feeds_q_arr, 1.0) or
                                    condenser == "partial" or
                                    not np.allclose(p_arr, p_arr[0]) or
                                    not np.allclose(E_arr, 1.0) or
                                    len(pa_flow_arr) > 0 or
                                    vapor_eos is not None):
        raise ValueError(
            "Wang-Henke solver supports only a single saturated-liquid "
            "feed (q=1), no side draws, total condenser, uniform "
            "pressure, full equilibrium (E=1), no pump-arounds, and "
            "modified-Raoult VLE (no vapor_eos).  Use "
            "method='naphtali_sandholm' for any of: multi-feed, "
            "side draws, q-fraction feeds, partial condenser, "
            "pressure profile, Murphree efficiency, pump-arounds, "
            "or γ-φ-EOS coupling.")

    # Single-feed legacy aliases (used for the heuristic T-profile and the
    # Wang-Henke iteration only)
    feed_stage = int(feeds_stage_arr[0])
    feed_F = float(feeds_F_arr[0])
    feed_z = feeds_z_arr[0].copy()
    feed_T = float(feeds_T_arr[0]) if feed_T is None else feed_T

    if len(species_names) != C:
        raise ValueError(f"species_names length {len(species_names)} != C={C}")
    if not (1 <= feed_stage <= n_stages):
        raise ValueError(f"feed_stage {feed_stage} not in [1, {n_stages}]")

    species_names = list(species_names)
    reactions = list(reactions)
    R = len(reactions)
    reactive_set = set(int(s) for s in reactive_stages)
    if reactive_set and not reactive_set.issubset(set(range(1, n_stages + 1))):
        raise ValueError(f"reactive_stages out of range: {reactive_stages}")
    if R == 0 and reactive_set:
        raise ValueError("reactive_stages specified but reactions=[]")

    # Map reaction species indices to canonical species_names ordering
    species_idx_in_rxn = []
    for rxn in reactions:
        idxs = []
        for sp in rxn.species_names:
            if sp not in species_names:
                raise KeyError(f"Reaction species '{sp}' not in column "
                               f"species_names {species_names}")
            idxs.append(species_names.index(sp))
        species_idx_in_rxn.append(idxs)

    # Initial T profile
    if T_init is None:
        # Heuristic: feed bubble-T at feed stage, gentle ramp top to bottom
        T_feed = _bubble_point_T(feed_z, float(p_arr[feed_stage - 1]),
                                 psat_funcs, activity_model,
                                 T_init=feed_T or 350.0)
        T_arr = np.linspace(T_feed - 5.0, T_feed + 25.0, n_stages)
    else:
        T_arr = np.asarray(T_init, dtype=float).copy()
        if T_arr.size != n_stages:
            raise ValueError(f"T_init length {T_arr.size} != n_stages={n_stages}")

    # Initial x profile
    if x_init is None:
        x_arr = np.tile(feed_z, (n_stages, 1)).astype(float)
    else:
        x_arr = np.asarray(x_init, dtype=float).copy()
        if x_arr.shape != (n_stages, C):
            raise ValueError(f"x_init shape {x_arr.shape} != ({n_stages}, {C})")

    # Initial flows from CMO + reflux
    D = float(distillate_rate)
    L_top = reflux_ratio * D                # liquid from condenser back to stage 1
    V_top = (reflux_ratio + 1.0) * D        # vapor leaving stage 1 to condenser

    # Flow profile under constant molar overflow (saturated-liquid feed
    # assumption) and zero net reaction-mole-change:
    #   Stages 1..f-1   (rectifying):       L_j = L_top,    V_j = V_top
    #   Stages f..N-1   (stripping below feed, above reboiler):
    #                                        L_j = L_top+F, V_j = V_top
    #   Stage N         (reboiler):          L_N = B,       V_N = V_top
    # Reaction-induced mole changes are folded into B in the iteration.
    L = np.full(n_stages, L_top)
    V = np.full(n_stages, V_top)
    # Stripping section (excluding the reboiler stage N=n_stages):
    for j in range(feed_stage - 1, n_stages - 1):
        L[j] = L_top + feed_F
    # Bottoms = F - D for zero mole-change reactions; updated in iteration
    B = feed_F - D
    L[-1] = B

    # Stage holdup (only affects reported xi units)
    if stage_holdup is None:
        stage_holdup = max(L_top, 1.0)

    xi_arr = np.zeros((n_stages, R))
    y_arr = np.zeros_like(x_arr)

    # Dispatch on solver method
    if method == "naphtali_sandholm":
        if energy_balance:
            if h_V_funcs is None or h_L_funcs is None:
                raise ValueError(
                    "energy_balance=True requires h_V_funcs and h_L_funcs "
                    "(lists of C callables T -> J/mol per species).")
            if len(h_V_funcs) != C or len(h_L_funcs) != C:
                raise ValueError(
                    f"h_V_funcs and h_L_funcs must each have length C={C}.")
            # Warm-start: solve the easier CMO+bubble-point problem first,
            # then use that profile as initial guess for the energy-balance
            # Newton.  EB Newton is notably less robust than CMO N-S for
            # stiff activity models (UNIFAC, NRTL with strong nonidealities)
            # because dropping CMO doubles the per-stage variable count.
            r_warm = _naphtali_sandholm_solve(
                n_stages=n_stages,
                feeds_stage=feeds_stage_arr, feeds_F=feeds_F_arr,
                feeds_z=feeds_z_arr,
                feeds_q=feeds_q_arr,
                liquid_draws_arr=liquid_draws_arr,
                vapor_draws_arr=vapor_draws_arr,
                distillate_rate=D, reflux_ratio=reflux_ratio,
                p_arr=p_arr, species_names=species_names,
                activity_model=activity_model, psat_funcs=psat_funcs,
                reactions=reactions, reactive_set=reactive_set,
                species_idx_in_rxn=species_idx_in_rxn,
                T_init=T_arr, x_init=x_arr, xi_init=xi_arr,
                E_arr=E_arr,
                pa_draw=pa_draw_arr, pa_return=pa_return_arr,
                pa_flow=pa_flow_arr, pa_L_add=pa_L_add_arr,
                ss_draw=ss_draw_arr, ss_return=ss_return_arr,
                ss_n_stages=ss_n_stages_arr, ss_flow=ss_flow_arr,
                ss_bottoms=ss_bottoms_arr, ss_pressure=ss_pressure_arr,
                ss_mode=ss_mode_arr, ss_steam_flow=ss_steam_flow_arr,
                ss_steam_z=ss_steam_z_arr, ss_steam_T=ss_steam_T_arr,
                vapor_eos=vapor_eos, V_L_arr=V_L_arr,
                phi_sat_funcs=phi_sat_funcs,
                max_newton_iter=max_newton_iter, tol=newton_tol,
                fd_step=fd_step, verbose=verbose,
            )
            if not r_warm.converged and verbose:
                print("  (warm-start CMO N-S did not converge; "
                      "proceeding to EB anyway)")
            r_eb = _naphtali_sandholm_solve_with_energy(
                n_stages=n_stages,
                feeds_stage=feeds_stage_arr, feeds_F=feeds_F_arr,
                feeds_z=feeds_z_arr, feeds_T=feeds_T_arr,
                feeds_q=feeds_q_arr,
                liquid_draws_arr=liquid_draws_arr,
                vapor_draws_arr=vapor_draws_arr,
                distillate_rate=D, reflux_ratio=reflux_ratio,
                p_arr=p_arr, species_names=species_names,
                activity_model=activity_model, psat_funcs=psat_funcs,
                reactions=reactions, reactive_set=reactive_set,
                species_idx_in_rxn=species_idx_in_rxn,
                h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
                T_init=r_warm.T, x_init=r_warm.x,
                V_init=r_warm.V, L_init=r_warm.L, xi_init=r_warm.xi,
                E_arr=E_arr,
                pa_draw=pa_draw_arr, pa_return=pa_return_arr,
                pa_flow=pa_flow_arr, pa_dT=pa_dT_arr,
                pa_L_add=pa_L_add_arr,
                ss_draw=ss_draw_arr, ss_return=ss_return_arr,
                ss_n_stages=ss_n_stages_arr, ss_flow=ss_flow_arr,
                ss_bottoms=ss_bottoms_arr, ss_pressure=ss_pressure_arr,
                ss_mode=ss_mode_arr, ss_steam_flow=ss_steam_flow_arr,
                ss_steam_z=ss_steam_z_arr, ss_steam_T=ss_steam_T_arr,
                vapor_eos=vapor_eos, V_L_arr=V_L_arr,
                phi_sat_funcs=phi_sat_funcs,
                max_newton_iter=max_newton_iter, tol=newton_tol,
                fd_step=fd_step, verbose=verbose,
            )
            r_eb.condenser = condenser
            return r_eb
        r_cmo = _naphtali_sandholm_solve(
            n_stages=n_stages,
            feeds_stage=feeds_stage_arr, feeds_F=feeds_F_arr,
            feeds_z=feeds_z_arr,
            feeds_q=feeds_q_arr,
            liquid_draws_arr=liquid_draws_arr,
            vapor_draws_arr=vapor_draws_arr,
            distillate_rate=D, reflux_ratio=reflux_ratio,
            p_arr=p_arr, species_names=species_names,
            activity_model=activity_model, psat_funcs=psat_funcs,
            reactions=reactions, reactive_set=reactive_set,
            species_idx_in_rxn=species_idx_in_rxn,
            T_init=T_arr, x_init=x_arr, xi_init=xi_arr,
            E_arr=E_arr,
            pa_draw=pa_draw_arr, pa_return=pa_return_arr,
            pa_flow=pa_flow_arr, pa_L_add=pa_L_add_arr,
            ss_draw=ss_draw_arr, ss_return=ss_return_arr,
            ss_n_stages=ss_n_stages_arr, ss_flow=ss_flow_arr,
            ss_bottoms=ss_bottoms_arr, ss_pressure=ss_pressure_arr,
                ss_mode=ss_mode_arr, ss_steam_flow=ss_steam_flow_arr,
                ss_steam_z=ss_steam_z_arr, ss_steam_T=ss_steam_T_arr,
            vapor_eos=vapor_eos, V_L_arr=V_L_arr,
            phi_sat_funcs=phi_sat_funcs,
            max_newton_iter=max_newton_iter, tol=newton_tol,
            fd_step=fd_step, verbose=verbose,
        )
        r_cmo.condenser = condenser
        return r_cmo
    elif method != "wang_henke":
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use 'wang_henke' or 'naphtali_sandholm'.")
    if energy_balance:
        raise ValueError(
            "energy_balance=True is only supported with "
            "method='naphtali_sandholm'.")

    last_dT = math.inf
    last_dx = math.inf

    # Mass-balance source on a stage from its current xi (in mol/time)
    def reaction_source(j_zero_idx: int) -> np.ndarray:
        """Total mol/time source per species i on stage j due to xi_j."""
        if R == 0 or (j_zero_idx + 1) not in reactive_set:
            return np.zeros(C)
        src = np.zeros(C)
        for r, rxn in enumerate(reactions):
            for sp_local_idx, nu_local in enumerate(rxn.nu):
                i = species_idx_in_rxn[r][sp_local_idx]
                src[i] += nu_local * xi_arr[j_zero_idx, r]
        return src

    for outer in range(max_outer_iter):
        # 1. K-values at current state
        K_arr = np.zeros((n_stages, C))
        gamma_arr = np.zeros((n_stages, C))
        for j in range(n_stages):
            psat_j = np.array([f(T_arr[j]) for f in psat_funcs])
            x_norm = x_arr[j] / max(x_arr[j].sum(), 1e-30)
            gamma_j = np.asarray(activity_model.gammas(T_arr[j], x_norm))
            K_arr[j] = gamma_j * psat_j / p_arr[j]
            gamma_arr[j] = gamma_j

        # 2. Inner sub-iteration: with K and T frozen, simultaneously
        # solve for (x, xi) so that BOTH mass balance and chemical
        # equilibrium are satisfied. The decoupled approach (compute
        # xi from current x, then x from current xi) does not converge
        # because xi from a "closed-stage" equilibrium calculation
        # underestimates the true open-stage steady-state extent.
        x_inner = x_arr.copy()
        xi_inner = xi_arr.copy()
        max_inner = 30
        for inner in range(max_inner):
            # 2a. Tridiagonal mass balance per component i, given xi_inner
            x_new = np.zeros_like(x_arr)
            for i in range(C):
                a = np.zeros(n_stages)
                b = np.zeros(n_stages)
                c = np.zeros(n_stages)
                d = np.zeros(n_stages)
                # Reaction source on each stage from xi_inner
                rxn_src = np.zeros(n_stages)
                for stage in reactive_set:
                    j = stage - 1
                    for r, rxn_obj in enumerate(reactions):
                        # Find this species's nu in this reaction (0 if absent)
                        for sp_local_idx, nu_local in enumerate(rxn_obj.nu):
                            if species_idx_in_rxn[r][sp_local_idx] == i:
                                rxn_src[j] += nu_local * xi_inner[j, r]
                # j = 0
                b[0] = -(D * K_arr[0, i] + L[0])
                c[0] = V[1] * K_arr[1, i] if n_stages > 1 else 0.0
                d[0] = -rxn_src[0]
                if feed_stage == 1:
                    d[0] -= feed_F * feed_z[i]
                for j in range(1, n_stages - 1):
                    a[j] = L[j-1]
                    b[j] = -(V[j] * K_arr[j, i] + L[j])
                    c[j] = V[j+1] * K_arr[j+1, i]
                    feed_term = feed_F * feed_z[i] if (j+1) == feed_stage else 0.0
                    d[j] = -feed_term - rxn_src[j]
                if n_stages > 1:
                    a[-1] = L[-2]
                    b[-1] = -(V[-1] * K_arr[-1, i] + B)
                    d[-1] = -rxn_src[n_stages - 1]
                    if feed_stage == n_stages:
                        d[-1] -= feed_F * feed_z[i]
                try:
                    x_new[:, i] = _solve_tridiagonal(a, b, c, d)
                except Exception:
                    x_new[:, i] = x_inner[:, i].copy()
            x_new = np.maximum(x_new, 0.0)
            row_sums = x_new.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
            x_new = x_new / row_sums

            # 2b. Compute the "batch" extent correction on each reactive
            # stage: this is the ADDITIONAL extent that, applied in a
            # closed-stage sense, would bring the current x_new[j] onto
            # the K_a = K_eq surface.  We then ADD this increment to the
            # running xi_inner (with damping) rather than replacing.
            #
            # This integration-style update is the key fix: at the fixed
            # point the increment vanishes only when x_new is already at
            # equilibrium.  Atom conservation is automatic because the
            # tridiagonal solution always satisfies the per-species mass
            # balance with whatever xi_inner is supplied (and ν*xi is
            # always atom-conserving).
            xi_increment = np.zeros_like(xi_inner)
            for stage in reactive_set:
                j = stage - 1
                xi_increment[j] = _solve_extent_at_stage(
                    x_new[j], T_arr[j], activity_model,
                    reactions, species_idx_in_rxn,
                    n_L=stage_holdup, tol=1e-10, maxiter=200)

            # Damping: blend the increment into xi_inner; blend x_new
            # into x_inner.  Both blends shrink to zero at the fixed
            # point where x_new is already at equilibrium.
            inner_damping = 0.5
            xi_inner_new = xi_inner + inner_damping * xi_increment
            x_inner_new = ((1.0 - inner_damping) * x_inner
                            + inner_damping * x_new)

            d_x_inner = float(np.max(np.abs(x_inner_new - x_inner)))
            if R > 0 and xi_inner.size > 0:
                d_xi_inner = float(np.max(np.abs(xi_increment)))
            else:
                d_xi_inner = 0.0
            x_inner = x_inner_new
            xi_inner = xi_inner_new
            if d_x_inner < 1e-7 and d_xi_inner < 1e-9:
                break

        # 3. Use converged inner result
        x_blend = (1.0 - damping) * x_arr + damping * x_inner
        xi_arr = (1.0 - damping) * xi_arr + damping * xi_inner

        # 3b. Update B (bottoms flow) from total mass balance:
        if R > 0:
            dn_per_rxn = np.array([float(rxn.nu.sum()) for rxn in reactions])
            dN_rxn = float((xi_arr * dn_per_rxn).sum())
        else:
            dN_rxn = 0.0
        B = feed_F + dN_rxn - D
        L[-1] = B

        # 4. Bubble-point T update
        T_new = T_arr.copy()
        for j in range(n_stages):
            T_new[j] = _bubble_point_T(
                x_blend[j], float(p_arr[j]), psat_funcs, activity_model,
                T_init=T_arr[j], tol=1e-5, maxiter=60)

        T_blend = (1.0 - damping) * T_arr + damping * T_new

        # 5. Recompute y from K * x
        y_new = K_arr * x_blend
        y_sums = y_new.sum(axis=1, keepdims=True)
        y_sums = np.where(y_sums < 1e-12, 1.0, y_sums)
        y_arr = y_new / y_sums

        # 6. Convergence check
        dT = float(np.max(np.abs(T_blend - T_arr)))
        dx = float(np.max(np.abs(x_blend - x_arr)))
        if verbose:
            print(f"  outer {outer}: dT={dT:.3e}, dx={dx:.3e}, "
                  f"x_D = {y_arr[0].round(3)}, x_B = {x_blend[-1].round(3)}")
        T_arr = T_blend
        x_arr = x_blend
        last_dT, last_dx = dT, dx
        if dT < tol and dx < tol:
            # Final clean recomputation of y
            for j in range(n_stages):
                psat_j = np.array([f(T_arr[j]) for f in psat_funcs])
                x_norm = x_arr[j] / max(x_arr[j].sum(), 1e-30)
                gamma_j = np.asarray(activity_model.gammas(T_arr[j], x_norm))
                K_arr[j] = gamma_j * psat_j / p_arr[j]
            y_arr = K_arr * x_arr
            ysums = y_arr.sum(axis=1, keepdims=True)
            ysums = np.where(ysums < 1e-12, 1.0, ysums)
            y_arr = y_arr / ysums
            return ColumnResult(
                converged=True, iterations=outer + 1,
                n_stages=n_stages, species_names=tuple(species_names),
                T=T_arr.copy(), p=p_arr.copy(), L=L.copy(), V=V.copy(),
                x=x_arr.copy(), y=y_arr.copy(), xi=xi_arr.copy(),
                D=D, B=B, feed_stage=feed_stage,
                feed_F=feed_F, feed_z=feed_z.copy(),
                reflux_ratio=reflux_ratio,
                reactive_stages=tuple(sorted(reactive_set)),
                message=f"converged in {outer + 1} outer iters "
                        f"(dT={dT:.2e}, dx={dx:.2e})")

    # Did not converge
    return ColumnResult(
        converged=False, iterations=max_outer_iter,
        n_stages=n_stages, species_names=tuple(species_names),
        T=T_arr.copy(), p=p_arr.copy(), L=L.copy(), V=V.copy(),
        x=x_arr.copy(), y=y_arr.copy(), xi=xi_arr.copy(),
        D=D, B=B, feed_stage=feed_stage,
        feed_F=feed_F, feed_z=feed_z.copy(),
        reflux_ratio=reflux_ratio,
        reactive_stages=tuple(sorted(reactive_set)),
        message=f"did not converge in {max_outer_iter} iters "
                f"(dT={last_dT:.2e}, dx={last_dx:.2e})")
