"""Multi-stage liquid-liquid extraction column at steady state.

Solves a countercurrent extraction column where two partially miscible
liquid phases (raffinate R and extract E) flow in opposite directions
through ``n_stages`` equilibrium stages.  Heavy feed F enters stage 0;
light solvent S enters stage n-1.  Two product streams leave: extract
E_0 (top) and raffinate R_{n-1} (bottom).

Stage numbering convention
--------------------------
                          E_0 (extract product)
                           ^
       F (feed) ----+------+------+
                    |             |
                    |   stage 0   |
                    +------+------+
                           |
                          ...
                           |
                    +------+------+
                    |   stage j   |
                    +------+------+
                          ...
                           |
                    +------+------+
                    |  stage n-1  |
                    +------+------+
                                  |
                                  v
                          R_{n-1} (raffinate product)        ^---- S (solvent)

Raffinate (heavy) flows 0 -> n-1 with rate R_j leaving stage j.
Extract  (light) flows n-1 -> 0 with rate E_j leaving stage j.
Boundary inputs: R_{-1} = F with comp z_F; E_n = S with comp z_S.

Algorithm
---------
Naphtali-Sandholm simultaneous Newton.  Per stage j:

    Variant                          unknowns                    residuals
    ---------------------------------------------------------------------
    Plain LLE                        x^R, x^E, R, E              M, iso-act, Sum_R, Sum_E
                                     (2C+2)                      (2C+2)

    + chemistry on reactive stage    ... + xi (R_chem)           ... + K_a=K_eq (R_chem)
    + energy balance                 ... + T                     ... + H

Residuals are evaluated in this fixed slot order:
M (C), iso-activity (C), Sum_R, Sum_E, [H], [chemistry].

Equilibrium uses ``gamma^R x^R - gamma^E x^E = 0`` directly rather than
log form; this is robust at near-zero mole fractions where one species
disappears from one phase entirely.

Initialization
--------------
Newton seeded from a single LLE flash on the overall (F+S) mixture,
with the heavier-side phase assigned to the raffinate.  This non-trivial
asymmetric guess avoids the trivial collapse where x^R = x^E sits as
a saddle satisfying every residual identically.  For the energy-balance
variant, T_j initializes linearly between feed_T and solvent_T;
xi_j initializes to zero on reactive stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ExtractionColumnResult:
    """Solution of a multi-stage liquid-liquid extraction column."""
    converged: bool
    iterations: int
    n_stages: int
    species_names: Tuple[str, ...]
    T: np.ndarray           # (n_stages,)  --  isothermal returns repeated value
    x_R: np.ndarray         # (n_stages, C)
    x_E: np.ndarray         # (n_stages, C)
    R: np.ndarray           # (n_stages,)
    E: np.ndarray           # (n_stages,)
    xi: np.ndarray          # (n_stages, R_chem)
    F: float
    S: float
    z_F: np.ndarray
    z_S: np.ndarray
    reactive_stages: Tuple[int, ...] = ()
    reaction_phase: str = "E"
    energy_balance: bool = False
    message: str = ""

    @property
    def x_raffinate_product(self) -> np.ndarray:
        return self.x_R[-1]

    @property
    def x_extract_product(self) -> np.ndarray:
        return self.x_E[0]

    @property
    def R_product(self) -> float:
        return float(self.R[-1])

    @property
    def E_product(self) -> float:
        return float(self.E[0])

    def recovery(self, species: str) -> float:
        """Fraction of `species` in the FEED that ends up in the
        extract product."""
        if species not in self.species_names:
            raise KeyError(f"unknown species: {species}")
        i = self.species_names.index(species)
        moles_in_feed = self.F * float(self.z_F[i])
        if moles_in_feed <= 0.0:
            return float("nan")
        return float(self.E_product * float(self.x_E[0, i]) / moles_in_feed)

    def conversion(self, species: str) -> float:
        """Fraction of `species` originally in the feed+solvent that
        has been consumed by reaction.  Negative for products
        (production fraction)."""
        if species not in self.species_names:
            raise KeyError(f"unknown species: {species}")
        i = self.species_names.index(species)
        moles_in = self.F * float(self.z_F[i]) + self.S * float(self.z_S[i])
        moles_out = (self.R_product * float(self.x_R[-1, i])
                      + self.E_product * float(self.x_E[0, i]))
        if moles_in <= 0.0:
            return float("nan")
        return float((moles_in - moles_out) / moles_in)


def lle_extraction_column(
    n_stages: int,
    feed_F: float,
    feed_z: Sequence[float],
    solvent_S: float,
    solvent_z: Sequence[float],
    T: Optional[float] = None,
    *,
    species_names: Sequence[str],
    activity_model,
    reactions: Sequence = (),
    reactive_stages: Sequence[int] = (),
    reaction_phase: str = "E",
    energy_balance: bool = False,
    feed_T: Optional[float] = None,
    solvent_T: Optional[float] = None,
    h_L_funcs: Optional[Sequence[Callable[[float], float]]] = None,
    init_x_R: Optional[np.ndarray] = None,
    init_x_E: Optional[np.ndarray] = None,
    init_R: Optional[np.ndarray] = None,
    init_E: Optional[np.ndarray] = None,
    init_T: Optional[np.ndarray] = None,
    init_xi: Optional[np.ndarray] = None,
    max_newton_iter: int = 60,
    tol: float = 1e-7,
    fd_step: float = 1e-7,
    verbose: bool = False,
) -> ExtractionColumnResult:
    """Solve a countercurrent liquid-liquid extraction column.

    Parameters
    ----------
    n_stages, feed_F, feed_z, solvent_S, solvent_z : column inputs.
    T : float, optional
        Column temperature [K].  Used when ``energy_balance=False``.
        Required if not energy-balanced.
    species_names, activity_model : as usual.
    reactions : sequence of LiquidPhaseReaction
        Liquid-phase reactions, default empty (non-reactive column).
    reactive_stages : sequence of 1-indexed stage numbers
        Stages on which reactions occur.
    reaction_phase : {"R", "E"}
        Phase the reactions occur in.  Stoichiometric source enters
        each stage's component balance regardless; this flag selects
        which phase's gammas and mole fractions are used in the
        K_a = K_eq equilibrium constraint.  Default "E" (extract phase
        is typical for solvent-driven complex-formation chemistry).
    energy_balance : bool
        If True, solve T_j on every stage from per-stage enthalpy
        balance.  Requires h_L_funcs, feed_T, solvent_T.
    feed_T, solvent_T : float
        Inlet temperatures [K].  Required when energy_balance=True.
    h_L_funcs : sequence of C callables T -> J/mol
        Pure-component liquid enthalpies.  Both phases share the same
        per-species liquid enthalpy at the local T (ideal-mixing
        assumption -- excess enthalpy is neglected).
    init_* : optional initial guesses; see source.

    Returns
    -------
    ExtractionColumnResult
    """
    C = len(species_names)
    z_F = np.asarray(feed_z, dtype=float)
    z_S = np.asarray(solvent_z, dtype=float)
    if len(z_F) != C:
        raise ValueError(f"len(feed_z)={len(z_F)} != C={C}")
    if len(z_S) != C:
        raise ValueError(f"len(solvent_z)={len(z_S)} != C={C}")
    F = float(feed_F)
    S = float(solvent_S)
    if F <= 0 or S <= 0:
        raise ValueError("feed_F and solvent_S must be positive")
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if reaction_phase not in ("R", "E"):
        raise ValueError("reaction_phase must be 'R' or 'E'")

    R_chem = len(reactions)
    reactive_set = set(reactive_stages)
    if reactive_set and not all(1 <= j <= n_stages for j in reactive_set):
        raise ValueError(f"reactive_stages must be 1-indexed in [1, n_stages]; "
                         f"got {sorted(reactive_set)}")

    if energy_balance:
        if h_L_funcs is None:
            raise ValueError(
                "energy_balance=True requires h_L_funcs "
                "(list of C callables T -> J/mol per species).")
        if len(h_L_funcs) != C:
            raise ValueError(f"h_L_funcs must have length C={C}.")
        if feed_T is None or solvent_T is None:
            raise ValueError(
                "energy_balance=True requires feed_T and solvent_T.")
    else:
        if T is None:
            raise ValueError(
                "T (column temperature) is required when energy_balance=False.")

    # Reaction stoichiometry in column-species ordering
    if R_chem > 0:
        species_idx_in_rxn: List[List[int]] = []
        species_list = list(species_names)
        for rxn in reactions:
            idxs = []
            for sp in rxn.species_names:
                if sp not in species_list:
                    raise ValueError(
                        f"reaction species {sp!r} not in column species")
                idxs.append(species_list.index(sp))
            species_idx_in_rxn.append(idxs)
        nu_full = np.zeros((R_chem, C))
        for r, rxn in enumerate(reactions):
            for k, nu_local in enumerate(rxn.nu):
                nu_full[r, species_idx_in_rxn[r][k]] = nu_local
        dH_rxn_arr = np.array([float(getattr(r, "dH_rxn", 0.0))
                                for r in reactions])
    else:
        nu_full = np.zeros((0, C))
        dH_rxn_arr = np.zeros(0)

    # ---------- per-stage layout ----------
    has_T = bool(energy_balance)
    n_vars_per_stage: list[int] = []
    for j in range(n_stages):
        is_reactive = (j + 1) in reactive_set
        n = 2 * C + 2 + (1 if has_T else 0) + (R_chem if is_reactive else 0)
        n_vars_per_stage.append(n)
    var_offsets = np.array([0] + list(np.cumsum(n_vars_per_stage)))
    n_total = int(var_offsets[-1])

    def slot_xR(j):  return var_offsets[j], var_offsets[j] + C
    def slot_xE(j):  return var_offsets[j] + C, var_offsets[j] + 2 * C
    def slot_R(j):   return var_offsets[j] + 2 * C
    def slot_E(j):   return var_offsets[j] + 2 * C + 1
    def slot_T(j):
        return var_offsets[j] + 2 * C + 2 if has_T else None
    def slot_xi(j):
        if (j + 1) not in reactive_set or R_chem == 0:
            return None
        start = var_offsets[j] + 2 * C + 2 + (1 if has_T else 0)
        return start, start + R_chem

    def unpack(w: np.ndarray):
        x_R = np.zeros((n_stages, C))
        x_E = np.zeros((n_stages, C))
        R_arr = np.zeros(n_stages)
        E_arr = np.zeros(n_stages)
        T_arr = np.zeros(n_stages)
        xi_arr = np.zeros((n_stages, max(R_chem, 1)))
        for j in range(n_stages):
            xs, xe = slot_xR(j); x_R[j] = w[xs:xe]
            xs, xe = slot_xE(j); x_E[j] = w[xs:xe]
            R_arr[j] = w[slot_R(j)]
            E_arr[j] = w[slot_E(j)]
            if has_T:
                T_arr[j] = w[slot_T(j)]
            sxi = slot_xi(j)
            if sxi is not None:
                xi_arr[j, :R_chem] = w[sxi[0]:sxi[1]]
        if not has_T:
            T_arr.fill(float(T))
        return x_R, x_E, R_arr, E_arr, T_arr, xi_arr

    def pack(x_R, x_E, R_arr, E_arr, T_arr, xi_arr):
        w = np.zeros(n_total)
        for j in range(n_stages):
            xs, xe = slot_xR(j); w[xs:xe] = x_R[j]
            xs, xe = slot_xE(j); w[xs:xe] = x_E[j]
            w[slot_R(j)] = R_arr[j]
            w[slot_E(j)] = E_arr[j]
            if has_T:
                w[slot_T(j)] = T_arr[j]
            sxi = slot_xi(j)
            if sxi is not None:
                w[sxi[0]:sxi[1]] = xi_arr[j, :R_chem]
        return w

    # ---------- enthalpy constants ----------
    if energy_balance:
        h_F = float(sum(z_F[i] * h_L_funcs[i](float(feed_T)) for i in range(C)))
        h_S = float(sum(z_S[i] * h_L_funcs[i](float(solvent_T)) for i in range(C)))
        H_scale = max(F + S, 1.0) * 1e4
    else:
        h_F = h_S = 0.0
        H_scale = 1.0

    # ---------- residuals ----------
    def residuals(w: np.ndarray) -> np.ndarray:
        x_R, x_E, R_arr, E_arr, T_arr, xi_arr = unpack(w)
        gam_R = np.zeros((n_stages, C))
        gam_E = np.zeros((n_stages, C))
        h_R_arr = np.zeros(n_stages)
        h_E_arr = np.zeros(n_stages)
        for j in range(n_stages):
            T_j = T_arr[j]
            xR_safe = np.maximum(x_R[j], 1e-30); xR_safe /= xR_safe.sum()
            xE_safe = np.maximum(x_E[j], 1e-30); xE_safe /= xE_safe.sum()
            gam_R[j] = np.asarray(activity_model.gammas(T_j, xR_safe))
            gam_E[j] = np.asarray(activity_model.gammas(T_j, xE_safe))
            if energy_balance:
                h_R_arr[j] = sum(x_R[j, i] * h_L_funcs[i](T_j) for i in range(C))
                h_E_arr[j] = sum(x_E[j, i] * h_L_funcs[i](T_j) for i in range(C))

        F_vec = np.zeros(n_total)
        for j in range(n_stages):
            off = var_offsets[j]
            is_reactive = (j + 1) in reactive_set

            # M
            for i in range(C):
                in_R_i = (F * z_F[i] if j == 0
                          else R_arr[j - 1] * x_R[j - 1, i])
                in_E_i = (S * z_S[i] if j == n_stages - 1
                          else E_arr[j + 1] * x_E[j + 1, i])
                out_i = R_arr[j] * x_R[j, i] + E_arr[j] * x_E[j, i]
                rxn_src = 0.0
                if is_reactive and R_chem > 0:
                    for r in range(R_chem):
                        rxn_src += nu_full[r, i] * xi_arr[j, r]
                F_vec[off + i] = in_R_i + in_E_i + rxn_src - out_i

            # E (iso-activity)
            for i in range(C):
                F_vec[off + C + i] = (gam_R[j, i] * x_R[j, i]
                                       - gam_E[j, i] * x_E[j, i])

            # S
            F_vec[off + 2 * C] = float(x_R[j].sum() - 1.0)
            F_vec[off + 2 * C + 1] = float(x_E[j].sum() - 1.0)

            cursor = off + 2 * C + 2
            # H
            if energy_balance:
                in_h = (F * h_F if j == 0
                        else R_arr[j - 1] * h_R_arr[j - 1])
                in_h += (S * h_S if j == n_stages - 1
                         else E_arr[j + 1] * h_E_arr[j + 1])
                out_h = R_arr[j] * h_R_arr[j] + E_arr[j] * h_E_arr[j]
                rxn_h = 0.0
                if is_reactive and R_chem > 0:
                    for r in range(R_chem):
                        rxn_h += -dH_rxn_arr[r] * xi_arr[j, r]
                F_vec[cursor] = (in_h + rxn_h - out_h) / H_scale
                cursor += 1

            # Chemistry
            if is_reactive and R_chem > 0:
                if reaction_phase == "E":
                    gam_use = gam_E[j]; x_use = x_E[j]
                else:
                    gam_use = gam_R[j]; x_use = x_R[j]
                log_gx = (np.log(np.maximum(gam_use, 1e-30))
                          + np.log(np.maximum(x_use, 1e-30)))
                for r in range(R_chem):
                    ln_Ka = float((nu_full[r] * log_gx).sum())
                    F_vec[cursor + r] = ln_Ka - reactions[r].ln_K_eq(T_arr[j])

        return F_vec

    # ---------- initialization ----------
    # When chemistry or energy balance is on, the extra unknowns make the
    # cold-start problem stiff -- Newton can wander and blow up.  Cure
    # this by nested warm-starts:
    #   chemistry only             : seed from non-reactive isothermal solve
    #   chemistry + energy balance : seed from reactive isothermal solve
    # The warm-start uses the SAME activity model and column geometry,
    # and is cheap (5-10 Newton iters typically).
    needs_warmstart = (R_chem > 0 or energy_balance) \
                       and init_x_R is None and init_x_E is None
    if needs_warmstart:
        # Stage 1: non-reactive isothermal at feed_T (or T if isothermal)
        T_for_warmstart = float(feed_T) if energy_balance else float(T)
        try:
            r_warm = lle_extraction_column(
                n_stages=n_stages,
                feed_F=F, feed_z=z_F,
                solvent_S=S, solvent_z=z_S,
                T=T_for_warmstart,
                species_names=species_names,
                activity_model=activity_model,
                max_newton_iter=max_newton_iter, tol=max(tol, 1e-6),
                fd_step=fd_step, verbose=False,
            )
            if r_warm.converged:
                init_x_R = r_warm.x_R.copy()
                init_x_E = r_warm.x_E.copy()
                init_R = r_warm.R.copy()
                init_E = r_warm.E.copy()
            elif verbose:
                print("  (non-reactive warm-start did not converge; "
                      "proceeding from cold start)")
        except Exception:
            pass

        # Stage 2: reactive isothermal (only if both chemistry and energy)
        if R_chem > 0 and energy_balance and init_x_R is not None:
            try:
                r_warm2 = lle_extraction_column(
                    n_stages=n_stages,
                    feed_F=F, feed_z=z_F,
                    solvent_S=S, solvent_z=z_S,
                    T=T_for_warmstart,
                    species_names=species_names,
                    activity_model=activity_model,
                    reactions=reactions, reactive_stages=reactive_stages,
                    reaction_phase=reaction_phase,
                    init_x_R=init_x_R, init_x_E=init_x_E,
                    init_R=init_R, init_E=init_E,
                    max_newton_iter=max_newton_iter, tol=max(tol, 1e-6),
                    fd_step=fd_step, verbose=False,
                )
                if r_warm2.converged:
                    init_x_R = r_warm2.x_R.copy()
                    init_x_E = r_warm2.x_E.copy()
                    init_R = r_warm2.R.copy()
                    init_E = r_warm2.E.copy()
                    if r_warm2.xi.shape[1] >= R_chem:
                        if init_xi is None:
                            init_xi = np.zeros((n_stages, max(R_chem, 1)))
                        init_xi[:, :R_chem] = r_warm2.xi[:, :R_chem]
                elif verbose:
                    print("  (reactive isothermal warm-start did not "
                          "converge; proceeding)")
            except Exception:
                pass

    if init_x_R is None or init_x_E is None:
        from ..activity.lle import LLEFlash
        z_overall = (F * z_F + S * z_S) / (F + S)
        x1_init = 0.7 * z_F + 0.3 * z_S; x1_init /= x1_init.sum()
        x2_init = 0.3 * z_F + 0.7 * z_S; x2_init /= x2_init.sum()
        T_for_flash = float(feed_T) if energy_balance else float(T)
        flash = LLEFlash(activity_model)
        try:
            sol = flash.solve(T_for_flash, z_overall, x1_init, x2_init,
                              maxiter=200, tol=1e-7)
            xR0, xE0 = sol.x1, sol.x2
            if np.linalg.norm(sol.x1 - z_F) > np.linalg.norm(sol.x2 - z_F):
                xR0, xE0 = sol.x2, sol.x1
        except Exception:
            xR0 = 0.7 * z_F + 0.3 * z_S; xR0 /= xR0.sum()
            xE0 = 0.3 * z_F + 0.7 * z_S; xE0 /= xE0.sum()
        if init_x_R is None:
            init_x_R = np.tile(xR0, (n_stages, 1))
        if init_x_E is None:
            init_x_E = np.tile(xE0, (n_stages, 1))
    if init_R is None:
        init_R = np.full(n_stages, F)
    if init_E is None:
        init_E = np.full(n_stages, S)
    if init_T is None:
        if energy_balance:
            init_T = np.linspace(float(feed_T), float(solvent_T), n_stages)
        else:
            init_T = np.full(n_stages, float(T))
    if init_xi is None:
        init_xi = np.zeros((n_stages, max(R_chem, 1)))

    w = pack(np.asarray(init_x_R, dtype=float),
             np.asarray(init_x_E, dtype=float),
             np.asarray(init_R, dtype=float),
             np.asarray(init_E, dtype=float),
             np.asarray(init_T, dtype=float),
             np.asarray(init_xi, dtype=float))

    F_curr = residuals(w)
    norm_curr = float(np.max(np.abs(F_curr)))
    converged = norm_curr < tol
    last_iter = 0

    for newton_iter in range(max_newton_iter):
        last_iter = newton_iter
        if verbose:
            print(f"  LLE-NS iter {newton_iter}: ||F||_inf = {norm_curr:.3e}")
        if norm_curr < tol:
            converged = True
            break

        J = np.zeros((n_total, n_total))
        for k in range(n_total):
            h_k = max(fd_step * abs(w[k]), fd_step)
            w_p = w.copy(); w_p[k] += h_k
            w_m = w.copy(); w_m[k] -= h_k
            J[:, k] = (residuals(w_p) - residuals(w_m)) / (2.0 * h_k)

        try:
            dw = np.linalg.solve(J, -F_curr)
        except np.linalg.LinAlgError:
            dw, *_ = np.linalg.lstsq(J, -F_curr, rcond=None)

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

    x_R, x_E, R_arr, E_arr, T_arr, xi_arr = unpack(w)

    flows_positive = bool((R_arr > 0).all() and (E_arr > 0).all())
    phase_split_max = float(np.max(np.abs(x_R - x_E)))
    collapsed = phase_split_max < 1e-3
    pathology = (not flows_positive) or collapsed
    if pathology:
        converged = False
        why = []
        if not flows_positive:
            why.append(f"non-positive flow (R_min={R_arr.min():.2e}, "
                       f"E_min={E_arr.min():.2e})")
        if collapsed:
            why.append(f"phases collapsed (max|x_R - x_E|={phase_split_max:.2e})")
        msg = ("LLE-NS converged to non-physical state: "
               + "; ".join(why)
               + ".  This typically means the overall (F+S) mixture "
               "lies outside the binodal at this T -- there is no "
               "two-phase column operation point.  Try a different "
               "S/F ratio, lower T, or different solvent.")
    else:
        msg = (f"LLE-NS converged in {last_iter + 1} Newton iters, "
               f"||F||={norm_curr:.2e}"
               if converged else
               f"LLE-NS did not converge in {max_newton_iter} iters, "
               f"||F||={norm_curr:.2e}")

    return ExtractionColumnResult(
        converged=converged, iterations=last_iter + 1,
        n_stages=n_stages, species_names=tuple(species_names),
        T=T_arr.copy(), x_R=x_R, x_E=x_E, R=R_arr, E=E_arr,
        xi=xi_arr[:, :R_chem].copy() if R_chem > 0 else np.zeros((n_stages, 0)),
        F=F, S=S, z_F=z_F.copy(), z_S=z_S.copy(),
        reactive_stages=tuple(sorted(reactive_set)),
        reaction_phase=reaction_phase,
        energy_balance=energy_balance,
        message=msg,
    )
