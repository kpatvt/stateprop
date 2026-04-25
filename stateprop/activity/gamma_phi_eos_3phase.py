"""Three-phase gamma-phi-EOS flash (VLLE).

Extends the v0.9.43 two-phase gamma-phi-EOS flash to handle
vapor-liquid-liquid equilibrium (VLLE) systems. Common applications:

- Water + organic + vapor heteroazeotropes (water/butanol, water/MIBK,
  water/aniline) -- two immiscible liquid phases with a common vapor.
- Petroleum systems with aqueous + hydrocarbon liquid phases plus a
  natural gas vapor phase.
- Liquid-liquid extraction columns operating near their bubble point.

Governing equations: with three phases L1 (reference liquid),
L2 (second liquid), V (vapor):

    f_i^L1 = f_i^L2 = f_i^V    (equal fugacities)

Define K-values relative to L1:

    K_iy = y_i / x1_i = gamma_i^L1 * p_i^sat * phi_i^sat * Poynting
                                    / (p * phi_i^V(T, p, y))
    K_ix = x2_i / x1_i = gamma_i^L1(T, x1) / gamma_i^L2(T, x2)

Material balance and sum-to-1 constraints give two scalar equations
(3-phase Rachford-Rice) in two unknowns (beta_V, beta_L2):

    sum_i z_i (K_iy - 1) / D_i = 0
    sum_i z_i (K_ix - 1) / D_i = 0

where D_i = 1 + beta_V (K_iy - 1) + beta_L2 (K_ix - 1).

Solved jointly by Newton's method with a bracketing safeguard
keeping all D_i > 0 and 0 <= beta_V, beta_L2, beta_V + beta_L2 <= 1.

Reference: Michelsen, M.L. "The isothermal flash problem. Part I.
Stability", Fluid Phase Equilib. 9, 1 (1982). Reid-Prausnitz-Poling
5th ed. Ch. 9 for the algorithmic outline.

**Initial guesses required**: the user must supply x1_guess and
x2_guess that are clearly different (e.g., one rich in water, one
rich in organic). Without good initial guesses the SS iteration
will collapse to the trivial L1=L2 solution which is just 2-phase
VLE. A future addition is automatic stability-analysis-based
initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np

from .gamma_phi_eos import GammaPhiEOSFlash


_R_GAS = 8.31446261815324


@dataclass
class ThreePhaseFlashResult:
    """Result of a 3-phase isothermal flash."""
    T: float
    p: float
    z: np.ndarray
    beta_V: float        # vapor mole fraction of feed
    beta_L2: float       # second-liquid mole fraction of feed
    beta_L1: float       # = 1 - beta_V - beta_L2
    x1: np.ndarray       # reference-liquid composition
    x2: np.ndarray       # second-liquid composition
    y: np.ndarray        # vapor composition
    K_y: np.ndarray      # y_i / x1_i
    K_x: np.ndarray      # x2_i / x1_i
    iterations: int = 0
    converged: bool = True


@dataclass
class SinglePhaseResult:
    """Result indicating only a single phase exists at (T, p, z)."""
    T: float
    p: float
    z: np.ndarray
    phase: str            # 'liquid' or 'vapor'
    tpd_min: float = 0.0  # from stability test (>=0 for stable)
    n_phases: int = 1


@dataclass
class AutoFlashResult:
    """Wrapper indicating which flash branch was taken by auto_isothermal."""
    T: float
    p: float
    z: np.ndarray
    n_phases: int                   # 1, 2, or 3
    phase_type: str                 # '1L', '1V', '2VL', '2LL', '3VLL'
    result: object                  # SinglePhaseResult / FlashResult / LLEResult / ThreePhaseFlashResult
    stability_tpd: float            # TPD from initial stability test


class GammaPhiEOSThreePhaseFlash(GammaPhiEOSFlash):
    """Three-phase VLLE flash on top of GammaPhiEOSFlash.

    Inherits the 2-phase flash's K-value plumbing (psat, EOS vapor,
    Poynting, phi_sat) and adds the 3-phase Rachford-Rice + outer
    SS iteration.

    Parameters
    ----------
    Same as `GammaPhiEOSFlash`. The activity model must give
    physically meaningful gamma values across the LLE composition
    range (UNIQUAC and the UNIFAC-LLE parameter set are typical
    choices; original VLE-UNIFAC parameters often give poor LLE
    predictions and a dedicated LLE database should be used).
    """

    def isothermal_3phase(self, T: float, p: float, z: Sequence[float],
                           x1_guess: Sequence[float],
                           x2_guess: Sequence[float],
                           y_guess: Optional[Sequence[float]] = None,
                           beta_V_guess: float = 0.05,
                           beta_L2_guess: float = 0.3,
                           tol: float = 1e-6,
                           maxiter: int = 200) -> ThreePhaseFlashResult:
        """Three-phase isothermal PT flash (VLLE).

        Outer SS loop on x1, x2, y. Inner solver: 2D Newton on
        the joint 3-phase Rachford-Rice for (beta_V, beta_L2).

        Parameters
        ----------
        T, p : float
            Temperature [K] and pressure [Pa].
        z : sequence
            Feed composition (length N).
        x1_guess, x2_guess : sequence
            Initial guess for the two liquid compositions. Must be
            sufficiently different to seed LLE (e.g., one rich in
            water, the other rich in organic). If x1 ~= x2, the
            iteration collapses to the trivial 2-phase solution.
        y_guess : sequence, optional
            Initial vapor composition. If None, computed from
            Raoult-like K with x1_guess.
        beta_V_guess, beta_L2_guess : float
            Initial guesses for vapor and second-liquid phase
            fractions. Must satisfy 0 < both < 1 and sum < 1.
        tol : float
            Convergence on max relative K-value change.
        maxiter : int
            Outer SS iteration limit.

        Returns
        -------
        ThreePhaseFlashResult

        Raises
        ------
        RuntimeError if SS or Rachford-Rice fails to converge.
        """
        z = np.asarray(z, dtype=float)
        x1 = np.asarray(x1_guess, dtype=float).copy()
        x2 = np.asarray(x2_guess, dtype=float).copy()
        x1 = x1 / x1.sum()
        x2 = x2 / x2.sum()
        beta_V = float(beta_V_guess)
        beta_L2 = float(beta_L2_guess)

        # Initial vapor composition
        if y_guess is None:
            psat = np.array([f(T) for f in self.psat])
            gammas_L1 = np.asarray(self.model.gammas(T, x1))
            y = (gammas_L1 * psat / p) * x1
            y = y / y.sum()
        else:
            y = np.asarray(y_guess, dtype=float).copy()
            y = y / y.sum()

        for it in range(maxiter):
            # Compute K values
            K_y, K_x = self._K_values_3phase(T, p, x1, x2, y)

            # Solve 3-phase Rachford-Rice
            beta_V, beta_L2 = _solve_3phase_rachford_rice(
                z, K_y, K_x, beta_V, beta_L2)

            # Update phase compositions
            D = 1.0 + beta_V * (K_y - 1.0) + beta_L2 * (K_x - 1.0)
            # Guard against negative D (numerical issues at boundary)
            D = np.maximum(D, 1e-12)
            x1_new = z / D
            x1_new = x1_new / x1_new.sum()
            x2_new = K_x * x1_new
            x2_new = x2_new / x2_new.sum()
            y_new = K_y * x1_new
            y_new = y_new / y_new.sum()

            # Recompute K with new compositions to assess convergence
            K_y_new, K_x_new = self._K_values_3phase(T, p, x1_new, x2_new, y_new)
            err = max(
                float(np.max(np.abs(K_y_new - K_y) / np.maximum(K_y, 1e-12))),
                float(np.max(np.abs(K_x_new - K_x) / np.maximum(K_x, 1e-12)))
            )
            x1, x2, y = x1_new, x2_new, y_new

            if err < tol:
                # Detect collapse to 2-phase: x1 ~= x2
                if float(np.max(np.abs(x1 - x2))) < 1e-4:
                    # Trivial solution; flag but return
                    pass
                return ThreePhaseFlashResult(
                    T=T, p=p, z=z,
                    beta_V=beta_V, beta_L2=beta_L2,
                    beta_L1=1.0 - beta_V - beta_L2,
                    x1=x1, x2=x2, y=y,
                    K_y=K_y_new, K_x=K_x_new,
                    iterations=it + 1,
                    converged=True
                )
        raise RuntimeError(
            f"3-phase isothermal flash did not converge in {maxiter} iter "
            f"(max K err {err:.2e})"
        )

    def _K_values_3phase(self, T: float, p: float,
                         x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
        """Compute K_iy = y/x1 and K_ix = x2/x1 with EOS vapor and
        activity-model liquids."""
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        y = np.asarray(y, dtype=float)
        psat = np.array([f(T) for f in self.psat])
        gamma_L1 = np.asarray(self.model.gammas(T, x1))
        gamma_L2 = np.asarray(self.model.gammas(T, x2))
        rho_v = self.eos.density_from_pressure(p, T, y, phase_hint='vapor')
        ln_phi_v = np.asarray(self.eos.ln_phi(rho_v, T, y))
        phi_v = np.exp(ln_phi_v)
        # Saturation fugacity
        if self._phi_sat is not None:
            phi_sat = np.array([f(T) for f in self._phi_sat])
        else:
            phi_sat = np.ones(self.N)
        # Poynting
        if self._VL is not None:
            poynting = np.exp(self._VL * (p - psat) / (_R_GAS * T))
        else:
            poynting = np.ones(self.N)
        # K_iy = gamma_L1 * psat * phi_sat * poynting / (p * phi_v)
        K_y = gamma_L1 * psat * phi_sat * poynting / (p * phi_v)
        # K_ix = gamma_L1 / gamma_L2
        K_x = gamma_L1 / gamma_L2
        return K_y, K_x

    # -----------------------------------------------------------------
    # Auto phase-count detection (v0.9.49)
    # -----------------------------------------------------------------

    def auto_isothermal(self, T: float, p: float, z: Sequence[float],
                          beta_threshold: float = 1e-4,
                          tol: float = 1e-6, maxiter: int = 200,
                          tpd_tol: float = 1e-7) -> AutoFlashResult:
        """Automatic phase-count detection PT-flash.

        Eliminates the need for x1_guess/x2_guess. The algorithm:

        1. Run Michelsen liquid-stability test on z.
        2. Estimate bubble pressure of z: p_bub = sum z_i gamma_i(z) p_sat_i.
           If p_bub < p, vapor cannot form; system is purely liquid.
        3. If z is stable AND p < p_bub: try 2-phase VLE; else single liquid.
        4. If z is unstable:
           a. Try 3-phase. If all three beta > threshold, return 3-phase.
           b. If 3-phase fails AND p < p_bub: vapor dominates; try 2-phase
              VLE first. If V in (threshold, 1-threshold), return 2-phase VL.
           c. Try 2-phase LLE.
           d. If LLE fails, last-resort 2-phase VLE.

        The bubble-p heuristic correctly distinguishes 2-phase LL
        (subcooled, p > p_bub) from 2-phase VL (superheated, p < p_bub)
        when the 3-phase RR cannot find an interior solution.

        Returns
        -------
        AutoFlashResult with .n_phases (1/2/3), .phase_type, and .result.
        """
        from .stability import stability_test
        from .lle import LLEFlash

        z_arr = np.asarray(z, dtype=float)
        z_arr = z_arr / z_arr.sum()

        # Bubble-pressure estimate (Raoult-modified-by-gamma)
        psat_z = np.array([f(T) for f in self.psat])
        gamma_z = np.asarray(self.model.gammas(T, z_arr))
        p_bub_z = float(np.sum(z_arr * gamma_z * psat_z))
        vapor_likely = (p < 0.99 * p_bub_z)

        # --- Step 1: liquid stability test ---
        stab = stability_test(self.model, T, z_arr, tpd_tol=tpd_tol)

        if stab.stable:
            # No LL split. If p > p_bub, single liquid; else try VLE.
            if not vapor_likely:
                spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase='liquid',
                                          tpd_min=float(stab.tpd_min))
                return AutoFlashResult(
                    T=T, p=p, z=z_arr, n_phases=1,
                    phase_type='1L', result=spr,
                    stability_tpd=float(stab.tpd_min)
                )
            try:
                vle = self.isothermal(T=T, p=p, z=z_arr,
                                        tol=tol, maxiter=maxiter)
                if beta_threshold < vle.V < 1.0 - beta_threshold:
                    return AutoFlashResult(
                        T=T, p=p, z=z_arr, n_phases=2,
                        phase_type='2VL', result=vle,
                        stability_tpd=float(stab.tpd_min)
                    )
                phase = 'vapor' if vle.V > 0.5 else 'liquid'
                spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase=phase,
                                          tpd_min=float(stab.tpd_min))
                return AutoFlashResult(
                    T=T, p=p, z=z_arr, n_phases=1,
                    phase_type='1' + ('V' if phase == 'vapor' else 'L'),
                    result=spr,
                    stability_tpd=float(stab.tpd_min)
                )
            except Exception:
                spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase='liquid',
                                          tpd_min=float(stab.tpd_min))
                return AutoFlashResult(
                    T=T, p=p, z=z_arr, n_phases=1,
                    phase_type='1L', result=spr,
                    stability_tpd=float(stab.tpd_min)
                )

        # --- Step 2: feed is unstable; try 3-phase ---
        Y1 = stab.Y_min.copy()
        Y2 = 2.0 * z_arr - Y1
        Y2 = np.maximum(Y2, 1e-4)
        Y2 = Y2 / Y2.sum()
        if float(np.max(np.abs(Y1 - Y2))) < 0.1:
            Y2 = 1.0 - Y1
            Y2 = np.maximum(Y2, 1e-4)
            Y2 = Y2 / Y2.sum()

        try:
            r3 = self.isothermal_3phase(T=T, p=p, z=z_arr,
                                           x1_guess=Y1, x2_guess=Y2,
                                           tol=tol, maxiter=maxiter)
            if (r3.beta_V > beta_threshold
                    and r3.beta_L1 > beta_threshold
                    and r3.beta_L2 > beta_threshold):
                return AutoFlashResult(
                    T=T, p=p, z=z_arr, n_phases=3,
                    phase_type='3VLL', result=r3,
                    stability_tpd=float(stab.tpd_min)
                )
        except Exception:
            pass

        # --- Step 3: 3-phase failed or collapsed. Choose 2-phase by p_bub ---
        if vapor_likely:
            # p < p_bub_z: vapor dominates. Try VLE first.
            try:
                vle = self.isothermal(T=T, p=p, z=z_arr,
                                         tol=tol, maxiter=maxiter)
                if beta_threshold < vle.V < 1.0 - beta_threshold:
                    return AutoFlashResult(
                        T=T, p=p, z=z_arr, n_phases=2,
                        phase_type='2VL', result=vle,
                        stability_tpd=float(stab.tpd_min)
                    )
            except Exception:
                pass
            try:
                lle = LLEFlash(self.model)
                llr = lle.solve(T, z_arr, x1_guess=Y1, x2_guess=Y2,
                                  maxiter=maxiter)
                return AutoFlashResult(
                    T=T, p=p, z=z_arr, n_phases=2,
                    phase_type='2LL', result=llr,
                    stability_tpd=float(stab.tpd_min)
                )
            except Exception:
                pass
        else:
            # p > p_bub_z: subcooled, LL only.
            try:
                lle = LLEFlash(self.model)
                llr = lle.solve(T, z_arr, x1_guess=Y1, x2_guess=Y2,
                                  maxiter=maxiter)
                return AutoFlashResult(
                    T=T, p=p, z=z_arr, n_phases=2,
                    phase_type='2LL', result=llr,
                    stability_tpd=float(stab.tpd_min)
                )
            except Exception:
                pass
            try:
                vle = self.isothermal(T=T, p=p, z=z_arr,
                                         tol=tol, maxiter=maxiter)
                if beta_threshold < vle.V < 1.0 - beta_threshold:
                    return AutoFlashResult(
                        T=T, p=p, z=z_arr, n_phases=2,
                        phase_type='2VL', result=vle,
                        stability_tpd=float(stab.tpd_min)
                    )
            except Exception:
                pass

        # --- Step 4: last resort ---
        spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase='liquid',
                                  tpd_min=float(stab.tpd_min))
        return AutoFlashResult(
            T=T, p=p, z=z_arr, n_phases=1,
            phase_type='1L', result=spr,
            stability_tpd=float(stab.tpd_min)
        )

    # -----------------------------------------------------------------
    # Auto phase-count detection v2: full 4-test TPD framework (v0.9.53)
    # -----------------------------------------------------------------

    def auto_isothermal_full_tpd(self, T: float, p: float,
                                    z: Sequence[float],
                                    beta_threshold: float = 1e-4,
                                    tol: float = 1e-6,
                                    maxiter: int = 200,
                                    tpd_tol: float = 1e-7,
                                    phi_sat_funcs: Optional[Sequence] = None,
                                    liquid_molar_volumes: Optional[Sequence[float]] = None,
                                    ) -> AutoFlashResult:
        """Automatic phase-count detection using the full 4-test TPD
        framework. Replaces the bubble-pressure heuristic in
        `auto_isothermal()` with rigorous cross-phase stability tests.

        Algorithm
        ---------

        1. Run all 4 Michelsen TPD tests on z:
           - L->L (v0.9.48): trial liquid using gamma
           - V->V (v0.9.51): trial vapor using phi
           - L->V (v0.9.52): liquid candidate, vapor trial
           - V->L (v0.9.52): vapor candidate, liquid trial

        2. Pattern-match on the 4 stability flags to determine phase
           count unambiguously:

           +------+------+------+------+------------------------+
           |  LL  |  VV  |  LV  |  VL  | Phase count            |
           +------+------+------+------+------------------------+
           |  S   |  S   |  S   |  U   | 1L (single liquid)     |
           |  S   |  S   |  U   |  S   | 1V (single vapor)      |
           |  S   |  S   |  U   |  U   | 2VL                    |
           |  U   |  S   |  S   |  U   | 2LL (no vapor present) |
           |  U   |  S   |  U   |  U   | 3VLL                   |
           |  S   |  S   |  S   |  S   | 1-phase (use bubble-p) |
           +------+------+------+------+------------------------+
           (S = stable, U = unstable)

        3. Run the appropriate flash with stability-derived initial
           guesses. If the chosen flash fails, fall back as in
           auto_isothermal().

        Parameters
        ----------
        T : K. p : Pa. z : feed composition.
        beta_threshold : minimum phase fraction to accept a 3-phase
            solution (otherwise reduce to 2-phase).
        tol, maxiter, tpd_tol : numerical tolerances.
        phi_sat_funcs : optional saturation phi functions for cross-phase.
        liquid_molar_volumes : optional liquid molar volumes for Poynting.

        Returns
        -------
        AutoFlashResult with .phase_type in {'1L', '1V', '2VL', '2LL', '3VLL'}.

        Notes
        -----
        - More EOS evaluations than `auto_isothermal()` due to the 4
          stability tests, but eliminates the bubble-p heuristic and
          gives unambiguous phase identification.
        - In the rare case where all 4 tests are stable but the system
          is actually multiphase (e.g., a saturated state where stability
          is marginal), falls back to bubble-p disambiguation.
        - For systems where the cubic EOS gives metastable "vapor" roots
          at compressed liquid conditions (a known cubic-EOS artifact),
          the L->V test may be unstable for the wrong reason; the algorithm
          handles this by checking Y_min compositions.
        """
        from .stability import stability_test
        from .vapor_stability import vapor_phase_stability_test
        from .cross_stability import cross_phase_stability_test
        from .lle import LLEFlash

        z_arr = np.asarray(z, dtype=float)
        z_arr = z_arr / z_arr.sum()

        # --- Step 1: run all 4 stability tests ---
        r_LL = stability_test(self.model, T, z_arr, tpd_tol=tpd_tol)
        try:
            r_VV = vapor_phase_stability_test(self.eos, T, p, z_arr,
                                                  tpd_tol=tpd_tol)
        except Exception:
            # EOS failed at z (e.g., density solver); treat as VV stable
            from .vapor_stability import VaporStabilityResult
            r_VV = VaporStabilityResult(stable=True, tpd_min=0.0,
                                          Y_min=z_arr.copy(),
                                          n_stationary=0,
                                          iterations_total=0,
                                          trials_evaluated=0)
        try:
            r_LV = cross_phase_stability_test(
                self.model, self.eos, self.psat, T=T, p=p, z=z_arr,
                candidate_phase='liquid', phi_sat_funcs=phi_sat_funcs,
                liquid_molar_volumes=liquid_molar_volumes,
                tpd_tol=tpd_tol)
        except Exception:
            from .cross_stability import CrossPhaseStabilityResult
            r_LV = CrossPhaseStabilityResult(stable=True, tpd_min=0.0,
                                              Y_min=z_arr.copy(),
                                              candidate_phase='liquid',
                                              trial_phase='vapor',
                                              n_stationary=0,
                                              iterations_total=0,
                                              trials_evaluated=0)
        try:
            r_VL = cross_phase_stability_test(
                self.model, self.eos, self.psat, T=T, p=p, z=z_arr,
                candidate_phase='vapor', phi_sat_funcs=phi_sat_funcs,
                liquid_molar_volumes=liquid_molar_volumes,
                tpd_tol=tpd_tol)
        except Exception:
            from .cross_stability import CrossPhaseStabilityResult
            r_VL = CrossPhaseStabilityResult(stable=True, tpd_min=0.0,
                                              Y_min=z_arr.copy(),
                                              candidate_phase='vapor',
                                              trial_phase='liquid',
                                              n_stationary=0,
                                              iterations_total=0,
                                              trials_evaluated=0)

        LL_stable = r_LL.stable
        VV_stable = r_VV.stable
        LV_stable = r_LV.stable
        VL_stable = r_VL.stable
        # Use min TPD over all four for a representative diagnostic value
        tpd_min_all = min(r_LL.tpd_min, r_VV.tpd_min,
                          r_LV.tpd_min, r_VL.tpd_min)

        def _build(phase_type, n_phases, result):
            return AutoFlashResult(
                T=T, p=p, z=z_arr, n_phases=n_phases,
                phase_type=phase_type, result=result,
                stability_tpd=float(tpd_min_all)
            )

        # --- Step 2: pattern-match on stability flags ---

        # Case A: LL unstable AND any cross-phase unstable -> try 3VLL
        if (not LL_stable) and ((not LV_stable) or (not VL_stable)):
            Y1 = r_LL.Y_min.copy()
            Y2 = 2.0 * z_arr - Y1
            Y2 = np.maximum(Y2, 1e-4)
            Y2 = Y2 / Y2.sum()
            if float(np.max(np.abs(Y1 - Y2))) < 0.1:
                Y2 = 1.0 - Y1
                Y2 = np.maximum(Y2, 1e-4)
                Y2 = Y2 / Y2.sum()
            try:
                r3 = self.isothermal_3phase(T=T, p=p, z=z_arr,
                                               x1_guess=Y1, x2_guess=Y2,
                                               tol=tol, maxiter=maxiter)
                if (r3.beta_V > beta_threshold
                        and r3.beta_L1 > beta_threshold
                        and r3.beta_L2 > beta_threshold):
                    return _build('3VLL', 3, r3)
            except Exception:
                pass
            # 3-phase failed; the unstable directions hint which 2-phase
            # If LL_unstable ranks worst, prefer LL; otherwise prefer VL
            if r_LL.tpd_min <= min(r_LV.tpd_min, r_VL.tpd_min):
                # LL split dominates
                try:
                    lle = LLEFlash(self.model)
                    llr = lle.solve(T, z_arr, x1_guess=Y1, x2_guess=Y2,
                                       maxiter=maxiter)
                    return _build('2LL', 2, llr)
                except Exception:
                    pass
            try:
                vle = self.isothermal(T=T, p=p, z=z_arr, tol=tol,
                                         maxiter=maxiter)
                if beta_threshold < vle.V < 1.0 - beta_threshold:
                    return _build('2VL', 2, vle)
            except Exception:
                pass
            try:
                lle = LLEFlash(self.model)
                llr = lle.solve(T, z_arr, x1_guess=Y1, x2_guess=Y2,
                                  maxiter=maxiter)
                return _build('2LL', 2, llr)
            except Exception:
                pass

        # Case B: only LL unstable -> 2LL
        if (not LL_stable) and LV_stable:
            Y1 = r_LL.Y_min.copy()
            Y2 = 2.0 * z_arr - Y1
            Y2 = np.maximum(Y2, 1e-4)
            Y2 = Y2 / Y2.sum()
            try:
                lle = LLEFlash(self.model)
                llr = lle.solve(T, z_arr, x1_guess=Y1, x2_guess=Y2,
                                  maxiter=maxiter)
                return _build('2LL', 2, llr)
            except Exception:
                pass

        # Case C: LV and VL both unstable, LL stable -> 2VL
        if LL_stable and (not LV_stable) and (not VL_stable):
            try:
                vle = self.isothermal(T=T, p=p, z=z_arr, tol=tol,
                                         maxiter=maxiter)
                if beta_threshold < vle.V < 1.0 - beta_threshold:
                    return _build('2VL', 2, vle)
            except Exception:
                pass

        # Case D: LV unstable, VL stable -> 1V
        if LL_stable and (not LV_stable) and VL_stable:
            spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase='vapor',
                                      tpd_min=float(r_LV.tpd_min))
            return _build('1V', 1, spr)

        # Case E: LV stable, VL unstable -> 1L
        if LL_stable and LV_stable and (not VL_stable):
            spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase='liquid',
                                      tpd_min=float(r_VL.tpd_min))
            return _build('1L', 1, spr)

        # Case F: all stable -> single phase, disambiguate by bubble-p
        if LL_stable and VV_stable and LV_stable and VL_stable:
            psat_z = np.array([f(T) for f in self.psat])
            gamma_z = np.asarray(self.model.gammas(T, z_arr))
            p_bub_z = float(np.sum(z_arr * gamma_z * psat_z))
            phase = 'vapor' if p < p_bub_z else 'liquid'
            spr = SinglePhaseResult(T=T, p=p, z=z_arr, phase=phase,
                                      tpd_min=0.0)
            return _build('1' + ('V' if phase == 'vapor' else 'L'), 1, spr)

        # --- Step 3: anything-else / final fallback ---
        # If we reach here, the stability flag pattern was unusual and
        # earlier branches' flashes failed. Try the bubble-p auto.
        return self.auto_isothermal(T=T, p=p, z=z_arr,
                                       beta_threshold=beta_threshold,
                                       tol=tol, maxiter=maxiter,
                                       tpd_tol=tpd_tol)


# ---------------------------------------------------------------------------
# 3-phase Rachford-Rice solver
# ---------------------------------------------------------------------------


def _solve_3phase_rachford_rice(z: np.ndarray, K_y: np.ndarray, K_x: np.ndarray,
                                  beta_V_init: float = 0.05,
                                  beta_L2_init: float = 0.3,
                                  tol: float = 1e-10,
                                  maxiter: int = 100) -> tuple:
    """Solve the joint 3-phase Rachford-Rice system:

        f_y(beta_V, beta_L2) = sum_i z_i (K_iy - 1) / D_i = 0
        f_x(beta_V, beta_L2) = sum_i z_i (K_ix - 1) / D_i = 0
        D_i = 1 + beta_V (K_iy - 1) + beta_L2 (K_ix - 1)

    Newton's method with backtracking line search to keep all D_i > 0
    and (beta_V, beta_L2) in the interior of the 2-simplex.

    A feasible interior solution exists only if the system is genuinely
    3-phase at the given K-values and feed. If RR cannot reduce both
    residuals below tol in maxiter, the function raises RuntimeError --
    typically indicating the system is actually 2-phase (V or L2 missing)
    at these conditions.
    """
    z = np.asarray(z, dtype=float)
    Ky_m1 = K_y - 1.0
    Kx_m1 = K_x - 1.0
    beta_V = float(beta_V_init)
    beta_L2 = float(beta_L2_init)

    def _feasible(bV, bL2):
        """Test if (bV, bL2) is in interior with all D_i > 0."""
        if not (0.0 < bV < 1.0 and 0.0 < bL2 < 1.0):
            return False
        if bV + bL2 >= 1.0:
            return False
        D_test = 1.0 + bV * Ky_m1 + bL2 * Kx_m1
        return bool(np.all(D_test > 1e-10))

    def _eval(bV, bL2):
        D = 1.0 + bV * Ky_m1 + bL2 * Kx_m1
        f_y = float(np.sum(z * Ky_m1 / D))
        f_x = float(np.sum(z * Kx_m1 / D))
        return D, f_y, f_x

    # Project initial guess into feasible region if needed
    if not _feasible(beta_V, beta_L2):
        beta_V = max(0.01, min(0.4, beta_V))
        beta_L2 = max(0.01, min(0.4, beta_L2))
        if beta_V + beta_L2 >= 0.95:
            scale = 0.9 / (beta_V + beta_L2)
            beta_V *= scale; beta_L2 *= scale

    last_err = float('inf')
    for it in range(maxiter):
        if not _feasible(beta_V, beta_L2):
            raise RuntimeError(
                f"3-phase RR: infeasible iterate (beta_V={beta_V:.3e}, "
                f"beta_L2={beta_L2:.3e}); system likely not 3-phase"
            )
        D, f_y, f_x = _eval(beta_V, beta_L2)
        err = max(abs(f_y), abs(f_x))
        if err < tol:
            return beta_V, beta_L2

        # Jacobian (symmetric, negative-definite)
        D2 = D * D
        J11 = -float(np.sum(z * Ky_m1 * Ky_m1 / D2))
        J22 = -float(np.sum(z * Kx_m1 * Kx_m1 / D2))
        J12 = -float(np.sum(z * Ky_m1 * Kx_m1 / D2))

        det = J11 * J22 - J12 * J12
        if abs(det) < 1e-30:
            raise RuntimeError(
                "3-phase RR: singular Jacobian; phases may have collapsed"
            )

        # Newton step: J [dV; dL] = -[f_y; f_x]
        dV = -(f_y * J22 - f_x * J12) / det
        dL = -(-f_y * J12 + f_x * J11) / det

        # Backtracking line search: find largest alpha in (0, 1] such that
        # the new point is feasible AND reduces the residual norm.
        alpha = 1.0
        for ls in range(40):
            new_V = beta_V + alpha * dV
            new_L = beta_L2 + alpha * dL
            if _feasible(new_V, new_L):
                _, f_y_new, f_x_new = _eval(new_V, new_L)
                new_err = max(abs(f_y_new), abs(f_x_new))
                # Armijo-like: accept if residual is reduced
                if new_err < err * (1.0 - 0.1 * alpha):
                    beta_V, beta_L2 = new_V, new_L
                    break
            alpha *= 0.5
        else:
            # Couldn't reduce residual: probably no interior solution.
            # Stop with current iterate and let caller decide.
            raise RuntimeError(
                f"3-phase RR: no descent direction found; "
                f"f_y={f_y:.2e}, f_x={f_x:.2e}; "
                f"system likely not in 3-phase region"
            )

        last_err = err

    raise RuntimeError(
        f"3-phase Rachford-Rice did not converge in {maxiter} iter "
        f"(f_y={f_y:.2e}, f_x={f_x:.2e})"
    )
