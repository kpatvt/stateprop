"""Reactive absorber column for amine-CO2 systems (v0.9.104).

Couples the v0.9.103 carbamate AmineSystem to a multi-stage absorber
column.  Each stage assumes thermodynamic equilibrium between vapor
CO2 and liquid amine (the standard reactive-absorption simplification:
chemistry is fast compared to mass transfer, so each tray is at
chemical equilibrium internally — the only finite-rate process is
inter-stage transfer).

The column is solved by a stage-by-stage Newton-Raphson iteration on
the loading α_n and vapor composition y_n profiles.

Geometry and conventions
------------------------
* N stages, numbered 1 (top) to N (bottom)
* Liquid (amine) flows down: enters top as L_0 (lean, α=α_lean),
  leaves bottom as L_N (rich, α=α_rich)
* Vapor (CO2 + inert) flows up: enters bottom as V_{N+1} (high CO2
  mole fraction y_{N+1}), leaves top as V_1 (cleaned, low y_1)
* Each stage n: vapor exiting (y_n) is in chemical equilibrium with
  liquid exiting (α_n) at the stage temperature T_n (theoretical stage)
* Mass balance per stage:
      L · α_{n-1} + G · y_{n+1} = L · α_n + G · y_n
  where L, G are molar flow rates (assumed constant — dilute approx)
* Equilibrium relation:
      y_n = P_CO2(α_n, T_n) / P_total

Algorithm
---------
We have 2N unknowns: α_n and y_n for n = 1..N.  Two equations per
stage (equilibrium + mass balance) gives 2N equations.

Newton iteration:
1. Reduce to N unknowns by using equilibrium to eliminate y_n in favor
   of α_n.  Then mass balance per stage becomes:
        L · α_{n-1} + G · y_{n+1} - L · α_n - G · y_eq(α_n, T_n) = 0
2. Solve this banded N x N system for α_n.
3. Each Jacobian element involves d(y_eq)/dα, computed by finite
   difference (cheap because each speciation is fast).

For typical absorbers (10-30 stages), this converges in 5-15 Newton
iterations.

Engineering inputs
------------------
For absorber design, the user typically specifies:
    L (liquid molar flow rate, mol/s)
    G (vapor molar flow rate, mol/s)
    α_lean (lean liquid loading, mol/mol)
    y_N+1 (inlet vapor CO2 mole fraction)
    P (total pressure, bar)
    T_n profile (or constant T)
    N (number of theoretical stages)
    total_amine (liquid amine concentration, mol/kg solvent)

Outputs:
    α_rich (rich liquid loading, mol/mol)
    y_1 (cleaned vapor CO2 mole fraction)
    α_n, y_n profiles through the column
    CO2 recovery = (y_{N+1} - y_1) / y_{N+1}

References
----------
* Kister, H. Z. (1992).  Distillation Design.  McGraw-Hill.
* Stichlmair, J., Fair, J. (1998).  Distillation: Principles and
  Practice.  Wiley-VCH.
* Posey, M. L. (1996).  Thermodynamic model for acidic gases loaded
  aqueous alkanolamine solutions.  PhD thesis, UT Austin.
* Aspen RateSep / ProTreat: industrial rate-based codes that this
  module's equilibrium-stage approach is the simpler counterpart to.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np

from .amines import Amine, AmineSystem, AmineSpeciationResult, lookup_amine


_R = 8.314462618


# =====================================================================
# Result dataclass
# =====================================================================

@dataclass
class AmineColumnResult:
    """Result of an AmineColumn.solve() call.

    Attributes
    ----------
    alpha : list of float
        Liquid loading [mol CO2 / mol amine] at each stage exit
        (length N, alpha[0] = stage-1 loading, alpha[N-1] = α_rich).
    y : list of float
        Vapor CO2 mole fraction at each stage exit (length N,
        y[0] = y_1 = cleaned gas, y[N-1] = y_N at bottom).
    T : list of float
        Stage temperatures [K].
    alpha_rich : float
        Rich liquid loading exiting the bottom (= alpha[-1]).
    y_top : float
        Vapor CO2 mole fraction exiting the top (= y[0]).
    co2_recovery : float
        Fraction of inlet CO2 absorbed: (y_in - y_top) / y_in.
    pH : list of float
        Liquid pH at each stage (from speciation).
    converged : bool
    iterations : int
    L : float
        Liquid molar flow rate [mol/s].
    G : float
        Vapor molar flow rate [mol/s].
    LG_ratio : float
        L/G ratio (key absorber design parameter).
    """
    alpha: List[float]
    y: List[float]
    T: List[float]
    alpha_rich: float
    y_top: float
    co2_recovery: float
    pH: List[float]
    converged: bool
    iterations: int
    L: float
    G: float
    LG_ratio: float

    def speciation_at_stage(self, n: int) -> AmineSpeciationResult:
        """Return full speciation result at stage n (1-indexed)."""
        if not (1 <= n <= len(self.alpha)):
            raise IndexError(f"stage {n} out of range 1..{len(self.alpha)}")
        # Re-speciate at this α, T (cached if column stored systems internally)
        return None    # placeholder; see solve() for full speciation


# =====================================================================
# AmineColumn class
# =====================================================================

class AmineColumn:
    """Multi-stage absorber column for amine-CO2 systems.

    Solves a counter-current absorber where liquid amine flows down
    and gas (CO2 + inert) flows up, with theoretical equilibrium
    stages.  Uses :class:`AmineSystem` for the chemical equilibrium
    at each stage.

    Parameters
    ----------
    amine : Amine or str
        The alkanolamine.  String is looked up in the bundled DB.
    total_amine : float
        Total amine concentration [mol/kg solvent], on the liquid feed.
    n_stages : int
        Number of theoretical equilibrium stages.

    Examples
    --------
    >>> from stateprop.electrolyte import AmineColumn
    >>> # Typical post-combustion CO2 capture absorber:
    >>> # 12% CO2 in flue gas, 30 wt% MEA, 40 °C, 20 stages
    >>> col = AmineColumn("MEA", total_amine=5.0, n_stages=20)
    >>> r = col.solve(
    ...     L=10.0,           # liquid molar flow [mol/s]
    ...     G=15.0,           # vapor molar flow [mol/s]
    ...     alpha_lean=0.20,
    ...     y_in=0.12,        # 12% CO2 inlet
    ...     P=1.013,          # 1 atm
    ...     T=313.15,         # isothermal 40 °C
    ... )
    >>> r.alpha_rich
    0.485
    >>> r.co2_recovery
    0.92
    """

    def __init__(self,
                  amine: Union[Amine, str],
                  total_amine: float,
                  n_stages: int):
        self.amine = (lookup_amine(amine) if isinstance(amine, str)
                       else amine)
        self.total_amine = float(total_amine)
        self.n_stages = int(n_stages)
        if self.n_stages < 1:
            raise ValueError("n_stages must be >= 1")
        # Internal AmineSystem for equilibrium calls
        self._sys = AmineSystem(self.amine, self.total_amine)

    # -----------------------------------------------------------------
    # Equilibrium relation: y_eq(α, T, P)
    # -----------------------------------------------------------------
    def _y_eq(self, alpha: float, T: float, P: float) -> float:
        """Equilibrium CO2 mole fraction in vapor at given α and T.

        y* = P_CO2(α, T) / P_total.
        """
        if alpha <= 1e-9:
            return 0.0
        try:
            res = self._sys.speciate(alpha=alpha, T=T)
            return res.P_CO2 / P
        except Exception:
            return 1.0     # blow up signal

    # -----------------------------------------------------------------
    # Forward solve
    # -----------------------------------------------------------------
    def solve(self,
                L: float,
                G: float,
                alpha_lean: float,
                y_in: float,
                P: float = 1.013,
                T: Union[float, Sequence[float]] = 313.15,
                adiabatic: bool = False,
                T_liquid_in: Optional[float] = None,
                T_gas_in: Optional[float] = None,
                wt_frac_amine: float = 0.30,
                cp_gas: float = 33.0,
                max_iter: int = 100,
                tol: float = 1e-8,
                verbose: bool = False) -> AmineColumnResult:
        """Solve the absorber column for steady-state α and y profiles.

        Two operating modes:

        **Isothermal (default, ``adiabatic=False``):** T profile fixed
        by user (constant or per-stage); only α_n is the Newton
        unknown.  Fast convergence (5-15 iterations).

        **Adiabatic (``adiabatic=True``):** T profile is solved jointly
        from per-stage energy balance.  Captures the "absorber
        temperature bulge" (exothermic CO₂ absorption heats the liquid
        on its way down; peak T typically 10-15 K above feed for
        30 wt% MEA at typical conditions).  Newton solves for
        (α_n, T_n) coupled.  Convergence in 10-30 iterations.

        Parameters
        ----------
        L : float
            Liquid molar flow rate [mol amine/s].  Constant assumed.
        G : float
            Vapor molar flow rate [mol/s] (mostly inert + CO2).
        alpha_lean : float
            Lean liquid loading at top inlet [mol CO2 / mol amine].
        y_in : float
            Inlet vapor CO2 mole fraction at the bottom.
        P : float, default 1.013 bar
            Total pressure.
        T : float or sequence, default 313.15 K
            Stage temperatures for **isothermal** mode (ignored if
            ``adiabatic=True``).
        adiabatic : bool, default False
            If True, solve T profile from energy balance.  Requires
            T_liquid_in and T_gas_in.
        T_liquid_in : float, optional
            Liquid feed temperature [K] for adiabatic mode.  Defaults
            to first element of T or 313.15.
        T_gas_in : float, optional
            Gas feed temperature [K] for adiabatic mode.  Defaults to
            T_liquid_in (typical absorber assumption).
        wt_frac_amine : float, default 0.30
            For energy balance: weight fraction of amine in solvent.
        cp_gas : float, default 33.0 J/(mol·K)
            Average vapor heat capacity (typical for N₂+CO₂ mix).
        max_iter, tol, verbose : Newton settings
        """
        # Handle adiabatic mode separately for clarity
        if adiabatic:
            return self._solve_adiabatic(
                L=L, G=G, alpha_lean=alpha_lean, y_in=y_in,
                P=P, T_liquid_in=T_liquid_in, T_gas_in=T_gas_in,
                wt_frac_amine=wt_frac_amine, cp_gas=cp_gas,
                max_iter=max_iter, tol=tol, verbose=verbose)

        # Isothermal mode (original v0.9.104 logic):
        N = self.n_stages
        L_amine = L * (self.amine.MW * 0.001)   # not used; L is in moles
        # We work in the convention where:
        #   L is molar flow of liquid amine (i.e., moles amine/s)
        #   G is molar flow of total vapor (i.e., moles vapor/s)
        # Mass balance for CO2: L · α + G · y is conserved across each
        # stage.  Operating line slope is L/G in (Δy/Δα) units.

        if np.isscalar(T):
            Ts = [float(T)] * N
        else:
            Ts = list(T)
            if len(Ts) != N:
                raise ValueError(f"T profile must have {N} entries")

        # Initial guess: linear loading from α_lean to α_rich
        # Estimate α_rich from overall mass balance, assuming y_top ≈ 0
        alpha_rich_guess = alpha_lean + (G / L) * y_in
        alpha_rich_guess = min(alpha_rich_guess, 0.95)
        alpha = np.linspace(alpha_lean, alpha_rich_guess, N + 1)[1:]
        # alpha[0] = stage-1 loading, alpha[N-1] = stage-N loading

        converged = False
        prev_norm = np.inf
        for outer in range(max_iter):
            # Compute equilibrium y_n at each stage from current α_n
            y = np.array([self._y_eq(alpha[n], Ts[n], P) for n in range(N)])

            # Mass balance residual at each stage:
            #   L · α_{n-1} + G · y_{n+1} = L · α_n + G · y_n
            # boundary: α_0 = α_lean (above stage 1), y_{N+1} = y_in
            # F_n = L · α_{n-1} + G · y_{n+1} - L · α_n - G · y_n
            F = np.zeros(N)
            for n in range(N):
                a_above = alpha_lean if n == 0 else alpha[n - 1]
                y_below = y_in if n == N - 1 else y[n + 1]
                F[n] = (L * a_above + G * y_below
                         - L * alpha[n] - G * y[n])

            norm_F = float(np.linalg.norm(F, ord=np.inf))
            scale = max(L * abs(alpha_rich_guess - alpha_lean),
                          G * y_in, 1e-12)
            if verbose:
                print(f"  iter {outer:3d}: ||F||_∞ = {norm_F:.3e}, "
                       f"scaled = {norm_F/scale:.3e}")

            if norm_F / scale < tol:
                converged = True
                break

            # Build Jacobian J[i,j] = ∂F_i / ∂α_j  (banded tridiagonal)
            # F_i = L · α_{i-1} + G · y_{i+1}(α_{i+1}) - L · α_i - G · y_i(α_i)
            # ∂F_i/∂α_{i-1} = +L
            # ∂F_i/∂α_i     = -L - G · dy_i/dα_i
            # ∂F_i/∂α_{i+1} = +G · dy_{i+1}/dα_{i+1}
            J = np.zeros((N, N))
            eps = 1e-5
            dy_dalpha = np.zeros(N)
            for n in range(N):
                a_perturb = alpha[n] + eps
                if a_perturb >= 1.0:
                    a_perturb = alpha[n] - eps
                    dy_dalpha[n] = ((self._y_eq(alpha[n], Ts[n], P)
                                       - self._y_eq(a_perturb, Ts[n], P))
                                       / eps)
                else:
                    dy_dalpha[n] = ((self._y_eq(a_perturb, Ts[n], P)
                                       - self._y_eq(alpha[n], Ts[n], P))
                                       / eps)
            for i in range(N):
                if i > 0:
                    J[i, i - 1] = +L
                J[i, i] = -L - G * dy_dalpha[i]
                if i < N - 1:
                    J[i, i + 1] = +G * dy_dalpha[i + 1]

            try:
                dalpha = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # Fall back to damped fixed-point
                dalpha = -0.1 * F / max(np.linalg.norm(F), 1e-12)

            # Damp step to keep α in [0, 0.95]
            damp = 1.0
            new_alpha = alpha + damp * dalpha
            while np.any(new_alpha < 0) or np.any(new_alpha > 0.95):
                damp *= 0.5
                new_alpha = alpha + damp * dalpha
                if damp < 1e-6:
                    break
            # Cap step in α-space to avoid wild excursions
            max_step = float(np.max(np.abs(damp * dalpha)))
            if max_step > 0.2:
                damp *= 0.2 / max_step
                new_alpha = alpha + damp * dalpha
            alpha = new_alpha

            # Detect divergence
            if outer > 5 and norm_F > 10 * prev_norm:
                # Reset to a uniform initial guess
                alpha = np.linspace(alpha_lean,
                                       alpha_lean + 0.5 * (G/L) * y_in,
                                       N + 1)[1:]
            prev_norm = norm_F

        # Final speciation pass to get pH and other outputs
        y_final = [self._y_eq(alpha[n], Ts[n], P) for n in range(N)]
        pH_final = []
        for n in range(N):
            try:
                res = self._sys.speciate(alpha=alpha[n], T=Ts[n])
                pH_final.append(res.pH)
            except Exception:
                pH_final.append(float("nan"))

        recovery = (y_in - y_final[0]) / y_in if y_in > 0 else 0.0

        return AmineColumnResult(
            alpha=list(alpha), y=y_final, T=Ts,
            alpha_rich=float(alpha[-1]),
            y_top=float(y_final[0]),
            co2_recovery=float(recovery),
            pH=pH_final,
            converged=converged, iterations=outer + 1,
            L=L, G=G, LG_ratio=L / G,
        )

    # -----------------------------------------------------------------
    # Adiabatic solver: T_n unknown, energy balance per stage
    # -----------------------------------------------------------------
    def _solve_adiabatic(self,
                            L: float, G: float,
                            alpha_lean: float, y_in: float,
                            P: float = 1.013,
                            T_liquid_in: Optional[float] = None,
                            T_gas_in: Optional[float] = None,
                            wt_frac_amine: float = 0.30,
                            cp_gas: float = 33.0,
                            max_iter: int = 100,
                            tol: float = 1e-8,
                            verbose: bool = False) -> AmineColumnResult:
        """Adiabatic absorber: T_n solved from per-stage energy balance.

        Newton system: 2N unknowns x = [α_1..α_N, T_1..T_N]
                       2N equations: F_n = mass balance, E_n = energy
                                       balance (both = 0).

        Energy balance per stage (interior):
            L · cp_L · (T_n - T_above) + G · cp_V · (T_n - T_below)
              + L · ΔH_abs · (α_above - α_n) = 0

        where α_above and T_above come from above (stage n-1, or
        feed for stage 1), and T_below from below (stage n+1, or
        gas feed for stage N).  ΔH_abs is the integral heat of
        absorption (negative); rearranged so positive Δα releases heat.

        Boundary conditions:
            α_0 = alpha_lean,  T_above_stage_1 = T_liquid_in
            y_{N+1} = y_in,    T_below_stage_N = T_gas_in
        """
        N = self.n_stages
        if T_liquid_in is None:
            T_liquid_in = 313.15
        if T_gas_in is None:
            T_gas_in = T_liquid_in

        # Solvent thermal mass per mol amine
        # cp_L (per mol amine) = cp_solvent · (kg solvent / mol amine)
        cp_sol_per_kg = self.amine.cp_solution(wt_frac_amine, T_liquid_in)
        kg_sol_per_mol_amine = (self.amine.MW * 1e-3) / wt_frac_amine
        cp_L_per_mol = cp_sol_per_kg * kg_sol_per_mol_amine    # J/(mol amine · K)

        delta_H_abs = self.amine.delta_H_abs    # J/mol CO2 (negative)

        # Initial guess: linear α, T from feed temperature
        alpha_rich_guess = alpha_lean + (G / L) * y_in
        alpha_rich_guess = min(alpha_rich_guess, 0.95)
        alpha = np.linspace(alpha_lean, alpha_rich_guess, N + 1)[1:]
        # T initial: linearly bumped up by ~10 K (typical bulge)
        T = np.linspace(T_liquid_in, T_liquid_in + 10.0, N)

        # Newton variables: x = [α_1, ..., α_N, T_1, ..., T_N]
        x = np.concatenate([alpha, T])

        converged = False
        for outer in range(max_iter):
            alpha = x[:N]
            T = x[N:]

            # Compute equilibrium y_n at each stage
            y = np.zeros(N)
            for n in range(N):
                y[n] = self._y_eq(alpha[n], T[n], P)

            # Mass-balance residuals  (CO2 only, water/amine carrier
            # treated as constant)
            F = np.zeros(N)
            for n in range(N):
                a_above = alpha_lean if n == 0 else alpha[n - 1]
                y_below = y_in if n == N - 1 else y[n + 1]
                F[n] = (L * a_above + G * y_below
                         - L * alpha[n] - G * y[n])

            # Energy-balance residuals
            # E_n = L·cp_L·(T_n - T_above) + G·cp_V·(T_n - T_below)
            #         - L · |ΔH_abs| · (α_n - α_above)
            # (ΔH_abs < 0, and Δα > 0 going down — heat released raises T)
            E = np.zeros(N)
            for n in range(N):
                T_above = T_liquid_in if n == 0 else T[n - 1]
                T_below = T_gas_in if n == N - 1 else T[n + 1]
                a_above = alpha_lean if n == 0 else alpha[n - 1]
                E[n] = (L * cp_L_per_mol * (T[n] - T_above)
                         + G * cp_gas * (T[n] - T_below)
                         - L * abs(delta_H_abs) * (alpha[n] - a_above))

            R = np.concatenate([F, E])
            scale_F = max(L * abs(alpha_rich_guess - alpha_lean),
                            G * y_in, 1e-12)
            scale_E = L * cp_L_per_mol * 10.0 + 1e-9
            norm = max(float(np.max(np.abs(F))) / scale_F,
                          float(np.max(np.abs(E))) / scale_E)
            if verbose:
                print(f"  iter {outer:3d}: ||F||={np.max(np.abs(F)):.2e}, "
                       f"||E||={np.max(np.abs(E)):.2e}, "
                       f"scaled = {norm:.2e}")
            if norm < tol:
                converged = True
                break

            # Numerical Jacobian (2N x 2N)
            J = np.zeros((2 * N, 2 * N))
            eps_a = 1e-6
            eps_T = 1e-3   # K
            for j in range(N):
                # Perturb α_j
                xp = x.copy()
                xp[j] += eps_a
                Rp = self._adiabatic_residual(xp, N, alpha_lean, y_in,
                                                  T_liquid_in, T_gas_in,
                                                  L, G, cp_L_per_mol,
                                                  cp_gas, delta_H_abs, P)
                J[:, j] = (Rp - R) / eps_a
                # Perturb T_j
                xp = x.copy()
                xp[N + j] += eps_T
                Rp = self._adiabatic_residual(xp, N, alpha_lean, y_in,
                                                  T_liquid_in, T_gas_in,
                                                  L, G, cp_L_per_mol,
                                                  cp_gas, delta_H_abs, P)
                J[:, N + j] = (Rp - R) / eps_T

            try:
                dx = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dx = -0.1 * R / max(np.linalg.norm(R), 1e-12)

            # Damp step
            damp = 1.0
            new_x = x + damp * dx
            # Constraints: α in [0, 0.95], T > T_feed - 50
            T_floor = min(T_liquid_in, T_gas_in) - 5.0
            T_ceil = max(T_liquid_in, T_gas_in) + 80.0
            while (np.any(new_x[:N] < 0) or np.any(new_x[:N] > 0.95)
                    or np.any(new_x[N:] < T_floor)
                    or np.any(new_x[N:] > T_ceil)):
                damp *= 0.5
                new_x = x + damp * dx
                if damp < 1e-6:
                    break
            # Cap step in α (max 0.2) and T (max 10 K)
            d_alpha_max = float(np.max(np.abs(damp * dx[:N])))
            d_T_max = float(np.max(np.abs(damp * dx[N:])))
            if d_alpha_max > 0.2:
                damp = min(damp, 0.2 / d_alpha_max)
            if d_T_max > 10.0:
                damp = min(damp, 10.0 / d_T_max)
            x = x + damp * dx

        # Final pass
        alpha = x[:N].tolist()
        T = x[N:].tolist()
        y_final = [self._y_eq(alpha[n], T[n], P) for n in range(N)]
        pH_final = []
        for n in range(N):
            try:
                res = self._sys.speciate(alpha=alpha[n], T=T[n])
                pH_final.append(res.pH)
            except Exception:
                pH_final.append(float("nan"))

        recovery = (y_in - y_final[0]) / y_in if y_in > 0 else 0.0
        return AmineColumnResult(
            alpha=alpha, y=y_final, T=T,
            alpha_rich=float(alpha[-1]),
            y_top=float(y_final[0]),
            co2_recovery=float(recovery),
            pH=pH_final,
            converged=converged, iterations=outer + 1,
            L=L, G=G, LG_ratio=L / G,
        )

    def _adiabatic_residual(self, x, N, alpha_lean, y_in,
                              T_liquid_in, T_gas_in, L, G,
                              cp_L_per_mol, cp_gas, delta_H_abs, P):
        """Helper for finite-difference Jacobian: re-evaluate residuals."""
        alpha = x[:N]
        T = x[N:]
        y = np.array([self._y_eq(alpha[n], T[n], P) for n in range(N)])
        F = np.zeros(N)
        E = np.zeros(N)
        for n in range(N):
            a_above = alpha_lean if n == 0 else alpha[n - 1]
            y_below = y_in if n == N - 1 else y[n + 1]
            F[n] = (L * a_above + G * y_below
                     - L * alpha[n] - G * y[n])
            T_above = T_liquid_in if n == 0 else T[n - 1]
            T_below = T_gas_in if n == N - 1 else T[n + 1]
            E[n] = (L * cp_L_per_mol * (T[n] - T_above)
                     + G * cp_gas * (T[n] - T_below)
                     - L * abs(delta_H_abs) * (alpha[n] - a_above))
        return np.concatenate([F, E])

    # -----------------------------------------------------------------
    def stages_for_recovery(self,
                              L: float,
                              G: float,
                              alpha_lean: float,
                              y_in: float,
                              target_recovery: float,
                              P: float = 1.013,
                              T: float = 313.15,
                              max_stages: int = 50) -> int:
        """Find the smallest number of stages giving the target CO2 recovery.

        Parameters
        ----------
        target_recovery : float
            Required (y_in - y_top) / y_in fraction (e.g., 0.90 for 90%).
        max_stages : int, default 50

        Returns
        -------
        int
            Smallest integer N such that solve(...) gives recovery >=
            target_recovery, or max_stages if not achievable.
        """
        for N_try in range(1, max_stages + 1):
            self.n_stages = N_try
            r = self.solve(L=L, G=G, alpha_lean=alpha_lean,
                              y_in=y_in, P=P, T=T)
            if r.co2_recovery >= target_recovery:
                return N_try
        return max_stages


# =====================================================================
# Convenience: equilibrium curve generator
# =====================================================================

def amine_equilibrium_curve(amine: Union[Amine, str],
                              total_amine: float,
                              alpha_range: Sequence[float],
                              T: float = 313.15,
                              P: float = 1.013) -> List[float]:
    """Generate the y* vs α equilibrium curve for an amine system.

    Useful for McCabe-Thiele construction or absorber tray sizing.

    Parameters
    ----------
    amine : Amine or str
    total_amine : float
        Liquid amine concentration [mol/kg solvent].
    alpha_range : sequence of float
        Loadings α at which to evaluate y*.
    T : float, default 313.15 (40 °C)
    P : float, default 1.013 (1 atm)

    Returns
    -------
    list of float
        y* values at each α.
    """
    sys = AmineSystem(amine, total_amine)
    out = []
    for a in alpha_range:
        if a <= 1e-9:
            out.append(0.0)
            continue
        try:
            r = sys.speciate(alpha=a, T=T)
            out.append(r.P_CO2 / P)
        except Exception:
            out.append(float("nan"))
    return out
