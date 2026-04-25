"""UNIFAC group-contribution activity coefficient model.

Predicts activity coefficients from the molecular structure (number
of subgroups per component) without requiring binary interaction
data fits. Built on top of UNIQUAC's combinatorial+residual
decomposition, with the residual term computed via Lyngby's
solution-of-groups concept:

    ln gamma_i^R = sum_k nu_k^(i) [ln Gamma_k - ln Gamma_k^(i)]

where Gamma_k is the residual group activity coefficient in the
mixture and Gamma_k^(i) is in pure-component-i environment.

Each Gamma_k follows a UNIQUAC-like residual formula applied to
groups instead of molecules, with group-group interaction
parameters Psi_mn = exp(-a_mn / T) where a_mn is the main-group
interaction parameter (in K).

Parameters from `unifac_database.SUBGROUPS` and
`unifac_database.A_MAIN`. Users specifying their own subgroups can
extend the database, OR pass per-component subgroup-counts and
custom R/Q/A tables directly to the constructor.

Reference:
    Fredenslund, A., Jones, R.L., Prausnitz, J.M., AIChE J. 21,
    1086 (1975).
    Hansen, H.K. et al., Ind. Eng. Chem. Res. 30, 2352 (1991).
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np

from . import unifac_database as _db
from .excess import GibbsExcessTDerivatives


class UNIFAC(GibbsExcessTDerivatives):
    """UNIFAC activity-coefficient model (group contribution).

    Parameters
    ----------
    subgroups_per_component : list of dict
        For each component, a dict {group_name: count}. Group names
        must exist in `stateprop.activity.unifac_database.SUBGROUPS`
        (or a custom database passed via `database`).
        Example: water = {'H2O': 1}, ethanol = {'CH3': 1, 'CH2': 1, 'OH': 1}.
    database : module-like, optional
        Object with `SUBGROUPS` dict and `A_MAIN` dict, defaults to
        the built-in `unifac_database`. Custom databases must follow
        the same structure.

    Notes
    -----
    Z = 10 coordination number, same as UNIQUAC.
    Group-group Psi_mn = exp(-a_mn / T) with a_mn in K (already
    pre-divided by R in published tables).
    """

    Z_COORD = 10.0

    def __init__(self, subgroups_per_component, database=None):
        self._db = database if database is not None else _db
        self._N = len(subgroups_per_component)
        self._subgroup_specs = subgroups_per_component
        # Build the union of subgroups across all components, indexed.
        all_subgroup_names = set()
        for spec in subgroups_per_component:
            for name in spec:
                if name not in self._db.SUBGROUPS:
                    raise KeyError(f"Unknown subgroup '{name}'")
                all_subgroup_names.add(name)
        self._sg_names = sorted(all_subgroup_names,
                                 key=lambda n: self._db.SUBGROUPS[n][0])
        self._n_sg = len(self._sg_names)
        # Per-component subgroup count matrix nu[i, k] = nu_k^(i)
        nu = np.zeros((self._N, self._n_sg))
        for i, spec in enumerate(subgroups_per_component):
            for name, count in spec.items():
                k = self._sg_names.index(name)
                nu[i, k] = float(count)
        self._nu = nu
        # R_k, Q_k arrays for the indexed subgroups
        R_sg = np.array([self._db.SUBGROUPS[n][2] for n in self._sg_names])
        Q_sg = np.array([self._db.SUBGROUPS[n][3] for n in self._sg_names])
        self._R_sg = R_sg
        self._Q_sg = Q_sg
        # Main-group index for each subgroup
        self._main_of_sg = np.array([self._db.SUBGROUPS[n][1]
                                      for n in self._sg_names], dtype=int)
        # Build the (n_sg, n_sg) a_mn lookup matrix at the subgroup level
        a_lookup = np.zeros((self._n_sg, self._n_sg))
        for i_sg, mi in enumerate(self._main_of_sg):
            for j_sg, mj in enumerate(self._main_of_sg):
                if mi == mj:
                    a_lookup[i_sg, j_sg] = 0.0
                else:
                    try:
                        a_lookup[i_sg, j_sg] = self._db.A_MAIN[int(mi)][int(mj)]
                    except KeyError:
                        raise KeyError(
                            f"Missing UNIFAC interaction parameter "
                            f"a_{mi},{mj} for subgroup pair "
                            f"({self._sg_names[i_sg]}, {self._sg_names[j_sg]})"
                        )
        self._a_lookup = a_lookup
        # UNIQUAC-style r_i, q_i for each component (sum of group contributions)
        self._r = nu @ R_sg            # (N,)
        self._q = nu @ Q_sg            # (N,)
        self._l = (self.Z_COORD / 2.0) * (self._r - self._q) - (self._r - 1.0)
        # Pure-component group activity coefficients ln Gamma_k^(i) computed
        # lazily per T (we just need to solve the group residual at
        # x_pure_i = e_i; can be cached per (T, i) pair).
        self._lnGamma_pure_cache = {}

    @property
    def N(self) -> int:
        return self._N

    # ---- Pickle support: drop the module reference (not picklable) -----

    def __getstate__(self):
        state = self.__dict__.copy()
        # Module objects can't be pickled; database is only used in __init__.
        # Drop the reference; restoration sets it to None which is harmless
        # since all lookups are already pre-built (a_lookup, R_sg, etc).
        state['_db'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def r(self) -> np.ndarray:
        return self._r.copy()

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    def _Psi(self, T: float) -> np.ndarray:
        """Subgroup-level Psi_mn = exp(-a_mn / T)."""
        return np.exp(-self._a_lookup / float(T))

    def _ln_group_gammas(self, T: float, X_grp: np.ndarray) -> np.ndarray:
        """Group residual activity coefficients ln Gamma_k for a given
        group mole fraction array X_grp[k] of length n_sg.

        Uses UNIQUAC-style residual formula on the group level.
        """
        Psi = self._Psi(T)                         # (n_sg, n_sg)
        Q = self._Q_sg
        # Theta_k = X_k Q_k / sum X_l Q_l
        denom = float(np.sum(X_grp * Q))
        theta = X_grp * Q / denom                  # (n_sg,)
        # ln Gamma_k = Q_k [1 - ln(sum_m theta_m Psi_mk)
        #                   - sum_m (theta_m Psi_km / sum_n theta_n Psi_nm)]
        # Same structure as UNIQUAC residual.
        S = theta @ Psi                            # (n_sg,) -- sum_n theta_n Psi_nm for each m
        # ln(sum_m theta_m Psi_mk) at k = (theta @ Psi.T)[k] -- but theta @ Psi already
        # gives sum_n theta_n Psi[n, m] for each m. Substituting m → k: same vector.
        # So sum_m theta_m Psi_mk = S[k].
        # Term2: sum_m (theta_m Psi_km / sum_n theta_n Psi_nm) = sum_m (Psi_km theta_m / S_m)
        # = (Psi @ (theta / S))[k]
        term2 = Psi @ (theta / S)                  # (n_sg,)
        return Q * (1.0 - np.log(S) - term2)

    def _ln_group_gammas_pure(self, T: float, i: int) -> np.ndarray:
        """ln Gamma_k^(i) for pure component i at temperature T."""
        cache_key = (round(float(T), 8), i)
        if cache_key in self._lnGamma_pure_cache:
            return self._lnGamma_pure_cache[cache_key]
        # Pure component: x = e_i. Group composition given by row i of nu.
        nu_i = self._nu[i]
        total = float(np.sum(nu_i))
        if total == 0.0:
            res = np.zeros(self._n_sg)
        else:
            X_pure = nu_i / total                  # (n_sg,) group mole fractions
            res = self._ln_group_gammas(T, X_pure)
        self._lnGamma_pure_cache[cache_key] = res
        return res

    def _ln_combinatorial(self, x: np.ndarray) -> np.ndarray:
        """Combinatorial activity coefficient (UNIQUAC form, original UNIFAC)."""
        sum_xr = float(np.sum(x * self._r))
        sum_xq = float(np.sum(x * self._q))
        phi_over_x = self._r / sum_xr
        theta_over_phi = (self._q / self._r) * (sum_xr / sum_xq)
        sum_xl = float(np.sum(x * self._l))
        return (np.log(phi_over_x)
                + (self.Z_COORD / 2.0) * self._q * np.log(theta_over_phi)
                + self._l
                - phi_over_x * sum_xl)

    def _ln_residual(self, T: float, x: np.ndarray) -> np.ndarray:
        """Residual activity coefficient via solution-of-groups."""
        N = self._N
        # Mixture group mole fractions
        nu_per_x = (x[:, None] * self._nu).sum(axis=0)
        total = float(nu_per_x.sum())
        X_mix = nu_per_x / total
        lnGamma_mix = self._ln_group_gammas(T, X_mix)
        lngamma_R = np.empty(N)
        for i in range(N):
            lnGamma_pure_i = self._ln_group_gammas_pure(T, i)
            lngamma_R[i] = float(np.sum(self._nu[i] * (lnGamma_mix - lnGamma_pure_i)))
        return lngamma_R

    def lngammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size != self._N:
            raise ValueError(f"x must have length {self._N}")
        return self._ln_combinatorial(x) + self._ln_residual(T, x)

    # -----------------------------------------------------------------
    # Analytical T-derivatives (override FD mixin defaults)
    # -----------------------------------------------------------------

    def _dPsi_dT(self, T: float) -> np.ndarray:
        """d(Psi_mn)/dT for the original UNIFAC form Psi = exp(-a/T):
            dPsi/dT = Psi * a / T^2
        Modified UNIFAC variants override this with their own form.
        """
        T = float(T)
        Psi = self._Psi(T)
        return Psi * self._a_lookup / (T * T)

    def _d_ln_group_gammas_dT(self, T: float,
                                X_grp: np.ndarray) -> np.ndarray:
        """d(ln Gamma_k)/dT at fixed group composition X_grp.

        ln Gamma_k = Q_k [1 - ln(S_k) - sum_m theta_m Psi_km / S_m]
            where S_m = sum_n theta_n Psi_nm, theta = X*Q / sum(X*Q),
            theta is T-independent.

        d(ln Gamma_k)/dT = Q_k [-dS_k/dT/S_k
                                  - sum_m theta_m (dPsi_km/S_m
                                                    - Psi_km dS_m/S_m^2)]
        """
        Psi = self._Psi(T)
        dPsi = self._dPsi_dT(T)
        Q = self._Q_sg
        denom = float(np.sum(X_grp * Q))
        theta = X_grp * Q / denom
        S = theta @ Psi              # (n_sg,)  S_m
        dS = theta @ dPsi            # dS_m/dT
        # Term: -dS_k/dT / S_k
        term_A = -dS / S
        # Term: sum_m theta_m dPsi_km / S_m
        first = dPsi @ (theta / S)
        # Term: sum_m theta_m Psi_km dS_m / S_m^2
        second = Psi @ (theta * dS / (S * S))
        return Q * (term_A - first + second)

    def _d_ln_group_gammas_pure_dT(self, T: float, i: int) -> np.ndarray:
        """d(ln Gamma_k^(i))/dT for pure component i."""
        nu_i = self._nu[i]
        total = float(np.sum(nu_i))
        if total == 0.0:
            return np.zeros(self._n_sg)
        X_pure = nu_i / total
        return self._d_ln_group_gammas_dT(T, X_pure)

    def dlngammas_dT(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Analytical d(ln gamma_i)/dT for UNIFAC.

        Combinatorial is T-independent so contributes zero. Residual:
            ln gamma_i^R = sum_k nu_k^i (ln Gamma_k^mix - ln Gamma_k^pure_i)
        d(ln gamma_i^R)/dT = sum_k nu_k^i (dln Gamma_k^mix/dT
                                              - dln Gamma_k^pure_i/dT)
        """
        x = np.asarray(x, dtype=float)
        if x.size != self._N:
            raise ValueError(f"x must have length {self._N}")
        # Mixture group mole fractions
        nu_per_x = (x[:, None] * self._nu).sum(axis=0)
        total = float(nu_per_x.sum())
        X_mix = nu_per_x / total
        d_lnGamma_mix = self._d_ln_group_gammas_dT(T, X_mix)

        d_lng_R = np.empty(self._N)
        for i in range(self._N):
            d_lnGamma_pure_i = self._d_ln_group_gammas_pure_dT(T, i)
            d_lng_R[i] = float(np.sum(self._nu[i]
                                       * (d_lnGamma_mix - d_lnGamma_pure_i)))
        return d_lng_R   # combinatorial contributes 0

    def gammas(self, T: float, x: Sequence[float]) -> np.ndarray:
        return np.exp(self.lngammas(T, x))

    def gE_over_RT(self, T: float, x: Sequence[float]) -> float:
        """Excess Gibbs / RT via gE = sum x_i ln gamma_i."""
        return float(np.sum(np.asarray(x) * self.lngammas(T, x)))
