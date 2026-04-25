"""UNIFAC-Lyngby (Larsen-Rasmussen-Fredenslund 1987) modified UNIFAC variant.

Differs from original UNIFAC in:

1. **Simplified combinatorial term** using 2/3 power:

       omega_i = x_i r_i^(2/3) / sum_j x_j r_j^(2/3)
       ln gamma_i^C = ln(omega_i / x_i) + 1 - omega_i / x_i

   No surface-area q term -- entirely volume-based. This was found to
   give better VLE for asymmetric alkane/alcohol mixtures than original
   UNIFAC.

2. **Temperature-dependent group interaction parameters** with three
   coefficients per main-group pair:

       Psi_mn(T) = exp(-(a_mn + b_mn (T - T_ref)
                          + c_mn (T ln(T_ref / T) + T - T_ref)) / T)

   with T_ref = 298.15 K.

The Lyngby parameter table differs from both original UNIFAC and
Dortmund. As with Dortmund, the full database is not bundled (commercial
provenance via DDBST). Defaults to original UNIFAC database with
b = c = 0; result is then "UNIFAC with Lyngby combinatorial only".

Reference: Larsen, B.L., Rasmussen, P., Fredenslund, A. "A Modified
UNIFAC Group-Contribution Model for Prediction of Phase Equilibria and
Heats of Mixing", Ind. Eng. Chem. Res. 26, 2274 (1987).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from .unifac import UNIFAC


class UNIFAC_Lyngby(UNIFAC):
    """Modified UNIFAC (Lyngby) activity coefficient model.

    Parameters
    ----------
    subgroups_per_component : list of dict
        Same format as UNIFAC.
    database : module-like, optional
        Object with SUBGROUPS, A_MAIN dicts plus optional B_MAIN, C_MAIN.
        Defaults to original UNIFAC database with b=c=0.
    T_ref : float
        Reference temperature for T-dependence (default 298.15 K).
    """

    T_REF = 298.15

    def __init__(self, subgroups_per_component, database=None,
                 T_ref: float = None):
        super().__init__(subgroups_per_component, database)
        if T_ref is not None:
            self.T_REF = float(T_ref)
        # 2/3 power for combinatorial
        self._r_23 = self._r ** (2.0 / 3.0)
        # Build B, C lookups (default zero)
        n_sg = self._n_sg
        b_lookup = np.zeros((n_sg, n_sg))
        c_lookup = np.zeros((n_sg, n_sg))
        B_db = getattr(self._db, 'B_MAIN', None)
        C_db = getattr(self._db, 'C_MAIN', None)
        for i_sg, mi in enumerate(self._main_of_sg):
            for j_sg, mj in enumerate(self._main_of_sg):
                if int(mi) == int(mj):
                    continue
                if B_db is not None:
                    try:
                        b_lookup[i_sg, j_sg] = B_db[int(mi)][int(mj)]
                    except KeyError:
                        pass
                if C_db is not None:
                    try:
                        c_lookup[i_sg, j_sg] = C_db[int(mi)][int(mj)]
                    except KeyError:
                        pass
        self._b_lookup = b_lookup
        self._c_lookup = c_lookup

    def _Psi(self, T: float) -> np.ndarray:
        """Lyngby T-dependent: Psi_mn = exp(-(a + b(T-T_ref) + c(...))/T)
        where the c term is c (T ln(T_ref/T) + T - T_ref)."""
        T = float(T)
        T_ref = self.T_REF
        # The Lyngby c term is c_mn * (T*ln(T_ref/T) + T - T_ref)
        c_term = self._c_lookup * (T * np.log(T_ref / T) + T - T_ref)
        b_term = self._b_lookup * (T - T_ref)
        return np.exp(-(self._a_lookup + b_term + c_term) / T)

    def _dPsi_dT(self, T: float) -> np.ndarray:
        """For Lyngby:
            f(T) = a + b(T - T_ref) + c (T ln(T_ref/T) + T - T_ref)
            ln Psi = -f / T
            f'(T) = b + c (ln(T_ref/T) - 1 + 1) = b + c ln(T_ref/T)
              [since d/dT (T ln(T_ref/T) + T - T_ref) = ln(T_ref/T) + T*(-1/T) + 1
                                                       = ln(T_ref/T)]
            d(ln Psi)/dT = -f'/T + f/T^2
            dPsi/dT = Psi * [-f'(T)/T + f(T)/T^2]
        """
        T = float(T)
        T_ref = self.T_REF
        Psi = self._Psi(T)
        # f(T)
        c_term = self._c_lookup * (T * np.log(T_ref / T) + T - T_ref)
        b_term = self._b_lookup * (T - T_ref)
        f_T = self._a_lookup + b_term + c_term
        # f'(T)
        f_prime = self._b_lookup + self._c_lookup * np.log(T_ref / T)
        return Psi * (-f_prime / T + f_T / (T * T))

    def _ln_combinatorial(self, x: np.ndarray) -> np.ndarray:
        """Lyngby simplified combinatorial:
        ln gamma_i^C = ln(omega_i / x_i) + 1 - omega_i / x_i
        where omega_i = x_i r_i^(2/3) / sum_j x_j r_j^(2/3),
        so omega_i / x_i = r_i^(2/3) / sum_j x_j r_j^(2/3).
        """
        sum_xr_23 = float(np.sum(x * self._r_23))
        omega_over_x = self._r_23 / sum_xr_23
        return np.log(omega_over_x) + 1.0 - omega_over_x
