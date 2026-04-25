"""UNIFAC-Dortmund (Weidlich-Gmehling 1987) modified UNIFAC variant.

Two principal differences from original UNIFAC:

1. **Modified combinatorial term**: uses 3/4 power volume fraction
   to better handle alkane/alcohol asymmetry:

       V'_i = r_i^(3/4) / sum_j x_j r_j^(3/4)
       V_i  = r_i / sum_j x_j r_j
       F_i  = q_i / sum_j x_j q_j
       ln gamma_i^C = 1 - V'_i + ln V'_i - 5 q_i [1 - V_i/F_i + ln(V_i/F_i)]

2. **Temperature-dependent group interaction parameters**:

       Psi_mn(T) = exp(-(a_mn + b_mn T + c_mn T^2) / T)

   with a_mn, b_mn, c_mn fit per main-group pair to a wide-T VLE database.

The Dortmund group-interaction parameter set differs from original UNIFAC
and is maintained by DDBST (Dortmund Data Bank Software & Separation
Technology). The parameter table is partly commercial (full database not
free); users should provide their own custom database via the
`database` argument or extend the built-in starter database.

When the database lacks B_MAIN/C_MAIN, those default to zeros. The
result is "UNIFAC with modified Dortmund combinatorial only" -- still
useful for demonstrating the combinatorial improvement, but loses
accuracy gains from refit residual parameters.

Reference: Weidlich, U., Gmehling, J. "A Modified UNIFAC Model. 1.
Prediction of VLE, hE, and gamma_inf", Ind. Eng. Chem. Res. 26, 1372
(1987). Also Gmehling, Li, Schiller (1993).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from .unifac import UNIFAC


class UNIFAC_Dortmund(UNIFAC):
    """Modified UNIFAC (Dortmund) activity coefficient model.

    Parameters
    ----------
    subgroups_per_component : list of dict
        Same format as UNIFAC: per component a dict {group_name: count}.
    database : module-like, optional
        Object with SUBGROUPS, A_MAIN dicts (required) plus optional
        B_MAIN, C_MAIN dicts for the T-dependent residual.
        Defaults to original UNIFAC database with b=c=0.

    Notes
    -----
    The combinatorial modification (3/4 power) is always applied
    regardless of whether the database has B_MAIN/C_MAIN. This changes
    gamma values for asymmetric mixtures (alkanes + alcohols, etc.).
    """

    def __init__(self, subgroups_per_component, database=None):
        super().__init__(subgroups_per_component, database)
        # Modified volume-fraction parameter: r_i^(3/4)
        self._r_34 = self._r ** 0.75
        # Build B and C lookups (zero if not in database)
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
                        pass   # default 0
                if C_db is not None:
                    try:
                        c_lookup[i_sg, j_sg] = C_db[int(mi)][int(mj)]
                    except KeyError:
                        pass
        self._b_lookup = b_lookup
        self._c_lookup = c_lookup

    def _Psi(self, T: float) -> np.ndarray:
        """Dortmund T-dependent: Psi_mn = exp(-(a + bT + cT^2) / T)."""
        T = float(T)
        return np.exp(-(self._a_lookup
                          + self._b_lookup * T
                          + self._c_lookup * T * T) / T)

    def _dPsi_dT(self, T: float) -> np.ndarray:
        """For Dortmund: ln Psi = -a/T - b - cT
        d(ln Psi)/dT = a/T^2 - c
        dPsi/dT = Psi (a/T^2 - c)."""
        T = float(T)
        Psi = self._Psi(T)
        return Psi * (self._a_lookup / (T * T) - self._c_lookup)

    def _ln_combinatorial(self, x: np.ndarray) -> np.ndarray:
        """Modified Dortmund combinatorial:
        ln gamma_i^C = 1 - V'_i + ln V'_i
                       - 5 q_i [1 - V_i/F_i + ln(V_i/F_i)]
        """
        sum_xr_34 = float(np.sum(x * self._r_34))
        sum_xr = float(np.sum(x * self._r))
        sum_xq = float(np.sum(x * self._q))
        V_prime = self._r_34 / sum_xr_34   # modified volume frac / x_i
        V = self._r / sum_xr               # standard volume frac / x_i
        F = self._q / sum_xq               # surface frac / x_i
        VF = V / F
        return (1.0 - V_prime + np.log(V_prime)
                - (self.Z_COORD / 2.0) * self._q * (1.0 - VF + np.log(VF)))
