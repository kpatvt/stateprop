"""
GERG-2008-style binary departure functions for mixture residual Helmholtz.

The multi-fluid mixture residual Helmholtz is:

    alpha_r(delta, tau, x) = sum_i x_i * alpha_r_oi(delta, tau)
                              + Delta_alpha_r(delta, tau, x)

where the departure term is:

    Delta_alpha_r(delta, tau, x) = sum_{i<j} x_i * x_j * F_ij * alpha_r_ij(delta, tau)

with
    - F_ij: scalar weighting for each binary pair (tabulated)
    - alpha_r_ij(delta, tau): a pure-fluid-like Helmholtz sum with polynomial
      and "generalized exponential" terms.

The generalized exponential term in GERG-2008 has the form:

    n * delta^d * tau^t * exp(-eta * (delta - epsilon)^2 - beta * (delta - gamma))

Note the distinctive form: the exponent is a polynomial in delta only (no tau
dependence). This is different from the Gaussian bell-curves used in pure-fluid
EOS, where the exponent is quadratic in both (delta-epsilon) and (tau-gamma).

Reference: Kunz & Wagner, "The GERG-2008 Wide-Range Equation of State for
Natural Gases and Other Mixtures: An Expansion of GERG-2004", J. Chem. Eng.
Data 57, 3032-3091 (2012). Equation numbers below refer to this reference.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class DepartureTerm:
    """A single term of a binary departure function alpha_r_ij.

    Two kinds:
      - Polynomial: n * delta^d * tau^t
      - Generalized exponential: n * delta^d * tau^t
                                 * exp(-eta * (delta - epsilon)^2
                                       - beta * (delta - gamma))

    For polynomial terms, eta = epsilon = beta = gamma = 0.
    For exp terms, they take their GERG-2008 tabulated values.
    """
    n: float
    d: float
    t: float
    # For generalized exponential term. Zero for polynomial.
    eta: float = 0.0
    epsilon: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    kind: str = "polynomial"   # or "exponential"


@dataclass
class DepartureFunction:
    """One binary departure function alpha_r_ij(delta, tau).

    This is the "alpha_r_ij" piece; the x_i * x_j * F_ij weighting is applied
    externally by the mixture when building the total departure contribution.
    """
    terms: List[DepartureTerm] = field(default_factory=list)
    # Metadata (optional)
    pair: Tuple[str, str] = ("", "")

    def evaluate(self, delta, tau):
        """Return (alpha_r_ij, and its five derivatives) at (delta, tau).

        Returns a 6-tuple (A, A_d, A_t, A_dd, A_tt, A_dt)
        where A = alpha_r_ij and A_* are partial derivatives w.r.t. delta and tau.
        """
        A = 0.0
        A_d = 0.0
        A_t = 0.0
        A_dd = 0.0
        A_tt = 0.0
        A_dt = 0.0

        ln_delta = np.log(delta) if delta > 0 else 0.0
        ln_tau = np.log(tau) if tau > 0 else 0.0

        for term in self.terms:
            n, d, t = term.n, term.d, term.t

            if term.kind == "polynomial":
                # base = n * delta^d * tau^t
                base = n * np.exp(d * ln_delta + t * ln_tau)
                A    += base
                # delta-derivatives of a polynomial term:
                #   d/d(delta) (delta^d) = d * delta^(d-1)
                #   so d(base)/d(delta) = d * base / delta  (for delta > 0)
                A_d  += d * base / delta
                A_t  += t * base / tau
                A_dd += d * (d - 1.0) * base / (delta * delta)
                A_tt += t * (t - 1.0) * base / (tau * tau)
                A_dt += d * t * base / (delta * tau)

            elif term.kind == "exponential":
                # term = n * delta^d * tau^t * exp(-eta * (delta - epsilon)^2
                #                                  - beta * (delta - gamma))
                eta = term.eta
                eps = term.epsilon
                beta_ = term.beta
                gam = term.gamma

                # Polynomial factor and its derivatives
                poly = n * np.exp(d * ln_delta + t * ln_tau)
                # exponent and its delta-derivatives
                g = -eta * (delta - eps) ** 2 - beta_ * (delta - gam)
                g_d = -2.0 * eta * (delta - eps) - beta_
                g_dd = -2.0 * eta
                # exp factor
                E = np.exp(g)

                # term value
                val = poly * E
                A += val

                # First derivatives
                # d(poly)/d(delta) = d * poly / delta
                # d(term)/d(delta) = (d/delta + g_d) * val
                val_d = (d / delta + g_d) * val
                val_t = (t / tau) * val          # no tau dependence in E
                A_d += val_d
                A_t += val_t

                # Second derivatives
                # d^2(term)/d(delta)^2:
                #   Let f = d/delta + g_d. Then val = poly * E gives
                #   val_d = f * val, so val_dd = (df/d(delta)) * val + f * val_d
                #   df/d(delta) = -d/delta^2 + g_dd
                #   val_dd = (-d/delta^2 + g_dd) * val + f * val_d
                f = d / delta + g_d
                val_dd = (-d / (delta * delta) + g_dd) * val + f * val_d
                A_dd += val_dd

                # d^2(term)/d(tau)^2:
                #   no coupling to E, so val_tt = t*(t-1)/tau^2 * val
                val_tt = t * (t - 1.0) / (tau * tau) * val
                A_tt += val_tt

                # d^2(term)/d(delta)d(tau):
                #   d/d(tau) of val_d = d/d(tau) [(d/delta + g_d) * val]
                #                     = (d/delta + g_d) * val_t
                #                     = f * (t/tau) * val
                val_dt = f * (t / tau) * val
                A_dt += val_dt

        return A, A_d, A_t, A_dd, A_tt, A_dt

    @classmethod
    def from_dict(cls, d, pair=("", "")):
        """Build a DepartureFunction from a dict (typically JSON-loaded)."""
        terms = []
        for item in d.get("polynomial", []):
            terms.append(DepartureTerm(
                n=float(item["n"]),
                d=float(item["d"]),
                t=float(item["t"]),
                kind="polynomial",
            ))
        for item in d.get("exponential", []):
            terms.append(DepartureTerm(
                n=float(item["n"]),
                d=float(item["d"]),
                t=float(item["t"]),
                eta=float(item.get("eta", 0.0)),
                epsilon=float(item.get("epsilon", 0.0)),
                beta=float(item.get("beta", 0.0)),
                gamma=float(item.get("gamma", 0.0)),
                kind="exponential",
            ))
        return cls(terms=terms, pair=pair)


def evaluate_total_departure(x, delta, tau, departures):
    """Evaluate the full Delta_alpha_r(delta, tau, x) and its 6 derivatives.

    Parameters
    ----------
    x : array (N,)              mole fractions
    delta, tau : float          reduced coordinates
    departures : dict[(i,j), (F_ij, DepartureFunction)]
                                keyed by (i, j) with i < j

    Returns
    -------
    Delta    : float
    Delta_d  : float           d Delta / d delta
    Delta_t  : float
    Delta_dd : float
    Delta_tt : float
    Delta_dt : float
    dDelta_dxi : ndarray (N,)  d Delta / d x_i (explicit, at fixed delta, tau,
                                x_{k != i}; used by ln_phi)
    """
    N = len(x)
    Delta = 0.0
    Delta_d = 0.0
    Delta_t = 0.0
    Delta_dd = 0.0
    Delta_tt = 0.0
    Delta_dt = 0.0
    dD_dx = np.zeros(N)

    for (i, j), (F_ij, alpha_ij) in departures.items():
        A, A_d, A_t, A_dd, A_tt, A_dt = alpha_ij.evaluate(delta, tau)
        xi_xj_F = x[i] * x[j] * F_ij

        Delta    += xi_xj_F * A
        Delta_d  += xi_xj_F * A_d
        Delta_t  += xi_xj_F * A_t
        Delta_dd += xi_xj_F * A_dd
        Delta_tt += xi_xj_F * A_tt
        Delta_dt += xi_xj_F * A_dt

        # d/d(x_k) (x_i * x_j * F_ij * alpha_ij) at fixed delta, tau:
        #   k = i:  x_j * F_ij * A
        #   k = j:  x_i * F_ij * A
        #   k other: 0
        dD_dx[i] += x[j] * F_ij * A
        dD_dx[j] += x[i] * F_ij * A

    return Delta, Delta_d, Delta_t, Delta_dd, Delta_tt, Delta_dt, dD_dx
