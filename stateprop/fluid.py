"""
Fluid: container class for one Helmholtz equation of state.

The JSON format mirrors the presentation used in reference papers like
Span & Wagner (1996) for CO2, Setzmann & Wagner (1991) for methane, and the
IAPWS formulation for water. Each fluid file contains:

    {
      "name": "CarbonDioxide",
      "molar_mass": 0.0440098,        # kg/mol (optional, needed for speed of sound)
      "gas_constant": 8.31451,        # J/(mol K) -- use the R that the paper used
                                        # (for older EOS this is NOT CODATA 8.314462618)
      "critical": {"T": 304.1282, "rho": 10624.9063, "p": 7377300.0},
      "triple":   {"T": 216.592, "p": 517950.0},
      "limits":   {"Tmin": 216.592, "Tmax": 1100.0, "pmax": 800e6, "rhomax": 37200.0},

      "ideal": [
          {"type": "a1",         "a": 8.37304456, "b": -3.70454304},
          {"type": "log_tau",    "a": 2.5},
          {"type": "PE",         "a": 1.99427042, "b": 3.15163},
          ...
      ],

      "residual_polynomial": [
          {"n": 0.38856823203161, "d": 1, "t": 0.00},
          {"n": 2.9385475942395,  "d": 1, "t": 0.75},
          ...
      ],

      "residual_exponential": [
          {"n": -0.62497968.........., "d": 1, "t": 1.50, "c": 1},
          ...
      ],

      "residual_gaussian": [
          {"n": ...., "d": 2, "t": 1.0, "eta": 25.0, "epsilon": 1.0,
           "beta": 325.0, "gamma": 1.16},
          ...
      ],

      "saturation": {             # optional ancillary equations (not required
          "Tc": 304.1282,           # for property evaluation, only for
          "pc": 7377300.0,          # saturation solver initial guesses)
          ...
      }
    }

The class exposes:

    fluid.name, fluid.R, fluid.T_c, fluid.rho_c, fluid.p_c, fluid.molar_mass
    fluid.pack()  ->  tuple of Numba-friendly arrays to pass into the kernels

"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Mapping from JSON ideal-term "type" strings to the integer codes expected
# by the Numba kernel in core.py. The kernel supports codes 0..7.
# ---------------------------------------------------------------------------
_IDEAL_CODE = {
    "a1":        0,   # constant (added to alpha_0)
    "a2_tau":    1,   # a * tau  (the second lead coefficient)
    "log_tau":   2,
    "log_delta": 3,
    "power_tau": 4,
    "PE":        5,   # Planck-Einstein
    "PE_cosh":   6,
    "PE_sinh":   7,
}


def _ensure_arr(lst, keys, dtype=np.float64):
    """Extract fields `keys` from list of dicts `lst` into stacked arrays.

    Returns a tuple of 1D numpy arrays (one per key) of shape (len(lst),).
    If `lst` is empty, returns empty arrays of the correct shape.
    """
    if not lst:
        return tuple(np.zeros(0, dtype=dtype) for _ in keys)
    arrs = []
    for k in keys:
        arrs.append(np.array([entry[k] for entry in lst], dtype=dtype))
    return tuple(arrs)


@dataclass
class Fluid:
    """Container for a single-component Helmholtz EOS.

    Build with ``Fluid.from_dict(d)`` or ``Fluid.from_json(path)``.

    Attributes
    ----------
    name : str
    R : float                 gas constant used by this EOS [J/(mol K)]
    T_c, rho_c, p_c : float   critical constants
    T_min, T_max : float      validity range
    molar_mass : float or None    [kg/mol] (needed only for speed of sound)
    """
    name: str
    R: float
    T_c: float
    rho_c: float
    p_c: float
    T_min: float = 0.0
    T_max: float = np.inf
    p_max: float = np.inf
    rho_max: float = np.inf
    T_triple: float = 0.0
    p_triple: float = 0.0
    molar_mass: Optional[float] = None

    # Packed arrays (created at load time)
    _pn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _pd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _pt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _en: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _ed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _et: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _ec: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _gn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _gd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _gt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _geta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _geps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _gbeta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _ggam: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Non-analytic (IAPWS-95 / Span-Wagner near-critical) term arrays
    _na: np.ndarray = field(default_factory=lambda: np.zeros(0))      # n
    _nb: np.ndarray = field(default_factory=lambda: np.zeros(0))      # b
    _nB: np.ndarray = field(default_factory=lambda: np.zeros(0))      # B
    _nC: np.ndarray = field(default_factory=lambda: np.zeros(0))      # C
    _nD: np.ndarray = field(default_factory=lambda: np.zeros(0))      # D
    _nA: np.ndarray = field(default_factory=lambda: np.zeros(0))      # A
    _nbeta: np.ndarray = field(default_factory=lambda: np.zeros(0))   # beta
    _naa: np.ndarray = field(default_factory=lambda: np.zeros(0))     # a

    _ideal_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    _ideal_a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _ideal_b: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Keep the raw dict around for debugging / round-tripping
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fluid":
        crit = d["critical"]
        limits = d.get("limits", {})
        triple = d.get("triple", {})

        # Residual polynomial terms
        poly = d.get("residual_polynomial", [])
        pn, pd, pt = _ensure_arr(poly, ["n", "d", "t"])

        # Residual exponential terms
        expo = d.get("residual_exponential", [])
        en, ed, et, ec = _ensure_arr(expo, ["n", "d", "t", "c"])

        # Residual Gaussian terms
        gaus = d.get("residual_gaussian", [])
        gn, gd, gt, geta, geps, gbeta, ggam = _ensure_arr(
            gaus, ["n", "d", "t", "eta", "epsilon", "beta", "gamma"]
        )

        # Non-analytic residual terms (IAPWS-95 / Span-Wagner near-critical)
        nonan = d.get("residual_nonanalytic", [])
        na, naa, nb, nB, nC, nD, nA, nbeta = _ensure_arr(
            nonan, ["n", "a", "b", "B", "C", "D", "A", "beta"]
        )

        # Ideal terms
        ideal = d.get("ideal", [])
        codes = []
        a_arr = []
        b_arr = []
        for entry in ideal:
            tname = entry["type"]
            if tname == "a1":
                # 'a1' in some papers is really two numbers (a + b*tau). Accept both.
                codes.append(_IDEAL_CODE["a1"])
                a_arr.append(float(entry.get("a", 0.0)))
                b_arr.append(0.0)
                if "b" in entry:
                    # Encode the b*tau half as a separate term
                    codes.append(_IDEAL_CODE["a2_tau"])
                    a_arr.append(float(entry["b"]))
                    b_arr.append(0.0)
            else:
                if tname not in _IDEAL_CODE:
                    raise ValueError(
                        f"Unknown ideal term type '{tname}' in fluid '{d.get('name','?')}'"
                    )
                codes.append(_IDEAL_CODE[tname])
                a_arr.append(float(entry.get("a", 0.0)))
                b_arr.append(float(entry.get("b", 0.0)))

        return cls(
            name=d.get("name", "unknown"),
            R=float(d.get("gas_constant", 8.314462618)),
            T_c=float(crit["T"]),
            rho_c=float(crit["rho"]),
            p_c=float(crit.get("p", np.nan)),
            T_min=float(limits.get("Tmin", 0.0)),
            T_max=float(limits.get("Tmax", np.inf)),
            p_max=float(limits.get("pmax", np.inf)),
            rho_max=float(limits.get("rhomax", np.inf)),
            T_triple=float(triple.get("T", 0.0)),
            p_triple=float(triple.get("p", 0.0)),
            molar_mass=(float(d["molar_mass"]) if "molar_mass" in d else None),
            _pn=pn, _pd=pd, _pt=pt,
            _en=en, _ed=ed, _et=et, _ec=ec,
            _gn=gn, _gd=gd, _gt=gt,
            _geta=geta, _geps=geps, _gbeta=gbeta, _ggam=ggam,
            _na=na, _naa=naa, _nb=nb, _nB=nB, _nC=nC, _nD=nD, _nA=nA, _nbeta=nbeta,
            _ideal_codes=np.array(codes, dtype=np.int64),
            _ideal_a=np.array(a_arr, dtype=np.float64),
            _ideal_b=np.array(b_arr, dtype=np.float64),
            _raw=d,
        )

    @classmethod
    def from_json(cls, path: str) -> "Fluid":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    # ---- utilities --------------------------------------------------------
    def pack(self) -> Tuple:
        """Return all JIT-compatible arrays in the canonical order expected
        by the kernels in core.py / properties.py.

        Order:
            (R, rho_c, T_c,
             pn, pd, pt,
             en, ed, et, ec,
             gn, gd, gt, g_eta, g_eps, g_beta, g_gamma,
             n_a, n_b, n_B, n_C, n_D, n_A, n_beta,     # non-analytic
             ideal_codes, ideal_a, ideal_b)
        """
        return (
            self.R, self.rho_c, self.T_c,
            self._pn, self._pd, self._pt,
            self._en, self._ed, self._et, self._ec,
            self._gn, self._gd, self._gt,
            self._geta, self._geps, self._gbeta, self._ggam,
            self._na, self._naa, self._nb, self._nB, self._nC, self._nD,
            self._nA, self._nbeta,
            self._ideal_codes, self._ideal_a, self._ideal_b,
        )

    def reduce(self, rho: float, T: float) -> Tuple[float, float]:
        """Return (delta, tau) reduced coordinates."""
        return rho / self.rho_c, self.T_c / T

    def __repr__(self) -> str:
        return (f"Fluid(name={self.name!r}, T_c={self.T_c:.3f}, "
                f"rho_c={self.rho_c:.3f}, p_c={self.p_c:.3e}, "
                f"n_poly={self._pn.size}, n_exp={self._en.size}, "
                f"n_gauss={self._gn.size}, n_nonan={self._na.size})")


def load_fluid(path_or_name: str) -> Fluid:
    """Load a fluid from an explicit path, or look it up in the packaged
    ``fluids/`` directory by name (case-insensitive, with or without '.json')."""
    # Direct path?
    if os.path.isfile(path_or_name):
        return Fluid.from_json(path_or_name)

    # Try packaged location
    here = os.path.dirname(os.path.abspath(__file__))
    fluids_dir = os.path.normpath(os.path.join(here, "..", "fluids"))

    # Exact-name search (with and without .json)
    for ext in ("", ".json"):
        candidate = os.path.join(fluids_dir, path_or_name + ext)
        if os.path.isfile(candidate):
            return Fluid.from_json(candidate)

    # Case-insensitive search
    if os.path.isdir(fluids_dir):
        want = path_or_name.lower().replace(".json", "")
        for fn in os.listdir(fluids_dir):
            if fn.lower().replace(".json", "") == want:
                return Fluid.from_json(os.path.join(fluids_dir, fn))

    raise FileNotFoundError(
        f"Could not find a fluid '{path_or_name}' either as a file path "
        f"or in the packaged fluids directory ({fluids_dir})"
    )
