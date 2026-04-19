# stateprop

Numba-accelerated evaluation of single-component thermodynamic state
properties via multiparameter Helmholtz equations of state (Span–Wagner,
IAPWS-95, GERG-style).

Fluids are defined in JSON files whose layout mirrors the tables of the
reference papers where the coefficients were published. Adding a new fluid is
a data-entry exercise: no Python code changes needed.

## What it computes

From the reduced Helmholtz energy `α(δ, τ) = α⁰(δ, τ) + αʳ(δ, τ)` and its six
partial derivatives, the library returns:

- Pressure `p`, compressibility factor `Z`
- Internal energy `u`, enthalpy `h`, entropy `s`, Gibbs energy `g`
- Isochoric and isobaric heat capacities `cv`, `cp`
- Thermodynamic speed of sound `w`
- Fugacity coefficient `φ`
- Joule–Thomson coefficient `μ_JT`
- Pressure derivatives `(∂p/∂ρ)_T` and `(∂p/∂T)_ρ`
- Vapor–liquid saturation (phase equilibrium) via an analytic-Jacobian Newton
- Density from `(p, T)` via a 1-D safeguarded Newton
- **Flashes:** PT, PH, PS, TV, UV — returning phase, quality, and the full
  thermodynamic state
- **Phase envelopes:** adaptive tracing of the coexistence curve from triple
  to critical point, with dome extraction in any thermodynamic coordinates

All of these follow from standard identities applied to the six derivatives of
`α`, which are computed in a single pass by a Numba-jitted kernel.

## Installation

```bash
pip install -e .                 # pure-Python mode (slow but works everywhere)
pip install -e ".[fast]"         # adds Numba for ~50-100x speedup
pip install -e ".[dev]"          # plus test tooling
```

## Quick start

```python
import stateprop as h

# Load a fluid from the packaged JSON library
co2 = h.load_fluid("carbondioxide")

# Evaluate properties at a state point (rho in mol/m^3, T in K)
rho, T = 12000.0, 320.0
print("p    =", h.pressure(rho, T, co2), "Pa")
print("cp   =", h.cp(rho, T, co2), "J/(mol K)")
print("w    =", h.speed_of_sound(rho, T, co2), "m/s")
print("µ_JT =", h.joule_thomson_coefficient(rho, T, co2), "K/Pa")

# Vapor-liquid equilibrium at a given temperature
rho_L, rho_V, p_sat = h.saturation_pT(270.0, co2)
print(f"p_sat(270 K) = {p_sat*1e-6:.4f} MPa")
print(f"rho_L = {rho_L:.2f}, rho_V = {rho_V:.2f} mol/m^3")

# Density from pressure and temperature
from stateprop.saturation import density_from_pressure
rho = density_from_pressure(10e6, 350.0, co2, phase="vapor")

# Array / vector evaluation
import numpy as np
rhos = np.linspace(100, 15000, 1000)
Ts = np.full_like(rhos, 320.0)
ps = h.pressure(rhos, Ts, co2)     # returns a 1000-element numpy array
```

## Flash algorithms

Given any two independent thermodynamic variables, determine the full state
and phase.  All flashes return a `FlashResult` with the phase label, T, p,
density, caloric properties, and — if two-phase — quality and saturated
liquid/vapor densities.

```python
import stateprop as sp
water = sp.load_fluid("water")

# PT flash -- classify phase, compute properties
r = sp.flash_pt(10e6, 700.0, water)
# r.phase == 'supercritical', r.h, r.s, r.cp, r.w all populated

# PH flash -- the workhorse for heat-exchanger and throttle calculations.
# If (p, h) lies inside the coexistence dome, returns quality automatically.
r = sp.flash_ph(1e6, 2500e3 * water.molar_mass, water)
if r.phase == "two_phase":
    print(f"Quality x = {r.quality:.4f}, T = {r.T:.2f} K")

# PS flash -- for isentropic expansion (turbines, nozzles)
inlet = sp.flash_pt(10e6, 773.15, water)            # steam at 10 MPa, 500 C
exit  = sp.flash_ps(0.01e6, inlet.s, water)         # isentropic to 10 kPa
print(f"Work = {(inlet.h - exit.h)/water.molar_mass*1e-3:.1f} kJ/kg, "
      f"exit quality = {exit.quality:.3f}")

# TS flash -- unambiguous for all subcritical and supercritical T
r = sp.flash_ts(450.0, s_target, water)             # returns phase, p, etc.

# TH flash -- has a physical ambiguity at subcritical T: compressed-liquid
# enthalpy is h_L_sat + p*v*(1-T*alpha_p), which sits above h_L_sat, overlapping
# the two-phase enthalpy band [h_L_sat, h_V_sat]. The default behavior returns
# the two-phase equilibrium state. To force the compressed-liquid branch:
r = sp.flash_th(300.0, h_target, water)                       # -> two-phase (default)
r = sp.flash_th(300.0, h_target, water, phase_hint="liquid")  # -> compressed liquid

# TV and UV flash -- natural variables for dynamic simulation
state = sp.flash_uv(u_target=5000.0, v=2e-4, fluid=water)
```

Supported flashes: `flash_pt`, `flash_ph`, `flash_ps`, `flash_th`, `flash_ts`,
`flash_tv`, `flash_uv`.  Each uses analytic Jacobians where available (cp for
PH, cp/T for PS, cv for UV, `(∂h/∂ρ)_T` and `(∂s/∂ρ)_T = -(1/ρ²)(∂p/∂T)_ρ`
from the Maxwell relation for TH/TS) and falls back to safeguarded damped
Newton iterations.

## Phase envelope tracing

Generate the full vapor-liquid coexistence curve for any fluid with a single
call:

```python
env = h.trace_phase_envelope(water, n_points=120)

# env.T, env.p, env.rho_L, env.rho_V, env.h_L, env.h_V, env.s_L, env.s_V
# are parallel arrays from the triple point to the critical point.

# Mass-based versions for engineering use
mb = env.as_mass_based()
# mb['p_MPa'], mb['rho_L'] (kg/m^3), mb['h_V'] (kJ/kg), mb['h_vap'], etc.

# Extract the dome as a closed curve in any thermodynamic coordinates
x, y = env.dome_coordinates(x_kind="s_kg", y_kind="T")   # T-s dome
x, y = env.dome_coordinates(x_kind="h_kg", y_kind="p_MPa")  # Mollier-style
```

The grid is adaptive in temperature, clustering points toward the critical
region where properties change rapidly. Spurious near-critical solutions are
filtered automatically. See `examples/plot_phase_envelope.py` for a
four-panel plot (p-T, T-s, p-h, T-ρ) of water.

## Included fluids

| Fluid            | Reference                                   | Validity         | Notes |
| ---------------- | ------------------------------------------- | ---------------- | ----- |
| `carbondioxide`  | Span & Wagner, *JPCRD* **25**, 1509 (1996)  | 216–1100 K, ≤ 800 MPa | 39 analytic terms (critical-enhancement terms omitted) |
| `nitrogen`       | Span et al., *JPCRD* **29**, 1361 (2000)    | 63–2000 K, ≤ 2200 MPa | 36 analytic terms |
| `water`          | IAPWS R6-95(2018) / Wagner & Pruß (2002)    | 273–1273 K, ≤ 1000 MPa | Full IAPWS-95: 56 terms incl. 2 non-analytic critical-enhancement terms |

## Adding a fluid

A fluid JSON file has four blocks that mirror the presentation in the
reference papers:

```json
{
  "name":         "CarbonDioxide",
  "molar_mass":   0.0440098,
  "gas_constant": 8.31451,
  "critical":     {"T": 304.1282, "rho": 10624.9063, "p": 7377300.0},
  "triple":       {"T": 216.592,  "p": 517950.0},
  "limits":       {"Tmin": 216.592, "Tmax": 1100.0, "pmax": 800e6},

  "ideal": [
    {"type": "log_delta", "a": 1.0},
    {"type": "a1",        "a":  8.37304456, "b": -3.70454304},
    {"type": "log_tau",   "a":  2.5},
    {"type": "PE",        "a":  1.99427042, "b":  3.15163}
    // ... more Planck-Einstein / power_tau / PE_cosh / PE_sinh terms
  ],

  "residual_polynomial": [
    {"n":  0.38856823203161, "d": 1, "t": 0.00},
    {"n":  2.9385475942395,  "d": 1, "t": 0.75}
    // ... term form:  n * delta^d * tau^t
  ],

  "residual_exponential": [
    {"n":  2.1658961543220, "d": 1, "t": 1.50, "c": 1}
    // ... term form:  n * delta^d * tau^t * exp(-delta^c)
  ],

  "residual_gaussian": [
    {"n": -213.65488688320, "d": 2, "t": 1.00,
     "eta": 25.0, "epsilon": 1.00, "beta": 325.0, "gamma": 1.16}
    // ... term form:  n * delta^d * tau^t *
    //                 exp(-eta*(delta-epsilon)^2 - beta*(tau-gamma)^2)
  ],

  "residual_nonanalytic": [
    // IAPWS-95 / Span-Wagner near-critical term:
    //   phi_i = n * Delta^b * delta * psi
    // with
    //   psi   = exp(-C (delta-1)^2 - D (tau-1)^2)
    //   Delta = theta^2 + B ((delta-1)^2)^a
    //   theta = (1 - tau) + A ((delta-1)^2)^(1/(2 beta))
    //
    // These capture the critical-enhancement behavior that
    // polynomial/exponential terms cannot.
    {"n": -0.14874640856724, "a": 3.5, "b": 0.85,
     "B": 0.2, "C": 28.0, "D": 700.0, "A": 0.32, "beta": 0.3}
  ]
}
```

Supported ideal-term `type` values:

| Type         | Contribution to `α⁰`          |
| ------------ | ----------------------------- |
| `a1`         | `a + b*τ`                     |
| `log_tau`    | `a · ln(τ)`                   |
| `log_delta`  | `a · ln(δ)`                   |
| `power_tau`  | `a · τᵇ`                      |
| `PE`         | `a · ln(1 − exp(−b·τ))`       |
| `PE_cosh`    | `a · ln(cosh(b·τ))`           |
| `PE_sinh`    | `a · ln\|sinh(b·τ)\|`         |

The `PE_cosh`/`PE_sinh` forms appear in GERG-2008 and several recent
multiparameter EOS.

## Performance

On a modern laptop (CO2, 39 residual terms + 8 ideal terms) with Numba:

| kernel               | time/call |
| -------------------- | --------- |
| `α_r + 5 derivs`     | ~1 µs     |
| pressure             | ~1 µs     |
| full property set    | ~2 µs    |

Pure-Python mode (no Numba): ~100 µs/call — still useful for scripting and
prototyping.

Running `python examples/benchmark.py` prints a timing table.

## Tests

```bash
python tests/test_co2.py      # core kernel + properties + saturation for CO2
python tests/test_fluids.py   # nitrogen + synthetic cosh/sinh coverage
python tests/test_water.py    # IAPWS-95 Tables 6, 7, 8 verification
python tests/test_flash.py    # PT/PH/PS/TH/TS/TV/UV flashes + phase envelope
```

All four test files end with `OVERALL: ALL TESTS PASSED` on current master.

## Numerical validation

- `α⁰` and `αʳ` derivatives are verified against finite differences to ~1e-8.
- Thermodynamic identities (e.g. `h = u + p/ρ`) hold to machine precision.
- Ideal-gas limits (Z → 1 at vanishing density) are recovered.
- CO2 supercritical speed of sound at (T=750 K, ρ=6367 mol/m³) = 485.1 m/s,
  matching published values.
- CO2 saturation pressure at T=240 K = 1.282 MPa, matches NIST to within 0.1%.
- Nitrogen saturation matches NIST to within 1–2% across 70–110 K.

## Mixtures (v0.3)

The `stateprop.mixture` submodule extends the pure-component framework to
multicomponent systems using the Kunz–Wagner multi-fluid approximation
(the foundation of GERG-2008).

### What's implemented

**Physics:**
- **Reducing functions:** Kunz–Wagner binary mixing rules with asymmetric
  `β_ij, γ_ij` parameters (stored in `fluids/binaries/`, default 1.0 for
  unspecified pairs).
- **Residual Helmholtz:** Mole-weighted pure-component `αʳ`'s evaluated at
  the mixture-reduced `(δ, τ)`, **plus** a binary departure function
  `Δαʳ(δ, τ, x) = Σ_{i<j} x_i x_j F_ij αʳ_ij(δ, τ)` when provided in the
  binary JSON. Supports both polynomial and generalized-exponential term
  types (`n δ^d τ^t exp[-η(δ-ε)² - β(δ-γ)]`, the GERG-2008 form).
- **Ideal-gas part:** Mole-weighted pure `α⁰` plus entropy of mixing
  `x_i ln(x_i)`.
- **Fugacity coefficients** from the exact composition-derivative identity
  (with the correct departure contribution via Euler's theorem on
  degree-2-homogeneous `Δαʳ`). Validated against finite differences to
  ~1e-8, with and without departure active, across 30+ test points.

**Algorithms:**
- **PT flash** with Michelsen TPD stability test and successive substitution
  on K-factors. Fugacity equality at convergence holds to machine precision.
- **State-function flashes:** T-β, p-β, PH, PS, TH, TS. Outer
  Newton-secant with bracketed bisection fallback.
- **Bubble- and dew-point solvers** (dedicated, not delegated to flash):
  `bubble_point_p(T, z)`, `bubble_point_T(p, z)`, `dew_point_p(T, z)`,
  `dew_point_T(p, z)`. Use Michelsen pressure-correction; the `T`-variant
  uses bracketed Newton in `1/T` with trivial-solution rejection.

### Example: mixture with departure function

```python
import numpy as np
from stateprop.mixture import load_mixture, flash_pt, bubble_point_p

# Without departure (simple multi-fluid, ~1% accuracy)
mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.9, 0.1])

# With departure (loads `fluids/binaries/<binary_set>.json`)
mx = load_mixture(['carbondioxide', 'nitrogen'],
                  composition=[0.9, 0.1],
                  binary_set='test_co2_n2')

# Two-phase flash
r = flash_pt(p=5e6, T=240.0, z=mx.x, mixture=mx)
print(f"phase={r.phase}, β={r.beta:.3f}")
print(f"liquid: x = {r.x}, ρ_L = {r.rho_L:.1f} mol/m³")
print(f"vapor:  y = {r.y}, ρ_V = {r.rho_V:.1f} mol/m³")

# Bubble-point pressure at T=240K
b = bubble_point_p(240.0, mx.x, mx)
print(f"bubble pressure = {b.p/1e6:.2f} MPa, incipient y = {b.y}")
```

### Adding GERG-2008 binary parameters

To use full GERG-2008 accuracy for a pair, create a JSON file in
`fluids/binaries/` with reducing-function parameters and the departure
function's polynomial + exponential term tables:

```json
{
  "pairs": [
    {
      "pair": ["methane", "nitrogen"],
      "beta_T": 0.998721377,
      "gamma_T": 1.013950311,
      "beta_v": 0.998098830,
      "gamma_v": 0.979273013,
      "F": 1.0,
      "departure": {
        "polynomial": [
          {"n": -0.00987659, "d": 1, "t": 0.0},
          {"n": 0.00589894, "d": 2, "t": 1.85}
        ],
        "exponential": [
          {"n": 1.234567e-3, "d": 3, "t": 7.85,
           "eta": 1.0, "epsilon": 0.5, "beta": 1.0, "gamma": 0.5}
        ]
      }
    }
  ]
}
```

Load via `load_mixture([...], binary_set='<filename-without-.json>')`.

### What's not (yet)

- **Pre-populated GERG-2008 coefficients** for the 21 natural-gas components
  × 210 binary pairs. The framework loads whatever JSON files are provided;
  the shipped `test_co2_n2.json` has *synthetic* coefficients for testing
  only. Production use requires loading published GERG-2008 tables.
- **Population of components:** only CO2, N2, and water (reusing the pure
  JSONs) are wired up. Methane, ethane, propane, etc. need JSON files added
  to `fluids/components/` (same format as pure fluids).

### Known limitations on the bubble/dew solvers

Some mixtures have **no physical bubble or dew point** at certain (T, p).
For example, CO2+N2 at 50/50 composition and T=220K is two-phase at all
pressures — β never reaches 0 or 1. The solvers correctly raise
`RuntimeError` in these cases with a clear message, rather than returning a
trivial-solution artifact at unphysical T or p.

### Tests

Run the full test battery:
```
python tests/run_mixture_tests.py      # mixture suite (107 checks)
python -m tests.test_co2               # pure CO2
python -m tests.test_water             # IAPWS-95 water
python -m tests.test_fluids            # cross-fluid consistency
python -m tests.test_flash             # pure-component flash
```

All 107 mixture tests + all pre-existing pure-component tests pass. The
mixture suite includes FD validation of all 5 derivatives of the departure
function, FD validation of the full `ln_phi` with departure active, and a
full-circuit PT flash with fugacity equality.

## Cubic EOS (v0.4)

The `stateprop.cubic` submodule provides generalized two-parameter cubic
equations of state, suitable for mixtures of ~any fluid with known critical
parameters and acentric factor. This is complementary to the multiparameter
Helmholtz path: cubics give ~1% accuracy but work for any fluid; Helmholtz
EOSes give ~0.01% accuracy but need tabulated coefficients per fluid.

### Supported EOS families

| EOS | ε | σ | Ωₐ | Ω_b | Z_c (EOS) | α(T_r, ω) |
|-----|---|---|----|-----|-----------|-----------|
| van der Waals | 0 | 0 | 27/64 | 1/8 | 0.375 | 1 |
| Redlich–Kwong | 0 | 1 | 0.42748 | 0.08664 | 1/3 | 1/√T_r |
| Soave–Redlich–Kwong | 0 | 1 | 0.42748 | 0.08664 | 1/3 | `[1+m(1-√T_r)]²` |
| Peng–Robinson | 1−√2 | 1+√2 | 0.45724 | 0.07780 | 0.307 | `[1+m(1-√T_r)]²` |

Soave-type α: `m = a₀ + a₁ω + a₂ω²` with `(a₀, a₁, a₂) = (0.480, 1.574, -0.176)` for
SRK and `(0.37464, 1.54226, -0.26992)` for PR.

### Example: pure fluid

```python
from stateprop.cubic import PR, SRK

# Create a Peng-Robinson EOS for CO2
co2 = PR(T_c=304.13, p_c=7.377e6, acentric_factor=0.224)

# Residual Helmholtz alpha^r(delta, tau) and all 5 derivatives
A, A_d, A_t, A_dd, A_tt, A_dt = co2.alpha_r_derivs(delta=0.5, tau=1.2)

# Pressure at (rho, T)
p = co2.pressure(rho=5000.0, T=280.0)

# Pure-fluid saturation
p_sat = co2.saturation_p(T=250.0)    # -> ~1.77 MPa (< 1% off NIST)
```

### Example: mixture with van der Waals one-fluid mixing

```python
from stateprop.cubic import CubicMixture, PR, flash_pt, bubble_point_p

# Natural gas-ish mixture (PR). Cp polynomials enable caloric properties.
ch4 = PR(T_c=190.56, p_c=4.599e6, acentric_factor=0.011, name="methane",
         ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
n2  = PR(T_c=126.19, p_c=3.396e6, acentric_factor=0.039, name="nitrogen",
         ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
co2 = PR(T_c=304.13, p_c=7.377e6, acentric_factor=0.224, name="CO2",
         ideal_gas_cp_poly=(22.26, 0.05981, -3.501e-5, 7.469e-9))

mx = CubicMixture(
    [ch4, n2, co2],
    composition=[0.85, 0.10, 0.05],
    k_ij={(0,1): 0.025, (0,2): 0.091, (1,2): -0.017},  # binary interaction params
)

# PT flash -- now returns real h, s (not placeholders)
r = flash_pt(p=3e6, T=170.0, z=mx.x, mixture=mx)
print(f"phase: {r.phase}, beta: {r.beta}")
print(f"h = {r.h:.1f} J/mol, s = {r.s:.3f} J/(mol K)")
print(f"liquid: x = {r.x}, rho_L = {r.rho_L:.1f} mol/m^3")
print(f"vapor:  y = {r.y}, rho_V = {r.rho_V:.1f} mol/m^3")

# Bubble-point pressure
b = bubble_point_p(T=170.0, z=mx.x, mixture=mx)
print(f"bubble_p = {b.p/1e6:.3f} MPa, incipient y = {b.y}")
```

### Caloric properties and state-function flashes

Once `ideal_gas_cp_poly` is provided per component, the mixture's caloric
properties (`h`, `s`, `u`) are returned by every flash automatically. The
state-function flashes invert the usual `PT -> property` relationship:

```python
from stateprop.cubic import flash_pt, flash_ph, flash_ps, flash_th, flash_ts

r_pt = flash_pt(p=2e6, T=150.0, z=mx.x, mixture=mx)

# PH: "what T gives this h at this p?"
r_ph = flash_ph(p=2e6, h_target=r_pt.h, z=mx.x, mixture=mx)
assert abs(r_ph.T - 150.0) < 0.01   # recovers the source T

# PS: "what T gives this s at this p?"
r_ps = flash_ps(p=2e6, s_target=r_pt.s, z=mx.x, mixture=mx)

# TH / TS: "what p gives this h (or s) at this T?"
r_th = flash_th(T=250.0, h_target=some_h, z=mx.x, mixture=mx)
r_ts = flash_ts(T=250.0, s_target=some_s, z=mx.x, mixture=mx)
```

The residual caloric parts are FD-validated to 1e-6 against the temperature
derivative of `alpha_r` at fixed density. The state-function flashes recover
source (T, p) to 1e-4 relative error across single-phase and two-phase
conditions.

**Ideal-gas model options:**
- Pass `ideal_gas_cp_poly=(a0, a1, a2, ...)` for polynomial `Cp(T) = sum a_k T^k` in J/(mol K). This is the intended path for realistic caloric properties.
- Omit and get a default `Cp = 3.5 R` (diatomic-like constant), sufficient for framework-only tests but not quantitatively accurate.
- Reference state is configurable via `T_ref`, `p_ref`, `h_ref`, `s_ref` per component (default: `T_ref=298.15 K`, `p_ref=101325 Pa`, `h_ref=s_ref=0`).

### NIST comparison

PR and SRK pure-component saturation vs NIST for CO2:

| T [K] | NIST [MPa] | PR [MPa] | PR err | SRK [MPa] | SRK err |
|-------|-----------:|---------:|-------:|----------:|--------:|
| 220 | 0.599 | 0.596 | −0.5% | 0.600 | +0.1% |
| 240 | 1.283 | 1.271 | −0.9% | 1.286 | +0.3% |
| 260 | 2.419 | 2.405 | −0.6% | 2.435 | +0.7% |
| 280 | 4.161 | 4.160 | ≈0%   | 4.199 | +0.9% |

Both SRK and PR match NIST to under 1% over this range. PR does slightly
better on CO2 near the critical; SRK's empirical α function compensates well
at moderate subcritical conditions.

### What's implemented

**Pure-fluid physics:**
- PR, SRK, RK, vdW cubic EOS as closed-form residual Helmholtz
- All 5 derivatives of α^r in (δ, τ) — FD-validated to ~1e-9
- Pure-fluid saturation solver (ln φ_L = ln φ_V)
- Pressure consistency `p(direct cubic) == ρRT(1 + δ α^r_δ)` to machine precision

**Mixture physics:**
- van der Waals one-fluid mixing with k_ij binary interaction parameters
  (dict or full matrix)
- Mixture `ln φ_i` derived from first principles via `∂(nα^r)/∂n_i` — FD-
  validated to ~1e-9 on binary and ternary systems
- **Caloric properties (v0.4.1): h, s, u** for the mixture at any (T, ρ, x),
  assembled from the closed-form cubic residual plus per-component ideal-gas
  Cp polynomials. Residual parts FD-validated to ~1e-6

**Algorithms:**
- **PT flash** with Michelsen TPD stability test; fugacity equality at
  convergence holds to 1e-11 on tested cases. Returns h and s populated.
- **Bubble-p, bubble-T, dew-p, dew-T** via Michelsen pressure/temperature
  corrections with trivial-solution detection and bracketed fallback
- **State-function flashes (v0.4.1): PH, PS, TH, TS** as outer Newton-secant
  wrappers around PT flash. Round-trips recover source T or p to 1e-4 or
  better across single-phase and two-phase states

### What's not (yet)

- **Volume translation** (e.g., Peneloux-type). Would improve liquid density
  accuracy at the cost of framework complexity.
- **α-function variants** beyond classical Soave (e.g., Twu-Coon, Mathias-Copeman).
  Adding these requires a modest extension to `CubicEOS.alpha_func`.
- **PR-1978** modified m(ω) correlation for heavier acentric factors. The
  classical (1976) form is used throughout.

### Run the cubic tests

```
python tests/run_cubic_tests.py
```

All 59 cubic tests pass: 4 pure-component α^r FD checks, 2 pressure-
consistency checks, SRK & PR saturation vs NIST for CO2, mixture
pressure-consistency (4 states), binary ln φ FD (9 states), ternary ln φ FD,
PT flash with fugacity equality (binary + ternary), supercritical detection,
stability test, 11 bubble/dew point convergence checks, 3 caloric FD checks,
Cp recovery check, flash_pt h/s population, and 14 state-function flash
round-trips (PH, PS, TH, TS across multiple phase regimes).

## References

- R. Span and W. Wagner, *J. Phys. Chem. Ref. Data* **25**, 1509 (1996).
- R. Span, E. W. Lemmon, R. T. Jacobsen, W. Wagner, A. Yokozeki,
  *J. Phys. Chem. Ref. Data* **29**, 1361 (2000).
- R. Span, *Multiparameter Equations of State* (Springer, 2000).
- O. Kunz and W. Wagner, *J. Chem. Eng. Data* **57**, 3032 (2012).
- M. L. Michelsen, *Fluid Phase Equilibria* **9**, 1 (1982) — stability.
- M. L. Michelsen, *Fluid Phase Equilibria* **9**, 21 (1982) — flash.
- M. L. Michelsen and J. M. Mollerup, *Thermodynamic Models: Fundamentals and
  Computational Aspects*, 2nd ed. (Tie-Line, 2007) — cubic EOS algorithms.
- G. Soave, *Chem. Eng. Sci.* **27**, 1197 (1972) — SRK α function.
- D.-Y. Peng and D. B. Robinson, *Ind. Eng. Chem. Fund.* **15**, 59 (1976).

## License

MIT.
