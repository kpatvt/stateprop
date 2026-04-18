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

## References

- R. Span and W. Wagner, *J. Phys. Chem. Ref. Data* **25**, 1509 (1996).
- R. Span, E. W. Lemmon, R. T. Jacobsen, W. Wagner, A. Yokozeki,
  *J. Phys. Chem. Ref. Data* **29**, 1361 (2000).
- R. Span, *Multiparameter Equations of State* (Springer, 2000).
- O. Kunz and W. Wagner, *J. Chem. Eng. Data* **57**, 3032 (2012).

## License

MIT.
