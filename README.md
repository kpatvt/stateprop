# stateprop

Numba-accelerated thermodynamic property evaluation for pure fluids and mixtures via Helmholtz, cubic, SAFT, and activity-coefficient frameworks. Built around the multiparameter Helmholtz EOS (Span–Wagner, IAPWS-95, GERG-2008), with cubic-EOS, PC-SAFT, and γ-φ machinery layered on top. Fluids are defined in JSON files mirroring the published reference papers; adding a new fluid is a data-entry task.

## Installation

```bash
pip install stateprop                 # pure-Python
pip install "stateprop[fast]"         # adds Numba (~50-100x speedup)
pip install "stateprop[dev]"          # plus test tooling and chemicals
```

The package bundles ~150 fluid definitions (3 reference Helmholtz EOS, 125 CoolProp-derived fluids, 21 GERG-2008 simplified fluids) and binary parameter sets. No external data files are needed at runtime.

## Quick start

```python
import stateprop as sp

# Pure-fluid Helmholtz EOS
co2 = sp.load_fluid("carbondioxide")
print(sp.pressure(12000.0, 320.0, co2))           # Pa
print(sp.cp(12000.0, 320.0, co2))                 # J/(mol K)
rho_L, rho_V, p_sat = sp.saturation_pT(270.0, co2)

# Pure-fluid flash
water = sp.load_fluid("water")
r = sp.flash_ph(1e6, 2500e3 * water.molar_mass, water)
print(r.phase, r.T, r.quality if r.phase == "two_phase" else r.s)

# Multi-component mixture (GERG-2008)
from stateprop.mixture.mixture import load_mixture
from stateprop.mixture.flash import flash_pt as gerg_flash
mix = load_mixture(
    ["gerg2008/methane", "gerg2008/ethane", "gerg2008/propane"],
    composition=[0.85, 0.10, 0.05], binary_set="gerg2008",
)
r = gerg_flash(50e5, 250.0, [0.85, 0.10, 0.05], mix)

# Cubic EOS mixture (PR/SRK/PR78/...)
from stateprop.cubic import PR, CubicMixture, flash_pt
methane = PR(T_c=190.56, p_c=4.599e6, acentric_factor=0.011)
ethane  = PR(T_c=305.32, p_c=4.872e6, acentric_factor=0.099)
mx = CubicMixture([methane, ethane], composition=[0.7, 0.3])
r = flash_pt(20e5, 220.0, [0.7, 0.3], mx)

# Activity-coefficient flash with auto-detected phase count
from stateprop.activity import GammaPhiEOSThreePhaseFlash, AntoinePsat
from stateprop.activity.compounds import make_unifac
uf = make_unifac(["ethanol", "water"])
psat = [AntoinePsat(A=5.37229, B=1670.409, C=-40.191),
        AntoinePsat(A=4.6543, B=1435.264, C=-64.848)]
flash = GammaPhiEOSThreePhaseFlash(activity_model=uf, psat_funcs=psat,
                                     vapor_eos=mx)
r = flash.auto_isothermal_full_tpd(T=350, p=101325, z=[0.5, 0.5])
print(r.phase_type)   # '1L', '1V', '2VL', '2LL', or '3VLL'
```

See `tests/` for further usage patterns; each test runner doubles as an executable demonstration of the corresponding subsystem.

## Capabilities

| Capability | Helmholtz (pure) | Helmholtz mixture (GERG) | Cubic | SAFT | γ-φ / activity |
|---|---|---|---|---|---|
| Pressure, density, residual derivatives | ✓ | ✓ | ✓ | ✓ | — |
| Caloric (h, s, u, cp, cv, w, μ_JT) | ✓ | ✓ | ✓ | ✓ | partial |
| Fugacity coefficient `ln φ` | ✓ | ✓ | ✓ | ✓ | n/a |
| Activity coefficient `γ` | n/a | n/a | n/a | n/a | NRTL/UNIQUAC/UNIFAC + variants |
| Saturation (pure) | ✓ | n/a | ✓ | ✓ | n/a |
| Density-from-pressure | ✓ | ✓ | ✓ | ✓ | n/a |
| Bubble / dew points (P or T) | n/a | ✓ | ✓ | ✓ | ✓ |
| PT flash (2-phase) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3-phase flash (VLLE) | n/a | ✗ | ✓ | ✗ | ✓ |
| LLE flash (2-phase liquid) | n/a | n/a | n/a | n/a | ✓ |
| PH / PS / TH / TS flash | ✓ | ✓ | ✓ | partial | ✗ |
| TV / UV flash | ✓ | ✓ | ✓ | ✗ | ✗ |
| **PV flash** (v0.9.56) | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Pα / Tα flash** (specified vapor frac, v0.9.56) | ✗ | ✗ | ✓ | ✗ | ✗ |
| Phase envelope tracing | n/a | ✓ | ✓ | ✗ | ✗ |
| Mixture critical points | n/a | ✗ | ✓ | ✗ | n/a |
| Michelsen stability TPD (4 quadrants) | ✗ | ✗ | partial | ✗ | ✓ |
| Auto phase-count flash | n/a | ✗ | n/a | ✗ | ✓ |
| Parameter regression (NRTL/UNIQUAC) | n/a | n/a | n/a | n/a | ✓ |
| Compound database (~50 molecules) | n/a | n/a | via `chemdb` | n/a | ✓ |
| LLE-fitted UNIFAC + extension API | n/a | n/a | n/a | n/a | Magnussen 1981 starter set |
| Volume translation (Péneloux) | n/a | n/a | ✓ | n/a | n/a |
| α-function variants (MC / Twu / PRSV) | n/a | n/a | ✓ | n/a | n/a |
| Transport (viscosity, k, σ) | ✓ Chung + Brock-Bird | ✓ | ✓ | ✓ | n/a |
| Compound lookup from `chemicals` databank | n/a | partial | ✓ | partial | ✓ |

## Tests

13 test runners, ~1240 tests total:

```bash
python tests/run_cubic_tests.py                 # 268 tests
python tests/run_activity_tests.py              # 289 tests
python tests/run_mixture_tests.py               # 248 tests
python tests/run_saft_tests.py                  # 147 tests
python tests/run_gerg_tests.py                  #  49 tests
python tests/run_gerg_validation.py             #  52 tests
python tests/run_gerg_caloric_validation.py     #  74 tests
python tests/run_uv_flash_tests.py              #  45 tests
python tests/run_chemicals_interface_tests.py   #  61 tests
python tests/run_chemdb_tests.py                #  10 tests (8 require chemicals)
python tests/run_converter_tests.py             #  26 tests
python tests/run_coolprop_fluids_tests.py       #  29 tests
python tests/run_transport_tests.py             #  39 tests
```

## Adding a fluid

Drop a JSON file modeled on the published reference paper into `stateprop/fluids/` (or load it from any path with `load_fluid("/path/to/fluid.json")`). The schema mirrors the reference EOS structure: ideal-gas Cp polynomial / Planck-Einstein contributions, residual polynomial / exponential / Gaussian terms, critical and reducing parameters, and saturation ancillaries. Existing files are usable as templates; the kernel handles 9 distinct term types covering virtually all Helmholtz-form EOS in the literature. CoolProp JSON files can be bulk-converted via `stateprop.converter.coolprop_to_stateprop`.

## Version history

### Helmholtz EOS foundation (v0.1 – v0.3)
Core single-fluid Helmholtz machinery: 9 residual + ideal-gas term types, Numba kernel for `α(δ,τ)` and its 5 derivatives in a single pass, density Newton, saturation by analytic-Jacobian Newton. PT/PH/PS/TH/TS/TV/UV flash for pure fluids. JSON-driven fluid library.

### Mixtures (v0.3 – v0.5)
Multicomponent Helmholtz mixtures with GERG-2008-style departure functions and reducing-rule binary parameters. Mixture flash machinery (PT, PH, PS, TV, UV) with shared SS+Broyden hybrid solver. Phase envelope tracing from triple to critical point.

### Cubic EOS (v0.4 – v0.5)
PR / PR-1978 / SRK / RK / VDW with van der Waals one-fluid mixing rules. Bubble/dew Newton solvers, two-phase PT flash with Michelsen stability test, mixture critical points (Heidemann-Khalil), phase envelope tracer. State-function flashes: PH, PS, TH, TS, TV, UV.

### CoolProp interoperability (v0.6 – v0.9.3)
- v0.6.1–v0.6.3: bug fixes in IAPWS-95 and ideal-gas conversions.
- v0.7.0: UV flash for mixtures (transient-simulation natural variables).
- v0.8.0: `chemicals` databank integration — build cubic components by name.
- v0.8.1: bulk-import all CoolProp `dev/fluids/` JSON files via converter.
- v0.9.0–v0.9.3: extended kernel term set covering 125 CoolProp fluids; caloric bugfixes for `CP0PolyT` and `PlanckEinsteinGeneralized`.

### Mixture flash maturity (v0.9.4 – v0.9.6)
v0.9.4 made arbitrary-component flash (n ≥ 4) numerically robust. v0.9.5 added warm-start density caching and Broyden-on-K acceleration (~1.5–2× faster). v0.9.6 micro-optimizations.

### Cubic EOS extensions (v0.9.7 – v0.9.20)
PR-1978 m(ω) correlation, Mathias-Copeman / Twu / PRSV α functions, Péneloux volume translation, Newton bubble/dew solvers, analytic envelope-tracer Jacobian, three-phase (VLLE) PT flash for cubic mixtures.

### PC-SAFT (v0.9.21 – v0.9.37)
- v0.9.21–v0.9.27: PC-SAFT pure + mixture, with hard-chain + dispersion + association (2B, 3B, 4C schemes).
- v0.9.28: dedicated 4C water parameter fit.
- v0.9.29–v0.9.30: analytic composition Jacobian via derivative identity.
- v0.9.31: Newton bubble/dew for SAFT mixtures.
- v0.9.32: transport properties (Chung viscosity, Brock-Bird thermal conductivity).
- v0.9.33: Macleod-Sugden parachor surface tension.
- v0.9.34–v0.9.37: fully analytic A_rho / A_rhorho / A_rhoi; optimal corr-FD step sizes.

### Activity coefficient framework (v0.9.40 – v0.9.47)
- v0.9.40: γ-φ flash for low-pressure VLE with NRTL/UNIQUAC/UNIFAC.
- v0.9.42: excess thermodynamic properties (`hE`, `sE`, `cpE`).
- v0.9.43: γ-φ-EOS flash for 1–30 bar VLE (vapor non-ideality from cubic/SAFT/GERG).
- v0.9.44: analytical T-derivatives of γ replacing v0.9.42 numerical paths.
- v0.9.45: batch grid generation with warm-start (~1.5–1.8× speedup).
- v0.9.46: three-phase γ-φ-EOS flash (VLLE) with 2D Newton + line search.
- v0.9.47: LLE flash + parameter regression (`regress_lle`, `regress_vle`, NRTL/UNIQUAC factories).

### Michelsen TPD framework (v0.9.48 – v0.9.53)
Complete 4-quadrant tangent-plane-distance machinery for the γ-φ framework:
- v0.9.48: liquid-against-liquid stability test (γ-based).
- v0.9.49: auto 3-phase flash via stability + bubble-p heuristic; 5 phase types (1L/1V/2VL/2LL/3VLL).
- v0.9.50: pre-built compound database — ~50 molecules, UNIQUAC r/q computed from group sums (matches DECHEMA to 4 decimals).
- v0.9.51: vapor-against-vapor stability test (φ-based).
- v0.9.52: cross-phase stability tests (L→V, V→L) using γ + φ jointly.
- v0.9.53: auto-flash with full 4-test TPD framework — replaces bubble-p heuristic with rigorous pattern-matched phase-count detection.

### LLE-UNIFAC and validation infrastructure (v0.9.54 – v0.9.55)
- v0.9.54: `UNIFAC_LLE` class with Magnussen-1981 LLE-fitted parameter overrides (4 critical aqueous-organic main-group pairs bundled; user-extensible via `extra_overrides`).
- v0.9.55: coverage reporting (`lle_coverage`), benchmark validation harness against published mutual solubilities (4 canonical aqueous-organic systems), JSON import/export for custom parameter sets.

### Cubic flash completion (v0.9.56)
PV flash (given P and v, find T), Pα flash (specified vapor fraction at fixed p, find T), Tα flash (specified vapor fraction at fixed T, find p). Completes the cubic-EOS state-function flash family (PT, PH, PS, TH, TS, TV, UV, PV, Pα, Tα, bubble, dew).

## References

- R. Span and W. Wagner, *J. Phys. Chem. Ref. Data* **25**, 1509 (1996).
- O. Kunz and W. Wagner, *J. Chem. Eng. Data* **57**, 3032 (2012) — GERG-2008.
- M. L. Michelsen, *Fluid Phase Equilibria* **9**, 1, 21 (1982) — stability + flash.
- M. L. Michelsen and J. M. Mollerup, *Thermodynamic Models: Fundamentals and Computational Aspects*, 2nd ed. (Tie-Line, 2007).
- D.-Y. Peng and D. B. Robinson, *Ind. Eng. Chem. Fund.* **15**, 59 (1976); GPA RR-28 (1978).
- G. Soave, *Chem. Eng. Sci.* **27**, 1197 (1972) — SRK α.
- A. Fredenslund, R. L. Jones, J. M. Prausnitz, *AIChE J.* **21**, 1086 (1975) — UNIFAC.
- H. K. Hansen et al., *Ind. Eng. Chem. Res.* **30**, 2352 (1991) — UNIFAC parameters.
- T. Magnussen, P. Rasmussen, A. Fredenslund, *Ind. Eng. Chem. Process Des. Dev.* **20**, 331 (1981) — UNIFAC-LLE.
- J. Gross and G. Sadowski, *Ind. Eng. Chem. Res.* **40**, 1244 (2001) — PC-SAFT.
- R. A. Heidemann and A. M. Khalil, *AIChE J.* **26**, 769 (1980) — cubic critical points.

## License

MIT.
