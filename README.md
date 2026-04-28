# stateprop

Numba-accelerated thermodynamic property evaluation for pure fluids and
mixtures via Helmholtz, cubic, SAFT, and activity-coefficient
frameworks. Built around the multiparameter Helmholtz EOS
(Span-Wagner, IAPWS-95, GERG-2008), with cubic-EOS, PC-SAFT, and γ-φ
machinery layered on top. Fluids are defined in JSON files mirroring
the published reference papers; adding a new fluid is a data-entry
task.

## Installation

```bash
pip install stateprop                 # pure-Python
pip install "stateprop[fast]"         # adds Numba (~50-100x speedup)
pip install "stateprop[dev]"          # plus test tooling and chemicals
```

The package bundles ~150 fluid definitions (3 reference Helmholtz EOS,
125 CoolProp-derived fluids, 21 GERG-2008 simplified fluids) and
binary parameter sets. No external data files are needed at runtime.

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

See `examples/` for a 65-example curriculum walking from pure-fluid
properties up through integrated process flowsheets, and `tests/` for
further usage patterns.

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
| PV flash, Pα / Tα flash | ✗ | ✗ | ✓ | ✗ | ✗ |
| Phase envelope tracing | n/a | ✓ | ✓ | ✗ | ✗ |
| Mixture critical points | n/a | ✗ | ✓ | ✗ | n/a |
| Auto phase-count flash | n/a | ✗ | n/a | ✗ | ✓ |
| Parameter regression (NRTL/UNIQUAC) | n/a | n/a | n/a | n/a | ✓ |
| Volume translation (Péneloux) | n/a | n/a | ✓ | n/a | n/a |
| α-function variants (MC / Twu / PRSV) | n/a | n/a | ✓ | n/a | n/a |
| Transport (viscosity, k, σ) | ✓ Chung + Brock-Bird | ✓ | ✓ | ✓ | n/a |
| Compound lookup from `chemicals` databank | n/a | partial | ✓ | partial | ✓ |

### Domain modules

- **Distillation columns** — Naphtali-Sandholm Newton solver with
  arbitrary feed and side-draw topology, side strippers, partial /
  total condensers, Murphree efficiency, tray hydraulics with
  Souders-Brown flooding, pump-arounds, design-mode specifications.
  Reactive distillation and reactive extraction with energy balance.
- **Refinery characterization** — TBP-curve discretization into
  pseudo-components with Riazi-Daubert / Lee-Kesler / Edmister
  correlation network. End-to-end atmospheric crude tower with side
  strippers producing realistic naphtha / kerosene / diesel / residue
  cuts.
- **Reaction equilibrium** — single and multi-reaction systems, Gibbs
  minimization with phase splitting (V-L, L-L, V-L-L, V-L-L-S),
  liquid-phase reactions coupled to activity models, real-gas K_eq
  corrections.
- **Aqueous electrolytes** — Pitzer single-salt (30+ salts) and
  multi-component activity (Harvie-Møller-Weare seawater preset),
  high-temperature extension to 200 °C, Davies γ correction,
  Setschenow neutral-species corrections, mineral solubility and
  saturation indices for 15+ minerals (gypsum, calcite, anhydrite,
  halite, barite, etc.).
- **Sour water** — full NH₃ / H₂S / CO₂ / H₂O speciation; single-stage
  strippers, two-stage flowsheets with HCl + NaOH dosing, automatic
  acid-dose sizing for H₂S recovery targets.
- **CO₂ capture amine flowsheets** — full carbamate / bicarbonate
  chemistry for MEA, DEA, MDEA, AMP, piperazine; PDH and Chen-Song
  activity models; rigorous Naphtali-Sandholm absorber and stripper
  columns; integrated capture flowsheets with heat-exchanger network
  closure; solvent screening across amines.

## Examples

The `examples/` directory contains a 65-example progressive curriculum.
28 of them carry built-in validation against published reference data
(NIST fluids, Robinson-Stokes activity coefficients, Marshall-Slusher
mineral solubility, Doherty-Malone azeotropes, Notz capture-flowsheet
energy demand, industrial pipeline pressure drop, crude-tower product
slates).

```bash
# Run any example standalone
PYTHONPATH=. python examples/06_compressed_natural_gas_storage.py

# Run the full curriculum as a regression suite
python tests/run_examples_tests.py
```

See `examples/00_README.md` for a tier-by-tier index.

## Tests

13 test runners, ~1240 tests total:

```bash
python tests/run_cubic_tests.py                 # 268 tests
python tests/run_activity_tests.py              # 289 tests
python tests/run_mixture_tests.py               # 248 tests
python tests/run_saft_tests.py                  # 147 tests
python tests/run_gerg_tests.py                  #  49 tests
python tests/run_distillation_tests.py          # 208 tests
python tests/run_electrolyte_tests.py           # 469 tests
python tests/run_validation_tests.py            # 199 tests
python tests/run_examples_tests.py              # 65 examples
```

## Adding a fluid

Drop a JSON file modeled on the published reference paper into
`stateprop/fluids/` (or load it from any path with
`load_fluid("/path/to/fluid.json")`). The schema mirrors the reference
EOS structure: ideal-gas Cp polynomial / Planck-Einstein contributions,
residual polynomial / exponential / Gaussian terms, critical and
reducing parameters, and saturation ancillaries. Existing files are
usable as templates; the kernel handles 9 distinct term types covering
virtually all Helmholtz-form EOS in the literature. CoolProp JSON
files can be bulk-converted via `stateprop.converter.coolprop_to_stateprop`.

## Version history

See `CHANGELOG.md` for the full per-release history from v0.1 through
v0.9.119.

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
- K. S. Pitzer, *Activity Coefficients in Electrolyte Solutions*, 2nd ed., CRC Press (1991).
- C. Harvie, N. Møller, J. H. Weare, *Geochim. Cosmochim. Acta* **48**, 723 (1984).

## License

MIT.
