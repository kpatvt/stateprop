# stateprop examples

A 65-example curriculum showing what the library can do, organized as a
progressive walk from single-fluid properties to integrated process
flowsheets.  Every example runs standalone:

```bash
PYTHONPATH=. python examples/<file>.py
```

## What this library does

**Pure-fluid thermodynamics** — reference-grade Helmholtz EOS (water,
methane, nitrogen, etc.) for property evaluation, saturation curves,
phase envelopes, Joule-Thomson behavior, gas storage calculations,
Rankine cycles.

**Mixture EOS** — Peng-Robinson and SRK cubics with volume translation,
GERG-2008 multi-fluid for natural gas, PC-SAFT with Wertheim
association for polar and hydrogen-bonding fluids.  Density, fugacity,
and full state-function flashes (PT, PH, PS, TH, TS).

**Activity coefficient models** — NRTL with parameter regression,
UNIQUAC, classical UNIFAC, Dortmund-modified UNIFAC, with a built-in
group-contribution database for ~130 compounds.

**Phase behavior** — two-phase and three-phase flash, LLE extraction,
critical-locus tracing, residue-curve maps for ternary distillation
design, stability analysis via tangent-plane distance.

**Aqueous electrolyte chemistry** — Pitzer single- and multi-salt
activity (Harvie-Møller-Weare seawater preset), high-temperature
extension to 200 °C, mineral solubility and saturation indices for
15+ minerals (gypsum, calcite, anhydrite, halite, barite…), CO₂
sequestration geochemistry, ocean acidification.

**Sour water** — full NH₃ / H₂S / CO₂ / H₂O speciation with optional
Davies γ corrections; single-stage strippers, two-stage flowsheets
with HCl + NaOH dosing, automatic acid-dose sizing for H₂S recovery
targets.

**CO₂ capture amine flowsheets** — full carbamate / bicarbonate
chemistry for MEA, DEA, MDEA, AMP, piperazine; Naphtali-Sandholm
absorber and stripper columns; integrated capture flowsheets with
heat-exchanger network closure; solvent screening across amines.

**Distillation** — multi-stage columns with arbitrary feed and
side-draw topology, side strippers, partial / total condensers,
Murphree efficiency, tray hydraulics with Souders-Brown flooding.
Reactive distillation and reactive extraction with energy balance.

**Refinery characterization** — TBP-curve discretization into
pseudo-components with Riazi-Daubert / Lee-Kesler / Edmister
correlation network.  Atmospheric crude tower with side strippers
producing realistic naphtha / kerosene / diesel / residue cuts.

**Reaction equilibrium** — single and multi-reaction systems, Gibbs
minimization with phase splitting (V-L, L-L, V-L-L, V-L-L-S),
liquid-phase reactions coupled to activity models.

**Transport** — Chung viscosity and thermal conductivity, Stiel-Thodos,
Wilke and Wassiljewa mixing rules, Brock-Bird and Macleod-Sugden
surface tension.

## Status

65 examples, all passing.  28 ship with built-in validation against
published reference data — 191 reproduced numbers covering NIST
fluids, Robinson-Stokes activity coefficients, Marshall-Slusher
mineral solubility, Doherty-Malone azeotropes, Notz capture-flowsheet
energy demand, industrial pipeline pressure drop, and crude-tower
product slates.

## How to navigate

| If you want to... | See |
|---|---|
| Pure-fluid properties at (T, p) | Tier 1 |
| Mixture flash, dewpoint design, pipeline calculations | Tier 2 |
| VLE / LLE with activity coefficients | Tier 3 |
| Phase behavior diagnostics, residue curves | Tier 4 |
| Polar fluids (PC-SAFT), transport coefficients | Tier 5 |
| Reaction equilibrium and Gibbs minimization | Tier 6 |
| Distillation column design, reactive separations | Tier 7 |
| Aqueous electrolytes, scaling, geochemistry | Tier 8 |
| Sour-water stripping | Tier 9 |
| CO₂ capture amine flowsheets | Tier 10 |
| Multi-discipline integration, consistency checks | Tier 11 |

★ marks examples with built-in published-data validation.

### Tier 1 — Pure fluids (Helmholtz reference EOS)

| 01 | Saturation curves | 05 | Property grid (mole-basis) |
| 02 | Steam tables | 05b | Property grid (mass-basis) |
| 03 | Rankine cycle | 06 ★ | Compressed natural gas storage |
| 04 | Phase envelope | 07 ★ | Joule-Thomson inversion locus |

### Tier 2 — Mixtures and cubic EOSes

| 08 | Natural gas flash | 16 / 16b / 16c | Database integration |
| 09 | Cubic phase envelope | 28 ★ | State-function flashes (PH/PS/TH/TS) |
| 11 ★ | LNG flash: GERG vs PR | 33 ★ | Pipeline pressure drop |
| 12 ★ | Volume-translation workflow | 13 / 14 | Pseudo-components, TBP |
| 15 ★ | Pipeline dewpoint design | | |

### Tier 3 — Activity coefficients and LLE

| 17 | Binary VLE | 20 | NRTL parameter regression |
| 18 | LLE — water-butanol | 23 ★ | UNIFAC: classic vs Dortmund |
| 19 | LLE — extraction column | | |

### Tier 4 — Phase-behavior diagnostics

| 21 ★ | Residue-curve map | 25 | Stability TPD diagnostic |
| 22 ★ | Excess-enthalpy validation | 26 | Partial-vaporization drum |
| 24 | Auto-flash phase typing | 27 ★ | Critical-locus binary |

### Tier 5 — Polar fluids and transport

| 29 | PC-SAFT pure and mixture | 32 | Transport (μ, k, σ) |
| 31 ★ | Water-alcohol density: PC-SAFT vs PR | | |

### Tier 6 — Chemical reaction equilibrium

| 34 | Reaction equilibrium classics | 37 | Liquid-phase esterification |
| 35 | Steam-methane reforming | 38 ★ | Gibbs minimization — methanol |
| 36 | Real-gas K_eq | | |

### Tier 7 — Distillation and reactive separations

| 39 | Benzene-toluene | 44 | Reactive distillation — methyl acetate |
| 40 | Q-spec and partial condenser | 45 | Reactive distillation + energy balance |
| 41 | Multi-feed and side-draw | 46 | Reactive flash — esterification |
| 42 ★ | Tray sizing and flooding | 47 | Reactive extraction + energy balance |
| 43 ★ | CPR Jacobian speedup | 48 ★ | Crude atmospheric tower |

### Tier 8 — Aqueous electrolytes and geochemistry

| 49 | Pitzer single-salt | 52 ★ | CO₂ aquifer sequestration |
| 50 | Multi-electrolyte brines | 53 ★ | Seawater carbonate chemistry |
| 51 | Mineral solubility and scaling | 54 ★ | Geothermal brine, high-T Pitzer |

### Tier 9 — Sour water

| 55 | Sour-water stripper basic | 57 ★ | Pitzer / Davies γ corrections |
| 56 ★ | Two-stage flowsheet (HCl + NaOH) | | |

### Tier 10 — CO₂ capture amine flowsheets

| 58 ★ | Amine absorber design | 62 ★ | Chen-Song vs PDH at high T |
| 59 ★ | Amine stripper design | 63 ★ | Solvent screening — MEA/DEA/MDEA |
| 60 ★ | Capture flowsheet (PDH) | | |

### Tier 11 — Multi-discipline integration

| 66 ★ | Mineral dissolution + Pitzer | 67 ★ | Thermodynamic consistency checks |

## Running the examples as a test suite

```bash
python tests/run_examples_tests.py
```

Each example exits non-zero on validation failure, suitable for CI.
