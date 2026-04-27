# stateprop examples — comprehensive plan

> **Goal**: a curated set of worked examples that simultaneously serves as
> the library's tutorial layer and as its end-to-end validation suite.
> Each example is a real engineering problem with published reference
> data and demonstrates a specific capability. The collection covers the
> full breadth of the library and is the recommended on-ramp for new
> users approaching 1.0.

## Design principles

1. **Real applications, not toy problems.** Every example is something
   a chemical engineer actually does — design a column, size a flash
   drum, predict scaling, model a capture plant. Test cases come from
   the standard reference textbooks (Smith-Van Ness, Seader-Henley,
   Gmehling-Kolbe, Pitzer, Kohl-Riesenfeld) or from peer-reviewed
   regression studies.

2. **Validate against published data.** Each example has at least one
   numerical comparison against a literature reference, with explicit
   tolerance and citation. This makes the example a regression test:
   if the library breaks, the example fails.

3. **Self-contained + minimal.** A single Python file, runnable from
   the repo root with `python examples/<file>.py`. No external data
   files (use bundled chemsep / GERG / esper2023 / rehner2023). No
   plotting in headless environments unless explicitly opted in.

4. **Progressive complexity.** Examples form a learning path:
   *properties → flash → mixtures → activity → distillation →
   reaction → electrolyte → integrated plants*. A new user can read
   them in order; an advanced user can jump to the relevant one.

5. **Consistent header.** Each example begins with:
   - One-line "what this demonstrates"
   - Reference (paper, textbook, dataset)
   - Expected runtime (rough)
   - The library APIs invoked (so users find it via grep)

6. **Embedded validation.** Each script `assert`s its key quantitative
   results against the reference. Running an example *is* a test pass.
   A future CI job that runs all examples gives a second tier of
   regression coverage on top of the unit-test suite.

## Audit of existing examples (39 scripts)

The repository ships with substantial example coverage already. Below
maps existing examples to the design principles above.

### Strong coverage (keep as-is, light polishing)

| Existing | What it does | Status |
|---|---|---|
| `saturation_curve.py` | CO₂ vapor pressure triple→critical | Good — Helmholtz reference |
| `steam_tables.py` | IAPWS-95 steam in engineering format | Good — IAPWS validation |
| `rankine_cycle.py` | Steam Rankine, full state points | Good — closes energy balance |
| `plot_phase_envelope.py` | P-T, T-ρ, T-h coexistence dome | Good — pure-fluid envelope |
| `property_grid.py`, `property_grid_mass_based.py` | Tabulated properties on (T,p) grid | Good |
| `natural_gas_flash.py` | GERG-2008 5-component flash | Good — multi-fluid validation |
| `cubic_phase_envelope.py` | Critical + envelope tracing, PR vs SRK | Good |
| `binary_vle_unifac_uniquac_nrtl.py` | Ethanol-water, three γ models | Good |
| `lle_water_butanol.py` | UNIFAC-LLE + LLE flash + coverage | Good |
| `nrtl_lle_regression.py` | Fit NRTL to LLE tie-line data | Good — under-discovered |
| `auto_flash_phase_types.py` | 5-regime phase detection | Good |
| `stability_tpd_diagnostic.py` | 4-quadrant TPD with truth table | Good |
| `partial_vaporization.py` | Pα and Tα flash for drum sizing | Good |
| `saft_pure_and_mixture.py` | PC-SAFT with Wertheim association | Good |
| `transport_properties.py` | Chung viscosity, Brock-Bird ST | Good (only transport example — see gaps) |
| `reaction_equilibrium.py` | WGS, methanol synth, NH₃ synth K(T) | Good |
| `steam_methane_reforming.py` | Multi-reaction with WGS | Good |
| `real_gas_K_eq.py` | High-P fugacity correction (industrial) | Good — under-discovered |
| `liquid_phase_esterification.py` | Liquid γ-coupled equilibrium | Good |
| `reactive_flash_esterification.py` | Reactive flash with VLE | Good |
| `distillation_benzene_toluene.py` | Binary distillation + Fenske check | Good |
| `distillation_q_fraction_partial_condenser.py` | Effect of feed q | Good |
| `distillation_multifeed_sidedraw.py` | Ternary multi-feed + side draw | Good |
| `reactive_distillation_esterification.py` | RD methyl acetate synthesis | Good |
| `reactive_distillation_energy_balance.py` | RD with full EB | Good |
| `reactive_extraction_with_energy_balance.py` | RX column with EB | Good |
| `pseudo_components.py` | NBP+SG → Tc/Pc/ω | Good |
| `tbp_discretization.py` | TBP curves → cuts | Good |
| `chemsep_database.py` | 446-compound DIPPR DB | Good |
| `pcsaft_database.py` | Esper2023 + Rehner2023 | Good |
| `database_integration.py` | Cross-EOS workflow | Good |
| `crude_distillation_with_side_strippers.py` | Refinery atmospheric tower | **Headline example** |
| `electrolyte_thermodynamics.py` | Pitzer, 18 salts, Davies/DH compare | Good |
| `sour_water_stripper.py` | NH₃/H₂S sour-water column | Good (predates v0.9.111 N-S — see notes) |
| `multi_electrolyte_brines.py` | Mixed brines, seawater | Good |
| `mineral_scaling.py` | Mineral solubility / saturation index | Good |
| `lle_extraction_acetone_water_benzene.py` | Liquid-liquid extraction | Good |
| `benchmark.py` | Performance characterization | Good |

### Gaps to fill

Examples needed to bring breadth coverage to ~95%. Each is a real
engineering problem with a clear validation reference.

#### Gap 1 — Volume translation in production (v0.9.119)

There's no example for the volume-translation lookup module. The
feature is shipped, tested at the unit level, but not demonstrated
as part of a real workflow.

#### Gap 2 — CPR-compressed Jacobians visible to users (v0.9.117)

The 4-10× speedup is invisible. Users don't know it's there. Needs
a worked example showing column timing scaling with N_stages and
explicit comparison of CPR vs dense.

#### Gap 3 — Chen-Song ENRTL applied to a real amine plant (v0.9.118)

The Chen-Song activity model is plugged into AmineSystem, but no
example shows the *engineering consequence* — that switching from
PDH to Chen-Song changes the predicted regenerator P_CO₂ by a factor
of 2-3× at industrial conditions, with implications for column
design.

#### Gap 4 — Full CO₂ capture flowsheet (v0.9.108-119)

The `CaptureFlowsheet` integrator is the headline integrated capability
of the post-v0.9.107 work, with both bespoke and rigorous N-S solvers,
tray sizing, and energy targets. Currently only exercised in unit
tests. Needs a full worked example: design a 50 MWe coal-plant CO₂
capture system, sweep solvent rate, find Q/ton minimum.

#### Gap 5 — Two-stage sour-water flowsheet (v0.9.113)

The acid-dosed two-stage flowsheet is genuinely useful for refinery
operators. Currently only in unit tests. Needs an example showing
selectivity gains and the `find_acid_dose_for_h2s_recovery()` solver.

#### Gap 6 — Tray hydraulics / column sizing (v0.9.113-115)

Tray-sizing infrastructure is fully built (`tray_hydraulics.py` is 893
lines) but not exposed in any example. Users designing real columns
need to see this.

#### Gap 7 — Multicomponent GERG-2008 with departure functions

`stateprop.mixture` has ~6,000 lines and 21 GERG fluids with 210 binary
pairs (15 with departure terms). Current coverage is one example
(natural-gas flash). Missing: cryogenic separation, LNG, hydrogen
mixtures, supercritical CO₂.

#### Gap 8 — Mineral-CO₂-brine geochemistry (v0.9.101-102)

The complexation + mineral solubility framework supports geochemistry
problems (CO₂ sequestration in saline aquifers, scale prediction,
cement chemistry). One existing `mineral_scaling.py` covers basics.
Missing: a real CO₂ injection scenario.

#### Gap 9 — Reactive absorber / stripper integrated thermal design

`AmineColumn`, `AmineStripper`, `lean_rich_exchanger`,
`StripperCondenser` are exposed but not pieced together in an example.
The `electrolyte_thermodynamics.py` only goes as far as Pitzer. Need
an example that goes from "given a flue gas, design the absorber" all
the way to an energy-integrated flowsheet.

#### Gap 10 — Process module

Zero examples. The module wraps amine column setup with steady-state
helpers; it deserves at least one minimal example.

## Proposed example set (target: 50 examples)

The structure below is a curriculum. Numbers in `[brackets]` are
estimated lines of code; ★ marks new examples to be created; existing
examples are listed for completeness so the curriculum is contiguous.

### Tier 1 — Single-component thermodynamics

The fundamentals — anyone using the library starts here.

1. `01_saturation_curve.py` (= existing `saturation_curve.py`) — pure
   CO₂ saturation [80 lines]
2. `02_steam_tables.py` (= existing `steam_tables.py`) — IAPWS-95
   tabulation [120 lines]
3. `03_rankine_cycle.py` (= existing `rankine_cycle.py`) — power
   cycle [200 lines]
4. `04_phase_envelope_pure.py` (= existing `plot_phase_envelope.py`)
   [120 lines]
5. `05_property_grid.py` (= existing) — tabulate properties on (T,p)
   [80 lines]
6. ★ `06_compressed_natural_gas_storage.py` [180 lines] — Methane
   isothermal compression from 1 to 250 bar, density and energy via
   Setzmann-Wagner; compare to ideal gas. **Reference**: Helmholtz EOS
   tables (Setzmann-Wagner 1991). **Validation**: ρ within 0.1% at all
   states.
7. ★ `07_joule_thomson_inversion.py` [150 lines] — Methane and N₂
   inversion temperature curves; demonstrate `joule_thomson_coefficient`
   sign change. **Reference**: Lemmon-Span 2006 N₂; Setzmann-Wagner
   1991 CH₄.

### Tier 2 — Multi-component cubic EOS

Standard process-engineering territory. Cubic is the workhorse.

8. `08_natural_gas_flash.py` (= existing `natural_gas_flash.py`) —
   GERG vs PR comparison [130 lines]
9. `09_cubic_phase_envelope.py` (= existing `cubic_phase_envelope.py`)
   — ternary envelope [130 lines]
10. ★ `10_co2_eor_phase_behavior.py` [200 lines] — CO₂ injection into
    a 5-component oil for enhanced oil recovery; minimum miscibility
    pressure (MMP) via tie-line slope criterion at varying T.
    **Reference**: Stalkup 1983, Pina-Martinez-Privat 2022. **Validation**:
    MMP within 5% of slim-tube data at one reference T.
11. ★ `11_lng_two_phase_flash.py` [180 lines] — Liquefaction PT-flash
    of typical LNG composition (75% C1, 15% C2, etc.) at -160 °C, 1 atm;
    demonstrates GERG departure functions matter. PR vs GERG:
    ρ_liquid differs by ~3-5%. **Validation**: GERG ρ_liquid within
    0.5% of NIST LNG tables.
12. ★ `12_volume_translation_workflow.py` [150 lines] — n-Octane
    saturated liquid density vs T from 220 K to 500 K, no shift / SRK
    Peneloux / PR auto-VT. Demonstrates `cubic_from_name(volume_shift='auto')`.
    **Reference**: NIST WebBook. **Validation**: VT improves error by
    ≥ 30% over no-shift across temperature range.
13. `13_pseudo_components.py` (= existing) [180 lines]
14. `14_tbp_discretization.py` (= existing) [220 lines]
15. ★ `15_natural_gas_dewpoint_design.py` [200 lines] — Pipeline
    natural-gas dehydration spec: dew point P at -10 °C with TEG
    contactor; calculation chain: composition → cubic flash →
    dew-point T → process design. **Reference**: GPSA Engineering Data
    Book chapter on gas dehydration.
16. `16_cubic_database_integration.py` (= existing
    `database_integration.py`) [240 lines]

### Tier 3 — Activity-coefficient framework

VLE / LLE for non-ideal liquid mixtures. The core of separations work.

17. `17_binary_vle.py` (= existing `binary_vle_unifac_uniquac_nrtl.py`)
    — UNIFAC, UNIQUAC, NRTL [180 lines]
18. `18_lle_water_butanol.py` (= existing) [120 lines]
19. `19_lle_extraction.py` (= existing
    `lle_extraction_acetone_water_benzene.py`) [140 lines]
20. `20_nrtl_lle_regression.py` (= existing) — fit binary parameters
    [150 lines]
21. ★ `21_acetone_water_chloroform_residue_curves.py` [220 lines] —
    Construct residue curves for the acetone-water-chloroform ternary
    via `gamma_phi` on a UNIFAC backbone; identify the ternary
    azeotrope and feasibility regions for distillation.
    **Reference**: Doherty-Malone 2001 textbook.
22. ★ `22_excess_enthalpy_validation.py` [150 lines] — H_E predictions
    from UNIFAC vs NRTL vs UNIQUAC for ethanol-water, methanol-water,
    acetone-water at 25 °C. **Reference**: DECHEMA Heats of Mixing
    series. **Validation**: H_E within published-data scatter (~10%).
23. ★ `23_unifac_dortmund_vs_classic.py` [170 lines] — Three binary
    systems where Dortmund-modified UNIFAC outperforms classic
    UNIFAC (e.g., water-glycol, alkane-aromatic). **Reference**:
    Gmehling-Kolbe 2012.

### Tier 4 — Phase-behavior / flash diagnostics

When standard PT-flash isn't enough.

24. `24_auto_flash_phase_types.py` (= existing) [200 lines]
25. `25_stability_tpd_diagnostic.py` (= existing) [160 lines]
26. `26_partial_vaporization_drum.py` (= existing
    `partial_vaporization.py`) [200 lines]
27. ★ `27_critical_locus_binary.py` [180 lines] — Trace the critical
    locus of CO₂-methane and CO₂-ethane binaries in (T, p, x) space;
    identify Type I vs Type III phase behavior. **Reference**:
    Diamantonis-Economou 2011. **Validation**: critical T at one
    composition within 2 K.
28. ★ `28_state_function_flashes.py` [160 lines] — Five flash
    specifications on the same propane mixture: PT, PH, PS, TH, TS;
    cross-check internal consistency. Demonstrates flash type matters
    for process simulation (e.g., turbines need PS, valves need PH).

### Tier 5 — PC-SAFT and transport

For polar fluids and engineering-grade transport coefficients.

29. `29_saft_pure_and_mixture.py` (= existing) [170 lines]
30. `30_pcsaft_database.py` (= existing) [240 lines]
31. ★ `31_water_alcohol_density_pcsaft.py` [200 lines] — Liquid
    density of methanol-water and ethanol-water across composition
    at 25 °C; PC-SAFT (with association) vs PR-MC. PC-SAFT typically
    wins by 8-15% on these polar systems. **Reference**: Gmehling
    H_E volume data. **Validation**: PC-SAFT errors within 2%.
32. `32_transport_properties.py` (= existing) [200 lines]
33. ★ `33_natural_gas_pipeline_pressure_drop.py` [240 lines] — Steady-
    state isothermal pipeline calculation: 100 km, 5 bar inlet drop;
    needs viscosity (Chung), density (GERG-2008), Reynolds → friction
    factor → Δp. End-to-end engineering calculation, demonstrates the
    transport module in context. **Reference**: Mokhatab pipeline
    handbook 2015.

### Tier 6 — Reaction equilibrium

From single-reaction K(T) to multi-reaction Gibbs minimization.

34. `34_reaction_equilibrium_classics.py` (= existing
    `reaction_equilibrium.py`) [200 lines]
35. `35_steam_methane_reforming.py` (= existing) [180 lines]
36. `36_real_gas_K_eq_industrial.py` (= existing `real_gas_K_eq.py`)
    — high-P fugacity [220 lines]
37. `37_liquid_phase_esterification.py` (= existing) — γ-coupled K_eq
    [220 lines]
38. ★ `38_gibbs_minimization_methanol.py` [180 lines] — Methanol synthesis
    over a 4-species system (CO, CO₂, H₂, MeOH) via
    `gibbs_minimize_TP`. Sweep T, p, feed-CO₂ ratio; show extent
    surfaces. **Reference**: classic methanol-synthesis curves
    (Bissett 1977 + many others).

### Tier 7 — Distillation columns

Naphtali-Sandholm framework with all features.

39. `39_distillation_benzene_toluene.py` (= existing) — basic NS column
    [110 lines]
40. `40_distillation_q_partial_condenser.py` (= existing) [180 lines]
41. `41_distillation_multifeed_sidedraw.py` (= existing) [200 lines]
42. ★ `42_tray_sizing_and_flooding.py` [200 lines] — Take the
    benzene-toluene column from #39, size it for 75% flooding via
    `tray_hydraulics.py`, plot the diameter-vs-stage profile.
    **Reference**: Kister 1992 chapter on tray hydraulics.
    **Validation**: diameter within ±15% of Kister textbook example.
43. ★ `43_cpr_jacobian_speedup.py` [180 lines] — Time the same column
    at N=10, 20, 30, 40 stages with default CPR vs forced dense FD;
    plot scaling. Shows the v0.9.117 win in user-visible terms.
44. `44_reactive_distillation_methyl_acetate.py` (= existing
    `reactive_distillation_esterification.py`) [220 lines]
45. `45_reactive_distillation_energy_balance.py` (= existing) [180 lines]
46. `46_reactive_flash_esterification.py` (= existing) [240 lines]
47. `47_reactive_extraction_energy_balance.py` (= existing) [180 lines]
48. `48_crude_atmospheric_tower.py` (= existing
    `crude_distillation_with_side_strippers.py`) — **Headline
    refinery example** [320 lines]

### Tier 8 — Aqueous electrolytes

The library's distinctive strength.

49. `49_pitzer_single_salt.py` (= existing
    `electrolyte_thermodynamics.py`) [220 lines]
50. `50_multi_electrolyte_brines.py` (= existing) [280 lines]
51. `51_mineral_scaling.py` (= existing) [280 lines]
52. ★ `52_co2_sequestration_in_aquifer.py` [260 lines] — CO₂ injection
    into a saline-brine aquifer at reservoir T (60 °C) and p (200 bar):
    dissolution, brine-CO₂ density change (drives convective mixing),
    pH evolution, calcite saturation index. **Reference**: Duan-Sun
    2003 CO₂ solubility, Gilfillan 2009 sequestration data.
53. ★ `53_seawater_carbonate_chemistry.py` [220 lines] — Surface and
    deep ocean: pH, total alkalinity, DIC, ΩCalcite, ΩAragonite as a
    function of pCO₂. Relevant for ocean acidification studies.
    **Reference**: Doney 2009, Millero 2007.
54. ★ `54_geothermal_brine_at_300C.py` [200 lines] — High-T brine
    with NaCl, KCl, CaCl₂ at 300 °C in a geothermal well; uses the
    v0.9.116 high-T Pitzer extension. **Reference**: Møller 1988
    plus Pabalan-Pitzer 1988. **Validation**: γ_± at 1 m NaCl, 300 °C
    within 1.5%.

### Tier 9 — Sour water

Real refinery operations problem.

55. ★ `55_sour_water_stripper_basic.py` (replaces existing
    `sour_water_stripper.py` to use v0.9.111 N-S framework) [240 lines]
    — 15-stage steam-stripped column, full N-S convergence, energy
    balance, Murphree efficiency. **Reference**: Eckert 1988 sour-
    water stripper design.
56. ★ `56_sour_water_two_stage.py` [240 lines] — Two-stage flowsheet
    with acid dosing (v0.9.113): selective H₂S recovery in stage 1
    (acid-dosed) followed by NH₃ recovery in stage 2. Use
    `find_acid_dose_for_h2s_recovery()` to size the acid stream.
    **Reference**: typical refinery practice (Beychok handbook).
    **Validation**: H₂S recovery > 95%, NH₃ < 10% at acid dose.
57. ★ `57_sour_water_with_pitzer_corrections.py` [180 lines] — Same
    column at brine-loaded conditions (5 mol/kg NaCl background);
    `pitzer_corrections=True` activates Setschenow + Davies γ. Shows
    the v0.9.116 corrections in context.

### Tier 10 — CO₂ capture

The integrated chemistry-coupled separation flowsheet.

58. ★ `58_amine_absorber_design.py` [240 lines] — Stand-alone MEA
    absorber: 12-stage column, CO₂ recovery vs L/G ratio, lean-loading
    sweep. Shows how loading α determines absorber height. Uses
    `amine_absorber_ns()` (v0.9.114). **Reference**: Cousins 2011 pilot
    plant data.
59. ★ `59_amine_stripper_design.py` [220 lines] — Companion stripper
    column: 15 stages, reboiler + condenser, energy balance, Q/ton CO₂
    minimization vs reflux ratio. Uses `amine_stripper_ns()`.
60. ★ `60_amine_capture_flowsheet_pdh.py` [280 lines] — Complete
    CaptureFlowsheet (v0.9.108) for a 50 MWe coal-plant flue gas
    (G=15 mol/s, y_CO2=12%): absorber + lean-rich HX + stripper +
    condenser, all coupled, with the bespoke α-Newton solver.
    **Reference**: Notz 2012 / Cousins 2011 baseline. **Validation**:
    Q/ton within 0.3 GJ/t of typical values (3.5-4.5 GJ/t).
61. ★ `61_amine_capture_flowsheet_rigorous.py` [280 lines] — Same
    flowsheet via `solver='ns'` (v0.9.115), with tray sizing
    (`size_trays=True`). Compare bespoke vs N-S: shows
    accuracy/runtime trade-off and the tower diameters.
62. ★ `62_chen_song_vs_pdh_at_high_T.py` [220 lines] — Side-by-side
    P_CO₂(α, T) prediction by activity model = pdh / chen_song /
    davies for 30 wt% MEA. Plots match published Jou-Mather-Otto 1995
    data within ±50% on Chen-Song vs ±100% on PDH. Shows the
    v0.9.118 improvement quantitatively.
63. ★ `63_solvent_screening_mea_mdea_dea.py` [240 lines] — Same
    flowsheet swept across three amines (MEA, MDEA, DEA) at fixed
    capacity. Compare Q/ton, solvent flow, regenerator pressure.
    Shows why pilot-scale studies do this benchmarking.

### Tier 11 — Reaction-coupled separations and other reactive systems

64. `64_reactive_distillation_methyl_acetate.py` (= existing) [220 lines]
65. ★ `65_acid_gas_removal_with_co2_capture.py` [280 lines] —
    Combined sour-water stripper + amine capture for refinery
    fuel-gas treatment. Demonstrates the chemistry-coupled separations
    differentiator: refinery operators rarely have a pure
    "either-or" between these two unit operations.
66. ★ `66_aqueous_phase_reaction_with_pitzer.py` [200 lines] — Hydrolysis
    or precipitation reaction in a brine background; demonstrates
    `liquid_phase` reaction module + Pitzer activity coupling. Real
    use case: brine treatment in oil & gas operations.

### Tier 12 — Cross-cutting / advanced

67. ★ `67_thermo_consistency_checks.py` [200 lines] — Demonstrate
    Maxwell-relation consistency, exact `(∂h/∂p)_T = v - T(∂v/∂T)_p`,
    Gibbs-Helmholtz, Clausius-Clapeyron at saturation. Uses three
    pure-component EOSs (CO₂ Helmholtz, CO₂ PR, CO₂ PC-SAFT) to
    show cross-EOS internal consistency.
68. ★ `68_phase_envelope_with_critical_point.py` [220 lines] —
    Trace a phase envelope, locate the cricondentherm, cricondenbar,
    and critical point on a 5-component natural-gas mixture; show
    the analytic Heidemann-Khalil critical detector. **Reference**:
    Whitson-Brule 2000 textbook example.

## Implementation path

The plan is to bring the example set from 39 files (current) to 68
files (target), through five phased commits. Each phase ships
independently and is internally validated.

### Phase A — Reorganize existing examples (1 commit, ~200 lines change)

Reorder the existing 39 files into the tier-numbered structure above.
Renaming, no new content. Result: anyone browsing `examples/` sees a
clear curriculum.

- Add a `00_README.md` in `examples/` that describes the curriculum
- Rename existing files with `NN_` prefixes per the plan above
- Update `examples/README.md` (top-level) to point to `00_README.md`
- Add a `run_all_examples.py` script that exercises every example as
  a smoke test (timing budget: each ≤ 60 s, total ≤ 30 min)

### Phase B — Validation harness (1 commit, ~300 lines)

Build the harness that turns examples into regression tests:

- `tests/test_examples_run.py` — imports each example as a module,
  catches assertion failures, fails if any example breaks
- Each new ★ example begins with assertions against published values
- `make examples-test` target in a Makefile (or equivalent)
- Document the convention: every example has at least one `assert`
  and a clear citation in the docstring

### Phase C — Headline new examples (5 commits, ~1500 lines)

The 5 highest-leverage new examples that close the most-cited gaps:

1. `12_volume_translation_workflow.py` (gap 1, v0.9.119)
2. `43_cpr_jacobian_speedup.py` (gap 2, v0.9.117)
3. `60_amine_capture_flowsheet_pdh.py` (gap 4, headline)
4. `62_chen_song_vs_pdh_at_high_T.py` (gap 3, v0.9.118)
5. `42_tray_sizing_and_flooding.py` (gap 6)

These five alone bring the post-v0.9.107 work into the user-visible
domain.

### Phase D — Deeper application examples (8 commits, ~2400 lines)

The next-most-valuable domain examples:

6. `10_co2_eor_phase_behavior.py` (gap 7)
7. `11_lng_two_phase_flash.py` (gap 7)
8. `52_co2_sequestration_in_aquifer.py` (gap 8)
9. `54_geothermal_brine_at_300C.py` (gap 8)
10. `56_sour_water_two_stage.py` (gap 5)
11. `61_amine_capture_flowsheet_rigorous.py`
12. `63_solvent_screening_mea_mdea_dea.py`
13. `65_acid_gas_removal_with_co2_capture.py`

### Phase E — Closing breadth gaps (16 commits, ~3000 lines)

The remaining ★ examples to bring breadth coverage to 95+%. Each is
small (150-220 lines) and self-contained.

- `06_compressed_natural_gas_storage.py`
- `07_joule_thomson_inversion.py`
- `15_natural_gas_dewpoint_design.py`
- `21_acetone_water_chloroform_residue_curves.py`
- `22_excess_enthalpy_validation.py`
- `23_unifac_dortmund_vs_classic.py`
- `27_critical_locus_binary.py`
- `28_state_function_flashes.py`
- `31_water_alcohol_density_pcsaft.py`
- `33_natural_gas_pipeline_pressure_drop.py`
- `38_gibbs_minimization_methanol.py`
- `53_seawater_carbonate_chemistry.py`
- `55_sour_water_stripper_basic.py` (replaces existing)
- `57_sour_water_with_pitzer_corrections.py`
- `58_amine_absorber_design.py`
- `59_amine_stripper_design.py`
- `66_aqueous_phase_reaction_with_pitzer.py`
- `67_thermo_consistency_checks.py`
- `68_phase_envelope_with_critical_point.py`

### Phase F — Documentation finalization (1 commit, ~400 lines)

Produce the user-facing tutorial-doc structure:

- `docs/getting_started.md` — points new users to Tier 1
- `docs/tutorials/` — Markdown narrative wrapping each tier
  (~3-5 paragraphs per tier explaining when to reach for these tools)
- README badges showing example coverage and validation pass rate
- A "navigating the library" decision tree:
  *Property at (T, p)* → Helmholtz / cubic / PC-SAFT decision
  *Two-phase flash* → cubic / GERG decision
  *Liquid-liquid* → activity-coefficient framework
  *Distillation column* → reactive vs non-reactive, EB vs CMO
  *Aqueous chemistry* → Pitzer / amine / sour-water decision

## Success criteria

The example set hits its targets when:

1. Every public class / function has at least one example that
   constructs it (search `grep -l ClassName examples/`)
2. Every benchmark in `tests/run_validation_tests.py` is mirrored
   by an example that *demonstrates* the same capability with prose
   and reference data (so users see the "why" not just the "test
   asserts X")
3. Running `python -m pytest examples/` (after phase B) gives ≥ 95%
   pass rate
4. A new user can answer "how do I do X with stateprop?" by browsing
   `examples/00_README.md` for any X drawn from the table of
   contents above
5. The post-v0.9.107 work (capture flowsheet, sour-water flowsheet,
   tray sizing, Chen-Song, CPR Jacobian, volume translation) all
   have user-visible examples

## What this is NOT

This plan does not include:

- API reference documentation (different deliverable, autogenerated
  from docstrings via Sphinx or similar)
- Internal-architecture explanation docs (different audience, will
  follow later)
- Performance optimization tuning (covered by the existing
  `benchmark.py`)
- Test infrastructure beyond the example-as-test convention (the
  `tests/run_*.py` framework is already substantial)

## Estimated effort

For the budget-conscious view: phases A and B are foundational
(2 commits, ~500 lines, 2 sessions of work). Phase C closes the
most-visible v0.9.116-119 gap with 5 examples (~1500 lines,
3-5 sessions). Phases D and E add breadth and can be parallelized
across multiple sessions. Phase F is final polish.

A pragmatic minimum path to 1.0: phases A + B + C alone, deferring D
and E to v1.x. That brings the curriculum-organization and the
post-v0.9.107 visibility up to 1.0 quality without committing to all
24 new examples.
