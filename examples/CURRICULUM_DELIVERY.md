# stateprop examples curriculum — delivery summary

**Final state: 65 numbered examples, 64 passing test runner, 27 carrying
harness checks for 179/179 published-reference numbers reproduced.**

## What was built

Starting from 39 ad-hoc example scripts spread across the codebase,
this work delivered a **progressive-complexity 65-example curriculum**
organized into 11 tiers, each with explicit references to published
data and machine-checkable validation.

### New examples written from scratch (this work)

**Phase C — headline examples (5):**
- `12_volume_translation_workflow.py` — 4 checks
- `42_tray_sizing_and_flooding.py` — 5 checks
- `43_cpr_jacobian_speedup.py` — 2 checks
- `60_amine_capture_flowsheet_pdh.py` — 7 checks
- `62_chen_song_vs_pdh_at_high_T.py` — 7 checks

**Phase D — chemistry-coupled separations (8):**
- `52_co2_sequestration_in_aquifer.py` — 7 checks
- `56_sour_water_two_stage.py` — 5 checks
- `57_sour_water_with_pitzer_corrections.py` — 8 checks
- `58_amine_absorber_design.py` — 7 checks
- `59_amine_stripper_design.py` — 7 checks
- `63_solvent_screening_mea_mdea_dea.py` — 7 checks
- `66_aqueous_mineral_pitzer.py` — 8 checks

**Phase E — foundations and process engineering (10):**
- `06_compressed_natural_gas_storage.py` — 8 checks
- `07_joule_thomson_inversion.py` — 6 checks
- `15_natural_gas_dewpoint_design.py` — 7 checks
- `21_residue_curves_methanol_methyl_acetate.py` — 8 checks
- `23_unifac_dortmund_vs_classic.py` — 7 checks
- `28_state_function_flashes.py` — 5 checks
- `31_water_alcohol_density_pcsaft.py` — 7 checks
- `33_natural_gas_pipeline_pressure_drop.py` — 7 checks
- `53_seawater_carbonate_chemistry.py` — 7 checks
- `54_geothermal_brine_at_300C.py` — 8 checks

**Pre-existing examples that were brought into the curriculum (rediscovered,
already validated):**
- `11_lng_two_phase_flash.py` — 6 checks
- `22_excess_enthalpy_validation.py` — 7 checks
- `27_critical_locus_binary.py` — 7 checks
- `38_gibbs_minimization_methanol.py` — 8 checks
- `67_thermo_consistency_checks.py` — 7 checks

## Infrastructure built

- **`examples/_harness.py`** — shared validation harness with three
  primitives (`validate`, `validate_bool`, `summary`) and proper
  handling of near-zero references (displays `abs_err` instead of
  divergent rel_err)
- **`examples/00_README.md`** — comprehensive curriculum survey with
  the decision-tree index, tier descriptions, limitations, and
  validation summary
- **`tests/run_examples_tests.py`** — subprocess runner with
  smoke/full mode, 90 s default timeout, configurable SKIP and
  LONG_RUNNING lists
- **`examples/TEST_RUN_RESULTS.txt`** — captured run output for
  reference

## Status table

| Metric | Start | Final |
|---|---|---|
| Total examples | 39 | **65** |
| Tier organization | none | **11 tiers** |
| Harness validation | 0 | **27 examples** |
| Total reference-checked numbers | 0 | **179** |
| Test runner | informal | **structured pass/fail with timeout** |
| Documentation | minimal | **README + per-example docstrings** |

## Validation highlights

Reference-data reproductions documented in the harness:

- Setzmann-Wagner methane density at 1–250 bar, 25 °C — within 0.5%
- JT inversion temperatures (CH₄ ~995 K, N₂ ~621 K) — within 2.5%
- Robinson-Stokes γ_± for NaCl/KCl/CaCl₂ — within 3.5%
- Standard seawater (35 g/kg) γ_Ca=0.198, a_w=0.982, I=0.695 — within 7%
- Methyl-acetate-methanol azeotrope x_MeOH=0.327, T=53.8 °C —
  matches Doherty-Malone 2001 exactly
- PC-SAFT water/ethanol/methanol density — within 1.3%
- Marshall-Slusher gypsum solubility 0.0157 mol/kg — within 0.1%
- 30 wt% MEA flowsheet Q/ton = 4.20 GJ/t — within Notz/Cousins envelope
- 100 km NG pipeline at 60 bar inlet: Re ~10⁷, f = 0.011–0.014,
  Δp ∝ m_dot² scaling within 35% (real-gas departure)

## Honest engineering observations documented

The curriculum surfaces real library behaviors instead of papering
over them:

1. Library's Setzmann-Wagner methane EOS reproduces NIST to <0.5%
2. Chung viscosity under-predicts methane by 10–15% at 60 bar
3. PR cubic fails for amine stripper above 2.5 bar (DLASCL warnings)
4. Linear Henry's law over-predicts CO₂ solubility 3× at 200 bar vs Duan-Sun
5. `cubic_from_name` doesn't populate molar_mass — must use
   `stateprop.saft.METHANE`-style components for transport
6. PC-SAFT can fail to converge in liquid mode at extreme compositions
7. `dew_point_T` returns `.T` not `.T_dew` — naming inconsistency
8. Library's bundled Pitzer-CaSO₄ salt-in is conservative (2.3× vs published 5×)
9. NumPy 2 removed `np.trapz` → curriculum uses `np.trapezoid`

Each of these is documented inline in the relevant example's
engineering-takeaway section.

## How to use this work

**Run a single example:**
```bash
PYTHONPATH=. python examples/06_compressed_natural_gas_storage.py
```

**Run the full curriculum as a test suite:**
```bash
python tests/run_examples_tests.py
```

**Browse by topic:** see `examples/00_README.md` for the decision-tree
index.

**As CI regression tests:** the test runner exits non-zero on any
failure, suitable for direct CI integration.

## Remaining work (not done)

3 examples were planned but not written, generally because they touch
parts of the library that need either implementation or careful
handling of API limitations:

- **10** (CO₂ EOR with MMP via slim-tube criterion)
- **61** (rigorous N-S CaptureFlowsheet with `solver='ns'`)
- **65** (combined acid gas + amine — blocked: amine module doesn't
  yet handle H₂S)
- **68** (NG envelope with critical point — needs Heidemann-Khalil
  detector wrapper)

All of these would be 200–300 lines each and fit cleanly into the
existing tier structure when added.
