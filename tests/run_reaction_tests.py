"""Tests for stateprop.reaction module (v0.9.61)."""
from __future__ import annotations
import sys
import math
import numpy as np

sys.path.insert(0, '.')


_passed = 0
_failed = 0
_failures = []


def check(label, ok):
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        _failures.append(label)
        print(f"  FAIL  {label}")


def section(name):
    print(f"\n[{name}]")


# ------------------------------------------------------------------
# Thermochemistry consistency
# ------------------------------------------------------------------

def test_species_self_consistency_at_298():
    """All species must satisfy H(298.15) = Hf_298 and S(298.15) = Sf_298
    after auto-calibration of Shomate F/G."""
    from stateprop.reaction.thermo import BUILTIN_SPECIES
    section("test_species_self_consistency_at_298")
    bad = []
    for sp in BUILTIN_SPECIES.values():
        # Skip H2O since its lowest-T Shomate block starts at 500K
        if sp.name == 'H2O':
            continue
        dH = sp.H(298.15) - sp.Hf_298
        dS = sp.S(298.15) - sp.Sf_298
        if abs(dH) > 1.0 or abs(dS) > 1e-3:
            bad.append((sp.name, dH, dS))
    check(f"All non-H2O species satisfy H/S at 298.15 K: {bad}", not bad)


def test_elements_have_zero_formation():
    """Standard-state elements: H2, N2, O2 have Hf_298 = Gf_298 = 0."""
    from stateprop.reaction.thermo import BUILTIN_SPECIES
    section("test_elements_have_zero_formation")
    for name in ['H2', 'N2', 'O2']:
        sp = BUILTIN_SPECIES[name]
        check(f"{name} Hf_298 = 0", sp.Hf_298 == 0.0)
        check(f"{name} Gf_298 = 0", sp.Gf_298 == 0.0)


# ------------------------------------------------------------------
# Reaction K_eq values
# ------------------------------------------------------------------

def test_water_gas_shift_K_eq():
    """K_eq for CO + H2O = CO2 + H2 at 800 K should be approx 4.24
    (published values: 4.0-4.5)."""
    from stateprop.reaction import Reaction
    section("test_water_gas_shift_K_eq")
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
                                products={'CO2': 1, 'H2': 1})
    K = rxn.K_eq(800.0)
    check(f"K_eq(800K) = {K:.3f} in [3.0, 6.0]", 3.0 < K < 6.0)
    # WGS is exothermic: dH < 0
    dH = rxn.dH_rxn(800.0)
    check(f"WGS exothermic: dH(800K) = {dH/1000:.2f} kJ/mol < 0",
          dH < 0)


def test_methanol_synthesis_K_eq_temperature_trend():
    """CO + 2 H2 = CH3OH is exothermic, so K_eq should DECREASE with T."""
    from stateprop.reaction import Reaction
    section("test_methanol_synthesis_K_eq_temperature_trend")
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                products={'CH3OH': 1})
    Ks = [rxn.K_eq(T) for T in [400, 500, 600, 700]]
    check(f"K_eq monotonically decreasing: {[f'{k:.2e}' for k in Ks]}",
          all(Ks[i] > Ks[i+1] for i in range(len(Ks)-1)))


def test_ammonia_synthesis_K_eq_high_at_low_T():
    """N2 + 3 H2 = 2 NH3 is exothermic; K_eq large at 400 K, small at 800 K."""
    from stateprop.reaction import Reaction
    section("test_ammonia_synthesis_K_eq_high_at_low_T")
    rxn = Reaction.from_names(reactants={'N2': 1, 'H2': 3},
                                products={'NH3': 2})
    K_400 = rxn.K_eq(400.0)
    K_800 = rxn.K_eq(800.0)
    check(f"K(400) >> K(800): {K_400:.2e} vs {K_800:.2e}",
          K_400 > 100 * K_800)
    # Sanity: K(400) > 1, K(800) < 1
    check(f"K(400) > 1: {K_400:.2f}", K_400 > 1)
    check(f"K(800) < 1: {K_800:.2e}", K_800 < 1)


# ------------------------------------------------------------------
# Equilibrium extent solver
# ------------------------------------------------------------------

def test_extent_solver_water_gas_shift():
    """At 800K, 10 bar, 1 mol CO + 1 mol H2O should give ~67% CO conversion."""
    from stateprop.reaction import Reaction
    section("test_extent_solver_water_gas_shift")
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
                                products={'CO2': 1, 'H2': 1})
    r = rxn.equilibrium_extent_ideal_gas(
        T=800.0, p=10e5, n_initial=[1.0, 1.0, 0.0, 0.0])
    check(f"Converged: {r.converged}", r.converged)
    conv = 1 - r.n_eq[0] / 1.0
    check(f"CO conversion in (0.5, 0.85): {conv:.3f}", 0.5 < conv < 0.85)
    # Mole fractions sum to 1
    check(f"y sums to 1: {r.y_eq.sum():.6f}",
          abs(r.y_eq.sum() - 1.0) < 1e-9)
    # K from product partial pressures matches K_eq(T)
    p_ref = 1e5
    K_check = ((r.y_eq[2] * r.p / p_ref) * (r.y_eq[3] * r.p / p_ref)
               / ((r.y_eq[0] * r.p / p_ref) * (r.y_eq[1] * r.p / p_ref)))
    check(f"K from y matches K_eq: {K_check:.4f} vs {r.K_eq:.4f}",
          abs(K_check - r.K_eq) < 1e-3)


def test_extent_solver_le_chatelier_pressure():
    """Methanol synthesis CO + 2H2 = CH3OH. Higher pressure should INCREASE
    conversion (Le Chatelier: net mole reduction)."""
    from stateprop.reaction import Reaction
    section("test_extent_solver_le_chatelier_pressure")
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                products={'CH3OH': 1})
    r_low = rxn.equilibrium_extent_ideal_gas(
        T=500.0, p=10e5, n_initial=[1.0, 2.0, 0.0])
    r_high = rxn.equilibrium_extent_ideal_gas(
        T=500.0, p=200e5, n_initial=[1.0, 2.0, 0.0])
    conv_low = 1 - r_low.n_eq[0] / 1.0
    conv_high = 1 - r_high.n_eq[0] / 1.0
    check(f"Higher p => higher conversion: {conv_low:.3f} vs {conv_high:.3f}",
          conv_high > conv_low)


def test_extent_solver_le_chatelier_inerts():
    """Ammonia synthesis N2 + 3H2 = 2NH3. Adding inerts at fixed total p
    should DECREASE conversion (lowers reactant partial pressures)."""
    from stateprop.reaction import Reaction
    section("test_extent_solver_le_chatelier_inerts")
    rxn = Reaction.from_names(reactants={'N2': 1, 'H2': 3},
                                products={'NH3': 2})
    r_no_inert = rxn.equilibrium_extent_ideal_gas(
        T=500.0, p=200e5, n_initial=[1.0, 3.0, 0.0], n_inert=0.0)
    r_with_inert = rxn.equilibrium_extent_ideal_gas(
        T=500.0, p=200e5, n_initial=[1.0, 3.0, 0.0], n_inert=2.0)
    check(f"Inerts reduce extent: {r_no_inert.xi:.4f} vs {r_with_inert.xi:.4f}",
          r_with_inert.xi < r_no_inert.xi)


def test_extent_solver_thermo_consistency():
    """At equilibrium: dG_rxn(T) + RT*ln(prod (y_i*p/p_ref)^nu_i) = 0."""
    from stateprop.reaction import Reaction
    from stateprop.reaction.thermo import R_GAS
    section("test_extent_solver_thermo_consistency")
    rxn = Reaction.from_names(reactants={'N2': 1, 'H2': 3},
                                products={'NH3': 2})
    T, p = 500.0, 200e5
    r = rxn.equilibrium_extent_ideal_gas(
        T=T, p=p, n_initial=[1.0, 3.0, 0.0])
    # dG_rxn + RT * Sum_i nu_i * ln(y_i * p/p_ref) = 0 at equilibrium
    nu = rxn.nu
    p_ref = 1e5
    rt_ln_term = R_GAS * T * float((nu * np.log(r.y_eq * p / p_ref)).sum())
    residual = rxn.dG_rxn(T) + rt_ln_term
    # Tolerance: at 500 K, RT ~ 4150 J/mol; relative tolerance 1e-4
    check(f"Equilibrium thermo residual = {residual:.2f} J/mol (~ 0)",
          abs(residual) < 1.0)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_get_species_unknown_raises():
    """get_species on unknown name raises KeyError."""
    from stateprop.reaction import get_species
    section("test_get_species_unknown_raises")
    raised = False
    try:
        get_species('nonexistent_species_xyz')
    except KeyError:
        raised = True
    check("KeyError raised for unknown species", raised)


def test_get_species_synonyms():
    """Synonyms resolve."""
    from stateprop.reaction import get_species
    section("test_get_species_synonyms")
    pairs = [('water', 'H2O'), ('h2o', 'H2O'), ('methane', 'CH4'),
             ('co', 'CO'), ('hydrogen', 'H2'), ('nitrogen', 'N2'),
             ('methanol', 'CH3OH')]
    all_ok = all(get_species(syn).name == get_species(canonical).name
                  for syn, canonical in pairs)
    check(f"All {len(pairs)} synonyms resolve", all_ok)


def test_reaction_extent_zero_initial_product_ok():
    """Solver should handle n_initial = 0 for products (correct lower bound)."""
    from stateprop.reaction import Reaction
    section("test_reaction_extent_zero_initial_product_ok")
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
                                products={'CO2': 1, 'H2': 1})
    r = rxn.equilibrium_extent_ideal_gas(
        T=800.0, p=10e5, n_initial=[1.0, 1.0, 0.0, 0.0])
    check(f"Converged with zero product initial: {r.converged}", r.converged)


# ------------------------------------------------------------------
# Multi-reaction equilibrium (v0.9.62)
# ------------------------------------------------------------------

def test_multi_reaction_construction():
    """MultiReaction merges species across reactions, builds R x N stoich matrix."""
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_construction")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    check(f"R = 2: {system.R}", system.R == 2)
    check(f"N = 5 (CH4, H2O, CO, H2, CO2): {system.N}", system.N == 5)
    check(f"Species names {system.species_names}",
          set(system.species_names) == {'CH4', 'H2O', 'CO', 'H2', 'CO2'})
    # Check stoichiometry matrix shape
    check(f"nu shape (2, 5): {system.nu.shape}", system.nu.shape == (2, 5))
    # dn for steam reforming: 1+3-1-1 = 2; for WGS: 1+1-1-1 = 0
    check(f"dn = [2, 0]: {system.dn}",
          np.allclose(system.dn, [2.0, 0.0]))


def test_multi_reaction_linear_dependence_rejected():
    """Linearly dependent reactions raise ValueError."""
    from stateprop.reaction import MultiReaction, Reaction
    section("test_multi_reaction_linear_dependence_rejected")
    # Two identical reactions are linearly dependent
    r1 = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
                                products={'CO2': 1, 'H2': 1})
    r2 = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
                                products={'CO2': 1, 'H2': 1})
    raised = False
    try:
        MultiReaction([r1, r2])
    except ValueError as e:
        raised = "linearly independent" in str(e).lower() or "rank" in str(e).lower()
    check("Linearly dependent reactions raise ValueError", raised)


def test_multi_reaction_smr_steam_reforming():
    """Steam methane reforming at T=1100K, p=1bar, S/C=3.

    Industrial reforming: CH4 conversion 95-99%, H2 yield ~3.3 mol/mol CH4,
    CO/CO2 ratio ~2 at high T, dropping toward 1 with temperature decrease.
    """
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_smr_steam_reforming")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    r = system.equilibrium_ideal_gas(
        T=1100.0, p=1e5, n_initial={'CH4': 1.0, 'H2O': 3.0})
    check(f"Converged: {r.converged}", r.converged)

    n_CH4 = r.n_eq[r.species.index('CH4')]
    n_H2 = r.n_eq[r.species.index('H2')]
    n_CO = r.n_eq[r.species.index('CO')]
    n_CO2 = r.n_eq[r.species.index('CO2')]

    conv = 1.0 - n_CH4 / 1.0
    check(f"CH4 conversion in [0.95, 1.0]: {conv:.4f}", 0.95 < conv < 1.0)

    h2_yield = n_H2 / 1.0
    check(f"H2 yield in [3.0, 3.5] per mol CH4: {h2_yield:.3f}",
          3.0 < h2_yield < 3.5)

    # CO + CO2 should equal initial moles of C (i.e., n_CH4_in - n_CH4_out)
    c_balance = n_CO + n_CO2 + n_CH4
    check(f"Carbon balance: {c_balance:.4f} = 1.0", abs(c_balance - 1.0) < 1e-6)


def test_multi_reaction_K_consistency_at_solution():
    """At equilibrium, K_r computed from y-mole fractions must match K_r(T)
    for ALL reactions simultaneously."""
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_K_consistency_at_solution")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    r = system.equilibrium_ideal_gas(
        T=1000.0, p=10e5, n_initial={'CH4': 1.0, 'H2O': 2.0})

    p_ref = 1e5
    # K_r from y
    Ks_check = np.array([np.prod((r.y_eq * r.p / p_ref) ** system.nu[i])
                          for i in range(system.R)])
    rel_err = np.max(np.abs(Ks_check - r.K_eq) / r.K_eq)
    check(f"K_r consistency: max rel err = {rel_err:.2e}", rel_err < 1e-6)


def test_multi_reaction_mass_balance():
    """Element balances must be exact at equilibrium (regardless of extents)."""
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_mass_balance")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    n0 = {'CH4': 1.0, 'H2O': 3.0}
    r = system.equilibrium_ideal_gas(T=1100.0, p=1e5, n_initial=n0)

    atoms = {'CH4': {'C': 1, 'H': 4}, 'H2O': {'H': 2, 'O': 1},
             'CO':  {'C': 1, 'O': 1}, 'H2': {'H': 2},
             'CO2': {'C': 1, 'O': 2}}
    bad_diffs = []
    for elem in ['C', 'H', 'O']:
        ein = sum(n * atoms[name].get(elem, 0) for name, n in n0.items())
        eout = sum(r.n_eq[r.species.index(name)] * atoms[name].get(elem, 0)
                    for name in r.species)
        if abs(ein - eout) > 1e-9:
            bad_diffs.append((elem, ein, eout))
    check(f"All element balances exact: {bad_diffs}", not bad_diffs)


def test_multi_reaction_le_chatelier_pressure():
    """For SMR at fixed T, INCREASING pressure should DECREASE H2 yield
    (reaction 1 has dn=+2: high p shifts back toward reactants)."""
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_le_chatelier_pressure")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    r_low = system.equilibrium_ideal_gas(T=900.0, p=1e5,
                                           n_initial={'CH4': 1.0, 'H2O': 3.0})
    r_high = system.equilibrium_ideal_gas(T=900.0, p=30e5,
                                            n_initial={'CH4': 1.0, 'H2O': 3.0})
    n_H2_low = r_low.n_eq[r_low.species.index('H2')]
    n_H2_high = r_high.n_eq[r_high.species.index('H2')]
    check(f"High-p H2 yield < low-p: {n_H2_high:.3f} < {n_H2_low:.3f}",
          n_H2_high < n_H2_low)


def test_multi_reaction_inert_dilution_helps_reforming():
    """For SMR, ADDING inerts at fixed total p should INCREASE H2 yield
    (the reforming step has dn>0; inerts dilute reactants, but Le Chatelier
    on a dn>0 reaction at fixed p favors products with more dilution)."""
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_inert_dilution_helps_reforming")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    r0 = system.equilibrium_ideal_gas(T=900.0, p=10e5,
                                        n_initial={'CH4': 1.0, 'H2O': 3.0},
                                        n_inert=0.0)
    r1 = system.equilibrium_ideal_gas(T=900.0, p=10e5,
                                        n_initial={'CH4': 1.0, 'H2O': 3.0},
                                        n_inert=10.0)
    n_H2_clean = r0.n_eq[r0.species.index('H2')]
    n_H2_dilute = r1.n_eq[r1.species.index('H2')]
    check(f"Inerts increase H2 yield: {n_H2_dilute:.3f} > {n_H2_clean:.3f}",
          n_H2_dilute > n_H2_clean)


def test_multi_reaction_disjoint_reactions():
    """Two reactions with no shared species should also work."""
    from stateprop.reaction import MultiReaction
    section("test_multi_reaction_disjoint_reactions")
    # Two independent decompositions:
    #   2 NO  = N2 + O2     (R1)
    #   2 NO2 = 2 NO + O2   (R2)
    # These do share NO and O2 actually, so let me use truly disjoint:
    #   N2 + 3 H2 = 2 NH3   (R1)
    #   CO + H2O = CO2 + H2 (R2; H2 shared so not truly disjoint either)
    # Just verify it converges with shared species:
    system = MultiReaction.from_specs([
        {'reactants': {'N2': 1, 'H2': 3}, 'products': {'NH3': 2}},
        {'reactants': {'CO': 1, 'H2O': 1}, 'products': {'CO2': 1, 'H2': 1}},
    ])
    r = system.equilibrium_ideal_gas(
        T=700.0, p=50e5,
        n_initial={'N2': 1.0, 'H2': 3.0, 'CO': 1.0, 'H2O': 1.0})
    check(f"Two coupled reactions converge: {r.converged}", r.converged)


# ------------------------------------------------------------------
# Real-gas K_eq corrections (v0.9.63)
# ------------------------------------------------------------------

def _build_methanol_pr_mixture():
    """PR mixture for methanol synthesis CO/H2/CH3OH (in that order)."""
    from stateprop.cubic import PR, CubicMixture
    co    = PR(T_c=132.85, p_c=3.494e6, acentric_factor=0.045)
    h2    = PR(T_c=33.145, p_c=1.296e6, acentric_factor=-0.219)
    ch3oh = PR(T_c=512.60, p_c=8.084e6, acentric_factor=0.5625)
    return CubicMixture([co, h2, ch3oh], composition=[0.33, 0.66, 0.01])


def test_real_gas_low_pressure_matches_ideal():
    """At p = 1 bar the real-gas correction should agree with ideal-gas
    to within ~0.1% (phi very close to 1)."""
    from stateprop.reaction import Reaction
    section("test_real_gas_low_pressure_matches_ideal")
    pr_mix = _build_methanol_pr_mixture()
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                products={'CH3OH': 1})
    r_id = rxn.equilibrium_extent_ideal_gas(T=500.0, p=1e5,
                                               n_initial=[1.0, 2.0, 0.0])
    r_rg = rxn.equilibrium_extent_real_gas(T=500.0, p=1e5,
                                              n_initial=[1.0, 2.0, 0.0],
                                              eos=pr_mix)
    check(f"Both converged: ideal={r_id.converged}, real={r_rg.converged}",
          r_id.converged and r_rg.converged)
    diff = abs(r_id.xi - r_rg.xi)
    check(f"Real-gas xi matches ideal at 1 bar (diff = {diff:.6f})",
          diff < 1e-3)


def test_real_gas_methanol_synthesis_high_p():
    """Methanol synthesis at 100 bar: real-gas (PR) should give HIGHER
    conversion than ideal-gas (CH3OH has low phi at high p)."""
    from stateprop.reaction import Reaction
    section("test_real_gas_methanol_synthesis_high_p")
    pr_mix = _build_methanol_pr_mixture()
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                products={'CH3OH': 1})
    r_id = rxn.equilibrium_extent_ideal_gas(T=500.0, p=100e5,
                                               n_initial=[1.0, 2.0, 0.0])
    r_rg = rxn.equilibrium_extent_real_gas(T=500.0, p=100e5,
                                              n_initial=[1.0, 2.0, 0.0],
                                              eos=pr_mix)
    conv_id = 1 - r_id.n_eq[0] / 1.0
    conv_rg = 1 - r_rg.n_eq[0] / 1.0
    check(f"Real-gas conversion higher than ideal at 100 bar: "
          f"{conv_rg:.3f} > {conv_id:.3f}", conv_rg > conv_id + 0.05)


def test_real_gas_K_y_satisfies_equilibrium_with_phi():
    """At the real-gas solution, K_eq(T) = Prod (y_i phi_i p/p_ref)^nu_i
    must hold to high precision."""
    from stateprop.reaction import Reaction
    section("test_real_gas_K_y_satisfies_equilibrium_with_phi")
    pr_mix = _build_methanol_pr_mixture()
    rxn = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                products={'CH3OH': 1})
    r = rxn.equilibrium_extent_real_gas(T=500.0, p=50e5,
                                           n_initial=[1.0, 2.0, 0.0],
                                           eos=pr_mix)
    rho = pr_mix.density_from_pressure(r.p, r.T, r.y_eq, phase_hint='vapor')
    ln_phi = pr_mix.ln_phi(rho, r.T, r.y_eq)
    p_ref = 1e5
    nu = rxn.nu
    K_check = np.prod((r.y_eq * np.exp(ln_phi) * r.p / p_ref) ** nu)
    rel_err = abs(K_check - r.K_eq) / r.K_eq
    check(f"K_eq from y*phi*p matches: {K_check:.4e} vs {r.K_eq:.4e} "
          f"(rel err = {rel_err:.2e})", rel_err < 1e-5)


def test_real_gas_multi_reaction_smr():
    """Steam methane reforming at moderate p: real-gas correction should
    be small (CH4/H2/CO/H2O are near-ideal) but solver must still work."""
    from stateprop.reaction import MultiReaction
    from stateprop.cubic import PR, CubicMixture
    section("test_real_gas_multi_reaction_smr")
    system = MultiReaction.from_specs([
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    # Build PR EOS in the same species ordering as the MultiReaction
    pr_params = {
        'CH4': PR(T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142),
        'H2O': PR(T_c=647.096, p_c=22.064e6, acentric_factor=0.3443),
        'CO':  PR(T_c=132.85,  p_c=3.494e6,  acentric_factor=0.045),
        'H2':  PR(T_c=33.145,  p_c=1.296e6,  acentric_factor=-0.219),
        'CO2': PR(T_c=304.13,  p_c=7.3773e6, acentric_factor=0.22394),
    }
    components = [pr_params[name] for name in system.species_names]
    pr_mix = CubicMixture(components, composition=[0.2]*5)

    r = system.equilibrium_real_gas(
        T=1000.0, p=30e5,
        n_initial={'CH4': 1.0, 'H2O': 3.0},
        eos=pr_mix)
    check(f"Real-gas SMR converged: {r.converged}", r.converged)

    # Verify K_eq consistency using y * phi * p/p_ref
    rho = pr_mix.density_from_pressure(r.p, r.T, r.y_eq, phase_hint='vapor')
    ln_phi = pr_mix.ln_phi(rho, r.T, r.y_eq)
    p_ref = 1e5
    Ks_check = np.array([
        np.prod((r.y_eq * np.exp(ln_phi) * r.p / p_ref) ** system.nu[i])
        for i in range(system.R)])
    rel_err = np.max(np.abs(Ks_check - r.K_eq) / r.K_eq)
    check(f"Real-gas K_r consistency: max rel err = {rel_err:.2e}",
          rel_err < 1e-4)


# ------------------------------------------------------------------
# Liquid-phase reactions with activity coefficients (v0.9.64)
# ------------------------------------------------------------------

class _IdealMixModel:
    """Trivial activity model: gamma_i = 1 for all species."""
    def __init__(self, N):
        self.N = N
    def gammas(self, T, x):
        return np.ones(len(x))


def test_liquid_phase_reaction_construction():
    """LiquidPhaseReaction constructs with K_eq_298 + dH_rxn or callable."""
    from stateprop.reaction import LiquidPhaseReaction
    section("test_liquid_phase_reaction_construction")
    rxn = LiquidPhaseReaction(['A', 'B', 'C', 'D'], [-1, -1, +1, +1],
                                K_eq_298=4.0, dH_rxn=-2.3e3)
    check(f"K_eq(298) = 4.0: {rxn.K_eq(298.15):.4f}",
          abs(rxn.K_eq(298.15) - 4.0) < 1e-6)
    # Vant Hoff: K should decrease above 298 if dH < 0
    check(f"Endothermic check: K(363) < K(298) for dH<0",
          rxn.K_eq(363.15) < rxn.K_eq(298.15))
    # Custom callable
    rxn2 = LiquidPhaseReaction(['A', 'B'], [-1, +1],
                                  ln_K_eq_T=lambda T: 1.0 - 1000.0/T)
    check(f"Custom ln_K_eq_T callable used",
          abs(rxn2.K_eq(500.0) - math.exp(1.0 - 1000.0/500.0)) < 1e-12)


def test_liquid_phase_ideal_solution_matches_analytic():
    """For equimolar A + B = C + D with K=4 and ideal solution, the
    analytic solution is xi = sqrt(K)/(1+sqrt(K))."""
    from stateprop.reaction import LiquidPhaseReaction
    section("test_liquid_phase_ideal_solution_matches_analytic")
    rxn = LiquidPhaseReaction(['A', 'B', 'C', 'D'], [-1, -1, +1, +1],
                                K_eq_298=4.0, dH_rxn=0.0)
    r = rxn.equilibrium_extent(T=298.15, n_initial=[1.0, 1.0, 0.0, 0.0],
                                  activity_model=_IdealMixModel(4))
    xi_analytic = 2.0 / (1.0 + 2.0)   # sqrt(4)/(1+sqrt(4)) = 2/3
    check(f"Converged: {r.converged}", r.converged)
    check(f"xi matches analytic 2/3: {r.xi:.6f} vs {xi_analytic:.6f}",
          abs(r.xi - xi_analytic) < 1e-6)
    check(f"K_a at solution = K_eq = 4.0: {r.K_a:.4f}",
          abs(r.K_a - 4.0) < 1e-6)


def test_liquid_phase_unifac_esterification():
    """Acetic acid + ethanol = ethyl acetate + water with UNIFAC."""
    from stateprop.reaction import LiquidPhaseReaction
    from stateprop.activity.compounds import make_unifac
    section("test_liquid_phase_unifac_esterification")
    rxn = LiquidPhaseReaction(
        species_names=['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'],
        nu=[-1, -1, +1, +1], K_eq_298=4.0, dH_rxn=-2.3e3,
    )
    uf = make_unifac(['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'])
    r = rxn.equilibrium_extent(T=333.15,
                                  n_initial=[1.0, 1.0, 0.0, 0.0],
                                  activity_model=uf)
    check(f"Converged: {r.converged}", r.converged)
    # UNIFAC predicts conversion typically lower than ideal due to
    # high γ for water and ester
    check(f"Conversion in (0.3, 0.7): {r.xi:.3f}", 0.3 < r.xi < 0.7)
    # K_a at the solution must equal K_eq(T)
    check(f"K_a equals K_eq: {r.K_a:.4f} vs {r.K_eq:.4f}",
          abs(r.K_a - r.K_eq) / r.K_eq < 1e-5)


def test_liquid_phase_le_chatelier_excess_reactant():
    """Excess of one reactant should INCREASE conversion of the other."""
    from stateprop.reaction import LiquidPhaseReaction
    from stateprop.activity.compounds import make_unifac
    section("test_liquid_phase_le_chatelier_excess_reactant")
    rxn = LiquidPhaseReaction(
        species_names=['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'],
        nu=[-1, -1, +1, +1], K_eq_298=4.0, dH_rxn=-2.3e3,
    )
    uf = make_unifac(['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'])
    r_eq = rxn.equilibrium_extent(T=333.15, n_initial=[1.0, 1.0, 0.0, 0.0],
                                     activity_model=uf)
    r_ex = rxn.equilibrium_extent(T=333.15, n_initial=[1.0, 3.0, 0.0, 0.0],
                                     activity_model=uf)
    conv_eq = (1 - r_eq.n_eq[0])
    conv_ex = (1 - r_ex.n_eq[0])
    check(f"Excess EtOH increases AcOH conversion: {conv_ex:.3f} > {conv_eq:.3f}",
          conv_ex > conv_eq)


def test_multi_liquid_phase_construction():
    """MultiLiquidPhaseReaction merges species, builds stoich matrix."""
    from stateprop.reaction import (LiquidPhaseReaction,
                                       MultiLiquidPhaseReaction)
    section("test_multi_liquid_phase_construction")
    r1 = LiquidPhaseReaction(['acetic_acid', 'ethanol',
                                'ethyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=4.0)
    r2 = LiquidPhaseReaction(['acetic_acid', 'methanol',
                                'methyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=5.0)
    system = MultiLiquidPhaseReaction([r1, r2])
    check(f"R = 2: {system.R}", system.R == 2)
    check(f"N = 6 species (union): {system.N}", system.N == 6)
    # Both have AcOH and water shared; ethanol-EtOAc unique to R1, MeOH-MeOAc unique to R2
    check(f"acetic_acid in canonical species",
          'acetic_acid' in system.species_names)
    check(f"water in canonical species",
          'water' in system.species_names)


def test_multi_liquid_phase_competing_esterifications():
    """AcOH + EtOH and AcOH + MeOH simultaneously, with competing AcOH."""
    from stateprop.reaction import (LiquidPhaseReaction,
                                       MultiLiquidPhaseReaction)
    from stateprop.activity.compounds import make_unifac
    section("test_multi_liquid_phase_competing_esterifications")
    r1 = LiquidPhaseReaction(['acetic_acid', 'ethanol',
                                'ethyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=4.0, dH_rxn=-2.3e3)
    r2 = LiquidPhaseReaction(['acetic_acid', 'methanol',
                                'methyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=5.0, dH_rxn=-3.5e3)
    system = MultiLiquidPhaseReaction([r1, r2])
    uf = make_unifac(list(system.species_names))
    r = system.equilibrium(T=333.15,
                              n_initial={'acetic_acid': 2.0,
                                         'ethanol': 1.0,
                                         'methanol': 1.0},
                              activity_model=uf)
    check(f"Converged: {r.converged}", r.converged)
    # Mass balance: water produced = total esters produced = xi_1 + xi_2
    n_water = r.n_eq[r.species.index('water')]
    n_etoac = r.n_eq[r.species.index('ethyl_acetate')]
    n_meoac = r.n_eq[r.species.index('methyl_acetate')]
    check(f"Water = EtOAc + MeOAc: {n_water:.4f} = {n_etoac + n_meoac:.4f}",
          abs(n_water - (n_etoac + n_meoac)) < 1e-6)
    # AcOH consumed = total esters formed
    n_acoh_in = 2.0
    n_acoh_eq = r.n_eq[r.species.index('acetic_acid')]
    consumed = n_acoh_in - n_acoh_eq
    check(f"AcOH consumed = total esters: {consumed:.4f} ~ "
          f"{n_etoac + n_meoac:.4f}",
          abs(consumed - (n_etoac + n_meoac)) < 1e-6)


def test_multi_liquid_phase_K_a_consistency():
    """At equilibrium, K_a,r = Prod(gamma_i * x_i)^nu[r,i] = K_eq,r(T)
    must hold for each reaction simultaneously."""
    from stateprop.reaction import (LiquidPhaseReaction,
                                       MultiLiquidPhaseReaction)
    from stateprop.activity.compounds import make_unifac
    section("test_multi_liquid_phase_K_a_consistency")
    r1 = LiquidPhaseReaction(['acetic_acid', 'ethanol',
                                'ethyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=4.0, dH_rxn=-2.3e3)
    r2 = LiquidPhaseReaction(['acetic_acid', 'methanol',
                                'methyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=5.0, dH_rxn=-3.5e3)
    system = MultiLiquidPhaseReaction([r1, r2])
    uf = make_unifac(list(system.species_names))
    r = system.equilibrium(T=333.15,
                              n_initial={'acetic_acid': 2.0,
                                         'ethanol': 1.0,
                                         'methanol': 1.0},
                              activity_model=uf)
    Ks_check = np.array([np.prod((r.gamma_eq * r.x_eq) ** system.nu[i])
                          for i in range(system.R)])
    rel_err = np.max(np.abs(Ks_check - r.K_eq) / r.K_eq)
    check(f"K_a,r = K_eq,r consistency: max rel err = {rel_err:.2e}",
          rel_err < 1e-5)


# ------------------------------------------------------------------
# Reactive flash (v0.9.65)
# ------------------------------------------------------------------

def _esterification_setup():
    """Common setup for reactive-flash tests: Antoine + UNIFAC + reaction."""
    from stateprop.reaction import LiquidPhaseReaction
    from stateprop.activity.compounds import make_unifac

    def antoine(A, B, C):
        def psat(T):
            return 133.322 * 10**(A - B/(T + C))
        return psat
    psat_acoh   = antoine(7.55716, 1642.540, -39.764)
    psat_etoh   = antoine(8.20417, 1642.890, -42.85)
    psat_etoac  = antoine(7.10179, 1244.95,  -55.84)
    psat_water  = antoine(8.07131, 1730.630, -39.574)
    species = ['acetic_acid', 'ethanol', 'ethyl_acetate', 'water']
    psats = [psat_acoh, psat_etoh, psat_etoac, psat_water]
    uf = make_unifac(species)
    rxn = LiquidPhaseReaction(species_names=species, nu=[-1, -1, +1, +1],
                                K_eq_298=4.0, dH_rxn=-2.3e3)
    return species, psats, uf, rxn


def test_reactive_flash_subcooled_matches_no_vle():
    """At T well below bubble point, reactive flash should give the
    SAME extent as the pure-liquid (no-VLE) equilibrium."""
    from stateprop.reaction import reactive_flash_TP
    section("test_reactive_flash_subcooled_matches_no_vle")
    species, psats, uf, rxn = _esterification_setup()

    r_flash = reactive_flash_TP(
        T=320.0, p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
        activity_model=uf, psat_funcs=psats, reactions=[rxn],
        species_names=species, tol=1e-7, maxiter=80)
    r_liq = rxn.equilibrium_extent(T=320.0, n_initial=[1.0, 1.0, 0.0, 0.0],
                                      activity_model=uf)
    check(f"Flash converged: {r_flash.converged}", r_flash.converged)
    check(f"V/(V+L) ~ 0 (subcooled)",
          r_flash.V / (r_flash.V + r_flash.L) < 1e-3)
    check(f"xi matches no-VLE: {r_flash.xi[0]:.4f} vs {r_liq.xi:.4f}",
          abs(r_flash.xi[0] - r_liq.xi) < 1e-4)


def test_reactive_flash_boiling_higher_conversion():
    """At boiling conditions, reactive flash should give HIGHER extent
    than pure-liquid equilibrium (Le Chatelier on vapor product removal)."""
    from stateprop.reaction import reactive_flash_TP
    section("test_reactive_flash_boiling_higher_conversion")
    species, psats, uf, rxn = _esterification_setup()
    r_flash = reactive_flash_TP(
        T=355.0, p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
        activity_model=uf, psat_funcs=psats, reactions=[rxn],
        species_names=species, tol=1e-7, maxiter=80)
    r_liq = rxn.equilibrium_extent(T=355.0, n_initial=[1.0, 1.0, 0.0, 0.0],
                                      activity_model=uf)
    check(f"Flash converged: {r_flash.converged}", r_flash.converged)
    check(f"V/(V+L) > 0.1 (two-phase): {r_flash.V/(r_flash.V+r_flash.L):.3f}",
          r_flash.V / (r_flash.V + r_flash.L) > 0.1)
    check(f"xi(flash) > xi(no-VLE): {r_flash.xi[0]:.4f} > {r_liq.xi:.4f}",
          r_flash.xi[0] > r_liq.xi + 0.05)


def test_reactive_flash_K_a_matches_K_eq():
    """At convergence, K_a from x and γ must match K_eq(T) to tol."""
    from stateprop.reaction import reactive_flash_TP
    section("test_reactive_flash_K_a_matches_K_eq")
    species, psats, uf, rxn = _esterification_setup()
    r_flash = reactive_flash_TP(
        T=355.0, p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
        activity_model=uf, psat_funcs=psats, reactions=[rxn],
        species_names=species, tol=1e-7, maxiter=80)
    rel_err = abs(r_flash.K_a[0] - r_flash.K_eq[0]) / r_flash.K_eq[0]
    check(f"K_a/K_eq consistency: rel_err = {rel_err:.2e}",
          rel_err < 1e-5)


def test_reactive_flash_mass_balance():
    """Total atoms conserved across feed and (V + L)."""
    from stateprop.reaction import reactive_flash_TP
    section("test_reactive_flash_mass_balance")
    species, psats, uf, rxn = _esterification_setup()
    F = 2.0
    z = np.array([0.5, 0.5, 0.0, 0.0])
    r = reactive_flash_TP(
        T=355.0, p=101325.0, F=F, z=z,
        activity_model=uf, psat_funcs=psats, reactions=[rxn],
        species_names=species, tol=1e-7, maxiter=80)
    # Element atoms: AcOH=C2H4O2, EtOH=C2H6O, EtOAc=C4H8O2, H2O=H2O
    atoms = np.array([
        [2, 4, 2],  # AcOH: C, H, O
        [2, 6, 1],  # EtOH
        [4, 8, 2],  # EtOAc
        [0, 2, 1],  # H2O
    ])
    n_in = z * F
    n_out = r.V * r.y + r.L * r.x
    elements_in  = atoms.T @ n_in
    elements_out = atoms.T @ n_out
    diff = float(np.max(np.abs(elements_in - elements_out)))
    check(f"Element balance (C, H, O) max diff = {diff:.2e}",
          diff < 1e-6)


def test_reactive_flash_no_reactions_pure_vle():
    """With reactions=[], reactive_flash_TP should reduce to pure VLE flash."""
    from stateprop.reaction import reactive_flash_TP
    section("test_reactive_flash_no_reactions_pure_vle")
    species, psats, uf, rxn = _esterification_setup()
    r = reactive_flash_TP(
        T=355.0, p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
        activity_model=uf, psat_funcs=psats, reactions=[],
        species_names=species, tol=1e-7, maxiter=80)
    check(f"Pure-VLE flash converged: {r.converged}", r.converged)
    check(f"xi has length 0: {r.xi.size}", r.xi.size == 0)
    # V + L should equal F (no reactions, no mole changes)
    check(f"V + L = F: {r.V + r.L:.4f} = {2.0}",
          abs(r.V + r.L - 2.0) < 1e-9)


# ------------------------------------------------------------------
# Reactive distillation column tests (v0.9.65)
# ------------------------------------------------------------------

def _column_setup():
    """Reusable setup: AcOH + EtOH = EtOAc + H2O at 1 atm."""
    from stateprop.reaction import LiquidPhaseReaction
    from stateprop.activity.compounds import make_unifac

    def antoine(A, B, C):
        return lambda T: 133.322 * 10.0**(A - B/(T - C))
    psats = [antoine(7.55716, 1642.540, 39.764),  # AcOH
             antoine(8.20417, 1642.890, 42.85),    # EtOH
             antoine(7.10179, 1244.95,  55.84),    # EtOAc
             antoine(8.07131, 1730.630, 39.574)]   # H2O
    species = ['acetic_acid', 'ethanol', 'ethyl_acetate', 'water']
    rxn = LiquidPhaseReaction(species, [-1, -1, +1, +1],
                                K_eq_298=4.0, dH_rxn=-2.3e3)
    uf = make_unifac(species)
    return species, rxn, uf, psats


def test_rd_column_construction():
    """Column constructs with valid args; result fields populated."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_construction")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=4, feed_stage=2, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3],
        max_outer_iter=80, tol=2e-3, damping=0.3)
    check(f"n_stages = 4: {r.n_stages}", r.n_stages == 4)
    check(f"feed_stage = 2: {r.feed_stage}", r.feed_stage == 2)
    check(f"reactive_stages = (2, 3): {r.reactive_stages}",
          r.reactive_stages == (2, 3))
    check(f"x has shape (4, 4): {r.x.shape}", r.x.shape == (4, 4))


def test_rd_column_non_reactive_mass_balance():
    """Column with no reactions = pure VLE: F z_i = D x_D,i + B x_B,i."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_non_reactive_mass_balance")
    species, _, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.0, 0.4, 0.3, 0.3], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[], reactive_stages=[],
        max_outer_iter=80, tol=1e-3, damping=0.3)
    check(f"Converged: {r.converged} ({r.iterations} iters)", r.converged)
    F_in = 1.0 * np.array([0.0, 0.4, 0.3, 0.3])
    mol_out = r.D * r.x_D + r.B * r.x_B
    err = float(np.max(np.abs(F_in - mol_out)))
    check(f"Per-species mass balance: max err = {err:.2e}", err < 1e-3)


def test_rd_column_no_reaction_zero_conversion():
    """Column with reactions=[] should report ~0 conversion."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_no_reaction_zero_conversion")
    species, _, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[], reactive_stages=[],
        max_outer_iter=80, tol=1e-3, damping=0.3)
    conv = r.conversion('acetic_acid')
    check(f"AcOH conv ~ 0 (no reaction): {conv:.4%}", abs(conv) < 5e-3)


def test_rd_column_element_balance():
    """Reactive column conserves atoms (C, H, O)."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_element_balance")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3, 4, 5],
        max_outer_iter=100, tol=2e-3, damping=0.3)
    check(f"Converged: {r.converged}", r.converged)
    # AcOH=C2H4O2, EtOH=C2H6O1, EtOAc=C4H8O2, H2O=H2O
    elements = {'C': np.array([2, 2, 4, 0]),
                  'H': np.array([4, 6, 8, 2]),
                  'O': np.array([2, 1, 2, 1])}
    F_in = 1.0 * np.array([0.5, 0.5, 0., 0.])
    mol_out = r.D * r.x_D + r.B * r.x_B
    max_rel = 0.0
    for sym, w in elements.items():
        in_ = float((F_in * w).sum())
        out_ = float((mol_out * w).sum())
        max_rel = max(max_rel, abs(in_ - out_) / max(in_, 1e-12))
    check(f"Max element-balance rel err: {max_rel:.2e}", max_rel < 5e-3)


def test_rd_column_temperature_profile():
    """Reboiler must be hotter than top stage (basic physical sanity)."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_temperature_profile")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3, 4, 5],
        max_outer_iter=80, tol=2e-3, damping=0.3)
    check(f"Converged: {r.converged}", r.converged)
    check(f"T_top = {r.T[0]:.1f} K < T_reb = {r.T[-1]:.1f} K",
          r.T[-1] > r.T[0])


def test_rd_column_K_a_at_reactive_stages():
    """At converged solution, every reactive stage must satisfy
    K_a = K_eq(T_j) within solver tolerance."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_K_a_at_reactive_stages")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3, 4, 5],
        max_outer_iter=100, tol=2e-3, damping=0.3)
    check(f"Converged: {r.converged}", r.converged)
    nu = np.array([-1., -1., +1., +1.])
    max_dev = 0.0
    for stage in r.reactive_stages:
        j = stage - 1
        T_j = r.T[j]
        x_j = r.x[j]
        gammas = np.asarray(uf.gammas(T_j, x_j))
        K_a = float(np.prod((gammas * x_j) ** nu))
        K_target = rxn.K_eq(T_j)
        max_dev = max(max_dev, abs(K_a - K_target) / K_target)
    check(f"Max |K_a - K_eq|/K_eq across reactive stages: {max_dev:.2e}",
          max_dev < 5e-2)


def test_rd_column_NS_quadratic_convergence():
    """Naphtali-Sandholm should converge in ~10-20 Newton iterations
    with the line-search achieving quadratic convergence at the end."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_NS_quadratic_convergence")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3, 4, 5],
        method="naphtali_sandholm",
        max_newton_iter=30, newton_tol=1e-7)
    check(f"N-S converged: {r.converged} ({r.iterations} iters)",
          r.converged and r.iterations < 25)


def test_rd_column_NS_high_precision_K_a():
    """N-S enforces K_a = K_eq to near-machine precision (~1e-8)."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_NS_high_precision_K_a")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3, 4, 5],
        method="naphtali_sandholm",
        max_newton_iter=30, newton_tol=1e-7)
    check(f"N-S converged: {r.converged}", r.converged)
    nu = np.array([-1., -1., +1., +1.])
    max_dev = 0.0
    for stage in r.reactive_stages:
        j = stage - 1
        gammas = np.asarray(uf.gammas(r.T[j], r.x[j]))
        K_a = float(np.prod((gammas * r.x[j]) ** nu))
        K_target = rxn.K_eq(r.T[j])
        max_dev = max(max_dev, abs(K_a - K_target) / K_target)
    # N-S should beat WH by ~3 orders of magnitude on K_a closure
    check(f"N-S K_a closure: {max_dev:.2e}", max_dev < 1e-6)


def test_rd_column_NS_machine_precision_atom_balance():
    """N-S element balance should close to ~1e-10 (machine precision)."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_NS_machine_precision_atom_balance")
    species, rxn, uf, psats = _column_setup()
    r = reactive_distillation_column(
        n_stages=6, feed_stage=3, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[2, 3, 4, 5],
        method="naphtali_sandholm",
        max_newton_iter=30, newton_tol=1e-7)
    check(f"N-S converged: {r.converged}", r.converged)
    elements = {'C': np.array([2, 2, 4, 0]),
                  'H': np.array([4, 6, 8, 2]),
                  'O': np.array([2, 1, 2, 1])}
    F_in = 1.0 * np.array([0.5, 0.5, 0., 0.])
    mol_out = r.D * r.x_D + r.B * r.x_B
    max_rel = 0.0
    for sym, w in elements.items():
        in_ = float((F_in * w).sum())
        out_ = float((mol_out * w).sum())
        max_rel = max(max_rel, abs(in_ - out_) / max(in_, 1e-12))
    check(f"N-S atom balance: {max_rel:.2e}", max_rel < 1e-8)


def test_rd_column_WH_NS_agree():
    """Wang-Henke and Naphtali-Sandholm should agree on the same column
    state to within their respective tolerances (cross-validation)."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_WH_NS_agree")
    species, rxn, uf, psats = _column_setup()
    cfg = dict(n_stages=6, feed_stage=3, feed_F=1.0,
                feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
                reflux_ratio=2.0, distillate_rate=0.5, pressure=101325.,
                species_names=species, activity_model=uf, psat_funcs=psats,
                reactions=[rxn], reactive_stages=[2, 3, 4, 5])
    r_wh = reactive_distillation_column(**cfg, method="wang_henke",
                                         max_outer_iter=120, tol=2e-3,
                                         damping=0.3)
    r_ns = reactive_distillation_column(**cfg, method="naphtali_sandholm",
                                         max_newton_iter=30, newton_tol=1e-7)
    check(f"WH conv: {r_wh.converged}, NS conv: {r_ns.converged}",
          r_wh.converged and r_ns.converged)
    diff_conv = abs(r_wh.conversion('acetic_acid')
                    - r_ns.conversion('acetic_acid'))
    check(f"AcOH conv agreement: WH={r_wh.conversion('acetic_acid'):.4%} "
          f"NS={r_ns.conversion('acetic_acid'):.4%} diff={diff_conv:.2e}",
          diff_conv < 1e-3)
    diff_T = float(np.max(np.abs(r_wh.T - r_ns.T)))
    check(f"T-profile agreement: max diff = {diff_T:.2e} K", diff_T < 0.5)


# v0.9.67 — Energy-balance reactive distillation
def _eb_enthalpy_funcs():
    """Approximate constant-Cp + h_vap_ref enthalpy model for the
    AcOH/EtOH/EtOAc/H2O esterification system.  Reference state:
    saturated liquid at T_REF=298.15 K with h_L^*=0."""
    T_REF = 298.15
    Cp_L = np.array([124.0, 113.0, 170.0, 75.3])
    Cp_V = np.array([67.0, 73.0, 113.0, 34.0])
    h_vap_298 = np.array([23700.0, 42000.0, 35000.0, 44000.0])
    h_L_funcs = [(lambda T, i=i: Cp_L[i] * (T - T_REF)) for i in range(4)]
    h_V_funcs = [(lambda T, i=i: h_vap_298[i] + Cp_V[i] * (T - T_REF))
                 for i in range(4)]
    return h_V_funcs, h_L_funcs


def test_rd_column_energy_balance_converges():
    """Energy-balance N-S converges and returns physically sensible
    profiles (T_reb > T_top, V_top fixed at (R+1)D, L_bottom = B)."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_energy_balance_converges")
    species, rxn, uf, psats = _column_setup()
    h_V_funcs, h_L_funcs = _eb_enthalpy_funcs()
    r = reactive_distillation_column(
        n_stages=8, feed_stage=4, feed_F=1.0,
        feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[3, 4, 5, 6],
        method="naphtali_sandholm",
        energy_balance=True,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        max_newton_iter=40, newton_tol=1e-7)
    check(f"Converged: {r.converged} in {r.iterations} iters", r.converged)
    check(f"T_reb {r.T[-1]:.1f} > T_top {r.T[0]:.1f}", r.T[-1] > r.T[0])
    V_top_expected = (2.0 + 1.0) * 0.5
    check(f"V[0] = {r.V[0]:.4f} = (R+1)*D = {V_top_expected:.4f}",
          abs(r.V[0] - V_top_expected) < 1e-9)
    B_expected = 1.0 - 0.5      # F - D for atom-conservative reaction
    check(f"L[-1] = {r.L[-1]:.4f} = B = {B_expected:.4f}",
          abs(r.L[-1] - B_expected) < 1e-6)


def test_rd_column_energy_balance_per_stage_closure():
    """Per-stage H_j residual must close to << H_scale at convergence."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_energy_balance_per_stage_closure")
    species, rxn, uf, psats = _column_setup()
    h_V_funcs, h_L_funcs = _eb_enthalpy_funcs()
    feed_T = 350.0
    F = 1.0
    feed_z = np.array([0.5, 0.5, 0., 0.])
    r = reactive_distillation_column(
        n_stages=8, feed_stage=4, feed_F=F, feed_z=feed_z, feed_T=feed_T,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325., species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[3, 4, 5, 6],
        method="naphtali_sandholm", energy_balance=True,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        max_newton_iter=40, newton_tol=1e-8)
    check(f"Converged: {r.converged}", r.converged)
    # Recompute per-stage h_L and h_V matching the solver convention
    h_L_vals = np.array([sum(r.x[j, i] * h_L_funcs[i](r.T[j])
                             for i in range(4)) for j in range(r.n_stages)])
    h_V_vals = np.array([sum(r.y[j, i] * h_V_funcs[i](r.T[j])
                             for i in range(4)) for j in range(r.n_stages)])
    h_F = float((feed_z * np.array([h_L_funcs[i](feed_T)
                                     for i in range(4)])).sum())
    H_scale = max(F, 1.0) * 1e4
    max_scaled = 0.0
    for j in range(1, r.n_stages - 1):
        in_h = r.L[j - 1] * h_L_vals[j - 1] + r.V[j + 1] * h_V_vals[j + 1]
        if (j + 1) == 4:   # feed stage
            in_h += F * h_F
        out_h = r.L[j] * h_L_vals[j] + r.V[j] * h_V_vals[j]
        rxn_h = (-rxn.dH_rxn) * float(r.xi[j].sum()) \
                if (j + 1) in {3, 4, 5, 6} else 0.0
        H_resid_scaled = (in_h + rxn_h - out_h) / H_scale
        max_scaled = max(max_scaled, abs(H_resid_scaled))
    check(f"Per-stage max |H_scaled| = {max_scaled:.2e}", max_scaled < 1e-6)


def test_rd_column_energy_balance_breaks_CMO():
    """With non-zero dH_rxn and a feed below column T, energy balance
    should produce flow profiles that DIFFER from CMO (by more than
    floating-point noise) — otherwise the H equation isn't doing
    anything.  Threshold is loose because dH_rxn is small (-2.3 kJ/mol)
    in the standard esterification setup; the bigger effect comes from
    the subcooled-feed enthalpy deficit."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_energy_balance_breaks_CMO")
    species, rxn, uf, psats = _column_setup()
    h_V_funcs, h_L_funcs = _eb_enthalpy_funcs()
    cfg = dict(n_stages=8, feed_stage=4, feed_F=1.0,
               feed_z=[0.5, 0.5, 0., 0.], feed_T=320.,    # subcooled
               reflux_ratio=2.0, distillate_rate=0.5,
               pressure=101325., species_names=species,
               activity_model=uf, psat_funcs=psats,
               reactions=[rxn], reactive_stages=[3, 4, 5, 6])
    r_cmo = reactive_distillation_column(**cfg, method="naphtali_sandholm",
                                          max_newton_iter=30,
                                          newton_tol=1e-7)
    r_eb = reactive_distillation_column(**cfg, method="naphtali_sandholm",
                                         energy_balance=True,
                                         h_V_funcs=h_V_funcs,
                                         h_L_funcs=h_L_funcs,
                                         max_newton_iter=40,
                                         newton_tol=1e-7)
    check(f"Both converged: CMO={r_cmo.converged} EB={r_eb.converged}",
          r_cmo.converged and r_eb.converged)
    max_V_diff = float(np.max(np.abs(r_cmo.V - r_eb.V)))
    max_L_diff = float(np.max(np.abs(r_cmo.L[:-1] - r_eb.L[:-1])))
    rel_V = max_V_diff / max(float(r_cmo.V.max()), 1e-12)
    rel_L = max_L_diff / max(float(r_cmo.L.max()), 1e-12)
    check(f"V profile differs: max abs diff = {max_V_diff:.3f} "
          f"(rel {rel_V:.1%})", rel_V > 0.01)
    check(f"L profile differs: max abs diff = {max_L_diff:.3f} "
          f"(rel {rel_L:.1%})", rel_L > 0.01)


def test_rd_column_energy_balance_validation():
    """energy_balance=True without h_*_funcs raises a clean error;
    wang_henke + energy_balance also raises."""
    from stateprop.reaction import reactive_distillation_column
    section("test_rd_column_energy_balance_validation")
    species, rxn, uf, psats = _column_setup()
    cfg = dict(n_stages=4, feed_stage=2, feed_F=1.0,
               feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
               reflux_ratio=2.0, distillate_rate=0.5,
               pressure=101325., species_names=species,
               activity_model=uf, psat_funcs=psats,
               reactions=[rxn], reactive_stages=[2, 3])

    raised = False
    try:
        reactive_distillation_column(**cfg, method="naphtali_sandholm",
                                      energy_balance=True)
    except ValueError:
        raised = True
    check("energy_balance=True without h_*_funcs raises ValueError",
          raised)

    raised = False
    try:
        reactive_distillation_column(**cfg, method="wang_henke",
                                      energy_balance=True,
                                      max_outer_iter=10, tol=1e-3,
                                      damping=0.3)
    except ValueError:
        raised = True
    check("wang_henke + energy_balance=True raises ValueError", raised)


def test_gamma_phi_eos_low_p_matches_modified_raoult():
    """At low p (<= 1 bar) with a well-behaved binary, the γ-φ-EOS
    K-values must agree with modified Raoult to within ~3% — the
    residual deviation comes only from the cubic EOS predicting
    φ_V slightly less than 1."""
    section("test_gamma_phi_eos_low_p_matches_modified_raoult")
    from stateprop.activity.gamma_phi_eos import GammaPhiEOSFlash
    from stateprop.activity.compounds import make_unifac
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ["benzene", "toluene"]
    pure_eos = [PR(562.05, 4.895e6, 0.2110),
                PR(591.75, 4.108e6, 0.2640)]
    mix = CubicMixture(pure_eos)
    uf = make_unifac(species)
    def antoine(A, B, C):
        return lambda T: 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    psats = [antoine(6.90565, 1211.033, 220.79),
             antoine(6.95464, 1344.8,   219.482)]

    flash = GammaPhiEOSFlash(uf, psats, mix)
    res = flash.isothermal(T=400.0, p=1e5, z=[0.5, 0.5])
    psat_arr = np.array([f(400.0) for f in psats])
    gam = uf.gammas(400.0, res.x)
    K_raoult = gam * psat_arr / 1e5
    rel_diff = np.max(np.abs((res.K - K_raoult) / K_raoult))
    print(f"    K_γφ={res.K}, K_Raoult={K_raoult}, "
          f"max rel diff = {rel_diff:.3%}")
    check("γ-φ-EOS at 1 bar agrees with Raoult to within 3%",
          rel_diff < 0.03)


def test_gamma_phi_eos_high_p_diverges_from_raoult():
    """At p ~ 30 bar, modified Raoult becomes inadequate; γ-φ-EOS
    K-values must differ from Raoult by at least 50% for at least
    one species."""
    section("test_gamma_phi_eos_high_p_diverges_from_raoult")
    from stateprop.activity.gamma_phi_eos import GammaPhiEOSFlash
    from stateprop.activity.compounds import make_unifac
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ["benzene", "toluene"]
    pure_eos = [PR(562.05, 4.895e6, 0.2110),
                PR(591.75, 4.108e6, 0.2640)]
    mix = CubicMixture(pure_eos)
    uf = make_unifac(species)
    def antoine(A, B, C):
        return lambda T: 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    psats = [antoine(6.90565, 1211.033, 220.79),
             antoine(6.95464, 1344.8,   219.482)]

    flash = GammaPhiEOSFlash(uf, psats, mix)
    res = flash.isothermal(T=400.0, p=30e5, z=[0.5, 0.5])
    psat_arr = np.array([f(400.0) for f in psats])
    gam = uf.gammas(400.0, res.x)
    K_raoult = gam * psat_arr / 30e5
    max_rel = np.max(np.abs((res.K - K_raoult) / K_raoult))
    print(f"    K_γφ={res.K}, K_Raoult={K_raoult}, max rel diff = {max_rel:.2%}")
    check("γ-φ-EOS at 30 bar differs from Raoult by > 50%",
          max_rel > 0.5)


def test_reactive_flash_vapor_eos_dispatch():
    """reactive_flash_TP: vapor_eos=None preserves modified-Raoult
    behavior bit-identically; vapor_eos=mix routes through γ-φ-EOS."""
    section("test_reactive_flash_vapor_eos_dispatch")
    from stateprop.reaction.reactive_flash import reactive_flash_TP
    from stateprop.reaction.liquid_phase import LiquidPhaseReaction
    from stateprop.activity.compounds import make_unifac
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ["acetic_acid", "ethanol", "ethyl_acetate", "water"]
    data = {"acetic_acid":   (591.95,  5.786e6,  0.4665),
            "ethanol":       (514.0,   6.137e6,  0.6452),
            "ethyl_acetate": (523.2,   3.880e6,  0.3664),
            "water":         (647.13, 22.064e6,  0.3449)}
    pure_eos = [PR(*data[s]) for s in species]
    mix = CubicMixture(pure_eos)
    uf = make_unifac(species)
    def antoine(A, B, C):
        return lambda T: 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    psats = [antoine(7.55716, 1642.540, 233.386),
             antoine(8.20417, 1642.890, 230.300),
             antoine(7.10179, 1244.951, 217.881),
             antoine(8.07131, 1730.630, 233.426)]
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=[-1, -1, +1, +1], K_eq_298=4.0)

    # vapor_eos=None: must be identical to existing path
    base = dict(T=350.0, p=1e5, F=1.0, z=[0.4, 0.4, 0.1, 0.1],
                activity_model=uf, psat_funcs=psats,
                reactions=[rxn], species_names=species)
    r0 = reactive_flash_TP(**base)
    r0_explicit = reactive_flash_TP(**base, vapor_eos=None)
    diff = float(np.max(np.abs(r0.x - r0_explicit.x)))
    check("vapor_eos=None bit-identical", diff < 1e-15)

    # vapor_eos=mix: must converge and produce a valid result
    r1 = reactive_flash_TP(**base, vapor_eos=mix)
    print(f"    no-EOS xi={r0.xi[0]:.6f}, γ-φ xi={r1.xi[0]:.6f}")
    check("γ-φ-EOS reactive flash converges", r1.converged)
    check("γ-φ-EOS produces a valid xi",
          0 <= r1.xi[0] <= 1.0 and np.isfinite(r1.xi[0]))


def test_reactive_flash_vapor_eos_input_validation():
    """Mismatched pure_liquid_volumes / phi_sat_funcs lengths raise."""
    section("test_reactive_flash_vapor_eos_input_validation")
    from stateprop.reaction.reactive_flash import reactive_flash_TP
    from stateprop.reaction.liquid_phase import LiquidPhaseReaction
    from stateprop.activity.compounds import make_unifac
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ["benzene", "toluene"]
    pure_eos = [PR(562.05, 4.895e6, 0.2110),
                PR(591.75, 4.108e6, 0.2640)]
    mix = CubicMixture(pure_eos)
    uf = make_unifac(species)
    def antoine(A, B, C):
        return lambda T: 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    psats = [antoine(6.90565, 1211.033, 220.79),
             antoine(6.95464, 1344.8,   219.482)]
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=[-1, +1], K_eq_298=1.0)

    base = dict(T=400.0, p=1e5, F=1.0, z=[0.5, 0.5],
                activity_model=uf, psat_funcs=psats,
                reactions=[rxn], species_names=species,
                vapor_eos=mix)
    # Wrong length
    raised = False
    try:
        reactive_flash_TP(**base, pure_liquid_volumes=[1e-4, 2e-4, 3e-4])
    except ValueError:
        raised = True
    check("bad pure_liquid_volumes length raises", raised)
    raised = False
    try:
        reactive_flash_TP(**base, phi_sat_funcs=[lambda T: 1.0])
    except ValueError:
        raised = True
    check("bad phi_sat_funcs length raises", raised)


def test_gibbs_min_water_gas_shift_matches_K_eq():
    """Direct Gibbs min on WGS at 1000 K must give K_y matching the
    extent-of-reaction K_eq within ~1e-5 relative."""
    section("test_gibbs_min_water_gas_shift_matches_K_eq")
    from stateprop.reaction import (gibbs_minimize_from_thermo,
                                      get_species, Reaction)
    species = ['CO', 'H2O', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2,'O':1}, {'C':1,'O':2}, {'H':2}]
    n_init = [1.0, 1.0, 0.001, 0.001]
    res = gibbs_minimize_from_thermo(
        T=1000.0, p=1e5, species=sp_obj, formulas=formulas, n_init=n_init,
        phase='gas', tol=1e-12, maxiter=50)
    check("WGS Gibbs min converged", res.converged)
    K_y = (res.n[2] * res.n[3]) / (res.n[0] * res.n[1])
    rxn = Reaction.from_names({'CO':1, 'H2O':1}, {'CO2':1, 'H2':1})
    K_ref = rxn.K_eq(1000.0)
    rel = abs(K_y - K_ref) / K_ref
    print(f"    K_y={K_y:.6f}, K_eq_ref={K_ref:.6f}, rel.err={rel:.2e}")
    check("K_y from Gibbs min matches K_eq within 1e-5", rel < 1e-5)


def test_gibbs_min_smr_multi_reaction_no_reactions_specified():
    """Steam methane reforming + water-gas shift: 5 species, 2
    independent reactions, but Gibbs min only needs species and
    formulas. Both K_eqs must be satisfied at the converged
    composition."""
    section("test_gibbs_min_smr_multi_reaction_no_reactions_specified")
    from stateprop.reaction import (gibbs_minimize_from_thermo,
                                      get_species, Reaction)
    species = ['CH4', 'H2O', 'CO', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'H':4}, {'H':2,'O':1}, {'C':1,'O':1},
                {'C':1,'O':2}, {'H':2}]
    n_init = [1.0, 3.0, 0.01, 0.01, 0.01]
    res = gibbs_minimize_from_thermo(
        T=1000.0, p=1e5, species=sp_obj, formulas=formulas, n_init=n_init,
        phase='gas', tol=1e-12, maxiter=80)
    check("SMR Gibbs min converged", res.converged)
    n = dict(zip(species, res.n))
    N_tot = float(res.n.sum())
    y = {s: n[s] / N_tot for s in species}
    rxn_smr = Reaction.from_names({'CH4':1, 'H2O':1}, {'CO':1, 'H2':3})
    rxn_wgs = Reaction.from_names({'CO':1, 'H2O':1}, {'CO2':1, 'H2':1})
    K_smr_obs = (y['CO'] * y['H2']**3) / (y['CH4'] * y['H2O']) * (1e5/1e5)**2
    K_wgs_obs = (y['CO2'] * y['H2']) / (y['CO'] * y['H2O'])
    e_smr = abs(K_smr_obs - rxn_smr.K_eq(1000.0)) / rxn_smr.K_eq(1000.0)
    e_wgs = abs(K_wgs_obs - rxn_wgs.K_eq(1000.0)) / rxn_wgs.K_eq(1000.0)
    print(f"    SMR K_y={K_smr_obs:.4f}, ref={rxn_smr.K_eq(1000.0):.4f}, "
          f"err={e_smr:.2e}")
    print(f"    WGS K_y={K_wgs_obs:.4f}, ref={rxn_wgs.K_eq(1000.0):.4f}, "
          f"err={e_wgs:.2e}")
    check("SMR K_eq satisfied within 1e-5", e_smr < 1e-5)
    check("WGS K_eq satisfied within 1e-5", e_wgs < 1e-5)


def test_gibbs_min_atom_balance_machine_precision():
    """Atom balance must hold to ~1e-12 at convergence."""
    section("test_gibbs_min_atom_balance_machine_precision")
    from stateprop.reaction import gibbs_minimize_from_thermo, get_species
    species = ['CH4', 'H2O', 'CO', 'CO2', 'H2', 'O2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'H':4}, {'H':2,'O':1}, {'C':1,'O':1},
                {'C':1,'O':2}, {'H':2}, {'O':2}]
    n_init = [1.0, 2.0, 0.01, 0.01, 0.01, 0.001]
    res = gibbs_minimize_from_thermo(
        T=1500.0, p=1e5, species=sp_obj, formulas=formulas, n_init=n_init,
        phase='gas', tol=1e-12, maxiter=80)
    check("converged", res.converged)
    print(f"    atom balance residual: {res.atom_balance_residual:.2e}")
    check("atom balance < 1e-10", res.atom_balance_residual < 1e-10)


def test_gibbs_min_monotone_decrease():
    """G should be monotonically non-increasing through the iteration
    (line-search guarantees this)."""
    section("test_gibbs_min_monotone_decrease")
    from stateprop.reaction import gibbs_minimize_from_thermo, get_species
    species = ['CO', 'H2O', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2,'O':1}, {'C':1,'O':2}, {'H':2}]
    res = gibbs_minimize_from_thermo(
        T=800.0, p=1e5, species=sp_obj, formulas=formulas,
        n_init=[1.0, 1.0, 0.001, 0.001],
        phase='gas', tol=1e-12, maxiter=50)
    G = res.G_history
    diffs = np.diff(G)
    print(f"    G_history (first 5): {[f'{g:.3e}' for g in G[:5]]}")
    print(f"    max increase between iters: {max(diffs):.2e}")
    check("G monotonically non-increasing (within 1e-6 of |G|)",
          max(diffs) < 1e-6 * max(abs(g) for g in G))


def test_gibbs_min_le_chatelier_pressure():
    """Methanol synthesis CO + 2H2 -> CH3OH (dn=-2) shifts to
    products at higher pressure."""
    section("test_gibbs_min_le_chatelier_pressure")
    from stateprop.reaction import gibbs_minimize_from_thermo, get_species
    species = ['CO', 'H2', 'CH3OH']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2}, {'C':1,'H':4,'O':1}]
    n_init = [1.0, 2.0, 0.001]
    T = 500.0
    res_lo = gibbs_minimize_from_thermo(
        T=T, p=1e5,    # 1 bar
        species=sp_obj, formulas=formulas, n_init=n_init,
        phase='gas', tol=1e-12)
    res_hi = gibbs_minimize_from_thermo(
        T=T, p=100e5,   # 100 bar
        species=sp_obj, formulas=formulas, n_init=n_init,
        phase='gas', tol=1e-12)
    methanol_lo = float(res_lo.n[2])
    methanol_hi = float(res_hi.n[2])
    print(f"    n[CH3OH] at 1 bar: {methanol_lo:.4f}")
    print(f"    n[CH3OH] at 100 bar: {methanol_hi:.4f}")
    check("higher p shifts to methanol (Le Chatelier)",
          methanol_hi > methanol_lo + 0.1)


def test_gibbs_min_n_init_positive_required():
    """n_init must be strictly positive (log(0) is undefined)."""
    section("test_gibbs_min_n_init_positive_required")
    from stateprop.reaction import gibbs_minimize_from_thermo, get_species
    species = ['CO', 'H2O', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2,'O':1}, {'C':1,'O':2}, {'H':2}]
    raised = False
    try:
        gibbs_minimize_from_thermo(
            T=1000.0, p=1e5, species=sp_obj, formulas=formulas,
            n_init=[1.0, 1.0, 0.0, 0.0],
            phase='gas')
    except ValueError:
        raised = True
    check("n_init with zero raises", raised)


def test_gibbs_min_solid_boudouard():
    """Boudouard equilibrium 2 CO <-> CO2 + C(s) shifts from
    products at low T to reactants at high T (textbook
    temperature-reversal of an exothermic reaction)."""
    section("test_gibbs_min_solid_boudouard")
    from stateprop.reaction import gibbs_minimize_TP, get_species
    species = ['CO', 'CO2', 'C(s)']
    formulas = [{'C':1,'O':1}, {'C':1,'O':2}, {'C':1}]
    sp_co = get_species('CO')
    sp_co2 = get_species('CO2')
    mu_funcs = [sp_co.Gf, sp_co2.Gf, lambda T: 0.0]   # graphite ref state
    phases = ['gas', 'gas', 'solid']
    n_init = [2.0, 0.001, 0.001]

    res_low = gibbs_minimize_TP(
        T=600.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, n_init=n_init,
        phase_per_species=phases, tol=1e-12, maxiter=100)
    res_high = gibbs_minimize_TP(
        T=1200.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, n_init=n_init,
        phase_per_species=phases, tol=1e-12, maxiter=100)
    print(f"    600 K:  n[CO]={res_low.n[0]:.4f}, n[C(s)]={res_low.n[2]:.4f}")
    print(f"    1200 K: n[CO]={res_high.n[0]:.4f}, n[C(s)]={res_high.n[2]:.4f}")
    check("low T: Boudouard near completion (CO consumed)",
          res_low.n[0] < 0.05)
    check("high T: Boudouard reversed (CO restored)",
          res_high.n[0] > 1.5)
    check("atom balance preserved at low T",
          res_low.atom_balance_residual < 1e-10)
    check("atom balance preserved at high T",
          res_high.atom_balance_residual < 1e-10)
    check("solid C(s) stoichiometry equals CO2 produced (n_C(s) = n_CO2)",
          abs(res_high.n[2] - res_high.n[1]) < 1e-10)


def test_gibbs_min_phase_split_high_T_matches_pure_gas():
    """At high T where everything is supercritical/vapor, the phase-
    split solver must reproduce the pure-gas Gibbs minimization."""
    section("test_gibbs_min_phase_split_high_T_matches_pure_gas")
    from stateprop.reaction import (gibbs_minimize_TP,
                                      gibbs_minimize_TP_phase_split,
                                      get_species)
    species = ['CO', 'H2O', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2,'O':1}, {'C':1,'O':2}, {'H':2}]
    mu_funcs = [s.Gf for s in sp_obj]
    def psat_high(T): return 1e9
    def psat_h2o(T): return 10**(8.07131 - 1730.63/((T-273.15)+233.426)) * 133.322
    psat_funcs = [psat_high, psat_h2o, psat_high, psat_high]
    n_init = [1.0, 1.0, 0.001, 0.001]

    res_gas = gibbs_minimize_TP(
        T=1000.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, n_init=n_init, phase='gas', tol=1e-12)
    res_ps = gibbs_minimize_TP_phase_split(
        T=1000.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=n_init, tol=1e-10, maxiter=80)
    print(f"    β = {res_ps.beta:.6f} (expect 1)")
    print(f"    max diff: {np.max(np.abs(res_ps.n - res_gas.n)):.2e}")
    check("phase-split β = 1 at high T", res_ps.beta > 0.99999)
    check("phase-split agrees with pure-gas to 1e-10 at high T",
          np.max(np.abs(res_ps.n - res_gas.n)) < 1e-10)


def test_gibbs_min_phase_split_two_phase_region():
    """Esterification at 360 K, 1 bar: real two-phase region (some
    vapor, some liquid).  Algorithm must converge with 0 < β < 1."""
    section("test_gibbs_min_phase_split_two_phase_region")
    from stateprop.reaction import gibbs_minimize_TP_phase_split
    from stateprop.activity.compounds import make_unifac
    species = ['acetic_acid', 'ethanol', 'ethyl_acetate', 'water']
    formulas = [{'C':2,'H':4,'O':2}, {'C':2,'H':6,'O':1},
                {'C':4,'H':8,'O':2}, {'H':2,'O':1}]
    mu_funcs = [lambda T: -389e3, lambda T: -174e3,
                lambda T: -332e3, lambda T: -228.6e3]
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T-273.15)+C)) * 133.322
    psats = [antoine(7.55716, 1642.540, 233.386),
             antoine(8.20417, 1642.890, 230.300),
             antoine(7.10179, 1244.951, 217.881),
             antoine(8.07131, 1730.630, 233.426)]
    uf = make_unifac(species)
    res = gibbs_minimize_TP_phase_split(
        T=360.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psats,
        activity_model=uf,
        n_init=[1.0, 1.0, 0.001, 0.001], tol=1e-9, maxiter=80)
    print(f"    converged: {res.converged}, iters: {res.iterations}")
    print(f"    β = {res.beta:.4f}, xi (≈ ethyl_acetate) = {res.n[2]:.4f}")
    print(f"    atom balance: {res.atom_balance_residual:.2e}")
    check("phase-split converged in two-phase region", res.converged)
    check("0 < β < 1 (real two-phase split)",
          0.05 < res.beta < 0.95)
    check("nontrivial chemistry conversion", res.n[2] > 0.1)
    check("atom balance preserved", res.atom_balance_residual < 1e-9)


def test_gibbs_min_phase_split_T_monotone():
    """As T increases through the bubble→dew range, β must increase
    monotonically."""
    section("test_gibbs_min_phase_split_T_monotone")
    from stateprop.reaction import gibbs_minimize_TP_phase_split
    from stateprop.activity.compounds import make_unifac
    species = ['acetic_acid', 'ethanol', 'ethyl_acetate', 'water']
    formulas = [{'C':2,'H':4,'O':2}, {'C':2,'H':6,'O':1},
                {'C':4,'H':8,'O':2}, {'H':2,'O':1}]
    mu_funcs = [lambda T: -389e3, lambda T: -174e3,
                lambda T: -332e3, lambda T: -228.6e3]
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T-273.15)+C)) * 133.322
    psats = [antoine(7.55716, 1642.540, 233.386),
             antoine(8.20417, 1642.890, 230.300),
             antoine(7.10179, 1244.951, 217.881),
             antoine(8.07131, 1730.630, 233.426)]
    uf = make_unifac(species)
    betas = []
    for T in [350.0, 360.0, 365.0, 370.0, 380.0]:
        res = gibbs_minimize_TP_phase_split(
            T=T, p=1e5, species_names=species, formulas=formulas,
            mu_standard_funcs=mu_funcs, psat_funcs=psats,
            activity_model=uf,
            n_init=[1.0, 1.0, 0.001, 0.001], tol=1e-9, maxiter=80)
        betas.append(res.beta)
    print(f"    β(T) = {[round(b, 3) for b in betas]}")
    diffs = [betas[i+1] - betas[i] for i in range(len(betas)-1)]
    check("β is monotonically non-decreasing in T",
          all(d >= -1e-6 for d in diffs))


def test_gibbs_min_phase_split_eos_low_p_matches_raoult():
    """At p = 1 bar, γ-φ-EOS phase-split Gibbs min must agree with
    modified-Raoult phase-split to within ~1e-4 absolute on the
    equilibrium product yield."""
    section("test_gibbs_min_phase_split_eos_low_p_matches_raoult")
    from stateprop.reaction import gibbs_minimize_TP_phase_split, get_species
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR

    species = ['CO', 'H2', 'CH3OH']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2}, {'C':1,'H':4,'O':1}]
    mu_funcs = [s.Gf for s in sp_obj]
    pure_eos = [PR(132.92, 3.494e6, 0.0480),
                PR(33.20, 1.297e6, -0.219),
                PR(512.6, 8.084e6, 0.5658)]
    mix = CubicMixture(pure_eos)
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T-273.15)+C)) * 133.322
    psat_funcs = [lambda T: 1e9, lambda T: 1e9,
                   antoine(7.89750, 1474.080, 229.130)]

    base = dict(T=500.0, p=1e5, species_names=species, formulas=formulas,
                mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
                n_init=[1.0, 2.0, 0.001], tol=1e-9, maxiter=80)
    res_no = gibbs_minimize_TP_phase_split(**base)
    res_eos = gibbs_minimize_TP_phase_split(**base, vapor_eos=mix)
    diff = abs(res_no.n[2] - res_eos.n[2])
    print(f"    no-EOS n[CH3OH] = {res_no.n[2]:.6f}")
    print(f"    γ-φ    n[CH3OH] = {res_eos.n[2]:.6f}")
    print(f"    |Δ| = {diff:.2e}")
    check("γ-φ at 1 bar matches Raoult to within 1e-4", diff < 1e-4)


def test_gibbs_min_phase_split_eos_high_p_diverges():
    """At p = 50 bar, γ-φ-EOS phase-split Gibbs min must differ from
    modified Raoult by at least 1% absolute on product yield."""
    section("test_gibbs_min_phase_split_eos_high_p_diverges")
    from stateprop.reaction import gibbs_minimize_TP_phase_split, get_species
    from stateprop.cubic.mixture import CubicMixture
    from stateprop.cubic.eos import PR
    from stateprop.activity import make_phi_sat_funcs

    species = ['CO', 'H2', 'CH3OH']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2}, {'C':1,'H':4,'O':1}]
    mu_funcs = [s.Gf for s in sp_obj]
    pure_eos = [PR(132.92, 3.494e6, 0.0480),
                PR(33.20, 1.297e6, -0.219),
                PR(512.6, 8.084e6, 0.5658)]
    mix = CubicMixture(pure_eos)
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T-273.15)+C)) * 133.322
    psat_funcs = [lambda T: 1e9, lambda T: 1e9,
                   antoine(7.89750, 1474.080, 229.130)]
    phi_sat = make_phi_sat_funcs(mix, psat_funcs)

    base = dict(T=500.0, p=50e5, species_names=species, formulas=formulas,
                mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
                n_init=[1.0, 2.0, 0.001], tol=1e-9, maxiter=80)
    res_no = gibbs_minimize_TP_phase_split(**base)
    res_eos = gibbs_minimize_TP_phase_split(**base, vapor_eos=mix,
                                             phi_sat_funcs=phi_sat)
    diff = abs(res_no.n[2] - res_eos.n[2])
    print(f"    no-EOS n[CH3OH] = {res_no.n[2]:.4f}")
    print(f"    γ-φ    n[CH3OH] = {res_eos.n[2]:.4f}")
    print(f"    |Δ| = {diff:.4f}")
    check("γ-φ at 50 bar differs from Raoult by >1%", diff > 0.01)
    check("both atom balances preserved",
          res_no.atom_balance_residual < 1e-9 and
          res_eos.atom_balance_residual < 1e-9)


def test_gibbs_min_phase_split_eos_dispatch_unchanged():
    """vapor_eos=None must give bit-identical output to v0.9.80
    phase-split solver."""
    section("test_gibbs_min_phase_split_eos_dispatch_unchanged")
    from stateprop.reaction import gibbs_minimize_TP_phase_split, get_species

    species = ['CO', 'H2O', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2,'O':1}, {'C':1,'O':2}, {'H':2}]
    mu_funcs = [s.Gf for s in sp_obj]
    def psat_high(T): return 1e9
    def psat_h2o(T): return 10**(8.07131 - 1730.63/((T-273.15)+233.426)) * 133.322
    psat_funcs = [psat_high, psat_h2o, psat_high, psat_high]
    base = dict(T=1000.0, p=1e5, species_names=species, formulas=formulas,
                mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
                n_init=[1.0, 1.0, 0.001, 0.001], tol=1e-10)
    r_default = gibbs_minimize_TP_phase_split(**base)
    r_explicit = gibbs_minimize_TP_phase_split(**base, vapor_eos=None)
    diff = float(np.max(np.abs(r_default.n - r_explicit.n)))
    print(f"    max |Δn| = {diff:.2e}")
    check("vapor_eos=None bit-identical", diff < 1e-15)


def test_gibbs_min_phase_split_solid_methane_cracking():
    """Methane cracking CH4 -> C(s) + 2 H2 at high T, p = 1 bar.
    Pure methane feed with C(s) seed.  Tests phase-split + solid
    combination."""
    section("test_gibbs_min_phase_split_solid_methane_cracking")
    from stateprop.reaction import gibbs_minimize_TP_phase_split, get_species
    species = ['CH4', 'H2', 'C(s)']
    sp = [get_species(s) for s in species[:2]]
    formulas = [{'C':1,'H':4}, {'H':2}, {'C':1}]
    mu_funcs = [s.Gf for s in sp] + [lambda T: 0.0]
    def psat_high(T): return 1e9
    psat_funcs = [psat_high, psat_high, lambda T: 0.0]
    phases = ['fluid', 'fluid', 'solid']

    res_lo = gibbs_minimize_TP_phase_split(
        T=800.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=[1.0, 0.001, 0.001],
        phase_per_species=phases, tol=1e-9, maxiter=80)
    res_hi = gibbs_minimize_TP_phase_split(
        T=1500.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=[1.0, 0.001, 0.001],
        phase_per_species=phases, tol=1e-9, maxiter=80)
    print(f"    800 K:  CH4={res_lo.n[0]:.4f}, H2={res_lo.n[1]:.4f}, "
          f"C(s)={res_lo.n[2]:.4f}")
    print(f"    1500 K: CH4={res_hi.n[0]:.4f}, H2={res_hi.n[1]:.4f}, "
          f"C(s)={res_hi.n[2]:.4f}")
    check("low T: low conversion to C(s)+H2", res_lo.n[0] > 0.5)
    check("high T: high conversion to C(s)+H2", res_hi.n[0] < 0.1)
    # Stoichiometry CH4 -> C(s) + 2 H2: net Δn[H2] = 2 Δn[C(s)]
    dH_lo = res_lo.n[1] - res_lo.n_init[1]
    dC_lo = res_lo.n[2] - res_lo.n_init[2]
    check("Δn[H2] = 2 Δn[C(s)] at low T",
          abs(dH_lo - 2 * dC_lo) < 1e-6)
    dH_hi = res_hi.n[1] - res_hi.n_init[1]
    dC_hi = res_hi.n[2] - res_hi.n_init[2]
    check("Δn[H2] = 2 Δn[C(s)] at high T",
          abs(dH_hi - 2 * dC_hi) < 1e-6)
    check("atom balance at low T preserved",
          res_lo.atom_balance_residual < 1e-10)
    check("atom balance at high T preserved",
          res_hi.atom_balance_residual < 1e-10)


def test_gibbs_min_phase_split_solid_no_coke_high_steam():
    """Steam reforming with high S:C ratio: methane fully converts,
    no graphite forms.  C(s) must end at floor (effectively zero)
    and the gas-phase K_eq's must match references."""
    section("test_gibbs_min_phase_split_solid_no_coke_high_steam")
    from stateprop.reaction import (gibbs_minimize_TP_phase_split,
                                      get_species, Reaction)
    species = ['CH4', 'H2O', 'CO', 'CO2', 'H2', 'C(s)']
    sp = [get_species(s) for s in species[:5]]
    formulas = [{'C':1,'H':4}, {'H':2,'O':1}, {'C':1,'O':1},
                {'C':1,'O':2}, {'H':2}, {'C':1}]
    mu_funcs = [s.Gf for s in sp] + [lambda T: 0.0]
    def psat_high(T): return 1e9
    def psat_h2o(T): return 10**(8.07131 - 1730.63/((T-273.15)+233.426)) * 133.322
    psat_funcs = [psat_high, psat_h2o, psat_high, psat_high,
                  psat_high, lambda T: 0.0]
    phases = ['fluid'] * 5 + ['solid']

    res = gibbs_minimize_TP_phase_split(
        T=1000.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=[1.0, 3.0, 0.001, 0.001, 0.001, 0.001],
        phase_per_species=phases, tol=1e-9, maxiter=80)
    print(f"    converged: {res.converged}, iters: {res.iterations}")
    print(f"    C(s) = {res.n[5]:.6f} (expect ~0)")
    print(f"    atom balance: {res.atom_balance_residual:.2e}")

    check("converged at high steam ratio", res.converged)
    check("C(s) at floor (no coke)", res.n[5] < 1e-6)
    check("atom balance preserved", res.atom_balance_residual < 1e-10)

    # Validate K_eq's at the converged composition (gas phase)
    n_fluid = res.n[:5]
    y = n_fluid / n_fluid.sum()
    rxn_smr = Reaction.from_names({'CH4':1, 'H2O':1}, {'CO':1, 'H2':3})
    rxn_wgs = Reaction.from_names({'CO':1, 'H2O':1}, {'CO2':1, 'H2':1})
    K_smr = (y[2] * y[4]**3) / (y[0] * y[1])    # at 1 bar
    K_wgs = (y[3] * y[4]) / (y[2] * y[1])
    e_smr = abs(K_smr - rxn_smr.K_eq(1000.0)) / rxn_smr.K_eq(1000.0)
    e_wgs = abs(K_wgs - rxn_wgs.K_eq(1000.0)) / rxn_wgs.K_eq(1000.0)
    print(f"    K_SMR rel.err = {e_smr:.2e}, K_WGS rel.err = {e_wgs:.2e}")
    check("SMR K_eq satisfied to 1e-5", e_smr < 1e-5)
    check("WGS K_eq satisfied to 1e-5", e_wgs < 1e-5)


def test_gibbs_min_phase_split_solid_phase_per_species_default():
    """phase_per_species=None must reproduce v0.9.81 phase-split
    bit-identically (no solids in the mix)."""
    section("test_gibbs_min_phase_split_solid_phase_per_species_default")
    from stateprop.reaction import gibbs_minimize_TP_phase_split, get_species
    species = ['CO', 'H2O', 'CO2', 'H2']
    sp_obj = [get_species(s) for s in species]
    formulas = [{'C':1,'O':1}, {'H':2,'O':1}, {'C':1,'O':2}, {'H':2}]
    mu_funcs = [s.Gf for s in sp_obj]
    def psat_high(T): return 1e9
    def psat_h2o(T): return 10**(8.07131 - 1730.63/((T-273.15)+233.426)) * 133.322
    psat_funcs = [psat_high, psat_h2o, psat_high, psat_high]
    base = dict(T=1000.0, p=1e5, species_names=species, formulas=formulas,
                mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
                n_init=[1.0, 1.0, 0.001, 0.001], tol=1e-10)
    r_default = gibbs_minimize_TP_phase_split(**base)
    r_explicit = gibbs_minimize_TP_phase_split(
        **base, phase_per_species=['fluid'] * 4)
    diff = float(np.max(np.abs(r_default.n - r_explicit.n)))
    print(f"    max |Δn| = {diff:.2e}")
    check("phase_per_species=None bit-identical to all-fluid", diff < 1e-15)


def test_gibbs_min_solid_reactivation_methane_cracking():
    """v0.9.83 active-set reactivation: methane cracking with C(s)
    seeded at the floor (1e-25).  Without reactivation, the algorithm
    cannot form C(s) and would not converge to the correct
    equilibrium.  With reactivation, supersaturation of C(s) is
    detected on the first iteration and the species is added back to
    the active set; the result matches the case with a positive
    seed."""
    section("test_gibbs_min_solid_reactivation_methane_cracking")
    from stateprop.reaction import gibbs_minimize_TP_phase_split, get_species
    species = ['CH4', 'H2', 'C(s)']
    sp = [get_species(s) for s in species[:2]]
    formulas = [{'C':1,'H':4}, {'H':2}, {'C':1}]
    mu_funcs = [s.Gf for s in sp] + [lambda T: 0.0]
    def psat_high(T): return 1e9
    psat_funcs = [psat_high, psat_high, lambda T: 0.0]
    phases = ['fluid', 'fluid', 'solid']

    res_react = gibbs_minimize_TP_phase_split(
        T=1500.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=[1.0, 0.001, 1e-25],   # C(s) at floor
        phase_per_species=phases, tol=1e-9, maxiter=80)
    res_seed = gibbs_minimize_TP_phase_split(
        T=1500.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=[1.0, 0.001, 0.001],   # positive seed
        phase_per_species=phases, tol=1e-9, maxiter=80)
    print(f"    seed-at-floor: n[CH4]={res_react.n[0]:.4f}, "
          f"n[C(s)]={res_react.n[2]:.4f}, conv={res_react.converged}, "
          f"iters={res_react.iterations}")
    print(f"    positive seed: n[CH4]={res_seed.n[0]:.4f}, "
          f"n[C(s)]={res_seed.n[2]:.4f}, conv={res_seed.converged}, "
          f"iters={res_seed.iterations}")
    check("reactivation case converged", res_react.converged)
    check("reactivation produces same final state as positive-seed",
          abs(res_react.n[2] - res_seed.n[2]) < 1e-3)
    check("reactivation atom balance preserved",
          res_react.atom_balance_residual < 1e-10)


def test_gibbs_min_LL_split_water_butanol():
    """Water + n-butanol LLE at 298 K (no chemistry) using
    UNIFAC-LLE.  The two phases must satisfy μ_i^L1 = μ_i^L2, i.e.
    γ_i^L1 x_i^L1 = γ_i^L2 x_i^L2 for both species."""
    section("test_gibbs_min_LL_split_water_butanol")
    from stateprop.reaction import gibbs_minimize_TP_LL_split
    from stateprop.activity.compounds import make_unifac_lle

    species = ['water', '1-butanol']
    formulas = [{'H':2, 'O':1}, {'C':4, 'H':10, 'O':1}]
    mu_funcs = [lambda T: 0.0, lambda T: 0.0]   # no chemistry
    uf_lle = make_unifac_lle(species)

    res = gibbs_minimize_TP_LL_split(
        T=298.15, p=1e5,
        species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, activity_model=uf_lle,
        n_init=[1.0, 1.0],
        x1_seed=[0.95, 0.05],
        x2_seed=[0.50, 0.50],
        tol=1e-8, maxiter=80)
    print(f"    converged: {res.converged}")
    print(f"    phase 1: x = {res.x1}, γ = {res.gammas1}")
    print(f"    phase 2: x = {res.x2}, γ = {res.gammas2}")
    a1 = res.gammas1 * res.x1
    a2 = res.gammas2 * res.x2
    print(f"    activities: a_1 = {a1}, a_2 = {a2}")
    check("LL flash converged", res.converged)
    check("activities equal between phases for water",
          abs(a1[0] - a2[0]) < 1e-4)
    check("activities equal between phases for butanol",
          abs(a1[1] - a2[1]) < 1e-4)
    # Phase 1 should be water-rich (>95% water), phase 2 butanol-rich
    check("phase 1 is water-rich (x_water > 0.9)", res.x1[0] > 0.9)
    check("phase 2 is butanol-rich (x_butanol > 0.5)", res.x2[1] > 0.5)
    check("atom balance preserved", res.atom_balance_residual < 1e-10)


def test_gibbs_min_LL_split_x1_x2_collapse_handled():
    """If x1_seed and x2_seed are too similar, the LL flash should
    collapse and the algorithm should fall back to single-phase."""
    section("test_gibbs_min_LL_split_x1_x2_collapse_handled")
    from stateprop.reaction import gibbs_minimize_TP_LL_split
    from stateprop.activity.compounds import make_unifac_lle

    species = ['water', '1-butanol']
    formulas = [{'H':2, 'O':1}, {'C':4, 'H':10, 'O':1}]
    mu_funcs = [lambda T: 0.0, lambda T: 0.0]
    uf_lle = make_unifac_lle(species)

    # Identical seeds → collapse
    res = gibbs_minimize_TP_LL_split(
        T=298.15, p=1e5,
        species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, activity_model=uf_lle,
        n_init=[1.0, 1.0],
        x1_seed=[0.5, 0.5],
        x2_seed=[0.5, 0.5],
        tol=1e-8, maxiter=80)
    # The LL flash internally raises ValueError; the wrapper falls back
    # gracefully to single-phase.  The algorithm should still terminate
    # with a sensible (non-NaN) result.
    print(f"    n = {res.n}, converged = {res.converged}")
    check("collapsed seeds doesn't crash",
          np.all(np.isfinite(res.n)))


def test_gibbs_min_VLL_split_2LL_matches_LL_split():
    """v0.9.84 dispatch: when the system is in the 2LL regime (no
    vapor), the VLLE Gibbs minimizer must reduce bit-identically to
    the v0.9.83 LL Gibbs minimizer.  Water/butanol at 298 K is purely
    LL — the dispatch must pick the 2LL branch and produce a result
    matching ``gibbs_minimize_TP_LL_split`` on the same system."""
    section("test_gibbs_min_VLL_split_2LL_matches_LL_split")
    from stateprop.reaction import (gibbs_minimize_TP_VLL_split,
                                      gibbs_minimize_TP_LL_split)
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['water', '1-butanol']
    formulas = [{'H':2, 'O':1}, {'C':4, 'H':10, 'O':1}]
    eos_water = PR(T_c=647.1, p_c=22.06e6, acentric_factor=0.345)
    eos_butan = PR(T_c=563.0, p_c=4.42e6, acentric_factor=0.594)
    mix = CubicMixture([eos_water, eos_butan])
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T - 273.15) + C)) * 133.322
    psat_funcs = [antoine(8.07131, 1730.63, 233.426),
                  antoine(7.36366, 1305.198, 173.427)]
    uf_lle = make_unifac_lle(species)
    mu_funcs = [lambda T: 0.0, lambda T: 0.0]

    res_vll = gibbs_minimize_TP_VLL_split(
        T=298.0, p=1.013e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        activity_model=uf_lle, vapor_eos=mix,
        n_init=[1.0, 1.0],
        x1_seed=[0.95, 0.05], x2_seed=[0.30, 0.70],
        tol=1e-7, maxiter=80)
    res_ll = gibbs_minimize_TP_LL_split(
        T=298.0, p=1.013e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, activity_model=uf_lle,
        n_init=[1.0, 1.0],
        x1_seed=[0.95, 0.05], x2_seed=[0.30, 0.70],
        tol=1e-7, maxiter=80)
    diff_x1 = float(np.max(np.abs(res_vll.x1 - res_ll.x1)))
    diff_x2 = float(np.max(np.abs(res_vll.x2 - res_ll.x2)))
    diff_beta = abs(res_vll.beta_L2 - res_ll.beta)
    print(f"    VLLE β_V = {res_vll.beta_V:.6f} (expect 0)")
    print(f"    diff x1 = {diff_x1:.2e}, x2 = {diff_x2:.2e}, "
          f"β = {diff_beta:.2e}")
    check("VLLE β_V = 0 in 2LL regime", res_vll.beta_V < 1e-8)
    check("VLLE x1 matches LLE x1", diff_x1 < 1e-6)
    check("VLLE x2 matches LLE x2", diff_x2 < 1e-6)
    check("VLLE βL2 matches LLE β", diff_beta < 1e-6)
    a1 = res_vll.gammas1 * res_vll.x1
    a2 = res_vll.gammas2 * res_vll.x2
    check("activities equal between L1 and L2 (water)",
          abs(a1[0] - a2[0]) < 1e-4)
    check("activities equal between L1 and L2 (butanol)",
          abs(a1[1] - a2[1]) < 1e-4)


def test_gibbs_min_VLL_split_all_vapor_at_high_T():
    """v0.9.84 dispatch: at high T the 3-phase flash should collapse
    to all-vapor (β_V = 1, β_L1 = β_L2 = 0)."""
    section("test_gibbs_min_VLL_split_all_vapor_at_high_T")
    from stateprop.reaction import gibbs_minimize_TP_VLL_split
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['water', '1-butanol']
    formulas = [{'H':2, 'O':1}, {'C':4, 'H':10, 'O':1}]
    eos_water = PR(T_c=647.1, p_c=22.06e6, acentric_factor=0.345)
    eos_butan = PR(T_c=563.0, p_c=4.42e6, acentric_factor=0.594)
    mix = CubicMixture([eos_water, eos_butan])
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T - 273.15) + C)) * 133.322
    psat_funcs = [antoine(8.07131, 1730.63, 233.426),
                  antoine(7.36366, 1305.198, 173.427)]
    uf_lle = make_unifac_lle(species)
    mu_funcs = [lambda T: 0.0, lambda T: 0.0]

    res = gibbs_minimize_TP_VLL_split(
        T=600.0, p=1.013e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        activity_model=uf_lle, vapor_eos=mix,
        n_init=[1.0, 1.0],
        x1_seed=[0.95, 0.05], x2_seed=[0.30, 0.70],
        tol=1e-7, maxiter=80)
    print(f"    β_V = {res.beta_V:.6f} (expect ~1)")
    print(f"    β_L1 = {res.beta_L1:.6f}, β_L2 = {res.beta_L2:.6f} (expect ~0)")
    check("β_V → 1 at high T", res.beta_V > 0.99)
    check("β_L1 → 0 at high T", res.beta_L1 < 0.01)
    check("β_L2 → 0 at high T", res.beta_L2 < 0.01)
    check("atom balance preserved", res.atom_balance_residual < 1e-10)


def test_gibbs_min_VLL_split_input_validation():
    """v0.9.84 input validation: bad seed lengths must raise."""
    section("test_gibbs_min_VLL_split_input_validation")
    from stateprop.reaction import gibbs_minimize_TP_VLL_split
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['water', '1-butanol']
    formulas = [{'H':2, 'O':1}, {'C':4, 'H':10, 'O':1}]
    eos_water = PR(T_c=647.1, p_c=22.06e6, acentric_factor=0.345)
    eos_butan = PR(T_c=563.0, p_c=4.42e6, acentric_factor=0.594)
    mix = CubicMixture([eos_water, eos_butan])
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T - 273.15) + C)) * 133.322
    psat_funcs = [antoine(8.07131, 1730.63, 233.426),
                  antoine(7.36366, 1305.198, 173.427)]
    uf_lle = make_unifac_lle(species)
    mu_funcs = [lambda T: 0.0, lambda T: 0.0]

    raised = False
    try:
        gibbs_minimize_TP_VLL_split(
            T=298.0, p=1e5, species_names=species, formulas=formulas,
            mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
            activity_model=uf_lle, vapor_eos=mix,
            n_init=[1.0, 1.0],
            x1_seed=[0.95],   # wrong length
            x2_seed=[0.30, 0.70])
    except ValueError:
        raised = True
    check("bad x1_seed length raises ValueError", raised)


def test_gibbs_min_VLL_split_3VLL_water_acetone_hexane():
    """v0.9.85: TRUE 3-phase VLLE on a ternary system known to be
    in the 3VLL region.

    Water + acetone + n-hexane has atom-balance rank 3 (det A != 0),
    so the chemistry cannot move the composition away from n_init.
    At T=323 K with z=(0.3, 0.20, 0.50), UNIFAC-LLE + PR places the
    system in the 3VLL region with β_V≈0.26, β_L1≈0.29, β_L2≈0.45.
    All three phases must coexist; activities must be equal across
    L1 and L2 (and consistent with vapor μ at phase equilibrium)."""
    section("test_gibbs_min_VLL_split_3VLL_water_acetone_hexane")
    from stateprop.reaction import gibbs_minimize_TP_VLL_split
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['water', 'acetone', 'n-hexane']
    formulas = [{'H':2, 'O':1}, {'C':3, 'H':6, 'O':1}, {'C':6, 'H':14}]
    mix = CubicMixture([
        PR(T_c=647.1, p_c=22.06e6, acentric_factor=0.345),
        PR(T_c=508.1, p_c=4.70e6,  acentric_factor=0.307),
        PR(T_c=507.6, p_c=3.025e6, acentric_factor=0.301),
    ])
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T - 273.15) + C)) * 133.322
    psat_funcs = [antoine(8.07131, 1730.63, 233.426),
                  antoine(7.11714, 1210.595, 229.664),
                  antoine(6.87024, 1168.72,  224.21)]
    uf = make_unifac_lle(species)

    res = gibbs_minimize_TP_VLL_split(
        T=323.0, p=1.013e5, species_names=species, formulas=formulas,
        mu_standard_funcs=[lambda T: 0.0] * 3, psat_funcs=psat_funcs,
        activity_model=uf, vapor_eos=mix,
        n_init=[0.3, 0.2, 0.5],
        x1_seed=[0.93, 0.07, 0.005], x2_seed=[0.005, 0.17, 0.825],
        beta_V_seed=0.25, beta_L2_seed=0.45,
        tol=1e-7, maxiter=80)
    print(f"    β_V={res.beta_V:.4f}, β_L1={res.beta_L1:.4f}, β_L2={res.beta_L2:.4f}")
    print(f"    x1={res.x1}")
    print(f"    x2={res.x2}")
    print(f"    y ={res.y_vapor}")

    check("3VLL: β_V > 0.05", res.beta_V > 0.05)
    check("3VLL: β_L1 > 0.05", res.beta_L1 > 0.05)
    check("3VLL: β_L2 > 0.05", res.beta_L2 > 0.05)
    check("3VLL: β sums to 1",
          abs(res.beta_V + res.beta_L1 + res.beta_L2 - 1.0) < 1e-6)

    a1 = res.gammas1 * res.x1
    a2 = res.gammas2 * res.x2
    for i, s in enumerate(species):
        check(f"a_{s} equal in L1 and L2 to 1e-6",
              abs(a1[i] - a2[i]) < 1e-6)

    check("atom balance preserved", res.atom_balance_residual < 1e-12)
    check("L1 is water-rich (x_water > 0.9)", res.x1[0] > 0.9)
    check("L2 is hexane-rich (x_hexane > 0.7)", res.x2[2] > 0.7)
    check("vapor concentrated in acetone (y_acetone > 0.30)",
          res.y_vapor[1] > 0.30)


def test_gibbs_min_VLLS_split_reduces_to_VLL_no_solids():
    """v0.9.85 dispatch: when no solids are present, the 4-phase
    solver must reduce bit-identically to the 3-phase solver."""
    section("test_gibbs_min_VLLS_split_reduces_to_VLL_no_solids")
    from stateprop.reaction import (gibbs_minimize_TP_VLL_split,
                                      gibbs_minimize_TP_VLLS_split)
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['water', 'acetone', 'n-hexane']
    formulas = [{'H':2, 'O':1}, {'C':3, 'H':6, 'O':1}, {'C':6, 'H':14}]
    mix = CubicMixture([
        PR(T_c=647.1, p_c=22.06e6, acentric_factor=0.345),
        PR(T_c=508.1, p_c=4.70e6,  acentric_factor=0.307),
        PR(T_c=507.6, p_c=3.025e6, acentric_factor=0.301),
    ])
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T - 273.15) + C)) * 133.322
    psat_funcs = [antoine(8.07131, 1730.63, 233.426),
                  antoine(7.11714, 1210.595, 229.664),
                  antoine(6.87024, 1168.72, 224.21)]
    uf = make_unifac_lle(species)
    base = dict(T=323.0, p=1.013e5, species_names=species, formulas=formulas,
                mu_standard_funcs=[lambda T: 0.0] * 3, psat_funcs=psat_funcs,
                activity_model=uf, vapor_eos=mix,
                n_init=[0.3, 0.2, 0.5],
                x1_seed=[0.93, 0.07, 0.005],
                x2_seed=[0.005, 0.17, 0.825],
                beta_V_seed=0.25, beta_L2_seed=0.45,
                tol=1e-7)
    r3 = gibbs_minimize_TP_VLL_split(**base)
    r4 = gibbs_minimize_TP_VLLS_split(**base)
    diff_n = float(np.max(np.abs(r3.n - r4.n)))
    diff_bV = abs(r3.beta_V - r4.beta_V)
    diff_bL2 = abs(r3.beta_L2 - r4.beta_L2)
    print(f"    max |Δn| = {diff_n:.2e}, ΔβV = {diff_bV:.2e}, "
          f"ΔβL2 = {diff_bL2:.2e}")
    check("VLLS without solids = VLL bit-identically", diff_n < 1e-15)
    check("βV identical", diff_bV < 1e-15)
    check("βL2 identical", diff_bL2 < 1e-15)


def test_gibbs_min_VLLS_split_methane_cracking_reduces_to_VLE_S():
    """v0.9.85 dispatch: when there is no LL split (single fluid
    phase), the 4-phase solver must reduce to the v0.9.82 phase-split
    solver with solids.  Methane cracking gives n[C(s)]=0.965 at
    1500K — must match v0.9.82 result."""
    section("test_gibbs_min_VLLS_split_methane_cracking_reduces_to_VLE_S")
    from stateprop.reaction import (gibbs_minimize_TP_VLLS_split,
                                      gibbs_minimize_TP_phase_split,
                                      get_species)
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['CH4', 'H2', 'C(s)']
    sp = [get_species(s) for s in species[:2]]
    formulas = [{'C':1,'H':4}, {'H':2}, {'C':1}]
    mu_funcs = [s.Gf for s in sp] + [lambda T: 0.0]
    def psat_high(T): return 1e9
    psat_funcs = [psat_high, psat_high, lambda T: 0.0]
    phases = ['fluid', 'fluid', 'solid']

    mix = CubicMixture([
        PR(T_c=190.6, p_c=4.6e6, acentric_factor=0.011),
        PR(T_c=33.2, p_c=1.3e6, acentric_factor=-0.22),
    ])

    class IdealLiq:
        N = 2
        def gammas(self, T, x): return np.ones(2)

    res4 = gibbs_minimize_TP_VLLS_split(
        T=1500.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        activity_model=IdealLiq(), vapor_eos=mix,
        n_init=[1.0, 0.001, 0.001],
        x1_seed=[0.5, 0.5, 0.0], x2_seed=[0.5, 0.5, 0.0],
        phase_per_species=phases, tol=1e-9, maxiter=80)
    res2 = gibbs_minimize_TP_phase_split(
        T=1500.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        n_init=[1.0, 0.001, 0.001],
        phase_per_species=phases, tol=1e-9, maxiter=80)
    diff_n = float(np.max(np.abs(res4.n - res2.n)))
    print(f"    4-phase: n[CH4]={res4.n[0]:.4f}, n[C(s)]={res4.n[2]:.4f}")
    print(f"    v0.9.82: n[CH4]={res2.n[0]:.4f}, n[C(s)]={res2.n[2]:.4f}")
    print(f"    max |Δn| = {diff_n:.2e}")
    check("4-phase solver matches v0.9.82 methane cracking",
          diff_n < 1e-3)
    check("4-phase converged", res4.converged)
    check("β_V → 1 at high T (no liquid)", res4.beta_V > 0.99)
    check("atom balance preserved", res4.atom_balance_residual < 1e-10)


def test_gibbs_min_VLLS_split_combined_3VLL_with_inert_solid():
    """v0.9.85 ultimate dispatch: 3VLL fluid system + solid that is
    inert (μ°_solid is too high for it to form).  The fluid result
    must match the pure 3VLL result; the solid must remain at floor."""
    section("test_gibbs_min_VLLS_split_combined_3VLL_with_inert_solid")
    from stateprop.reaction import (gibbs_minimize_TP_VLL_split,
                                      gibbs_minimize_TP_VLLS_split)
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species_full = ['water', 'acetone', 'n-hexane', 'X(s)']
    formulas_full = [{'H':2, 'O':1}, {'C':3, 'H':6, 'O':1}, {'C':6, 'H':14}, {'X':1}]
    species_fluid = species_full[:3]
    formulas_fluid = formulas_full[:3]
    mix = CubicMixture([
        PR(T_c=647.1, p_c=22.06e6, acentric_factor=0.345),
        PR(T_c=508.1, p_c=4.70e6,  acentric_factor=0.307),
        PR(T_c=507.6, p_c=3.025e6, acentric_factor=0.301),
        PR(T_c=300.0, p_c=1.0e6,   acentric_factor=0.0),    # solid placeholder
    ])
    def antoine(A, B, C):
        return lambda T: 10**(A - B/((T - 273.15) + C)) * 133.322
    psat_funcs_fluid = [antoine(8.07131, 1730.63, 233.426),
                        antoine(7.11714, 1210.595, 229.664),
                        antoine(6.87024, 1168.72, 224.21)]
    psat_funcs_full = psat_funcs_fluid + [lambda T: 0.0]
    uf = make_unifac_lle(species_fluid)
    mu_funcs_full = [lambda T: 0.0] * 3 + [lambda T: 500e3]
    phases = ['fluid', 'fluid', 'fluid', 'solid']

    res4 = gibbs_minimize_TP_VLLS_split(
        T=323.0, p=1.013e5, species_names=species_full, formulas=formulas_full,
        mu_standard_funcs=mu_funcs_full, psat_funcs=psat_funcs_full,
        activity_model=uf, vapor_eos=mix,
        n_init=[0.3, 0.2, 0.5, 1e-25],
        x1_seed=[0.93, 0.07, 0.005, 0.0],
        x2_seed=[0.005, 0.17, 0.825, 0.0],
        beta_V_seed=0.25, beta_L2_seed=0.45,
        phase_per_species=phases, tol=1e-7, maxiter=80)
    # Compare against pure-3VLL result on a 3-species mix (sliced).
    mix_fluid = CubicMixture(mix.components[:3])
    res3 = gibbs_minimize_TP_VLL_split(
        T=323.0, p=1.013e5, species_names=species_fluid, formulas=formulas_fluid,
        mu_standard_funcs=[lambda T: 0.0] * 3, psat_funcs=psat_funcs_fluid,
        activity_model=uf, vapor_eos=mix_fluid,
        n_init=[0.3, 0.2, 0.5],
        x1_seed=[0.93, 0.07, 0.005], x2_seed=[0.005, 0.17, 0.825],
        beta_V_seed=0.25, beta_L2_seed=0.45,
        tol=1e-7)
    diff_fluid = float(np.max(np.abs(res4.n[:3] - res3.n)))
    print(f"    4-phase fluid: n = {res4.n[:3]}")
    print(f"    3-phase result: n = {res3.n}")
    print(f"    n[X(s)] = {res4.n[3]:.2e} (must stay at floor)")
    check("inert solid stays at floor",
          res4.n[3] < 1e-20)
    check("fluid moles match 3-phase result", diff_fluid < 1e-12)
    check("βV matches 3-phase",
          abs(res4.beta_V - res3.beta_V) < 1e-8)
    check("βL2 matches 3-phase",
          abs(res4.beta_L2 - res3.beta_L2) < 1e-8)


def test_gibbs_min_VLLS_split_full_4phase_real_chemistry():
    """v0.9.86 ultimate test: ALL FOUR PHASES SIMULTANEOUSLY ACTIVE
    with real chemistry.

    System: CO + CO2 + N2 + water + n-hexane + C(s) at T=300K, p=1bar.

    Phase configuration at convergence:
      - V:  inert N2 (mostly) + trace water/hexane vapor + traces of CO2
      - L1: nearly pure water (water-rich liquid, ~99.99%)
      - L2: nearly pure n-hexane (hexane-rich liquid, ~99.92%)
      - S:  graphite C(s)

    Chemistry at convergence (extent inferred from Δn):
      - Dominant: hexane partial combustion via CO oxidation
            7 CO + C6H14 → 7 H2O + 13 C(s)
        ΔG very negative at 300 K because Gf(CO)=-137 kJ/mol,
        Gf(H2O)=-228 kJ/mol, with stable C(s) and stable hexane
        (μ°_hex set to a deeply negative value to keep hexane
        thermodynamically stable as the solvent).
      - Reverse-Boudouard: trace CO2 + C(s) → 2 CO supplies the
        small remaining CO needed by the hexane reaction.

    This test exercises every feature of the v0.9.85 four-phase
    Gibbs minimizer simultaneously: chemistry, multi-element atom
    balance, vapor-liquid equilibrium with non-ideal vapor (PR EOS),
    liquid-liquid split (UNIFAC-LLE for the polar/non-polar pair),
    and solid-phase stationarity with active-set re-activation.

    The activity model wraps UNIFAC-LLE for water+hexane and assigns
    a high γ (1e6) to gas species, modeling the negligible solubility
    of CO/CO2/N2 in liquid water and hexane.  A fluid-only vapor EOS
    is constructed internally by slicing the user-supplied
    CubicMixture to the fluid components.
    """
    section("test_gibbs_min_VLLS_split_full_4phase_real_chemistry")
    from stateprop.reaction import gibbs_minimize_TP_VLLS_split, get_species
    from stateprop.activity.compounds import make_unifac_lle
    from stateprop.cubic.eos import PR
    from stateprop.cubic.mixture import CubicMixture

    species = ['CO', 'CO2', 'N2', 'water', 'n-hexane', 'C(s)']
    formulas = [{'C': 1, 'O': 1}, {'C': 1, 'O': 2}, {'N': 2},
                {'H': 2, 'O': 1}, {'C': 6, 'H': 14}, {'C': 1}]
    sp = [get_species(s) for s in ['CO', 'CO2', 'N2', 'H2O']]
    mu_funcs = [sp[0].Gf, sp[1].Gf, sp[2].Gf, sp[3].Gf,
                lambda T: -200e3,        # hexane: deeply stable solvent
                lambda T: 0.0]           # C(s): reference state

    def antoine(A, B, C):
        return lambda T: 10**(A - B / ((T - 273.15) + C)) * 133.322
    psat_funcs = [lambda T: 1e9, lambda T: 1e9, lambda T: 1e9,
                  antoine(8.07131, 1730.63, 233.426),
                  antoine(6.87024, 1168.72, 224.21),
                  lambda T: 0.0]

    uf_base = make_unifac_lle(['water', 'n-hexane'])

    class WrappedActivity:
        N = 6
        base_idx = np.array([3, 4])
        def gammas(self, T, x):
            x_full = np.asarray(x).copy()
            x_base = x_full[self.base_idx]
            s = x_base.sum()
            x_base_norm = x_base / s if s > 1e-30 else np.full(2, 0.5)
            g_base = np.asarray(uf_base.gammas(T, x_base_norm))
            gammas = np.ones(self.N) * 1e6   # gases: very high γ in liquid
            gammas[self.base_idx] = g_base   # liquid pair: real UNIFAC-LLE
            return gammas

    mix = CubicMixture([
        PR(T_c=132.85, p_c=3.494e6, acentric_factor=0.045),
        PR(T_c=304.13, p_c=7.377e6, acentric_factor=0.225),
        PR(T_c=126.21, p_c=3.39e6,  acentric_factor=0.04),
        PR(T_c=647.1,  p_c=22.06e6, acentric_factor=0.345),
        PR(T_c=507.6,  p_c=3.025e6, acentric_factor=0.301),
        PR(T_c=300.0,  p_c=1.0e6,   acentric_factor=0.0),    # solid placeholder
    ])
    phases = ['fluid'] * 5 + ['solid']

    res = gibbs_minimize_TP_VLLS_split(
        T=300.0, p=1e5, species_names=species, formulas=formulas,
        mu_standard_funcs=mu_funcs, psat_funcs=psat_funcs,
        activity_model=WrappedActivity(), vapor_eos=mix,
        n_init=[0.1, 0.001, 1.0, 1.0, 1.0, 0.001],
        x1_seed=[1e-8, 1e-8, 1e-8, 0.99, 0.005, 0.0],
        x2_seed=[1e-8, 1e-8, 1e-8, 0.005, 0.99, 0.0],
        beta_V_seed=0.4, beta_L2_seed=0.3,
        phase_per_species=phases, tol=1e-6, maxiter=100)

    print(f"    converged: {res.converged}, iters: {res.iterations}")
    print(f"    β_V={res.beta_V:.4f}, β_L1={res.beta_L1:.4f}, "
          f"β_L2={res.beta_L2:.4f}, n[C(s)]={res.n[5]:.4f}")

    # All four phases must be simultaneously active
    check("ALL 4 PHASES: β_V > 0.05",  res.beta_V > 0.05)
    check("ALL 4 PHASES: β_L1 > 0.05", res.beta_L1 > 0.05)
    check("ALL 4 PHASES: β_L2 > 0.05", res.beta_L2 > 0.05)
    check("ALL 4 PHASES: n[C(s)] > 0.05", res.n[5] > 0.05)

    # β_V + β_L1 + β_L2 = 1 (fluid-phase fractions sum to 1)
    check("fluid β sums to 1",
          abs(res.beta_V + res.beta_L1 + res.beta_L2 - 1.0) < 1e-6)

    # Atom balance preserved across the chemistry
    check("atom balance preserved", res.atom_balance_residual < 1e-10)

    # Phase identity checks
    iCO, iCO2, iN2, iH2O, iHEX, iCs = 0, 1, 2, 3, 4, 5
    check("L1 is water-rich (x_water > 0.99)", res.x1[iH2O] > 0.99)
    check("L2 is hexane-rich (x_hexane > 0.99)", res.x2[iHEX] > 0.99)
    check("vapor is mostly N2 (y_N2 > 0.5)", res.y_vapor[iN2] > 0.5)

    # Chemistry happened: CO consumed, C(s) produced, water produced
    check("CO consumed (n_CO < 0.001)", res.n[iCO] < 0.001)
    check("C(s) produced (n_Cs > 0.05)", res.n[iCs] > 0.05)
    check("water produced (Δn_H2O > 0.05)",
          res.n[iH2O] - res.n_init[iH2O] > 0.05)

    # Inert N2 is conserved exactly
    check("N2 preserved exactly", abs(res.n[iN2] - 1.0) < 1e-9)

    # Stoichiometry of dominant reaction:
    #   7 CO + C6H14 → 7 H2O + 13 C(s)
    # Plus reverse-Boudouard: CO2 + C(s) → 2 CO providing trace CO.
    # Direct atom accounting: H balance checks self-consistency.
    H_init = (2 * res.n_init[iH2O] + 14 * res.n_init[iHEX])
    H_final = (2 * res.n[iH2O] + 14 * res.n[iHEX])
    check("H atom balance (independent of stoichiometry assumed)",
          abs(H_init - H_final) < 1e-10)
    O_init = (1 * res.n_init[iCO] + 2 * res.n_init[iCO2]
              + 1 * res.n_init[iH2O])
    O_final = (1 * res.n[iCO] + 2 * res.n[iCO2] + 1 * res.n[iH2O])
    check("O atom balance",
          abs(O_init - O_final) < 1e-10)
    C_init = (res.n_init[iCO] + res.n_init[iCO2]
              + 6 * res.n_init[iHEX] + res.n_init[iCs])
    C_final = (res.n[iCO] + res.n[iCO2]
               + 6 * res.n[iHEX] + res.n[iCs])
    check("C atom balance",
          abs(C_init - C_final) < 1e-10)


def main():
    for fn in [
        test_species_self_consistency_at_298,
        test_elements_have_zero_formation,
        test_water_gas_shift_K_eq,
        test_methanol_synthesis_K_eq_temperature_trend,
        test_ammonia_synthesis_K_eq_high_at_low_T,
        test_extent_solver_water_gas_shift,
        test_extent_solver_le_chatelier_pressure,
        test_extent_solver_le_chatelier_inerts,
        test_extent_solver_thermo_consistency,
        test_get_species_unknown_raises,
        test_get_species_synonyms,
        test_reaction_extent_zero_initial_product_ok,
        # v0.9.62 — Multi-reaction equilibrium
        test_multi_reaction_construction,
        test_multi_reaction_linear_dependence_rejected,
        test_multi_reaction_smr_steam_reforming,
        test_multi_reaction_K_consistency_at_solution,
        test_multi_reaction_mass_balance,
        test_multi_reaction_le_chatelier_pressure,
        test_multi_reaction_inert_dilution_helps_reforming,
        test_multi_reaction_disjoint_reactions,
        # v0.9.63 — Real-gas K_eq corrections via EOS fugacity coefficients
        test_real_gas_low_pressure_matches_ideal,
        test_real_gas_methanol_synthesis_high_p,
        test_real_gas_K_y_satisfies_equilibrium_with_phi,
        test_real_gas_multi_reaction_smr,
        # v0.9.64 — Liquid-phase reactions with activity coefficients
        test_liquid_phase_reaction_construction,
        test_liquid_phase_ideal_solution_matches_analytic,
        test_liquid_phase_unifac_esterification,
        test_liquid_phase_le_chatelier_excess_reactant,
        test_multi_liquid_phase_construction,
        test_multi_liquid_phase_competing_esterifications,
        test_multi_liquid_phase_K_a_consistency,
        # v0.9.65 — Reactive flash (VLE + chemical equilibrium)
        test_reactive_flash_subcooled_matches_no_vle,
        test_reactive_flash_boiling_higher_conversion,
        test_reactive_flash_K_a_matches_K_eq,
        test_reactive_flash_mass_balance,
        test_reactive_flash_no_reactions_pure_vle,
        # v0.9.65 — Reactive distillation column
        test_rd_column_construction,
        test_rd_column_non_reactive_mass_balance,
        test_rd_column_no_reaction_zero_conversion,
        test_rd_column_element_balance,
        test_rd_column_temperature_profile,
        test_rd_column_K_a_at_reactive_stages,
        # v0.9.66 — Naphtali-Sandholm simultaneous Newton solver
        test_rd_column_NS_quadratic_convergence,
        test_rd_column_NS_high_precision_K_a,
        test_rd_column_NS_machine_precision_atom_balance,
        test_rd_column_WH_NS_agree,
        # v0.9.67 — Energy-balance reactive distillation
        test_rd_column_energy_balance_converges,
        test_rd_column_energy_balance_per_stage_closure,
        test_rd_column_energy_balance_breaks_CMO,
        test_rd_column_energy_balance_validation,
        # v0.9.77 — γ-φ-EOS coupling for high-pressure reactive flash
        test_gamma_phi_eos_low_p_matches_modified_raoult,
        test_gamma_phi_eos_high_p_diverges_from_raoult,
        test_reactive_flash_vapor_eos_dispatch,
        test_reactive_flash_vapor_eos_input_validation,
        # v0.9.79 — Direct Gibbs minimization with element constraints
        test_gibbs_min_water_gas_shift_matches_K_eq,
        test_gibbs_min_smr_multi_reaction_no_reactions_specified,
        test_gibbs_min_atom_balance_machine_precision,
        test_gibbs_min_monotone_decrease,
        test_gibbs_min_le_chatelier_pressure,
        test_gibbs_min_n_init_positive_required,
        # v0.9.80 — Solid phases + phase split
        test_gibbs_min_solid_boudouard,
        test_gibbs_min_phase_split_high_T_matches_pure_gas,
        test_gibbs_min_phase_split_two_phase_region,
        test_gibbs_min_phase_split_T_monotone,
        # v0.9.81 — γ-φ-EOS in phase-split Gibbs minimizer
        test_gibbs_min_phase_split_eos_low_p_matches_raoult,
        test_gibbs_min_phase_split_eos_high_p_diverges,
        test_gibbs_min_phase_split_eos_dispatch_unchanged,
        # v0.9.82 — Solid + phase split combined
        test_gibbs_min_phase_split_solid_methane_cracking,
        test_gibbs_min_phase_split_solid_no_coke_high_steam,
        test_gibbs_min_phase_split_solid_phase_per_species_default,
        # v0.9.83 — Active-set reactivation + LL split
        test_gibbs_min_solid_reactivation_methane_cracking,
        test_gibbs_min_LL_split_water_butanol,
        test_gibbs_min_LL_split_x1_x2_collapse_handled,
        # v0.9.84 — VLLE combined (vapor + 2 liquids + chemistry)
        test_gibbs_min_VLL_split_2LL_matches_LL_split,
        test_gibbs_min_VLL_split_all_vapor_at_high_T,
        test_gibbs_min_VLL_split_input_validation,
        # v0.9.85 — 3VLL ternary + 4-phase reactive equilibrium (V+2L+S)
        test_gibbs_min_VLL_split_3VLL_water_acetone_hexane,
        test_gibbs_min_VLLS_split_reduces_to_VLL_no_solids,
        test_gibbs_min_VLLS_split_methane_cracking_reduces_to_VLE_S,
        test_gibbs_min_VLLS_split_combined_3VLL_with_inert_solid,
        # v0.9.86 — full V+L1+L2+S+real chemistry (all phases simultaneously active)
        test_gibbs_min_VLLS_split_full_4phase_real_chemistry,
    ]:
        try:
            fn()
        except Exception as e:
            global _failed
            _failed += 1
            _failures.append(f"{fn.__name__}: {e}")
            print(f"  FAIL  {fn.__name__}: {e}")

    print()
    print("=" * 60)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    if _failures:
        print("\nFailures:")
        for f in _failures:
            print(f"  - {f}")
    print("=" * 60)
    return _failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
