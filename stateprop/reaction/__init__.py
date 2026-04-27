"""Reaction equilibrium calculations (v0.9.61).

Single-reaction ideal-gas equilibrium based on NIST Shomate
thermochemistry. Built around three primitives:

  - SpeciesThermo: Hf_298, Sf_298, Gf_298, and Shomate Cp(T) for one species
  - Reaction: a stoichiometric collection of species; provides
              dH_rxn(T), dG_rxn(T), K_eq(T)
  - equilibrium_extent_ideal_gas(): solves for the reaction extent xi
              at fixed (T, p) with given initial mole numbers

Examples
--------

Water-gas shift CO + H2O = CO2 + H2:

    >>> from stateprop.reaction import Reaction
    >>> rxn = Reaction.from_names(
    ...     reactants={'CO': 1, 'H2O': 1},
    ...     products={'CO2': 1, 'H2': 1})
    >>> rxn.K_eq(800.0)               # ~3.2 (mildly product-favored)
    >>> rxn.dH_rxn(800.0) / 1000      # -41 kJ/mol (exothermic)
    >>> result = rxn.equilibrium_extent_ideal_gas(
    ...     T=800.0, p=10e5,
    ...     n_initial=[1.0, 1.0, 0.0, 0.0])
    >>> result.xi   # extent of reaction
    >>> result.y_eq # equilibrium mole fractions

Methanol synthesis CO + 2 H2 = CH3OH:

    >>> rxn = Reaction.from_names(
    ...     reactants={'CO': 1, 'H2': 2},
    ...     products={'CH3OH': 1})

Bundled species (16): H2O, CO, CO2, H2, N2, O2, CH4, NH3, CH3OH,
C2H4, C2H6, C3H8, NO, NO2, SO2, HCl, HCN.

Out-of-scope (deferred to future versions):
  - Multi-reaction equilibrium (Lagrangian / Gibbs minimization)
  - Real-gas K_eq corrections via fugacity coefficients
  - Reactive distillation (gas-liquid coupled equilibrium)
  - Liquid-phase reactions with activity coefficients
"""
from .thermo import (SpeciesThermo, ShomateCoeffs, BUILTIN_SPECIES,
                     get_species, list_species, R_GAS)
from .equilibrium import Reaction, EquilibriumResult
from .multi import MultiReaction, MultiEquilibriumResult
from .liquid_phase import (LiquidPhaseReaction, LiquidEquilibriumResult,
                            MultiLiquidPhaseReaction,
                            LiquidMultiEquilibriumResult)
from .reactive_flash import reactive_flash_TP, ReactiveFlashResult
from .reactive_column import reactive_distillation_column, ColumnResult
from .gibbs_min import (gibbs_minimize_TP, gibbs_minimize_from_thermo,
                         gibbs_minimize_TP_phase_split,
                         gibbs_minimize_TP_LL_split,
                         gibbs_minimize_TP_VLL_split,
                         gibbs_minimize_TP_VLLS_split,
                         GibbsMinResult, GibbsMinPhaseSplitResult,
                         GibbsMinLLSplitResult, GibbsMinVLLSplitResult,
                         GibbsMin4PhaseSplitResult)


__all__ = [
    "SpeciesThermo",
    "ShomateCoeffs",
    "BUILTIN_SPECIES",
    "get_species",
    "list_species",
    "R_GAS",
    "Reaction",
    "EquilibriumResult",
    "MultiReaction",
    "MultiEquilibriumResult",
    "LiquidPhaseReaction",
    "LiquidEquilibriumResult",
    "MultiLiquidPhaseReaction",
    "LiquidMultiEquilibriumResult",
    "reactive_flash_TP",
    "ReactiveFlashResult",
    "reactive_distillation_column",
    "ColumnResult",
    "gibbs_minimize_TP",
    "gibbs_minimize_from_thermo",
    "gibbs_minimize_TP_phase_split",
    "gibbs_minimize_TP_LL_split",
    "gibbs_minimize_TP_VLL_split",
    "gibbs_minimize_TP_VLLS_split",
    "GibbsMinResult",
    "GibbsMinPhaseSplitResult",
    "GibbsMinLLSplitResult",
    "GibbsMinVLLSplitResult",
    "GibbsMin4PhaseSplitResult",
]
