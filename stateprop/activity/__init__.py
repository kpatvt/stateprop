"""Activity-coefficient models for non-ideal liquid mixtures (v0.9.39).

Three classical models:

- NRTL (Non-Random Two-Liquid, Renon 1968)
- UNIQUAC (Universal Quasi-Chemical, Abrams-Prausnitz 1975)
- UNIFAC (UNIQUAC Functional-group Activity Coefficients,
  Fredenslund-Jones-Prausnitz 1975; group-contribution method)

These are gE-models for the liquid phase: given mole fractions x and
temperature T, they return activity coefficients gamma_i that
characterize departure from Raoult's law. Used in gamma-phi flash
formulations for highly non-ideal mixtures (alcohols + water,
distillation azeotropes, LLE, etc.) where SAFT/cubic EOS alone are
either unreliable or require parameter regression that gE models
sidestep.

Each model implements a common interface:

    model.gammas(T, x)              -> ndarray of activity coefficients
    model.lngammas(T, x)            -> ndarray of ln(gamma_i)
    model.gE_over_RT(T, x)          -> excess Gibbs / RT (scalar)

For UNIFAC, the constructor takes a subgroups specification per
component plus access to a tabulated database of subgroup R, Q
values and main-group interaction parameters. A small starter
database is provided in `stateprop.activity.unifac_database`;
users can extend it with additional groups as needed.
"""

from .nrtl import NRTL
from .uniquac import UNIQUAC
from .unifac import UNIFAC
from .unifac_dortmund import UNIFAC_Dortmund
from .unifac_lyngby import UNIFAC_Lyngby
from .unifac_lle import UNIFAC_LLE, make_lle_database, LLE_OVERRIDES
from .lle_coverage import (lle_coverage, lle_coverage_summary,
                              CoverageReport, validate_against_benchmarks,
                              format_benchmark_results, BenchmarkResult,
                              save_overrides_to_json,
                              load_overrides_from_json,
                              LLE_BENCHMARKS)
from .gamma_phi import (GammaPhiFlash, AntoinePsat,
                          BubbleResult, DewResult, FlashResult)
from .gamma_phi_eos import GammaPhiEOSFlash, make_phi_sat_funcs
from .gamma_phi_eos_3phase import (GammaPhiEOSThreePhaseFlash,
                                     ThreePhaseFlashResult,
                                     SinglePhaseResult, AutoFlashResult)
from .lle import LLEFlash, LLEResult
from .stability import stability_test, StabilityResult
from .vapor_stability import vapor_phase_stability_test, VaporStabilityResult
from .cross_stability import (cross_phase_stability_test,
                                CrossPhaseStabilityResult)
from . import batch
from . import regression
from . import compounds

__all__ = ['NRTL', 'UNIQUAC', 'UNIFAC',
           'UNIFAC_Dortmund', 'UNIFAC_Lyngby', 'UNIFAC_LLE',
           'make_lle_database', 'LLE_OVERRIDES',
           'lle_coverage', 'lle_coverage_summary', 'CoverageReport',
           'validate_against_benchmarks', 'format_benchmark_results',
           'BenchmarkResult', 'save_overrides_to_json',
           'load_overrides_from_json', 'LLE_BENCHMARKS',
           'GammaPhiFlash', 'GammaPhiEOSFlash', 'make_phi_sat_funcs',
           'GammaPhiEOSThreePhaseFlash', 'ThreePhaseFlashResult',
           'SinglePhaseResult', 'AutoFlashResult',
           'LLEFlash', 'LLEResult',
           'stability_test', 'StabilityResult',
           'vapor_phase_stability_test', 'VaporStabilityResult',
           'cross_phase_stability_test', 'CrossPhaseStabilityResult',
           'AntoinePsat',
           'BubbleResult', 'DewResult', 'FlashResult',
           'batch', 'regression', 'compounds']
