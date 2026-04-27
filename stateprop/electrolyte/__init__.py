"""Electrolyte solution thermodynamics (v0.9.96).

Models for activity coefficients and osmotic properties of aqueous
electrolyte solutions:

    * `PitzerModel`  — Pitzer ion-interaction model (1973, 1991), the
      standard for aqueous electrolytes. Bundled parameter sets for 18
      common salts at 25 °C.

    * `eNRTL`        — electrolyte NRTL (Chen 1982). Bundled
      parameters for 7 common salts at 25 °C.

    * Utilities      — Debye-Hückel A coefficient, Davies equation,
      ionic strength, molality ↔ mole-fraction conversions.

Quick start
-----------

    >>> from stateprop.electrolyte import PitzerModel
    >>> nacl = PitzerModel("NaCl")
    >>> nacl.gamma_pm(molality=1.0, T=298.15)
    0.6577...
    >>> nacl.osmotic_coefficient(molality=1.0)
    0.9355...
    >>> nacl.water_activity(molality=1.0)
    0.96686...

Convention notes
----------------
* Molality m is in mol/(kg solvent), the standard electrolyte basis.
* Charges z are signed integers (e.g. z_Na = +1, z_Cl = -1).
* Stoichiometric coefficients ν_M, ν_X are the cation/anion counts in
  the salt formula (NaCl: ν_M = ν_X = 1; CaCl2: ν_M = 1, ν_X = 2).
* The mean ionic activity coefficient γ_± is defined per
  γ_±^ν = γ_M^ν_M · γ_X^ν_X with ν = ν_M + ν_X.

References
----------
* Pitzer, K. S. (1991). *Activity Coefficients in Electrolyte
  Solutions* (2nd ed., CRC Press).
* Chen, C.-C. et al. (1982). Local composition model for excess
  Gibbs energy of electrolyte systems. AIChE J. 28, 588.
* Robinson, R. A., Stokes, R. H. (1959). *Electrolyte Solutions*
  (2nd ed., Butterworths). Tabulated γ± and φ data.
"""
from .utils import (
    ionic_strength,
    molality_to_mole_fraction,
    mole_fraction_to_molality,
    water_density,
    water_dielectric,
    debye_huckel_A,
    davies_log_gamma_pm,
    debye_huckel_log_gamma_pm,
)
from .pitzer import (
    PitzerSalt, PitzerModel,
    lookup_salt, list_salts,
    lookup_salt_high_T, list_salts_high_T,    # v0.9.116
)
from .enrtl import (
    eNRTLSalt, eNRTL,
    lookup_enrtl,
)
from .multi_pitzer import (
    MultiPitzerSystem, IonInfo,
    E_theta, MixingParam,
)
from .minerals import (
    Mineral, MineralSystem,
    lookup_mineral, list_minerals,
    saturation_index, solubility_in_water,
)
from .complexation import (
    Complex, Speciation, SpeciationResult,
    lookup_complex, list_complexes, solve_speciation,
)
from .amines import (
    Amine, AmineSystem, AmineSpeciationResult,
    lookup_amine, list_amines,
)
from .amine_column import (
    AmineColumn, AmineColumnResult,
    amine_equilibrium_curve,
)
from .amine_stripper import (
    AmineStripper, AmineStripperResult,
    P_water_sat,
    StripperCondenser, StripperCondenserResult,
    stripper_with_condenser,
)
from .heat_exchanger import (
    CrossHeatExchanger, CrossExchangerResult,
    lean_rich_exchanger,
)
from .flowsheet import (
    CaptureFlowsheet, CaptureFlowsheetResult,
)
from . import sour_water
from .sour_water_column import (
    SourWaterActivityModel,
    sour_water_stripper, SourWaterStripperResult,
    build_psat_funcs, build_enthalpy_funcs,
)
from .sour_water_flowsheet import (
    sour_water_two_stage_flowsheet, SourWaterFlowsheetResult,
    find_acid_dose_for_h2s_recovery,
)
from .amine_column_ns import (
    AmineActivityModel, AmineNSResult,
    amine_stripper_ns, amine_absorber_ns,
    build_amine_psat_funcs, build_amine_enthalpy_funcs,
)

__all__ = [
    # Utilities
    "ionic_strength",
    "molality_to_mole_fraction", "mole_fraction_to_molality",
    "water_density", "water_dielectric", "debye_huckel_A",
    "davies_log_gamma_pm", "debye_huckel_log_gamma_pm",
    # Pitzer
    "PitzerSalt", "PitzerModel",
    "lookup_salt", "list_salts",
    "lookup_salt_high_T", "list_salts_high_T",     # v0.9.116
    # Multi-Pitzer (v0.9.98)
    "MultiPitzerSystem", "IonInfo", "E_theta", "MixingParam",
    # Mineral solubility (v0.9.101)
    "Mineral", "MineralSystem",
    "lookup_mineral", "list_minerals",
    "saturation_index", "solubility_in_water",
    # Aqueous complexation (v0.9.102)
    "Complex", "Speciation", "SpeciationResult",
    "lookup_complex", "list_complexes", "solve_speciation",
    # Amine carbamate equilibria (v0.9.103)
    "Amine", "AmineSystem", "AmineSpeciationResult",
    "lookup_amine", "list_amines",
    # Amine column reactive absorber (v0.9.104)
    "AmineColumn", "AmineColumnResult", "amine_equilibrium_curve",
    # Reactive stripper / heat balance (v0.9.105)
    "AmineStripper", "AmineStripperResult", "P_water_sat",
    # Lean-rich heat exchanger (v0.9.106)
    "CrossHeatExchanger", "CrossExchangerResult", "lean_rich_exchanger",
    # Coupled T-solver + stripper condenser (v0.9.107)
    "StripperCondenser", "StripperCondenserResult", "stripper_with_condenser",
    # Capture flowsheet integrator (v0.9.108)
    "CaptureFlowsheet", "CaptureFlowsheetResult",
    # eNRTL
    "eNRTLSalt", "eNRTL", "lookup_enrtl",
    # Sour water
    "sour_water",
    # Sour water stripper column (v0.9.111)
    "SourWaterActivityModel", "sour_water_stripper",
    "SourWaterStripperResult", "build_psat_funcs", "build_enthalpy_funcs",
    # Two-stage sour-water flowsheet (v0.9.113)
    "sour_water_two_stage_flowsheet", "SourWaterFlowsheetResult",
    "find_acid_dose_for_h2s_recovery",
    # Amine column via N-S solver (v0.9.114)
    "AmineActivityModel", "AmineNSResult",
    "amine_stripper_ns", "amine_absorber_ns",
    "build_amine_psat_funcs", "build_amine_enthalpy_funcs",
]
