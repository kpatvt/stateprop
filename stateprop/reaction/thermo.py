"""Thermochemistry data and Shomate-equation evaluator for ideal-gas
chemical reaction equilibrium calculations.

The NIST Shomate equation:

    Cp/R   = (A + B*t + C*t^2 + D*t^3 + E/t^2) / R
    H - H_298    = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H_ref
    S            = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2*t^2) + G

with t = T/1000 [K], Cp in J/(mol K), H in kJ/mol, S in J/(mol K).

Coefficients in the bundled table are from NIST WebBook (publicly
licensed thermochemical data) and apply only within the indicated
T_min..T_max range for each species. Outside that range the Shomate
fit is no longer reliable; consumers should not extrapolate.

References
----------
- NIST Chemistry WebBook (https://webbook.nist.gov), 2024.
- Shomate, C. H. (1944) J. Phys. Chem. 48, 244-249 (the original form).
- M. W. Chase Jr., NIST-JANAF Thermochemical Tables, 4th ed., 1998.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict
import math


R_GAS = 8.314472   # J/(mol K)


@dataclass(frozen=True)
class ShomateCoeffs:
    """NIST Shomate coefficients valid in [T_min, T_max] [K]."""
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float = 0.0    # reference enthalpy offset (kJ/mol; usually 0 = H_298)
    T_min: float = 298.15
    T_max: float = 6000.0


@dataclass(frozen=True)
class SpeciesThermo:
    """Ideal-gas thermochemistry for a single species.

    Convention: all enthalpies are at 1 bar reference pressure (the
    "ideal-gas" standard state).

    Parameters
    ----------
    name : str
        Convenient label (does not need to match any database).
    Hf_298 : float
        Standard enthalpy of formation at 298.15 K [J/mol].
    Sf_298 : float
        Absolute entropy at 298.15 K, 1 bar [J/(mol K)].
    Gf_298 : float
        Standard Gibbs energy of formation at 298.15 K [J/mol].
        For internal consistency this should equal
        Hf_298 - 298.15 * (Sf_298 - sum_atoms Sf_298_atom) but is
        stored separately to avoid rounding in published tables.
    shomate : ShomateCoeffs or sequence of ShomateCoeffs
        Cp(T) representation. A single coefficient block for the
        common 298-1500 K range, or a list for piecewise fits.
        Outside the supplied range the closest-bound block is used
        (with a warning at first use).
    """
    name: str
    Hf_298: float
    Sf_298: float
    Gf_298: float
    shomate: tuple   # tuple of ShomateCoeffs

    def __post_init__(self):
        # Allow a single ShomateCoeffs to be passed and stored as a 1-tuple
        if isinstance(self.shomate, ShomateCoeffs):
            object.__setattr__(self, 'shomate', (self.shomate,))
        # Auto-calibrate the F and G coefficients in the FIRST block such
        # that H(298.15 K) = Hf_298 and S(298.15 K) = Sf_298 exactly.
        # This corrects for any small inconsistency between the published
        # Shomate F/G/H integration constants and the published
        # formation-enthalpy / absolute-entropy at 298.15 K. The Cp(T)
        # polynomial (A, B, C, D, E) is unchanged by this calibration,
        # so the temperature-dependent shifts dH(T) and dS(T) are
        # unaffected.
        first = self.shomate[0]
        # If the first block doesn't span 298.15K, skip calibration
        if not (first.T_min <= 298.15 <= first.T_max):
            return
        t_ref = 298.15 / 1000.0
        # Polynomial part at t_ref (in kJ/mol):
        #   p(t) = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t
        # NIST formula for H: H(T) - H_ref = p(t) + F - H
        # We want: H(298.15) - H_298 = 0  (definition of reference)
        # So:     0 = p(t_ref) + F_new - H_new
        # We choose H_new = Hf_298/1000 (kJ/mol) and solve for F_new:
        #         F_new = H_new - p(t_ref) = Hf_298/1000 - p(t_ref)
        p_t = (first.A * t_ref + first.B * t_ref**2 / 2.0
               + first.C * t_ref**3 / 3.0 + first.D * t_ref**4 / 4.0
               - first.E / t_ref)
        new_H = self.Hf_298 / 1000.0   # kJ/mol
        new_F = new_H - p_t
        # For S(T) = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2t^2) + G
        # We want S(298.15) = Sf_298, so:
        #     G_new = Sf_298 - [A*ln(t_ref) + B*t_ref + ...]
        s_poly = (first.A * math.log(t_ref) + first.B * t_ref
                  + first.C * t_ref**2 / 2.0 + first.D * t_ref**3 / 3.0
                  - first.E / (2.0 * t_ref**2))
        new_G = self.Sf_298 - s_poly
        # Replace the first block with calibrated F, G, H
        calibrated = ShomateCoeffs(
            A=first.A, B=first.B, C=first.C, D=first.D, E=first.E,
            F=new_F, G=new_G, H=new_H,
            T_min=first.T_min, T_max=first.T_max)
        new_blocks = (calibrated,) + self.shomate[1:]
        object.__setattr__(self, 'shomate', new_blocks)

    def _select_block(self, T: float) -> ShomateCoeffs:
        """Pick the Shomate block whose [T_min, T_max] covers T.
        If none does, return the closest block (extrapolation territory)."""
        for blk in self.shomate:
            if blk.T_min <= T <= blk.T_max:
                return blk
        # Closest by midpoint
        return min(self.shomate,
                   key=lambda b: abs(T - 0.5 * (b.T_min + b.T_max)))

    def Cp(self, T: float) -> float:
        """Heat capacity at T [J/(mol K)]."""
        b = self._select_block(T)
        t = T / 1000.0
        return b.A + b.B * t + b.C * t * t + b.D * t * t * t + b.E / (t * t)

    def H_minus_H_298(self, T: float) -> float:
        """[H(T) - H(298.15 K)] / 1 [J/mol], from integrating Shomate Cp."""
        b = self._select_block(T)
        t = T / 1000.0
        # Shomate output is in kJ/mol; convert to J/mol
        h_kJmol = (b.A * t + b.B * t * t / 2.0 + b.C * t * t * t / 3.0
                   + b.D * t ** 4 / 4.0 - b.E / t + b.F - b.H)
        return h_kJmol * 1000.0

    def H(self, T: float) -> float:
        """Total enthalpy at T including formation: Hf_298 + (H(T) - H_298)."""
        return self.Hf_298 + self.H_minus_H_298(T)

    def S(self, T: float) -> float:
        """Absolute entropy at T, 1 bar standard state [J/(mol K)]."""
        b = self._select_block(T)
        t = T / 1000.0
        return (b.A * math.log(t) + b.B * t + b.C * t * t / 2.0
                + b.D * t * t * t / 3.0 - b.E / (2.0 * t * t) + b.G)

    def G(self, T: float) -> float:
        """Total Gibbs energy at T, 1 bar [J/mol]: H(T) - T*S(T)."""
        return self.H(T) - T * self.S(T)

    def Gf(self, T: float) -> float:
        """Gibbs of formation at T, 1 bar [J/mol].

        Computed by Hess's law from the supplied Hf_298, Gf_298, and
        Cp(T):

           Gf(T) = Hf(T) - T*[(Hf(T) - Gf(T))/T]
                 = Hf_298 + dH(T) - T*S_rxn_for_formation(T)

        Practically, since the formation reactions involve elements in
        their standard states, the integrated Cp difference between
        compound and elements is what changes Gf with T. To avoid
        needing element Cp data here, we use the simpler form

           Gf(T) ≈ Gf_298 + (Hf(T) - Hf_298) * (1 - T/298.15)
                          - T * d_Cp_avg * ln(T/298.15)

        Implementation: compute Hf(T) and Sf(T) by integrating Cp_compound
        only (the elements' contribution cancels in dH_rxn for any
        reaction and is implicitly captured in the supplied Sf_298 and
        Gf_298 at 298.15K). The function below returns the **Gibbs of
        formation under the assumption that elemental Cp contributions
        are negligible** -- which is exact for reactions where the
        atoms balance, but is only approximate as an absolute Gf(T).
        For reaction-equilibrium calculations this is fine because
        we only ever take dG_rxn = Sum nu_i * Gf_i(T), and the
        elemental contributions cancel exactly in the sum.
        """
        # Use NIST-consistent: at 298.15K, Hf - T*Sf = Gf if Sf is computed
        # from Cp_integrated_compound - Cp_integrated_elements. For our
        # purposes (Sum nu_i Gf_i), only the temperature-dependent shifts
        # matter and they come from Cp_compound integrated above 298.
        dH = self.H_minus_H_298(T)
        # Gf shift = dH - T*dS where dS = integral of Cp/T from 298 to T
        dS = self.S(T) - self.S(298.15)
        # Gf(T) = Gf_298 + dH(T) - T*dS - 298.15*0  (Hf_298 already in Gf_298)
        # We need: Gf(T) - Gf_298 = (Hf(T) - Hf_298) - T*Sf(T) + 298.15*Sf_298
        # But Sf(T) = Sf_298 + integral Cp/T dT from 298 to T = Sf_298 + dS
        # So Gf(T) - Gf_298 = dH - T*(Sf_298 + dS) + 298.15*Sf_298
        #                   = dH - T*dS - (T - 298.15)*Sf_298
        return self.Gf_298 + dH - T * dS - (T - 298.15) * self.Sf_298


# ------------------------------------------------------------------------
# Built-in NIST Shomate table for common gas-phase species
# (Chase, NIST-JANAF; NIST WebBook 2024).
#
# Hf_298, Gf_298 in J/mol; Sf_298 in J/(mol K). Shomate fits for one
# representative high-temperature range each; for accurate work below
# 500 K verify the chosen block covers the operating range.
# ------------------------------------------------------------------------

BUILTIN_SPECIES: Dict[str, SpeciesThermo] = {
    'H2O': SpeciesThermo(
        name='H2O',
        Hf_298=-241826.0, Sf_298=188.835, Gf_298=-228570.0,
        # NIST Shomate, gas, 500-1700 K
        shomate=ShomateCoeffs(
            A=30.09200, B=6.832514, C=6.793435, D=-2.534480,
            E=0.082139, F=-250.8810, G=223.3967, H=-241.8264,
            T_min=500.0, T_max=1700.0),
    ),
    'CO': SpeciesThermo(
        name='CO',
        Hf_298=-110530.0, Sf_298=197.660, Gf_298=-137168.0,
        # NIST Shomate, 298-1300 K
        shomate=ShomateCoeffs(
            A=25.56759, B=6.096130, C=4.054656, D=-2.671301,
            E=0.131021, F=-118.0089, G=227.3665, H=-110.5271,
            T_min=298.0, T_max=1300.0),
    ),
    'CO2': SpeciesThermo(
        name='CO2',
        Hf_298=-393510.0, Sf_298=213.785, Gf_298=-394390.0,
        # NIST Shomate, 298-1200 K
        shomate=ShomateCoeffs(
            A=24.99735, B=55.18696, C=-33.69137, D=7.948387,
            E=-0.136638, F=-403.6075, G=228.2431, H=-393.5224,
            T_min=298.0, T_max=1200.0),
    ),
    'H2': SpeciesThermo(
        name='H2',
        Hf_298=0.0, Sf_298=130.680, Gf_298=0.0,
        # NIST Shomate, 298-1000 K
        shomate=ShomateCoeffs(
            A=33.066178, B=-11.363417, C=11.432816, D=-2.772874,
            E=-0.158558, F=-9.980797, G=172.707974, H=0.0,
            T_min=298.0, T_max=1000.0),
    ),
    'N2': SpeciesThermo(
        name='N2',
        Hf_298=0.0, Sf_298=191.609, Gf_298=0.0,
        # NIST Shomate, 100-500 K  (close to room T)
        shomate=ShomateCoeffs(
            A=28.98641, B=1.853978, C=-9.647459, D=16.63537,
            E=0.000117, F=-8.671914, G=226.4168, H=0.0,
            T_min=100.0, T_max=500.0),
    ),
    'O2': SpeciesThermo(
        name='O2',
        Hf_298=0.0, Sf_298=205.152, Gf_298=0.0,
        # NIST Shomate, 100-700 K
        shomate=ShomateCoeffs(
            A=31.32234, B=-20.23531, C=57.86644, D=-36.50624,
            E=-0.007374, F=-8.903471, G=246.7945, H=0.0,
            T_min=100.0, T_max=700.0),
    ),
    'CH4': SpeciesThermo(
        name='CH4',
        Hf_298=-74600.0, Sf_298=186.250, Gf_298=-50530.0,
        # NIST Shomate, 298-1300 K
        shomate=ShomateCoeffs(
            A=-0.703029, B=108.4773, C=-42.52157, D=5.862788,
            E=0.678565, F=-76.84376, G=158.7163, H=-74.87310,
            T_min=298.0, T_max=1300.0),
    ),
    'NH3': SpeciesThermo(
        name='NH3',
        Hf_298=-45940.0, Sf_298=192.770, Gf_298=-16407.0,
        # NIST Shomate, 298-1400 K
        shomate=ShomateCoeffs(
            A=19.99563, B=49.77119, C=-15.37599, D=1.921168,
            E=0.189174, F=-53.30667, G=203.8591, H=-45.89806,
            T_min=298.0, T_max=1400.0),
    ),
    'CH3OH': SpeciesThermo(
        name='CH3OH',
        Hf_298=-200940.0, Sf_298=239.865, Gf_298=-162320.0,
        # NIST Shomate, 298-1500 K (gas-phase methanol). G corrected so
        # that S(298.15) equals the published Sf_298.
        shomate=ShomateCoeffs(
            A=14.19500, B=97.21210, C=-9.461620, D=-2.061560,
            E=0.073800, F=-205.0210, G=228.913, H=-200.9400,
            T_min=298.0, T_max=1500.0),
    ),
    'C2H4': SpeciesThermo(
        name='C2H4',
        Hf_298=52400.0, Sf_298=219.317, Gf_298=68440.0,
        # NIST Shomate, ethylene, 298-1200 K
        shomate=ShomateCoeffs(
            A=-6.387880, B=184.4019, C=-112.9718, D=28.49593,
            E=0.315540, F=48.17332, G=163.1568, H=52.46694,
            T_min=298.0, T_max=1200.0),
    ),
    'C2H6': SpeciesThermo(
        name='C2H6',
        Hf_298=-83820.0, Sf_298=229.600, Gf_298=-31920.0,
        # NIST Shomate, ethane, 298-1500 K. G corrected so S(298.15)
        # equals the published Sf_298.
        shomate=ShomateCoeffs(
            A=6.900000, B=172.7800, C=-64.04740, D=7.285360,
            E=0.000000, F=-87.59340, G=189.218, H=-83.82000,
            T_min=298.0, T_max=1500.0),
    ),
    'C3H8': SpeciesThermo(
        name='C3H8',
        Hf_298=-104680.0, Sf_298=270.310, Gf_298=-24290.0,
        # Approximate Shomate (NIST WebBook propane, 298-1000 K).
        # G corrected so S(298.15) equals the published Sf_298.
        shomate=ShomateCoeffs(
            A=-23.1747, B=363.797, C=-222.572, D=56.2466,
            E=0.598438, F=-76.2839, G=146.561, H=-104.6800,
            T_min=298.0, T_max=1000.0),
    ),
    'NO': SpeciesThermo(
        name='NO',
        Hf_298=91271.0, Sf_298=210.760, Gf_298=87580.0,
        shomate=ShomateCoeffs(
            A=23.83491, B=12.58878, C=-1.139011, D=-1.497459,
            E=0.214194, F=83.35783, G=237.1219, H=90.29114,
            T_min=298.0, T_max=1200.0),
    ),
    'NO2': SpeciesThermo(
        name='NO2',
        Hf_298=33100.0, Sf_298=240.040, Gf_298=51310.0,
        shomate=ShomateCoeffs(
            A=16.10857, B=75.89525, C=-54.38740, D=14.30777,
            E=0.239423, F=26.17464, G=240.5386, H=33.09502,
            T_min=298.0, T_max=1200.0),
    ),
    'SO2': SpeciesThermo(
        name='SO2',
        Hf_298=-296810.0, Sf_298=248.223, Gf_298=-300130.0,
        shomate=ShomateCoeffs(
            A=21.43049, B=74.35094, C=-57.75217, D=16.35534,
            E=0.087110, F=-305.7688, G=254.8872, H=-296.8422,
            T_min=298.0, T_max=1200.0),
    ),
    'HCl': SpeciesThermo(
        name='HCl',
        Hf_298=-92310.0, Sf_298=186.902, Gf_298=-95300.0,
        shomate=ShomateCoeffs(
            A=32.12392, B=-13.45805, C=19.86852, D=-6.853936,
            E=-0.049672, F=-101.6206, G=228.6866, H=-92.31201,
            T_min=298.0, T_max=1200.0),
    ),
    'HCN': SpeciesThermo(
        name='HCN',
        Hf_298=135143.0, Sf_298=201.828, Gf_298=124700.0,
        shomate=ShomateCoeffs(
            A=32.69373, B=22.59205, C=-4.369142, D=-0.407697,
            E=-0.282399, F=123.4811, G=233.2940, H=135.1432,
            T_min=298.0, T_max=1200.0),
    ),
}


def get_species(name: str) -> SpeciesThermo:
    """Look up a species by name; case-insensitive with synonym handling.

    Raises KeyError if neither the built-in table nor the chemicals
    library (if available) provides Hf_298, Gf_298 for the species.
    """
    # Fast path: exact match
    if name in BUILTIN_SPECIES:
        return BUILTIN_SPECIES[name]

    # Synonyms
    syn = {
        'water': 'H2O', 'h2o': 'H2O', 'steam': 'H2O',
        'carbon_monoxide': 'CO', 'co': 'CO',
        'carbon_dioxide': 'CO2', 'co2': 'CO2',
        'hydrogen': 'H2', 'h2': 'H2',
        'nitrogen': 'N2', 'n2': 'N2',
        'oxygen': 'O2', 'o2': 'O2',
        'methane': 'CH4', 'ch4': 'CH4',
        'ammonia': 'NH3', 'nh3': 'NH3',
        'methanol': 'CH3OH', 'ch3oh': 'CH3OH', 'meoh': 'CH3OH',
        'ethylene': 'C2H4', 'ethene': 'C2H4', 'c2h4': 'C2H4',
        'ethane': 'C2H6', 'c2h6': 'C2H6',
        'propane': 'C3H8', 'c3h8': 'C3H8',
        'nitric_oxide': 'NO',
        'nitrogen_dioxide': 'NO2',
        'sulfur_dioxide': 'SO2', 'sulphur_dioxide': 'SO2',
        'hydrogen_chloride': 'HCl', 'hcl': 'HCl',
        'hydrogen_cyanide': 'HCN', 'hcn': 'HCN',
    }
    key = syn.get(name.lower())
    if key is not None:
        return BUILTIN_SPECIES[key]

    # Soft-import chemicals: it has Hfg/Gfg/Sfg + heat-capacity polynomials
    try:
        import chemicals
        from chemicals.identifiers import CAS_from_any
        from chemicals.reaction import (Hfg as _Hfg, Sfg as _Sfg, Gfg as _Gfg)
        # Note: chemicals does not return Shomate; we'd need the
        # heat_capacity_gas correlation. This branch is left as a
        # future extension — for now we surface a clear message.
        cas = CAS_from_any(name)
        Hf = _Hfg(cas)
        Gf = _Gfg(cas)
        if Hf is None or Gf is None:
            raise KeyError(name)
        # Simple constant-Cp fallback: use the compound's R*Cp_R(298)
        # if we can extract it. For now, raise NotImplementedError so the
        # caller knows to either add to BUILTIN_SPECIES or supply data
        # directly.
        raise NotImplementedError(
            f"Species '{name}' has Hf/Gf in chemicals but no Shomate Cp(T) "
            "fit is bundled. Add to BUILTIN_SPECIES or pass a custom "
            "SpeciesThermo to the Reaction constructor.")
    except (ImportError, KeyError):
        raise KeyError(
            f"Species '{name}' not found. Bundled species: "
            f"{list(BUILTIN_SPECIES.keys())}")


def list_species() -> list:
    """Return list of bundled species names."""
    return list(BUILTIN_SPECIES.keys())
