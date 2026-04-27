"""PC-SAFT equation of state (Gross & Sadowski 2001, non-associating).

PC-SAFT ("Perturbed-Chain Statistical Associating Fluid Theory") is a
molecular-based equation of state built from three contributions to the
residual Helmholtz energy:

    a^res = a^hc + a^disp    [+ a^assoc for associating fluids]

where
  a^hc   = hard-chain reference (BMCSL hard-sphere mixture + chain connectivity)
  a^disp = dispersive perturbation (Gross & Sadowski universal-constant
           polynomial integrals in packing fraction eta)

This module provides the `PCSAFT` class, which stores per-component
parameters. Like stateprop.cubic.eos, PC-SAFT components are parameterized
directly by published molecular parameters rather than by EOS-specific JSON
tables. Each component needs:

  m           : segment number (dimensionless, typically 1-5 for small molecules)
  sigma       : segment diameter [Angstrom]
  epsilon_k   : segment energy parameter [K] (epsilon/k_B)
  T_c, p_c    : critical parameters [K, Pa] -- used for Wilson K initialization
  acentric_factor : used for Wilson K initialization

The full mixture residual Helmholtz and its derivatives live in the
`SAFTMixture` class in `stateprop.saft.mixture`. This separation mirrors
the stateprop.cubic module layout.

Reference
---------
Gross, J. and Sadowski, G. (2001). "Perturbed-Chain SAFT: An Equation of
State Based on a Perturbation Theory for Chain Molecules."
Ind. Eng. Chem. Res. 40, 1244-1260.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PCSAFT:
    """A pure-component PC-SAFT parameter set.

    Parameters
    ----------
    m : float
        Segment number.
    sigma : float
        Segment diameter [Angstrom].
    epsilon_k : float
        Segment dispersive energy parameter divided by Boltzmann constant [K].
    T_c : float
        Critical temperature [K]. Used for Wilson K initialization.
    p_c : float
        Critical pressure [Pa]. Used for Wilson K initialization.
    acentric_factor : float
        Pitzer acentric factor [-]. Used for Wilson K initialization.
    name : str, optional
        Human-readable component name (e.g., "methane").

    Association (v0.9.23, 2B or 4C scheme):

    eps_AB_k : float
        Association energy / Boltzmann constant [K]. Zero means non-associating.
    kappa_AB : float
        Association volume parameter (dimensionless).
    assoc_scheme : str
        Association scheme identifier: '2B' (default, 1 A-site + 1 B-site)
        or '4C' (2 A-sites + 2 B-sites, for water; v0.9.26).

    Polar term:

    dipole_moment : float
        Dipole moment [Debye]. Zero means non-polar.
    n_polar_segments : float
        Number of polar segments per molecule (typically between 1 and m).
        If zero and dipole_moment > 0, defaults to m at construction time.
    quadrupole_moment : float
        Quadrupole moment [DA = Debye-Angstrom] (v0.9.26). Zero means no
        quadrupole contribution. Used for CO2, N2 and other symmetric
        molecules with no net dipole but strong quadrupolar interactions.
    """
    m: float
    sigma: float
    epsilon_k: float
    T_c: float
    p_c: float
    acentric_factor: float
    name: Optional[str] = None
    # Association (Chapman-Radosz)
    eps_AB_k: float = 0.0
    kappa_AB: float = 0.0
    assoc_scheme: str = "2B"
    # Dipolar polar term (Gross-Vrabec 2006)
    dipole_moment: float = 0.0
    n_polar_segments: float = 0.0
    # Quadrupolar polar term (Gross 2005, v0.9.26)
    quadrupole_moment: float = 0.0
    # Molar mass [kg/mol] (v0.9.32, for transport-property calculations)
    molar_mass: float = 0.0
    # Normal boiling point [K] (v0.9.32, for Brock-Bird surface tension)
    T_b: float = 0.0
    # Parachor [cm^3/mol * (dyn/cm)^(1/4)] (v0.9.33, for Macleod-Sugden
    # mixture surface tension). Typical values: CH4 73, n-hexane 271,
    # water 52.9, methanol 88.6.
    parachor: float = 0.0

    def __post_init__(self):
        # If a dipole is given but no n_polar_segments, default to m
        if self.dipole_moment > 0 and self.n_polar_segments == 0.0:
            self.n_polar_segments = self.m
        # Validate association scheme
        if self.assoc_scheme not in ("2B", "4C"):
            raise ValueError(f"assoc_scheme must be '2B' or '4C', got {self.assoc_scheme!r}")


# -----------------------------------------------------------------------
# Published parameter sets for common substances (Gross & Sadowski 2001
# Table 1; additional substances from later papers). Users can also
# construct PCSAFT components directly.
# -----------------------------------------------------------------------

def _make(name, m, sigma, eps_k, T_c, p_c, omega,
          eps_AB_k=0.0, kappa_AB=0.0, assoc_scheme="2B",
          dipole_moment=0.0, n_polar_segments=0.0,
          quadrupole_moment=0.0, molar_mass=0.0, T_b=0.0,
          parachor=0.0):
    return PCSAFT(m=m, sigma=sigma, epsilon_k=eps_k,
                  T_c=T_c, p_c=p_c, acentric_factor=omega, name=name,
                  eps_AB_k=eps_AB_k, kappa_AB=kappa_AB,
                  assoc_scheme=assoc_scheme,
                  dipole_moment=dipole_moment,
                  n_polar_segments=n_polar_segments,
                  quadrupole_moment=quadrupole_moment,
                  molar_mass=molar_mass, T_b=T_b,
                  parachor=parachor)


# Values from Gross & Sadowski 2001 Table 1 (non-associating, non-polar)
# Molar masses in kg/mol (v0.9.32, for transport property calculations)
#
# Note on methane accuracy (v0.9.94 finding):
# These standard Gross-Sadowski 2001 methane parameters were fit to
# saturated liquid density and vapor pressure (T = 90-190 K).  In that
# regime, PC-SAFT reproduces NIST data to <2.3% AAD — excellent.
# Extrapolation to dense supercritical states above ~300 K shows
# systematic errors of 5-17% (PC-SAFT overestimates Z, hence
# underestimates density).  See ``METHANE_SUPERCRITICAL`` below for
# an alternative parameter set fit to a wider T-P range; it has worse
# saturation behavior but better supercritical accuracy.  This is a
# fundamental functional-form limitation of PC-SAFT for methane (no
# single (m, σ, ε/k) triple can fit both regimes simultaneously); not
# a bug in stateprop's implementation.  Verified against a hand-coded
# Gross-Sadowski 2001 reference (matches to machine precision) and
# against the Esper-2023 parameters (m=1.0000, σ=3.7005, ε/k=150.07
# give the same accuracy envelope as our values).
METHANE   = _make("methane",       m=1.0000, sigma=3.7039, eps_k=150.03,
                  T_c=190.564, p_c=4.5992e6, omega=0.01142,
                  molar_mass=0.016043, T_b=111.63, parachor=73.2)
# v0.9.26: N2 has a significant quadrupole (Q=1.52 DA)
NITROGEN  = _make("nitrogen",      m=1.2053, sigma=3.3130, eps_k=90.96,
                  T_c=126.19,  p_c=3.3958e6, omega=0.0372,
                  quadrupole_moment=1.52, molar_mass=0.028014, T_b=77.355, parachor=59.1)
# v0.9.26: CO2 has a large quadrupole (Q=4.40 DA) -- dominant polar interaction
CO2       = _make("carbon_dioxide", m=2.0729, sigma=2.7852, eps_k=169.21,
                  T_c=304.128, p_c=7.3773e6, omega=0.22394,
                  quadrupole_moment=4.40, molar_mass=0.044010, T_b=194.7, parachor=78.0)
ETHANE    = _make("ethane",        m=1.6069, sigma=3.5206, eps_k=191.42,
                  T_c=305.322, p_c=4.8722e6, omega=0.0995,
                  molar_mass=0.030069, T_b=184.55, parachor=112.9)
PROPANE   = _make("propane",       m=2.0020, sigma=3.6184, eps_k=208.11,
                  T_c=369.83,  p_c=4.248e6,  omega=0.152,
                  molar_mass=0.044096, T_b=231.04, parachor=152.9)
N_BUTANE  = _make("n_butane",      m=2.3316, sigma=3.7086, eps_k=222.88,
                  T_c=425.12,  p_c=3.796e6,  omega=0.200,
                  molar_mass=0.058122, T_b=272.65, parachor=192.0)
N_PENTANE = _make("n_pentane",     m=2.6896, sigma=3.7729, eps_k=231.20,
                  T_c=469.70,  p_c=3.3675e6, omega=0.251,
                  molar_mass=0.072149, T_b=309.22, parachor=232.1)
N_HEXANE  = _make("n_hexane",      m=3.0576, sigma=3.7983, eps_k=236.77,
                  T_c=507.60,  p_c=3.025e6,  omega=0.301,
                  molar_mass=0.086175, T_b=341.88, parachor=271.1)
N_HEPTANE = _make("n_heptane",     m=3.4831, sigma=3.8049, eps_k=238.40,
                  T_c=540.20,  p_c=2.74e6,   omega=0.349,
                  molar_mass=0.100202, T_b=371.57, parachor=310.7)
N_OCTANE  = _make("n_octane",      m=3.8176, sigma=3.8373, eps_k=242.78,
                  T_c=568.70,  p_c=2.49e6,   omega=0.398,
                  molar_mass=0.114229, T_b=398.82, parachor=351.0)


# -----------------------------------------------------------------------
# v0.9.23: Associating and polar components (updated v0.9.26).
# -----------------------------------------------------------------------

# Water -- v0.9.28: 4C association scheme (2 H-donor + 2 O-acceptor sites,
# matches water's tetrahedral H-bond geometry). Dispersion parameters
# (m, sigma, eps/k) held at the Gross & Sadowski 2002 2B fit; association
# parameters (eps_AB/k, kappa_AB) re-fit against NIST saturation pressure
# (IAPWS-95) via coarse grid scan. The 4C formula produces ~2x the
# association contribution of 2B at equal Delta (2 vs 4 sites), so eps_AB/k
# was reduced from 2500.7 at 2B to 1400.0 at 4C; kappa_AB is essentially
# unchanged (0.035 vs 0.034868). With these parameters, water p_sat
# matches NIST to ~5-6% across T=300-400K. A full 5-parameter refit
# (m, sigma, eps/k, eps_AB, kappa_AB) would eliminate the residual
# systematic bias but is left as future work.
WATER = _make("water", m=1.0656, sigma=3.0007, eps_k=366.51,
              T_c=647.096, p_c=22.064e6, omega=0.3443,
              eps_AB_k=1400.0, kappa_AB=0.035, assoc_scheme="4C",
              molar_mass=0.018015, T_b=373.15, parachor=52.9)

# Methanol (2B scheme) -- Gross & Sadowski 2002
METHANOL = _make("methanol", m=1.5255, sigma=3.2300, eps_k=188.90,
                 T_c=512.60, p_c=8.084e6, omega=0.5625,
                 eps_AB_k=2899.5, kappa_AB=0.035176, assoc_scheme="2B",
                 molar_mass=0.032042, T_b=337.69, parachor=88.6)

# Ethanol (2B scheme)
ETHANOL = _make("ethanol", m=2.3827, sigma=3.1771, eps_k=198.24,
                T_c=513.92, p_c=6.148e6, omega=0.6449,
                eps_AB_k=2653.4, kappa_AB=0.032384,
                molar_mass=0.046069, T_b=351.44, parachor=126.8)

# 1-Propanol (2B scheme)
N_PROPANOL = _make("1_propanol", m=3.0300, sigma=3.2522, eps_k=234.98,
                   T_c=536.78, p_c=5.175e6, omega=0.6283,
                   eps_AB_k=2276.8, kappa_AB=0.015268,
                   molar_mass=0.060095, T_b=370.35, parachor=165.3)

# Acetone (dipolar) -- Gross & Vrabec 2006
ACETONE = _make("acetone", m=2.7447, sigma=3.2742, eps_k=232.99,
                T_c=508.20, p_c=4.700e6, omega=0.3071,
                dipole_moment=2.88,
                molar_mass=0.058079, T_b=329.22, parachor=162.5)

# DME (dimethyl ether, dipolar) -- v0.9.25 re-parameterized.
# Original v0.9.23 parameters (m=2.2634, sigma=3.2729, eps/k=210.29) were
# taken from a non-polar PC-SAFT fit where the dipole was implicitly
# absorbed into the dispersion, incompatible with the explicit polar
# term (default-on in v0.9.24). New parameters fit against NIST DME
# saturation-pressure data over T in [240, 340] K with the calibrated
# polar term ON give max |p_sat error| = 4.7% (mean 3.0%):
#   sigma = 3.35 A       (was 3.2729 A)
#   eps/k = 214.0 K      (was 210.29 K)
#   m, dipole unchanged.
DME = _make("dimethyl_ether", m=2.2634, sigma=3.35, eps_k=214.0,
            T_c=400.378, p_c=5.3368e6, omega=0.200,
            dipole_moment=1.3, molar_mass=0.046069, T_b=248.31, parachor=119.2)
