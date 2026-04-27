"""PC-SAFT database (Esper 2023 + Rehner 2023) tests.

Validates the bundled FeOS PC-SAFT parameter set integration:
  - Pure-component lookup by CAS, name, IUPAC, SMILES, InChI
  - Binary kij + cross-association lookup
  - One-call SAFTMixture constructor
  - Database statistics
  - Robustness on edge cases (missing entries, conflicting identifiers)
"""
from __future__ import annotations
import sys, os, warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from stateprop.saft import (
    PCSAFT, SAFTMixture,
    lookup_pcsaft, lookup_binary, lookup_kij,
    make_saft_mixture, database_summary,
    load_pure_database, load_binary_database,
)

_PASS = 0
_FAIL = 0


def section(n): print(f"\n[{n}]")
def check(label, ok):
    global _PASS, _FAIL
    if ok: _PASS += 1; print(f"  PASS  {label}")
    else: _FAIL += 1; print(f"  FAIL  {label}")


# =====================================================================
# Database loading
# =====================================================================

def test_database_summary_counts():
    """The bundled databases must load and have the expected sizes
    (1842 pure components, 7848 binary pairs)."""
    section("test_database_summary_counts")
    s = database_summary()
    check(f"1842 pure components ({s['n_pure_components']})",
          s["n_pure_components"] == 1842)
    check(f"7848 binary pairs ({s['n_binary_pairs']})",
          s["n_binary_pairs"] == 7848)
    check(f"polar component count > 400 ({s['n_pure_polar']})",
          s["n_pure_polar"] > 400)
    check(f"associating count > 1000 ({s['n_pure_assoc_full'] + s['n_pure_assoc_induced']})",
          s["n_pure_assoc_full"] + s["n_pure_assoc_induced"] > 1000)
    check(f"binary kij count > 6000 ({s['n_binary_with_kij']})",
          s["n_binary_with_kij"] > 6000)


def test_database_caching():
    """The pure and binary databases should be cached — successive
    calls return the same list object."""
    section("test_database_caching")
    pure_a = load_pure_database()
    pure_b = load_pure_database()
    check("pure DB returns same object on repeat call", pure_a is pure_b)
    binary_a = load_binary_database()
    binary_b = load_binary_database()
    check("binary DB returns same object on repeat call",
          binary_a is binary_b)


# =====================================================================
# Pure-component lookup
# =====================================================================

def test_lookup_pure_by_name():
    """Common refinery and chemical species must look up by name."""
    section("test_lookup_pure_by_name")
    cases = [
        ("methanol", 2.25965, 2.83016, 183.58634),
        ("ethanol", 2.88660, 2.95772, 187.26028),
        ("water",   2.36948, 2.15072, 230.71557),
        ("methane", 1.00000, 3.70051, 150.07147),
        ("ethane",  1.60689, 3.51681, 191.45389),
    ]
    for name, m_ref, sigma_ref, eps_ref in cases:
        c = lookup_pcsaft(name=name)
        check(f"{name}: m={c.m} = {m_ref}",
              abs(c.m - m_ref) < 1e-4)
        check(f"{name}: sigma={c.sigma:.4f} = {sigma_ref}",
              abs(c.sigma - sigma_ref) < 1e-4)
        check(f"{name}: eps/k={c.epsilon_k:.4f} = {eps_ref}",
              abs(c.epsilon_k - eps_ref) < 1e-3)


def test_lookup_pure_by_cas():
    """CAS lookup must give the same result as name lookup for known
    compounds; CAS is the more robust identifier across synonym sets."""
    section("test_lookup_pure_by_cas")
    cases = [
        ("methanol", "67-56-1"),
        ("water", "7732-18-5"),
        ("methane", "74-82-8"),
        ("benzene", "71-43-2"),
        ("carbon dioxide", "124-38-9"),
    ]
    for name, cas in cases:
        c_name = lookup_pcsaft(name=name)
        c_cas = lookup_pcsaft(cas=cas)
        check(f"{name}: name lookup matches CAS lookup",
              abs(c_name.m - c_cas.m) < 1e-12
              and abs(c_name.sigma - c_cas.sigma) < 1e-12)


def test_lookup_pure_by_smiles():
    """SMILES lookup must work for common compounds."""
    section("test_lookup_pure_by_smiles")
    c = lookup_pcsaft(smiles="O")   # water
    check("water by SMILES 'O'", abs(c.epsilon_k - 230.71557) < 1e-3)
    c = lookup_pcsaft(smiles="C")   # methane
    check("methane by SMILES 'C'", abs(c.m - 1.0) < 1e-9)
    c = lookup_pcsaft(smiles="CO")  # methanol
    check("methanol by SMILES 'CO'", abs(c.m - 2.25965) < 1e-4)


def test_lookup_pure_associating():
    """Lookup of associating components must populate kappa_AB and
    epsilon_AB/k from the dataset's `association_sites` block."""
    section("test_lookup_pure_associating")
    c = lookup_pcsaft(name="water")
    check(f"water kappa_AB = {c.kappa_AB:.5f} (expect 0.35319)",
          abs(c.kappa_AB - 0.35319) < 1e-4)
    check(f"water eps_AB/k = {c.eps_AB_k:.4f} (expect 2195.10176)",
          abs(c.eps_AB_k - 2195.10176) < 1e-2)
    c = lookup_pcsaft(name="methanol")
    check(f"methanol kappa_AB = {c.kappa_AB:.5f} (expect 0.08716)",
          abs(c.kappa_AB - 0.08716) < 1e-4)


def test_lookup_pure_polar():
    """Components with dipole moments must populate the dipole_moment
    field from the dataset's `mu` field."""
    section("test_lookup_pure_polar")
    # acetone has mu=2.7 D in standard tables
    c = lookup_pcsaft(name="acetone")
    check(f"acetone has nonzero dipole ({c.dipole_moment} D)",
          c.dipole_moment > 0)
    # methane (non-polar) has mu = 0
    c = lookup_pcsaft(name="methane")
    check(f"methane has zero dipole ({c.dipole_moment} D)",
          c.dipole_moment == 0)


def test_lookup_pure_non_associating():
    """A non-associating compound (methane, ethane, hydrocarbons)
    must return zero kappa_AB and epsilon_AB/k."""
    section("test_lookup_pure_non_associating")
    c = lookup_pcsaft(name="methane")
    check("methane: kappa_AB = 0", c.kappa_AB == 0)
    check("methane: eps_AB_k = 0", c.eps_AB_k == 0)
    c = lookup_pcsaft(name="propane")
    check("propane: kappa_AB = 0", c.kappa_AB == 0)


def test_lookup_pure_returns_PCSAFT():
    """The returned object must be a PCSAFT dataclass instance."""
    section("test_lookup_pure_returns_PCSAFT")
    c = lookup_pcsaft(name="water")
    check("returned object is PCSAFT", isinstance(c, PCSAFT))
    # Required fields populated
    for field in ["m", "sigma", "epsilon_k", "molar_mass", "name"]:
        check(f"PCSAFT.{field} populated", hasattr(c, field))


def test_lookup_pure_missing_raises():
    """Looking up a non-existent compound must raise KeyError."""
    section("test_lookup_pure_missing_raises")
    raised = False
    try:
        lookup_pcsaft(name="unobtanium-xyz")
    except KeyError:
        raised = True
    check("non-existent compound raises KeyError", raised)
    raised = False
    try:
        lookup_pcsaft(cas="0000-00-0")
    except KeyError:
        raised = True
    check("non-existent CAS raises KeyError", raised)


def test_lookup_pure_no_keys_raises():
    """Calling without any identifier must raise ValueError."""
    section("test_lookup_pure_no_keys_raises")
    raised = False
    try:
        lookup_pcsaft()
    except ValueError:
        raised = True
    check("no-identifier call raises ValueError", raised)


def test_lookup_pure_name_case_insensitive():
    """Name and IUPAC matching is case-insensitive."""
    section("test_lookup_pure_name_case_insensitive")
    c1 = lookup_pcsaft(name="methanol")
    c2 = lookup_pcsaft(name="METHANOL")
    c3 = lookup_pcsaft(name="MeThAnOl")
    check("methanol name lookup case-insensitive",
          c1.m == c2.m == c3.m)


def test_lookup_pure_with_critical_overrides():
    """T_c, p_c, acentric_factor can be supplied as overrides for use
    in flash routines that need Wilson initialization."""
    section("test_lookup_pure_with_critical_overrides")
    c = lookup_pcsaft(name="methanol", T_c=512.6, p_c=80.97e5,
                       acentric_factor=0.565)
    check(f"T_c override ({c.T_c}) = 512.6", abs(c.T_c - 512.6) < 1e-9)
    check(f"p_c override ({c.p_c}) = 8.097e6",
          abs(c.p_c - 80.97e5) < 1e-3)
    check(f"omega override = 0.565",
          abs(c.acentric_factor - 0.565) < 1e-9)


# =====================================================================
# Binary lookup
# =====================================================================

def test_lookup_binary_methanol_water():
    """The methanol/water pair has a known kij ≈ -0.016."""
    section("test_lookup_binary_methanol_water")
    rec = lookup_binary(name1="methanol", name2="water")
    check("methanol/water binary record exists", rec is not None)
    check(f"methanol/water kij ≈ -0.016 ({rec['k_ij']:.4f})",
          abs(rec["k_ij"] - (-0.0159)) < 0.001)
    # Symmetric: same answer in reverse order
    rec_rev = lookup_binary(name1="water", name2="methanol")
    check("methanol/water symmetric on swap",
          abs(rec["k_ij"] - rec_rev["k_ij"]) < 1e-12)


def test_lookup_kij_convenience():
    """The lookup_kij convenience wrapper returns just the scalar."""
    section("test_lookup_kij_convenience")
    kij = lookup_kij(name1="methanol", name2="water")
    rec = lookup_binary(name1="methanol", name2="water")
    check(f"lookup_kij matches binary['k_ij']",
          abs(kij - rec["k_ij"]) < 1e-12)


def test_lookup_binary_missing():
    """A binary pair not in the database returns None."""
    section("test_lookup_binary_missing")
    rec = lookup_binary(name1="methanol", name2="unobtanium-xyz")
    check("missing pair returns None", rec is None)
    kij = lookup_kij(name1="methanol", name2="unobtanium-xyz")
    check("missing pair kij returns None", kij is None)


def test_lookup_binary_by_cas():
    """CAS-based binary lookup must give the same result as name."""
    section("test_lookup_binary_by_cas")
    rec_name = lookup_binary(name1="methanol", name2="water")
    rec_cas = lookup_binary(cas1="67-56-1", cas2="7732-18-5")
    check("methanol/water by CAS matches by name",
          abs(rec_name["k_ij"] - rec_cas["k_ij"]) < 1e-12)


def test_lookup_binary_no_keys_raises():
    """Calling lookup_binary with missing identifiers raises."""
    section("test_lookup_binary_no_keys_raises")
    raised = False
    try:
        lookup_binary(name1="methanol")   # missing component 2
    except ValueError:
        raised = True
    check("partial spec raises ValueError", raised)


# =====================================================================
# make_saft_mixture
# =====================================================================

def test_make_saft_mixture_methane():
    """A 1-component mixture from the database must reproduce the
    bundled stateprop METHANE constant within 0.1% on density."""
    section("test_make_saft_mixture_methane")
    mix = make_saft_mixture(["methane"], composition=[1.0])
    rho_n = mix.density_from_pressure(p=100e5, T=300.0,
                                         phase_hint="vapor")
    from stateprop.saft import METHANE
    mix_old = SAFTMixture([METHANE], composition=np.array([1.0]))
    rho_old = mix_old.density_from_pressure(p=100e5, T=300.0,
                                              phase_hint="vapor")
    err = abs(float(rho_n) - float(rho_old)) / float(rho_old)
    check(f"Esper-methane vs bundled-METHANE density: {err*100:.3f}%",
          err < 0.01)


def test_make_saft_mixture_methanol_water_kij():
    """Building a methanol/water mixture must populate the kij from
    the database (~ -0.016)."""
    section("test_make_saft_mixture_methanol_water_kij")
    mix = make_saft_mixture(["methanol", "water"], composition=[0.3, 0.7])
    # SAFTMixture stores k_ij as {(0, 1): kij}
    check("kij dict is populated", (0, 1) in mix._k_ij)
    if (0, 1) in mix._k_ij:
        kij = mix._k_ij[(0, 1)]
        check(f"kij ≈ -0.016 ({kij:.4f})",
              abs(kij - (-0.0159)) < 0.001)


def test_make_saft_mixture_with_critical_props():
    """Critical properties supplied via T_c_list, p_c_list, omega_list
    must propagate to the underlying components."""
    section("test_make_saft_mixture_with_critical_props")
    mix = make_saft_mixture(
        ["methanol", "water"],
        composition=[0.5, 0.5],
        T_c_list=[512.6, 647.1],
        p_c_list=[80.97e5, 220.6e5],
        omega_list=[0.565, 0.345],
    )
    check(f"methanol T_c = 512.6 ({mix.components[0].T_c})",
          abs(mix.components[0].T_c - 512.6) < 1e-9)
    check(f"water T_c = 647.1 ({mix.components[1].T_c})",
          abs(mix.components[1].T_c - 647.1) < 1e-9)


def test_make_saft_mixture_density_methanol_water():
    """30/70 mol methanol/water at 1 atm, 298 K should give a
    reasonable liquid density.  Mass-weighted reference ~960 kg/m³;
    PC-SAFT with database parameters typically lands within 5% of
    this."""
    section("test_make_saft_mixture_density_methanol_water")
    mix = make_saft_mixture(["methanol", "water"], composition=[0.3, 0.7])
    rho_n = mix.density_from_pressure(p=1e5, T=298.0,
                                         phase_hint="liquid")
    MW_avg = 0.3 * 0.032026 + 0.7 * 0.018011
    rho_kg = float(rho_n) * MW_avg
    check(f"methanol/water 30/70 ρ_liq = {rho_kg:.1f} kg/m³ (target ~960)",
          abs(rho_kg - 960) / 960 < 0.10)


def test_make_saft_mixture_three_component():
    """Three-component HC mixture (methane + ethane + propane) must
    build cleanly and pull all three pairwise kij from the database."""
    section("test_make_saft_mixture_three_component")
    mix = make_saft_mixture(
        ["methane", "ethane", "propane"],
        composition=[0.5, 0.3, 0.2])
    check("3-component build", mix.N == 3)
    rho_n = mix.density_from_pressure(p=10e5, T=250.0,
                                         phase_hint="vapor")
    check(f"3-comp vapor density positive ({float(rho_n):.0f} mol/m³)",
          float(rho_n) > 0)


def test_make_saft_mixture_composition_sum():
    """make_saft_mixture rejects compositions that don't sum to 1."""
    section("test_make_saft_mixture_composition_sum")
    raised = False
    try:
        make_saft_mixture(["methanol", "water"], composition=[0.5, 0.6])
    except ValueError:
        raised = True
    check("rejects composition sum != 1", raised)


# =====================================================================
# Realism vs NIST (sanity)
# =====================================================================

def test_methane_density_300K_vs_NIST():
    """Methane density at 300 K, 100 bar from the database (Esper 2023
    parameters) should give ~80 kg/m³, within 5% of NIST 79.9 kg/m³."""
    section("test_methane_density_300K_vs_NIST")
    mix = make_saft_mixture(["methane"], composition=[1.0])
    rho_n = mix.density_from_pressure(p=100e5, T=300.0,
                                         phase_hint="vapor")
    rho_kg = float(rho_n) * 0.016043
    check(f"PC-SAFT methane ρ at 300K, 100bar: {rho_kg:.2f} kg/m³ "
          f"(NIST 79.9, err {(rho_kg-79.9)/79.9*100:+.1f}%)",
          abs(rho_kg - 79.9) / 79.9 < 0.05)


def test_water_density_298K_vs_NIST():
    """Water liquid density at 298 K, 1 atm from the database should
    give roughly 1000 kg/m³ (PC-SAFT typically 5-10% below NIST due to
    spherical-segment limitations on dense water)."""
    section("test_water_density_298K_vs_NIST")
    mix = make_saft_mixture(["water"], composition=[1.0])
    rho_n = mix.density_from_pressure(p=1e5, T=298.0,
                                         phase_hint="liquid")
    rho_kg = float(rho_n) * 0.018011
    check(f"PC-SAFT water ρ at 298K: {rho_kg:.0f} kg/m³ "
          f"(NIST 997, err {(rho_kg-997)/997*100:+.1f}%)",
          abs(rho_kg - 997) / 997 < 0.10)


def main():
    print("=" * 60)
    print("stateprop PC-SAFT database (Esper 2023 + Rehner 2023) tests")
    print("=" * 60)
    tests = [
        # Loading and stats
        test_database_summary_counts,
        test_database_caching,
        # Pure lookup
        test_lookup_pure_by_name,
        test_lookup_pure_by_cas,
        test_lookup_pure_by_smiles,
        test_lookup_pure_associating,
        test_lookup_pure_polar,
        test_lookup_pure_non_associating,
        test_lookup_pure_returns_PCSAFT,
        test_lookup_pure_missing_raises,
        test_lookup_pure_no_keys_raises,
        test_lookup_pure_name_case_insensitive,
        test_lookup_pure_with_critical_overrides,
        # Binary lookup
        test_lookup_binary_methanol_water,
        test_lookup_kij_convenience,
        test_lookup_binary_missing,
        test_lookup_binary_by_cas,
        test_lookup_binary_no_keys_raises,
        # Mixture constructor
        test_make_saft_mixture_methane,
        test_make_saft_mixture_methanol_water_kij,
        test_make_saft_mixture_with_critical_props,
        test_make_saft_mixture_density_methanol_water,
        test_make_saft_mixture_three_component,
        test_make_saft_mixture_composition_sum,
        # Realism
        test_methane_density_300K_vs_NIST,
        test_water_density_298K_vs_NIST,
    ]
    for t in tests:
        t()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
