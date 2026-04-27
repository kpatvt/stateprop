"""Compressed natural gas storage: real-gas behavior of methane.

What this demonstrates
----------------------
Methane in a CNG storage tank departs strongly from ideal-gas behavior
above ~50 bar.  Storage capacity, energy content, and compression work
all depend on the real-gas equation of state.

This example sweeps methane density from 1 bar through 250 bar at
ambient temperature using the Setzmann-Wagner 1991 reference EOS,
then computes:

- Storage density (kg/m³) and the gain over ideal-gas
- Compressibility factor Z = pv/RT vs pressure
- Energy required to compress from atmospheric to storage pressure
  (isothermal — a lower bound on compressor duty)
- Mass-equivalent volume (1 m³ at 250 bar holds how many m³ of gas?)

We compare against ideal-gas predictions and NIST reference points.

Reference
---------
Setzmann, U.; Wagner, W. (1991). A new equation of state and tables
of thermodynamic properties for methane covering the range from the
melting line to 625 K at pressures up to 1000 MPa.  J. Phys. Chem.
Ref. Data 20, 1061-1155.

NIST Chemistry WebBook (NIST Standard Reference Database 69) —
methane reference points at 298.15 K.

Approximate runtime: ~2 seconds.

Public APIs invoked
-------------------
- stateprop.load_fluid
- stateprop.density_from_pressure
- stateprop.enthalpy, stateprop.entropy
- stateprop.compressibility_factor

"""
import sys
sys.path.insert(0, '.')

import numpy as np

from stateprop import (
    load_fluid, density_from_pressure,
    enthalpy, entropy, compressibility_factor,
)
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Compressed natural gas storage — methane real-gas behavior")
print("=" * 70)
print()

methane = load_fluid("methane")
T_AMB = 298.15  # 25 °C
M_CH4 = methane.molar_mass   # kg/mol
R = 8.314462618

# Pressure range: 1 bar to 250 bar (typical CNG tank conditions)
pressures_bar = np.array([1, 5, 10, 20, 50, 100, 150, 200, 250])
pressures = pressures_bar * 1e5

print(f"  Methane reference EOS: Setzmann-Wagner 1991")
print(f"  Conditions: T = {T_AMB-273.15:.0f} °C (ambient)")
print(f"  Pressure range: {pressures_bar[0]} - {pressures_bar[-1]} bar")
print()

print(f"  {'p (bar)':>8s}  {'ρ (kg/m³)':>10s}  {'ρ ideal':>9s}  "
      f"{'ρ/ρ ideal':>10s}  {'Z':>6s}")
print(f"  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*6}")

rho_data = []
for p_bar, p in zip(pressures_bar, pressures):
    rho_mol = density_from_pressure(p, T_AMB, methane, phase="vapor")
    rho_kg = rho_mol * M_CH4
    rho_ideal_mol = p / (R * T_AMB)
    rho_ideal_kg = rho_ideal_mol * M_CH4
    Z = compressibility_factor(rho_mol, T_AMB, methane)
    rho_data.append((p_bar, rho_mol, rho_kg, rho_ideal_kg, Z))
    print(f"  {p_bar:>8.0f}  {rho_kg:>10.2f}  {rho_ideal_kg:>9.2f}  "
            f"{rho_kg/rho_ideal_kg:>10.4f}  {Z:>6.4f}")

# ------------------------------------------------------------------
# Storage capacity comparison
# ------------------------------------------------------------------
print()
print("Storage capacity: 1 m³ tank at 250 bar holds how much gas?")
print()

p_atm = 1.013e5
p_store = 250e5
rho_store = density_from_pressure(p_store, T_AMB, methane, phase="vapor")
rho_atm = density_from_pressure(p_atm, T_AMB, methane, phase="vapor")
mass_per_tank = rho_store * M_CH4 * 1.0   # kg of CH4 in 1 m³ tank
volume_atm_equiv = mass_per_tank / (rho_atm * M_CH4)

print(f"  Tank volume:               1.000 m³")
print(f"  CH₄ mass at 250 bar, 25°C: {mass_per_tank:.1f} kg")
print(f"  Equivalent atm-pressure volume: {volume_atm_equiv:.1f} m³")
print(f"  Compression ratio (mass-equivalent): "
        f"{volume_atm_equiv:.0f}× ambient")
print(f"  Ideal-gas naïve estimate would be: "
        f"{p_store / p_atm:.0f}× (off by "
        f"{(volume_atm_equiv - p_store/p_atm) / (p_store/p_atm) * 100:+.1f}%)")

# ------------------------------------------------------------------
# Isothermal compression work: ∫ p dv from atmospheric to storage
# ------------------------------------------------------------------
# For 1 mol of gas, the reversible isothermal work is
#     W = - ∫ p dv = ∫_p_atm^p_store v dp = R*T*∫ Z(p)/p dp
# We use the trapezoidal rule on a log-pressure grid.
print()
print("Isothermal compression work (lower bound on compressor duty):")

# Dense p-grid from 1 to 250 bar
p_grid_bar = np.logspace(np.log10(1.0), np.log10(250.0), 50)
p_grid = p_grid_bar * 1e5
Z_grid = np.zeros_like(p_grid)
for i, p in enumerate(p_grid):
    rho = density_from_pressure(p, T_AMB, methane, phase="vapor")
    Z_grid[i] = compressibility_factor(rho, T_AMB, methane)

# W_real = R T ∫ (Z/p) dp from p_atm to p_store
ln_p = np.log(p_grid)
integrand = Z_grid
W_real = R * T_AMB * np.trapezoid(integrand, ln_p)
W_ideal = R * T_AMB * np.log(p_store / p_grid[0])

print(f"  Real-gas compression work:  {W_real/1000:.2f} kJ/mol "
      f"= {W_real/M_CH4/1e6:.2f} MJ/kg CH₄")
print(f"  Ideal-gas comparison:       {W_ideal/1000:.2f} kJ/mol")
print(f"  Real/ideal ratio:           {W_real/W_ideal:.4f} "
      f"(< 1 means less work due to Z<1 at high p)")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation against NIST reference data (T = 25 °C):")

# Reference points: stateprop's Setzmann-Wagner 1991 is the NIST
# reference EOS for methane.  These values are the published reference
# (NIST Chemistry WebBook, methane fluid file derived from Setzmann-
# Wagner).  We round to 4 figures so small library-version drift
# wouldn't cause spurious failures.
nist = [
    (1.013e5,   0.6567),   # 1 atm
    (50e5,     35.27),
    (100e5,    75.99),
    (200e5,   157.09),
    (250e5,   188.20),
]

for p, rho_ref in nist:
    rho = density_from_pressure(p, T_AMB, methane, phase="vapor")
    rho_kg = rho * M_CH4
    validate(f"Methane density at {p/1e5:.0f} bar, 25 °C",
              reference=rho_ref, computed=rho_kg,
              units="kg/m³", tol_rel=0.005,
              source="NIST Chemistry WebBook (Setzmann-Wagner 1991)")

# Z-factor at 250 bar should be ~0.85
rho_250 = density_from_pressure(250e5, T_AMB, methane, phase="vapor")
Z_250 = compressibility_factor(rho_250, T_AMB, methane)
validate("Compressibility factor Z at 250 bar, 25 °C",
          reference=0.857, computed=Z_250,
          units="-", tol_rel=0.02,
          source="NIST WebBook (typical CNG storage condition)")

# Real compression work is less than ideal at these conditions (Z < 1)
validate_bool("Real compression work < ideal at high p (Z < 1)",
                condition=(W_real < W_ideal),
                detail=f"W_real={W_real/1000:.2f} < W_ideal={W_ideal/1000:.2f} kJ/mol",
                source="Theoretical: Z < 1 → reduced compression work")

# Mass-equivalent storage ratio: real-gas storage gives more mass than
# ideal-gas at same P (because Z < 1 means denser gas).
validate_bool("Real-gas CNG storage holds more mass than ideal estimate",
                condition=(volume_atm_equiv > p_store / p_atm),
                detail=f"real {volume_atm_equiv:.0f}× vs "
                f"ideal {p_store/p_atm:.0f}×")

summary()
