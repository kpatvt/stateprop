"""
Example: complete steam Rankine cycle analysis using IAPWS-95.

This is the worked example of every thermodynamics textbook: compute the
state at each corner of a simple Rankine cycle (pump -> boiler -> turbine
-> condenser), then the net work, heat input, and thermal efficiency.

The calculations use:
  - PT flash to set the condenser-saturation state
  - PS flash (isentropic) to model ideal pump and turbine
  - PH flash to close the cycle after heat addition / rejection

Run:
    python examples/03_rankine_cycle.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as he


# ---- helpers ---------------------------------------------------------------

def kg(state, fluid, field):
    """Convert a molar value to a mass-specific value (kJ/kg or kJ/kg-K)."""
    M = fluid.molar_mass
    return getattr(state, field) / M * 1e-3


def print_state(label, state, fluid):
    """Pretty-print one state."""
    M = fluid.molar_mass
    print(f"  {label}")
    print(f"    phase  = {state.phase}")
    print(f"    T      = {state.T:.3f} K  ({state.T - 273.15:+.2f} degC)")
    print(f"    p      = {state.p*1e-6:.4f} MPa")
    print(f"    rho    = {state.rho * M:.4f} kg/m^3")
    if state.quality is not None:
        print(f"    x      = {state.quality:.6f}")
    print(f"    h      = {kg(state, fluid, 'h'):.3f} kJ/kg")
    print(f"    s      = {kg(state, fluid, 's'):.5f} kJ/(kg K)")


# ---- Cycle definition ------------------------------------------------------

def rankine_ideal(p_boiler_MPa, p_cond_MPa, T_superheat_K, fluid):
    """Simulate a simple ideal Rankine cycle with superheat.

    States:
        1: saturated liquid at condenser pressure  (exit of condenser)
        2: compressed liquid after isentropic pump to boiler pressure
        3: superheated steam exiting boiler at (p_boiler, T_superheat)
        4: two-phase mixture after isentropic turbine expansion to condenser pressure

    Returns a dict with per-state results and overall efficiency.
    """
    p_boil = p_boiler_MPa * 1e6
    p_cond = p_cond_MPa  * 1e6

    # State 1: saturated liquid at condenser pressure
    # Get T_sat and rho_L via saturation_pT, then compute full state.
    # Using flash_ph at h = h_L_sat gives a liquid-phase FlashResult.
    # Simpler: use the private helper path to get the saturated liquid state.
    T_sat_cond, rho_L_cond, rho_V_cond = _sat_state(p_cond, fluid)
    s1_mol = he.entropy(rho_L_cond, T_sat_cond, fluid)
    h1_mol = he.enthalpy(rho_L_cond, T_sat_cond, fluid)
    # Build a FlashResult manually for state 1
    state1 = he.FlashResult(
        phase="liquid",
        T=T_sat_cond, p=p_cond,
        rho=rho_L_cond,
        u=he.internal_energy(rho_L_cond, T_sat_cond, fluid),
        h=h1_mol, s=s1_mol,
    )

    # State 2: after ideal (isentropic) pump to boiler pressure
    # Use flash_ps(p_boil, s1)
    state2 = he.flash_ps(p_boil, s1_mol, fluid)

    # State 3: superheated steam at boiler outlet (p_boil, T_superheat)
    state3 = he.flash_pt(p_boil, T_superheat_K, fluid)

    # State 4: after ideal (isentropic) turbine expansion to condenser pressure
    state4 = he.flash_ps(p_cond, state3.s, fluid)

    # Work and heat (per unit mass, positive if delivered by the fluid to the surroundings)
    w_pump    = kg(state2, fluid, 'h') - kg(state1, fluid, 'h')   # work INPUT (positive)
    q_boiler  = kg(state3, fluid, 'h') - kg(state2, fluid, 'h')   # heat INPUT (positive)
    w_turbine = kg(state3, fluid, 'h') - kg(state4, fluid, 'h')   # work OUTPUT (positive)
    q_cond    = kg(state4, fluid, 'h') - kg(state1, fluid, 'h')   # heat REJECTED (positive)

    w_net = w_turbine - w_pump
    eta = w_net / q_boiler
    # First-law check: w_net should equal q_boiler - q_cond
    first_law_residual = w_net - (q_boiler - q_cond)

    return {
        "states": (state1, state2, state3, state4),
        "w_pump":    w_pump,
        "w_turbine": w_turbine,
        "w_net":     w_net,
        "q_boiler":  q_boiler,
        "q_cond":    q_cond,
        "eta":       eta,
        "first_law_residual": first_law_residual,
    }


def _sat_state(p, fluid):
    """Find (T_sat, rho_L, rho_V) at given saturation pressure p."""
    # Use the same internal helper from flash.py
    from stateprop.flash import _T_sat_from_p
    return _T_sat_from_p(p, fluid)


# ---- Driver ----------------------------------------------------------------

def main():
    fluid = he.load_fluid("water")
    print("=" * 76)
    print(f"IDEAL RANKINE CYCLE  --  working fluid: {fluid.name}")
    print("=" * 76)

    # Typical steam-power-plant conditions
    p_boiler_MPa  = 10.0
    p_cond_MPa    = 0.01
    T_superheat_K = 773.15   # 500 degC

    print(f"  Boiler pressure      p3 = {p_boiler_MPa} MPa")
    print(f"  Condenser pressure   p1 = {p_cond_MPa} MPa")
    print(f"  Superheat temperature T3 = {T_superheat_K} K = {T_superheat_K-273.15} degC")
    print()

    r = rankine_ideal(p_boiler_MPa, p_cond_MPa, T_superheat_K, fluid)

    s1, s2, s3, s4 = r["states"]
    print("STATES")
    print_state("State 1 (saturated liquid at condenser):", s1, fluid)
    print_state("State 2 (compressed liquid after pump):", s2, fluid)
    print_state("State 3 (superheated steam after boiler):", s3, fluid)
    print_state("State 4 (wet steam after turbine):", s4, fluid)

    print()
    print("ENERGY FLOWS per unit mass (kJ/kg):")
    print(f"  Pump work input          w_pump    = {r['w_pump']:8.3f}")
    print(f"  Boiler heat input        q_boil    = {r['q_boiler']:8.3f}")
    print(f"  Turbine work output      w_turbine = {r['w_turbine']:8.3f}")
    print(f"  Condenser heat rejected  q_cond    = {r['q_cond']:8.3f}")
    print(f"  Net work                 w_net     = {r['w_net']:8.3f}")
    print()
    print(f"  First-law residual  w_net - (q_boiler - q_cond) = {r['first_law_residual']:.3e} kJ/kg")
    print(f"  (should be zero to machine precision)")
    print()
    print(f"  Thermal efficiency   eta = w_net / q_boiler = {r['eta']*100:.2f}%")
    print(f"  Carnot limit at these temperatures:  "
          f"{(1 - s1.T / s3.T) * 100:.2f}%")
    print()

    # ---- Parameter study: efficiency vs boiler pressure ----
    print("=" * 76)
    print("PARAMETER STUDY: Rankine efficiency vs boiler pressure")
    print(f"(condenser at {p_cond_MPa} MPa, superheat to {T_superheat_K} K)")
    print("=" * 76)
    print(f"{'p_boiler [MPa]':>18} {'eta [%]':>10} {'w_net [kJ/kg]':>18} {'x_exit':>10}")
    print("-" * 60)
    for p_b in [2.0, 5.0, 10.0, 15.0, 20.0]:
        try:
            rr = rankine_ideal(p_b, p_cond_MPa, T_superheat_K, fluid)
            s4 = rr["states"][3]
            x = s4.quality if s4.quality is not None else 1.0
            print(f"{p_b:>18.2f} {rr['eta']*100:>10.3f} {rr['w_net']:>18.3f} {x:>10.4f}")
        except Exception as e:
            print(f"{p_b:>18.2f}  <failed: {e}>")


if __name__ == "__main__":
    main()
