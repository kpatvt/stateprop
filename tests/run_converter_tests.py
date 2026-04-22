"""Tests for tools/convert_coolprop.py.

The converter is exercised on:
  1. A small hand-built CoolProp-style fluid (known coefficients)
  2. A SYNTHETIC CoolProp form of stateprop's existing CO2 fluid,
     verifying the converter -> loader pipeline produces a fluid
     that gives identical pressures to the original
  3. Edge cases: unsupported term types, malformed input
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))         # stateprop
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "tools")))  # converter

from convert_coolprop import (
    convert_fluid, UnsupportedTermType, CoolPropSchemaError,
)
from stateprop.fluid import Fluid
import stateprop as _sp


def _p(f, T, rho):
    """Convenience: pressure (Pa) for fluid object f at given T and rho."""
    Z = _sp.compressibility_factor(rho, T, f)
    return Z * rho * f.R * T


PASSED = 0
FAILED = 0
FAILURES = []

def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        FAILURES.append((label, detail))
        print(f"  FAIL  {label}: {detail}")


def run_test(fn):
    print(f"\n[{fn.__name__}]")
    try:
        fn()
    except Exception as e:
        global FAILED
        FAILED += 1
        FAILURES.append((fn.__name__, f"EXCEPTION: {type(e).__name__}: {e}"))
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 1: Build a synthetic CoolProp form of stateprop's CO2 file and verify
# round-trip pressure equality.
#
# This is the strongest test: it proves the converter, when given a CoolProp
# JSON whose mathematical content matches stateprop's existing CO2 fluid,
# produces a stateprop JSON that gives bit-for-bit identical pressures.
# ---------------------------------------------------------------------------

def _synthesize_coolprop_from_stateprop(sp_dict):
    """Build the CoolProp-style dict that the converter is designed to consume,
    starting from a stateprop dict. This is the exact inverse of the converter's
    field renames and array (de)flattening operations."""

    # Reduce: stateprop's "critical" carries T and rho (mol/m^3)
    crit = sp_dict["critical"]
    Tc = crit["T"]
    rhoc = crit["rho"]

    # Build alpha0 array. Stateprop's per-term form is dict-of-scalars; CoolProp
    # uses vectorized blocks. We must group consecutive PE / power_tau into one
    # block each since the converter then unrolls them. (Single-term blocks are
    # also valid CoolProp output; we just emit one block per stateprop entry to
    # keep this test simple and obvious.)
    alpha0 = []
    i = 0
    ideal = sp_dict["ideal"]
    while i < len(ideal):
        term = ideal[i]
        ttype = term["type"]
        if ttype == "log_delta":
            # CoolProp leaves log(delta) implicit; the converter re-adds it
            i += 1
            continue
        elif ttype == "a1":
            alpha0.append({
                "type": "IdealGasHelmholtzLead",
                "a1": term["a"],
                "a2": term.get("b", 0.0),
            })
            i += 1
        elif ttype == "log_tau":
            alpha0.append({"type": "IdealGasHelmholtzLogTau", "a": term["a"]})
            i += 1
        elif ttype == "power_tau":
            # Group consecutive power_tau entries into one block
            ns, ts = [term["a"]], [term["b"]]
            j = i + 1
            while j < len(ideal) and ideal[j]["type"] == "power_tau":
                ns.append(ideal[j]["a"]); ts.append(ideal[j]["b"])
                j += 1
            alpha0.append({"type": "IdealGasHelmholtzPower", "n": ns, "t": ts})
            i = j
        elif ttype == "PE":
            ns, ts = [term["a"]], [term["b"]]
            j = i + 1
            while j < len(ideal) and ideal[j]["type"] == "PE":
                ns.append(ideal[j]["a"]); ts.append(ideal[j]["b"])
                j += 1
            alpha0.append({
                "type": "IdealGasHelmholtzPlanckEinstein", "n": ns, "t": ts,
            })
            i = j
        else:
            raise NotImplementedError(f"synth path doesn't yet emit {ttype}")

    # Build alphar array. Stateprop separates polynomial / exponential into
    # two lists; CoolProp packs them into one ResidualHelmholtzPower block
    # (l=0 = polynomial, l>0 = exponential with c=l).
    alphar = []
    poly = sp_dict.get("residual_polynomial", [])
    expo = sp_dict.get("residual_exponential", [])
    if poly or expo:
        n_arr, d_arr, t_arr, l_arr = [], [], [], []
        for p in poly:
            n_arr.append(p["n"]); d_arr.append(p["d"])
            t_arr.append(p["t"]); l_arr.append(0)
        for e in expo:
            n_arr.append(e["n"]); d_arr.append(e["d"])
            t_arr.append(e["t"]); l_arr.append(e["c"])
        alphar.append({
            "type": "ResidualHelmholtzPower",
            "n": n_arr, "d": d_arr, "t": t_arr, "l": l_arr,
        })
    gaus = sp_dict.get("residual_gaussian", [])
    if gaus:
        alphar.append({
            "type": "ResidualHelmholtzGaussian",
            "n":   [g["n"]   for g in gaus],
            "d":   [g["d"]   for g in gaus],
            "t":   [g["t"]   for g in gaus],
            "eta": [g["eta"] for g in gaus],
            "epsilon": [g["epsilon"] for g in gaus],
            "beta":    [g["beta"]    for g in gaus],
            "gamma":   [g["gamma"]   for g in gaus],
        })
    nonan = sp_dict.get("residual_nonanalytic", [])
    if nonan:
        alphar.append({
            "type": "ResidualHelmholtzNonAnalytic",
            "n": [na["n"] for na in nonan],
            "a": [na["a"] for na in nonan],
            "b": [na["b"] for na in nonan],
            "B": [na["B"] for na in nonan],
            "C": [na["C"] for na in nonan],
            "D": [na["D"] for na in nonan],
            "A": [na["A"] for na in nonan],
            "beta": [na["beta"] for na in nonan],
        })

    return {
        "INFO": {"NAME": sp_dict["name"]},
        "EOS": [{
            "gas_constant": sp_dict["gas_constant"],
            "molar_mass": sp_dict["molar_mass"],
            "STATES": {
                "reducing": {"T": Tc, "rhomolar": rhoc},
                "critical": {"T": Tc, "rhomolar": rhoc,
                             "p": crit.get("p")},
            },
            "T_min": sp_dict.get("limits", {}).get("Tmin", 200.0),
            "T_max": sp_dict.get("limits", {}).get("Tmax", 1100.0),
            "p_max": sp_dict.get("limits", {}).get("pmax", 1e9),
            "alpha0": alpha0,
            "alphar": alphar,
            "BibTeX_EOS": "Span-Wagner-JPCRD-1996",
        }],
    }


def test_co2_round_trip_pressure_match():
    """Synthesize CoolProp form of CO2, convert, and verify identical pressures."""
    # Load the original
    co2_path = os.path.normpath(os.path.join(
        HERE, "..", "stateprop", "fluids", "carbondioxide.json"))
    with open(co2_path) as f:
        sp_orig = json.load(f)

    # Synthesize CoolProp form
    cp_form = _synthesize_coolprop_from_stateprop(sp_orig)

    # Run through the converter
    sp_converted = convert_fluid(cp_form)

    # Sanity: the converted dict should have the same number of residual terms
    check(
        f"poly count: orig {len(sp_orig.get('residual_polynomial', []))}, "
        f"converted {len(sp_converted.get('residual_polynomial', []))}",
        len(sp_orig.get("residual_polynomial", [])) ==
        len(sp_converted.get("residual_polynomial", [])),
    )
    check(
        f"expo count: orig {len(sp_orig.get('residual_exponential', []))}, "
        f"converted {len(sp_converted.get('residual_exponential', []))}",
        len(sp_orig.get("residual_exponential", [])) ==
        len(sp_converted.get("residual_exponential", [])),
    )
    check(
        f"gaus count: orig {len(sp_orig.get('residual_gaussian', []))}, "
        f"converted {len(sp_converted.get('residual_gaussian', []))}",
        len(sp_orig.get("residual_gaussian", [])) ==
        len(sp_converted.get("residual_gaussian", [])),
    )

    # Load both through stateprop and compare pressure at several state points
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(sp_converted, tf)
        tmp_path = tf.name
    try:
        f_orig = Fluid.from_json(co2_path)
        f_conv = Fluid.from_json(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Compare pressure at a grid of points spanning gas, liquid, supercritical
    test_points = [
        (250.0,    0.5),     # cold gas
        (300.0,  100.0),     # gas
        (300.0, 5000.0),     # liquid-like
        (310.0,12000.0),     # supercritical near critical
        (500.0,  500.0),     # high T gas
        (500.0,15000.0),     # high T dense
        (800.0, 1000.0),     # very high T gas
    ]
    max_rel = 0.0
    for T, rho in test_points:
        p1 = _p(f_orig, T, rho)
        p2 = _p(f_conv, T, rho)
        rel = abs(p1 - p2) / max(abs(p1), 1.0)
        max_rel = max(max_rel, rel)
        check(
            f"p(T={T:.0f}, rho={rho:.0f}): orig={p1:.6e}, conv={p2:.6e}, "
            f"rel={rel:.2e}",
            rel < 1e-12,
            f"rel={rel:.2e} > 1e-12",
        )
    print(f"  max relative pressure error across {len(test_points)} points: "
          f"{max_rel:.2e}")


# ---------------------------------------------------------------------------
# Test 2: Same round-trip for water (which has Gaussian + non-analytic terms)
# ---------------------------------------------------------------------------

def test_water_round_trip_pressure_match():
    """Water tests Gaussian and non-analytic residual term paths."""
    water_path = os.path.normpath(os.path.join(
        HERE, "..", "stateprop", "fluids", "water.json"))
    with open(water_path) as f:
        sp_orig = json.load(f)
    cp_form = _synthesize_coolprop_from_stateprop(sp_orig)
    sp_converted = convert_fluid(cp_form)

    check(
        "non-analytic count preserved",
        len(sp_orig.get("residual_nonanalytic", [])) ==
        len(sp_converted.get("residual_nonanalytic", []))
    )
    check(
        "gaussian count preserved",
        len(sp_orig.get("residual_gaussian", [])) ==
        len(sp_converted.get("residual_gaussian", []))
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(sp_converted, tf)
        tmp_path = tf.name
    try:
        f_orig = Fluid.from_json(water_path)
        f_conv = Fluid.from_json(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Avoid the immediate critical region where non-analytic terms dominate
    # and where small numerical differences can amplify
    test_points = [
        (300.0,    1.0),       # subcritical vapor
        (400.0,   10.0),       # superheated gas
        (300.0, 55000.0),      # liquid
        (700.0,  1000.0),      # supercritical gas-like
        (1000.0, 5000.0),      # very high T
    ]
    for T, rho in test_points:
        p1 = _p(f_orig, T, rho)
        p2 = _p(f_conv, T, rho)
        rel = abs(p1 - p2) / max(abs(p1), 1.0)
        check(
            f"p(T={T:.0f}, rho={rho:.0f}): orig={p1:.4e}, conv={p2:.4e}, "
            f"rel={rel:.2e}",
            rel < 1e-12,
            f"rel={rel:.2e} > 1e-12",
        )


# ---------------------------------------------------------------------------
# Test 3: PlanckEinsteinFunctionT (theta in Kelvin) maps correctly
# ---------------------------------------------------------------------------

def test_planck_einstein_function_T():
    """A CoolProp PlanckEinsteinFunctionT term with theta in Kelvin should
    convert to a stateprop PE term with b = theta / T_c."""
    cp = {
        "INFO": {"NAME": "FakeFluid"},
        "EOS": [{
            "gas_constant": 8.314472,
            "molar_mass": 0.044096,
            "STATES": {
                "reducing": {"T": 200.0, "rhomolar": 5000.0},
                "critical": {"T": 200.0, "rhomolar": 5000.0, "p": 5e6},
            },
            "T_min": 100.0, "T_max": 1000.0, "p_max": 1e8,
            "alpha0": [
                {"type": "IdealGasHelmholtzLead", "a1": 0.0, "a2": 0.0},
                {"type": "IdealGasHelmholtzLogTau", "a": 2.5},
                {"type": "IdealGasHelmholtzPlanckEinsteinFunctionT",
                 "n": [3.0, 5.0],
                 "t": [400.0, 800.0]},   # in Kelvin
            ],
            "alphar": [
                {"type": "ResidualHelmholtzPower",
                 "n": [1.0], "d": [1], "t": [0.0], "l": [0]},
            ],
        }],
    }
    out = convert_fluid(cp)
    pes = [t for t in out["ideal"] if t["type"] == "PE"]
    check(f"two PE terms emitted (got {len(pes)})", len(pes) == 2)
    if len(pes) >= 2:
        # b should be theta / T_c = 400/200 = 2.0  and  800/200 = 4.0
        check(f"first PE b = theta/T_c (got {pes[0]['b']}, expected 2.0)",
              abs(pes[0]["b"] - 2.0) < 1e-12)
        check(f"second PE b = theta/T_c (got {pes[1]['b']}, expected 4.0)",
              abs(pes[1]["b"] - 4.0) < 1e-12)


# ---------------------------------------------------------------------------
# Test 4: Unsupported types raise UnsupportedTermType
# ---------------------------------------------------------------------------

def test_cp0_polyt_t_minus_1_now_supported():
    """CP0PolyT with t=-1 now converts (v0.9.2) using the new tau_log_tau
    ideal-term kernel (code 8 in core.py). The cp/R contribution is
    c_k * tau and the alpha_0 contribution includes a tau*ln(tau) term."""
    cp = {
        "INFO": {"NAME": "FakeFluid"},
        "EOS": [{
            "gas_constant": 8.314, "molar_mass": 0.018,
            "STATES": {"reducing": {"T": 600, "rhomolar": 17000},
                       "critical": {"T": 600, "rhomolar": 17000, "p": 22e6}},
            "alpha0": [
                {"type": "IdealGasHelmholtzLead", "a1": 0.0, "a2": 0.0},
                {"type": "IdealGasHelmholtzLogTau", "a": 2.5},
                {"type": "IdealGasHelmholtzCP0PolyT",
                 "T0": 298.15, "Tc": 600, "c": [1.0], "t": [-1]},
            ],
            "alphar": [{"type": "ResidualHelmholtzPower",
                        "n":[1],"d":[1],"t":[0],"l":[0]}],
        }],
    }
    try:
        out = convert_fluid(cp)
        types = [t["type"] for t in out["ideal"]]
        check(f"CP0PolyT t=-1 emits tau_log_tau term (got types={types})",
              "tau_log_tau" in types,
              f"missing tau_log_tau: {types}")
    except Exception as e:
        check("CP0PolyT t=-1 converts without error", False,
              f"{type(e).__name__}: {e}")


def test_cp0_constant_now_supported():
    """CP0Constant should now convert successfully (v0.9.1)."""
    cp = {
        "INFO": {"NAME": "FakeFluid"},
        "EOS": [{
            "gas_constant": 8.314472, "molar_mass": 0.044,
            "STATES": {"reducing": {"T": 369.295, "rhomolar": 6058.22},
                       "critical": {"T": 369.295, "rhomolar": 6058.22,
                                    "p": 4.99e6}},
            "alpha0": [
                {"type": "IdealGasHelmholtzLead", "a1": -15.86, "a2": 11.68},
                {"type": "IdealGasHelmholtzLogTau", "a": -1.0},
                {"type": "IdealGasHelmholtzCP0Constant",
                 "T0": 273.15, "Tc": 369.295, "cp_over_R": 4.00526},
            ],
            "alphar": [{"type": "ResidualHelmholtzPower",
                        "n":[1.0],"d":[1],"t":[0.0],"l":[0]}],
        }],
    }
    try:
        out = convert_fluid(cp)
        # Check expected stateprop terms generated
        types = [t["type"] for t in out["ideal"]]
        check(f"CP0Constant produces stateprop terms (got {types})",
              "log_tau" in types and "a1" in types,
              f"missing log_tau/a1: {types}")
    except Exception as e:
        check("CP0Constant converts without error", False,
              f"{type(e).__name__}: {e}")


def test_unsupported_alphar_raises():
    cp = {
        "INFO": {"NAME": "FakeFluid"},
        "EOS": [{
            "gas_constant": 8.314, "molar_mass": 0.018,
            "STATES": {"reducing": {"T": 600, "rhomolar": 17000},
                       "critical": {"T": 600, "rhomolar": 17000, "p": 22e6}},
            "alpha0": [{"type": "IdealGasHelmholtzLead", "a1":0, "a2":0},
                       {"type": "IdealGasHelmholtzLogTau", "a": 2.5}],
            "alphar": [{"type": "ResidualHelmholtzAssociating",
                        "a": 1.0, "m": 1.5, "vbarn": 1.4e-5, "kappabar": 0.001,
                        "epsilonbar": 100.0}],
        }],
    }
    try:
        convert_fluid(cp)
        check("Associating term raises UnsupportedTermType", False, "no exception")
    except UnsupportedTermType:
        check("Associating term raises UnsupportedTermType", True)
    except Exception as e:
        check("Associating term raises UnsupportedTermType", False,
              f"got {type(e).__name__} instead")


# ---------------------------------------------------------------------------
# Test 5: Schema errors on malformed input
# ---------------------------------------------------------------------------

def test_missing_eos_block_raises():
    try:
        convert_fluid({"INFO": {"NAME": "X"}})
        check("missing EOS raises CoolPropSchemaError", False, "no exception")
    except CoolPropSchemaError:
        check("missing EOS raises CoolPropSchemaError", True)


def test_eos_index_out_of_range():
    try:
        convert_fluid({"INFO": {"NAME": "X"}, "EOS": []})
        check("empty EOS list raises CoolPropSchemaError", False, "no exception")
    except CoolPropSchemaError:
        check("empty EOS list raises CoolPropSchemaError", True)


# ---------------------------------------------------------------------------
# Test 6: Full pipeline -- write a converted fluid and load via stateprop
# ---------------------------------------------------------------------------

def test_round_trip_through_loader():
    """End-to-end check that the converter produces a JSON the stateprop
    loader can read and use. Independent of the per-term comparison above."""
    co2_path = os.path.normpath(os.path.join(
        HERE, "..", "stateprop", "fluids", "carbondioxide.json"))
    with open(co2_path) as f:
        sp_orig = json.load(f)
    cp_form = _synthesize_coolprop_from_stateprop(sp_orig)
    out = convert_fluid(cp_form)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(out, tf)
        tmp_path = tf.name
    try:
        f_loaded = Fluid.from_json(tmp_path)
        # Use it for a real calculation
        p = _p(f_loaded, 300.0, 1000.0)
        check(f"converted-CO2 loads + pressure(T=300, rho=1000) "
              f"= {p:.3e} Pa (sane)",
              1e5 < p < 1e9, f"p={p}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    tests = [
        test_co2_round_trip_pressure_match,
        test_water_round_trip_pressure_match,
        test_planck_einstein_function_T,
        test_cp0_polyt_t_minus_1_now_supported,
        test_cp0_constant_now_supported,
        test_unsupported_alphar_raises,
        test_missing_eos_block_raises,
        test_eos_index_out_of_range,
        test_round_trip_through_loader,
    ]
    for t in tests:
        run_test(t)
    print(f"\n{'='*60}")
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    if FAILURES:
        for n, d in FAILURES:
            print(f"  - {n}: {d}")
    print('='*60)
    sys.exit(0 if FAILED == 0 else 1)
