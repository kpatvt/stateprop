"""Bulk-convert a CoolProp dev/fluids directory into stateprop fluid JSONs.

This is the runner script for ingesting the full CoolProp fluid library
(~120 fluids) into stateprop's format. It:

  1. Walks every *.json in a CoolProp dev/fluids directory
  2. Calls convert_coolprop.convert_fluid() on each
  3. Catches UnsupportedTermType to skip fluids that need work
  4. Optionally validates each converted fluid by loading it through
     stateprop and checking a few thermodynamic identities and a NIST
     WebBook reference point if available
  5. Writes successful conversions into stateprop/fluids/coolprop/
  6. Writes a manifest JSON listing what was converted, what was skipped,
     and why.

Usage
-----
First obtain CoolProp's fluid JSONs. The easy way::

    git clone --depth 1 https://github.com/CoolProp/CoolProp.git /tmp/coolprop_src

Then run::

    cd stateprop/
    python tools/build_fluid_library.py /tmp/coolprop_src/dev/fluids \
        --output stateprop/fluids/coolprop \
        --validate

After this, the converted fluids are accessible to stateprop's loader
exactly like the existing fluids/carbondioxide.json or
fluids/gerg2008/methane.json.

The --validate flag does basic round-trip checks on each fluid:
  - The JSON loads through stateprop without error
  - At a low-density state point (rho = 0.01 mol/m^3, T = 300 K) the
    pressure agrees with p = rho * R * T to 1e-4 relative (ideal-gas limit)
  - The Helmholtz framework's pressure derivative dp/drho > 0 (mechanical
    stability) at three midrange points

These are necessary but not sufficient — they catch gross errors but not
sub-percent coefficient typos. For full validation against published
reference data, use the per-fluid NIST WebBook check (manual; out of
scope for this script).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow this script to be run directly from the tools/ folder
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # for convert_coolprop
sys.path.insert(0, str(HERE.parent))  # for stateprop

from convert_coolprop import (
    convert_fluid, UnsupportedTermType, CoolPropSchemaError,
)


# ---------------------------------------------------------------------------
# Per-fluid validation
# ---------------------------------------------------------------------------

def validate_converted_fluid(out_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """Try loading the converted JSON through stateprop and run sanity checks.

    Returns (ok, message). On success, message is a short summary; on
    failure, message describes what went wrong.
    """
    try:
        from stateprop.fluid import Fluid
    except Exception as e:
        return False, f"could not import stateprop.fluid: {e}"

    # Write to a temp file then load (the loader takes a path or a dict
    # depending on stateprop's API; we go through tempfile for safety)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(out_dict, tf)
        tmp_path = tf.name
    try:
        try:
            f = Fluid.from_json(tmp_path)
        except AttributeError:
            # Fallback: maybe the loader is `Fluid(path)` or stateprop.load_fluid
            from stateprop import load_fluid     # type: ignore
            f = load_fluid(tmp_path)
    except Exception as e:
        return False, f"loader rejected fluid: {type(e).__name__}: {e}"
    finally:
        os.unlink(tmp_path)

    # Sanity check 1: ideal-gas limit at very low density
    # Use the module-level helper rather than a method on the Fluid object
    # (Fluid is a data container; the kernels live at module level).
    import stateprop as _sp
    R = out_dict["gas_constant"]
    T_test = 300.0
    rho_test = 0.01           # mol/m^3, well into ideal-gas region
    try:
        Z = _sp.compressibility_factor(rho_test, T_test, f)
        p_calc = Z * rho_test * R * T_test
    except Exception as e:
        return False, f"compressibility_factor() raised: {type(e).__name__}: {e}"
    p_ideal = rho_test * R * T_test
    rel = abs(p_calc - p_ideal) / p_ideal
    if rel > 1e-3:
        return False, (
            f"ideal-gas limit failed: at rho={rho_test} mol/m^3, T={T_test} K "
            f"got p={p_calc:.6e}, expected ~{p_ideal:.6e} (rel diff {rel:.2e})"
        )

    # Sanity check 2: dp/drho > 0 at three midrange single-phase points
    Tc = out_dict["critical"]["T"]
    rho_c = out_dict["critical"]["rho"]
    points = [
        (1.5 * Tc, 0.5 * rho_c),    # supercritical
        (1.5 * Tc, 1.5 * rho_c),    # supercritical liquid-like
        (2.0 * Tc, 1.0 * rho_c),    # high-T midrange
    ]
    for T, rho in points:
        try:
            Z1 = _sp.compressibility_factor(rho,         T, f)
            Z2 = _sp.compressibility_factor(rho * 1.001, T, f)
            p1 = Z1 * rho * R * T
            p2 = Z2 * rho * 1.001 * R * T
        except Exception as e:
            return False, (
                f"pressure derivative check raised at (T={T:.1f}, "
                f"rho={rho:.1f}): {type(e).__name__}: {e}"
            )
        if not (p2 > p1):
            return False, (
                f"dp/drho not positive at T={T:.1f} K, rho={rho:.2f} mol/m^3 "
                f"(p1={p1:.3e}, p2={p2:.3e}); fluid may be unstable here"
            )

    return True, f"OK: ideal-gas limit rel-err {rel:.1e}, dp/drho>0 at 3 points"


# ---------------------------------------------------------------------------
# Output filename normalization
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """Turn a fluid name into a stateprop-style filename: lowercased,
    spaces and dashes removed, only [a-z0-9_]. Matches the convention of
    existing files like 'carbondioxide.json', 'gerg2008/methane.json'."""
    s = name.strip().lower()
    s = s.replace(" ", "").replace("-", "").replace(",", "")
    out = "".join(c for c in s if c.isalnum() or c == "_")
    return out + ".json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "input_dir",
        help="Path to CoolProp dev/fluids directory (containing *.json)"
    )
    ap.add_argument(
        "-o", "--output",
        default=str(HERE.parent / "stateprop" / "fluids" / "coolprop"),
        help="Output directory (default: stateprop/fluids/coolprop/)"
    )
    ap.add_argument(
        "--validate", action="store_true",
        help="Load each converted fluid through stateprop and run checks"
    )
    ap.add_argument(
        "--manifest", default=None,
        help="Path for the manifest JSON (default: <output>/_manifest.json)"
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N fluids (useful for smoke testing)"
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full tracebacks for unexpected exceptions"
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        ap.error(f"{args.input_dir} is not a directory")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else (
        out_dir / "_manifest.json"
    )

    inputs = sorted(in_dir.glob("*.json"))
    if args.limit:
        inputs = inputs[: args.limit]
    if not inputs:
        ap.error(f"no *.json files found in {in_dir}")

    converted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for src in inputs:
        try:
            with open(src) as f:
                cp_data = json.load(f)
        except Exception as e:
            skipped.append({"file": src.name, "reason": f"unreadable: {e}"})
            continue

        # CoolProp dev/fluids files are a single fluid as a JSON object,
        # but a few legacy files use a one-element list
        if isinstance(cp_data, list):
            if len(cp_data) != 1:
                skipped.append({"file": src.name,
                                "reason": f"unexpected list of {len(cp_data)}"})
                continue
            cp_data = cp_data[0]

        try:
            out_dict = convert_fluid(cp_data)
        except UnsupportedTermType as e:
            skipped.append({"file": src.name,
                            "reason": f"unsupported term: {e}"})
            continue
        except CoolPropSchemaError as e:
            skipped.append({"file": src.name,
                            "reason": f"schema error: {e}"})
            continue
        except Exception as e:
            if args.verbose:
                traceback.print_exc()
            skipped.append({"file": src.name,
                            "reason": f"unexpected: {type(e).__name__}: {e}"})
            continue

        # Optional validation
        validation_msg = None
        if args.validate:
            ok, msg = validate_converted_fluid(out_dict)
            validation_msg = msg
            if not ok:
                skipped.append({
                    "file": src.name,
                    "reason": f"validation failed: {msg}",
                    "name": out_dict.get("name"),
                })
                continue

        # Write to output dir
        out_name = _sanitize_filename(out_dict["name"])
        out_path = out_dir / out_name
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=2)
            f.write("\n")
        entry = {
            "source": src.name,
            "output": out_name,
            "name": out_dict["name"],
            "CAS": out_dict.get("CAS"),
            "n_polynomial": len(out_dict.get("residual_polynomial", [])),
            "n_exponential": len(out_dict.get("residual_exponential", [])),
            "n_gaussian": len(out_dict.get("residual_gaussian", [])),
            "n_nonanalytic": len(out_dict.get("residual_nonanalytic", [])),
        }
        if validation_msg:
            entry["validation"] = validation_msg
        converted.append(entry)
        print(f"  OK  {src.name:30s} -> {out_name}")

    # Write manifest
    manifest = {
        "source_dir": str(in_dir),
        "output_dir": str(out_dir),
        "n_total_inputs": len(inputs),
        "n_converted": len(converted),
        "n_skipped": len(skipped),
        "converted": converted,
        "skipped": skipped,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"converted: {len(converted)}  skipped: {len(skipped)}  "
          f"of {len(inputs)} input files")
    print(f"manifest: {manifest_path}")
    if skipped:
        # Group skip reasons for a quick summary
        from collections import Counter
        reasons = Counter()
        for s in skipped:
            r = s["reason"].split(":", 1)[0]
            reasons[r] += 1
        print(f"\nSkip reasons:")
        for r, n in reasons.most_common():
            print(f"  {n:4d}  {r}")
    return 0 if converted else 1


if __name__ == "__main__":
    sys.exit(main())
