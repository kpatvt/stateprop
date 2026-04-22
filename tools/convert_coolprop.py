"""Convert CoolProp fluid JSON files to stateprop's JSON schema.

CoolProp publishes high-quality reference Helmholtz equations of state
for ~120 fluids in machine-readable JSON form at:

    https://github.com/CoolProp/CoolProp/tree/master/dev/fluids

This module converts those JSONs into the schema stateprop's loader
expects (see stateprop/fluid.py). The two schemas describe the same
underlying mathematics; the conversion is mostly:

  - field renaming (CoolProp's `IdealGasHelmholtzPlanckEinstein` ->
    stateprop's `PE`, etc.)
  - vector unrolling (CoolProp uses parallel arrays `n: [...], t: [...]`
    while stateprop uses arrays of dicts `[{n:..., t:...}, ...]`)
  - splitting CoolProp's `ResidualHelmholtzPower` into stateprop's
    separate `residual_polynomial` (l=0) and `residual_exponential`
    (l>0, c=l) sections
  - top-level metadata (critical, triple, limits, gas_constant)

Usage
-----
Programmatic::

    from convert_coolprop import convert_fluid
    out_dict = convert_fluid(coolprop_json_dict)
    json.dump(out_dict, open("propane.json", "w"), indent=2)

CLI::

    python convert_coolprop.py path/to/n-Propane.json -o propane.json

Or to bulk-convert a CoolProp dev/fluids directory, see
build_fluid_library.py in this same `tools/` folder.

Supported CoolProp term types
-----------------------------
Ideal-gas (alpha0):
  - IdealGasHelmholtzLead                   (a1 + a2*tau)            -> a1
  - IdealGasHelmholtzLogTau                 (a*ln(tau))              -> log_tau
  - IdealGasHelmholtzPower                  (sum n_i * tau^t_i)      -> power_tau
  - IdealGasHelmholtzPlanckEinstein         (sum n_i ln(1-exp(-t_i tau)),
                                             t_i in REDUCED form)     -> PE
  - IdealGasHelmholtzPlanckEinsteinFunctionT (theta_i in KELVIN;
                                              converted to t_i = theta_i/T_c) -> PE
  - IdealGasHelmholtzPlanckEinsteinGeneralized (cosh/sinh form)      -> PE_cosh / PE_sinh
                                                                       (only the strict
                                                                       Planck-Einstein subset
                                                                       is currently extracted;
                                                                       others raise.)
  - IdealGasHelmholtzEnthalpyEntropyOffset  (reference-state shift)  -> dropped
                                                                       (stateprop applies its
                                                                       own reference state at
                                                                       use time)

Residual (alphar):
  - ResidualHelmholtzPower                  (n,d,t,l)                -> residual_polynomial (l=0)
                                                                        + residual_exponential (l>0, c=l)
  - ResidualHelmholtzGaussian               (n,d,t,eta,epsilon,beta,gamma) -> residual_gaussian
  - ResidualHelmholtzNonAnalytic            (n,a,b,B,C,D,A,beta)     -> residual_nonanalytic

Unsupported (will raise UnsupportedTermType with a clear message):
  - IdealGasHelmholtzCP0Constant, IdealGasHelmholtzCP0PolyT,
    IdealGasHelmholtzCP0AlyLee   (these are c_p^0(T) polynomials that
                                  must first be integrated twice to
                                  produce alpha0 form; see teqp's
                                  convert_CoolProp_idealgas function)
  - ResidualHelmholtzExponential, ResidualHelmholtzGERG2008,
    ResidualHelmholtzGaoB         (rare specialty residual forms)

When a fluid uses one of the unsupported types, the converter raises
UnsupportedTermType, the build_fluid_library.py orchestrator skips that
fluid and logs it.

License & attribution
---------------------
The converter itself is part of stateprop (MIT). The COEFFICIENTS in
each output JSON are reproduced (via this conversion) from CoolProp's
JSON files, which themselves cite the original publication for each
fluid. The converter preserves the citation in the `reference` field
of the output. Anyone redistributing converted fluid files should
preserve those citations.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsupportedTermType(ValueError):
    """Raised when a CoolProp term type isn't yet handled by this converter.

    The orchestrator catches this to skip fluids that need work the converter
    can't currently do (e.g. cp0-polynomial integration).
    """


class CoolPropSchemaError(ValueError):
    """Raised when the input doesn't match the expected CoolProp JSON layout."""


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------

def convert_fluid(cp_data: Dict[str, Any], *,
                  eos_index: int = 0,
                  preserve_unknown_metadata: bool = False) -> Dict[str, Any]:
    """Convert a CoolProp fluid JSON dict into stateprop's schema.

    Parameters
    ----------
    cp_data : dict
        Parsed CoolProp fluid JSON. Must have an `EOS` array with at
        least `eos_index + 1` entries and an `INFO` block.
    eos_index : int
        Which EOS within the CoolProp file to use (most fluids have
        only one; some have alternates with `eos_index` selecting them).
    preserve_unknown_metadata : bool
        If True, includes a `_coolprop_extra` field with bibliographic
        and ancillary data not consumed by stateprop's loader. Default
        False to keep the output JSON small.

    Returns
    -------
    dict suitable for json.dump and consumption by Fluid.from_json.

    Raises
    ------
    UnsupportedTermType   if any term in alpha0 or alphar uses a form
                          this converter does not yet support.
    CoolPropSchemaError   if the structure is malformed or fields missing.
    """
    if "EOS" not in cp_data or not isinstance(cp_data["EOS"], list):
        raise CoolPropSchemaError("missing or non-list 'EOS' field")
    if len(cp_data["EOS"]) <= eos_index:
        raise CoolPropSchemaError(
            f"EOS index {eos_index} out of range (only {len(cp_data['EOS'])} EOS in file)"
        )

    eos = cp_data["EOS"][eos_index]
    info = cp_data.get("INFO", {})

    # --- top-level metadata ---
    name = info.get("NAME") or info.get("name") or cp_data.get("NAME") or "Unknown"
    cas = info.get("CAS")
    aliases = info.get("ALIASES", [])

    out: Dict[str, Any] = {"name": name}
    if cas:
        out["CAS"] = cas
    if aliases:
        out["aliases"] = list(aliases)

    # --- gas constant and molar mass (units are documented in CoolProp's schema) ---
    R = eos.get("gas_constant")
    if R is None:
        raise CoolPropSchemaError("missing gas_constant in EOS block")
    M = eos.get("molar_mass")
    if M is None:
        raise CoolPropSchemaError("missing molar_mass in EOS block")
    # CoolProp uses J/(mol K) and kg/mol throughout (consistent with stateprop)
    out["molar_mass"] = float(M)
    out["gas_constant"] = float(R)

    # --- critical / reducing parameters ---
    # CoolProp stores reducing-point info in EOS["STATES"]["reducing"] and
    # critical-point info (which sometimes differs slightly) in EOS["STATES"]["critical"].
    states = eos.get("STATES", {})
    reducing = states.get("reducing", {})
    critical = states.get("critical", reducing)

    # The reducing parameters are what the EOS uses to non-dimensionalize
    # (delta = rho/rho_r, tau = T_r/T). Stateprop's loader expects these
    # under "critical" too (matching CoolProp/REFPROP convention where the
    # reducing point IS the critical point for pure fluids).
    Tc = float(reducing.get("T", critical.get("T")))
    rhoc_molar = float(reducing.get("rhomolar", critical.get("rhomolar")))
    pc = critical.get("p", reducing.get("p"))
    out["critical"] = {"T": Tc, "rho": rhoc_molar}
    if pc is not None:
        out["critical"]["p"] = float(pc)

    # Acentric factor (Pitzer). CoolProp stores it as EOS.acentric. Needed
    # for Wilson K-factor initialization of flash calculations; without it
    # Wilson K often lies in the regime where Rachford-Rice has no valid
    # solution and flash_pt falls through to a trivial single-phase answer.
    acentric = eos.get("acentric")
    if acentric is not None:
        out["acentric"] = float(acentric)

    # --- triple point (optional) ---
    triple = states.get("triple_liquid", {})
    if triple:
        T_t = triple.get("T")
        p_t = triple.get("p")
        if T_t is not None or p_t is not None:
            tp_block = {}
            if T_t is not None:
                tp_block["T"] = float(T_t)
            if p_t is not None:
                tp_block["p"] = float(p_t)
            out["triple"] = tp_block

    # --- validity limits ---
    Tmin = eos.get("T_min", eos.get("Tmin"))
    Tmax = eos.get("T_max", eos.get("Tmax"))
    pmax = eos.get("p_max", eos.get("pmax"))
    rhomax_mass = eos.get("rhoLmax")  # CoolProp sometimes uses mass density here
    rhomax_molar = eos.get("rhomolar_max", eos.get("rhomolarmax"))
    if rhomax_molar is None and rhomax_mass is not None:
        rhomax_molar = rhomax_mass / out["molar_mass"]
    limits: Dict[str, float] = {}
    if Tmin is not None: limits["Tmin"] = float(Tmin)
    if Tmax is not None: limits["Tmax"] = float(Tmax)
    if pmax is not None: limits["pmax"] = float(pmax)
    if rhomax_molar is not None: limits["rhomax"] = float(rhomax_molar)
    if limits:
        out["limits"] = limits

    # --- citation ---
    bibtex_eos = eos.get("BibTeX_EOS") or eos.get("BibTeX")
    if bibtex_eos:
        out["reference"] = f"BibTeX: {bibtex_eos}"

    # --- ideal-gas (alpha0) and residual (alphar) terms ---
    alpha0 = eos.get("alpha0", [])
    alphar = eos.get("alphar", [])
    if not alpha0:
        raise CoolPropSchemaError("EOS has empty alpha0 (ideal-gas) block")
    if not alphar:
        raise CoolPropSchemaError("EOS has empty alphar (residual) block")

    out["ideal"] = _convert_alpha0(alpha0, T_reduce=Tc)
    poly, expo, gaus, nonan, dblexp, gaob = _convert_alphar(alphar)
    if poly:
        out["residual_polynomial"] = poly
    if expo:
        out["residual_exponential"] = expo
    if gaus:
        out["residual_gaussian"] = gaus
    if nonan:
        out["residual_nonanalytic"] = nonan
    if dblexp:
        out["residual_double_exponential"] = dblexp
    if gaob:
        out["residual_gaob"] = gaob

    if preserve_unknown_metadata:
        extras = {}
        for k in ("INCHI_KEY", "FORMULA", "SMILES", "CHEMSPIDER_ID", "PUBCHEM_CID"):
            if k in info:
                extras[k] = info[k]
        if extras:
            out["_coolprop_info"] = extras

    return out


# ---------------------------------------------------------------------------
# Ideal-gas (alpha0) term conversion
# ---------------------------------------------------------------------------

def _convert_alpha0(alpha0: List[Dict[str, Any]],
                    *, T_reduce: float) -> List[Dict[str, Any]]:
    """Convert CoolProp alpha0 array to stateprop ideal terms list.

    Always emits a leading {"type":"log_delta", "a":1.0} term, since
    stateprop's existing fluids start with that and the contribution is
    implicit in CoolProp (the ln(delta) term is added by the framework
    rather than appearing in the JSON).
    """
    out: List[Dict[str, Any]] = [{"type": "log_delta", "a": 1.0}]

    for term in alpha0:
        ttype = term.get("type")
        if ttype == "IdealGasHelmholtzLead":
            # alpha_0 += a1 + a2*tau
            out.append({
                "type": "a1",
                "a": float(term["a1"]),
                "b": float(term["a2"]),
            })

        elif ttype == "IdealGasHelmholtzLogTau":
            # alpha_0 += a * ln(tau)
            out.append({"type": "log_tau", "a": float(term["a"])})

        elif ttype == "IdealGasHelmholtzPower":
            # alpha_0 += sum n_i * tau^t_i
            for n_i, t_i in zip(term["n"], term["t"]):
                out.append({
                    "type": "power_tau",
                    "a": float(n_i),
                    "b": float(t_i),
                })

        elif ttype == "IdealGasHelmholtzPlanckEinstein":
            # alpha_0 += sum n_i * ln(1 - exp(-t_i * tau))
            # Here t_i is already in REDUCED form (theta_i / T_reduce).
            for n_i, t_i in zip(term["n"], term["t"]):
                out.append({
                    "type": "PE",
                    "a": float(n_i),
                    "b": float(t_i),
                })

        elif ttype == "IdealGasHelmholtzPlanckEinsteinFunctionT":
            # Same but theta_i is in KELVIN; convert to reduced form.
            # CoolProp uses key 'v' for the Kelvin theta values (not 't').
            # alpha_0 += sum n_i * ln(1 - exp(-(theta_i / T_reduce) * tau))
            theta_arr = term.get("v", term.get("t"))   # back-compat
            for n_i, theta_i in zip(term["n"], theta_arr):
                out.append({
                    "type": "PE",
                    "a": float(n_i),
                    "b": float(theta_i) / T_reduce,
                })

        elif ttype == "IdealGasHelmholtzPlanckEinsteinGeneralized":
            # CoolProp form: alpha_0 += sum_i n_i * ln(c_i + d_i * exp(theta_i * tau))
            # Note the sign: CoolProp uses exp(+theta*tau), so for physical
            # behavior (argument stays positive) theta is typically negative.
            # Stateprop's PE kernel uses exp(-b*tau), so we map b = -theta.
            #
            # Recognized special cases:
            #   (c=1, d=-1): standard Planck-Einstein
            #       alpha_0 = n * ln(1 - exp(theta*tau)) = n * ln(1 - exp(-(-theta)*tau))
            #       -> stateprop PE with a=n, b=-theta
            #   (c=0, d=2):  2*cosh/sinh form -> PE_sinh (sign preserved since sinh is odd)
            #   (c=0, d=1):  hyperbolic cosh form  -> PE_cosh
            #   General (c,d): stateprop's PE_general (a * ln(c + d*exp(-b*tau)))
            #       -> emit PE_general with a=n, b=-theta, c_const=c, d_const=d
            n_arr = term["n"]; t_arr = term["t"]
            c_arr = term.get("c", [1.0]*len(n_arr))
            d_arr = term.get("d", [-1.0]*len(n_arr))
            for n_i, t_i, c_i, d_i in zip(n_arr, t_arr, c_arr, d_arr):
                b_val = -float(t_i)  # flip sign to match stateprop's exp(-b*tau) convention
                if c_i == 1.0 and d_i == -1.0:
                    out.append({"type": "PE", "a": float(n_i), "b": b_val})
                elif c_i == 0.0 and abs(d_i) == 2.0:
                    # n * ln(d * exp(theta*tau)) with |d|=2 ~ ln(2) + ln|sinh|
                    # (the extra ln(2) is absorbed into the Lead term's a1)
                    out.append({"type": "PE_sinh",
                                "a": float(n_i), "b": b_val})
                elif c_i == 0.0 and abs(d_i) == 1.0:
                    out.append({"type": "PE_cosh",
                                "a": float(n_i), "b": b_val})
                else:
                    # Generic form: needs stateprop's PE_general term (added v0.9.2)
                    out.append({"type": "PE_general",
                                "a": float(n_i), "b": b_val,
                                "c": float(c_i), "d": float(d_i)})

        elif ttype == "IdealGasHelmholtzEnthalpyEntropyOffset":
            # CoolProp uses these to set h, s reference states (e.g. IIR,
            # NBP, ASHRAE). Stateprop applies its own reference-state offset
            # at use time, so we silently drop this term.
            pass

        elif ttype in ("IdealGasHelmholtzCP0Constant",
                       "IdealGasHelmholtzCP0PolyT",
                       "IdealGasHelmholtzCP0AlyLee"):
            # cp^0(T) polynomial forms. We integrate twice analytically and
            # emit equivalent contributions in stateprop's term-type vocabulary.
            # The math is documented in convert_cp0.py; see the helper below.
            new_terms = _convert_cp0_term(term, T_reduce=T_reduce)
            out.extend(new_terms)

        else:
            raise UnsupportedTermType(
                f"Unrecognized alpha0 term type: {ttype!r}"
            )

    return out


# ---------------------------------------------------------------------------
# Residual (alphar) term conversion
# ---------------------------------------------------------------------------

def _convert_alphar(alphar: List[Dict[str, Any]]):
    """Convert CoolProp alphar array to stateprop's six residual lists.

    Returns (polynomial, exponential, gaussian, nonanalytic, double_exponential,
    gaob) where each is a list of dicts in stateprop's per-term schema.
    Empty lists for any that the fluid doesn't use.
    """
    poly: List[Dict[str, Any]] = []
    expo: List[Dict[str, Any]] = []
    gaus: List[Dict[str, Any]] = []
    nonan: List[Dict[str, Any]] = []
    dblexp: List[Dict[str, Any]] = []
    gaob: List[Dict[str, Any]] = []

    for block in alphar:
        btype = block.get("type")

        if btype == "ResidualHelmholtzPower":
            # CoolProp combines polynomial (l=0) and exponential (l>0)
            # terms into one block with parallel arrays {n, d, t, l}.
            # Stateprop separates them into residual_polynomial and
            # residual_exponential (with c = l for the latter).
            n_arr = block["n"]; d_arr = block["d"]
            t_arr = block["t"]; l_arr = block["l"]
            if not (len(n_arr) == len(d_arr) == len(t_arr) == len(l_arr)):
                raise CoolPropSchemaError(
                    "ResidualHelmholtzPower: n/d/t/l arrays differ in length"
                )
            for n, d, t, l in zip(n_arr, d_arr, t_arr, l_arr):
                if l == 0:
                    poly.append({"n": float(n), "d": int(d), "t": float(t)})
                else:
                    expo.append({"n": float(n), "d": int(d), "t": float(t),
                                 "c": int(l)})

        elif btype == "ResidualHelmholtzGaussian":
            for n, d, t, eta, eps, beta, gamma in zip(
                block["n"], block["d"], block["t"],
                block["eta"], block["epsilon"],
                block["beta"], block["gamma"]
            ):
                gaus.append({
                    "n": float(n), "d": int(d), "t": float(t),
                    "eta": float(eta), "epsilon": float(eps),
                    "beta": float(beta), "gamma": float(gamma),
                })

        elif btype == "ResidualHelmholtzNonAnalytic":
            for n, a, b, B, C, D, A, beta in zip(
                block["n"], block["a"], block["b"],
                block["B"], block["C"], block["D"],
                block["A"], block["beta"]
            ):
                nonan.append({
                    "n": float(n),
                    "a": float(a), "b": float(b),
                    "B": float(B), "C": float(C), "D": float(D),
                    "A": float(A), "beta": float(beta),
                })

        elif btype == "ResidualHelmholtzExponential":
            # Generalized form: n * delta^d * tau^t * exp(-g * delta^l).
            # When g==0 and l==0 this degenerates to a polynomial term;
            # otherwise it's an exponential with the new g-multiplier (added
            # to stateprop's kernel in v0.9.2).
            n_arr = block["n"]; d_arr = block["d"]
            t_arr = block["t"]; l_arr = block["l"]; g_arr = block["g"]
            if not (len(n_arr) == len(d_arr) == len(t_arr)
                    == len(l_arr) == len(g_arr)):
                raise CoolPropSchemaError(
                    "ResidualHelmholtzExponential: n/d/t/l/g arrays differ in length"
                )
            for n, d, t, l, g in zip(n_arr, d_arr, t_arr, l_arr, g_arr):
                if l == 0 and g == 0:
                    poly.append({"n": float(n), "d": int(d), "t": float(t)})
                else:
                    expo.append({"n": float(n), "d": int(d), "t": float(t),
                                 "c": int(l), "g": float(g)})

        elif btype == "ResidualHelmholtzGaoB":
            # Gao-Bubble form (Gao 2020 Ammonia EOS): n*δ^d*τ^t *
            # exp(eta*(δ-ε)² + 1/(β*(τ-γ)² + b))
            # Handled by stateprop's _alpha_r_gaob kernel (v0.9.3).
            keys = ("n", "d", "t", "eta", "epsilon", "beta", "gamma", "b")
            arrs = [block[k] for k in keys]
            lengths = {len(a) for a in arrs}
            if len(lengths) != 1:
                raise CoolPropSchemaError(
                    f"ResidualHelmholtzGaoB arrays have inconsistent lengths: "
                    f"{[len(a) for a in arrs]}"
                )
            for n, d, t, eta, eps, beta, gamma, b in zip(*arrs):
                gaob.append({
                    "n": float(n), "d": int(d), "t": float(t),
                    "eta": float(eta), "epsilon": float(eps),
                    "beta": float(beta), "gamma": float(gamma),
                    "b": float(b),
                })

        elif btype == "ResidualHelmholtzDoubleExponential":
            # Methanol's double-decay form: n*δ^d*τ^t*exp(-gd*δ^ld - gt*τ^lt).
            # Maps directly to stateprop's residual_double_exponential
            # (kernel added v0.9.3).
            keys = ("n", "d", "t", "ld", "lt", "gd", "gt")
            arrs = [block[k] for k in keys]
            lengths = {len(a) for a in arrs}
            if len(lengths) != 1:
                raise CoolPropSchemaError(
                    f"ResidualHelmholtzDoubleExponential arrays have "
                    f"inconsistent lengths: {[len(a) for a in arrs]}"
                )
            for n, d, t, ld, lt, gd, gt in zip(*arrs):
                dblexp.append({
                    "n": float(n), "d": int(d), "t": float(t),
                    "ld": float(ld), "lt": float(lt),
                    "gd": float(gd), "gt": float(gt),
                })

        elif btype == "ResidualHelmholtzLemmon2005":
            # Lemmon's 2005 generalized form (R125):
            #   n * δ^d * τ^t * exp(-δ^l - τ^m)
            # Decomposes by (l, m):
            #   (0, 0): polynomial (no exponential decay at all)
            #   (l>0, 0): standard exponential exp(-δ^l); map to residual_exponential
            #   (0, m>0): tau-only decay exp(-τ^m); map to residual_double_exponential
            #             with ld=0, gd=0 (suppresses the delta decay term: -0*δ^0=0)
            #   (l>0, m>0): both; map to residual_double_exponential with
            #               ld=l, lt=m, gd=1, gt=1
            keys = ("n", "d", "t", "l", "m")
            arrs = [block[k] for k in keys]
            lengths = {len(a) for a in arrs}
            if len(lengths) != 1:
                raise CoolPropSchemaError(
                    f"ResidualHelmholtzLemmon2005 arrays have inconsistent lengths"
                )
            for n, d, t, l, m in zip(*arrs):
                if l == 0 and m == 0:
                    poly.append({"n": float(n), "d": int(d), "t": float(t)})
                elif l > 0 and m == 0:
                    expo.append({"n": float(n), "d": int(d), "t": float(t),
                                 "c": int(l), "g": 1.0})
                elif l == 0 and m > 0:
                    dblexp.append({
                        "n": float(n), "d": int(d), "t": float(t),
                        "ld": 0.0, "lt": float(m),
                        "gd": 0.0, "gt": 1.0,
                    })
                else:  # both l>0 and m>0
                    dblexp.append({
                        "n": float(n), "d": int(d), "t": float(t),
                        "ld": float(l), "lt": float(m),
                        "gd": 1.0, "gt": 1.0,
                    })

        elif btype in ("ResidualHelmholtzGERG2008",
                       "ResidualHelmholtzAssociating"):
            raise UnsupportedTermType(
                f"alphar block '{btype}' uses a specialty residual term form "
                f"(rare; appears in only a handful of fluids). Adding support "
                f"requires extending stateprop's residual kernel in core.py."
            )

        else:
            raise UnsupportedTermType(
                f"Unrecognized alphar block type: {btype!r}"
            )

    return poly, expo, gaus, nonan, dblexp, gaob


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CP0 polynomial / Aly-Lee conversion to stateprop alpha0 term types
# ---------------------------------------------------------------------------
#
# CoolProp packages the temperature-dependent ideal-gas heat capacity in three
# alternate forms that bypass the standard Helmholtz alpha_0 representation:
#
#   IdealGasHelmholtzCP0Constant {T0, Tc, cp_over_R}
#       cp^0(T)/R = c (constant)
#
#   IdealGasHelmholtzCP0PolyT    {T0, Tc, c[], t[]}
#       cp^0(T)/R = sum_k c_k * (T/Tc)^t_k
#
#   IdealGasHelmholtzCP0AlyLee   {T0, Tc, c[5]}
#       cp^0(T)/R = c0 + c1*(c2/T/sinh(c2/T))^2 + c3*(c4/T/cosh(c4/T))^2
#
# These forms must be integrated twice (once to get h^0(T), once more divided
# by T^2 to enter alpha_0) and the result re-expressed in stateprop's term
# vocabulary. The standard derivation runs as follows.
#
# Starting from cp^0/R = c_k * (T/T_c)^t_k = c_k * tau^(-t_k), the contribution
# to the dimensionless ideal-gas Helmholtz energy alpha_0 (separating off the
# integration constants which CoolProp absorbs into Lead a1, a2 and which we
# leave alone) is:
#
#   For t_k != 0 and t_k != -1:
#     alpha_0_cp = -c_k/(t_k*(t_k+1)) * tau^(-t_k)
#                  - c_k/(t_k+1) * tau_0^(-(t_k+1)) * tau
#                  + c_k/t_k     * tau_0^(-t_k)
#     where tau_0 = T_c / T_0.
#
#   For t_k == 0 (constant cp):
#     alpha_0_cp = c_k * (1 - ln(tau_0)) - (c_k/tau_0) * tau + c_k * ln(tau)
#
#   For t_k == -1 (cp/R proportional to tau):
#     alpha_0_cp = -c_k * tau * ln(tau) + c_k*(1+ln(tau_0))*tau - c_k * tau_0
#     Uses stateprop's tau_log_tau ideal-term type (code 8, added v0.9.2).
#
# The Aly-Lee sinh^2 and cosh^2 terms are converted to the equivalent
# PE_sinh and PE_cosh forms, which stateprop's kernel already supports
# directly with semantics:
#     PE_sinh: alpha_0 += a * ln|sinh(b*tau)|
#     PE_cosh: alpha_0 += a * ln(cosh(b*tau))
# These reproduce the Aly-Lee cv contributions exactly (verified algebraically;
# see the comments below for the mapping).


def _cp0_polynomial_contribution(c_k: float, t_k: float, tau_0: float,
                                 T_c: float, T_0: float):
    """Return list of stateprop ideal-term dicts for one (c_k, t_k) entry of
    CoolProp's CP0PolyT. The CoolProp convention is
        cp^0/R = sum c_k * T^t_k  (raw T in KELVIN, NOT (T/T_c)^t_k)
    The double integration to alpha_0 gives (for t_k != 0, -1):
        alpha_0_cp = -c_k * T_c^t_k / (t_k*(t_k+1)) * tau^(-t_k)
                     - c_k * T_0^(t_k+1) / ((t_k+1)*T_c) * tau
                     + c_k * T_0^t_k / t_k
    Special cases t_k=0 and t_k=-1 are handled separately (the general
    formula has divisions by zero there).

    tau_0 = T_c/T_0; passed in for consistency with CP0Constant/AlyLee.
    T_c and T_0 are raw values in Kelvin.

    NOTE (v0.9.2): Fixed from v0.9.1 which treated coefficients as (T/T_c)^t_k
    based, which under-predicted cp for fluids with large polynomial
    coefficients (HFE143m, n-Undecane, etc). Previous versions' fluids
    had small polynomial corrections so the error was numerically invisible
    in pressure-based spot checks. Caloric spot checks in
    run_coolprop_fluids_tests.py now guard against regression.
    """
    terms = []
    if t_k == 0:
        # cp/R = c_k constant -> three contributions:
        #   constant + tau-linear absorbed into a1
        #   ln(tau) absorbed into log_tau
        # Same as CP0Constant. No T_c scaling needed (T^0 = 1).
        terms.append({
            "type": "a1",
            "a": c_k * (1.0 - _safe_log(tau_0)),
            "b": -c_k / tau_0,
        })
        terms.append({"type": "log_tau", "a": c_k})
        return terms

    if t_k == -1:
        # cp/R = c_k / T gives a tau*ln(tau) term.
        # With T=T_c/tau: cp/R = c_k * tau / T_c, so the effective "reduced"
        # coefficient is (c_k / T_c).
        # alpha_0_cp = -(c_k/T_c) * tau * ln(tau)
        #              + (c_k/T_c) * (1 + ln(tau_0)) * tau
        #              - (c_k/T_c) * tau_0
        ck_red = c_k / T_c
        terms.append({"type": "tau_log_tau", "a": -ck_red})
        terms.append({
            "type": "a1",
            "a": -ck_red * tau_0,
            "b": ck_red * (1.0 + _safe_log(tau_0)),
        })
        return terms

    # General case: t_k != 0 and t_k != -1.
    # Each "c_k * T^t_k" contribution to cp/R produces:
    #   power_tau: a = -c_k * T_c^t_k / (t_k*(t_k+1)), b = -t_k
    #   a1: const = c_k * T_0^t_k / t_k,
    #       tau-linear = -c_k * T_0^(t_k+1) / ((t_k+1)*T_c)
    Tc_pow = T_c ** t_k          # T_c^t_k
    T0_pow = T_0 ** t_k          # T_0^t_k
    T0_pow_plus = T_0 ** (t_k + 1.0)  # T_0^(t_k+1)
    inv_t = 1.0 / t_k
    inv_tp1 = 1.0 / (t_k + 1.0)
    terms.append({
        "type": "power_tau",
        "a": -c_k * Tc_pow * inv_t * inv_tp1,
        "b": -t_k,
    })
    terms.append({
        "type": "a1",
        "a": c_k * T0_pow * inv_t,                        # constant
        "b": -c_k * T0_pow_plus * inv_tp1 / T_c,          # tau-linear
    })
    return terms


def _safe_log(x: float) -> float:
    """ln(x) for positive x, with a clear error otherwise. tau_0 should always
    be positive (T_c, T_0 > 0), so any failure here is a data error."""
    import math
    if x <= 0:
        raise CoolPropSchemaError(
            f"CP0 term has non-positive tau_0 ({x}); check Tc and T0 fields"
        )
    return math.log(x)


def _convert_cp0_term(term, *, T_reduce: float):
    """Dispatch to the appropriate CP0 form converter."""
    ttype = term["type"]

    # Resolve T_0 and the reducing T_c the term was written against. CoolProp
    # always stores Tc explicitly in the CP0 term so we use it (it should
    # match T_reduce but doesn't always exactly due to rounding).
    Tc_term = float(term.get("Tc", T_reduce))
    T0 = float(term["T0"])
    tau_0 = Tc_term / T0

    if ttype == "IdealGasHelmholtzCP0Constant":
        c = float(term["cp_over_R"])
        return _cp0_polynomial_contribution(c, 0.0, tau_0, Tc_term, T0)

    if ttype == "IdealGasHelmholtzCP0PolyT":
        out = []
        for c_k, t_k in zip(term["c"], term["t"]):
            out.extend(_cp0_polynomial_contribution(
                float(c_k), float(t_k), tau_0, Tc_term, T0))
        return out

    if ttype == "IdealGasHelmholtzCP0AlyLee":
        # cp/R = c_0 + c_1*(c_2/T/sinh(c_2/T))^2 + c_3*(c_4/T/cosh(c_4/T))^2
        # The c_0 part is identical to a CP0Constant.
        # The (c_1, c_2) sinh^2 part maps to PE_sinh with (a=c_1, b=c_2/T_c).
        # The (c_3, c_4) cosh^2 part maps to PE_cosh with (a=-c_3, b=c_4/T_c).
        # Verification: cv_R contribution from PE_sinh(a, b) is
        #   +a * (b*tau)^2 / sinh^2(b*tau), which equals c_1*(c_2/T)^2/sinh^2(c_2/T)
        # when a=c_1, b=c_2/T_c (since b*tau = c_2/T_c * T_c/T = c_2/T). Similarly
        # PE_cosh(a, b) gives -a * (b*tau)^2 / cosh^2(b*tau) so a=-c_3 reproduces
        # the +c_3*(c_4/T)^2/cosh^2(c_4/T) target.
        coeffs = term["c"]
        if len(coeffs) != 5:
            raise CoolPropSchemaError(
                f"IdealGasHelmholtzCP0AlyLee expects 5 coefficients, "
                f"got {len(coeffs)}"
            )
        c0, c1, c2, c3, c4 = (float(x) for x in coeffs)
        out = []
        # constant term -> same as CP0Constant
        out.extend(_cp0_polynomial_contribution(c0, 0.0, tau_0, Tc_term, T0))
        # sinh^2 vibrational term
        if c2 != 0.0:
            out.append({"type": "PE_sinh", "a": c1, "b": c2 / T_reduce})
        # cosh^2 "anti-vibrational" / 2-state term
        if c4 != 0.0:
            out.append({"type": "PE_cosh", "a": -c3, "b": c4 / T_reduce})
        return out

    raise CoolPropSchemaError(
        f"_convert_cp0_term called on unexpected type {ttype!r}"
    )


def _main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="Path to CoolProp fluid JSON")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path for stateprop JSON (default: stdout)")
    ap.add_argument("--eos-index", type=int, default=0,
                    help="Which EOS within the file (default: 0)")
    ap.add_argument("--preserve-metadata", action="store_true",
                    help="Include CoolProp identifier metadata in output")
    args = ap.parse_args()

    with open(args.input) as f:
        cp = json.load(f)
    # CoolProp files contain a single fluid as a dict at top level, OR a list
    if isinstance(cp, list):
        if len(cp) != 1:
            raise SystemExit(
                f"Expected exactly one fluid in {args.input}, got {len(cp)}"
            )
        cp = cp[0]
    out = convert_fluid(cp, eos_index=args.eos_index,
                        preserve_unknown_metadata=args.preserve_metadata)
    text = json.dumps(out, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.output}", flush=True)
    else:
        print(text)


if __name__ == "__main__":
    _main()
