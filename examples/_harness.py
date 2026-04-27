"""
examples_harness — utilities for the stateprop examples-as-tests
infrastructure.

Each example in `examples/` is simultaneously a tutorial and a
regression test.  This module provides the convention that makes
that work:

    from examples._harness import validate, summary, run_smoke

    # ... compute things ...

    validate(
        "MEA + 30 wt% + α=0.5 + 40 °C: P_CO2",
        reference=0.20,
        computed=res.P_CO2,
        units="bar",
        tol_rel=0.50,
        source="Jou-Mather-Otto 1995",
    )

    summary()    # prints pass/fail at end of script

The harness keeps track of every `validate()` call in the current
process.  When `summary()` is invoked it prints a concise table and
exits with a nonzero code if any check failed.  This makes the
example act like a self-contained test.

For pytest integration, see `tests/test_examples_run.py` which
imports each example as a subprocess and parses its summary output.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Check:
    name: str
    reference: float
    computed: float
    units: str
    tol_rel: float
    source: str
    rel_err: float
    passed: bool


# Module-level state — fine for a script's lifetime
_checks: List[Check] = []


def validate_bool(name: str,
                       *,
                       condition: bool,
                       detail: str = "",
                       source: str = "") -> bool:
    """Record a boolean validation check.

    For pass/fail conditions that aren't naturally a numerical
    comparison — e.g., "speedup ≥ 2×", "list contains expected
    keys", "no NaNs in result".

    Parameters
    ----------
    name : str
        Human-readable name of the check.
    condition : bool
        The thing that must be true.
    detail : str
        Optional context shown in the inline output.
    source : str
        Citation / reference.

    Returns
    -------
    bool
        Same as ``condition``.  Recorded for ``summary()``.
    """
    passed = bool(condition)
    _checks.append(Check(
        name=name, reference=1.0,
        computed=1.0 if passed else 0.0,
        units="bool", tol_rel=0.0, source=source,
        rel_err=0.0 if passed else 1.0,
        passed=passed,
    ))
    flag = "PASS" if passed else "FAIL"
    line = f"  [{flag}] {name}"
    if detail:
        line += f": {detail}"
    print(line)
    if source and not passed:
        print(f"         source: {source}")
    return passed


def validate(name: str,
              *,
              reference: float,
              computed: float,
              units: str = "-",
              tol_rel: float = 0.05,
              source: str = "",
              tol_abs: Optional[float] = None) -> bool:
    """Record a validation check against a published reference.

    Parameters
    ----------
    name : str
        Human-readable name of the quantity being checked.
    reference : float
        Published / textbook value.
    computed : float
        Value produced by the example code.
    units : str
        Display units (informational only).
    tol_rel : float, default 0.05
        Relative tolerance.  The check passes if
        ``|computed - reference| / max(|reference|, ε) ≤ tol_rel``,
        with ``ε`` chosen to avoid divide-by-zero on small references.
    source : str
        Citation for the reference value (textbook, paper, dataset).
    tol_abs : float, optional
        Absolute tolerance — if supplied, check passes if either the
        relative or absolute tolerance is satisfied.

    Returns
    -------
    bool
        True if the check passed, False otherwise.  The check is also
        recorded for the eventual ``summary()`` call.
    """
    eps = max(abs(reference) * 1e-9, 1e-15)
    abs_err = abs(computed - reference)
    rel_err = abs_err / max(abs(reference), eps)

    passed = rel_err <= tol_rel
    if tol_abs is not None and abs_err <= tol_abs:
        passed = True

    _checks.append(Check(
        name=name, reference=reference, computed=computed,
        units=units, tol_rel=tol_rel, source=source,
        rel_err=rel_err, passed=passed,
    ))

    # Print inline so users running the example see the result.
    # When reference is near zero, the relative error blows up; show
    # absolute error instead in that case.
    flag = "PASS" if passed else "FAIL"
    if abs(reference) < 1e-9:
        msg = (f"  [{flag}] {name}: "
               f"computed={computed:.6g} {units}, "
               f"reference={reference:.6g} {units}, "
               f"abs_err={abs_err:.3g}")
    else:
        msg = (f"  [{flag}] {name}: "
               f"computed={computed:.6g} {units}, "
               f"reference={reference:.6g} {units}, "
               f"rel_err={rel_err*100:.2f}%")
    print(msg)
    if source and not passed:
        print(f"         source: {source}")
    return passed


def summary(*, exit_on_fail: bool = True) -> bool:
    """Print a pass/fail summary and (optionally) exit nonzero on failure.

    Returns True if all checks passed, False otherwise.

    The default behavior of exiting on failure makes examples behave
    like CLI tests — running ``python examples/01_foo.py`` returns 0
    on pass and nonzero on regression.  The harness in
    ``tests/test_examples_run.py`` relies on this.
    """
    n = len(_checks)
    n_pass = sum(1 for c in _checks if c.passed)
    n_fail = n - n_pass

    print()
    print("=" * 60)
    print(f"VALIDATION SUMMARY: {n_pass}/{n} checks passed")
    if n_fail:
        print(f"  {n_fail} FAILURES:")
        for c in _checks:
            if not c.passed:
                if abs(c.reference) < 1e-9:
                    abs_err = abs(c.computed - c.reference)
                    print(f"    - {c.name}: abs_err={abs_err:.3g}")
                else:
                    print(f"    - {c.name}: rel_err={c.rel_err*100:.2f}% > "
                          f"tol={c.tol_rel*100:.2f}%")
                if c.source:
                    print(f"      reference source: {c.source}")
    print("=" * 60)

    all_pass = (n_fail == 0)
    if not all_pass and exit_on_fail:
        sys.exit(1)
    return all_pass


def reset() -> None:
    """Clear recorded checks.  Useful for interactive use / pytest."""
    _checks.clear()


def get_checks() -> List[Check]:
    """Return the list of recorded checks (read-only intent)."""
    return list(_checks)


# ----------------------------------------------------------------------
# Smoke-mode utilities
#
# Some examples take more than a few seconds (e.g., crude tower).  When
# being run by the example test harness, we want a short-circuit option
# so the test suite finishes in a reasonable time.  Examples can opt
# into this by checking ``smoke_mode()`` and reducing problem size.
# ----------------------------------------------------------------------

def smoke_mode() -> bool:
    """Return True if running under the example test harness.

    Set the environment variable ``STATEPROP_EXAMPLES_SMOKE=1`` to
    activate.  Examples can use this to reduce grid resolution, skip
    plotting, or run a smaller sub-case so that ``test_examples_run.py``
    finishes in bounded time.
    """
    return os.environ.get("STATEPROP_EXAMPLES_SMOKE", "") == "1"


def maybe_plot(plot_fn) -> None:
    """Invoke a plot function only if not in smoke mode and matplotlib
    is available.  Pattern:

        from examples._harness import maybe_plot
        def _plot():
            import matplotlib.pyplot as plt
            ...
            plt.show()
        maybe_plot(_plot)
    """
    if smoke_mode():
        return
    try:
        plot_fn()
    except ImportError:
        pass
    except Exception as e:
        print(f"  (plot skipped: {e})")
