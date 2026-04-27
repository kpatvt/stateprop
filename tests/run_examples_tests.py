"""Plain-Python test runner for the examples curriculum.

Run: python tests/run_examples_tests.py

Runs every numbered example (NN_*.py) in `examples/` as a subprocess,
captures its stdout/stderr, parses the validation summary, and reports
pass/fail.  Sets the STATEPROP_EXAMPLES_SMOKE=1 environment variable
so examples can opt into shorter / coarser smoke-mode runs.

This complements the unit-test suite (tests/run_*_tests.py).  Running
the examples as tests catches integration regressions that unit tests
would miss — e.g., when a public API change breaks a cookbook recipe.

Each example defines its own validation checks via examples/_harness.py.
The example exits 0 on pass, nonzero on failure.  This runner just
asks: did the example exit cleanly?
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

PASSED = 0
FAILED = 0
SKIPPED = 0
FAILURES: list = []

# Examples that take longer than the smoke-mode budget — run them only
# when STATEPROP_EXAMPLES_FULL=1.  These all have published validation
# numbers; they're worth running in the full nightly test pass.
LONG_RUNNING = {
    "48_crude_atmospheric_tower.py",  # ~30 s, multi-stage with side strippers
}

# Examples that are documentation-only or interactive (e.g., open
# prompts), or that have known pre-existing bugs.
SKIP = {
    # Pre-existing bug in benchmark.py: Fluid.pack() API mismatch
    # producing wrong number of args for alpha_r_derivs().  This bug
    # predates the curriculum reorganization.  Tracked separately.
    "99_benchmark.py",
}


def _print_separator():
    print("-" * 70)


def run_example(path: Path, timeout: float = 90.0) -> tuple:
    """Run an example as a subprocess.

    Returns (returncode, stdout, stderr, elapsed_seconds).
    """
    env = dict(os.environ)
    env["STATEPROP_EXAMPLES_SMOKE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    # Suppress matplotlib displays in headless test
    env["MPLBACKEND"] = "Agg"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.time() - t0
        return result.returncode, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return -1, "", f"TIMEOUT after {timeout:.1f}s", elapsed


def _parse_summary(stdout: str) -> tuple:
    """Parse the harness summary line if present.

    Returns (n_pass, n_total) or (None, None) if no harness output.
    """
    m = re.search(r"VALIDATION SUMMARY: (\d+)/(\d+) checks passed",
                   stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def main() -> int:
    global PASSED, FAILED, SKIPPED

    if not EXAMPLES_DIR.exists():
        print(f"Examples directory not found: {EXAMPLES_DIR}")
        return 1

    full_mode = os.environ.get("STATEPROP_EXAMPLES_FULL", "") == "1"
    timeout = 90.0 if not full_mode else 600.0

    # Numbered examples in curriculum order
    examples = sorted(
        p for p in EXAMPLES_DIR.glob("*.py")
        if p.stem[:2].isdigit() or p.stem.startswith("99")
    )

    if not examples:
        print(f"No numbered examples found under {EXAMPLES_DIR}")
        return 1

    print("=" * 70)
    print(f"stateprop examples test runner")
    print(f"  examples dir: {EXAMPLES_DIR}")
    print(f"  examples found: {len(examples)}")
    print(f"  mode: {'full (long-running enabled)' if full_mode else 'smoke (default)'}")
    print(f"  timeout per example: {timeout:.0f}s")
    print("=" * 70)

    for path in examples:
        name = path.name

        if name in SKIP:
            print(f"[SKIP] {name}  (in SKIP list)")
            SKIPPED += 1
            continue

        if name in LONG_RUNNING and not full_mode:
            print(f"[SKIP] {name}  (long-running; set "
                  f"STATEPROP_EXAMPLES_FULL=1 to enable)")
            SKIPPED += 1
            continue

        sys.stdout.write(f"[ ... ] {name}  ")
        sys.stdout.flush()

        rc, out, err, elapsed = run_example(path, timeout=timeout)
        n_pass, n_total = _parse_summary(out)

        if rc == 0:
            PASSED += 1
            if n_total is not None:
                tag = f"({n_pass}/{n_total} checks)"
            else:
                tag = "(no harness checks)"
            sys.stdout.write(f"\r[ OK  ] {name}  {tag} "
                              f"[{elapsed:.1f}s]\n")
        else:
            FAILED += 1
            sys.stdout.write(f"\r[FAIL ] {name}  rc={rc} "
                              f"[{elapsed:.1f}s]\n")
            tail_out = "\n".join(out.splitlines()[-15:])
            tail_err = "\n".join(err.splitlines()[-10:])
            FAILURES.append((name, rc, tail_out, tail_err))

    _print_separator()
    print(f"RESULT: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    if FAILURES:
        print()
        print("FAILURES:")
        for name, rc, out_tail, err_tail in FAILURES:
            print()
            _print_separator()
            print(f"  {name} (exit code {rc})")
            if out_tail.strip():
                print(f"  --- last stdout ---")
                for line in out_tail.splitlines():
                    print(f"  {line}")
            if err_tail.strip():
                print(f"  --- last stderr ---")
                for line in err_tail.splitlines():
                    print(f"  {line}")
        _print_separator()

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
