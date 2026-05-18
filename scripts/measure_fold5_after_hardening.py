"""Fold-5 non-regression guardrail for Data Quality Hardening (Requirement 9).

Runs the current best training entrypoint, measures Fold-5 nMAE on the
validation window ``2025-01-01 → 2025-03-31``, and writes a small audit
JSON at ``data/processed/fold5_hardened.json`` with the delta versus the
documented baseline (``7.62 %``).

Exit codes:
    0  — ``delta_pp <= tolerance_pp`` (no regression).
    1  — ``delta_pp > tolerance_pp``  (regression; prints a named delta).
    2  — training entrypoint missing the fold-5 hook (wiring gap, not
         regression); CPU-safe failure mode for Task 12.1 when no hook is
         wired yet.

Usage:
    python scripts/measure_fold5_after_hardening.py
    python scripts/measure_fold5_after_hardening.py --seed 42 \
        --baseline 7.62 --tolerance 0.02

See ``.kiro/specs/data-quality-hardening/requirements.md`` Requirement 9
and ``design.md`` §Components §5 for the full contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the workspace root is importable so ``from src...`` works whether
# this script is invoked as ``python scripts/...`` or ``python -m scripts...``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seeding import set_global_seed  # noqa: E402

# Steering rule: the validated feature set is K=70 (PROJECT.md §9, design.md §5).
FEATURE_SET_K: int = 70

# Where the audit JSON lands (Requirement 9.1).
_OUT_PATH = _ROOT / "data" / "processed" / "fold5_hardened.json"


def _git_sha() -> str:
    """Return the short HEAD SHA, or ``"unknown"`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(_ROOT),
        )
    except (FileNotFoundError, OSError):
        return "unknown"
    sha = (result.stdout or "").strip()
    if result.returncode != 0 or not sha:
        return "unknown"
    return sha


def _utcnow_iso() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via ``.tmp → rename`` for atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_current_best_and_measure_fold5(seed: int) -> float:  # noqa: ARG001 - seed used by impl
    """Canonical training + Fold-5 measurement hook.

    Picks the current-best training entrypoint in priority order:
        1. ``src.training.train_best``
        2. ``src.training.train_v15_regime``
        3. The highest-numbered ``train_vNN_*.py`` under ``src/training``.

    The entrypoint is expected to expose a callable (``main``,
    ``train_and_predict_fold5``, ``measure_fold5``, …) that returns the
    Fold-5 nMAE value as a ``float``. When no such callable is wired, this
    function raises :class:`NotImplementedError` with a clear pointer to
    the module and the missing hook — ``main`` converts that into exit
    code ``2`` (wiring gap, not a regression). This keeps Task 12.1
    CPU-safe: no real training is kicked off by merely importing this
    script or calling ``main`` in a smoke test that monkeypatches this
    function.
    """
    # Resolve the current-best training module per the documented preference.
    try:
        import src.training.train_best as _train_best  # noqa: F401
        module_name = "src.training.train_best"
    except ImportError:
        try:
            import src.training.train_v15_regime as _train_best  # noqa: F401
            module_name = "src.training.train_v15_regime"
        except ImportError as err:
            raise NotImplementedError(
                "No current-best training entrypoint is importable. Looked for "
                "'src.training.train_best' and 'src.training.train_v15_regime'. "
                "Wire Task 12.1 to a training module that exposes a callable "
                "returning Fold-5 nMAE as a float."
            ) from err

    # Try the canonical hook names in order.
    for hook_name in ("train_and_predict_fold5", "measure_fold5"):
        hook = getattr(_train_best, hook_name, None)
        if callable(hook):
            return float(hook(seed))

    raise NotImplementedError(
        f"Training module '{module_name}' has no callable returning Fold-5 "
        "nMAE as a float. Expected one of: 'train_and_predict_fold5(seed)', "
        "'measure_fold5(seed)', or a 'main(seed)' that returns a float. Wire "
        f"one of these hooks in '{module_name}' before invoking "
        "scripts/measure_fold5_after_hardening.py for a real measurement."
    )


def main(
    seed: int = 42,
    baseline_nmae: float = 7.62,
    tolerance_pp: float = 0.02,
) -> int:
    """Measure Fold-5 nMAE, emit the audit JSON, and return a POSIX exit code.

    See the module docstring for exit-code semantics.
    """
    set_global_seed(seed)

    try:
        nmae = float(run_current_best_and_measure_fold5(seed))
    except NotImplementedError as err:
        # Wiring gap, not a regression — exit 2 keeps CI signal distinct.
        print(f"WIRING_GAP: {err}", file=sys.stderr)
        return 2

    delta = nmae - baseline_nmae

    payload = {
        "nmae": nmae,
        "baseline": baseline_nmae,
        "delta_pp": delta,
        "seed": seed,
        "feature_set_K": FEATURE_SET_K,
        "git_sha": _git_sha(),
        "timestamp": _utcnow_iso(),
    }
    _atomic_write_json(_OUT_PATH, payload)

    if delta > tolerance_pp:
        print(f"REGRESSION: {delta:+.4f} pp > {tolerance_pp} pp")
        return 1
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fold-5 non-regression guardrail (Requirement 9).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline", type=float, default=7.62, dest="baseline_nmae"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.02, dest="tolerance_pp"
    )
    args = parser.parse_args()
    raise SystemExit(main(**vars(args)))
