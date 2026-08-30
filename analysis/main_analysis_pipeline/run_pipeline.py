#!/usr/bin/env python3
"""Concise entry point for the active PELD-RLDH V5 analysis pipeline.

The authoritative analysis remains in ``scripts/``. This file only groups the
main steps, enforces the prespecified cohort gates, and provides a safe runner.
It does not duplicate model code or introduce a new analysis lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
SOURCE = ROOT / "PELD术后RLDH.xlsx"
EXPECTED_SOURCE_SHA256 = (
    "9b8ca4c6d72c53fe64e8847fd8beb537a43a87ac73de84bf7f3de4285f7533a0"
)
OUTPUT = ROOT / "outputs" / VERSION


def read_json(relative_path: str) -> dict[str, Any] | None:
    path = ROOT / relative_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_complete() -> bool:
    status = read_json(
        f"outputs/{VERSION}/stage0_1_audit/stage0_1_audit_status.json"
    )
    checks = (status or {}).get("hard_checks", {})
    return bool(checks) and all(checks.values())


def development_complete() -> bool:
    status = read_json(
        f"outputs/{VERSION}/model_development_h1_h4/development_status.json"
    )
    verification = read_json(
        f"outputs/{VERSION}/model_development_h1_h4/independent_verification.json"
    )
    return (
        (status or {}).get("status") == "FINAL_MODEL_FROZEN_H5_MAY_OPEN"
        and (verification or {}).get("status") == "PASS"
    )


def h5_complete() -> bool:
    status = read_json(
        f"outputs/{VERSION}/external_validation/h5_primary/h5_status.json"
    )
    return (status or {}).get("status") == "H5_ORIGINAL_RESULTS_LOCKED_H6_MAY_OPEN"


def h6_complete() -> bool:
    status = read_json(
        f"outputs/{VERSION}/external_validation/h6_stress_test/h6_status.json"
    )
    verification = read_json(
        f"outputs/{VERSION}/external_validation/independent_verification.json"
    )
    return (
        (status or {}).get("status")
        == "H6_ORIGINAL_RESULTS_LOCKED_EXTERNAL_VALIDATION_COMPLETE"
        and (verification or {}).get("status") == "PASS"
    )


def verification_complete(relative_path: str) -> bool:
    status = read_json(relative_path)
    return (status or {}).get("status") == "PASS"


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    purpose: str
    scripts: tuple[str, ...]
    prerequisite: Callable[[], bool] | None
    complete: Callable[[], bool]
    gate: str


STAGES = (
    Stage(
        key="audit",
        title="1. Data audit and cohort lock",
        purpose="Audit the updated workbook and freeze H1-H4, H5, and H6 by role.",
        scripts=("scripts/01_lock_and_audit_updated_data.py",),
        prerequisite=None,
        complete=audit_complete,
        gate="All schema, uniqueness, outcome, missingness, and finite-value checks pass.",
    ),
    Stage(
        key="development",
        title="2. H1-H4 development, nested IECV, model freeze",
        purpose="Compare prespecified candidates using development hospitals only.",
        scripts=(
            "scripts/02_develop_and_freeze_h1_h4.py",
            "scripts/03_verify_development_stop.py",
        ),
        prerequisite=audit_complete,
        complete=development_complete,
        gate="Frozen-model status allows H5 to open and independent verification passes.",
    ),
    Stage(
        key="h5",
        title="3. H5 primary external validation",
        purpose="Apply the untouched frozen model to H5 and lock original results.",
        scripts=("scripts/05_external_validate_h5.py",),
        prerequisite=development_complete,
        complete=h5_complete,
        gate="H5 original predictions are immutable before H6 is opened.",
    ),
    Stage(
        key="h6",
        title="4. H6 independent stress test",
        purpose="Apply the same frozen model to H6, then verify both external cohorts.",
        scripts=(
            "scripts/06_external_validate_h6.py",
            "scripts/07_verify_external_validation.py",
        ),
        prerequisite=h5_complete,
        complete=h6_complete,
        gate="H6 is locked and external-validation verification passes.",
    ),
    Stage(
        key="extensions",
        title="5. Frozen-model interpretation and robustness",
        purpose="Generate Stage 14 stability, subgroup, threshold, and sensitivity analyses.",
        scripts=(
            "scripts/08_stage14_extensions.py",
            "scripts/09_verify_stage14_extensions.py",
        ),
        prerequisite=h6_complete,
        complete=lambda: verification_complete(
            f"outputs/{VERSION}/stage14_extensions/stage14_independent_verification.json"
        ),
        gate="Stage 14 verification passes without refitting or reselection.",
    ),
    Stage(
        key="manuscript_validation",
        title="6. Manuscript-level validation",
        purpose="Assess calibration, dataset shift, risk strata, and equation reproducibility.",
        scripts=(
            "scripts/10_manuscript_validation_analyses.py",
            "scripts/11_verify_manuscript_validation.py",
        ),
        prerequisite=lambda: verification_complete(
            f"outputs/{VERSION}/stage14_extensions/stage14_independent_verification.json"
        ),
        complete=lambda: verification_complete(
            f"outputs/{VERSION}/stage15_manuscript_validation/stage15_independent_verification.json"
        ),
        gate="Stage 15 verification passes; no external cutoff optimization is allowed.",
    ),
    Stage(
        key="assets",
        title="7. Main manuscript tables and figures",
        purpose="Build and verify deterministic journal assets from locked results.",
        scripts=(
            "scripts/12_build_jama_manuscript_assets.py",
            "scripts/13_verify_jama_manuscript_assets.py",
        ),
        prerequisite=lambda: verification_complete(
            f"outputs/{VERSION}/stage15_manuscript_validation/stage15_independent_verification.json"
        ),
        complete=lambda: verification_complete(
            f"outputs/{VERSION}/manuscript_assets/asset_verification.json"
        ),
        gate="Asset verification passes; this step performs no model fitting.",
    ),
)


def check_source_lineage() -> None:
    if not SOURCE.exists():
        raise RuntimeError(f"Source workbook is missing: {SOURCE}")
    observed = sha256(SOURCE)
    if observed != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Source workbook hash changed. Do not reuse the V5 pipeline/results. "
            f"Expected {EXPECTED_SOURCE_SHA256}, observed {observed}."
        )


def selected_stages(from_stage: str, to_stage: str) -> tuple[Stage, ...]:
    keys = [stage.key for stage in STAGES]
    start = keys.index(from_stage)
    end = keys.index(to_stage)
    if start > end:
        raise ValueError("--from-stage must precede --to-stage")
    return STAGES[start : end + 1]


def print_plan(stages: tuple[Stage, ...], python: str) -> None:
    check_source_lineage()
    print(f"Lineage: {VERSION}")
    print(f"Source SHA-256: {EXPECTED_SOURCE_SHA256}")
    for stage in stages:
        state = "COMPLETE" if stage.complete() else "PENDING"
        print(f"\n[{state}] {stage.title}")
        print(f"  Purpose: {stage.purpose}")
        for script in stage.scripts:
            print(f"  Command: {python} {script}")
        print(f"  Gate: {stage.gate}")


def run(stages: tuple[Stage, ...], python: str) -> None:
    check_source_lineage()
    for stage in stages:
        if stage.complete():
            print(f"[SKIP] {stage.title}: verified outputs already exist.")
            continue
        if stage.prerequisite is not None and not stage.prerequisite():
            raise RuntimeError(f"Prerequisite gate failed before {stage.title}")
        print(f"[RUN] {stage.title}")
        for script in stage.scripts:
            subprocess.run(
                [python, str(ROOT / script)],
                cwd=ROOT,
                check=True,
            )
        if not stage.complete():
            raise RuntimeError(f"Completion gate failed after {stage.title}")
        print(f"[PASS] {stage.gate}")


def parse_args() -> argparse.Namespace:
    keys = [stage.key for stage in STAGES]
    parser = argparse.ArgumentParser(
        description="Plan or run the seven main stages of the active RLDH V5 pipeline."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute pending stages. Without this flag, only print the plan.",
    )
    parser.add_argument("--from-stage", choices=keys, default=keys[0])
    parser.add_argument("--to-stage", choices=keys, default=keys[-1])
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the authoritative scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = selected_stages(args.from_stage, args.to_stage)
    if args.run:
        run(stages, args.python)
    else:
        print_plan(stages, args.python)


if __name__ == "__main__":
    main()
