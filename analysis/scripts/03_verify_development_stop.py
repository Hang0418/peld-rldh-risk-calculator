#!/usr/bin/env python3
"""Independent structural and numerical verification of the v5 IECV decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rldh_model_utils import auc_i_squared, metric_row


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
DATA = ROOT / "data" / "frozen" / VERSION / "01_development_H1-H4.csv"
OUTPUT = ROOT / "outputs" / VERSION / "model_development_h1_h4"
PREDICTIONS = OUTPUT / "iecv_predictions.csv"
GATE = OUTPUT / "candidate_transportability_gate.csv"
STATUS = OUTPUT / "development_status.json"
FREEZE = ROOT / "outputs" / VERSION / "model_freeze" / "frozen_model.joblib"
EXTERNAL_OUTPUT = ROOT / "outputs" / VERSION / "external_validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True))


def main() -> None:
    development = pd.read_csv(DATA)
    predictions = pd.read_csv(PREDICTIONS)
    gate = pd.read_csv(GATE)
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    candidates = sorted(predictions["candidate"].unique())
    any_eligible = bool(gate["eligible"].any())
    expected_status = (
        "FINAL_MODEL_FROZEN_H5_MAY_OPEN"
        if any_eligible
        else "STOP_NO_CANDIDATE_PASSED_TRANSPORTABILITY_GATES"
    )
    checks: dict[str, bool] = {
        "development_contains_h1_h4_only": set(development["Hospital"].unique()) == {1, 2, 3, 4},
        "predictions_contain_h1_h4_only": set(predictions["Hospital"].unique()) == {1, 2, 3, 4},
        "eight_candidates": len(candidates) == 8,
        "prediction_rows_exact": len(predictions) == len(development) * len(candidates),
        "prediction_keys_unique": not predictions.duplicated(["candidate", "Hospital", "id"]).any(),
        "probabilities_finite": bool(np.isfinite(predictions[["raw_probability", "probability"]]).all().all()),
        "probabilities_bounded": bool(
            predictions[["raw_probability", "probability"]].ge(0).all().all()
            and predictions[["raw_probability", "probability"]].le(1).all().all()
        ),
        "status_matches_recalculated_gate_path": status["status"] == expected_status,
        "status_external_false": status["external_data_opened"] is False,
        "frozen_model_presence_matches_gate_path": FREEZE.exists() == any_eligible,
        "no_external_output": not EXTERNAL_OUTPUT.exists(),
    }

    source_outcomes = development[["Hospital", "id", "Recurrence"]]
    merged = predictions.merge(source_outcomes, on=["Hospital", "id"], suffixes=("_prediction", "_source"))
    checks["outcomes_match_source"] = bool(
        (merged["Recurrence_prediction"] == merged["Recurrence_source"]).all()
    )

    maximum_metric_difference = 0.0
    recalculated_rows = []
    for candidate, candidate_data in predictions.groupby("candidate"):
        pooled = metric_row(candidate_data["Recurrence"].to_numpy(), candidate_data["probability"].to_numpy())
        center_rows = []
        for hospital, center_data in candidate_data.groupby("Hospital"):
            row = {"Hospital": hospital, **metric_row(center_data["Recurrence"], center_data["probability"])}
            center_rows.append(row)
        center_frame = pd.DataFrame(center_rows)
        recalculated = {
            "candidate": candidate,
            **pooled,
            "minimum_center_auroc": float(center_frame["auroc"].min()),
            "minimum_center_slope": float(center_frame["calibration_slope"].min()),
            "maximum_center_slope": float(center_frame["calibration_slope"].max()),
            "auroc_i_squared": auc_i_squared(center_frame),
        }
        recalculated_rows.append(recalculated)
        observed = gate[gate["candidate"] == candidate].iloc[0]
        for metric in [
            "auroc",
            "auprc",
            "brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "oe_ratio",
            "minimum_center_auroc",
            "minimum_center_slope",
            "maximum_center_slope",
            "auroc_i_squared",
        ]:
            difference = abs(float(observed[metric]) - float(recalculated[metric]))
            maximum_metric_difference = max(maximum_metric_difference, difference)
            checks[f"metric_{candidate}_{metric}"] = close(float(observed[metric]), float(recalculated[metric]))

    recalculated = pd.DataFrame(recalculated_rows)
    recalculated["eligible"] = (
        (recalculated["minimum_center_auroc"] >= 0.65)
        & recalculated["calibration_slope"].between(0.70, 1.30)
        & (recalculated["minimum_center_slope"] >= 0.50)
        & (recalculated["maximum_center_slope"] <= 1.50)
        & (recalculated["auroc_i_squared"] <= 50.0)
    )
    checks["recalculated_eligibility_matches_gate"] = bool(
        np.array_equal(
            recalculated.sort_values("candidate")["eligible"].to_numpy(),
            gate.sort_values("candidate")["eligible"].to_numpy(),
        )
    )
    if any_eligible:
        checks["selected_candidate_is_eligible"] = bool(
            gate.loc[gate["eligible"], "candidate"].eq(status["selected_candidate"]).any()
        )
        checks["frozen_model_hash_matches_status"] = (
            sha256(FREEZE) == status["model_sha256"] if FREEZE.exists() else False
        )

    core_files = [
        DATA,
        PREDICTIONS,
        GATE,
        OUTPUT / "candidate_center_metrics.csv",
        OUTPUT / "stable_feature_selection.csv",
        STATUS,
    ]
    verification = {
        "version": VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "maximum_metric_difference": maximum_metric_difference,
        "file_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in core_files},
    }
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    if verification["status"] != "PASS":
        raise RuntimeError(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
