#!/usr/bin/env python3
"""Independently verify frozen-model H5/H6 predictions, metrics, and hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rldh_model_utils import apply_platt, metric_row


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
BASE = ROOT / "outputs" / VERSION
DATA = ROOT / "data" / "frozen" / VERSION
MODEL = BASE / "model_freeze" / "frozen_model.joblib"
H5 = BASE / "external_validation" / "h5_primary"
H6 = BASE / "external_validation" / "h6_stress_test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True))


def verify_cohort(
    label: str,
    source_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    probability_columns: dict[str, str],
    expected_hospital: int,
    artifact: dict,
    checks: dict[str, bool],
) -> float:
    source = pd.read_csv(source_path)
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    checks[f"{label}_hospital_role"] = set(source["Hospital"].unique()) == {expected_hospital}
    checks[f"{label}_row_count"] = len(source) == len(predictions)
    checks[f"{label}_prediction_keys_unique"] = not predictions.duplicated(["Hospital", "id"]).any()
    merged = predictions.merge(
        source[["Hospital", "id", "Recurrence"]],
        on=["Hospital", "id"],
        suffixes=("_prediction", "_source"),
        validate="one_to_one",
    )
    checks[f"{label}_outcomes_match"] = bool(
        (merged["Recurrence_prediction"] == merged["Recurrence_source"]).all()
    )
    direct_raw = artifact["pipeline"].predict_proba(source[artifact["features"]])[:, 1]
    direct_original = apply_platt(artifact["calibrator"], direct_raw)
    checks[f"{label}_raw_predictions_match_frozen_model"] = bool(
        np.allclose(direct_raw, predictions["raw_probability"], rtol=1e-12, atol=1e-12)
    )
    checks[f"{label}_original_predictions_match_frozen_model"] = bool(
        np.allclose(
            direct_original,
            predictions["original_probability"],
            rtol=1e-12,
            atol=1e-12,
        )
    )
    maximum_difference = 0.0
    for analysis, probability_column in probability_columns.items():
        recalculated = metric_row(predictions["Recurrence"], predictions[probability_column])
        observed = metrics[metrics["analysis"] == analysis].iloc[0]
        for metric, value in recalculated.items():
            difference = abs(float(observed[metric]) - float(value))
            maximum_difference = max(maximum_difference, difference)
            checks[f"{label}_{analysis}_{metric}"] = close(float(observed[metric]), float(value))
    return maximum_difference


def main() -> None:
    artifact = joblib.load(MODEL)
    h5_status = json.loads((H5 / "h5_status.json").read_text(encoding="utf-8"))
    h6_status = json.loads((H6 / "h6_status.json").read_text(encoding="utf-8"))
    checks = {
        "h5_status_locked": h5_status["status"] == "H5_ORIGINAL_RESULTS_LOCKED_H6_MAY_OPEN",
        "h6_status_complete": h6_status["status"]
        == "H6_ORIGINAL_RESULTS_LOCKED_EXTERNAL_VALIDATION_COMPLETE",
        "model_hash_matches_h5": sha256(MODEL) == h5_status["model_sha256"],
        "model_hash_matches_h6": sha256(MODEL) == h6_status["model_sha256"],
        "h5_prediction_hash_locked": sha256(H5 / "h5_predictions.csv")
        == h5_status["h5_predictions_sha256"],
        "h5_hash_unchanged_before_h6": sha256(H5 / "h5_predictions.csv")
        == h6_status["h5_predictions_sha256"],
        "h6_prediction_hash_locked": sha256(H6 / "h6_predictions.csv")
        == h6_status["h6_predictions_sha256"],
    }
    maximum_difference = verify_cohort(
        "h5",
        DATA / "02_external_H5_SEALED.csv",
        H5 / "h5_predictions.csv",
        H5 / "h5_metrics.csv",
        {
            "original_frozen_model": "original_probability",
            "secondary_intercept_only": "intercept_only_probability",
            "secondary_logistic_recalibration": "logistic_recalibrated_probability",
        },
        5,
        artifact,
        checks,
    )
    maximum_difference = max(
        maximum_difference,
        verify_cohort(
            "h6",
            DATA / "03_external_H6_SEALED.csv",
            H6 / "h6_predictions.csv",
            H6 / "h6_metrics.csv",
            {
                "original_frozen_model": "original_probability",
                "secondary_h5_intercept_transported": "h5_intercept_transported_probability",
                "secondary_h5_logistic_transported": "h5_logistic_transported_probability",
            },
            6,
            artifact,
            checks,
        ),
    )
    result = {
        "version": VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "maximum_metric_difference": maximum_difference,
    }
    path = BASE / "external_validation" / "independent_verification.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
