#!/usr/bin/env python3
"""Apply the untouched frozen v5 model to H6 after H5 results are locked."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from rldh_model_utils import EPSILON, apply_platt, bootstrap_metric_intervals, decision_curve, metric_row


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
FROZEN_DATA = ROOT / "data" / "frozen" / VERSION
MODEL_DIR = ROOT / "outputs" / VERSION / "model_freeze"
H5_DIR = ROOT / "outputs" / VERSION / "external_validation" / "h5_primary"
OUTPUT = ROOT / "outputs" / VERSION / "external_validation" / "h6_stress_test"
DATA = FROZEN_DATA / "03_external_H6_SEALED.csv"
MODEL = MODEL_DIR / "frozen_model.joblib"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    h5_status = json.loads((H5_DIR / "h5_status.json").read_text(encoding="utf-8"))
    lock = json.loads((FROZEN_DATA / "lock_manifest.json").read_text(encoding="utf-8"))
    if h5_status["status"] != "H5_ORIGINAL_RESULTS_LOCKED_H6_MAY_OPEN":
        raise RuntimeError("H5 original results are not locked")
    if sha256(MODEL) != h5_status["model_sha256"]:
        raise RuntimeError("Frozen model changed after H5")
    if sha256(DATA) != lock["files"]["external_h6_sealed"]["sha256"]:
        raise RuntimeError("H6 data hash mismatch")

    data = pd.read_csv(DATA)
    if set(data["Hospital"].unique()) != {6}:
        raise RuntimeError("H6 input contains an unexpected hospital")
    artifact = joblib.load(MODEL)
    raw = artifact["pipeline"].predict_proba(data[artifact["features"]])[:, 1]
    original = apply_platt(artifact["calibrator"], raw)
    lp = logit(np.clip(original, EPSILON, 1 - EPSILON))
    transported_intercept = expit(lp + h5_status["h5_intercept_update"])
    transported_logistic = expit(
        h5_status["h5_logistic_recalibration_intercept"]
        + h5_status["h5_logistic_recalibration_slope"] * lp
    )
    y_true = data["Recurrence"].to_numpy(int)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = data[["Hospital", "id", "Recurrence"]].copy()
    predictions["raw_probability"] = raw
    predictions["original_probability"] = original
    predictions["h5_intercept_transported_probability"] = transported_intercept
    predictions["h5_logistic_transported_probability"] = transported_logistic
    predictions.to_csv(OUTPUT / "h6_predictions.csv", index=False)

    metric_rows = []
    for label, column in [
        ("original_frozen_model", "original_probability"),
        ("secondary_h5_intercept_transported", "h5_intercept_transported_probability"),
        ("secondary_h5_logistic_transported", "h5_logistic_transported_probability"),
    ]:
        metric_rows.append({"analysis": label, **metric_row(y_true, predictions[column])})
        dca = decision_curve(y_true, predictions[column], THRESHOLDS)
        dca.insert(0, "analysis", label)
        dca.to_csv(OUTPUT / f"h6_dca_{label}.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "h6_metrics.csv", index=False)
    bootstrap_metric_intervals(
        predictions.rename(columns={"original_probability": "probability"}),
        "probability",
        repetitions=2000,
    ).to_csv(OUTPUT / "h6_original_bootstrap_intervals.csv", index=False)

    status = {
        "version": VERSION,
        "status": "H6_ORIGINAL_RESULTS_LOCKED_EXTERNAL_VALIDATION_COMPLETE",
        "model_sha256": sha256(MODEL),
        "h5_predictions_sha256": sha256(H5_DIR / "h5_predictions.csv"),
        "h6_data_sha256": sha256(DATA),
        "h6_predictions_sha256": sha256(OUTPUT / "h6_predictions.csv"),
        "original_results_are_immutable": True,
    }
    (OUTPUT / "h6_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "metrics": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
