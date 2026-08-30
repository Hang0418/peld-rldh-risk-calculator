#!/usr/bin/env python3
"""Apply the untouched frozen v5 model to H5 and lock original results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from rldh_model_utils import (
    EPSILON,
    apply_platt,
    bootstrap_metric_intervals,
    decision_curve,
    fit_platt,
    metric_row,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
FROZEN_DATA = ROOT / "data" / "frozen" / VERSION
MODEL_DIR = ROOT / "outputs" / VERSION / "model_freeze"
OUTPUT = ROOT / "outputs" / VERSION / "external_validation" / "h5_primary"
DATA = FROZEN_DATA / "02_external_H5_SEALED.csv"
MODEL = MODEL_DIR / "frozen_model.joblib"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intercept_update(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    lp = logit(np.clip(probabilities, EPSILON, 1 - EPSILON))
    return float(brentq(lambda value: np.sum(y_true - expit(lp + value)), -30, 30))


def main() -> None:
    freeze = json.loads((MODEL_DIR / "freeze_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((FROZEN_DATA / "lock_manifest.json").read_text(encoding="utf-8"))
    if freeze["status"] != "FINAL_MODEL_FROZEN_H5_MAY_OPEN":
        raise RuntimeError("Final model is not frozen")
    if sha256(MODEL) != freeze["model_sha256"]:
        raise RuntimeError("Frozen model hash mismatch")
    if sha256(DATA) != lock["files"]["external_h5_sealed"]["sha256"]:
        raise RuntimeError("H5 data hash mismatch")

    data = pd.read_csv(DATA)
    if set(data["Hospital"].unique()) != {5}:
        raise RuntimeError("H5 input contains an unexpected hospital")
    artifact = joblib.load(MODEL)
    features = artifact["features"]
    raw = artifact["pipeline"].predict_proba(data[features])[:, 1]
    original = apply_platt(artifact["calibrator"], raw)
    y_true = data["Recurrence"].to_numpy(int)
    intercept_delta = intercept_update(y_true, original)
    lp = logit(np.clip(original, EPSILON, 1 - EPSILON))
    intercept_only = expit(lp + intercept_delta)
    logistic_model = fit_platt(y_true, original)
    logistic = apply_platt(logistic_model, original)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = data[["Hospital", "id", "Recurrence"]].copy()
    predictions["raw_probability"] = raw
    predictions["original_probability"] = original
    predictions["intercept_only_probability"] = intercept_only
    predictions["logistic_recalibrated_probability"] = logistic
    predictions.to_csv(OUTPUT / "h5_predictions.csv", index=False)

    metric_rows = []
    for label, column in [
        ("original_frozen_model", "original_probability"),
        ("secondary_intercept_only", "intercept_only_probability"),
        ("secondary_logistic_recalibration", "logistic_recalibrated_probability"),
    ]:
        metric_rows.append({"analysis": label, **metric_row(y_true, predictions[column])})
        dca = decision_curve(y_true, predictions[column], THRESHOLDS)
        dca.insert(0, "analysis", label)
        dca.to_csv(OUTPUT / f"h5_dca_{label}.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "h5_metrics.csv", index=False)
    bootstrap_metric_intervals(
        predictions.rename(columns={"original_probability": "probability"}),
        "probability",
        repetitions=2000,
    ).to_csv(OUTPUT / "h5_original_bootstrap_intervals.csv", index=False)

    status = {
        "version": VERSION,
        "status": "H5_ORIGINAL_RESULTS_LOCKED_H6_MAY_OPEN",
        "model_sha256": sha256(MODEL),
        "h5_data_sha256": sha256(DATA),
        "h5_predictions_sha256": sha256(OUTPUT / "h5_predictions.csv"),
        "original_results_are_immutable": True,
        "h5_intercept_update": intercept_delta,
        "h5_logistic_recalibration_intercept": float(logistic_model.intercept_[0]),
        "h5_logistic_recalibration_slope": float(logistic_model.coef_[0, 0]),
    }
    (OUTPUT / "h5_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "metrics": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
