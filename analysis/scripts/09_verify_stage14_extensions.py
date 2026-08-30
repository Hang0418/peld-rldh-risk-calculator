#!/usr/bin/env python3
"""Independent verification of Stage 14 frozen-model extension artifacts."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from rldh_model_utils import apply_platt, metric_row


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
BASE = ROOT / "outputs" / VERSION
FROZEN = ROOT / "data" / "frozen" / VERSION
DEV_OUTPUT = BASE / "model_development_h1_h4"
H5_OUTPUT = BASE / "external_validation" / "h5_primary"
H6_OUTPUT = BASE / "external_validation" / "h6_stress_test"
OUTPUT = BASE / "stage14_extensions"
MODEL = BASE / "model_freeze" / "frozen_model.joblib"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


warnings.filterwarnings("ignore", message="Setting penalty=None will ignore.*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True))


def simple_metrics(y_true: pd.Series, probability: pd.Series) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def cohorts() -> dict[str, pd.DataFrame]:
    development = pd.read_csv(FROZEN / "01_development_H1-H4.csv")
    predictions = pd.read_csv(DEV_OUTPUT / "iecv_predictions.csv")
    selected = predictions[predictions["candidate"] == "M2_Stable_Core_RCS_Ridge"]
    development = development.merge(
        selected[["Hospital", "id", "probability"]], on=["Hospital", "id"], validate="one_to_one"
    )
    h5 = pd.read_csv(FROZEN / "02_external_H5_SEALED.csv").merge(
        pd.read_csv(H5_OUTPUT / "h5_predictions.csv")[["Hospital", "id", "original_probability"]],
        on=["Hospital", "id"], validate="one_to_one"
    ).rename(columns={"original_probability": "probability"})
    h6 = pd.read_csv(FROZEN / "03_external_H6_SEALED.csv").merge(
        pd.read_csv(H6_OUTPUT / "h6_predictions.csv")[["Hospital", "id", "original_probability"]],
        on=["Hospital", "id"], validate="one_to_one"
    ).rename(columns={"original_probability": "probability"})
    return {"H1-H4 IECV": development, "H5 primary": h5, "H6 stress": h6}


def main() -> None:
    manifest = json.loads((OUTPUT / "stage14_manifest.json").read_text(encoding="utf-8"))
    artifact = joblib.load(MODEL)
    data_by_cohort = cohorts()
    checks: dict[str, bool] = {
        "manifest_pending_verification": manifest["status"]
        == "STAGE14_EXTENSIONS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "frozen_model_hash_matches": sha256(MODEL) == manifest["frozen_model_sha256"],
        "no_refit_or_reselection": manifest["refit_or_reselection_performed"] is False,
        "external_not_used_for_selection": manifest["h5_h6_used_for_model_selection"] is False,
    }
    for relative, expected_hash in manifest["files"].items():
        checks[f"hash_{relative}"] = sha256(ROOT / relative) == expected_hash

    maximum_difference = 0.0
    predictions = pd.read_csv(DEV_OUTPUT / "iecv_predictions.csv")
    incremental = pd.read_csv(OUTPUT / "paired_incremental_performance_bootstrap.csv")
    core = predictions[predictions["candidate"] == "M2_Stable_Core_RCS_Ridge"]
    for comparator in ["M3_Full_ElasticNet", "M5_Full_ExtraTrees"]:
        other = predictions[predictions["candidate"] == comparator]
        paired = core[["Hospital", "id", "Recurrence", "probability"]].merge(
            other[["Hospital", "id", "probability"]],
            on=["Hospital", "id"], suffixes=("_core", "_other"), validate="one_to_one"
        )
        core_metrics = simple_metrics(paired["Recurrence"], paired["probability_core"])
        other_metrics = simple_metrics(paired["Recurrence"], paired["probability_other"])
        label = f"Stable_Core_RCS minus {comparator}"
        for metric in core_metrics:
            observed = incremental[(incremental["comparison"] == label) & (incremental["metric"] == metric)].iloc[0]
            expected = [core_metrics[metric], other_metrics[metric], core_metrics[metric] - other_metrics[metric]]
            found = [observed["core_estimate"], observed["comparator_estimate"], observed["difference_core_minus_comparator"]]
            for index, (left, right) in enumerate(zip(found, expected)):
                maximum_difference = max(maximum_difference, abs(float(left) - float(right)))
                checks[f"incremental_{comparator}_{metric}_{index}"] = close(float(left), float(right))

    age = pd.read_csv(OUTPUT / "age_under18_sensitivity.csv")
    for cohort, source in data_by_cohort.items():
        for analysis, subset in [("all_ages", source), ("exclude_age_under_18", source[source["Age"] >= 18])]:
            observed = age[(age["cohort"] == cohort) & (age["analysis"] == analysis)].iloc[0]
            expected = metric_row(subset["Recurrence"], subset["probability"])
            checks[f"age_{cohort}_{analysis}_excluded_n"] = int(observed["excluded_n"]) == len(source) - len(subset)
            for metric, value in expected.items():
                difference = abs(float(observed[metric]) - float(value))
                maximum_difference = max(maximum_difference, difference)
                checks[f"age_{cohort}_{analysis}_{metric}"] = close(float(observed[metric]), float(value))

    operating = pd.read_csv(OUTPUT / "prespecified_threshold_operating_points.csv")
    for cohort, source in data_by_cohort.items():
        y_true = source["Recurrence"].to_numpy(int)
        probability = source["probability"].to_numpy(float)
        for threshold in THRESHOLDS:
            row = operating[(operating["cohort"] == cohort) & np.isclose(operating["threshold"], threshold)].iloc[0]
            predicted = probability >= threshold
            tp = int(np.sum(predicted & (y_true == 1)))
            fp = int(np.sum(predicted & (y_true == 0)))
            tn = int(np.sum(~predicted & (y_true == 0)))
            fn = int(np.sum(~predicted & (y_true == 1)))
            expected = {
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp),
                "npv": tn / (tn + fn),
                "model_net_benefit": tp / len(source) - fp / len(source) * threshold / (1 - threshold),
            }
            for metric, value in expected.items():
                difference = abs(float(row[metric]) - value)
                maximum_difference = max(maximum_difference, difference)
                checks[f"operating_{cohort}_{threshold}_{metric}"] = close(float(row[metric]), value)

    stability = pd.read_csv(DEV_OUTPUT / "stable_feature_selection.csv")
    outer = stability[stability["training_hospitals"].str.count("-") == 2]
    matrix = pd.read_csv(OUTPUT / "predictor_stability_heatmap_matrix.csv", index_col=0)
    checks["stability_matrix_dimensions"] = matrix.shape == (16, 4)
    for row in outer.itertuples():
        held_out = next(str(value) for value in range(1, 5) if str(value) not in row.training_hospitals.split("-"))
        checks[f"stability_{row.variable}_{held_out}"] = close(
            float(matrix.loc[row.variable, held_out]), float(row.selection_frequency)
        )

    curves = pd.read_csv(OUTPUT / "continuous_predictor_marginal_risk_curves.csv")
    development = pd.read_csv(FROZEN / "01_development_H1-H4.csv")
    for variable, part in curves.groupby("variable"):
        for row in part.iloc[[0, len(part) // 2, -1]].itertuples():
            modified = development[artifact["features"]].copy()
            modified[variable] = row.value
            raw = artifact["pipeline"].predict_proba(modified)[:, 1]
            expected = float(apply_platt(artifact["calibrator"], raw).mean())
            difference = abs(row.marginal_adjusted_risk - expected)
            maximum_difference = max(maximum_difference, difference)
            checks[f"curve_{variable}_{row.Index}"] = close(row.marginal_adjusted_risk, expected)

    calibration = pd.read_csv(OUTPUT / "calibration_transport_summary.csv")
    expected_calibration = pd.concat(
        [
            pd.read_csv(H5_OUTPUT / "h5_metrics.csv").assign(cohort="H5 primary"),
            pd.read_csv(H6_OUTPUT / "h6_metrics.csv").assign(cohort="H6 stress"),
        ], ignore_index=True
    )
    checks["calibration_transport_rows_exact"] = calibration.equals(expected_calibration)

    forest = pd.read_csv(OUTPUT / "centre_performance_forest_data.csv")
    for centre, source in [(f"H{h}", p) for h, p in data_by_cohort["H1-H4 IECV"].groupby("Hospital")] + [
        ("H5", data_by_cohort["H5 primary"]), ("H6", data_by_cohort["H6 stress"])
    ]:
        observed = forest[forest["centre"] == centre].iloc[0]
        expected = metric_row(source["Recurrence"], source["probability"])
        for metric, value in expected.items():
            difference = abs(float(observed[metric]) - float(value))
            maximum_difference = max(maximum_difference, difference)
            checks[f"forest_{centre}_{metric}"] = close(float(observed[metric]), float(value))

    result = {
        "version": VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "maximum_metric_difference": maximum_difference,
    }
    (OUTPUT / "stage14_independent_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
