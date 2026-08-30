#!/usr/bin/env python3
"""Independent integrity checks for Stage 15 manuscript validation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from predict_from_published_equation import predict_from_equation
from rldh_model_utils import apply_platt


ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
BASE = ROOT / "outputs" / VERSION
FROZEN = ROOT / "data" / "frozen" / VERSION
DEV_OUTPUT = BASE / "model_development_h1_h4"
H5_OUTPUT = BASE / "external_validation" / "h5_primary"
H6_OUTPUT = BASE / "external_validation" / "h6_stress_test"
MODEL = BASE / "model_freeze" / "frozen_model.joblib"
OUTPUT = BASE / "stage15_manuscript_validation"
CONTINUOUS = [
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Sacral_slope/degrees",
    "Age",
    "Disc_height_index",
]
CATEGORICAL = ["Modic_group", "Pfirrmann_group", "Herniation_type"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohorts() -> dict[str, pd.DataFrame]:
    development = pd.read_csv(FROZEN / "01_development_H1-H4.csv")
    predictions = pd.read_csv(DEV_OUTPUT / "iecv_predictions.csv")
    selected = predictions.loc[
        predictions["candidate"].eq("M2_Stable_Core_RCS_Ridge"),
        ["Hospital", "id", "probability"],
    ]
    development = development.merge(selected, on=["Hospital", "id"], validate="one_to_one")
    h5 = pd.read_csv(FROZEN / "02_external_H5_SEALED.csv").merge(
        pd.read_csv(H5_OUTPUT / "h5_predictions.csv")[["Hospital", "id", "original_probability"]],
        on=["Hospital", "id"],
        validate="one_to_one",
    ).rename(columns={"original_probability": "probability"})
    h6 = pd.read_csv(FROZEN / "03_external_H6_SEALED.csv").merge(
        pd.read_csv(H6_OUTPUT / "h6_predictions.csv")[["Hospital", "id", "original_probability"]],
        on=["Hospital", "id"],
        validate="one_to_one",
    ).rename(columns={"original_probability": "probability"})
    return {"H1-H4 IECV": development, "H5 primary": h5, "H6 stress": h6}


def calibration_model() -> Pipeline:
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("model", LogisticRegression(C=np.inf, solver="lbfgs", max_iter=3000)),
        ]
    )


def continuous_smd(development: pd.Series, external: pd.Series) -> float:
    pooled = np.sqrt((development.var(ddof=1) + external.var(ddof=1)) / 2)
    return float((external.mean() - development.mean()) / pooled) if pooled > 0 else 0.0


def continuous_psi(development: pd.Series, external: pd.Series) -> float:
    internal = np.unique(development.quantile(np.linspace(0, 1, 11)).to_numpy(float))
    edges = np.concatenate(([-np.inf], internal[1:-1], [np.inf]))
    dev = pd.cut(development, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(float)
    ext = pd.cut(external, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(float)
    dev = np.clip(dev / dev.sum(), 1e-6, None)
    ext = np.clip(ext / ext.sum(), 1e-6, None)
    return float(np.sum((ext - dev) * np.log(ext / dev)))


def categorical_psi(development: pd.Series, external: pd.Series) -> float:
    levels = sorted(set(development.astype(str)) | set(external.astype(str)))
    dev = development.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    ext = external.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    dev = np.clip(dev, 1e-6, None)
    ext = np.clip(ext, 1e-6, None)
    return float(np.sum((ext - dev) * np.log(ext / dev)))


def check(condition: bool, label: str, checks: list[dict], detail: object = None) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL", "detail": detail})


def main() -> None:
    checks: list[dict] = []
    data = cohorts()

    manifest = json.loads((OUTPUT / "stage15_manifest.json").read_text(encoding="utf-8"))
    mismatches = [
        name for name, expected in manifest["files"].items()
        if not (ROOT / name).exists() or sha256(ROOT / name) != expected
    ]
    check(not mismatches, "manifest hashes", checks, mismatches)
    check(sha256(MODEL) == manifest["frozen_model_sha256"], "frozen model hash", checks)

    metadata = json.loads((OUTPUT / "stage15_run_metadata.json").read_text(encoding="utf-8"))
    check(metadata["model_refit"] is False, "no model refit", checks)
    check(metadata["feature_reselection"] is False, "no feature reselection", checks)
    check(metadata["external_cutoff_optimization"] is False, "no external cutoff optimization", checks)
    check(manifest["statistical_expansion_terminal"] is True, "terminal expansion flag", checks)

    calibration = pd.read_csv(OUTPUT / "flexible_calibration_curves.csv")
    check(len(calibration) == 360, "calibration grid rows", checks, len(calibration))
    check((calibration["successful_bootstraps"] == 500).all(), "500 successful calibration bootstraps", checks)
    check(
        ((calibration["lower_95"] <= calibration["observed_smoothed"])
         & (calibration["observed_smoothed"] <= calibration["upper_95"])).all(),
        "calibration estimates inside bands",
        checks,
    )
    calibration_max_diff = 0.0
    for name, frame in data.items():
        stored = calibration.loc[calibration["cohort"].eq(name)].sort_values("predicted_probability")
        model = calibration_model().fit(
            logit(np.clip(frame["probability"].to_numpy(float), 1e-6, 1 - 1e-6)).reshape(-1, 1),
            frame["Recurrence"].to_numpy(int),
        )
        rebuilt = model.predict_proba(logit(stored["predicted_probability"].to_numpy()).reshape(-1, 1))[:, 1]
        calibration_max_diff = max(
            calibration_max_diff,
            float(np.max(np.abs(rebuilt - stored["observed_smoothed"].to_numpy()))),
        )
    check(calibration_max_diff < 1e-12, "independent calibration point curves", checks, calibration_max_diff)

    cutpoint_record = json.loads((OUTPUT / "development_frozen_risk_cutpoints.json").read_text())
    rebuilt_cutpoints = np.quantile(data["H1-H4 IECV"]["probability"], [0.25, 0.5, 0.75])
    cutpoint_diff = float(np.max(np.abs(rebuilt_cutpoints - np.asarray(cutpoint_record["cutpoints"]))))
    check(cutpoint_diff < 1e-15, "development-only frozen quartiles", checks, cutpoint_diff)
    stored_strata = pd.read_csv(OUTPUT / "development_frozen_risk_strata.csv")
    bins = [-np.inf, *rebuilt_cutpoints.tolist(), np.inf]
    labels = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    strata_max_diff = 0.0
    count_match = True
    for cohort, frame in data.items():
        assigned = pd.cut(frame["probability"], bins=bins, labels=labels, include_lowest=True)
        for group in labels:
            part = frame.loc[assigned.eq(group)]
            row = stored_strata.loc[
                stored_strata["cohort"].eq(cohort) & stored_strata["risk_group"].eq(group)
            ].iloc[0]
            count_match &= int(row["n"]) == len(part) and int(row["events"]) == int(part["Recurrence"].sum())
            for column, value in {
                "observed_rate": part["Recurrence"].mean(),
                "mean_predicted_risk": part["probability"].mean(),
                "median_predicted_risk": part["probability"].median(),
            }.items():
                strata_max_diff = max(strata_max_diff, abs(float(row[column]) - float(value)))
    check(count_match, "risk-stratum counts and events", checks)
    check(strata_max_diff < 1e-14, "risk-stratum summary values", checks, strata_max_diff)

    stored_shift = pd.read_csv(OUTPUT / "dataset_shift_smd_psi.csv")
    shift_max_diff = 0.0
    development = data["H1-H4 IECV"]
    for external_name in ["H5 primary", "H6 stress"]:
        external = data[external_name]
        comparison = f"H1-H4 vs {external_name}"
        for feature in CONTINUOUS:
            row = stored_shift.loc[
                stored_shift["comparison"].eq(comparison) & stored_shift["predictor"].eq(feature)
            ].iloc[0]
            shift_max_diff = max(
                shift_max_diff,
                abs(float(row["signed_smd_external_minus_development"]) - continuous_smd(development[feature], external[feature])),
                abs(float(row["psi"]) - continuous_psi(development[feature], external[feature])),
            )
        for feature in CATEGORICAL:
            row = stored_shift.loc[
                stored_shift["comparison"].eq(comparison) & stored_shift["predictor"].eq(feature)
            ].iloc[0]
            shift_max_diff = max(
                shift_max_diff,
                abs(float(row["psi"]) - categorical_psi(development[feature], external[feature])),
            )
    check(shift_max_diff < 1e-12, "independent SMD and PSI reconstruction", checks, shift_max_diff)

    oof = pd.read_csv(OUTPUT / "domain_classifier_oof_predictions.csv")
    domain_summary = pd.read_csv(OUTPUT / "domain_classifier_summary.csv")
    domain_auc_diff = 0.0
    for comparison, part in oof.groupby("comparison"):
        rebuilt_auc = roc_auc_score(part["domain"], part["oof_domain_probability"])
        stored_auc = domain_summary.loc[domain_summary["comparison"].eq(comparison), "oof_domain_auc"].iloc[0]
        domain_auc_diff = max(domain_auc_diff, abs(float(rebuilt_auc) - float(stored_auc)))
    check(domain_auc_diff < 1e-14, "domain-classifier AUROC from saved OOF predictions", checks, domain_auc_diff)

    specification = json.loads((OUTPUT / "published_equation_spec.json").read_text(encoding="utf-8"))
    master = pd.concat([frame.assign(validation_cohort=name) for name, frame in data.items()], ignore_index=True)
    equation_probability = predict_from_equation(master, specification)
    artifact = joblib.load(MODEL)
    raw_probability = artifact["pipeline"].predict_proba(master[artifact["features"]])[:, 1]
    pipeline_probability = apply_platt(artifact["calibrator"], raw_probability)
    equation_max_diff = float(np.max(np.abs(equation_probability - pipeline_probability)))
    check(equation_max_diff < 1e-10, "standalone published equation reproduces pipeline", checks, equation_max_diff)
    full = pd.read_csv(OUTPUT / "published_equation_full_reproducibility.csv")
    check(len(full) == len(master) == 5088, "all-subject equation test size", checks, len(full))
    check(len(pd.read_csv(OUTPUT / "published_equation_random100_reproducibility.csv")) == 100, "random-100 test size", checks)
    status = json.loads((OUTPUT / "published_equation_reproducibility_status.json").read_text())
    check(status["status"] == "PASS" and status["maximum_absolute_difference_all"] < 1e-10, "published-equation status", checks, status)

    failed = [item for item in checks if item["status"] != "PASS"]
    result = {
        "version": VERSION,
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "checks": checks,
    }
    (OUTPUT / "stage15_independent_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
