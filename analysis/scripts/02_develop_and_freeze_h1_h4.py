#!/usr/bin/env python3
"""Nested IECV model development and final freeze using H1-H4 only."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from rldh_model_utils import (
    SEED,
    RestrictedCubicSpline,
    apply_platt,
    auc_i_squared,
    decision_curve,
    fit_platt,
    metric_row,
)


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
VERSION = "PELD_RLDH_V5_20260810"
DATA = ROOT / "data" / "frozen" / VERSION / "01_development_H1-H4.csv"
FIREWALL = ROOT / "data" / "frozen" / VERSION / "EXTERNAL_COHORT_FIREWALL.json"
OUTPUT = ROOT / "outputs" / VERSION / "model_development_h1_h4"
FREEZE = ROOT / "outputs" / VERSION / "model_freeze"

OUTCOME = "Recurrence"
GROUP = "Hospital"
IDENTIFIER = "id"
CONTINUOUS = [
    "Age",
    "BMI",
    "Disc_height_index",
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Lumbar_lordosis/degrees",
    "Sacral_slope/degrees",
]
CLINICAL = ["Age", "Gender", "BMI", "Smoking", "Alcoholism", "Hypertension", "Diabetes"]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_strategy: str
    nonlinear: bool
    complexity: int
    parameter_grid: tuple[dict[str, Any], ...]


def grid(**parameters: list[Any]) -> tuple[dict[str, Any], ...]:
    keys = list(parameters)
    return tuple(dict(zip(keys, values)) for values in itertools.product(*(parameters[key] for key in keys)))


CANDIDATES = [
    Candidate("M0_Clinical_Ridge", "ridge", "clinical", False, 1, grid(C=[0.1, 1.0, 10.0])),
    Candidate("M1_Stable_Core_Ridge", "ridge", "stable", False, 2, grid(C=[0.1, 1.0, 10.0])),
    Candidate("M2_Stable_Core_RCS_Ridge", "ridge", "stable", True, 3, grid(C=[0.05, 0.2, 1.0])),
    Candidate(
        "M3_Full_ElasticNet",
        "elasticnet",
        "full",
        False,
        4,
        grid(C=[0.05, 0.2, 1.0], l1_ratio=[0.25, 0.75]),
    ),
    Candidate(
        "M4_Full_RandomForest",
        "random_forest",
        "full",
        False,
        5,
        grid(max_depth=[3, 6], min_samples_leaf=[10, 30]),
    ),
    Candidate(
        "M5_Full_ExtraTrees",
        "extra_trees",
        "full",
        False,
        6,
        grid(max_depth=[4, 8], min_samples_leaf=[10, 30]),
    ),
    Candidate(
        "M6_Full_HistGradientBoosting",
        "hist_gradient_boosting",
        "full",
        False,
        7,
        grid(learning_rate=[0.03, 0.08], max_leaf_nodes=[7, 15], l2_regularization=[1.0]),
    ),
    Candidate(
        "M7_Full_XGBoost",
        "xgboost",
        "full",
        False,
        8,
        grid(max_depth=[2, 3], learning_rate=[0.03, 0.07], min_child_weight=[5, 15]),
    ),
]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictor_pool(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in {IDENTIFIER, GROUP, OUTCOME}]


def make_preprocessor(features: list[str], nonlinear: bool) -> ColumnTransformer:
    numeric = [feature for feature in features if feature in CONTINUOUS]
    categorical = [feature for feature in features if feature not in CONTINUOUS]
    if nonlinear and numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("rcs", RestrictedCubicSpline()),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def make_estimator(candidate: Candidate, parameters: dict[str, Any]):
    if candidate.family == "ridge":
        return LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=parameters["C"],
            max_iter=3000,
            random_state=SEED,
        )
    if candidate.family == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=parameters["C"],
            l1_ratio=parameters["l1_ratio"],
            max_iter=5000,
            random_state=SEED,
        )
    if candidate.family == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features="sqrt",
            random_state=SEED,
            n_jobs=1,
        )
    if candidate.family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features="sqrt",
            random_state=SEED,
            n_jobs=1,
        )
    if candidate.family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=parameters["learning_rate"],
            max_leaf_nodes=parameters["max_leaf_nodes"],
            l2_regularization=parameters["l2_regularization"],
            max_iter=250,
            random_state=SEED,
        )
    if candidate.family == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=parameters["max_depth"],
            learning_rate=parameters["learning_rate"],
            min_child_weight=parameters["min_child_weight"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(candidate.family)


def bootstrap_indices(data: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    indices = []
    for _, part in data.groupby([GROUP, OUTCOME], sort=True):
        part_indices = part.index.to_numpy()
        indices.extend(rng.choice(part_indices, size=len(part_indices), replace=True))
    return np.asarray(indices)


def coefficient_variable_map(preprocessor: ColumnTransformer, numeric: list[str], categorical: list[str]) -> list[str]:
    mapping = list(numeric)
    if categorical:
        encoder = preprocessor.named_transformers_["cat"].named_steps["one_hot"]
        for feature, categories in zip(categorical, encoder.categories_):
            mapping.extend([feature] * max(len(categories) - 1, 0))
    return mapping


def stable_selection(
    data: pd.DataFrame,
    all_features: list[str],
    repetitions: int,
    cache: dict[tuple[int, ...], dict[str, Any]],
) -> dict[str, Any]:
    key = tuple(sorted(int(value) for value in data[GROUP].unique()))
    if key in cache:
        return cache[key]
    numeric = [feature for feature in all_features if feature in CONTINUOUS]
    categorical = [feature for feature in all_features if feature not in CONTINUOUS]
    c_grid = [0.02, 0.05, 0.1, 0.2, 0.5]
    c_scores = []
    for c_value in c_grid:
        fold_losses = []
        for held_out in sorted(data[GROUP].unique()):
            training = data[data[GROUP] != held_out]
            validation = data[data[GROUP] == held_out]
            pipeline = Pipeline(
                [
                    ("preprocessor", make_preprocessor(all_features, False)),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            C=c_value,
                            max_iter=3000,
                            random_state=SEED,
                        ),
                    ),
                ]
            )
            pipeline.fit(training[all_features], training[OUTCOME])
            probabilities = pipeline.predict_proba(validation[all_features])[:, 1]
            fold_losses.append(log_loss(validation[OUTCOME], probabilities, labels=[0, 1]))
        c_scores.append((float(np.mean(fold_losses)), c_value))
    best_c = min(c_scores)[1]

    rng = np.random.default_rng(SEED + sum(key) * 101)
    selected_count = {feature: 0 for feature in all_features}
    absolute_sum = {feature: 0.0 for feature in all_features}
    for _ in range(repetitions):
        sampled = data.loc[bootstrap_indices(data, rng)]
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(all_features, False)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=best_c,
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        pipeline.fit(sampled[all_features], sampled[OUTCOME])
        coefficients = np.abs(pipeline.named_steps["model"].coef_[0])
        mapping = coefficient_variable_map(pipeline.named_steps["preprocessor"], numeric, categorical)
        for feature in all_features:
            feature_coefficients = coefficients[np.asarray(mapping) == feature]
            magnitude = float(feature_coefficients.max()) if len(feature_coefficients) else 0.0
            selected_count[feature] += int(magnitude > 1e-8)
            absolute_sum[feature] += magnitude

    stability = pd.DataFrame(
        {
            "variable": all_features,
            "selection_frequency": [selected_count[feature] / repetitions for feature in all_features],
            "mean_absolute_coefficient": [absolute_sum[feature] / repetitions for feature in all_features],
        }
    ).sort_values(["selection_frequency", "mean_absolute_coefficient"], ascending=False)
    selected = stability.loc[stability["selection_frequency"] >= 0.60, "variable"].tolist()[:8]
    if len(selected) < 3:
        selected = stability.head(5)["variable"].tolist()
    result = {"hospitals": list(key), "best_l1_c": best_c, "selected": selected, "stability": stability}
    cache[key] = result
    return result


def features_for(
    candidate: Candidate,
    training: pd.DataFrame,
    all_features: list[str],
    selection_cache: dict[tuple[int, ...], dict[str, Any]],
) -> list[str]:
    if candidate.feature_strategy == "clinical":
        return CLINICAL
    if candidate.feature_strategy == "full":
        return all_features
    return stable_selection(training, all_features, 30, selection_cache)["selected"]


def fit_pipeline(candidate: Candidate, parameters: dict[str, Any], training: pd.DataFrame, features: list[str]) -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(features, candidate.nonlinear)),
            ("model", make_estimator(candidate, parameters)),
        ]
    )
    pipeline.fit(training[features], training[OUTCOME])
    return pipeline


def tune_candidate(
    candidate: Candidate,
    training: pd.DataFrame,
    all_features: list[str],
    selection_cache: dict[tuple[int, ...], dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for parameter_index, parameters in enumerate(candidate.parameter_grid):
        fold_losses = []
        for held_out in sorted(training[GROUP].unique()):
            inner_training = training[training[GROUP] != held_out]
            inner_validation = training[training[GROUP] == held_out]
            features = features_for(candidate, inner_training, all_features, selection_cache)
            pipeline = fit_pipeline(candidate, parameters, inner_training, features)
            probabilities = pipeline.predict_proba(inner_validation[features])[:, 1]
            loss = float(log_loss(inner_validation[OUTCOME], probabilities, labels=[0, 1]))
            fold_losses.append(loss)
            rows.append(
                {
                    "candidate": candidate.name,
                    "parameter_index": parameter_index,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "held_out_hospital": int(held_out),
                    "features": json.dumps(features),
                    "log_loss": loss,
                }
            )
        rows.append(
            {
                "candidate": candidate.name,
                "parameter_index": parameter_index,
                "parameters": json.dumps(parameters, sort_keys=True),
                "held_out_hospital": "MEAN",
                "features": "nested",
                "log_loss": float(np.mean(fold_losses)),
            }
        )
    results = pd.DataFrame(rows)
    means = results[results["held_out_hospital"] == "MEAN"].sort_values(["log_loss", "parameter_index"])
    best_index = int(means.iloc[0]["parameter_index"])
    return dict(candidate.parameter_grid[best_index]), results


def training_oof_probabilities(
    candidate: Candidate,
    parameters: dict[str, Any],
    training: pd.DataFrame,
    all_features: list[str],
    selection_cache: dict[tuple[int, ...], dict[str, Any]],
) -> np.ndarray:
    probabilities = np.full(len(training), np.nan)
    positions = pd.Series(np.arange(len(training)), index=training.index)
    for held_out in sorted(training[GROUP].unique()):
        inner_training = training[training[GROUP] != held_out]
        inner_validation = training[training[GROUP] == held_out]
        features = features_for(candidate, inner_training, all_features, selection_cache)
        pipeline = fit_pipeline(candidate, parameters, inner_training, features)
        probabilities[positions.loc[inner_validation.index].to_numpy()] = pipeline.predict_proba(
            inner_validation[features]
        )[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("OOF calibration predictions incomplete")
    return probabilities


def candidate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    center_rows = []
    for candidate, candidate_data in predictions.groupby("candidate", sort=False):
        pooled = metric_row(candidate_data[OUTCOME].to_numpy(), candidate_data["probability"].to_numpy())
        pooled_rows.append({"candidate": candidate, **pooled})
        local_rows = []
        for hospital, center_data in candidate_data.groupby(GROUP):
            metrics = metric_row(center_data[OUTCOME].to_numpy(), center_data["probability"].to_numpy())
            row = {"candidate": candidate, "Hospital": int(hospital), **metrics}
            local_rows.append(row)
            center_rows.append(row)
        local_frame = pd.DataFrame(local_rows)
        pooled_rows[-1].update(
            {
                "minimum_center_auroc": float(local_frame["auroc"].min()),
                "minimum_center_slope": float(local_frame["calibration_slope"].min()),
                "maximum_center_slope": float(local_frame["calibration_slope"].max()),
                "auroc_i_squared": auc_i_squared(local_frame),
            }
        )
    return pd.DataFrame(pooled_rows), pd.DataFrame(center_rows)


def apply_gates(pooled: pd.DataFrame) -> pd.DataFrame:
    gate = pooled.copy()
    gate["gate_min_center_auroc"] = gate["minimum_center_auroc"] >= 0.65
    gate["gate_pooled_slope"] = gate["calibration_slope"].between(0.70, 1.30)
    gate["gate_all_center_slopes"] = (gate["minimum_center_slope"] >= 0.50) & (
        gate["maximum_center_slope"] <= 1.50
    )
    gate["gate_heterogeneity"] = gate["auroc_i_squared"] <= 50.0
    gate["eligible"] = gate[
        ["gate_min_center_auroc", "gate_pooled_slope", "gate_all_center_slopes", "gate_heterogeneity"]
    ].all(axis=1)
    return gate


def select_winner(gate: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    ranked = gate.merge(
        pd.DataFrame({"candidate": [candidate.name for candidate in CANDIDATES], "complexity": [candidate.complexity for candidate in CANDIDATES]}),
        on="candidate",
    )
    eligible = ranked[ranked["eligible"]].copy()
    if eligible.empty:
        return None, ranked
    eligible = eligible[eligible["minimum_center_auroc"] >= eligible["minimum_center_auroc"].max() - 0.01]
    eligible = eligible[eligible["auroc"] >= eligible["auroc"].max() - 0.01]
    best_brier = eligible["brier"].min()
    close_brier = eligible[eligible["brier"] < best_brier + 0.002].copy()
    close_brier = close_brier.sort_values(["complexity", "brier", "candidate"])
    winner = str(close_brier.iloc[0]["candidate"])
    ranked["selected"] = ranked["candidate"] == winner
    return winner, ranked


def extract_rcs_knots(pipeline: Pipeline, features: list[str]) -> dict[str, list[float]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    if "num" not in preprocessor.named_transformers_:
        return {}
    numeric_pipeline = preprocessor.named_transformers_["num"]
    if "rcs" not in numeric_pipeline.named_steps:
        return {}
    numeric_features = [feature for feature in features if feature in CONTINUOUS]
    spline = numeric_pipeline.named_steps["rcs"]
    return {feature: [float(value) for value in knots] for feature, knots in zip(numeric_features, spline.knots_)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FREEZE.mkdir(parents=True, exist_ok=True)
    firewall = json.loads(FIREWALL.read_text(encoding="utf-8"))
    if firewall["status"] != "SEALED":
        raise RuntimeError("External-cohort firewall is not sealed")
    data = pd.read_csv(DATA)
    if set(data[GROUP].unique()) != {1, 2, 3, 4}:
        raise RuntimeError("Development input contains non-development hospitals")
    all_features = predictor_pool(data)
    selection_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    prediction_rows = []
    tuning_frames = []
    outer_records = []

    for candidate in CANDIDATES:
        for held_out in sorted(data[GROUP].unique()):
            training = data[data[GROUP] != held_out].copy()
            validation = data[data[GROUP] == held_out].copy()
            best_parameters, tuning = tune_candidate(candidate, training, all_features, selection_cache)
            tuning["outer_held_out_hospital"] = int(held_out)
            tuning_frames.append(tuning)
            calibration_probabilities = training_oof_probabilities(
                candidate, best_parameters, training, all_features, selection_cache
            )
            calibrator = fit_platt(training[OUTCOME].to_numpy(), calibration_probabilities)
            final_features = features_for(candidate, training, all_features, selection_cache)
            final_pipeline = fit_pipeline(candidate, best_parameters, training, final_features)
            raw_probabilities = final_pipeline.predict_proba(validation[final_features])[:, 1]
            probabilities = apply_platt(calibrator, raw_probabilities)
            for row_index, (_, row) in enumerate(validation.iterrows()):
                prediction_rows.append(
                    {
                        "candidate": candidate.name,
                        "Hospital": int(row[GROUP]),
                        "id": row[IDENTIFIER],
                        "Recurrence": int(row[OUTCOME]),
                        "raw_probability": float(raw_probabilities[row_index]),
                        "probability": float(probabilities[row_index]),
                    }
                )
            outer_records.append(
                {
                    "candidate": candidate.name,
                    "held_out_hospital": int(held_out),
                    "best_parameters": best_parameters,
                    "final_features": final_features,
                    "calibration_intercept": float(calibrator.intercept_[0]),
                    "calibration_slope": float(calibrator.coef_[0, 0]),
                }
            )
            print(f"completed {candidate.name}, held-out H{held_out}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(OUTPUT / "iecv_predictions.csv", index=False)
    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    tuning_results.to_csv(OUTPUT / "nested_tuning_results.csv", index=False)
    write_json(OUTPUT / "outer_fold_configurations.json", outer_records)

    stability_frames = []
    for key, result in sorted(selection_cache.items()):
        frame = result["stability"].copy()
        frame.insert(0, "training_hospitals", "-".join(map(str, key)))
        frame["selected"] = frame["variable"].isin(result["selected"])
        frame["best_l1_c"] = result["best_l1_c"]
        stability_frames.append(frame)
    pd.concat(stability_frames, ignore_index=True).to_csv(OUTPUT / "stable_feature_selection.csv", index=False)

    pooled, center = candidate_metrics(predictions)
    pooled.to_csv(OUTPUT / "candidate_pooled_metrics.csv", index=False)
    center.to_csv(OUTPUT / "candidate_center_metrics.csv", index=False)
    gate = apply_gates(pooled)
    winner_name, ranked = select_winner(gate)
    ranked.to_csv(OUTPUT / "candidate_transportability_gate.csv", index=False)

    dca_frames = []
    for candidate, candidate_data in predictions.groupby("candidate"):
        dca = decision_curve(candidate_data[OUTCOME].to_numpy(), candidate_data["probability"].to_numpy(), THRESHOLDS)
        dca.insert(0, "candidate", candidate)
        dca_frames.append(dca)
    pd.concat(dca_frames, ignore_index=True).to_csv(OUTPUT / "iecv_decision_curve.csv", index=False)

    if winner_name is None:
        status = {
            "version": VERSION,
            "status": "STOP_NO_CANDIDATE_PASSED_TRANSPORTABILITY_GATES",
            "external_data_opened": False,
            "development_sha256": sha256(DATA),
        }
        write_json(OUTPUT / "development_status.json", status)
        print(json.dumps(status, indent=2))
        return

    winner = next(candidate for candidate in CANDIDATES if candidate.name == winner_name)
    best_parameters, final_tuning = tune_candidate(winner, data, all_features, selection_cache)
    final_tuning.to_csv(FREEZE / "final_model_tuning.csv", index=False)
    calibration_probabilities = training_oof_probabilities(winner, best_parameters, data, all_features, selection_cache)
    calibrator = fit_platt(data[OUTCOME].to_numpy(), calibration_probabilities)
    final_features = features_for(winner, data, all_features, selection_cache)
    final_pipeline = fit_pipeline(winner, best_parameters, data, final_features)
    rcs_knots = extract_rcs_knots(final_pipeline, final_features)

    artifact = {
        "version": VERSION,
        "candidate": winner.name,
        "features": final_features,
        "parameters": best_parameters,
        "pipeline": final_pipeline,
        "calibrator": calibrator,
        "thresholds": THRESHOLDS,
        "rcs_knots": rcs_knots,
    }
    model_path = FREEZE / "frozen_model.joblib"
    joblib.dump(artifact, model_path)
    freeze_manifest = {
        "version": VERSION,
        "status": "FINAL_MODEL_FROZEN_H5_MAY_OPEN",
        "selected_candidate": winner.name,
        "features": final_features,
        "parameters": best_parameters,
        "rcs_knots": rcs_knots,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "development_n": int(len(data)),
        "development_events": int(data[OUTCOME].sum()),
        "development_sha256": sha256(DATA),
        "model_sha256": sha256(model_path),
        "external_data_opened": False,
        "selection_rule": "transportability_then_accuracy_then_complexity",
    }
    write_json(FREEZE / "freeze_manifest.json", freeze_manifest)
    write_json(
        OUTPUT / "development_status.json",
        {
            **freeze_manifest,
            "eligible_candidates": ranked.loc[ranked["eligible"], "candidate"].tolist(),
            "candidate_count": len(CANDIDATES),
        },
    )
    print(json.dumps(freeze_manifest, indent=2))


if __name__ == "__main__":
    main()
