#!/usr/bin/env python3
"""Four manuscript-grade validation analyses for the frozen RLDH v5 model.

No model development, feature reselection, cutoff optimization, or external-cohort
feedback is performed. Outputs cover flexible calibration uncertainty, prediction
distributions and development-frozen risk strata, eight-predictor dataset shift,
and standalone published-equation reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib.pyplot as plt
import seaborn as sns

from predict_from_published_equation import predict_from_equation
from rldh_model_utils import apply_platt


VERSION = "PELD_RLDH_V5_20260810"
BASE = ROOT / "outputs" / VERSION
FROZEN = ROOT / "data" / "frozen" / VERSION
DEV_OUTPUT = BASE / "model_development_h1_h4"
H5_OUTPUT = BASE / "external_validation" / "h5_primary"
H6_OUTPUT = BASE / "external_validation" / "h6_stress_test"
MODEL = BASE / "model_freeze" / "frozen_model.joblib"
OUTPUT = BASE / "stage15_manuscript_validation"
FIGURES = OUTPUT / "figures"
SEED = 20260810
CONTINUOUS = [
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Sacral_slope/degrees",
    "Age",
    "Disc_height_index",
]
CATEGORICAL = ["Modic_group", "Pfirrmann_group", "Herniation_type"]
PREDICTORS = [
    "Modic_group",
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Pfirrmann_group",
    "Sacral_slope/degrees",
    "Age",
    "Disc_height_index",
    "Herniation_type",
]


warnings.filterwarnings("ignore", message="Setting penalty=None will ignore.*")
warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cohorts() -> dict[str, pd.DataFrame]:
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


def calibration_pipeline() -> Pipeline:
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


def calibration_curves(cohorts: dict[str, pd.DataFrame]) -> None:
    rows = []
    rng = np.random.default_rng(SEED)
    for cohort_index, (cohort, data) in enumerate(cohorts.items()):
        probability = data["probability"].to_numpy(float)
        y_true = data["Recurrence"].to_numpy(int)
        lower = max(0.001, float(np.quantile(probability, 0.01)))
        upper = min(0.60, float(np.quantile(probability, 0.99)))
        grid_probability = np.linspace(lower, upper, 120)
        grid_logit = logit(grid_probability).reshape(-1, 1)
        full = calibration_pipeline().fit(logit(np.clip(probability, 1e-6, 1 - 1e-6)).reshape(-1, 1), y_true)
        estimate = full.predict_proba(grid_logit)[:, 1]
        if cohort == "H1-H4 IECV":
            strata = [part.index.to_numpy() for _, part in data.groupby(["Hospital", "Recurrence"])]
        else:
            strata = [part.index.to_numpy() for _, part in data.groupby("Recurrence")]
        draws = []
        for _ in range(500):
            indices = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
            sampled = data.loc[indices]
            sampled_probability = sampled["probability"].to_numpy(float)
            sampled_y = sampled["Recurrence"].to_numpy(int)
            try:
                fitted = calibration_pipeline().fit(
                    logit(np.clip(sampled_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1),
                    sampled_y,
                )
                draws.append(fitted.predict_proba(grid_logit)[:, 1])
            except Exception:
                continue
        draw_array = np.asarray(draws)
        for index, predicted in enumerate(grid_probability):
            rows.append(
                {
                    "cohort": cohort,
                    "predicted_probability": predicted,
                    "observed_smoothed": estimate[index],
                    "lower_95": float(np.quantile(draw_array[:, index], 0.025)),
                    "upper_95": float(np.quantile(draw_array[:, index], 0.975)),
                    "successful_bootstraps": len(draws),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "flexible_calibration_curves.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (cohort, part) in zip(axes, frame.groupby("cohort", sort=False)):
        source = cohorts[cohort]
        maximum = max(0.25, float(part["predicted_probability"].max()) * 1.08, float(part["upper_95"].max()) * 1.05)
        ax.fill_between(part["predicted_probability"], part["lower_95"], part["upper_95"], color="#457B9D", alpha=0.22)
        ax.plot(part["predicted_probability"], part["observed_smoothed"], color="#1D3557", linewidth=2.4)
        ax.plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1)
        rug_y = np.full(len(source), maximum * 0.012)
        ax.plot(source["probability"], rug_y, "|", color="#E76F51", alpha=0.08, markersize=7)
        ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="Predicted recurrence probability", ylabel="Observed recurrence probability", title=cohort)
    fig.suptitle("Flexible calibration curves with stratified bootstrap 95% bands")
    fig.tight_layout()
    fig.savefig(FIGURES / "flexible_calibration_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = events / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def risk_distribution_and_strata(cohorts: dict[str, pd.DataFrame]) -> None:
    development_probability = cohorts["H1-H4 IECV"]["probability"].to_numpy(float)
    cutpoints = np.quantile(development_probability, [0.25, 0.50, 0.75])
    (OUTPUT / "development_frozen_risk_cutpoints.json").write_text(
        json.dumps(
            {
                "source": "H1-H4 selected-model out-of-hospital predictions",
                "quantiles": [0.25, 0.50, 0.75],
                "cutpoints": [float(value) for value in cutpoints],
                "external_reoptimization": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []
    distribution_rows = []
    bins = [-np.inf, *cutpoints.tolist(), np.inf]
    labels = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    for cohort, data in cohorts.items():
        assigned = pd.cut(data["probability"], bins=bins, labels=labels, include_lowest=True)
        working = data.assign(risk_group=assigned)
        distribution_rows.append(
            working[["Hospital", "id", "Recurrence", "probability", "risk_group"]].assign(cohort=cohort)
        )
        for group, part in working.groupby("risk_group", observed=True):
            events = int(part["Recurrence"].sum())
            lower, upper = wilson_interval(events, len(part))
            rows.append(
                {
                    "cohort": cohort,
                    "risk_group": str(group),
                    "n": int(len(part)),
                    "events": events,
                    "observed_rate": events / len(part),
                    "observed_lower_95": lower,
                    "observed_upper_95": upper,
                    "mean_predicted_risk": float(part["probability"].mean()),
                    "median_predicted_risk": float(part["probability"].median()),
                }
            )
    pd.concat(distribution_rows, ignore_index=True).to_csv(OUTPUT / "prediction_distribution_data.csv", index=False)
    strata = pd.DataFrame(rows)
    strata.to_csv(OUTPUT / "development_frozen_risk_strata.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (cohort, data) in zip(axes, cohorts.items()):
        plotting = data.assign(outcome=data["Recurrence"].map({0: "Non-recurrence", 1: "Recurrence"}))
        sns.histplot(data=plotting, x="probability", hue="outcome", stat="density", common_norm=False, element="step", fill=False, bins=30, ax=ax)
        ax.set(xlabel="Frozen-model predicted risk", ylabel="Density", title=cohort)
    fig.suptitle("Prediction distributions by observed recurrence")
    fig.tight_layout()
    fig.savefig(FIGURES / "prediction_distributions_by_outcome.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    order = labels
    for ax, (cohort, part) in zip(axes, strata.groupby("cohort", sort=False)):
        part = part.set_index("risk_group").loc[order].reset_index()
        positions = np.arange(4)
        ax.errorbar(
            positions,
            part["observed_rate"],
            yerr=[part["observed_rate"] - part["observed_lower_95"], part["observed_upper_95"] - part["observed_rate"]],
            fmt="o",
            color="#1D3557",
            capsize=3,
            label="Observed",
        )
        ax.plot(positions, part["mean_predicted_risk"], marker="s", color="#E76F51", label="Mean predicted")
        ax.set(xticks=positions, xticklabels=["Q1", "Q2", "Q3", "Q4"], xlabel="Development-frozen risk group", ylabel="Recurrence probability", title=cohort)
    axes[-1].legend(frameon=False)
    fig.suptitle("Observed recurrence across development-defined risk strata")
    fig.tight_layout()
    fig.savefig(FIGURES / "development_frozen_risk_strata.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def continuous_smd(development: pd.Series, external: pd.Series) -> float:
    pooled = np.sqrt((development.var(ddof=1) + external.var(ddof=1)) / 2)
    return float((external.mean() - development.mean()) / pooled) if pooled > 0 else 0.0


def continuous_psi(development: pd.Series, external: pd.Series) -> float:
    internal = np.unique(development.quantile(np.linspace(0, 1, 11)).to_numpy(float))
    edges = np.concatenate(([-np.inf], internal[1:-1], [np.inf]))
    dev_counts = pd.cut(development, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(float)
    ext_counts = pd.cut(external, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(float)
    dev_prop = np.clip(dev_counts / dev_counts.sum(), 1e-6, None)
    ext_prop = np.clip(ext_counts / ext_counts.sum(), 1e-6, None)
    return float(np.sum((ext_prop - dev_prop) * np.log(ext_prop / dev_prop)))


def categorical_psi(development: pd.Series, external: pd.Series) -> float:
    levels = sorted(set(development.astype(str)) | set(external.astype(str)))
    dev_prop = development.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    ext_prop = external.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    dev_prop = np.clip(dev_prop, 1e-6, None)
    ext_prop = np.clip(ext_prop, 1e-6, None)
    return float(np.sum((ext_prop - dev_prop) * np.log(ext_prop / dev_prop)))


def domain_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("continuous", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), CONTINUOUS),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CATEGORICAL),
        ],
        remainder="drop",
    )


def dataset_shift(cohorts: dict[str, pd.DataFrame]) -> None:
    development = cohorts["H1-H4 IECV"]
    shift_rows = []
    categorical_level_rows = []
    for external_name in ["H5 primary", "H6 stress"]:
        external = cohorts[external_name]
        for feature in CONTINUOUS:
            shift_rows.append(
                {
                    "comparison": f"H1-H4 vs {external_name}",
                    "predictor": feature,
                    "type": "continuous",
                    "signed_smd_external_minus_development": continuous_smd(development[feature], external[feature]),
                    "maximum_absolute_level_smd": np.nan,
                    "psi": continuous_psi(development[feature], external[feature]),
                }
            )
        for feature in CATEGORICAL:
            levels = sorted(set(development[feature].astype(str)) | set(external[feature].astype(str)))
            level_smds = []
            for level in levels:
                dev_p = float((development[feature].astype(str) == level).mean())
                ext_p = float((external[feature].astype(str) == level).mean())
                pooled = (dev_p + ext_p) / 2
                denominator = np.sqrt(max(pooled * (1 - pooled), 1e-12))
                smd = (ext_p - dev_p) / denominator
                level_smds.append(abs(smd))
                categorical_level_rows.append(
                    {
                        "comparison": f"H1-H4 vs {external_name}",
                        "predictor": feature,
                        "level": level,
                        "development_proportion": dev_p,
                        "external_proportion": ext_p,
                        "signed_level_smd": smd,
                    }
                )
            shift_rows.append(
                {
                    "comparison": f"H1-H4 vs {external_name}",
                    "predictor": feature,
                    "type": "categorical",
                    "signed_smd_external_minus_development": np.nan,
                    "maximum_absolute_level_smd": max(level_smds),
                    "psi": categorical_psi(development[feature], external[feature]),
                }
            )
    pd.DataFrame(shift_rows).to_csv(OUTPUT / "dataset_shift_smd_psi.csv", index=False)
    pd.DataFrame(categorical_level_rows).to_csv(OUTPUT / "dataset_shift_categorical_levels.csv", index=False)

    domain_rows = []
    domain_prediction_rows = []
    rng = np.random.default_rng(SEED)
    for comparison_index, external_name in enumerate(["H5 primary", "H6 stress"]):
        external = cohorts[external_name]
        combined = pd.concat(
            [development[PREDICTORS].assign(domain=0), external[PREDICTORS].assign(domain=1)],
            ignore_index=True,
        )
        pipeline = Pipeline(
            [
                ("preprocessor", domain_preprocessor()),
                ("model", LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", random_state=SEED)),
            ]
        )
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        probability = cross_val_predict(
            pipeline,
            combined[PREDICTORS],
            combined["domain"],
            cv=folds,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        auc = float(roc_auc_score(combined["domain"], probability))
        strata = [index.to_numpy() for _, index in combined.groupby("domain").groups.items()]
        draws = []
        for _ in range(2000):
            sampled_index = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
            draws.append(float(roc_auc_score(combined.loc[sampled_index, "domain"], probability[sampled_index])))
        comparison = f"H1-H4 vs {external_name}"
        domain_rows.append(
            {
                "comparison": comparison,
                "n_development": int((combined["domain"] == 0).sum()),
                "n_external": int((combined["domain"] == 1).sum()),
                "oof_domain_auc": auc,
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
                "cv": "stratified 5-fold",
                "interpretation": "descriptive case-mix separability, not model validation",
            }
        )
        domain_prediction_rows.append(
            pd.DataFrame(
                {
                    "comparison": comparison,
                    "row_index": np.arange(len(combined)),
                    "domain": combined["domain"],
                    "oof_domain_probability": probability,
                }
            )
        )
    pd.DataFrame(domain_rows).to_csv(OUTPUT / "domain_classifier_summary.csv", index=False)
    pd.concat(domain_prediction_rows, ignore_index=True).to_csv(OUTPUT / "domain_classifier_oof_predictions.csv", index=False)

    long_continuous = []
    for cohort, data in cohorts.items():
        long_continuous.append(data[CONTINUOUS].assign(cohort=cohort).melt(id_vars="cohort", var_name="predictor", value_name="value"))
    continuous_frame = pd.concat(long_continuous, ignore_index=True)
    grid = sns.FacetGrid(continuous_frame, col="predictor", col_wrap=3, hue="cohort", sharex=False, sharey=False, height=3.6)
    grid.map(sns.kdeplot, "value", common_norm=False)
    grid.add_legend()
    grid.set_axis_labels("Observed value", "Density")
    grid.fig.suptitle("Case-mix distributions of continuous frozen-model predictors", y=1.03)
    grid.savefig(FIGURES / "dataset_shift_continuous_distributions.png", dpi=240, bbox_inches="tight")
    plt.close(grid.fig)

    categorical_plot_rows = []
    for cohort, data in cohorts.items():
        for feature in CATEGORICAL:
            proportions = data[feature].astype(str).value_counts(normalize=True)
            for level, proportion in proportions.items():
                display_level = "None" if feature == "Modic_group" and level == "无" else level
                categorical_plot_rows.append(
                    {
                        "cohort": cohort,
                        "predictor": feature,
                        "level": display_level,
                        "proportion": proportion,
                    }
                )
    categorical_frame = pd.DataFrame(categorical_plot_rows)
    grid = sns.catplot(data=categorical_frame, x="level", y="proportion", hue="cohort", col="predictor", kind="bar", sharex=False, height=4.3, aspect=1.1)
    grid.set_axis_labels("Observed category", "Proportion")
    grid.set_xticklabels(rotation=30)
    grid.fig.suptitle("Case-mix distributions of categorical frozen-model predictors", y=1.04)
    grid.savefig(FIGURES / "dataset_shift_categorical_proportions.png", dpi=240, bbox_inches="tight")
    plt.close(grid.fig)


def export_equation_and_verify(artifact: dict, cohorts: dict[str, pd.DataFrame]) -> None:
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    ridge = artifact["pipeline"].named_steps["model"]
    numeric_pipeline = preprocessor.named_transformers_["num"]
    categorical_pipeline = preprocessor.named_transformers_["cat"]
    numeric_features = [feature for feature in artifact["features"] if feature in CONTINUOUS]
    categorical_features = [feature for feature in artifact["features"] if feature in CATEGORICAL]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    rcs = numeric_pipeline.named_steps["rcs"]
    scaler = numeric_pipeline.named_steps["scaler"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["one_hot"]
    specification = {
        "version": VERSION,
        "raw_predictors": artifact["features"],
        "numeric": {
            "features": numeric_features,
            "imputation_medians": [float(value) for value in numeric_imputer.statistics_],
            "knots": {
                feature: [float(value) for value in knots]
                for feature, knots in zip(numeric_features, rcs.knots_)
            },
            "rcs_definition": "Harrell restricted cubic spline basis exactly as specified in predict_from_published_equation.py",
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
        },
        "categorical": {
            "features": categorical_features,
            "imputation_modes": [str(value) for value in categorical_imputer.statistics_],
            "levels": {
                feature: [str(value) for value in levels]
                for feature, levels in zip(categorical_features, encoder.categories_)
            },
            "drop_index": {
                feature: int(drop_index)
                for feature, drop_index in zip(categorical_features, encoder.drop_idx_)
            },
        },
        "ridge_model": {
            "intercept": float(ridge.intercept_[0]),
            "coefficients": [float(value) for value in ridge.coef_[0]],
        },
        "platt_calibration": {
            "intercept": float(artifact["calibrator"].intercept_[0]),
            "slope": float(artifact["calibrator"].coef_[0, 0]),
            "probability_clip": [1e-6, 1 - 1e-6],
        },
        "model_sha256": sha256(MODEL),
    }
    spec_path = OUTPUT / "published_equation_spec.json"
    spec_path.write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")

    master = pd.concat(
        [data.assign(validation_cohort=cohort) for cohort, data in cohorts.items()],
        ignore_index=True,
    )
    pipeline_raw = artifact["pipeline"].predict_proba(master[artifact["features"]])[:, 1]
    pipeline_probability = apply_platt(artifact["calibrator"], pipeline_raw)
    equation_probability = predict_from_equation(master, specification)
    difference = np.abs(pipeline_probability - equation_probability)
    full = master[["validation_cohort", "Hospital", "id"]].copy()
    full["pipeline_probability"] = pipeline_probability
    full["equation_probability"] = equation_probability
    full["absolute_difference"] = difference
    full.to_csv(OUTPUT / "published_equation_full_reproducibility.csv", index=False)
    sample = full.sample(n=100, random_state=SEED).sort_values(["validation_cohort", "Hospital", "id"])
    sample.to_csv(OUTPUT / "published_equation_random100_reproducibility.csv", index=False)
    status = {
        "version": VERSION,
        "tested_all_n": int(len(full)),
        "tested_random_sample_n": 100,
        "maximum_absolute_difference_all": float(difference.max()),
        "maximum_absolute_difference_random100": float(sample["absolute_difference"].max()),
        "required_tolerance": 1e-10,
        "status": "PASS" if float(difference.max()) < 1e-10 else "FAIL",
        "joblib_not_loaded_by_published_equation_script": True,
    }
    (OUTPUT / "published_equation_reproducibility_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    if status["status"] != "PASS":
        raise RuntimeError("Published equation failed to reproduce frozen pipeline")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cohorts = load_cohorts()
    artifact = joblib.load(MODEL)
    calibration_curves(cohorts)
    risk_distribution_and_strata(cohorts)
    dataset_shift(cohorts)
    export_equation_and_verify(artifact, cohorts)
    metadata = {
        "version": VERSION,
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "random_seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "bootstrap_repetitions": {"calibration_curve": 500, "domain_auc": 2000},
        "model_refit": False,
        "feature_reselection": False,
        "external_cutoff_optimization": False,
    }
    (OUTPUT / "stage15_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    excluded = {"stage15_manifest.json", "stage15_independent_verification.json"}
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name not in excluded)
    manifest = {
        "version": VERSION,
        "status": "MANUSCRIPT_VALIDATION_ANALYSES_COMPLETE_PENDING_VERIFICATION",
        "frozen_model_sha256": sha256(MODEL),
        "statistical_expansion_terminal": True,
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    (OUTPUT / "stage15_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
