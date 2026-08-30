#!/usr/bin/env python3
"""Stage 14 extensions for the frozen PELD-RLDH v5 model.

Inputs are the locked H1-H6 datasets, frozen model, IECV predictions, and immutable
external predictions. The script does not refit, reselect, or alter the frozen model.
It creates interpretation, stability, paired incremental-value, age sensitivity,
calibration-transport, operating-point, subgroup, and centre-forest artifacts.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib.pyplot as plt
import seaborn as sns

from rldh_model_utils import apply_platt, metric_row


VERSION = "PELD_RLDH_V5_20260810"
BASE = ROOT / "outputs" / VERSION
FROZEN = ROOT / "data" / "frozen" / VERSION
DEVELOPMENT = FROZEN / "01_development_H1-H4.csv"
MODEL = BASE / "model_freeze" / "frozen_model.joblib"
DEV_OUTPUT = BASE / "model_development_h1_h4"
H5_OUTPUT = BASE / "external_validation" / "h5_primary"
H6_OUTPUT = BASE / "external_validation" / "h6_stress_test"
OUTPUT = BASE / "stage14_extensions"
FIGURES = OUTPUT / "figures"
SEED = 20260810
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
CONTINUOUS_FEATURES = [
    "Age",
    "BMI",
    "Disc_height_index",
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Lumbar_lordosis/degrees",
    "Sacral_slope/degrees",
]


warnings.filterwarnings("ignore", message="Setting penalty=None will ignore.*")
warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discrimination_accuracy(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def load_prediction_cohorts() -> dict[str, pd.DataFrame]:
    development = pd.read_csv(DEVELOPMENT)
    dev_predictions = pd.read_csv(DEV_OUTPUT / "iecv_predictions.csv")
    dev_predictions = dev_predictions[dev_predictions["candidate"] == "M2_Stable_Core_RCS_Ridge"]
    development = development.merge(
        dev_predictions[["Hospital", "id", "probability"]],
        on=["Hospital", "id"],
        validate="one_to_one",
    )
    h5 = pd.read_csv(FROZEN / "02_external_H5_SEALED.csv").merge(
        pd.read_csv(H5_OUTPUT / "h5_predictions.csv")[
            ["Hospital", "id", "original_probability"]
        ],
        on=["Hospital", "id"],
        validate="one_to_one",
    ).rename(columns={"original_probability": "probability"})
    h6 = pd.read_csv(FROZEN / "03_external_H6_SEALED.csv").merge(
        pd.read_csv(H6_OUTPUT / "h6_predictions.csv")[
            ["Hospital", "id", "original_probability"]
        ],
        on=["Hospital", "id"],
        validate="one_to_one",
    ).rename(columns={"original_probability": "probability"})
    return {"H1-H4 IECV": development, "H5 primary": h5, "H6 stress": h6}


def predictor_interpretation(artifact: dict, development: pd.DataFrame) -> None:
    features = artifact["features"]
    curves = []
    for feature in [value for value in features if value in CONTINUOUS_FEATURES]:
        lower, upper = development[feature].quantile([0.01, 0.99])
        grid = np.linspace(float(lower), float(upper), 100)
        for value in grid:
            standardized = development[features].copy()
            standardized[feature] = value
            raw = artifact["pipeline"].predict_proba(standardized)[:, 1]
            probability = apply_platt(artifact["calibrator"], raw)
            curves.append(
                {
                    "variable": feature,
                    "value": value,
                    "marginal_adjusted_risk": float(np.mean(probability)),
                }
            )
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(OUTPUT / "continuous_predictor_marginal_risk_curves.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (feature, part) in zip(axes.ravel(), curve_frame.groupby("variable", sort=False)):
        ax.plot(part["value"], part["marginal_adjusted_risk"], color="#1D3557", linewidth=2.4)
        ax.axvline(development[feature].median(), color="#E76F51", linestyle="--", linewidth=1.4)
        ax.set(title=feature, xlabel="Observed-scale value", ylabel="Marginal predicted risk")
    for ax in axes.ravel()[curve_frame["variable"].nunique() :]:
        ax.axis("off")
    fig.suptitle("Frozen stable-core RCS model: continuous predictor response profiles")
    fig.tight_layout()
    fig.savefig(FIGURES / "continuous_predictor_marginal_risk_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    model = artifact["pipeline"].named_steps["model"]
    numeric = [feature for feature in features if feature in CONTINUOUS_FEATURES]
    categorical = [feature for feature in features if feature not in CONTINUOUS_FEATURES]
    names = []
    for feature in numeric:
        names.extend([feature, f"{feature}_rcs1", f"{feature}_rcs2"])
    encoder = preprocessor.named_transformers_["cat"].named_steps["one_hot"]
    for feature, categories in zip(categorical, encoder.categories_):
        reference = str(categories[0])
        names.extend([f"{feature}: {level} vs {reference}" for level in categories[1:]])
    coefficients = model.coef_[0]
    if len(names) != len(coefficients):
        raise RuntimeError("Transformed feature names do not match frozen coefficients")
    platt_slope = float(artifact["calibrator"].coef_[0, 0])
    pd.DataFrame(
        {
            "transformed_term": names,
            "ridge_coefficient": coefficients,
            "effective_calibrated_logit_coefficient": coefficients * platt_slope,
        }
    ).to_csv(OUTPUT / "frozen_transformed_coefficient_table.csv", index=False)


def stability_heatmap() -> None:
    stability = pd.read_csv(DEV_OUTPUT / "stable_feature_selection.csv")
    outer = stability[stability["training_hospitals"].str.count("-") == 2].copy()
    outer["held_out_hospital"] = outer["training_hospitals"].map(
        lambda value: next(str(number) for number in range(1, 5) if str(number) not in value.split("-"))
    )
    matrix = outer.pivot(index="variable", columns="held_out_hospital", values="selection_frequency")
    ordering = outer.groupby("variable")["selection_frequency"].mean().sort_values(ascending=False).index
    matrix = matrix.loc[ordering]
    matrix.to_csv(OUTPUT / "predictor_stability_heatmap_matrix.csv")
    summary = outer.groupby("variable", as_index=False).agg(
        selected_outer_folds=("selected", "sum"),
        mean_selection_frequency=("selection_frequency", "mean"),
        mean_absolute_coefficient=("mean_absolute_coefficient", "mean"),
    ).sort_values(["mean_selection_frequency", "mean_absolute_coefficient"], ascending=False)
    summary.to_csv(OUTPUT / "predictor_stability_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Bootstrap selection frequency"},
        ax=ax,
    )
    ax.set(xlabel="Outer held-out hospital", ylabel="Candidate predictor", title="Training-only predictor stability across IECV domains")
    fig.tight_layout()
    fig.savefig(FIGURES / "predictor_stability_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def paired_incremental_bootstrap() -> None:
    predictions = pd.read_csv(DEV_OUTPUT / "iecv_predictions.csv")
    core = predictions[predictions["candidate"] == "M2_Stable_Core_RCS_Ridge"][
        ["Hospital", "id", "Recurrence", "probability"]
    ].rename(columns={"probability": "core_probability"})
    comparators = ["M3_Full_ElasticNet", "M5_Full_ExtraTrees"]
    rng = np.random.default_rng(SEED)
    rows = []
    for comparator in comparators:
        other = predictions[predictions["candidate"] == comparator][
            ["Hospital", "id", "probability"]
        ].rename(columns={"probability": "comparator_probability"})
        paired = core.merge(other, on=["Hospital", "id"], validate="one_to_one")
        point_core = discrimination_accuracy(paired["Recurrence"].to_numpy(), paired["core_probability"].to_numpy())
        point_other = discrimination_accuracy(paired["Recurrence"].to_numpy(), paired["comparator_probability"].to_numpy())
        strata = [group.index.to_numpy() for _, group in paired.groupby(["Hospital", "Recurrence"])]
        draws = {metric: [] for metric in point_core}
        for _ in range(2000):
            sampled_index = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
            sampled = paired.loc[sampled_index]
            y_true = sampled["Recurrence"].to_numpy()
            core_metrics = discrimination_accuracy(y_true, sampled["core_probability"].to_numpy())
            other_metrics = discrimination_accuracy(y_true, sampled["comparator_probability"].to_numpy())
            for metric in draws:
                draws[metric].append(core_metrics[metric] - other_metrics[metric])
        for metric, values in draws.items():
            rows.append(
                {
                    "comparison": f"Stable_Core_RCS minus {comparator}",
                    "metric": metric,
                    "core_estimate": point_core[metric],
                    "comparator_estimate": point_other[metric],
                    "difference_core_minus_comparator": point_core[metric] - point_other[metric],
                    "lower_95": float(np.quantile(values, 0.025)),
                    "upper_95": float(np.quantile(values, 0.975)),
                    "bootstrap_repetitions": 2000,
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "paired_incremental_performance_bootstrap.csv", index=False)


def age_sensitivity(cohorts: dict[str, pd.DataFrame]) -> None:
    rows = []
    for cohort, data in cohorts.items():
        for analysis, subset in [("all_ages", data), ("exclude_age_under_18", data[data["Age"] >= 18])]:
            rows.append(
                {
                    "cohort": cohort,
                    "analysis": analysis,
                    "excluded_n": int(len(data) - len(subset)),
                    **metric_row(subset["Recurrence"], subset["probability"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "age_under18_sensitivity.csv", index=False)


def calibration_transport_summary() -> None:
    h5 = pd.read_csv(H5_OUTPUT / "h5_metrics.csv").assign(cohort="H5 primary")
    h6 = pd.read_csv(H6_OUTPUT / "h6_metrics.csv").assign(cohort="H6 stress")
    frame = pd.concat([h5, h6], ignore_index=True)
    frame.to_csv(OUTPUT / "calibration_transport_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["brier", "log_loss", "oe_ratio"]):
        sns.barplot(data=frame, x="cohort", y=metric, hue="analysis", ax=ax)
        ax.set(xlabel="", title=metric.replace("_", " ").title())
        if metric == "oe_ratio":
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    axes[2].legend(title="Analysis", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("Original and secondary recalibration-transport analyses")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_transport_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def operating_point_table(cohorts: dict[str, pd.DataFrame]) -> None:
    rows = []
    for cohort, data in cohorts.items():
        y_true = data["Recurrence"].to_numpy(int)
        probability = data["probability"].to_numpy(float)
        prevalence = float(y_true.mean())
        for threshold in THRESHOLDS:
            predicted = probability >= threshold
            tp = int(np.sum(predicted & (y_true == 1)))
            fp = int(np.sum(predicted & (y_true == 0)))
            tn = int(np.sum(~predicted & (y_true == 0)))
            fn = int(np.sum(~predicted & (y_true == 1)))
            odds = threshold / (1 - threshold)
            rows.append(
                {
                    "cohort": cohort,
                    "threshold": threshold,
                    "sensitivity": tp / max(tp + fn, 1),
                    "specificity": tn / max(tn + fp, 1),
                    "ppv": tp / max(tp + fp, 1),
                    "npv": tn / max(tn + fn, 1),
                    "model_net_benefit": tp / len(data) - fp / len(data) * odds,
                    "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                    "treat_none_net_benefit": 0.0,
                    "descriptive_not_optimized": True,
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "prespecified_threshold_operating_points.csv", index=False)


def subgroup_assignments(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "Age group": pd.cut(data["Age"], [-np.inf, 50, 65, np.inf], right=False, labels=["<50", "50-64", ">=65"]),
        "Gender": data["Gender"].astype(str),
        "BMI group": pd.cut(data["BMI"], [-np.inf, 24, 28, np.inf], right=False, labels=["<24", "24-27.9", ">=28"]),
        "Modic": data["Modic_group"].astype(str),
        "Pfirrmann": data["Pfirrmann_group"].astype(str),
        "Herniation type": data["Herniation_type"].astype(str),
    }


def bootstrap_selected_metrics(data: pd.DataFrame, repetitions: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = {"auroc": [], "calibration_slope": [], "oe_ratio": []}
    for _ in range(repetitions):
        sampled = data.iloc[rng.choice(len(data), size=len(data), replace=True)]
        if sampled["Recurrence"].nunique() < 2:
            continue
        metrics = metric_row(sampled["Recurrence"], sampled["probability"])
        for metric in draws:
            draws[metric].append(metrics[metric])
    result = {}
    for metric, values in draws.items():
        result[f"{metric}_lower_95"] = float(np.quantile(values, 0.025)) if values else np.nan
        result[f"{metric}_upper_95"] = float(np.quantile(values, 0.975)) if values else np.nan
    result["successful_bootstraps"] = min(len(values) for values in draws.values())
    return result


def subgroup_transportability(cohorts: dict[str, pd.DataFrame]) -> None:
    rows = []
    definitions = [
        {"variable": "Age group", "definition": "<50; 50-64; >=65 years"},
        {"variable": "Gender", "definition": "Observed categories"},
        {"variable": "BMI group", "definition": "<24; 24-27.9; >=28 kg/m^2"},
        {"variable": "Modic", "definition": "Observed Modic_group categories"},
        {"variable": "Pfirrmann", "definition": "Observed Pfirrmann_group categories"},
        {"variable": "Herniation type", "definition": "Observed categories"},
    ]
    pd.DataFrame(definitions).to_csv(OUTPUT / "subgroup_definitions.csv", index=False)
    seed_offset = 0
    for cohort, data in cohorts.items():
        for variable, assignment in subgroup_assignments(data).items():
            working = data.assign(subgroup=assignment)
            for subgroup, part in working.groupby("subgroup", observed=True):
                if part["Recurrence"].nunique() < 2:
                    metrics = {
                        "n": int(len(part)),
                        "events": int(part["Recurrence"].sum()),
                        "prevalence": float(part["Recurrence"].mean()),
                        "auroc": np.nan,
                        "auprc": np.nan,
                        "brier": float(brier_score_loss(part["Recurrence"], part["probability"])),
                        "log_loss": float(log_loss(part["Recurrence"], part["probability"], labels=[0, 1])),
                        "calibration_intercept": np.nan,
                        "calibration_slope": np.nan,
                        "oe_ratio": float(part["Recurrence"].sum() / part["probability"].sum()),
                    }
                else:
                    metrics = metric_row(part["Recurrence"], part["probability"])
                precision = "adequate" if metrics["events"] >= 20 else "limited_events"
                intervals = (
                    bootstrap_selected_metrics(part, 300, SEED + seed_offset)
                    if metrics["events"] >= 10 and metrics["n"] - metrics["events"] >= 10
                    else {
                        "auroc_lower_95": np.nan,
                        "auroc_upper_95": np.nan,
                        "calibration_slope_lower_95": np.nan,
                        "calibration_slope_upper_95": np.nan,
                        "oe_ratio_lower_95": np.nan,
                        "oe_ratio_upper_95": np.nan,
                        "successful_bootstraps": 0,
                    }
                )
                seed_offset += 1
                rows.append(
                    {
                        "cohort": cohort,
                        "subgroup_variable": variable,
                        "subgroup": str(subgroup),
                        "precision_flag": precision,
                        **metrics,
                        **intervals,
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT / "subgroup_transportability_metrics.csv", index=False)


def centre_forest(cohorts: dict[str, pd.DataFrame]) -> None:
    rows = []
    development = cohorts["H1-H4 IECV"]
    centre_parts = [(f"H{hospital}", part) for hospital, part in development.groupby("Hospital")]
    centre_parts.extend([("H5", cohorts["H5 primary"]), ("H6", cohorts["H6 stress"])])
    for index, (centre, part) in enumerate(centre_parts):
        metrics = metric_row(part["Recurrence"], part["probability"])
        intervals = bootstrap_selected_metrics(part, 2000, SEED + 1000 + index)
        rows.append({"centre": centre, **metrics, **intervals})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "centre_performance_forest_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    positions = np.arange(len(frame))[::-1]
    axes[0].errorbar(
        frame["auroc"],
        positions,
        xerr=[frame["auroc"] - frame["auroc_lower_95"], frame["auroc_upper_95"] - frame["auroc"]],
        fmt="o",
        color="#1D3557",
        capsize=3,
    )
    axes[0].axvline(0.5, color="grey", linestyle="--")
    axes[0].set(xlabel="AUROC (bootstrap 95% CI)", yticks=positions, yticklabels=frame["centre"], title="Discrimination")
    axes[1].errorbar(
        frame["calibration_slope"],
        positions,
        xerr=[
            frame["calibration_slope"] - frame["calibration_slope_lower_95"],
            frame["calibration_slope_upper_95"] - frame["calibration_slope"],
        ],
        fmt="o",
        color="#E76F51",
        capsize=3,
    )
    axes[1].axvline(1, color="black", linestyle="--")
    axes[1].set(xlabel="Calibration slope (bootstrap 95% CI)", title="Calibration")
    fig.suptitle("Held-out development centres and sequential external validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "centre_performance_forest_plot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(MODEL)
    development = pd.read_csv(DEVELOPMENT)
    cohorts = load_prediction_cohorts()
    predictor_interpretation(artifact, development)
    stability_heatmap()
    paired_incremental_bootstrap()
    age_sensitivity(cohorts)
    calibration_transport_summary()
    operating_point_table(cohorts)
    subgroup_transportability(cohorts)
    centre_forest(cohorts)
    run_metadata = {
        "version": VERSION,
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "random_seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "bootstrap_repetitions": {
            "paired_incremental": 2000,
            "subgroup": 300,
            "centre_forest": 2000,
        },
    }
    (OUTPUT / "stage14_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    unhashed_control_files = {"stage14_manifest.json", "stage14_independent_verification.json"}
    files = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name not in unhashed_control_files
    )
    manifest = {
        "version": VERSION,
        "status": "STAGE14_EXTENSIONS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "frozen_model_sha256": sha256(MODEL),
        "refit_or_reselection_performed": False,
        "h5_h6_used_for_model_selection": False,
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    (OUTPUT / "stage14_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
