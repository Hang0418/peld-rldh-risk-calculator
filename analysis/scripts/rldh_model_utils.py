"""Shared, versioned utilities for the PELD-RLDH v5 model pipeline."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


SEED = 20260810
EPSILON = 1e-6


class RestrictedCubicSpline(BaseEstimator, TransformerMixin):
    """Harrell-style restricted cubic spline basis using training quantile knots."""

    def __init__(self, quantiles: tuple[float, ...] = (0.05, 0.35, 0.65, 0.95)):
        self.quantiles = quantiles

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RestrictedCubicSpline":
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        self.n_features_in_ = values.shape[1]
        self.knots_ = []
        for column in range(values.shape[1]):
            knots = np.quantile(values[:, column], self.quantiles)
            knots = np.maximum.accumulate(knots.astype(float))
            if np.unique(knots).size < 4:
                minimum, maximum = float(np.min(values[:, column])), float(np.max(values[:, column]))
                knots = np.linspace(minimum, maximum, 4)
            self.knots_.append(knots)
        return self

    @staticmethod
    def _basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
        first, penultimate, last = knots[0], knots[-2], knots[-1]
        scale = max((last - first) ** 2, EPSILON)
        columns = [values]
        for knot in knots[:-2]:
            positive_knot = np.maximum(values - knot, 0.0) ** 3
            positive_penultimate = np.maximum(values - penultimate, 0.0) ** 3
            positive_last = np.maximum(values - last, 0.0) ** 3
            denominator = max(last - penultimate, EPSILON)
            restricted = (
                positive_knot
                - ((last - knot) / denominator) * positive_penultimate
                + ((penultimate - knot) / denominator) * positive_last
            ) / scale
            columns.append(restricted)
        return np.column_stack(columns)

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        transformed = [self._basis(values[:, index], self.knots_[index]) for index in range(values.shape[1])]
        return np.column_stack(transformed)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = (
            list(input_features)
            if input_features is not None
            else [f"x{index}" for index in range(self.n_features_in_)]
        )
        output = []
        for name in names:
            output.extend([name, f"{name}_rcs1", f"{name}_rcs2"])
        return np.asarray(output, dtype=object)


def fit_platt(y_true: np.ndarray, probabilities: np.ndarray) -> LogisticRegression:
    x = logit(np.clip(probabilities, EPSILON, 1 - EPSILON)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, random_state=SEED)
    model.fit(x, y_true)
    return model


def apply_platt(model: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    x = logit(np.clip(probabilities, EPSILON, 1 - EPSILON)).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def calibration_parameters(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    lp = logit(np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)).reshape(-1, 1)
    if np.unique(y_true).size < 2:
        return math.nan, math.nan
    try:
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, random_state=SEED)
        model.fit(lp, y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return math.nan, math.nan


def metric_row(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    intercept, slope = calibration_parameters(y_true, probabilities)
    return {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "oe_ratio": float(y_true.sum() / probabilities.sum()),
    }


def auc_variance_hanley(auc: float, events: int, non_events: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    numerator = (
        auc * (1 - auc)
        + (events - 1) * (q1 - auc * auc)
        + (non_events - 1) * (q2 - auc * auc)
    )
    return max(numerator / max(events * non_events, 1), EPSILON**2)


def auc_i_squared(center_metrics: pd.DataFrame) -> float:
    estimates = center_metrics["auroc"].to_numpy(float)
    variances = np.asarray(
        [
            auc_variance_hanley(row.auroc, int(row.events), int(row.n - row.events))
            for row in center_metrics.itertuples()
        ],
        dtype=float,
    )
    weights = 1.0 / variances
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    q = float(np.sum(weights * (estimates - pooled) ** 2))
    degrees = max(len(estimates) - 1, 1)
    return float(max(0.0, (q - degrees) / max(q, EPSILON)) * 100.0)


def decision_curve(y_true: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(y_true)
    prevalence = float(y_true.mean())
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & (y_true == 1)))
        false_positive = int(np.sum(predicted & (y_true == 0)))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "model_net_benefit": true_positive / n - false_positive / n * odds,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_intervals(
    data: pd.DataFrame,
    probability_column: str,
    strata_columns: list[str] | None = None,
    repetitions: int = 1000,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope", "oe_ratio"]
    draws = []
    strata_columns = strata_columns or []
    if strata_columns:
        groups = [part.index.to_numpy() for _, part in data.groupby(strata_columns, dropna=False)]
    else:
        groups = [data.index.to_numpy()]
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        sampled = data.loc[indices]
        if sampled["Recurrence"].nunique() < 2:
            continue
        draws.append(metric_row(sampled["Recurrence"].to_numpy(), sampled[probability_column].to_numpy()))
    draw_frame = pd.DataFrame(draws)
    estimate = metric_row(data["Recurrence"].to_numpy(), data[probability_column].to_numpy())
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "estimate": estimate[metric],
                "lower_95": float(draw_frame[metric].quantile(0.025)),
                "upper_95": float(draw_frame[metric].quantile(0.975)),
                "successful_bootstraps": int(draw_frame[metric].notna().sum()),
            }
        )
    return pd.DataFrame(rows)
