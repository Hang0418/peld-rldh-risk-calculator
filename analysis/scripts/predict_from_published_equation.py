#!/usr/bin/env python3
"""Predict PELD-RLDH risk from the exported equation without loading joblib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT.parent / "model" / "model_specification.json"
EPSILON = 1e-6


def expit(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


def rcs_basis(values: np.ndarray, knots: list[float]) -> np.ndarray:
    knots_array = np.asarray(knots, dtype=float)
    first, penultimate, last = knots_array[0], knots_array[-2], knots_array[-1]
    scale = max((last - first) ** 2, EPSILON)
    columns = [values]
    for knot in knots_array[:-2]:
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


def predict_from_equation(data: pd.DataFrame, specification: dict) -> np.ndarray:
    transformed_parts = []
    numeric = specification["numeric"]
    for feature_index, feature in enumerate(numeric["features"]):
        values = pd.to_numeric(data[feature], errors="coerce").to_numpy(float)
        values = np.where(np.isnan(values), numeric["imputation_medians"][feature_index], values)
        transformed_parts.append(rcs_basis(values, numeric["knots"][feature]))
    numeric_matrix = np.column_stack(transformed_parts)
    numeric_matrix = (
        numeric_matrix - np.asarray(numeric["scaler_mean"], dtype=float)
    ) / np.asarray(numeric["scaler_scale"], dtype=float)

    categorical = specification["categorical"]
    categorical_columns = []
    for feature_index, feature in enumerate(categorical["features"]):
        values = data[feature].astype("object").copy()
        values = values.where(values.notna(), categorical["imputation_modes"][feature_index]).astype(str)
        levels = [str(value) for value in categorical["levels"][feature]]
        unknown = sorted(set(values.unique()) - set(levels))
        if unknown:
            raise ValueError(f"Unknown level(s) for {feature}: {unknown}")
        drop_index = categorical["drop_index"][feature]
        for level_index, level in enumerate(levels):
            if level_index == drop_index:
                continue
            categorical_columns.append((values == level).to_numpy(float))
    categorical_matrix = (
        np.column_stack(categorical_columns)
        if categorical_columns
        else np.empty((len(data), 0), dtype=float)
    )
    design = np.column_stack([numeric_matrix, categorical_matrix])
    model = specification["ridge_model"]
    linear_predictor = float(model["intercept"]) + design @ np.asarray(model["coefficients"], dtype=float)
    raw_probability = expit(linear_predictor)
    calibration = specification["platt_calibration"]
    calibrated_logit = float(calibration["intercept"]) + float(calibration["slope"]) * logit(raw_probability)
    return expit(calibrated_logit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    data = pd.read_csv(args.input_csv)
    required = specification["raw_predictors"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {missing}")
    output = data.copy()
    output["predicted_recurrence_probability"] = predict_from_equation(data, specification)
    output.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
