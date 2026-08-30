#!/usr/bin/env python3
"""Lock the updated workbook and create auditable, role-separated datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PELD术后RLDH.xlsx"
VERSION = "PELD_RLDH_V5_20260810"
FROZEN = ROOT / "data" / "frozen" / VERSION
OUTPUT = ROOT / "outputs" / VERSION / "stage0_1_audit"
SUPERSEDED_LOCK = (
    ROOT
    / "data"
    / "frozen"
    / "PELD_RLDH_V4_20260810"
    / "PELD_RLDH_V4_20260810_MASTER_LOCKED.xlsx"
)

EXPECTED_COLUMNS = [
    "id",
    "Hospital",
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "Alcoholism",
    "Hypertension",
    "Diabetes",
    "Herniation_type",
    "Operated_segment",
    "Pfirrmann_group",
    "Modic_group",
    "Disc_height_index",
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Lumbar_lordosis/degrees",
    "Sacral_slope/degrees",
    "Recurrence",
]

CONTINUOUS = [
    "Age",
    "BMI",
    "Disc_height_index",
    "sROM/degrees",
    "Cross_sectional_area/cm^2",
    "Lumbar_lordosis/degrees",
    "Sacral_slope/degrees",
]
CATEGORICAL = [
    column
    for column in EXPECTED_COLUMNS
    if column not in {"id", "Hospital", "Recurrence", *CONTINUOUS}
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(SOURCE, sheet_name="Merged_Data", engine="openpyxl")
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected schema: {data.columns.tolist()}")

    key_duplicates = int(data.duplicated(["Hospital", "id"]).sum())
    invalid_hospitals = sorted(set(data["Hospital"].dropna()) - set(range(1, 7)))
    invalid_outcomes = sorted(set(data["Recurrence"].dropna()) - {0, 1})
    nonfinite_numeric = {}
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        nonfinite_numeric[column] = int((~np.isfinite(data[column].astype(float))).sum())

    hard_checks = {
        "schema_exact": data.columns.tolist() == EXPECTED_COLUMNS,
        "n_rows_positive": len(data) > 0,
        "hospital_id_unique": key_duplicates == 0,
        "hospital_codes_valid": len(invalid_hospitals) == 0,
        "outcome_binary": len(invalid_outcomes) == 0,
        "no_missing_cells": int(data.isna().sum().sum()) == 0,
        "numeric_values_finite": sum(nonfinite_numeric.values()) == 0,
    }
    if not all(hard_checks.values()):
        raise RuntimeError(f"Locking checks failed: {hard_checks}")

    locked_xlsx = FROZEN / f"{VERSION}_MASTER_LOCKED.xlsx"
    shutil.copy2(SOURCE, locked_xlsx)
    master_csv = FROZEN / f"{VERSION}_MASTER_LOCKED.csv"
    data.to_csv(master_csv, index=False, lineterminator="\n")

    role_files = {
        "development_h1_h4": FROZEN / "01_development_H1-H4.csv",
        "external_h5_sealed": FROZEN / "02_external_H5_SEALED.csv",
        "external_h6_sealed": FROZEN / "03_external_H6_SEALED.csv",
    }
    data[data["Hospital"].isin([1, 2, 3, 4])].to_csv(
        role_files["development_h1_h4"], index=False, lineterminator="\n"
    )
    data[data["Hospital"] == 5].to_csv(
        role_files["external_h5_sealed"], index=False, lineterminator="\n"
    )
    data[data["Hospital"] == 6].to_csv(
        role_files["external_h6_sealed"], index=False, lineterminator="\n"
    )

    center_summary = (
        data.groupby("Hospital", as_index=False)
        .agg(n=("id", "size"), events=("Recurrence", "sum"))
        .assign(event_rate=lambda frame: frame["events"] / frame["n"])
    )
    center_summary.to_csv(OUTPUT / "center_summary.csv", index=False)

    missingness = pd.DataFrame(
        {
            "variable": data.columns,
            "missing_n": [int(data[column].isna().sum()) for column in data.columns],
            "missing_rate": [float(data[column].isna().mean()) for column in data.columns],
        }
    )
    missingness.to_csv(OUTPUT / "missingness.csv", index=False)

    plausibility_flags = pd.DataFrame(
        [
            {"rule": "Age < 18", "flagged_n": int((data["Age"] < 18).sum())},
            {
                "rule": "Age is not integer-valued",
                "flagged_n": int((data["Age"] % 1 != 0).sum()),
            },
            {
                "rule": "BMI outside 10-60",
                "flagged_n": int((~data["BMI"].between(10, 60)).sum()),
            },
            {
                "rule": "Disc height index outside 0-1",
                "flagged_n": int((~data["Disc_height_index"].between(0, 1)).sum()),
            },
            {
                "rule": "sROM outside 0-60 degrees",
                "flagged_n": int((~data["sROM/degrees"].between(0, 60)).sum()),
            },
            {
                "rule": "CSA outside 0-100 cm^2",
                "flagged_n": int((~data["Cross_sectional_area/cm^2"].between(0, 100)).sum()),
            },
            {
                "rule": "Lumbar lordosis outside 0-100 degrees",
                "flagged_n": int((~data["Lumbar_lordosis/degrees"].between(0, 100)).sum()),
            },
            {
                "rule": "Sacral slope outside 0-90 degrees",
                "flagged_n": int((~data["Sacral_slope/degrees"].between(0, 90)).sum()),
            },
        ]
    )
    plausibility_flags.to_csv(OUTPUT / "plausibility_flags.csv", index=False)

    continuous_rows = []
    for hospital, group in data.groupby("Hospital"):
        for column in CONTINUOUS:
            values = group[column].astype(float)
            continuous_rows.append(
                {
                    "Hospital": hospital,
                    "variable": column,
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    pd.DataFrame(continuous_rows).to_csv(OUTPUT / "continuous_by_center.csv", index=False)

    categorical_rows = []
    for hospital, group in data.groupby("Hospital"):
        for column in CATEGORICAL:
            counts = group[column].astype(str).value_counts(dropna=False)
            for level, count in counts.items():
                categorical_rows.append(
                    {
                        "Hospital": hospital,
                        "variable": column,
                        "level": level,
                        "n": int(count),
                        "proportion": float(count / len(group)),
                    }
                )
    pd.DataFrame(categorical_rows).to_csv(OUTPUT / "categorical_by_center.csv", index=False)

    outcome_group_rows = []
    for (hospital, outcome), group in data.groupby(["Hospital", "Recurrence"]):
        for column in CONTINUOUS:
            values = group[column].astype(float)
            outcome_group_rows.append(
                {
                    "Hospital": int(hospital),
                    "Recurrence": int(outcome),
                    "variable": column,
                    "kind": "continuous",
                    "n": int(values.notna().sum()),
                    "summary": (
                        f"{values.mean():.6g} ({values.std(ddof=1):.6g}); "
                        f"{values.median():.6g} [{values.quantile(0.25):.6g}, "
                        f"{values.quantile(0.75):.6g}]"
                    ),
                }
            )
        for column in CATEGORICAL:
            for level, count in group[column].astype(str).value_counts(dropna=False).items():
                outcome_group_rows.append(
                    {
                        "Hospital": int(hospital),
                        "Recurrence": int(outcome),
                        "variable": column,
                        "kind": "categorical",
                        "level": level,
                        "n": int(count),
                        "summary": f"{count} ({count / len(group):.6%})",
                    }
                )
    pd.DataFrame(outcome_group_rows).to_csv(
        OUTPUT / "baseline_by_center_and_outcome.csv", index=False
    )

    lineage_comparison = {"available": False}
    if SUPERSEDED_LOCK.exists():
        prior = pd.read_excel(SUPERSEDED_LOCK, sheet_name="Merged_Data", engine="openpyxl")
        comparison_keys = ["Hospital", "id"]
        if prior.columns.tolist() == data.columns.tolist():
            aligned = prior.merge(
                data,
                on=comparison_keys,
                how="outer",
                suffixes=("_v4", "_v5"),
                indicator=True,
            )
            changed_rows = np.zeros(len(aligned), dtype=bool)
            changed_cells = []
            for column in [value for value in data.columns if value not in comparison_keys]:
                left = aligned[f"{column}_v4"]
                right = aligned[f"{column}_v5"]
                changed = ~(left.eq(right) | (left.isna() & right.isna()))
                changed_rows |= changed.to_numpy()
                changed_cells.append(
                    {
                        "variable": column,
                        "changed_cells": int(changed.sum()),
                    }
                )
            pd.DataFrame(changed_cells).to_csv(
                OUTPUT / "superseded_v4_to_v5_changed_cells.csv", index=False
            )
            lineage_comparison = {
                "available": True,
                "prior_rows": int(len(prior)),
                "current_rows": int(len(data)),
                "left_only_rows": int((aligned["_merge"] == "left_only").sum()),
                "right_only_rows": int((aligned["_merge"] == "right_only").sum()),
                "rows_with_any_changed_value": int(changed_rows.sum()),
                "changed_cells": int(sum(row["changed_cells"] for row in changed_cells)),
            }
        else:
            lineage_comparison = {
                "available": True,
                "schema_match": False,
                "prior_columns": prior.columns.tolist(),
                "current_columns": data.columns.tolist(),
            }

    variable_dictionary = pd.DataFrame(
        [
            {
                "variable": column,
                "role": (
                    "identifier"
                    if column == "id"
                    else "domain"
                    if column == "Hospital"
                    else "outcome"
                    if column == "Recurrence"
                    else "candidate_predictor"
                ),
                "type": (
                    "identifier"
                    if column == "id"
                    else "domain"
                    if column == "Hospital"
                    else "binary_outcome"
                    if column == "Recurrence"
                    else "continuous"
                    if column in CONTINUOUS
                    else "categorical"
                ),
                "observed_levels_or_range": (
                    f"{data[column].min()} to {data[column].max()}"
                    if pd.api.types.is_numeric_dtype(data[column])
                    else " | ".join(sorted(data[column].astype(str).unique()))
                ),
                "unit": column.split("/", 1)[1] if "/" in column else "not encoded",
                "missing_n": int(data[column].isna().sum()),
            }
            for column in data.columns
        ]
    )
    variable_dictionary.to_csv(OUTPUT / "variable_dictionary.csv", index=False)

    file_registry = {
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "locked_workbook": {"path": str(locked_xlsx), "sha256": sha256(locked_xlsx)},
        "master_csv": {"path": str(master_csv), "sha256": sha256(master_csv)},
        **{
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in role_files.items()
        },
    }
    manifest = {
        "version": VERSION,
        "source_sheet": "Merged_Data",
        "n_rows": int(len(data)),
        "n_columns": int(data.shape[1]),
        "outcome": "Recurrence",
        "outcome_definition": "Binary recurrence indicator supplied by investigators (0/1)",
        "roles": {
            "H1-H4": "development_and_iecv",
            "H5": "primary_external_validation_sealed_until_freeze",
            "H6": "second_external_validation_stress_test_sealed_until_freeze",
        },
        "hard_checks": hard_checks,
        "plausibility_flags_are_descriptive_not_exclusions": plausibility_flags.to_dict(
            orient="records"
        ),
        "superseded_v4_comparison": lineage_comparison,
        "key_duplicates": key_duplicates,
        "invalid_hospitals": invalid_hospitals,
        "invalid_outcomes": invalid_outcomes,
        "files": file_registry,
    }
    write_json(FROZEN / "lock_manifest.json", manifest)
    write_json(OUTPUT / "stage0_1_audit_status.json", {**manifest, "status": "PASS_MODEL_DEVELOPMENT_MAY_START"})
    write_json(
        FROZEN / "EXTERNAL_COHORT_FIREWALL.json",
        {
            "status": "SEALED",
            "rule": "H5 and H6 cannot be read by model-development scripts before final freeze.",
            "h5_sha256": file_registry["external_h5_sealed"]["sha256"],
            "h6_sha256": file_registry["external_h6_sealed"]["sha256"],
        },
    )
    print(json.dumps({"status": "PASS_MODEL_DEVELOPMENT_MAY_START", "center_summary": center_summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
