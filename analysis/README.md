# Key analysis code

This directory contains the principal Python code for the active
`PELD_RLDH_V5_20260810` analysis lineage. It is a code-only reproducibility
release: no patient-level records, frozen cohort tables, fitted model binaries,
predictions, credentials, or local machine paths are included.

## Workflow

1. `01_lock_and_audit_updated_data.py` audits the workbook schema, identifiers,
   missingness, plausibility, outcomes, and hospital roles, then creates a
   versioned data lock.
2. `02_develop_and_freeze_h1_h4.py` performs prespecified nested
   leave-one-hospital-out internal-external cross-validation in H1-H4, applies
   transportability gates, and freezes the selected model.
3. `03_verify_development_stop.py` independently recomputes structural and
   numerical checks before any external cohort may be opened.
4. `05_external_validate_h5.py` applies the untouched frozen model to H5 and
   locks the original primary external-validation results.
5. `06_external_validate_h6.py` applies the same model to H6 only after H5 has
   been locked; `07_verify_external_validation.py` independently verifies both
   external cohorts.
6. `08_stage14_extensions.py` and `09_verify_stage14_extensions.py` implement
   frozen-model interpretation, stability, subgroup, threshold, and sensitivity
   analyses without model reselection.
Shared model and metric functions are in `scripts/rldh_model_utils.py`.
Standalone equation inference is in
`scripts/predict_from_published_equation.py`.

Numbering follows the archived analysis chronology; step 04 is not required by
the public core workflow.

## Directory contract

Place an authorized source workbook at:

```text
analysis/PELD术后RLDH.xlsx
```

The workbook must contain a `Merged_Data` worksheet with the exact schema
checked by `01_lock_and_audit_updated_data.py`. Patient-level analysis files
remain under ignored `analysis/data/` and `analysis/outputs/` directories and
must not be committed.

## Environment

The archived run used Python 3.12.13. Recreate the tested package environment
with:

```bash
python -m pip install -r analysis/requirements.txt
```

## Inspect or run

Display stage order, source-lineage checks, and current completion gates without
running analysis:

```bash
python analysis/main_analysis_pipeline/run_pipeline.py
```

Execute pending stages only after placing the authorized source workbook:

```bash
python analysis/main_analysis_pipeline/run_pipeline.py --run
```

The runner stops when the source hash changes or a prerequisite/completion gate
fails. H5 and H6 are never used to change the selected primary frozen model;
any recalibration analysis remains explicitly secondary.

## Public-release boundary

The scripts expose the analysis logic and prespecified gates. The repository
does not provide clinical data or a fitted Python model. The browser calculator
uses the separately published frozen equation in `../model/` and is intended for
research and reproducibility use only.
