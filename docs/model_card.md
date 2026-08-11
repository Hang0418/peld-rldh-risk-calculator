# Model card

## Intended use

The calculator reproduces the frozen PELD-RLDH model for research, independent verification, and manuscript dissemination. It accepts eight preoperative variables and returns an estimated recurrence probability.

It is not validated as a treatment-selection instrument, does not estimate causal effects, and has not undergone prospective clinical-impact evaluation. Local validation and calibration assessment are required before any routine clinical use.

## Model lineage

- Version: `PELD_RLDH_V5_20260810`
- Frozen model: Stable Core RCS Ridge
- SHA-256: `8e81fd2dc45af5ca6792eddf3cf93934d6b62406758be0c071a0c8e7ed41f0a2`
- Development design: nested leave-one-hospital-out internal-external cross-validation in H1–H4
- Locked validation: H5 external validation followed by H6 independent stress validation

## Output

The primary output is the original frozen calibrated probability. No H5- or H6-derived recalibration is applied. The optional quartile label uses thresholds from frozen H1–H4 predictions and is descriptive rather than a treatment threshold.

The operational RLDH definition, prediction horizon, follow-up completeness, and censoring rules remain author queries. The webpage intentionally does not assert a 2-year horizon.

## Known limitations

- Retrospective source data and center-specific measurement practices may affect transportability.
- External calibration showed some absolute-risk underestimation.
- H6 contained 36 events, limiting calibration precision.
- Values outside the observed development ranges have weaker empirical support.
- Feature-contribution bars are additive changes in calibrated model log-odds relative to a fixed reference profile; they are not causal explanations.
