# Software validation summary

## Numerical equivalence

The JavaScript browser equation was compared with the independent frozen Python implementation (`scripts/predict_from_published_equation.py` in the parent analysis workspace) on 72 deterministic synthetic cases spanning spline knots, observed-range boundaries, and all categorical levels.

- Cases: 72 synthetic combinations
- Patient data: none
- Acceptance tolerance: maximum absolute probability difference below `1e-10`
- Observed maximum absolute difference: `1.1102230246251565e-16`
- Status: PASS

## Additional checks

- Frozen SHA identifier matches the published-equation specification: PASS
- Example patient probability (`0.2939836687324339`) reproduces exactly within `1e-12`: PASS
- Production build completes: PASS
- Server-rendered page contains research-use, privacy, and unresolved-horizon safeguards: PASS
- ESLint static analysis: PASS after final validation

Run `node --test tests/model-integrity.test.mjs tests/rendered-html.test.mjs` to repeat the deterministic checks.
