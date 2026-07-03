# 2-Year RLDH Risk Calculator After PELD

This folder contains a static research-use web calculator for estimating the predicted probability of 2-year recurrent lumbar disc herniation after percutaneous endoscopic lumbar discectomy.

## Files

- `index.html`: calculator page
- `styles.css`: clinical-style page layout
- `calculator.js`: locked full-variable logistic-regression coefficients and calculation logic

## Intended Use

The calculator is intended for research-use risk communication and postoperative follow-up stratification. It should not be used as a stand-alone indication for revision surgery, fusion surgery, or treatment escalation.

## Deployment

The site is static and can be deployed through GitHub Pages, institutional hosting, or any static web server. For GitHub Pages, upload these files to a repository and enable Pages from the repository settings.

## Risk Strata

- Low risk: predicted risk <10%
- Intermediate risk: predicted risk 10%-30%
- High risk: predicted risk >30%

## Model Note

The calculator uses the locked full prespecified logistic-regression model. XGBoost was used separately for model-level interpretation in the manuscript.
