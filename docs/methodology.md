# Calculator methodology

The website is a display and inference layer for the published-equation specification. It does not fit or modify a model.

For each numeric predictor, the browser:

1. creates the exported Harrell restricted cubic spline basis;
2. applies the exported mean and scale to every basis column;
3. concatenates the encoded categorical columns;
4. evaluates the frozen ridge-logistic linear predictor; and
5. applies the frozen Platt intercept and slope.

The risk is calculated entirely in JavaScript. Model parameters are loaded from `public/model/model_specification.json`; inputs and outputs are not sent to a server or stored.

The contribution display compares the patient design vector with a fixed reference vector composed of the development medians and modes. Each difference is multiplied by the ridge coefficient and Platt slope. The resulting terms add on the calibrated log-odds scale.

The frozen probability is the sole primary output. The calculator does not use external-validation data to select features, tune hyperparameters, select a model, or recalibrate predictions.
