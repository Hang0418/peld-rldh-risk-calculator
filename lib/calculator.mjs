const EPSILON = 1e-6;

export const DEVELOPMENT_RANGES = {
  Age: [12, 88.2],
  "sROM/degrees": [1, 35],
  "Cross_sectional_area/cm^2": [1.114, 29.494],
  "Sacral_slope/degrees": [15, 49.53],
  Disc_height_index: [0.1, 0.5],
};

export const RISK_THRESHOLDS = [
  0.0268629595550639,
  0.04929256196429735,
  0.09761429417968598,
];

function expit(value) {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function logit(probability) {
  const clipped = Math.min(1 - EPSILON, Math.max(EPSILON, probability));
  return Math.log(clipped / (1 - clipped));
}

function positiveCube(value) {
  return Math.max(value, 0) ** 3;
}

export function rcsBasis(value, knots) {
  const first = knots[0];
  const penultimate = knots[knots.length - 2];
  const last = knots[knots.length - 1];
  const scale = Math.max((last - first) ** 2, EPSILON);
  const denominator = Math.max(last - penultimate, EPSILON);
  const columns = [value];
  for (const knot of knots.slice(0, -2)) {
    const restricted = (
      positiveCube(value - knot)
      - ((last - knot) / denominator) * positiveCube(value - penultimate)
      + ((penultimate - knot) / denominator) * positiveCube(value - last)
    ) / scale;
    columns.push(restricted);
  }
  return columns;
}

function numericValue(inputs, feature, fallback) {
  const value = Number(inputs[feature]);
  return Number.isFinite(value) ? value : fallback;
}

export function designVector(inputs, specification) {
  const numeric = specification.numeric;
  const rawNumeric = [];
  numeric.features.forEach((feature, featureIndex) => {
    const value = numericValue(inputs, feature, numeric.imputation_medians[featureIndex]);
    rawNumeric.push(...rcsBasis(value, numeric.knots[feature]));
  });
  const scaledNumeric = rawNumeric.map(
    (value, index) => (value - numeric.scaler_mean[index]) / numeric.scaler_scale[index],
  );

  const categoricalColumns = [];
  const categorical = specification.categorical;
  categorical.features.forEach((feature, featureIndex) => {
    const supplied = inputs[feature];
    const value = supplied === "" || supplied === null || supplied === undefined
      ? categorical.imputation_modes[featureIndex]
      : String(supplied);
    const levels = categorical.levels[feature].map(String);
    if (!levels.includes(value)) throw new Error(`Unknown level for ${feature}: ${value}`);
    levels.forEach((level, levelIndex) => {
      if (levelIndex !== categorical.drop_index[feature]) {
        categoricalColumns.push(value === level ? 1 : 0);
      }
    });
  });
  return [...scaledNumeric, ...categoricalColumns];
}

export function predictRisk(inputs, specification) {
  const design = designVector(inputs, specification);
  const model = specification.ridge_model;
  const linearPredictor = model.intercept + design.reduce(
    (sum, value, index) => sum + value * model.coefficients[index],
    0,
  );
  const rawProbability = expit(linearPredictor);
  const calibration = specification.platt_calibration;
  return expit(calibration.intercept + calibration.slope * logit(rawProbability));
}

export function developmentRiskStratum(probability) {
  if (probability <= RISK_THRESHOLDS[0]) return "Q1 — Lowest";
  if (probability <= RISK_THRESHOLDS[1]) return "Q2";
  if (probability <= RISK_THRESHOLDS[2]) return "Q3";
  return "Q4 — Highest";
}

export function contributionProfile(inputs, specification) {
  const reference = {};
  specification.numeric.features.forEach((feature, index) => {
    reference[feature] = specification.numeric.imputation_medians[index];
  });
  specification.categorical.features.forEach((feature, index) => {
    reference[feature] = specification.categorical.imputation_modes[index];
  });
  const patientDesign = designVector(inputs, specification);
  const referenceDesign = designVector(reference, specification);
  const coefficients = specification.ridge_model.coefficients;
  const plattSlope = specification.platt_calibration.slope;
  const numericContributions = specification.numeric.features.map((feature, index) => {
    const start = index * 3;
    const value = [0, 1, 2].reduce(
      (sum, offset) => sum + (patientDesign[start + offset] - referenceDesign[start + offset]) * coefficients[start + offset],
      0,
    ) * plattSlope;
    return { feature, value };
  });
  let cursor = 15;
  const categoricalContributions = specification.categorical.features.map((feature) => {
    const value = [0, 1].reduce(
      (sum, offset) => sum + (patientDesign[cursor + offset] - referenceDesign[cursor + offset]) * coefficients[cursor + offset],
      0,
    ) * plattSlope;
    cursor += 2;
    return { feature, value };
  });
  return [...numericContributions, ...categoricalContributions].sort(
    (a, b) => Math.abs(b.value) - Math.abs(a.value),
  );
}

export function validateInputs(inputs) {
  const errors = [];
  const warnings = [];
  const required = [
    "Modic_group", "sROM/degrees", "Cross_sectional_area/cm^2", "Pfirrmann_group",
    "Sacral_slope/degrees", "Age", "Disc_height_index", "Herniation_type",
  ];
  if (required.some((key) => inputs[key] === "" || inputs[key] === null || inputs[key] === undefined)) {
    errors.push("Complete all 8 predictor fields.");
    return { errors, warnings };
  }
  const hardRules = {
    Age: [0, 120, "Age must be between 0 and 120 years."],
    "sROM/degrees": [0, 90, "Sagittal range of motion must be between 0 and 90 degrees."],
    "Cross_sectional_area/cm^2": [Number.MIN_VALUE, 100, "Cross-sectional area must be greater than 0 and no more than 100 cm²."],
    "Sacral_slope/degrees": [0, 90, "Sacral slope must be between 0 and 90 degrees."],
    Disc_height_index: [Number.MIN_VALUE, 2, "Disc height index must be greater than 0 and no more than 2."],
  };
  Object.entries(hardRules).forEach(([feature, [minimum, maximum, message]]) => {
    const value = Number(inputs[feature]);
    if (!Number.isFinite(value) || value < minimum || value > maximum) errors.push(message);
  });
  if (errors.length > 0) return { errors, warnings };
  Object.entries(DEVELOPMENT_RANGES).forEach(([feature, [minimum, maximum]]) => {
    const value = Number(inputs[feature]);
    if (value < minimum || value > maximum) {
      warnings.push(`${feature.replaceAll("_", " ")} = ${value} is outside the observed range ${minimum}–${maximum}; reliability may be reduced.`);
    }
  });
  return { errors, warnings };
}
