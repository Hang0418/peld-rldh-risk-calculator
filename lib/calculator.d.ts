export type Inputs = Record<string, string | number>;
export type ModelSpec = {
  version: string;
  raw_predictors: string[];
  numeric: {
    features: string[];
    imputation_medians: number[];
    knots: Record<string, number[]>;
    scaler_mean: number[];
    scaler_scale: number[];
  };
  categorical: {
    features: string[];
    imputation_modes: string[];
    levels: Record<string, string[]>;
    drop_index: Record<string, number>;
  };
  ridge_model: { intercept: number; coefficients: number[] };
  platt_calibration: { intercept: number; slope: number; probability_clip: number[] };
  model_sha256: string;
};
export function rcsBasis(value: number, knots: number[]): number[];
export function designVector(inputs: Inputs, specification: ModelSpec): number[];
export function predictRisk(inputs: Inputs, specification: ModelSpec): number;
export function developmentRiskStratum(probability: number): string;
export function contributionProfile(inputs: Inputs, specification: ModelSpec): { feature: string; value: number }[];
export function validateInputs(inputs: Inputs): { errors: string[]; warnings: string[] };
export const DEVELOPMENT_RANGES: Record<string, [number, number]>;
export const RISK_THRESHOLDS: number[];
