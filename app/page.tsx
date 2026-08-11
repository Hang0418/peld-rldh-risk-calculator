"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  contributionProfile,
  developmentRiskStratum,
  predictRisk,
  validateInputs,
} from "@/lib/calculator.mjs";

type ModelSpec = {
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

type Inputs = Record<string, string | number>;

const EMPTY_INPUTS: Inputs = {
  Modic_group: "",
  "sROM/degrees": "",
  "Cross_sectional_area/cm^2": "",
  Pfirrmann_group: "",
  "Sacral_slope/degrees": "",
  Age: "",
  Disc_height_index: "",
  Herniation_type: "",
};

const EXAMPLE_INPUTS: Inputs = {
  Modic_group: "II–III",
  "sROM/degrees": 10,
  "Cross_sectional_area/cm^2": 9.5,
  Pfirrmann_group: "III–IV",
  "Sacral_slope/degrees": 28,
  Age: 55,
  Disc_height_index: 0.3,
  Herniation_type: "extrusion",
};

const LABELS: Record<string, string> = {
  Modic_group: "Modic change",
  "sROM/degrees": "Sagittal range of motion",
  "Cross_sectional_area/cm^2": "Cross-sectional area",
  Pfirrmann_group: "Pfirrmann grade",
  "Sacral_slope/degrees": "Sacral slope",
  Age: "Age",
  Disc_height_index: "Disc height index",
  Herniation_type: "Herniation type",
};

const PERFORMANCE = [
  { cohort: "H1–H4 nested IECV", n: "3,610", events: "295", auroc: "0.764", auprc: "0.255" },
  { cohort: "H5 external validation", n: "1,190", events: "117", auroc: "0.810", auprc: "0.419" },
  { cohort: "H6 stress validation", n: "288", events: "36", auroc: "0.833", auprc: "0.529" },
];

function Field({
  label,
  name,
  value,
  unit,
  help,
  onChange,
}: {
  label: string;
  name: string;
  value: string | number;
  unit?: string;
  help: string;
  onChange: (name: string, value: string) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <span className="input-shell">
        <input
          inputMode="decimal"
          name={name}
          type="number"
          step="any"
          value={value}
          onChange={(event) => onChange(name, event.target.value)}
          aria-describedby={`${name}-help`}
        />
        {unit && <span className="unit">{unit}</span>}
      </span>
      <span className="field-help" id={`${name}-help`}>{help}</span>
    </label>
  );
}

function SelectField({
  label,
  name,
  value,
  options,
  onChange,
}: {
  label: string;
  name: string;
  value: string | number;
  options: { value: string; label: string }[];
  onChange: (name: string, value: string) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select name={name} value={value} onChange={(event) => onChange(name, event.target.value)}>
        <option value="">Select one</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

export default function Home() {
  const [spec, setSpec] = useState<ModelSpec | null>(null);
  const [inputs, setInputs] = useState<Inputs>(EMPTY_INPUTS);
  const [risk, setRisk] = useState<number | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    fetch("./model/model_specification.json")
      .then((response) => {
        if (!response.ok) throw new Error("Model specification unavailable");
        return response.json();
      })
      .then((data: ModelSpec) => setSpec(data))
      .catch(() => setLoadError(true));
  }, []);

  const contributions = useMemo(
    () => (risk !== null && spec ? contributionProfile(inputs, spec) : []),
    [inputs, risk, spec],
  );
  const maxContribution = Math.max(0.01, ...contributions.map((item) => Math.abs(item.value)));

  function updateInput(name: string, value: string) {
    setInputs((current) => ({ ...current, [name]: value }));
    setRisk(null);
    setErrors([]);
    setWarnings([]);
  }

  function calculate(event: FormEvent) {
    event.preventDefault();
    if (!spec) return;
    const validation = validateInputs(inputs);
    setErrors(validation.errors);
    setWarnings(validation.warnings);
    if (validation.errors.length > 0) {
      setRisk(null);
      return;
    }
    setRisk(predictRisk(inputs, spec));
  }

  function loadExample() {
    setInputs(EXAMPLE_INPUTS);
    setErrors([]);
    setWarnings([]);
    if (spec) setRisk(predictRisk(EXAMPLE_INPUTS, spec));
  }

  function reset() {
    setInputs(EMPTY_INPUTS);
    setRisk(null);
    setErrors([]);
    setWarnings([]);
  }

  const riskPercent = risk === null ? 0 : risk * 100;
  const markerPosition = Math.min(100, Math.max(0, (riskPercent / 75) * 100));
  const stratum = risk === null ? null : developmentRiskStratum(risk);

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="PELD-RLDH calculator home">
          <span className="brand-mark">PR</span>
          <span>PELD-RLDH <strong>Risk Calculator</strong></span>
        </a>
        <nav aria-label="Page navigation">
          <a href="#calculator">Calculator</a>
          <a href="#model">Model</a>
          <a href="#methods">Methods</a>
          <a href="#limitations">Limitations</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow">Frozen multicenter prediction model · Version PELD_RLDH_V5_20260810</div>
        <h1>Estimate recurrence probability after PELD—transparently.</h1>
        <p className="hero-copy">
          A browser-based research calculator using 8 preoperative clinical and imaging predictors.
          The complete frozen equation runs locally on your device.
        </p>
        <div className="research-alert" role="note">
          <strong>Research-use prediction tool.</strong> Not intended to replace clinical judgment.
          Local calibration and prospective impact evaluation are required before routine clinical implementation.
        </div>
        <div className="privacy-line"><span className="privacy-dot" /> No patient information is transmitted or stored.</div>
      </section>

      <section className="calculator-section" id="calculator">
        <div className="section-heading">
          <span>01</span>
          <div>
            <p className="kicker">Frozen-model inference</p>
            <h2>Patient inputs and prediction</h2>
          </div>
        </div>

        <div className="calculator-grid">
          <form className="input-panel" onSubmit={calculate} noValidate>
            <div className="panel-heading">
              <div>
                <p className="kicker">Eight prespecified predictors</p>
                <h3>Enter preoperative values</h3>
              </div>
              <span className="step-pill">1 of 2</span>
            </div>

            <div className="form-grid">
              <SelectField label="Modic change" name="Modic_group" value={inputs.Modic_group} onChange={updateInput}
                options={[{ value: "无", label: "None" }, { value: "I", label: "Type I" }, { value: "II–III", label: "Type II–III" }]} />
              <SelectField label="Pfirrmann grade" name="Pfirrmann_group" value={inputs.Pfirrmann_group} onChange={updateInput}
                options={[{ value: "I–II", label: "I–II" }, { value: "III–IV", label: "III–IV" }, { value: "V", label: "V" }]} />
              <Field label="Sagittal range of motion" name="sROM/degrees" value={inputs["sROM/degrees"]} unit="degrees" help="Development range: 1–35" onChange={updateInput} />
              <Field label="Cross-sectional area" name="Cross_sectional_area/cm^2" value={inputs["Cross_sectional_area/cm^2"]} unit="cm²" help="Development range: 1.114–29.494" onChange={updateInput} />
              <Field label="Sacral slope" name="Sacral_slope/degrees" value={inputs["Sacral_slope/degrees"]} unit="degrees" help="Development range: 15–49.53" onChange={updateInput} />
              <Field label="Age" name="Age" value={inputs.Age} unit="years" help="Development range: 12–88.2" onChange={updateInput} />
              <Field label="Disc height index" name="Disc_height_index" value={inputs.Disc_height_index} help="Development range: 0.10–0.50" onChange={updateInput} />
              <SelectField label="Herniation type" name="Herniation_type" value={inputs.Herniation_type} onChange={updateInput}
                options={[{ value: "protrusion", label: "Protrusion" }, { value: "extrusion", label: "Extrusion" }, { value: "sequestration", label: "Sequestration" }]} />
            </div>

            {loadError && <div className="message error">The locked model file could not be loaded. Please reload the page.</div>}
            {errors.length > 0 && <div className="message error" role="alert"><strong>Check the inputs:</strong><ul>{errors.map((error) => <li key={error}>{error}</li>)}</ul></div>}
            {warnings.length > 0 && <div className="message warning"><strong>Outside development support:</strong><ul>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}

            <div className="form-actions">
              <button className="primary-button" type="submit" disabled={!spec}>Calculate predicted probability</button>
              <button className="text-button" type="button" onClick={loadExample}>Load example</button>
              <button className="text-button muted" type="button" onClick={reset}>Reset</button>
            </div>
          </form>

          <aside className={`result-panel ${risk === null ? "empty" : ""}`} aria-live="polite">
            <div className="panel-heading">
              <div>
                <p className="kicker">Patient-specific output</p>
                <h3>Frozen-model result</h3>
              </div>
              <span className="step-pill">2 of 2</span>
            </div>

            {risk === null ? (
              <div className="empty-state">
                <div className="empty-orbit"><span /></div>
                <h4>Ready when you are</h4>
                <p>Complete all 8 fields, then calculate the prediction. Use the example patient to test the exact frozen equation.</p>
              </div>
            ) : (
              <>
                <div className="risk-summary">
                  <p>Predicted RLDH probability</p>
                  <strong>{riskPercent.toFixed(1)}%</strong>
                  <span>Frozen original model · no local recalibration</span>
                </div>

                <div className="risk-scale" aria-label={`Predicted probability ${riskPercent.toFixed(1)} percent`}>
                  <div className="risk-gradient" />
                  <div className="risk-marker" style={{ left: `${markerPosition}%` }}><span>{riskPercent.toFixed(1)}%</span></div>
                  <div className="scale-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span></div>
                </div>

                <div className="stratum-card">
                  <span>Development-defined risk stratum</span>
                  <strong>{stratum}</strong>
                  <p>Defined from H1–H4 frozen predictions; thresholds were not optimized in H5 or H6.</p>
                </div>

                <div className="contribution-block">
                  <div className="contribution-heading">
                    <h4>Contributions to this model prediction</h4>
                    <span>Change in calibrated log-odds vs reference profile</span>
                  </div>
                  <div className="contribution-axis"><span>Lower prediction</span><span>Higher prediction</span></div>
                  {contributions.map((item) => {
                    const width = `${Math.max(3, (Math.abs(item.value) / maxContribution) * 46)}%`;
                    return (
                      <div className="contribution-row" key={item.feature}>
                        <span className="contribution-label">{LABELS[item.feature]}</span>
                        <div className="contribution-track">
                          <span className="centerline" />
                          <span className={`contribution-bar ${item.value >= 0 ? "positive" : "negative"}`} style={{ width }} />
                        </div>
                        <span className="contribution-value">{item.value >= 0 ? "+" : ""}{item.value.toFixed(2)}</span>
                      </div>
                    );
                  })}
                </div>

                <div className="interpretation-box">
                  <strong>Interpretation boundary</strong>
                  <p>This is a model prediction—not a causal effect or treatment recommendation. The fixed outcome horizon remains an author query and is therefore not asserted in this tool.</p>
                </div>
              </>
            )}
          </aside>
        </div>
      </section>

      <section className="evidence-section" id="model">
        <div className="section-heading light">
          <span>02</span>
          <div><p className="kicker">Evidence at a glance</p><h2>How the frozen model performed</h2></div>
        </div>
        <div className="performance-table" role="table" aria-label="Model performance by cohort">
          <div className="performance-row header" role="row"><span>Cohort</span><span>Patients</span><span>Events</span><span>AUROC</span><span>AUPRC</span></div>
          {PERFORMANCE.map((row) => <div className="performance-row" role="row" key={row.cohort}><strong>{row.cohort}</strong><span>{row.n}</span><span>{row.events}</span><span>{row.auroc}</span><span>{row.auprc}</span></div>)}
        </div>
        <p className="evidence-note">External calibration showed some absolute-risk underestimation. H6 contained 36 events, so its calibration estimates are less precise.</p>
      </section>

      <section className="details-grid" id="methods">
        <article>
          <p className="kicker">Why these predictors?</p>
          <h3>Transportability before leaderboard performance</h3>
          <p>Predictors were selected through training-domain stability analysis inside nested leave-one-hospital-out IECV. H5 and H6 were not used for feature selection, tuning, calibration, or model choice.</p>
          <ul className="predictor-list">{Object.values(LABELS).map((label) => <li key={label}>{label}</li>)}</ul>
        </article>
        <article>
          <p className="kicker">Transparent inference</p>
          <h3>The webpage is only a display layer</h3>
          <p>The calculator applies the exported spline knots, scaling parameters, ridge coefficients, categorical encoding, and Platt calibration exactly as frozen.</p>
          <div className="hash-card"><span>Model SHA-256</span><code>{spec?.model_sha256 ?? "Loading locked specification…"}</code></div>
        </article>
      </section>

      <section className="limitations" id="limitations">
        <div>
          <p className="kicker">Use boundaries</p>
          <h2>What this calculator does not establish</h2>
        </div>
        <ul>
          <li>No prospective clinical-impact evaluation has been performed.</li>
          <li>The model does not estimate treatment effects and its contributions are not causal.</li>
          <li>The operational RLDH definition, fixed horizon, follow-up completeness, and censoring rules require author confirmation.</li>
          <li>Predictions outside the observed development ranges require additional caution and local validation.</li>
        </ul>
      </section>

      <footer>
        <div><strong>PELD-RLDH Risk Calculator</strong><span>Research and reproducibility companion</span></div>
        <p>All calculations occur locally in the browser. No patient data are transmitted or retained.</p>
        <span>Model version: {spec?.version ?? "PELD_RLDH_V5_20260810"}</span>
      </footer>
    </main>
  );
}
