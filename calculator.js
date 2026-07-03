const beta = {
  "(Intercept)": -9.8078,
  Age: 0.0308,
  Sex_male: 0.4391,
  BMI: 0.1758,
  Smoking_yes: 0.3179,
  Alcoholism_yes: -0.1279,
  Hypertension_yes: 0.1815,
  Diabetes_yes: 0.8381,
  Operation_time: 0.0039,
  Segment_L5S1: -0.1941,
  Herniation_extrusion: 0.9474,
  Herniation_sequestration: 0.4775,
  Modic_any: 1.3338,
  Pfirrmann_IVV: 1.3681,
  Disc_height_index: -0.0289,
  sROM: 0.1769,
  Cross_sectional_area: 0.1293,
  Lumbar_lordosis: -0.0646,
  Sacral_slope: -0.0825
};

const defaults = {
  Age: 45,
  Sex_male: 0,
  BMI: 23.0,
  Smoking_yes: 0,
  Alcoholism_yes: 0,
  Hypertension_yes: 0,
  Diabetes_yes: 0,
  Segment_L5S1: 0,
  Operation_time: 65,
  Herniation_type: 'protrusion',
  Modic_any: 0,
  Pfirrmann_IVV: 0,
  Disc_height_index: 0.32,
  sROM: 8.0,
  Cross_sectional_area: 8.0,
  Lumbar_lordosis: 35.0,
  Sacral_slope: 22.0
};

const numericFields = [
  'Age', 'BMI', 'Operation_time', 'Disc_height_index', 'sROM',
  'Cross_sectional_area', 'Lumbar_lordosis', 'Sacral_slope'
];

const binaryFields = [
  'Sex_male', 'Smoking_yes', 'Alcoholism_yes', 'Hypertension_yes',
  'Diabetes_yes', 'Segment_L5S1', 'Modic_any', 'Pfirrmann_IVV'
];

function valueOf(id) {
  const el = document.getElementById(id);
  if (!el) return 0;
  const value = parseFloat(el.value);
  return Number.isFinite(value) ? value : 0;
}

function getInputs() {
  const herniation = document.getElementById('Herniation_type').value;
  return {
    Age: valueOf('Age'),
    Sex_male: valueOf('Sex_male'),
    BMI: valueOf('BMI'),
    Smoking_yes: valueOf('Smoking_yes'),
    Alcoholism_yes: valueOf('Alcoholism_yes'),
    Hypertension_yes: valueOf('Hypertension_yes'),
    Diabetes_yes: valueOf('Diabetes_yes'),
    Segment_L5S1: valueOf('Segment_L5S1'),
    Operation_time: valueOf('Operation_time'),
    Herniation_extrusion: herniation === 'extrusion' ? 1 : 0,
    Herniation_sequestration: herniation === 'sequestration' ? 1 : 0,
    Modic_any: valueOf('Modic_any'),
    Pfirrmann_IVV: valueOf('Pfirrmann_IVV'),
    Disc_height_index: valueOf('Disc_height_index'),
    sROM: valueOf('sROM'),
    Cross_sectional_area: valueOf('Cross_sectional_area'),
    Lumbar_lordosis: valueOf('Lumbar_lordosis'),
    Sacral_slope: valueOf('Sacral_slope')
  };
}

function calculateRisk() {
  const inputs = getInputs();
  let lp = beta['(Intercept)'];
  Object.keys(inputs).forEach((key) => {
    lp += (beta[key] || 0) * inputs[key];
  });
  const risk = 1 / (1 + Math.exp(-lp));
  return { risk, lp, inputs };
}

function riskCategory(risk) {
  if (risk < 0.10) {
    return {
      key: 'low',
      label: 'Low risk',
      text: 'The predicted risk is low. Routine postoperative follow-up may be appropriate, together with standard rehabilitation and symptom monitoring.'
    };
  }
  if (risk <= 0.30) {
    return {
      key: 'intermediate',
      label: 'Intermediate risk',
      text: 'The predicted risk is intermediate. Closer symptom monitoring, rehabilitation guidance, activity modification, and lifestyle counseling may be considered.'
    };
  }
  return {
    key: 'high',
    label: 'High risk',
    text: 'The predicted risk is high. More detailed counseling and follow-up planning may be appropriate. If high risk coexists with segmental instability, severe degeneration, or mechanical symptoms, alternative surgical strategies may be discussed as part of shared decision-making.'
  };
}

function renderResult() {
  const { risk } = calculateRisk();
  const category = riskCategory(risk);
  const percent = Math.max(0, Math.min(100, risk * 100));
  const riskText = percent.toFixed(1);
  const riskPercent = document.getElementById('risk-percent');
  const riskCategoryEl = document.getElementById('risk-category');
  const marker = document.getElementById('risk-marker');
  const interpretation = document.getElementById('interpretation-text');

  riskPercent.textContent = riskText;
  riskCategoryEl.textContent = category.label;
  riskCategoryEl.className = `category-pill ${category.key}`;
  marker.style.left = `${Math.min(100, Math.max(0, percent))}%`;
  interpretation.textContent = category.text;
}

function resetForm() {
  Object.entries(defaults).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.value = value;
  });
  renderResult();
}

function labelValue(id) {
  const el = document.getElementById(id);
  if (!el) return '';
  if (el.tagName === 'SELECT') {
    return el.options[el.selectedIndex].text;
  }
  return el.value;
}

function downloadReport() {
  const { risk, lp } = calculateRisk();
  const category = riskCategory(risk);
  const lines = [
    'PELD RLDH 2-Year Risk Calculator Report',
    `Date/time: ${new Date().toLocaleString()}`,
    '',
    `Predicted 2-year RLDH risk: ${(risk * 100).toFixed(1)}%`,
    `Risk category: ${category.label}`,
    `Linear predictor: ${lp.toFixed(4)}`,
    '',
    'Input variables:',
    `Age, years: ${labelValue('Age')}`,
    `Sex: ${labelValue('Sex_male')}`,
    `BMI, kg/m2: ${labelValue('BMI')}`,
    `Smoking: ${labelValue('Smoking_yes')}`,
    `Alcoholism: ${labelValue('Alcoholism_yes')}`,
    `Hypertension: ${labelValue('Hypertension_yes')}`,
    `Diabetes: ${labelValue('Diabetes_yes')}`,
    `Operative segment: ${labelValue('Segment_L5S1')}`,
    `Operation time, min: ${labelValue('Operation_time')}`,
    `Herniation type: ${labelValue('Herniation_type')}`,
    `Modic change: ${labelValue('Modic_any')}`,
    `Pfirrmann grade: ${labelValue('Pfirrmann_IVV')}`,
    `Disc height index: ${labelValue('Disc_height_index')}`,
    `sROM, degrees: ${labelValue('sROM')}`,
    `Cross-sectional area: ${labelValue('Cross_sectional_area')}`,
    `Lumbar lordosis, degrees: ${labelValue('Lumbar_lordosis')}`,
    `Sacral slope, degrees: ${labelValue('Sacral_slope')}`,
    '',
    'Suggested interpretation:',
    category.text,
    '',
    'Disclaimer:',
    'This research-use output is intended to support risk communication and postoperative follow-up stratification. It should not be used as a stand-alone indication for revision surgery, fusion surgery, or treatment escalation. Interpret together with symptoms, neurological findings, imaging results, segmental stability, surgeon judgment, and patient preference.'
  ];
  const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `PELD_RLDH_risk_report_${new Date().toISOString().slice(0, 10)}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function init() {
  document.getElementById('calculate-btn').addEventListener('click', renderResult);
  document.getElementById('reset-btn').addEventListener('click', resetForm);
  document.getElementById('download-btn').addEventListener('click', downloadReport);
  [...numericFields, ...binaryFields, 'Herniation_type'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', renderResult);
    if (el) el.addEventListener('change', renderResult);
  });
  renderResult();
}

document.addEventListener('DOMContentLoaded', init);
