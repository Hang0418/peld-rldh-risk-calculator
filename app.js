import {
  contributionProfile,
  developmentRiskStratum,
  predictRisk,
  validateInputs,
} from "./lib/calculator.mjs";

const form = document.querySelector("#risk-form");
const messages = document.querySelector("#messages");
const probability = document.querySelector("#probability");
const stratum = document.querySelector("#stratum");
const marker = document.querySelector("#marker");
const contributionSection = document.querySelector("#contributions");
const contributionBars = document.querySelector("#contribution-bars");
const modelVersion = document.querySelector("#model-version");

const labels = {
  Modic_group: "Modic change",
  "sROM/degrees": "Sagittal range of motion",
  "Cross_sectional_area/cm^2": "Cross-sectional area",
  Pfirrmann_group: "Pfirrmann grade",
  "Sacral_slope/degrees": "Sacral slope",
  Age: "Age",
  Disc_height_index: "Disc height index",
  Herniation_type: "Herniation type",
};

const example = {
  Modic_group: "II–III",
  "sROM/degrees": 10,
  "Cross_sectional_area/cm^2": 9.5,
  Pfirrmann_group: "III–IV",
  "Sacral_slope/degrees": 28,
  Age: 55,
  Disc_height_index: 0.3,
  Herniation_type: "extrusion",
};

let specification = null;

function getInputs() {
  return Object.fromEntries(
    [...new FormData(form).entries()].map(([key, value]) => {
      const numeric = ["sROM/degrees", "Cross_sectional_area/cm^2", "Sacral_slope/degrees", "Age", "Disc_height_index"];
      return [key, numeric.includes(key) && value !== "" ? Number(value) : value];
    }),
  );
}

function renderMessages(validation) {
  messages.replaceChildren();
  const groups = [
    ["error", "Check the inputs", validation.errors],
    ["warning", "Outside development support", validation.warnings],
  ];
  groups.forEach(([kind, title, items]) => {
    if (!items.length) return;
    const box = document.createElement("div");
    box.className = `message ${kind}`;
    const heading = document.createElement("strong");
    heading.textContent = title;
    const list = document.createElement("ul");
    items.forEach((item) => {
      const entry = document.createElement("li");
      entry.textContent = item;
      list.append(entry);
    });
    box.append(heading, list);
    messages.append(box);
  });
}

function renderContributions(inputs) {
  const contributions = contributionProfile(inputs, specification).slice(0, 5);
  const maximum = Math.max(...contributions.map(({ value }) => Math.abs(value)), 0.01);
  contributionBars.replaceChildren();
  contributions.forEach(({ feature, value }) => {
    const row = document.createElement("div");
    row.className = "contribution-row";
    const name = document.createElement("span");
    name.textContent = labels[feature] || feature;
    const track = document.createElement("div");
    track.className = "bar-track";
    const bar = document.createElement("i");
    bar.className = value >= 0 ? "positive" : "negative";
    bar.style.width = `${Math.max(4, Math.abs(value) / maximum * 100)}%`;
    const amount = document.createElement("code");
    amount.textContent = `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
    track.append(bar);
    row.append(name, track, amount);
    contributionBars.append(row);
  });
  contributionSection.hidden = false;
}

function calculate() {
  if (!specification) return;
  const inputs = getInputs();
  const validation = validateInputs(inputs);
  renderMessages(validation);
  if (validation.errors.length) return;
  const risk = predictRisk(inputs, specification);
  probability.textContent = `${(risk * 100).toFixed(1)}%`;
  stratum.textContent = `Development-cohort stratum: ${developmentRiskStratum(risk)}`;
  marker.style.left = `${Math.min(100, risk / 0.75 * 100)}%`;
  renderContributions(inputs);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  calculate();
});

form.addEventListener("reset", () => {
  setTimeout(() => {
    messages.replaceChildren();
    probability.textContent = "—";
    stratum.textContent = "Complete all eight fields.";
    marker.style.left = "0%";
    contributionSection.hidden = true;
  });
});

document.querySelector("#example").addEventListener("click", () => {
  Object.entries(example).forEach(([name, value]) => {
    form.elements.namedItem(name).value = value;
  });
  calculate();
});

fetch("./model/model_specification.json")
  .then((response) => {
    if (!response.ok) throw new Error("Frozen model specification is unavailable.");
    return response.json();
  })
  .then((data) => {
    specification = data;
    modelVersion.textContent = data.version;
  })
  .catch((error) => {
    messages.textContent = error.message;
    messages.className = "message error";
  });
