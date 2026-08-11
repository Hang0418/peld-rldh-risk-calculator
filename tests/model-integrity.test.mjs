import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { predictRisk, validateInputs } from "../lib/calculator.mjs";

const root = new URL("../", import.meta.url);
const specification = JSON.parse(await readFile(new URL("model/model_specification.json", root), "utf8"));
const references = JSON.parse(await readFile(new URL("tests/reference_predictions.json", root), "utf8"));
const lockedSha = (await readFile(new URL("model/model_sha256.txt", root), "utf8")).trim();

test("browser equation matches the independent Python equation", () => {
  let maximumAbsoluteDifference = 0;
  for (const reference of references.cases) {
    const { case_id: caseId, python_reference_probability: expected, ...inputs } = reference;
    const observed = predictRisk(inputs, specification);
    const absoluteDifference = Math.abs(observed - expected);
    maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, absoluteDifference);
    assert.ok(absoluteDifference < 1e-10, `${caseId}: |JS - Python| = ${absoluteDifference}`);
  }
  assert.equal(references.cases.length, 72);
  console.log(`maximum |JS - Python|: ${maximumAbsoluteDifference}`);
});

test("published specification identifies the locked model", () => {
  assert.equal(specification.model_sha256, lockedSha);
  assert.equal(references.provenance.model_sha256, lockedSha);
  assert.equal(references.provenance.case_type, "deterministic synthetic combinations; no patient records");
});

test("example patient is reproducible", () => {
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
  assert.ok(Math.abs(predictRisk(example, specification) - 0.2939836687324339) < 1e-12);
  assert.deepEqual(validateInputs(example), { errors: [], warnings: [] });
});
