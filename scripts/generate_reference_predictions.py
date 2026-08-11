#!/usr/bin/env python3
"""Generate deterministic synthetic reference predictions with the frozen Python equation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT_WORKSPACE = ROOT.parent
REFERENCE_IMPLEMENTATION = PARENT_WORKSPACE / "scripts" / "predict_from_published_equation.py"
SPECIFICATION = ROOT / "model" / "model_specification.json"
OUTPUT = ROOT / "tests" / "reference_predictions.json"


def load_reference_module():
    module_spec = importlib.util.spec_from_file_location("frozen_equation", REFERENCE_IMPLEMENTATION)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load {REFERENCE_IMPLEMENTATION}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_synthetic_cases() -> list[dict[str, object]]:
    modic = ["I", "II–III", "无"]
    pfirrmann = ["III–IV", "I–II", "V"]
    herniation = ["extrusion", "protrusion", "sequestration"]
    numeric_profiles = [
        (1.0, 1.114, 15.0, 12.0, 0.10),
        (2.0, 5.4509, 16.953, 37.0, 0.209),
        (5.0, 8.595, 22.54465, 50.4, 0.28915),
        (6.0, 9.7545, 26.1735, 54.35, 0.315),
        (8.0, 11.33495, 30.28665, 57.1, 0.339),
        (13.0, 19.1024, 36.1881, 68.0, 0.41),
        (35.0, 29.494, 49.53, 88.2, 0.50),
        (10.0, 9.5, 28.0, 55.0, 0.30),
    ]
    cases = []
    case_id = 1
    for profile_index, numeric in enumerate(numeric_profiles):
        srom, area, slope, age, dhi = numeric
        for category_index in range(9):
            cases.append(
                {
                    "case_id": f"S{case_id:03d}",
                    "Modic_group": modic[(profile_index + category_index) % 3],
                    "sROM/degrees": srom,
                    "Cross_sectional_area/cm^2": area,
                    "Pfirrmann_group": pfirrmann[(profile_index + category_index // 3) % 3],
                    "Sacral_slope/degrees": slope,
                    "Age": age,
                    "Disc_height_index": dhi,
                    "Herniation_type": herniation[(profile_index + category_index) % 3],
                }
            )
            case_id += 1
    return cases


def main() -> None:
    reference = load_reference_module()
    specification = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
    cases = build_synthetic_cases()
    frame = pd.DataFrame(cases)
    predictions = reference.predict_from_equation(frame, specification)
    payload = {
        "provenance": {
            "generator": str(REFERENCE_IMPLEMENTATION.relative_to(PARENT_WORKSPACE)),
            "model_version": specification["version"],
            "model_sha256": specification["model_sha256"],
            "case_type": "deterministic synthetic combinations; no patient records",
        },
        "cases": [
            {**case, "python_reference_probability": float(probability)}
            for case, probability in zip(cases, predictions, strict=True)
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} synthetic references to {OUTPUT}")


if __name__ == "__main__":
    main()
