# Predictor definitions

The site uses the exact raw names and categorical encodings stored in the frozen specification. Measurement protocols should be aligned with the final manuscript data dictionary before clinical interpretation.

| Display label | Model field | Accepted values | Development range |
|---|---|---|---|
| Modic change | `Modic_group` | None (`无`), I, II–III | categorical |
| Sagittal range of motion | `sROM/degrees` | degrees | 1–35 |
| Cross-sectional area | `Cross_sectional_area/cm^2` | cm² | 1.114–29.494 |
| Pfirrmann grade | `Pfirrmann_group` | I–II, III–IV, V | categorical |
| Sacral slope | `Sacral_slope/degrees` | degrees | 15–49.53 |
| Age | `Age` | years | 12–88.2 |
| Disc height index | `Disc_height_index` | ratio | 0.10–0.50 |
| Herniation type | `Herniation_type` | protrusion, extrusion, sequestration | categorical |

The browser blocks physiologically implausible values and warns, rather than blocks, values outside the observed development support.
