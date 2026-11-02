# Global comparisons

```console
python run.py --output output
python reaggregate_support.py --output support_output
```

Inputs are the saved QA/QC-weighted prediction grid, country and support assignments, record locations and calibrated latent draws. Outputs cover global concentrations, country rankings and gaps, paired regional ratios, night-light representativeness, mass conversion, database counts and reservoir-case metrics.

## Definitions

- Global means and Brazil/paired ratios use WGS84 grid-footprint weights. Country rankings and the developing-region share use the stored spherical weights. Weights describe prediction-grid footprints.
- Rankings use the 70-country roster. Night-light analysis uses all eligible countries and available years. The regional ratio uses its separate 23-country, 2020 paired design.
- Country attribution follows the UN193 crosswalk; development groups use the stored Natural Earth ECONOMY classification.
- High-concentration areas use the waterbody-specific area-weighted upper quartile. The >1,000 items/m³ comparison is reported separately.
- Mass calculations use 200 latent-model draws and 500 trait draws with the stored pairing. The within-domain conversion factor preserves ranks within that domain.

SI2 uses saved temporal-model estimates; SI23 uses saved reservoir predictions. Country ranks use 800 resamples and paired regional ratios use 500. Reference tables provide checks after aggregation.
