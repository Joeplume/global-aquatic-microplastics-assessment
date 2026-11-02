# Model comparisons and paper removal — SI21 and SI24

```console
python reproduce_supplement.py --output results
```

The entry aggregates saved held-out predictions. SI21 includes 20 model rows from record-random baselines, QA/QC H26 predictions, grouped neural comparisons and the stored grouped RF results. Each comparator retains its original training weights.

SI24 uses 180 removal-unit outputs, 60 paper-removal lists and paired full-cohort predictions. WIS is calculated for records from removed papers and paired by domain, round and fold. Geometric reference retention is documented separately in `DATA_NOTES.md` at the package root.

## Small-sample training settings

Three 80%-removal units used a 30-record inner early-stopping minimum: Estuary round 1 folds 9/10 and marine round 1 fold 7. Their inner partitions contain 33/40/37 records. Outputs include the combined results and a sensitivity analysis excluding these units.

Original model code and recovery procedures are in `supplementary_sources`; fitting configurations are in `supplementary_inputs`. The adjacent model module supplies the shared process tables and folds. Run the portable entry above for the packaged inputs.

Fitting parameters for the 30 baseline folds and 180 removal units are consolidated in `data/analysis_parameters.json`, including the three small-sample settings.
