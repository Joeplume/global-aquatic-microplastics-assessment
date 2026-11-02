# Validation by support stratum — SI15

```console
python reproduce_support_strata.py --output results
```

The entry joins 24,129 QA/QC record-random OOF predictions with the adopted record-level C-AOA labels and calculates the 12 SI15 rows.

Intervals use the saved training calibration: conditional observation SD for records with source references and environmental observation SD for unknown sources. Outputs include RMSE, WIS and interval coverage.

SI15 describes performance by full-data support stratum. The separate SI22 entry evaluates fold-local support.
