# Aquatic microplastic assessment — V22.7

Code and numerical inputs for concentration assessment, cross-study comparisons and monitoring-network analysis.

## Setup and run

Tested environment: Python 3.10 on Windows, with CPU PyTorch. Package versions are listed in `requirements.txt`.

```console
python -m pip install -r requirements.txt
python run_reproduce.py --stage results --output outputs
```

The public code package uses the numerical inputs supplied in the corresponding full reproduction bundle. Choose an empty output directory. Relative output paths are placed inside the package.

## Analysis entries

| Stage | Input and output |
|---|---|
| `results` | Saved predictions, environmental matching matrices and cost inputs → recalculated tables, 200 coverage-boundary replicates, REML point estimates, 24 baseline-cost scenarios, 16 basin-cost scenarios and 5,000-draw temporal simulations |
| `results-full` | Extended analysis with 2,000 REML bootstrap replicates per waterbody and all 216 basin-cost scenarios |
| `model-fold` | Prepared observations, process features and fixed folds → one 180-epoch training run |
| `model-all` | All domains and folds for the selected validation design |
| `verify` | Input-file integrity check |

```console
python run_reproduce.py --stage model-fold --domain freshwater --fold 1 --output trained_fold
python run_reproduce.py --stage model-all --design grouped10 --output trained_grouped
python run_reproduce.py --stage results-full --output extended_results
```

Model training and downstream table calculation have separate entries. For training with global-map export, see `modules/model/README.md`.

## Files

- `modules/`: model, global comparison, monitoring and validation calculations.
- `CLAIM_CODE_MAP.csv`: correspondence between SI tables and calculation entries.
- `DATA_NOTES.md`: data version, source-table availability and planning assumptions.
- `RUN_SUMMARY.json` in the output directory: executed jobs and their reports.
