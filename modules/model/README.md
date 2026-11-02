# H26 model

The model combines three waterbody-specific environmental networks with study-source offsets. Inputs include prepared observations, environmental-process features, QA/QC scores and fixed validation folds.

## Recalculate saved validation results

```console
python reaggregate_oof.py --output results/metrics
python reaggregate_support_validation.py --output results/support
```

The second command scores SI22 with its stored fold-local retained-set weights.

## Train

```console
python run_model.py --design grouped10 --domain freshwater --fold 1 --output results/fold1
python run_model.py --design grouped10 --domain all --fold all --with-maps --output results/maps
python aggregate_maps.py --training-output results/maps --output results/GRID10_qaqc.csv.gz
```

Available designs are `grouped10`, `random10`, `spatial`, `source`, `joint` and `grouped_comparison`. Random and grouped designs use ten folds; transfer designs use five. Fold numbering starts at 1. Training defaults to 180 epochs and 24 posterior draws. `--check-only` validates inputs and writes the run plan. `--epochs` selects a diagnostic training length.

Map export is available for `grouped10`. The aggregator takes the median environmental prediction across folds and combines within-fold and between-fold latent variance. Future-observation SD is stored separately. Newly trained maps have the original `GRID10_qaqc.csv.gz` schema; downstream calculations need the corresponding new map and derived inputs.

## Predictions and environment

Environmental predictions describe the concentration surface. Conditional predictions add an estimated source offset for represented sources. Operational predictions use that offset where available and otherwise use the environmental prediction. Metrics are reported separately for these targets and their evaluation cohorts.

Tested with Python 3.10 and CPU PyTorch 2.5.1; versions are pinned in the package requirements. At least 2 GiB available RAM is required; 8 GiB is preferable for map export. Outputs use the selected directory.
