# Monitoring and cost calculations

```console
python reproduce_monitoring.py --output output/monitoring --boundary-reps 200
python reproduce_basin_monitoring.py --stage all --output output/basins
python reproduce_temporal.py --output output/temporal
```

## Inputs and calculations

The network entry uses prepared source/query matrices, the stored 400-site sequence and the high-concentration mask. It refits robust scaling, PCA retaining 90% of variance and the 95th-percentile nearest-source matching radius, then recalculates coverage. Boundary resampling keeps site order and the concentration mask fixed.

The REML entry uses 24,129 saved OOF residuals and measurement-variance inputs. Station requirements refer to six observations over five years. Basin maps and temporal simulations use the separate cross-validation-calibrated variance components; SI18 reports the REML analysis.

Cost calculations use 2026 EUR scenario prices, service-group shared fees, institutional proximity and accessibility scenarios. The 95/305-site analysis covers 24 combinations. Basin analysis reports support/detection states, fixed-revisit packages, local variance and continued monitoring costs.

## Run options

- `--boundary-reps 200` calculates the coverage interval; the module default is three replicates.
- `--reml-reps 2000` adds the source-bootstrap station-count intervals; the default fits point estimates.
- `--full-cost-grid` expands the basin calculation from 16 paired mid-price scenarios to all 216 scenarios.
- Temporal simulation uses 5,000 realizations per scenario.

The original assembly and optimization scripts are in `src/original`. The default entries recalculate the stored layout. Reference tables provide numerical checks.
