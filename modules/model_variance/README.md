# Matched-study variance decomposition — SI3

```console
python reproduce_variance.py --output results
```

The calculation uses 975 matched records, 49 sources and 210 strata. Residuals are observed concentrations minus environmental OOF predictions. A crossed random-intercept REML model estimates geotime, analytical-method and study-source components.

`original_decomposition.py` supplies the fitting function; `original_qaqc_recompute.py` records its application. The measurement-variance contribution is capped at the remaining residual component, following the adopted calculation.

Outputs include residuals, variance estimates, convergence information, component shares and reference comparisons. Use the model environment with statsmodels.
