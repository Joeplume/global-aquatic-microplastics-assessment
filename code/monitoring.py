import numpy as np
import pandas as pd


def minimum_detectable_decline(sigma_log10, sites, occasions, rho=0.3):
    sites = np.maximum(np.asarray(sites, dtype=float), 1.0)
    occasions = max(float(occasions), 2.0)
    effective = sites * occasions / (1.0 + (occasions - 1.0) * rho)
    delta = 2.8 * np.asarray(sigma_log10, dtype=float) / np.sqrt(effective)
    return 100.0 * (1.0 - np.power(10.0, -delta))


def monitoring_analysis(predictions, settings):
    width = predictions["q975_log10_items_m3"] - predictions["q025_log10_items_m3"]
    sigma = width / 3.92
    rows = []
    for n in settings.get("site_budgets", [25, 50, 100, 250, 500]):
        values = minimum_detectable_decline(sigma, n, settings.get("occasions", 6), settings.get("rho", 0.3))
        rows.append({"fixed_sites": int(n), "median_mdc_pct": float(np.nanmedian(values)), "q10_mdc_pct": float(np.nanquantile(values, 0.1)), "q90_mdc_pct": float(np.nanquantile(values, 0.9))})
    return pd.DataFrame(rows)
