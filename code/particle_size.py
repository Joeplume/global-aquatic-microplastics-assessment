import numpy as np
import pandas as pd


def interval_mass(lower, upper, alpha):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    exponent = 1.0 - float(alpha)
    return np.log(upper / lower) if abs(exponent) < 1e-12 else (upper ** exponent - lower ** exponent) / exponent


def shared_alpha_correction(data, alpha, reference_lower_um, reference_upper_um):
    out = data.copy()
    lower = pd.to_numeric(out["particle_size_lower_um"], errors="coerce")
    upper = pd.to_numeric(out.get("analysis_particle_size_upper_um", reference_upper_um), errors="coerce").fillna(reference_upper_um)
    valid = lower.gt(0) & upper.gt(lower) & np.isfinite(lower) & np.isfinite(upper)
    observed = pd.Series(np.nan, index=out.index, dtype=float)
    observed.loc[valid] = interval_mass(lower.loc[valid], upper.loc[valid], float(alpha))
    reference = float(interval_mass(reference_lower_um, reference_upper_um, float(alpha)))
    factor = reference / observed
    out["shared_alpha"] = float(alpha)
    out["particle_size_interval_valid"] = valid
    out["particle_size_factor"] = factor.where(valid)
    out["concentration_size_corrected_items_m3"] = pd.to_numeric(out["concentration_items_m3"], errors="coerce") * out["particle_size_factor"]
    return out
