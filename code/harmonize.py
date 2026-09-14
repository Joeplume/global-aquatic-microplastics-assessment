import math
import re
import unicodedata
import numpy as np
import pandas as pd


COUNT_FACTORS = {
    "items/m3": 1.0,
    "item/m3": 1.0,
    "particles/m3": 1.0,
    "particle/m3": 1.0,
    "items/l": 1000.0,
    "item/l": 1000.0,
    "particles/l": 1000.0,
    "particle/l": 1000.0,
    "items/ml": 1000000.0,
    "particles/ml": 1000000.0,
    "items/dm3": 1000.0,
    "particles/dm3": 1000.0,
}

MASS_FACTORS = {
    "kg/m3": 1000000.0,
    "g/m3": 1000.0,
    "mg/m3": 1.0,
    "ug/m3": 0.001,
    "ng/m3": 0.000001,
    "g/l": 1000000.0,
    "mg/l": 1000.0,
    "ug/l": 1.0,
    "ng/l": 0.001,
}

POLYMER_DENSITY = {
    "pe": 0.94,
    "ldpe": 0.92,
    "hdpe": 0.96,
    "pp": 0.90,
    "ps": 1.05,
    "pet": 1.38,
    "pvc": 1.38,
    "pa": 1.14,
    "pmma": 1.18,
}


def canonical_unit(value):
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    for old, new in {"μ": "u", "µ": "u", "−": "-", "·": "", "litres": "l", "liters": "l", "litre": "l", "liter": "l", "per": "/", "pieces": "items", "piece": "item"}.items():
        text = text.replace(old, new)
    text = text.replace("m^-3", "/m3").replace("m-3", "/m3").replace("m^3", "m3")
    text = text.replace("ml^-1", "/ml").replace("ml-1", "/ml").replace("l^-1", "/l").replace("l-1", "/l")
    return re.sub(r"\s+", "", text)


def numbered_volume_factor(unit):
    if any(token in unit for token in ("/kg", "/g", "/m2", "/km2", "/sample", "/fish", "/organism", "%")):
        return np.nan
    if not re.search(r"item|particle|microplastic|piece|count|number|(^|[^a-z])mp([^a-z]|$)", unit):
        return np.nan
    match = re.search(r"/(\d+(?:\.\d+)?)(ml|l|dm3|m3)$", unit)
    if not match:
        return np.nan
    divisor = float(match.group(1))
    scale = {"ml": 1000000.0, "l": 1000.0, "dm3": 1000.0, "m3": 1.0}[match.group(2)]
    return scale / divisor


def unit_channel(unit):
    if unit in COUNT_FACTORS or np.isfinite(numbered_volume_factor(unit)):
        return "count_volume"
    if unit in MASS_FACTORS:
        return "mass_volume"
    if re.search(r"/(?:km2|m2|cm2)$", unit):
        return "area_density"
    if re.search(r"/(?:day|d|month|year|yr)$", unit):
        return "flux"
    if re.search(r"/(?:kg|g|sample|fish|organism)$", unit):
        return "non_volume"
    return "unresolved"


def particle_mass_from_traits(row):
    explicit = pd.to_numeric(row.get("mean_particle_mass_g"), errors="coerce")
    if np.isfinite(explicit) and explicit > 0:
        return float(explicit), "reported_particle_mass"
    diameter = pd.to_numeric(row.get("particle_size_mid_um"), errors="coerce")
    if not np.isfinite(diameter) or diameter <= 0:
        return np.nan, "unavailable"
    shape = str(row.get("dominant_shape", "fragment")).lower()
    polymer = str(row.get("dominant_polymer", "")).lower()
    density = next((value for name, value in POLYMER_DENSITY.items() if re.search(rf"(^|[^a-z]){name}([^a-z]|$)", polymer)), 1.05)
    if re.search(r"fiber|fibre|filament", shape):
        volume = math.pi * (diameter / 20.0) ** 2 * diameter
    elif re.search(r"film|sheet", shape):
        volume = diameter * diameter * diameter * 0.05
    elif re.search(r"bead|pellet|sphere", shape):
        volume = math.pi * diameter ** 3 / 6.0
    else:
        volume = 0.30 * diameter ** 3
    return density * volume * 1e-12, "trait_based_particle_mass"


def harmonize_units(data, value_col="original_concentration_value", unit_col="original_concentration_unit"):
    out = data.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    units = out[unit_col].map(canonical_unit)
    channels = units.map(unit_channel)
    count_factor = units.map(COUNT_FACTORS)
    numbered = units.map(numbered_volume_factor)
    count_factor = count_factor.where(count_factor.notna(), numbered)
    mass_factor = units.map(MASS_FACTORS)
    out["unit_canonical"] = units
    out["unit_channel"] = channels
    out["concentration_items_m3"] = values * count_factor
    out["concentration_mg_m3"] = values * mass_factor
    mass_to_count = []
    mass_basis = []
    for index, row in out.iterrows():
        if channels.at[index] == "mass_volume" and np.isfinite(out.at[index, "concentration_mg_m3"]):
            particle_mass, basis = particle_mass_from_traits(row)
            mass_to_count.append(out.at[index, "concentration_mg_m3"] * 1e-3 / particle_mass if np.isfinite(particle_mass) and particle_mass > 0 else np.nan)
            mass_basis.append(basis)
        else:
            mass_to_count.append(np.nan)
            mass_basis.append("not_applicable")
    mass_to_count = pd.Series(mass_to_count, index=out.index)
    fallback = pd.to_numeric(out["standardized_items_m3"], errors="coerce") if "standardized_items_m3" in out else pd.Series(np.nan, index=out.index)
    out["concentration_items_m3"] = out["concentration_items_m3"].where(out["concentration_items_m3"].notna(), mass_to_count)
    out["concentration_items_m3"] = out["concentration_items_m3"].where(out["concentration_items_m3"].notna(), fallback)
    out["mass_to_count_basis"] = mass_basis
    out["unit_conversion_source"] = np.select(
        [count_factor.notna(), mass_to_count.notna(), fallback.notna()],
        ["count_volume_rule", "mass_particle_model", "verified_standardized_value"],
        default="unresolved",
    )
    out["unit_harmonized"] = np.where(out["concentration_items_m3"].notna(), "items m-3", np.where(out["concentration_mg_m3"].notna(), "mg m-3", "unresolved"))
    return out
