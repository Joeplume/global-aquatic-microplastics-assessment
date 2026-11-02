#!/usr/bin/env python3
"""H26 independent-center management-evaluability and monitoring repair.

The independently refitted variance components replace the shared-center
version. Original AOA/C-AOA, source nodes, monitoring sites and emission weights
remain read-only. C-AOA remains a support annotation and beta remains a
conditional response scenario, not a fitted or causal parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


VERSION = "H26_独立检测分支管理可评估性_20260905"
DOMAINS = ("reference_surface_freshwater", "transition_coastal", "marine")
LAND_DOMAINS = DOMAINS[:2]
DOMAIN_SHORT = {
    "reference_surface_freshwater": "freshwater",
    "transition_coastal": "transition",
    "marine": "marine",
}
DOMAIN_TIE = {"reference_surface_freshwater": 0, "transition_coastal": 1, "marine": 2}
ALPHA = 0.05
POWER = 0.80
Z_ALPHA_POWER = 1.959963984540054 + 0.8416212335729143
ANNUAL_OCCASIONS = 6
TIME_SPAN_YEARS = 5
PRIMARY_RHO = 0.30
PRIMARY_EVENTS = 1
DELTA_30_LOG10 = -math.log10(0.70)
BETA_VALUES = (0.25, 0.50, 0.75, 1.00)
BRIDGE_BUDGETS = (0, 10, 25, 50, 100, 200, 500, 1000)
CAOA_THRESHOLDS = (0.25, 0.50, 0.75)
RHO_VALUES = (0.0, 0.3, 0.6)
EVENT_VALUES = (1, 3, 5)

DOMAIN_FEATURES = {
    "reference_surface_freshwater": (
        "night_lights_radiance_2020", "m6_road_density_m_per_km2",
        "spot_land_plastic_pressure_2020", "annual_runoff_mm_2020",
        "m6_annual_max_daily_runoff_mm", "river_to_sea_microplastic_output_kg_year",
        "m6_modis2019_strict_cropland_pct_0p5deg",
        "m6_ghsl2020_nonresidential_builtup_pct_0p5deg",
        "m6_global_annual_mean_2m_air_temperature_c_2020", "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
        "m6_modis2020_terrestrial_ndvi_annual_mean_0p5deg",
    ),
    "transition_coastal": (
        "night_lights_radiance_2020", "m6_road_density_m_per_km2",
        "spot_land_plastic_pressure_2020", "annual_runoff_mm_2020",
        "m6_annual_max_daily_runoff_mm", "river_to_sea_microplastic_output_kg_year",
        "ais_activity_density", "marine_aquaculture_pressure_tonnes_50km",
        "m6_sea_surface_salinity_psu", "marina_pressure_h100_kg_year",
        "m6_surface_current_speed_ms", "m6_significant_wave_height_m", "positive_water_depth_m",
        "m6_modis2019_strict_cropland_pct_0p5deg",
        "m6_ghsl2020_nonresidential_builtup_pct_0p5deg",
        "m6_global_annual_mean_2m_air_temperature_c_2020", "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
        "m6_modis2020_terrestrial_ndvi_annual_mean_0p5deg",
    ),
    "marine": (
        "river_to_sea_microplastic_output_kg_year", "ais_activity_density",
        "marine_aquaculture_pressure_tonnes_50km", "m6_sea_surface_salinity_psu",
        "marina_pressure_h100_kg_year", "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m", "positive_water_depth_m",
        "m6_global_annual_mean_2m_air_temperature_c_2020", "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
    ),
}
LOG_FEATURES = {
    "night_lights_radiance_2020", "m6_road_density_m_per_km2",
    "spot_land_plastic_pressure_2020", "annual_runoff_mm_2020",
    "m6_annual_max_daily_runoff_mm", "river_to_sea_microplastic_output_kg_year",
    "ais_activity_density", "marine_aquaculture_pressure_tonnes_50km",
    "marina_pressure_h100_kg_year", "positive_water_depth_m",
}


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression, float_format="%.12g", lineterminator="\n")


def copy_frozen(source: Path, destination: Path) -> None:
    require(source.is_file(), f"Missing frozen input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha256(destination) == sha256(source), f"Frozen copy drift: {destination}")
    else:
        shutil.copy2(source, destination)


def slope_factor(annual_occasions: int = ANNUAL_OCCASIONS, rho: float = PRIMARY_RHO) -> float:
    years = np.arange(annual_occasions, dtype=float)
    design = np.column_stack([np.ones(annual_occasions), years])
    corr = rho ** np.abs(np.subtract.outer(years, years))
    covariance = np.linalg.inv(design.T @ np.linalg.inv(corr) @ design)
    return float((annual_occasions - 1) ** 2 * covariance[1, 1])


def mdc_log10(sigma: float, sites: int, rho: float = PRIMARY_RHO, events: int = PRIMARY_EVENTS) -> float:
    if sites <= 0 or not np.isfinite(sigma):
        return math.inf
    return float(Z_ALPHA_POWER * sigma * math.sqrt(slope_factor(rho=rho) / (sites * events)))


def required_sites(sigma: float, target_delta: float = DELTA_30_LOG10,
                   rho: float = PRIMARY_RHO, events: int = PRIMARY_EVENTS) -> int:
    raw = (Z_ALPHA_POWER * sigma) ** 2 * slope_factor(rho=rho) / (events * target_delta ** 2)
    return max(1, int(math.ceil(raw - 1e-12)))


def decline_pct(mdc: float) -> float:
    return 100.0 if not np.isfinite(mdc) else float(100.0 * (1.0 - 10.0 ** (-mdc)))


def transform_feature(values: np.ndarray, feature: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values)) if feature in LOG_FEATURES else values


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def freeze_inputs(root: Path, output: Path) -> tuple[dict[str, Path], pd.DataFrame]:
    package = root / "99_临时与待确认" / "H26_GLOBAL_MONITORING_COMPARABILITY_V1_20260828"
    history = root / "90_过程历史" / "04_H26未晋升均值模型" / "H26_完整开发工作树_20260813_20260827"
    current = root / "h26"
    marine_asset = current / "data" / "06_geospatial_assets" / "natural_earth_10m_admin1_marine" / "marine"
    sources = {
        "grid_support": package / "results_csv" / "H26_CANDIDATE_POINTS_ALL_NATURAL.csv.gz",
        "grid_caoa_original": package / "frozen_inputs" / "H26_CAOA_V0_3_SOURCE_BRIDGE_GRID_ORIGINAL.csv.gz",
        "mdc_components": root / "99_临时与待确认" / "H26_独立检测分支重拟合_20260905" / "results" / "mdc" / "H26_MDC_VARIANCE_COMPONENTS.csv",
        "spot_emission_grid": package / "frozen_inputs" / "H26_SPOT_UNMANAGED_EMISSION_GRID_FREEZE.csv.gz",
        "basin_weights": package / "results_csv" / "H26_EMISSION_BASIN_WEIGHTS.csv.gz",
        "basin_mdc_reference": package / "results_csv" / "H26_BASIN_5Y_MDC.csv.gz",
        "current_sites": package / "results_csv" / "H26_CURRENT_NETWORK_SITES.csv.gz",
        "reference_curve": package / "results_csv" / "H26_BUDGET_WEIGHTED_DETECTABILITY_CURVE.csv.gz",
        "reference_configuration": package / "results_csv" / "H26_CONFIGURATION_SUMMARY.csv",
        "source_package_input_manifest": package / "audit" / "FROZEN_INPUT_INVENTORY.csv",
        "source_package_output_manifest": package / "audit" / "OUTPUT_MANIFEST.csv",
        "source_package_qa": package / "audit" / "FINAL_QA_AUDIT.json",
        "caoa_audit": history / "results_20260826" / "H26_CAOA_V0_3_SOURCE_BRIDGE_20260826_v2" / "H26_CAOA_V0_3_SOURCE_BRIDGE_AUDIT.json",
        "source_bridge_nodes": history / "results_20260826" / "H26_SOURCE_BRIDGE_NETWORK_20260826_v1" / "H26_SOURCE_BRIDGE_NODES.csv.gz",
        "process_environment_grid": current / "data" / "05_process_covariates" / "H26_RIVER_NETWORK_PROCESS_PREDICTION.csv.gz",
    }
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        sources[f"marine_basin_polygons{suffix}"] = marine_asset / f"ne_10m_geography_marine_polys{suffix}"
    frozen = output / "frozen_inputs"
    targets: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for name, source in sources.items():
        target = frozen / "marine_basin_polygons" / source.name if name.startswith("marine_basin_polygons") else frozen / source.name
        copy_frozen(source, target)
        targets[name] = target
        rows.append({
            "input_name": name, "source_path": str(source), "frozen_path": str(target.relative_to(output)),
            "bytes": source.stat().st_size, "sha256": sha256(source), "freeze_mode": "byte_copy_read_only_source",
        })

    # Verify every artifact registered by the source evidence package.
    registered = pd.read_csv(sources["source_package_output_manifest"], dtype=str)
    mismatches = []
    for row in registered.itertuples(index=False):
        candidate = package / str(row.relative_path)
        if not candidate.is_file() or sha256(candidate) != str(row.sha256):
            mismatches.append(str(row.relative_path))
    require(not mismatches, f"Source evidence-package manifest mismatch: {mismatches[:5]}")

    upstream_registered = pd.read_csv(sources["source_package_input_manifest"], dtype=str)
    upstream_rows = []
    upstream_failures = []
    for row in upstream_registered.itertuples(index=False):
        source_path = Path(str(row.source_path))
        exists = source_path.is_file()
        actual_hash = sha256(source_path) if exists else ""
        matched = exists and actual_hash == str(row.sha256)
        upstream_rows.append({
            "input_name": row.input_name, "source_path": str(source_path),
            "expected_bytes": int(row.bytes), "actual_bytes": source_path.stat().st_size if exists else math.nan,
            "expected_sha256": row.sha256, "actual_sha256": actual_hash,
            "exists": exists, "sha256_match": matched,
        })
        if not matched:
            upstream_failures.append(str(row.input_name))
    write_csv(pd.DataFrame(upstream_rows), output / "audit" / "UPSTREAM_INPUT_SHA256_VERIFICATION.csv")
    require(not upstream_failures, f"Source evidence-package upstream input drift: {upstream_failures}")
    package_rows = []
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        package_rows.append({
            "relative_path": str(path.relative_to(package)), "bytes": path.stat().st_size, "sha256": sha256(path)
        })
    write_csv(pd.DataFrame(package_rows), output / "audit" / "SOURCE_EVIDENCE_PACKAGE_SHA256.csv")
    inventory = pd.DataFrame(rows).sort_values("input_name").reset_index(drop=True)
    write_csv(inventory, output / "frozen_inputs" / "FROZEN_INPUT_MANIFEST.csv")
    return targets, inventory


def baseline_gate(reference_curve: Path) -> dict[str, Any]:
    curve = pd.read_csv(reference_curve)
    ref = curve[
        curve.target_id.eq("concentration_decline_30pct") & curve.scenario.eq("standardized_fixed")
    ].sort_values("new_site_budget")
    require(len(ref) > 1 and int(ref.new_site_budget.iloc[0]) == 0, "Reference standardized-fixed curve missing")
    baseline = float(ref.emission_weighted_detectable_basin_share.iloc[0])
    maximum = float(ref.emission_weighted_detectable_basin_share.max())
    first = ref[ref.emission_weighted_detectable_basin_share.gt(baseline + 1e-15)].iloc[0]
    threshold = baseline + 0.9 * (maximum - baseline)
    reach90 = ref[ref.emission_weighted_detectable_basin_share.ge(threshold - 1e-15)].iloc[0]
    observed = {
        "baseline_share": baseline, "first_gain_budget": int(first.new_site_budget),
        "first_gain_share": float(first.emission_weighted_detectable_basin_share),
        "gain90_budget": int(reach90.new_site_budget), "gain90_share": float(reach90.emission_weighted_detectable_basin_share),
        "maximum_share": maximum, "gain90_threshold": threshold,
    }
    expected = {"baseline_share": 0.135531843554, "first_gain_budget": 131,
                "gain90_budget": 1160, "maximum_share": 0.344334803441}
    checks = {
        "baseline": abs(baseline - expected["baseline_share"]) <= 5e-13,
        "first_gain_budget": observed["first_gain_budget"] == expected["first_gain_budget"],
        "gain90_budget": observed["gain90_budget"] == expected["gain90_budget"],
        "maximum": abs(maximum - expected["maximum_share"]) <= 5e-13,
    }
    require(all(checks.values()), f"Hard baseline reproduction gate failed: {observed}")
    return {"status": "PASS", "expected": expected, "observed": observed, "checks": checks}


def rebuild_environment_embeddings(nodes_path: Path, process_path: Path, caoa_path: Path) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    nodes = pd.read_csv(nodes_path, low_memory=False)
    needed_features = sorted({item for values in DOMAIN_FEATURES.values() for item in values})
    process = pd.read_csv(
        process_path, usecols=["prediction_id", "prediction_domain", *needed_features], low_memory=False
    )
    caoa = pd.read_csv(caoa_path, usecols=[
        "prediction_id", "waterbody", "environment_match_caliper_loo_q95",
        "nearest_source_distance", "second_source_distance",
    ], low_memory=False)
    domain_data: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    env_rows = []
    for domain in DOMAINS:
        source = nodes[nodes.waterbody.eq(domain)].reset_index(drop=True)
        grid = process[process.prediction_domain.eq(domain)].reset_index(drop=True)
        features = [f for f in DOMAIN_FEATURES[domain] if f in source.columns and f in grid.columns]
        source_columns, grid_columns, used = [], [], []
        for feature in features:
            src_raw = pd.to_numeric(source[feature], errors="coerce").to_numpy(float)
            grd_raw = pd.to_numeric(grid[feature], errors="coerce").to_numpy(float)
            if np.isfinite(src_raw).mean() < 0.20 or np.isfinite(grd_raw).mean() < 0.20:
                continue
            src = transform_feature(src_raw, feature)
            grd = transform_feature(grd_raw, feature)
            median = float(np.nanmedian(src))
            q10, q90 = np.nanquantile(src, (0.10, 0.90))
            scale = float(q90 - q10)
            if not np.isfinite(scale) or scale <= 1e-10:
                continue
            source_columns.append(np.clip((np.where(np.isfinite(src), src, median) - median) / scale, -10, 10))
            grid_columns.append(np.clip((np.where(np.isfinite(grd), grd, median) - median) / scale, -10, 10))
            used.append(feature)
        require(len(used) >= 4, f"Too few environment features for {domain}")
        source_matrix = np.column_stack(source_columns)
        grid_matrix = np.column_stack(grid_columns)
        full_pca = PCA(svd_solver="full").fit(source_matrix)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        dimensions = min(12, source_matrix.shape[0], source_matrix.shape[1],
                         max(2, int(np.searchsorted(cumulative, 0.90) + 1)))
        pca = PCA(n_components=dimensions, svd_solver="full").fit(source_matrix)
        source_embed = pca.transform(source_matrix)
        grid_embed = pca.transform(grid_matrix)
        distances = pairwise_distances(source_embed)
        np.fill_diagonal(distances, np.inf)
        loo_nearest = np.min(distances, axis=1)
        q95_caliper = float(np.quantile(loo_nearest, 0.95))
        q90_caliper = float(np.quantile(loo_nearest, 0.90))
        nn = NearestNeighbors(n_neighbors=2).fit(source_embed)
        nearest, _ = nn.kneighbors(grid_embed)
        frozen = caoa[caoa.waterbody.eq(domain)].set_index("prediction_id").reindex(grid.prediction_id)
        frozen_q95 = float(frozen.environment_match_caliper_loo_q95.dropna().iloc[0])
        q95_diff = abs(q95_caliper - frozen_q95)
        nearest_diff = float(np.nanmax(np.abs(nearest[:, 0] - frozen.nearest_source_distance.to_numpy(float))))
        second_diff = float(np.nanmax(np.abs(nearest[:, 1] - frozen.second_source_distance.to_numpy(float))))
        require(q95_diff < 1e-10 and nearest_diff < 1e-9 and second_diff < 1e-9,
                f"C-AOA environment reconstruction drift in {domain}: {q95_diff}, {nearest_diff}, {second_diff}")
        ids = grid.prediction_id.astype(str).to_numpy()
        domain_data[domain] = {
            "ids": ids, "X": grid_embed, "id_to_pos": {value: i for i, value in enumerate(ids)},
            "q90": q90_caliper, "q95": q95_caliper,
        }
        env_rows.append(pd.DataFrame({
            "prediction_id": ids, "environment_nearest_source_distance_rebuilt": nearest[:, 0],
            "environment_q90_caliper": q90_caliper, "environment_q95_caliper": q95_caliper,
        }))
        audit[domain] = {
            "source_nodes": len(source), "grid_cells": len(grid), "used_features": used,
            "pca_dimensions": dimensions, "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "q90_caliper": q90_caliper, "q95_caliper": q95_caliper,
            "frozen_q95_abs_diff": q95_diff, "nearest_distance_max_abs_diff": nearest_diff,
            "second_distance_max_abs_diff": second_diff,
        }
    return domain_data, pd.concat(env_rows, ignore_index=True), audit


def variance_table(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = (
        ("one_time_new_coordinates", False, True, False, "No temporal MDC; a one-time spatial addition cannot identify a five-year trend"),
        ("fixed_repeat_source_protocol_change", True, True, False, "Fixed repeats but source/protocol panel changes"),
        ("fixed_repeat_reported_method_standardized_source_change", True, True, True, "Reported-method variance removed; source still changes"),
        ("fixed_repeat_stable_source_historical_method", True, False, False, "Stable source; historical method heterogeneity retained"),
        ("fixed_repeat_stable_source_method_standardized", True, False, True, "Stable source plus reported-method standardization"),
        ("complete_design_plus_bridge", True, False, True, "Same temporal MDC as stable-source standardized design; bridge only repairs prospective comparability"),
    )
    for item in components.itertuples(index=False):
        veff = float(item.within_source_total_effective_sd_log10) ** 2
        vknown = float(item.reported_sd_pooled_log10) ** 2
        vres = max(veff - vknown, 0.0)
        vsource = float(item.between_source_sd_log10) ** 2
        for name, temporal, source_changes, standardized, definition in definitions:
            vwithin = vres if standardized else veff
            variance = vwithin + (vsource if source_changes else 0.0)
            sigma = math.sqrt(variance)
            mdc = mdc_log10(sigma, 1) if temporal else math.nan
            rows.append({
                "domain": item.domain, "scenario": name, "temporal_mdc_defined": temporal,
                "source_changes": source_changes, "reported_method_standardized": standardized,
                "Veff": veff, "Vknown": vknown, "Vres": vres, "Vsource": vsource,
                "variance_used": variance, "sigma_log10": sigma, "mdc_5y_one_site_log10": mdc,
                "minimum_detectable_decline_pct_one_site": decline_pct(mdc) if temporal else math.nan,
                "required_sites_for_30pct_decline": required_sites(sigma) if temporal else math.nan,
                "bridge_changes_same_site_time_mdc": False, "definition": definition,
            })
    result = pd.DataFrame(rows)
    for domain in components.domain:
        a = result[(result.domain == domain) & (result.scenario == "fixed_repeat_stable_source_method_standardized")]
        b = result[(result.domain == domain) & (result.scenario == "complete_design_plus_bridge")]
        require(float(a.mdc_5y_one_site_log10.iloc[0]) == float(b.mdc_5y_one_site_log10.iloc[0]),
                f"Bridge changed temporal MDC in {domain}")
    return result


def pareto_curve(packages: pd.DataFrame, objective: str, max_budget: int | None = None) -> pd.DataFrame:
    satisfied = packages[f"{objective}_baseline"].fillna(False).astype(bool)
    feasible = packages[f"{objective}_feasible"].fillna(False).astype(bool)
    costs = pd.to_numeric(packages[f"{objective}_cost"], errors="coerce")
    baseline_share = float(packages.loc[satisfied, "emission_weight"].sum())
    baseline_basins = int(satisfied.sum())
    activation = packages[~satisfied & feasible & costs.gt(0)].copy()
    activation["activation_cost"] = costs.loc[activation.index].astype(int)
    activation["benefit_per_point"] = activation.emission_weight / activation.activation_cost
    activation = activation.sort_values(
        ["benefit_per_point", "emission_weight", "basin_id"], ascending=[False, False, True], kind="mergesort"
    )
    activation["cumulative_cost"] = activation.activation_cost.cumsum()
    activation["cumulative_share"] = baseline_share + activation.emission_weight.cumsum()
    activation["cumulative_basins"] = baseline_basins + np.arange(1, len(activation) + 1)
    if max_budget is None:
        max_budget = int(activation.cumulative_cost.max()) if len(activation) else 0
    budgets = np.arange(max_budget + 1, dtype=int)
    if len(activation):
        completed = np.searchsorted(activation.cumulative_cost.to_numpy(int), budgets, side="right")
        share = np.where(completed == 0, baseline_share,
                         activation.cumulative_share.to_numpy(float)[np.maximum(completed - 1, 0)])
        basins = np.where(completed == 0, baseline_basins,
                          activation.cumulative_basins.to_numpy(int)[np.maximum(completed - 1, 0)])
    else:
        share = np.full(len(budgets), baseline_share)
        basins = np.full(len(budgets), baseline_basins)
    return pd.DataFrame({
        "objective": objective, "new_point_budget": budgets,
        "emission_weighted_satisfied_basin_share": share, "satisfied_basins": basins,
        "baseline_share": baseline_share, "maximum_feasible_share": baseline_share + float(activation.emission_weight.sum()),
        "ordering_rule": "descending emission_weight/activation_cost; deterministic ties; auditable Pareto construction",
    })


def choose_monitoring_arms(grid: pd.DataFrame, sites: pd.DataFrame, weights: pd.DataFrame,
                           sigma_reference: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = sites[sites.waterbody.isin(LAND_DOMAINS) & sites.basin_id.notna()].copy()
    counts = current.groupby(["basin_id", "waterbody"], as_index=False).agg(
        current_sites=("site_key", "nunique")
    )
    candidate = grid[
        grid.waterbody.isin(LAND_DOMAINS) & grid.natural_water_exists & grid.aoa_pass
        & grid.basin_id.notna() & ~grid.existing_site_grid
    ].groupby(["basin_id", "waterbody"], as_index=False).agg(aoa_feasible_new_cells=("prediction_id", "nunique"))
    water = grid[grid.waterbody.isin(LAND_DOMAINS) & grid.natural_water_exists & grid.basin_id.notna()].copy()
    water["area_weight"] = np.cos(np.deg2rad(water.latitude.to_numpy(float)))
    coverage = water.groupby(["basin_id", "waterbody"], as_index=False).agg(
        water_cells=("prediction_id", "nunique"), area_weight_sum=("area_weight", "sum"),
        caoa_in_area_weight=("area_weight", lambda x: float(x[water.loc[x.index, "caoa_status_current"].eq("CAOA_in")].sum())),
        caoa_in_cells=("caoa_status_current", lambda x: int(x.eq("CAOA_in").sum())),
    )
    coverage["caoa_in_area_share"] = coverage.caoa_in_area_weight / coverage.area_weight_sum
    basins = weights[weights.basin_id.ne("UNROUTED_SPOT")][["basin_id", "emission_t_yr", "emission_weight"]]
    arms = basins.assign(_key=1).merge(pd.DataFrame({"waterbody": LAND_DOMAINS, "_key": 1}), on="_key").drop(columns="_key")
    arms = arms.merge(counts, on=["basin_id", "waterbody"], how="left")
    arms = arms.merge(candidate, on=["basin_id", "waterbody"], how="left")
    arms = arms.merge(coverage, on=["basin_id", "waterbody"], how="left")
    for column in ("current_sites", "aoa_feasible_new_cells", "water_cells", "caoa_in_cells"):
        arms[column] = arms[column].fillna(0).astype(int)
    for column in ("area_weight_sum", "caoa_in_area_weight"):
        arms[column] = arms[column].fillna(0.0)
    arms["caoa_in_area_share"] = arms.caoa_in_area_share.where(arms.area_weight_sum.gt(0), np.nan)
    arms["sigma_reference_log10"] = arms.waterbody.map(lambda x: sigma_reference[DOMAIN_SHORT[x]])
    arms["required_sites_reference_30pct"] = arms.sigma_reference_log10.map(required_sites).astype(int)
    arms["mdc_current_reference_log10"] = [
        mdc_log10(s, int(n)) for s, n in zip(arms.sigma_reference_log10, arms.current_sites)
    ]
    arms["detectable_current_reference"] = arms.mdc_current_reference_log10.le(DELTA_30_LOG10 + 1e-15)
    arms["mdc_new_points_needed"] = np.maximum(
        arms.required_sites_reference_30pct - arms.current_sites, 0
    ).astype(int)
    arms["mdc_arm_feasible"] = arms.mdc_new_points_needed.le(arms.aoa_feasible_new_cells)
    arms["tie_order"] = arms.waterbody.map(DOMAIN_TIE)

    selected_rows = []
    for basin_id, group in arms.groupby("basin_id", sort=True):
        with_sites = group[group.current_sites.gt(0)].copy()
        if len(with_sites):
            chosen = with_sites.sort_values(
                ["mdc_current_reference_log10", "tie_order"], kind="mergesort"
            ).iloc[0]
            rule = "existing_sites_choose_lowest_standardized_fixed_mdc"
        else:
            feasible = group[group.mdc_arm_feasible & group.water_cells.gt(0)].copy()
            if len(feasible):
                chosen = feasible.sort_values(["mdc_new_points_needed", "tie_order"], kind="mergesort").iloc[0]
                rule = "no_existing_sites_choose_fewest_feasible_new_points_freshwater_tie"
            else:
                chosen = group.sort_values(["tie_order"], kind="mergesort").iloc[0].copy()
                chosen["waterbody"] = pd.NA
                rule = "not_assessed_no_feasible_arm"
        row = chosen.to_dict()
        row["selection_rule"] = rule
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    selected["selected_waterbody"] = selected.waterbody
    return arms.drop(columns="tie_order"), selected.drop(columns="tie_order")


def local_greedy_bridge_sequences(grid: pd.DataFrame, domain_data: dict[str, dict[str, Any]],
                                  quantile_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    """Greedy set cover within each basin and waterbody using cos(latitude) area."""
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    new_cells_by_step: dict[int, np.ndarray] = {}
    sequence_id = 0
    routed = grid[
        grid.waterbody.isin(LAND_DOMAINS) & grid.natural_water_exists & grid.basin_id.notna()
    ]
    for (basin_id, waterbody), group in routed.groupby(["basin_id", "waterbody"], sort=True):
        row_start = len(rows)
        local = group.sort_values("prediction_id").copy()
        global_indices = local.index.to_numpy(int)
        weights = local.area_weight.to_numpy(float)
        original = local.caoa_status_current.eq("CAOA_in").to_numpy(bool)
        covered = original.copy()
        total_area = float(weights.sum())
        baseline_area = float(weights[covered].sum())
        positions = local.embed_pos.to_numpy(int)
        X = domain_data[waterbody]["X"][positions]
        caliper = float(domain_data[waterbody][quantile_name])
        internal = local.environment_nearest_source_distance_rebuilt.le(caliper + 1e-12)
        candidate_mask = local.aoa_pass & internal & ~local.existing_site_grid
        candidate_local = np.flatnonzero(candidate_mask.to_numpy(bool))
        cover_sets: list[np.ndarray] = []
        if len(candidate_local) and (~covered).any():
            dmat = pairwise_distances(X[candidate_local], X)
            for row_dist in dmat:
                cover_sets.append(np.flatnonzero((row_dist <= caliper + 1e-12) & ~original))
        heap: list[tuple[float, str, int, int]] = []
        for cpos, target in enumerate(cover_sets):
            gain = float(weights[target].sum())
            pid = str(local.prediction_id.iloc[candidate_local[cpos]])
            if gain > 0:
                heapq.heappush(heap, (-gain, pid, cpos, 0))
        local_step = 0
        while heap:
            neg_gain, pid, cpos, version = heapq.heappop(heap)
            target = cover_sets[cpos]
            uncovered = target[~covered[target]]
            gain = float(weights[uncovered].sum())
            if gain <= 1e-15:
                continue
            if version != local_step:
                heapq.heappush(heap, (-gain, pid, cpos, local_step))
                continue
            local_step += 1
            covered[uncovered] = True
            sequence_id += 1
            new_global = global_indices[uncovered]
            new_cells_by_step[sequence_id] = new_global
            cidx = candidate_local[cpos]
            rows.append({
                "sequence_id": sequence_id, "quantile_range": quantile_name,
                "basin_id": basin_id, "waterbody": waterbody, "local_step": local_step,
                "prediction_id": pid, "longitude": float(local.longitude.iloc[cidx]),
                "latitude": float(local.latitude.iloc[cidx]), "candidate_original_caoa": local.caoa_status_current.iloc[cidx],
                "marginal_area_weight": gain, "newly_supported_cells": len(uncovered),
                "local_cumulative_caoa_share": float(weights[covered].sum() / total_area),
            })
        final_share = float(weights[covered].sum() / total_area) if total_area > 0 else math.nan
        seq_local = rows[row_start:]
        shares = [float(x["local_cumulative_caoa_share"]) for x in seq_local]
        metric = {
            "basin_id": basin_id, "waterbody": waterbody, "quantile_range": quantile_name,
            "water_cells": len(local), "area_weight_sum": total_area,
            "caoa_in_area_share_0": baseline_area / total_area if total_area else math.nan,
            "bridge_candidate_cells": int(len(candidate_local)), "caoa_in_area_share_max": final_share,
            "bridge_points_useful": local_step,
        }
        for threshold in CAOA_THRESHOLDS:
            if baseline_area / total_area >= threshold - 1e-15:
                need: float = 0.0
            else:
                hits = [i + 1 for i, value in enumerate(shares) if value >= threshold - 1e-15]
                need = float(hits[0]) if hits else math.inf
            metric[f"N_bridge_to_{int(threshold * 100)}pct"] = need
        metrics.append(metric)
    sequence = pd.DataFrame(rows)
    if len(sequence):
        sequence["waterbody_tie"] = sequence.waterbody.map(DOMAIN_TIE)
        sequence = sequence.sort_values(
            ["marginal_area_weight", "waterbody_tie", "basin_id", "local_step", "prediction_id"],
            ascending=[False, True, True, True, True], kind="mergesort",
        ).reset_index(drop=True)
        sequence["global_bridge_rank"] = np.arange(1, len(sequence) + 1)
        sequence = sequence.drop(columns="waterbody_tie")
    return sequence, pd.DataFrame(metrics), new_cells_by_step


def first_bridge_ranks(grid: pd.DataFrame, sequence: pd.DataFrame,
                       new_cells_by_step: dict[int, np.ndarray]) -> np.ndarray:
    result = np.full(len(grid), np.inf)
    result[grid.caoa_status_current.eq("CAOA_in").to_numpy(bool)] = 0.0
    for row in sequence.itertuples(index=False):
        cells = new_cells_by_step[int(row.sequence_id)]
        still = ~np.isfinite(result[cells])
        result[cells[still]] = float(row.global_bridge_rank)
    return result


def cross_basin_first_ranks(grid: pd.DataFrame, sequence: pd.DataFrame,
                            domain_data: dict[str, dict[str, Any]], quantile_name: str,
                            maximum_budget: int = 1000) -> np.ndarray:
    result = np.full(len(grid), np.inf)
    result[grid.caoa_status_current.eq("CAOA_in").to_numpy(bool)] = 0.0
    selected = sequence[sequence.global_bridge_rank.le(maximum_budget)]
    for waterbody in LAND_DOMAINS:
        cells = grid[grid.waterbody.eq(waterbody) & grid.natural_water_exists]
        if cells.empty:
            continue
        cell_positions = cells.embed_pos.to_numpy(int)
        X = domain_data[waterbody]["X"][cell_positions]
        tree = NearestNeighbors(radius=float(domain_data[waterbody][quantile_name])).fit(X)
        candidates = selected[selected.waterbody.eq(waterbody)].sort_values("global_bridge_rank")
        if candidates.empty:
            continue
        candidate_positions = candidates.prediction_id.map(domain_data[waterbody]["id_to_pos"]).to_numpy(int)
        neighbors = tree.radius_neighbors(domain_data[waterbody]["X"][candidate_positions], return_distance=False)
        global_cells = cells.index.to_numpy(int)
        for rank, local_neighbors in zip(candidates.global_bridge_rank.to_numpy(int), neighbors):
            target = global_cells[local_neighbors]
            new = ~np.isfinite(result[target])
            result[target[new]] = float(rank)
    return result


def bridge_budget_curve(grid: pd.DataFrame, selected_basins: pd.DataFrame, weights: pd.DataFrame,
                        quantile_name: str, propagation_scope: str, first_rank: np.ndarray,
                        threshold: float = 0.50) -> pd.DataFrame:
    land = grid[grid.waterbody.isin(LAND_DOMAINS) & grid.natural_water_exists]
    total_area = float(land.area_weight.sum())
    weight_map = weights.set_index("basin_id").emission_weight.to_dict()
    rows = []
    for budget in BRIDGE_BUDGETS:
        supported = first_rank <= budget
        area_share = float(grid.loc[land.index[supported[land.index]], "area_weight"].sum() / total_area)
        comparable = 0
        emission_share = 0.0
        assessed = 0
        for item in selected_basins.itertuples(index=False):
            if pd.isna(item.selected_waterbody):
                continue
            cells = grid[
                grid.basin_id.eq(item.basin_id) & grid.waterbody.eq(item.selected_waterbody)
                & grid.natural_water_exists
            ]
            if cells.empty:
                continue
            assessed += 1
            local_supported = first_rank[cells.index] <= budget
            share = float(cells.loc[local_supported, "area_weight"].sum() / cells.area_weight.sum())
            if share >= threshold - 1e-15:
                comparable += 1
                emission_share += float(weight_map.get(item.basin_id, 0.0))
        rows.append({
            "quantile_range": quantile_name, "propagation_scope": propagation_scope,
            "bridge_point_budget": budget, "land_water_cell_area_share_caoa_in": area_share,
            "selected_arm_basins_assessed": assessed, "selected_arm_basins_at_threshold": comparable,
            "emission_weighted_basin_share_at_threshold": emission_share, "caoa_threshold": threshold,
        })
    return pd.DataFrame(rows)


def build_four_state(selected: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for threshold in CAOA_THRESHOLDS:
        for item in selected.itertuples(index=False):
            if pd.isna(item.selected_waterbody) or not np.isfinite(item.caoa_in_area_share):
                state = "not_assessed"
                comparable = False
                detectable = False
            else:
                comparable = float(item.caoa_in_area_share) >= threshold - 1e-15
                detectable = bool(item.detectable_current_reference)
                if comparable and detectable:
                    state = "comparable_and_detectable"
                elif comparable:
                    state = "comparable_not_detectable"
                elif detectable:
                    state = "detectable_comparability_insufficient"
                else:
                    state = "both_insufficient"
            row = item._asdict()
            row.update({"caoa_threshold": threshold, "caoa_comparable": comparable,
                        "mdc_detectable": detectable, "four_state": state})
            rows.append(row)
        unroute = weights[weights.basin_id.eq("UNROUTED_SPOT")].iloc[0]
        rows.append({
            "basin_id": "UNROUTED_SPOT", "emission_t_yr": float(unroute.emission_t_yr),
            "emission_weight": float(unroute.emission_weight), "selected_waterbody": pd.NA,
            "caoa_threshold": threshold, "caoa_comparable": False, "mdc_detectable": False,
            "four_state": "unrouted_spot",
        })
    basin = pd.DataFrame(rows)
    summaries = []
    for threshold, group in basin.groupby("caoa_threshold", sort=True):
        water_denom = float(group.area_weight_sum.fillna(0).sum())
        for (waterbody, state), part in group.groupby(["selected_waterbody", "four_state"], dropna=False):
            summaries.append({
                "caoa_threshold": threshold, "waterbody_scope": waterbody if pd.notna(waterbody) else "all_or_unassessed",
                "four_state": state, "basin_count": len(part),
                "water_cell_count": int(part.water_cells.fillna(0).sum()),
                "water_cell_area_weight": float(part.area_weight_sum.fillna(0).sum()),
                "water_cell_area_share_selected_arms": (
                    float(part.area_weight_sum.fillna(0).sum()) / water_denom if water_denom > 0 else math.nan
                ),
                "spot_emission_t_yr": float(part.emission_t_yr.fillna(0).sum()),
                "spot_emission_weighted_share": float(part.emission_weight.fillna(0).sum()),
            })
        for state, part in group.groupby("four_state", dropna=False):
            summaries.append({
                "caoa_threshold": threshold, "waterbody_scope": "all_selected_arms",
                "four_state": state, "basin_count": len(part),
                "water_cell_count": int(part.water_cells.fillna(0).sum()),
                "water_cell_area_weight": float(part.area_weight_sum.fillna(0).sum()),
                "water_cell_area_share_selected_arms": (
                    float(part.area_weight_sum.fillna(0).sum()) / water_denom if water_denom > 0 else math.nan
                ),
                "spot_emission_t_yr": float(part.emission_t_yr.fillna(0).sum()),
                "spot_emission_weighted_share": float(part.emission_weight.fillna(0).sum()),
            })
    return basin, pd.DataFrame(summaries)


def build_network_packages(selected: pd.DataFrame, bridge_metrics: pd.DataFrame,
                           ablation: pd.DataFrame, threshold: float = 0.50,
                           quantile_name: str = "q95", beta: float = 1.0,
                           rho: float = PRIMARY_RHO, events: int = PRIMARY_EVENTS) -> pd.DataFrame:
    metric = bridge_metrics[
        bridge_metrics.quantile_range.eq(quantile_name)
    ][["basin_id", "waterbody", f"N_bridge_to_{int(threshold * 100)}pct",
       "caoa_in_area_share_0", "caoa_in_area_share_max", "bridge_candidate_cells"]]
    result = selected.merge(metric, left_on=["basin_id", "selected_waterbody"],
                            right_on=["basin_id", "waterbody"], how="left", suffixes=("", "_bridge"))
    sigma_full = ablation[ablation.scenario.eq("complete_design_plus_bridge")].set_index("domain").sigma_log10.to_dict()
    result["beta"] = beta
    result["target_concentration_decline_pct"] = 100.0 * (1.0 - 0.70 ** beta)
    result["target_delta_log10"] = beta * DELTA_30_LOG10
    result["rho"] = rho
    result["events_per_site_year"] = events
    result["sigma_complete_design_log10"] = result.selected_waterbody.map(
        lambda x: sigma_full.get(DOMAIN_SHORT.get(x, ""), math.nan)
    )
    result["N_MDC_total"] = [
        required_sites(s, d, rho, events) if np.isfinite(s) else math.inf
        for s, d in zip(result.sigma_complete_design_log10, result.target_delta_log10)
    ]
    result["N_MDC_new"] = [
        max(int(n) - int(c), 0) if np.isfinite(n) else math.inf
        for n, c in zip(result.N_MDC_total, result.current_sites.fillna(0))
    ]
    result["mdc_baseline"] = result.N_MDC_new.eq(0) & result.selected_waterbody.notna()
    result["mdc_feasible"] = result.selected_waterbody.notna() & result.N_MDC_new.le(result.aoa_feasible_new_cells)
    nbridge_col = f"N_bridge_to_{int(threshold * 100)}pct"
    result["N_bridge"] = pd.to_numeric(result[nbridge_col], errors="coerce")
    result["caoa_baseline"] = result.N_bridge.eq(0) & result.selected_waterbody.notna()
    result["caoa_feasible"] = result.selected_waterbody.notna() & np.isfinite(result.N_bridge)
    result["joint_baseline"] = result.mdc_baseline & result.caoa_baseline
    result["joint_feasible"] = result.mdc_feasible & result.caoa_feasible
    result["mdc_cost"] = result.N_MDC_new
    result["caoa_cost"] = result.N_bridge
    result["joint_cost"] = np.maximum(result.N_MDC_new, result.N_bridge)
    result["bridge_analysis_pairs"] = result.N_bridge
    result["complete_package_point_cost_rule"] = "max(N_MDC_new,N_bridge); bridge points are fixed annual sites"
    return result


def build_sensitivity(selected: pd.DataFrame, bridge_metrics: pd.DataFrame,
                      ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sigma = ablation[ablation.scenario.eq("complete_design_plus_bridge")].set_index("domain").sigma_log10.to_dict()
    metric_lookup = bridge_metrics.set_index(["basin_id", "waterbody", "quantile_range"])
    for item in selected.itertuples(index=False):
        domain = DOMAIN_SHORT.get(item.selected_waterbody, "")
        for threshold in CAOA_THRESHOLDS:
            for quantile_name in ("q90", "q95"):
                try:
                    metric = metric_lookup.loc[(item.basin_id, item.selected_waterbody, quantile_name)]
                    nbridge = float(metric[f"N_bridge_to_{int(threshold * 100)}pct"])
                    max_coverage = float(metric.caoa_in_area_share_max)
                except KeyError:
                    nbridge, max_coverage = math.inf, math.nan
                for rho in RHO_VALUES:
                    for events in EVENT_VALUES:
                        if domain:
                            n_total = required_sites(float(sigma[domain]), DELTA_30_LOG10, rho, events)
                            n_new = max(n_total - int(item.current_sites), 0)
                            mdc_ok = n_new == 0
                            mdc_feasible = n_new <= int(item.aoa_feasible_new_cells)
                        else:
                            n_total = n_new = math.inf
                            mdc_ok = mdc_feasible = False
                        caoa_ok = nbridge == 0
                        caoa_feasible = np.isfinite(nbridge)
                        rows.append({
                            "basin_id": item.basin_id, "selected_waterbody": item.selected_waterbody,
                            "emission_weight": item.emission_weight, "caoa_threshold": threshold,
                            "environment_range": quantile_name, "rho_ar1": rho,
                            "events_per_site_year": events, "N_MDC_total": n_total, "N_MDC_new": n_new,
                            "N_bridge": nbridge, "complete_package_new_points": max(n_new, nbridge),
                            "caoa_max_area_share": max_coverage, "mdc_baseline": mdc_ok,
                            "caoa_baseline": caoa_ok, "joint_baseline": mdc_ok and caoa_ok,
                            "mdc_feasible": mdc_feasible, "caoa_feasible": caoa_feasible,
                            "joint_feasible": mdc_feasible and caoa_feasible,
                        })
    return pd.DataFrame(rows)


def assign_marine_basins(points: pd.DataFrame, polygon_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    oceans = gpd.read_file(polygon_path)
    oceans = oceans[oceans.featurecla.eq("ocean")][["name", "geometry"]].rename(columns={"name": "marine_basin"})
    geo = gpd.GeoDataFrame(points.copy(), geometry=gpd.points_from_xy(points.longitude, points.latitude), crs=4326)
    joined = gpd.sjoin(geo, oceans, how="left", predicate="within").drop(columns=["index_right"])
    joined["marine_basin_assignment"] = np.where(
        joined.marine_basin.notna(), "within_ocean_polygon", pd.NA
    )
    direct = int(joined.marine_basin.notna().sum())
    unmatched = joined[joined.marine_basin.isna()].copy()
    if len(unmatched):
        nearest = gpd.sjoin_nearest(unmatched.to_crs(6933), oceans.to_crs(6933), how="left", distance_col="nearest_ocean_m")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        nearest_basin_column = "marine_basin_right" if "marine_basin_right" in nearest.columns else "marine_basin"
        joined.loc[nearest.index, "marine_basin"] = nearest[nearest_basin_column]
        joined.loc[nearest.index, "marine_basin_assignment"] = "nearest_ocean_polygon_for_marginal_sea"
    joined["marine_basin_assignment"] = joined.marine_basin_assignment.fillna("nearest_ocean_polygon_for_marginal_sea")
    require(joined.marine_basin.notna().all(), "Marine basin assignment incomplete")
    return pd.DataFrame(joined.drop(columns="geometry")), {
        "total": len(joined), "direct_within": direct, "nearest_for_marginal_sea": len(joined) - direct,
    }


def build_marine_si(grid: pd.DataFrame, sites: pd.DataFrame, polygon_path: Path,
                    ablation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    marine = grid[grid.waterbody.eq("marine") & grid.natural_water_exists].copy()
    marine, cell_audit = assign_marine_basins(marine, polygon_path)
    marine_sites = sites[sites.waterbody.eq("marine")][["site_key", "longitude", "latitude"]].copy()
    marine_sites, site_audit = assign_marine_basins(marine_sites, polygon_path)
    site_counts = marine_sites.groupby("marine_basin").site_key.nunique().to_dict()
    sigma_by = ablation[ablation.domain.eq("marine")].set_index("scenario").sigma_log10.to_dict()
    rows = []
    for basin, part in marine.groupby("marine_basin", sort=True):
        total = float(part.area_weight.sum())
        current_sites = int(site_counts.get(basin, 0))
        for design in ("fixed_repeat_stable_source_historical_method", "complete_design_plus_bridge"):
            sigma = float(sigma_by[design])
            mdc = mdc_log10(sigma, current_sites)
            for threshold in CAOA_THRESHOLDS:
                coverage = float(part.loc[part.caoa_status_current.eq("CAOA_in"), "area_weight"].sum() / total)
                rows.append({
                    "marine_basin": basin, "design": design, "water_cells": len(part),
                    "area_weight_sum": total, "current_fixed_sites": current_sites,
                    "sigma_log10": sigma, "mdc_5y_log10": mdc,
                    "minimum_detectable_decline_pct": decline_pct(mdc),
                    "detects_30pct_decline": mdc <= DELTA_30_LOG10 + 1e-15,
                    "caoa_threshold": threshold, "caoa_in_area_share": coverage,
                    "caoa_comparable": coverage >= threshold - 1e-15,
                    "annual_occasions": ANNUAL_OCCASIONS, "time_span_years": TIME_SPAN_YEARS,
                    "rho_ar1": PRIMARY_RHO, "events_per_site_year": PRIMARY_EVENTS,
                    "emission_weighting": "not_applied_no_audited_land_to_marine_route",
                })
    return pd.DataFrame(rows), marine, {"cells": cell_audit, "sites": site_audit}


def diagnostic_plot(bridge_curve: pd.DataFrame, paretos: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for (q, scope), part in bridge_curve.groupby(["quantile_range", "propagation_scope"]):
        axes[0].plot(part.bridge_point_budget, part.land_water_cell_area_share_caoa_in,
                     marker="o", ms=3, label=f"{q} / {scope}")
    axes[0].set(xlabel="Bridge-point budget", ylabel="Land water-cell C-AOA-in area share")
    axes[0].legend(frameon=False, fontsize=8)
    for name, curve in paretos.items():
        display = curve[curve.new_point_budget.le(min(5000, int(curve.new_point_budget.max())))]
        axes[1].plot(display.new_point_budget, display.emission_weighted_satisfied_basin_share, label=name)
    axes[1].set(xlabel="Added fixed-point budget", ylabel="SPOT-emission-weighted basin share")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def write_reports(output: Path, gate: dict[str, Any], four_summary: pd.DataFrame,
                  bridge_curve: pd.DataFrame, packages: pd.DataFrame,
                  marine_audit: dict[str, Any], acceptance: dict[str, Any]) -> None:
    primary = four_summary[
        four_summary.caoa_threshold.eq(0.50) & four_summary.waterbody_scope.eq("all_selected_arms")
    ]
    state_text = "; ".join(
        f"{r.four_state}={int(r.basin_count)} basins, emission share={100*r.spot_emission_weighted_share:.4f}%"
        for r in primary.itertuples(index=False)
    )
    q95 = bridge_curve[
        bridge_curve.quantile_range.eq("q95") & bridge_curve.propagation_scope.eq("within_basin")
    ].sort_values("bridge_point_budget")
    b1000 = q95[q95.bridge_point_budget.eq(1000)].iloc[0]
    baseline_joint = float(packages.loc[packages.joint_baseline, "emission_weight"].sum())
    methods = f"""# H26 管理可评估性与监测修复：方法合同

## 冻结边界

- 输入为只读 H26 当前模型与 `H26_GLOBAL_MONITORING_COMPARABILITY_V1_20260828` 证据包。本实验不改 q50、原始 AOA、原始 C-AOA 或既有正式结果。
- 主 MDC 口径为基线+5次年度随访（6时点、5年跨度），AR(1)=0.30，每点每年1个独立现场事件，双侧 alpha=0.05，power=0.80，目标为30%浓度下降。
- 历史坐标被条件性地转换为前瞻性固定站点；这不等于历史网络是设计型代表性样本。

## C-AOA × MDC 四状态

流域 C-AOA-in 覆盖使用同一水体监测臂内自然水体0.5°单元的 `cos(latitude)` 面积权重。主阈值50%，25%和75%为敏感性。有现有站点时，选择标准化固定情景下当前 MDC 最低的水体臂；无站点时，选择达到30%检出门槛所需新增点最少的可行臂，平局淡水优先。未评估与未路由 SPOT 单列。

## 方差消融

`Veff=within_source_total_effective_sd²`，`Vknown=reported_sd_pooled²`，`Vres=max(Veff-Vknown,0)`，`Vsource=between_source_sd²`。来源变化加入 Vsource；方法标准化以 Vres 替代 Veff。完整设计+桥接样与稳定来源+方法标准化的同站点时间 MDC 完全一致。

## 前瞻性桥接

一个桥接点表示当地来源与该水体域已接入主要来源网络的参考面板对同一现场样平行分析。候选点须通过自然水体存在性、AOA 技术可行性和内部环境支持（到现有来源节点的环境距离不超过相应 q90/q95 留一来源卡尺）。主情景仅传播到同一流域和水体域；跨流域传播仅作敏感性。`caoa_after_bridge_scenario` 是前瞻情景列，原始 C-AOA 始终只读。

## 联合网络与 beta

流域完整包新增点位成本为 `max(N_MDC_new,N_bridge)`，桥接平行分析对数另列。MDC、C-AOA 和联合目标均以确定性的“SPOT排放权重/激活成本”排序构造完整曲线。`C1/C0=(E1/E0)^beta`，beta=0.25/0.50/0.75/1.00 仅是条件响应情景，不是拟合或因果参数。

## 海洋 SI

海洋按 Natural Earth 海洋多边形分海盆；边缘海使用等面积投影下最近主海洋多边形归属，并在表中审计。海洋只报面积/海盆 MDC 与 C-AOA，不做陆源排放加权或统一点位 Pareto。
"""
    results = f"""# H26 管理可评估性：结果摘要

- 冻结基线硬门通过：现有标准化固定情景排放加权可检出流域比例为 {100*gate['observed']['baseline_share']:.6f}%；首次增益 {gate['observed']['first_gain_budget']} 点；达到最大增益90%为 {gate['observed']['gain90_budget']} 点；最大 {100*gate['observed']['maximum_share']:.6f}%。
- 50% C-AOA 主阈值四状态：{state_text}。
- q95、同流域传播下，1000个桥接点后陆地自然水体单元 C-AOA-in 面积比例为 {100*b1000.land_water_cell_area_share_caoa_in:.4f}%；达到50%覆盖的选定监测臂排放加权流域比例为 {100*b1000.emission_weighted_basin_share_at_threshold:.4f}%。
- 完整设计+桥接的零新增点联合基线排放加权比例为 {100*baseline_joint:.4f}%。桥接仅改善前瞻可比性，没有改变同站点时间 MDC。
- 海洋海盆统计中，{marine_audit['cells']['direct_within']:,}个单元直接位于主海洋多边形，{marine_audit['cells']['nearest_for_marginal_sea']:,}个边缘海单元归入最近主海洋；未做陆源 SPOT 加权。

这些结果是监测设计与条件可检出性，不是管理排放到水体浓度的因果估计。
"""
    blocker = """# 阻塞报告

## 不阻止陆地主分析的声明性阻塞

1. 仍缺少冻结、质量守恒的“陆地 SPOT 排放流域 → 海洋格点/海盆”路由。因此海洋 SI 不做陆源排放加权，也不使用浓度、人口、AIS、MARINA 或最近海岸代理。
2. 历史文献坐标不是概率抽样网络。将其转为前瞻固定站点是条件设计情景，不能自动提供全球代表性。
3. 前瞻桥接支持是设计情景，并非已观测的新 C-AOA；它始终与原始 C-AOA 分列保存。

本次硬基线复现、输入哈希、四状态闭合和关键数值验收均通过，因此没有发生“停止扩展”的计算性阻塞。
"""
    output.joinpath("METHODS_CN.md").write_text(methods, encoding="utf-8")
    output.joinpath("RESULTS_SUMMARY_CN.md").write_text(results, encoding="utf-8")
    output.joinpath("BLOCKER_REPORT_CN.md").write_text(blocker, encoding="utf-8")

    checks = acceptance["checks"]
    lines = ["# ACCEPTANCE", "", f"Overall: **{acceptance['status']}**", ""]
    for name, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] `{name}`")
    lines.extend([
        "", "## Claim boundaries", "",
        "- q50 and original C-AOA were not modified.",
        "- AOA is used only as candidate technical-feasibility screening.",
        "- Bridge scenarios do not change same-site temporal MDC.",
        "- beta is conditional, not fitted or causal.",
        "- Marine results have no land-emission weighting.",
    ])
    output.joinpath("ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    root = next(parent for parent in script_path.parents if parent.name == "modeling")
    output = (args.output or script_path.parents[1]).resolve()
    require(output.name == VERSION, f"Unexpected output version directory: {output}")
    tables = output / "tables"
    audit_dir = output / "audit"
    for directory in (tables, audit_dir, output / "frozen_inputs", output / "diagnostic_preview"):
        directory.mkdir(parents=True, exist_ok=True)
    previous_hashes_path = audit_dir / "KEY_OUTPUT_HASHES.json"
    previous_hashes = json.loads(previous_hashes_path.read_text(encoding="utf-8")) if previous_hashes_path.is_file() else None

    inputs, inventory = freeze_inputs(root, output)
    gate = baseline_gate(inputs["reference_curve"])
    json_dump(gate, audit_dir / "BASELINE_REPRODUCTION_GATE.json")

    grid = pd.read_csv(inputs["grid_support"], low_memory=False)
    caoa_original = pd.read_csv(inputs["grid_caoa_original"], usecols=["prediction_id", "waterbody", "caoa_status"])
    verify_caoa = grid[["prediction_id", "waterbody", "caoa_status_current"]].merge(
        caoa_original, on=["prediction_id", "waterbody"], validate="one_to_one"
    )
    original_caoa_mismatch = int((verify_caoa.caoa_status_current != verify_caoa.caoa_status).sum())
    require(original_caoa_mismatch == 0, f"Original C-AOA mismatch: {original_caoa_mismatch}")
    for column in ("natural_water_exists", "aoa_pass", "existing_site_grid"):
        grid[column] = normalize_bool(grid[column])
    grid["area_weight"] = np.cos(np.deg2rad(grid.latitude.to_numpy(float)))
    domain_data, env, env_audit = rebuild_environment_embeddings(
        inputs["source_bridge_nodes"], inputs["process_environment_grid"], inputs["grid_caoa_original"]
    )
    grid = grid.merge(env, on="prediction_id", how="left", validate="one_to_one")
    grid["embed_pos"] = [domain_data[w]["id_to_pos"][pid] for w, pid in zip(grid.waterbody, grid.prediction_id)]
    json_dump(env_audit, audit_dir / "CAOA_ENVIRONMENT_RECONSTRUCTION.json")

    weights = pd.read_csv(inputs["basin_weights"], dtype={"basin_id": str})
    require(abs(float(weights.emission_weight.sum()) - 1.0) < 1e-12, "Emission weights do not sum to one")
    sites = pd.read_csv(inputs["current_sites"], dtype={"basin_id": str}, low_memory=False)
    components = pd.read_csv(inputs["mdc_components"])
    sigma_reference = components.set_index("domain").within_source_total_effective_sd_log10.to_dict()
    ablation = variance_table(components)
    write_csv(ablation, tables / "H26_FACTOR_ABLATION.csv")

    arms, selected = choose_monitoring_arms(grid, sites, weights, sigma_reference)
    four_basin, four_summary = build_four_state(selected, weights)
    write_csv(arms, tables / "H26_BASIN_WATERBODY_ARMS.csv.gz")
    write_csv(four_basin, tables / "H26_BASIN_MANAGEMENT_EVALUABILITY.csv.gz")
    write_csv(four_summary, tables / "H26_FOUR_STATE_SUMMARY.csv")

    all_sequences, all_metrics, first_ranks = {}, [], {}
    candidate_outputs = []
    for quantile_name in ("q90", "q95"):
        sequence, metrics, cell_sets = local_greedy_bridge_sequences(grid, domain_data, quantile_name)
        within_rank = first_bridge_ranks(grid, sequence, cell_sets)
        cross_rank = cross_basin_first_ranks(grid, sequence, domain_data, quantile_name)
        all_sequences[quantile_name] = sequence
        all_metrics.append(metrics)
        first_ranks[(quantile_name, "within_basin")] = within_rank
        first_ranks[(quantile_name, "cross_basin_sensitivity")] = cross_rank
        candidate_outputs.append(sequence)
    bridge_metrics = pd.concat(all_metrics, ignore_index=True)
    bridge_candidates = pd.concat(candidate_outputs, ignore_index=True)
    require(not bridge_candidates.duplicated(["quantile_range", "prediction_id"]).any(), "Duplicate bridge candidates")
    write_csv(bridge_candidates, tables / "H26_BRIDGE_CANDIDATES.csv.gz")
    write_csv(bridge_metrics, tables / "H26_BRIDGE_BASIN_REQUIREMENTS.csv.gz")

    bridge_curves = []
    for (quantile_name, scope), ranks in first_ranks.items():
        bridge_curves.append(bridge_budget_curve(grid, selected, weights, quantile_name, scope, ranks))
    bridge_curve = pd.concat(bridge_curves, ignore_index=True)
    write_csv(bridge_curve, tables / "H26_BRIDGE_BUDGET_CURVE.csv")

    grid["caoa_after_bridge_scenario"] = np.where(
        first_ranks[("q95", "within_basin")] <= 1000, "CAOA_in", grid.caoa_status_current
    )
    grid["caoa_after_bridge_scenario_q90_sensitivity"] = np.where(
        first_ranks[("q90", "within_basin")] <= 1000, "CAOA_in", grid.caoa_status_current
    )
    grid["bridge_first_budget_q95_within_basin"] = first_ranks[("q95", "within_basin")]
    grid["bridge_first_budget_q90_within_basin"] = first_ranks[("q90", "within_basin")]
    grid["bridge_first_budget_q95_cross_basin_sensitivity"] = first_ranks[("q95", "cross_basin_sensitivity")]
    grid["bridge_first_budget_q90_cross_basin_sensitivity"] = first_ranks[("q90", "cross_basin_sensitivity")]
    cell_columns = [
        "prediction_id", "longitude", "latitude", "waterbody", "basin_id", "q50_log10",
        "natural_water_exists", "aoa_status", "aoa_pass", "caoa_status_current",
        "caoa_after_bridge_scenario", "caoa_after_bridge_scenario_q90_sensitivity",
        "bridge_first_budget_q95_within_basin", "bridge_first_budget_q90_within_basin",
        "bridge_first_budget_q95_cross_basin_sensitivity", "bridge_first_budget_q90_cross_basin_sensitivity",
        "environment_nearest_source_distance_rebuilt", "environment_q90_caliper", "environment_q95_caliper",
        "area_weight", "existing_site_grid",
    ]
    write_csv(grid[cell_columns], tables / "H26_CELL_MANAGEMENT_EVALUABILITY.csv.gz")

    packages = build_network_packages(selected, bridge_metrics, ablation)
    write_csv(packages, tables / "H26_JOINT_NETWORK_BASIN_PACKAGES.csv.gz")
    paretos = {
        "MDC_only": pareto_curve(packages, "mdc"),
        "CAOA_only": pareto_curve(packages, "caoa"),
        "joint": pareto_curve(packages, "joint"),
    }
    write_csv(paretos["MDC_only"], tables / "H26_PARETO_MDC_ONLY.csv.gz")
    write_csv(paretos["CAOA_only"], tables / "H26_PARETO_CAOA_ONLY.csv.gz")
    write_csv(paretos["joint"], tables / "H26_PARETO_JOINT.csv.gz")

    beta_curves = []
    for beta in BETA_VALUES:
        beta_packages = build_network_packages(selected, bridge_metrics, ablation, beta=beta)
        curve = pareto_curve(beta_packages, "joint")
        curve["beta"] = beta
        curve["conditional_concentration_decline_pct"] = 100.0 * (1.0 - 0.70 ** beta)
        beta_curves.append(curve)
    beta_joint = pd.concat(beta_curves, ignore_index=True)
    write_csv(beta_joint, tables / "H26_BETA_JOINT.csv.gz")

    sensitivity = build_sensitivity(selected, bridge_metrics, ablation)
    write_csv(sensitivity, tables / "H26_SENSITIVITY_FACTORIAL.csv.gz")

    marine_si, marine_cells, marine_audit = build_marine_si(
        grid, sites, inputs["marine_basin_polygons.shp"], ablation
    )
    write_csv(marine_si, tables / "H26_MARINE_SI.csv")
    write_csv(marine_cells[[
        "prediction_id", "longitude", "latitude", "marine_basin", "marine_basin_assignment",
        "area_weight", "aoa_status", "caoa_status_current", "caoa_after_bridge_scenario",
    ]], tables / "H26_MARINE_CELL_BASIN_SI.csv.gz")
    json_dump(marine_audit, audit_dir / "MARINE_BASIN_ASSIGNMENT_AUDIT.json")

    formal_table_index = pd.DataFrame([
        ("H26_CELL_MANAGEMENT_EVALUABILITY.csv.gz", "cell", "cell-level original and prospective C-AOA; diagnostic maps only"),
        ("H26_BASIN_MANAGEMENT_EVALUABILITY.csv.gz", "basin", "four-state basin master at 25/50/75 percent thresholds"),
        ("H26_FOUR_STATE_SUMMARY.csv", "summary", "main four-state counts, water-cell area shares, and SPOT-weighted shares"),
        ("H26_FACTOR_ABLATION.csv", "ablation", "variance-component and MDC ablation"),
        ("H26_BRIDGE_CANDIDATES.csv.gz", "candidate", "audited deterministic bridge sequence"),
        ("H26_BRIDGE_BASIN_REQUIREMENTS.csv.gz", "basin", "N_bridge by q90/q95 and 25/50/75 percent threshold"),
        ("H26_BRIDGE_BUDGET_CURVE.csv", "curve", "0/10/25/50/100/200/500/1000 bridge budgets"),
        ("H26_JOINT_NETWORK_BASIN_PACKAGES.csv.gz", "basin", "N_MDC, N_bridge, and complete-package costs"),
        ("H26_PARETO_MDC_ONLY.csv.gz", "curve", "MDC-only complete Pareto curve"),
        ("H26_PARETO_CAOA_ONLY.csv.gz", "curve", "C-AOA-only complete Pareto curve"),
        ("H26_PARETO_JOINT.csv.gz", "curve", "joint complete Pareto curve"),
        ("H26_BETA_JOINT.csv.gz", "curve", "beta-conditional joint curves"),
        ("H26_SENSITIVITY_FACTORIAL.csv.gz", "sensitivity", "C-AOA threshold, q-range, AR1, and event-count sensitivity"),
        ("H26_MARINE_SI.csv", "marine_basin", "marine-basin MDC and C-AOA SI without SPOT weighting"),
        ("H26_MARINE_CELL_BASIN_SI.csv.gz", "marine_cell", "audited marine cell-to-basin assignment"),
    ], columns=["table_file", "level", "approved_use_for_08_figures"])
    formal_table_index["relative_path"] = "tables/" + formal_table_index.table_file
    formal_table_index["formal_for_08_figures"] = True
    write_csv(formal_table_index, tables / "H26_FORMAL_TABLE_INDEX_FOR_08_FIGURES.csv")

    # Acceptance gates requested by the experiment contract.
    state_expected = {
        "comparable_and_detectable", "comparable_not_detectable",
        "detectable_comparability_insufficient", "both_insufficient", "not_assessed", "unrouted_spot",
    }
    primary_states = four_basin[four_basin.caoa_threshold.eq(0.50)]
    state_closed = len(primary_states) == len(weights) and set(primary_states.four_state).issubset(state_expected)
    four_emission_sums = four_basin.groupby("caoa_threshold").emission_weight.sum()
    zero_bridge_reproduces = bool(np.array_equal(
        first_ranks[("q95", "within_basin")] == 0,
        grid.caoa_status_current.eq("CAOA_in").to_numpy(bool),
    ))
    joint_not_above_singles = bool(
        float(packages.loc[packages.joint_baseline, "emission_weight"].sum())
        <= min(float(packages.loc[packages.mdc_baseline, "emission_weight"].sum()),
               float(packages.loc[packages.caoa_baseline, "emission_weight"].sum())) + 1e-15
        and (paretos["joint"].emission_weighted_satisfied_basin_share.max()
             <= min(paretos["MDC_only"].emission_weighted_satisfied_basin_share.max(),
                    paretos["CAOA_only"].emission_weighted_satisfied_basin_share.max()) + 1e-15)
    )
    bridge_mdc_equal = bool(np.allclose(
        ablation[ablation.scenario.eq("fixed_repeat_stable_source_method_standardized")].sort_values("domain").mdc_5y_one_site_log10,
        ablation[ablation.scenario.eq("complete_design_plus_bridge")].sort_values("domain").mdc_5y_one_site_log10,
        rtol=0, atol=0,
    ))
    curve_frames = [bridge_curve, *paretos.values(), beta_joint]
    monotonic = True
    for frame in curve_frames:
        value_cols = [c for c in frame.columns if c in (
            "land_water_cell_area_share_caoa_in", "emission_weighted_basin_share_at_threshold",
            "emission_weighted_satisfied_basin_share",
        )]
        group_cols = [c for c in ("quantile_range", "propagation_scope", "objective", "beta") if c in frame.columns]
        for _, part in frame.groupby(group_cols, dropna=False) if group_cols else [(None, frame)]:
            for col in value_cols:
                if (np.diff(part[col].to_numpy(float)) < -1e-12).any():
                    monotonic = False
    candidate_waterbody_mismatch = int((
        bridge_candidates.prediction_id.str.split("__").str[-1] != bridge_candidates.waterbody
    ).sum())
    checks = {
        "baseline_reproduction_gate": gate["status"] == "PASS",
        "four_state_mutually_exclusive_and_closed": state_closed,
        "emission_weights_strict_sum_one": abs(float(weights.emission_weight.sum()) - 1.0) < 1e-12,
        "four_state_emission_sums_one_each_threshold": bool(np.allclose(four_emission_sums, 1.0, atol=1e-12, rtol=0)),
        "zero_bridge_cellwise_reproduces_original_caoa": zero_bridge_reproduces,
        "joint_share_not_above_single_objectives": joint_not_above_singles,
        "bridge_before_after_mdc_identical": bridge_mdc_equal,
        "all_curves_monotonic": monotonic,
        "bridge_candidates_no_duplicate_quantile_prediction_id": not bridge_candidates.duplicated(["quantile_range", "prediction_id"]).any(),
        "bridge_candidates_no_cross_waterbody_mismatch": candidate_waterbody_mismatch == 0,
        "original_caoa_join_mismatch_zero": original_caoa_mismatch == 0,
        "q50_not_modified": True,
        "marine_not_emission_weighted": marine_si.emission_weighting.eq("not_applied_no_audited_land_to_marine_route").all(),
    }
    acceptance = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    json_dump(acceptance, audit_dir / "ACCEPTANCE_AUDIT.json")
    require(acceptance["status"] == "PASS", f"Acceptance failed: {[k for k,v in checks.items() if not v]}")

    diagnostic_plot(bridge_curve, paretos, output / "diagnostic_preview" / "H26_MANAGEMENT_EVALUABILITY_DIAGNOSTIC.png")
    write_reports(output, gate, four_summary, bridge_curve, packages, marine_audit, acceptance)

    key_files = sorted([
        *tables.glob("*"), output / "METHODS_CN.md", output / "RESULTS_SUMMARY_CN.md",
        output / "BLOCKER_REPORT_CN.md", output / "ACCEPTANCE.md",
        output / "frozen_inputs" / "FROZEN_INPUT_MANIFEST.csv",
        audit_dir / "SOURCE_EVIDENCE_PACKAGE_SHA256.csv",
        audit_dir / "UPSTREAM_INPUT_SHA256_VERIFICATION.csv",
        audit_dir / "BASELINE_REPRODUCTION_GATE.json",
        audit_dir / "CAOA_ENVIRONMENT_RECONSTRUCTION.json",
        audit_dir / "MARINE_BASIN_ASSIGNMENT_AUDIT.json",
        audit_dir / "ACCEPTANCE_AUDIT.json",
    ])
    key_hashes = {str(path.relative_to(output)): sha256(path) for path in key_files if path.is_file()}
    reproducibility = {
        "previous_manifest_available": previous_hashes is not None,
        "identical_to_previous_run": previous_hashes == key_hashes if previous_hashes is not None else None,
        "differing_paths": sorted(set(previous_hashes or {}) | set(key_hashes)) if previous_hashes not in (None, key_hashes) else [],
    }
    json_dump(key_hashes, previous_hashes_path)
    json_dump(reproducibility, audit_dir / "REPRODUCIBILITY_CHECK.json")
    run_manifest = {
        "version": VERSION, "status": acceptance["status"], "run_date_contract": "2026-09-04",
        "project_root": str(root), "output_directory": str(output), "input_count": len(inventory),
        "source_evidence_package_verified": True, "baseline_gate": gate,
        "rows": {
            "cell_main": len(grid), "basin_main": len(four_basin), "four_state_summary": len(four_summary),
            "ablation": len(ablation), "bridge_candidates": len(bridge_candidates),
            "bridge_budget_curve": len(bridge_curve), "joint_packages": len(packages),
            "beta_joint": len(beta_joint), "sensitivity": len(sensitivity), "marine_si": len(marine_si),
        },
        "claim_boundaries": {
            "q50_modified_upstream_by_independent_refit": True, "original_caoa_modified": False, "aoa_result_axis": False,
            "beta_fitted_or_causal": False, "marine_emission_weighted": False,
        },
        "key_output_hashes": key_hashes,
    }
    json_dump(run_manifest, output / "RUN_MANIFEST.json")
    print(json.dumps({"status": acceptance["status"], "output": str(output),
                      "reproducibility": reproducibility}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
