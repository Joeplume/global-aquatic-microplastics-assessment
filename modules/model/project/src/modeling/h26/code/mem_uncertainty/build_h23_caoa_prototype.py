"""Build and render a prototype calibration area of applicability (C-AOA).

The prototype asks whether a prediction cell has nearby training *sources*
from at least two known detection-method families in local environmental
covariate space.  It is deliberately narrower than a final calibration AOA:
particle-size protocol, QA/QC, sample volume, graph effective resistance, and
masked-edge recovery are not yet included.

Predictions are never masked.  The output is a support annotation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NATURAL_DOMAINS = (
    "reference_surface_freshwater",
    "transition_coastal",
    "marine",
)
DOMAIN_LABELS = {
    "reference_surface_freshwater": "Freshwater",
    "transition_coastal": "Transition / coastal",
    "marine": "Marine",
}

DOMAIN_FEATURES = {
    "reference_surface_freshwater": (
        "night_lights_radiance_2020",
        "m6_road_density_m_per_km2",
        "spot_land_plastic_pressure_2020",
        "annual_runoff_mm_2020",
        "m6_annual_max_daily_runoff_mm",
        "river_to_sea_microplastic_output_kg_year",
        "m6_modis2019_strict_cropland_pct_0p5deg",
        "m6_ghsl2020_nonresidential_builtup_pct_0p5deg",
        "m6_global_annual_mean_2m_air_temperature_c_2020",
        "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
        "m6_modis2020_terrestrial_ndvi_annual_mean_0p5deg",
    ),
    "transition_coastal": (
        "night_lights_radiance_2020",
        "m6_road_density_m_per_km2",
        "spot_land_plastic_pressure_2020",
        "annual_runoff_mm_2020",
        "m6_annual_max_daily_runoff_mm",
        "river_to_sea_microplastic_output_kg_year",
        "ais_activity_density",
        "marine_aquaculture_pressure_tonnes_50km",
        "m6_sea_surface_salinity_psu",
        "marina_pressure_h100_kg_year",
        "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
        "m6_modis2019_strict_cropland_pct_0p5deg",
        "m6_ghsl2020_nonresidential_builtup_pct_0p5deg",
        "m6_global_annual_mean_2m_air_temperature_c_2020",
        "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
        "m6_modis2020_terrestrial_ndvi_annual_mean_0p5deg",
    ),
    "marine": (
        "river_to_sea_microplastic_output_kg_year",
        "ais_activity_density",
        "marine_aquaculture_pressure_tonnes_50km",
        "m6_sea_surface_salinity_psu",
        "marina_pressure_h100_kg_year",
        "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
        "m6_global_annual_mean_2m_air_temperature_c_2020",
        "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
    ),
}

LOG_FEATURES = {
    "night_lights_radiance_2020",
    "m6_road_density_m_per_km2",
    "spot_land_plastic_pressure_2020",
    "annual_runoff_mm_2020",
    "m6_annual_max_daily_runoff_mm",
    "river_to_sea_microplastic_output_kg_year",
    "ais_activity_density",
    "marine_aquaculture_pressure_tonnes_50km",
    "marina_pressure_h100_kg_year",
    "positive_water_depth_m",
}

KNOWN_METHOD_FAMILIES = (
    "FTIR",
    "Raman",
    "LDIR",
    "thermal_mass",
    "visual_micro",
    "other",
)
STATUS_ORDER = ("C-AOA in", "C-AOA limited", "C-AOA out")
STATUS_CODE = {name: index for index, name in enumerate(STATUS_ORDER)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_family(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "na", "未识别", "unknown"}:
        return "unknown"
    if "ldir" in text:
        return "LDIR"
    if any(token in text for token in ("py-gc", "gc-ms", "thermal", "pyr")):
        return "thermal_mass"
    if "raman" in text:
        return "Raman"
    if "ftir" in text:
        return "FTIR"
    if any(token in text for token in ("microscope", "visual", "sem/eds", "sem_eds")):
        return "visual_micro"
    return "other"


def mode_or_unknown(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    return clean.mode().iloc[0] if len(clean) else "unknown"


def transform_feature(values: np.ndarray, feature: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if feature in LOG_FEATURES:
        result = np.sign(result) * np.log1p(np.abs(result))
    return result


def effective_method_count(
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    method_codes: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    known = method_codes[neighbor_indices] >= 0
    shifted = np.maximum(distances - distances[:, :1], 0.0)
    weights = np.exp(-0.5 * np.square(shifted / max(bandwidth, 1e-8))) * known
    family_weights = np.zeros((len(distances), len(KNOWN_METHOD_FAMILIES)), dtype=float)
    for code in range(len(KNOWN_METHOD_FAMILIES)):
        family_weights[:, code] = np.sum(weights * (method_codes[neighbor_indices] == code), axis=1)
    totals = family_weights.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        family_weights,
        totals,
        out=np.zeros_like(family_weights),
        where=totals > 0,
    )
    entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-15)), 0.0),
        axis=1,
    )
    effective = np.exp(entropy)
    effective[totals[:, 0] <= 0] = 1.0
    return effective


def fit_domain_support(
    records: pd.DataFrame,
    prediction: pd.DataFrame,
    domain: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors

    subset = records.loc[records["waterbody"].eq(domain)].copy()
    grid = prediction.loc[prediction["prediction_domain"].eq(domain)].copy()
    require(len(subset) > 50, f"Too few training records for {domain}: {len(subset)}")
    require(len(grid) > 0, f"No prediction rows for {domain}")

    source_meta = subset.groupby("source_group", sort=False).agg(
        source_rows=("stable_record_id", "size"),
        country=("country_iso3_v3", mode_or_unknown),
        detection_method=("detection_method", mode_or_unknown),
    )
    source_meta["method_family"] = source_meta["detection_method"].map(method_family)

    candidate_features = [feature for feature in DOMAIN_FEATURES[domain] if feature in subset and feature in grid]
    source_values = subset.groupby("source_group", sort=False)[candidate_features].median()
    source = source_meta.join(source_values, how="inner")

    used_features: list[str] = []
    source_columns: list[np.ndarray] = []
    grid_columns: list[np.ndarray] = []
    preprocessing: dict[str, dict[str, float]] = {}
    for feature in candidate_features:
        source_raw = pd.to_numeric(source[feature], errors="coerce").to_numpy(float)
        grid_raw = pd.to_numeric(grid[feature], errors="coerce").to_numpy(float)
        if np.isfinite(source_raw).mean() < 0.20 or np.isfinite(grid_raw).mean() < 0.20:
            continue
        source_transformed = transform_feature(source_raw, feature)
        grid_transformed = transform_feature(grid_raw, feature)
        median = float(np.nanmedian(source_transformed))
        q10, q90 = np.nanquantile(source_transformed, [0.10, 0.90])
        scale = float(q90 - q10)
        if not np.isfinite(scale) or scale <= 1e-10:
            continue
        source_scaled = np.clip((np.where(np.isfinite(source_transformed), source_transformed, median) - median) / scale, -10, 10)
        grid_scaled = np.clip((np.where(np.isfinite(grid_transformed), grid_transformed, median) - median) / scale, -10, 10)
        used_features.append(feature)
        source_columns.append(source_scaled)
        grid_columns.append(grid_scaled)
        preprocessing[feature] = {"median_transformed": median, "q10_q90_scale": scale}

    require(len(used_features) >= 4, f"Insufficient usable features for {domain}: {used_features}")
    source_matrix = np.column_stack(source_columns)
    grid_matrix = np.column_stack(grid_columns)
    full_pca = PCA(svd_solver="full").fit(source_matrix)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    dimensions = min(12, max(2, int(np.searchsorted(cumulative, 0.90) + 1)))
    pca = PCA(n_components=dimensions, svd_solver="full").fit(source_matrix)
    source_embedding = pca.transform(source_matrix)
    grid_embedding = pca.transform(grid_matrix)

    method_codes = np.array(
        [KNOWN_METHOD_FAMILIES.index(item) if item in KNOWN_METHOD_FAMILIES else -1 for item in source["method_family"]],
        dtype=int,
    )
    known_family_counts = {
        family: int(np.sum(source["method_family"].to_numpy() == family))
        for family in KNOWN_METHOD_FAMILIES
    }
    active_families = [family for family, count in known_family_counts.items() if count >= 2]
    require(len(active_families) >= 2, f"Fewer than two known method families for {domain}")

    pairwise = pairwise_distances(source_embedding)
    np.fill_diagonal(pairwise, np.inf)
    train_nearest = np.min(pairwise, axis=1)
    family_train_distances = []
    for family in active_families:
        mask = source["method_family"].to_numpy() == family
        family_train_distances.append(np.min(pairwise[:, mask], axis=1))
    family_train = np.column_stack(family_train_distances)
    train_bridge = np.partition(family_train, kth=1, axis=1)[:, 1]

    k_neighbors = min(25, len(source) - 1)
    order = np.argpartition(pairwise, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]
    order_distances = np.take_along_axis(pairwise, order, axis=1)
    sort_order = np.argsort(order_distances, axis=1)
    order = np.take_along_axis(order, sort_order, axis=1)
    order_distances = np.take_along_axis(order_distances, sort_order, axis=1)
    bandwidth = float(np.nanmedian(train_nearest[np.isfinite(train_nearest)]))
    train_effective = effective_method_count(order_distances, order, method_codes, bandwidth)

    thresholds = {
        "nearest_q95": float(np.nanquantile(train_nearest, 0.95)),
        "nearest_q99": float(np.nanquantile(train_nearest, 0.99)),
        "bridge_q90": float(np.nanquantile(train_bridge, 0.90)),
        "bridge_q95": float(np.nanquantile(train_bridge, 0.95)),
        "bridge_q99": float(np.nanquantile(train_bridge, 0.99)),
        "effective_q10": float(np.nanquantile(train_effective, 0.10)),
        "effective_q02": float(np.nanquantile(train_effective, 0.02)),
        "effective_median": float(np.nanmedian(train_effective)),
        "neighbor_bandwidth": bandwidth,
    }

    nearest_model = NearestNeighbors(n_neighbors=min(25, len(source))).fit(source_embedding)
    grid_distances, grid_neighbors = nearest_model.kneighbors(grid_embedding)
    grid_nearest = grid_distances[:, 0]
    grid_effective = effective_method_count(grid_distances, grid_neighbors, method_codes, bandwidth)

    family_grid_distances = []
    for family in active_families:
        mask = source["method_family"].to_numpy() == family
        family_model = NearestNeighbors(n_neighbors=1).fit(source_embedding[mask])
        family_distance, _ = family_model.kneighbors(grid_embedding)
        family_grid_distances.append(family_distance[:, 0])
    family_grid = np.column_stack(family_grid_distances)
    family_order = np.argsort(family_grid, axis=1)
    grid_bridge = np.take_along_axis(family_grid, family_order, axis=1)[:, 1]
    closest_family = np.array(active_families, dtype=object)[family_order[:, 0]]
    second_family = np.array(active_families, dtype=object)[family_order[:, 1]]

    in_support = (
        (grid_nearest <= thresholds["nearest_q95"])
        & (grid_bridge <= thresholds["bridge_q90"])
        & (grid_effective >= thresholds["effective_q10"])
    )
    limited_support = (
        ~in_support
        & (grid_nearest <= thresholds["nearest_q99"])
        & (grid_bridge <= thresholds["bridge_q99"])
        & (grid_effective >= max(1.05, thresholds["effective_q02"]))
    )
    status = np.full(len(grid), "C-AOA out", dtype=object)
    status[limited_support] = "C-AOA limited"
    status[in_support] = "C-AOA in"

    near_score = np.exp(-0.5 * np.square(grid_nearest / max(thresholds["nearest_q95"], 1e-8)))
    bridge_score = np.exp(-0.5 * np.square(grid_bridge / max(thresholds["bridge_q95"], 1e-8)))
    diversity_denominator = max(thresholds["effective_median"] - 1.0, 1e-8)
    diversity_score = np.clip((grid_effective - 1.0) / diversity_denominator, 0.0, 1.0)
    score = np.minimum(np.minimum(near_score, bridge_score), diversity_score)

    output = pd.DataFrame(
        {
            "prediction_id": grid["prediction_id"].astype(str).to_numpy(),
            "longitude": pd.to_numeric(grid["longitude"], errors="coerce").to_numpy(float),
            "latitude": pd.to_numeric(grid["latitude"], errors="coerce").to_numpy(float),
            "waterbody": domain,
            "caoa_status": status,
            "caoa_code": np.array([STATUS_CODE[item] for item in status], dtype=np.int8),
            "caoa_score_v0_1": score,
            "nearest_source_distance": grid_nearest,
            "two_method_bridge_distance": grid_bridge,
            "effective_method_families": grid_effective,
            "closest_known_method_family": closest_family,
            "second_known_method_family": second_family,
        }
    )
    audit = {
        "domain": domain,
        "training_records": int(len(subset)),
        "training_sources": int(len(source)),
        "prediction_cells": int(len(grid)),
        "candidate_features": candidate_features,
        "used_features": used_features,
        "excluded_from_geometry_by_design": [
            "plastic_waste_generation_t",
            "wgi_government_effectiveness_2020",
            "country label",
            "paper/source ID",
            "response concentration",
        ],
        "preprocessing": preprocessing,
        "pca_dimensions": int(dimensions),
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "method_family_source_counts": {
            str(key): int(value) for key, value in source["method_family"].value_counts().items()
        },
        "active_known_method_families": active_families,
        "thresholds_from_leave_one_source_out_training_distribution": thresholds,
        "status_counts": {str(key): int(value) for key, value in pd.Series(status).value_counts().items()},
    }
    return output, audit


def compute(args: argparse.Namespace) -> int:
    training_path = args.training.resolve()
    prediction_path = args.prediction.resolve()
    observation_path = args.observation.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (training_path, prediction_path, observation_path):
        require(path.exists(), f"Missing input: {path}")

    all_features = sorted({feature for features in DOMAIN_FEATURES.values() for feature in features})
    training_columns = [
        "stable_record_id",
        "country_iso3_v3",
        "detection_method",
        *all_features,
    ]
    prediction_columns = [
        "prediction_id",
        "longitude",
        "latitude",
        "prediction_domain",
        *all_features,
    ]
    training = pd.read_csv(training_path, usecols=training_columns, low_memory=False)
    prediction = pd.read_csv(prediction_path, usecols=prediction_columns, low_memory=False)
    observation = pd.read_csv(
        observation_path,
        usecols=["stable_record_id", "source_group", "waterbody", "detection_method"],
        low_memory=False,
    )
    observation = observation.rename(columns={"detection_method": "observation_detection_method"})
    records = training.merge(observation, on="stable_record_id", how="inner", validate="one_to_one")
    records["detection_method"] = records["observation_detection_method"].fillna(records["detection_method"])
    prediction = prediction.loc[prediction["prediction_domain"].isin(NATURAL_DOMAINS)].copy()
    require(len(records) == 34_853, f"Expected 34,853 observation records, found {len(records):,}")
    require(len(prediction) == 259_200, f"Expected 259,200 natural prediction cells, found {len(prediction):,}")

    outputs = []
    domain_audits = []
    for domain in NATURAL_DOMAINS:
        domain_output, domain_audit = fit_domain_support(records, prediction, domain)
        outputs.append(domain_output)
        domain_audits.append(domain_audit)
    grid = pd.concat(outputs, ignore_index=True)
    require(len(grid) == 259_200, "C-AOA grid row closure failed")
    require(grid["prediction_id"].nunique() == len(grid), "C-AOA prediction IDs are not unique")
    require(np.isfinite(grid[["longitude", "latitude", "caoa_score_v0_1"]].to_numpy(float)).all(), "Non-finite C-AOA outputs")

    grid_path = output_dir / "H23_CAOA_V0_1_GRID.csv.gz"
    grid.to_csv(grid_path, index=False, compression="gzip")
    counts = (
        grid.groupby(["waterbody", "caoa_status"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    for status in STATUS_ORDER:
        counts[f"{status.replace(' ', '_').lower()}_pct"] = 100.0 * counts[status] / counts["total"]
    counts.reset_index().to_csv(output_dir / "H23_CAOA_V0_1_COUNTS.csv", index=False, encoding="utf-8-sig")

    audit = {
        "status": "COMPLETED_PROTOTYPE_NOT_RELEASE",
        "version": "H23_CAOA_V0_1_LOCAL_ANALOG_METHOD_DIVERSITY",
        "definition": "Local environmental analog support from source-equal training centroids plus at least two known detection-method families.",
        "claim_boundary": "Diagnostic calibration-support annotation; not a probability, not a validated measurement-error correction, and not a mask.",
        "not_yet_included": [
            "particle-size protocol graph",
            "QA/QC protocol graph",
            "sample-volume support",
            "method-source graph effective resistance",
            "masked-edge recovery validation",
            "formal propensity overlap weighting",
        ],
        "inputs": {
            "training": str(training_path),
            "prediction": str(prediction_path),
            "observation_layer_records": str(observation_path),
            "training_sha256": sha256(training_path),
            "prediction_sha256": sha256(prediction_path),
            "observation_sha256": sha256(observation_path),
        },
        "output_grid": str(grid_path),
        "rows": int(len(grid)),
        "domain_audits": domain_audits,
    }
    (output_dir / "H23_CAOA_V0_1_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": audit["status"], "grid": str(grid_path), "rows": len(grid)}, ensure_ascii=False))
    return 0


def robinson_boundary(transformer) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.linspace(-90, 90, 721)
    x_left, y_left = transformer.transform(np.full_like(latitude, -180.0), latitude)
    x_right, y_right = transformer.transform(np.full_like(latitude, 180.0), latitude[::-1])
    top_lon = np.linspace(180, -180, 1441)
    bottom_lon = np.linspace(-180, 180, 1441)
    x_top, y_top = transformer.transform(top_lon, np.full_like(top_lon, 90.0))
    x_bottom, y_bottom = transformer.transform(bottom_lon, np.full_like(bottom_lon, -90.0))
    return (
        np.concatenate([x_left, x_top, x_right, x_bottom, x_left[:1]]),
        np.concatenate([y_left, y_top, y_right, y_bottom, y_left[:1]]),
    )


def project_grid(longitude: np.ndarray, latitude: np.ndarray, transformer) -> tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    x, y = transformer.transform(lon_grid, lat_grid)
    return np.asarray(x, float), np.asarray(y, float)


def plot(args: argparse.Namespace) -> int:
    import geopandas as gpd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from pyproj import CRS, Transformer

    grid_path = args.grid.resolve()
    countries_path = args.countries.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(grid_path.exists(), f"Missing C-AOA grid: {grid_path}")
    require(countries_path.exists(), f"Missing countries layer: {countries_path}")
    grid = pd.read_csv(grid_path, low_memory=False)
    require(len(grid) == 259_200, f"Expected 259,200 C-AOA rows, found {len(grid):,}")
    require(grid["prediction_id"].nunique() == len(grid), "C-AOA IDs are not unique")

    longitude = np.sort(grid["longitude"].unique())
    latitude = np.sort(grid["latitude"].unique())
    require((len(latitude), len(longitude)) == (360, 720), "Unexpected C-AOA grid shape")
    codes = grid.pivot(index="latitude", columns="longitude", values="caoa_code").reindex(index=latitude, columns=longitude).to_numpy(float)
    domain_values = grid.assign(
        domain_code=pd.Categorical(grid["waterbody"], categories=NATURAL_DOMAINS).codes
    )
    domains = domain_values.pivot(index="latitude", columns="longitude", values="domain_code").reindex(index=latitude, columns=longitude).to_numpy(float)
    require(np.isfinite(codes).all() and np.isfinite(domains).all(), "C-AOA grid is incomplete")

    robinson = CRS.from_proj4("+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", robinson, always_xy=True)
    lon_edges = np.arange(-180.0, 180.0 + 0.5, 0.5)
    lat_edges = np.arange(-90.0, 90.0 + 0.5, 0.5)
    edge_x, edge_y = project_grid(lon_edges, lat_edges, transformer)
    center_x, center_y = project_grid(longitude, latitude, transformer)
    boundary_x, boundary_y = robinson_boundary(transformer)
    countries = gpd.read_file(countries_path).to_crs(robinson)

    colors = ("#238B8E", "#F2B134", "#C94C4C")
    palette = ListedColormap(colors, name="caoa_in_limited_out")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], palette.N)
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    def draw_axis(axis, values: np.ndarray, outline_domains: bool) -> None:
        axis.pcolormesh(
            edge_x,
            edge_y,
            values,
            cmap=palette,
            norm=norm,
            shading="flat",
            linewidth=0,
            antialiased=False,
            rasterized=True,
            zorder=1,
        )
        if outline_domains:
            axis.contour(center_x, center_y, domains, levels=[0.5, 1.5], colors="#334A52", linewidths=0.22, alpha=0.7, zorder=2)
        countries.boundary.plot(ax=axis, color="#5A6267", linewidth=0.30, alpha=0.82, zorder=3)
        axis.plot(boundary_x, boundary_y, color="#3D484D", linewidth=0.65, zorder=4)
        axis.set_xlim(float(np.nanmin(edge_x)), float(np.nanmax(edge_x)))
        axis.set_ylim(float(np.nanmin(edge_y)), float(np.nanmax(edge_y)))
        axis.set_aspect("equal")
        axis.set_axis_off()

    legend = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in zip(STATUS_ORDER, colors)
    ]
    counts = grid.groupby(["waterbody", "caoa_status"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)

    fig, axis = plt.subplots(figsize=(14.2, 7.7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.17)
    draw_axis(axis, codes, outline_domains=True)
    axis.set_title("Prototype C-AOA v0.1: observation-calibration support", fontsize=16.5, pad=16, color="#263940")
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=3, frameon=False, fontsize=10.5)
    fig.text(
        0.5,
        0.035,
        "Source-equal local environmental analogs + known method-family diversity. Native 0.5° cells; predictions retained.\n"
        "Diagnostic prototype only: size/QA/volume graph and masked-recovery validation are not yet included.",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4B5358",
    )
    combined_png = output_dir / "H23_CAOA_V0_1_GLOBAL_NATURAL_ROBINSON.png"
    fig.savefig(combined_png, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(combined_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13.6, 14.7))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.93, bottom=0.085, hspace=0.13)
    fig.suptitle("Prototype C-AOA v0.1 by natural-water domain", fontsize=17.0, color="#263940")
    for index, (axis, domain) in enumerate(zip(axes, NATURAL_DOMAINS)):
        values = np.ma.masked_where(domains != index, codes)
        draw_axis(axis, values, outline_domains=False)
        row = counts.loc[domain]
        total = int(row.sum())
        axis.set_title(
            f"{DOMAIN_LABELS[domain]} — in {100 * row['C-AOA in'] / total:.1f}% | "
            f"limited {100 * row['C-AOA limited'] / total:.1f}% | out {100 * row['C-AOA out'] / total:.1f}%",
            fontsize=11.3,
            pad=7,
            color="#263940",
        )
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, fontsize=10.5)
    fig.text(
        0.5,
        0.014,
        "C-AOA is calibration support, not concentration and not environmental AOA. All prediction cells are retained.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4B5358",
    )
    panels_png = output_dir / "H23_CAOA_V0_1_THREE_DOMAIN_PANELS.png"
    fig.savefig(panels_png, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(panels_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    plot_audit = {
        "status": "COMPLETED_PROTOTYPE_NOT_RELEASE",
        "grid": str(grid_path),
        "grid_sha256": sha256(grid_path),
        "countries": str(countries_path),
        "projection": "Robinson",
        "renderer": "native_0p5_degree_pcolormesh_flat_no_interpolation",
        "predictions_retained": True,
        "outputs": [str(combined_png), str(combined_png.with_suffix('.pdf')), str(panels_png), str(panels_png.with_suffix('.pdf'))],
    }
    (output_dir / "H23_CAOA_V0_1_PLOT_AUDIT.json").write_text(
        json.dumps(plot_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plot_audit, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compute_parser = subparsers.add_parser("compute", help="Compute the prototype C-AOA grid")
    compute_parser.add_argument("--training", type=Path, required=True)
    compute_parser.add_argument("--prediction", type=Path, required=True)
    compute_parser.add_argument("--observation", type=Path, required=True)
    compute_parser.add_argument("--output-dir", type=Path, required=True)
    compute_parser.set_defaults(func=compute)
    plot_parser = subparsers.add_parser("plot", help="Render the computed prototype C-AOA grid")
    plot_parser.add_argument("--grid", type=Path, required=True)
    plot_parser.add_argument("--countries", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser.add_argument("--dpi", type=int, default=320)
    plot_parser.set_defaults(func=plot)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
