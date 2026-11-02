"""Train H26 marine process BNNs on untouched outer source-space folds.

This is the marine-only successor to the scalar-current ablations.  It accepts
the tables emitted by :mod:`build_h26_cmems_vector_process_features`.  The
primary candidate uses the 2016--2020 CMEMS vector field as a process layer; a
matched scalar-current baseline uses the identical CMEMS-supported records and
folds while changing only that process layer.  No latitude, longitude, regional
label, temperature, or shortwave-radiation covariate is admitted.

The bounded ``h12_background_compatible`` control restores the broad scalar
marine background used by H12 (salinity, current speed, wave height, depth and
surface climate) on the H26 table and validation scheme.  It is a diagnostic
candidate only: H12 predictions, latitude bands and regional labels are never
used as targets or losses.

The two observation arms deliberately share one environmental fit.  Detection
metadata changes only future-observation interval width; the paired q50 fields
are byte-identical by construction.  CMEMS ``source_missing`` and
``outside_physical_support`` grid cells remain present with NA predictions --
they are not zero-filled, interpolated, or hidden.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

import run_h24_simple_protocol_preview as h24
import run_h26_direct_s0_s3 as direct
import run_h26_hydro_residual_booster as hydro
import run_h26_modular_process_bnn as modular


DOMAIN = "marine"
CMEMS_STATUS = "h26_cmems_vector_process_status"
GRAPH_STATUS = "h26_ocean_transport_graph_status"
PERSISTENT_CONVERGENCE_GRAPH_STATUS = "h26_ocean_transport_persistent_convergence_status"
ACTIVE_STATUS = "h26_active_candidate_process_status"

# Direction is represented directly by the eastward/northward velocity
# components.  Magnitude, direction sin/cos, and u/v are deterministic
# re-expressions of one another, so the redundant magnitude/direction fields
# are not admitted to the mean.  This removes the old scalar-speed shortcut
# that generated the equatorial band while retaining seam-free vector flow.
SOURCE_FEATURES = (
    "river_to_sea_microplastic_output_kg_year",
    "ais_activity_density",
    "marine_aquaculture_pressure_tonnes_50km",
    "marina_pressure_h100_kg_year",
)
TRANSPORT_FEATURES = (
    "h26_cmems_u_ms_mean",
    "h26_cmems_v_ms_mean",
    "h26_cmems_eke_m2s2",
    "h26_cmems_positive_divergence_sinv",
    "h26_cmems_retention_days_per_100km",
    "m6_significant_wave_height_m",
    "positive_water_depth_m",
)
RETENTION_FEATURES = (
    "h26_cmems_positive_convergence_sinv",
    "h26_cmems_vorticity_sinv",
    "h26_cmems_mld_m_mean",
    "h26_cmems_mld_m_time_sd",
    "m6_sea_surface_salinity_psu",
)
# Provider spread is retained for support auditing only.  Letting data-quality
# uncertainty drive the environmental mean would reintroduce a support map as
# a concentration predictor.
PROVIDER_DIAGNOSTIC_FEATURES = (
    "h26_cmems_u_provider_sd_mean",
    "h26_cmems_v_provider_sd_mean",
    "h26_cmems_vector_provider_sd_mean",
    "h26_cmems_mld_provider_sd_mean",
)
VECTOR_BRANCHES = {
    "source": SOURCE_FEATURES,
    "transport": TRANSPORT_FEATURES,
    "retention": RETENTION_FEATURES,
}
DERIVED_DYNAMICS_BRANCHES = {
    "source": SOURCE_FEATURES,
    "transport": (
        "h26_cmems_eke_m2s2",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_retention_days_per_100km",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_abs_vorticity_sinv",
        "h26_cmems_mld_m_mean",
        "h26_cmems_mld_m_time_sd",
        "m6_sea_surface_salinity_psu",
    ),
}
PHYSICS_CONSTRAINED_BRANCHES = {
    "source": SOURCE_FEATURES,
    "transport": (
        "h26_cmems_eke_m2s2",
        "h26_cmems_abs_vorticity_sinv",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
        "h26_cmems_mld_m_time_sd",
        "m6_sea_surface_salinity_psu",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_retention_days_per_100km",
        "h26_cmems_mld_m_mean",
    ),
}
# Diagnostics on the unconstrained candidates showed that EKE and temporal MLD
# variability alone created an excessive equatorial low band.  This minimal
# process set retains source delivery and directionally interpretable transport
# / retention terms without using those unstable single-feature shortcuts.
MINIMAL_PHYSICS_BRANCHES = {
    "source": SOURCE_FEATURES,
    "transport": (
        "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_retention_days_per_100km",
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_FEATURES = (
    "h26_ocean_transport_accumulation_p80_log1p",
    "h26_ocean_transport_accumulation_p92_log1p",
    "h26_ocean_transport_accumulation_p98_log1p",
)
OCEAN_SOURCE_FORCING_SCORE = "h26_ocean_source_forcing_score_log1p"
OCEAN_MEDIUM_PERSISTENCE_ENRICHMENT = "h26_ocean_transport_persistence_enrichment_p92_p80_logratio"
OCEAN_PERSISTENCE_ENRICHMENT = "h26_ocean_transport_persistence_enrichment_p98_p92_logratio"
OCEAN_EXTERNAL_IMPORT = "h26_ocean_transport_external_import_p80_log1p"
OCEAN_EXTERNAL_IMPORT_FRACTION = "h26_ocean_transport_external_import_fraction_p80"
OCEAN_PERSISTENT_CONVERGENCE = "h26_cmems_persistent_convergence_sinv"
TRANSPORT_GRAPH_BRANCHES = {
    "source": SOURCE_FEATURES,
    "transport": (
        *TRANSPORT_GRAPH_FEATURES,
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_mld_m_mean",
    ),
}
# The core graph arm avoids duplicating the river source already injected into
# the graph and drops wave height, whose sparse legacy coverage otherwise turns
# valid CMEMS ocean cells into model-missing cells.  It changes support only;
# no response or regional diagnostic is used to choose these fields.
TRANSPORT_GRAPH_CORE_BRANCHES = {
    "source": (
        "ais_activity_density",
        "marine_aquaculture_pressure_tonnes_50km",
        "marina_pressure_h100_kg_year",
    ),
    "transport": (
        *TRANSPORT_GRAPH_FEATURES,
        "positive_water_depth_m",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_SOURCE_CORE_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        *TRANSPORT_GRAPH_FEATURES,
        "positive_water_depth_m",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_PERSISTENCE_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        "h26_ocean_transport_accumulation_p80_log1p",
        "positive_water_depth_m",
    ),
    "retention": (
        OCEAN_PERSISTENCE_ENRICHMENT,
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_EXTERNAL_IMPORT_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        OCEAN_EXTERNAL_IMPORT,
        OCEAN_EXTERNAL_IMPORT_FRACTION,
        "positive_water_depth_m",
    ),
    "retention": (
        OCEAN_PERSISTENCE_ENRICHMENT,
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        OCEAN_EXTERNAL_IMPORT_FRACTION,
        "positive_water_depth_m",
    ),
    "retention": (
        OCEAN_PERSISTENCE_ENRICHMENT,
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_positive_divergence_sinv",
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_SIGNED_NO_CONVERGENCE_BRANCHES = {
    "source": TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES["source"],
    "transport": TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES["transport"],
    "retention": (
        OCEAN_PERSISTENCE_ENRICHMENT,
        "h26_cmems_mld_m_mean",
    ),
}
TRANSPORT_GRAPH_SIGNED_PERSISTENT_CONVERGENCE_BRANCHES = {
    "source": TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES["source"],
    "transport": TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES["transport"],
    "retention": (
        OCEAN_PERSISTENCE_ENRICHMENT,
        OCEAN_PERSISTENT_CONVERGENCE,
        "h26_cmems_mld_m_mean",
    ),
}
# Single-variable transport challenger: use absolute advected accumulation
# rather than the imported *fraction*.  The fraction can be high where local
# source forcing is tiny and therefore create a strong equatorial contrast
# without implying a high accumulated exposure.  No response or latitude band
# is used in this response-free feature substitution.
TRANSPORT_GRAPH_SHARED_ACCUMULATION_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        "h26_ocean_transport_accumulation_p80_log1p",
        "positive_water_depth_m",
    ),
    "retention": TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES["retention"],
}
# Final bounded path-process challenger.  Each component is response-free and
# comes from the same directed multi-step ocean graph: local forcing, material
# delivered from non-self graph edges, and the additional accumulation gained
# under long persistence.  No local convergence/divergence, coordinate, region
# label or duplicate p80/p92/p98 accumulation fields enter this candidate.
TRANSPORT_PATH_POTENTIAL_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (OCEAN_EXTERNAL_IMPORT,),
    "retention": (
        OCEAN_MEDIUM_PERSISTENCE_ENRICHMENT,
        OCEAN_PERSISTENCE_ENRICHMENT,
    ),
}
PHYSICS_ORIENTATION = {
    "h26_cmems_positive_convergence_sinv": 1.0,
    "h26_cmems_positive_divergence_sinv": -1.0,
    "h26_cmems_retention_days_per_100km": 1.0,
    "h26_cmems_mld_m_mean": -1.0,
    "positive_water_depth_m": -1.0,
    **{name: 1.0 for name in TRANSPORT_GRAPH_FEATURES},
    OCEAN_SOURCE_FORCING_SCORE: 1.0,
    OCEAN_MEDIUM_PERSISTENCE_ENRICHMENT: 1.0,
    OCEAN_PERSISTENCE_ENRICHMENT: 1.0,
    OCEAN_EXTERNAL_IMPORT: 1.0,
    OCEAN_EXTERNAL_IMPORT_FRACTION: 1.0,
    OCEAN_PERSISTENT_CONVERGENCE: 1.0,
}
SCALAR_SAME_SUPPORT_BRANCHES = {
    "source": SOURCE_FEATURES,
    "transport": (
        "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
    ),
    "retention": ("m6_sea_surface_salinity_psu",),
}
# One bounded post-three-arm control requested for mechanism diagnosis.  It
# restores the H12-era broad marine background while retaining the H26 source
# forcing, observation layer and validation scheme.  Convergence/divergence is
# deliberately absent because the three-arm audit showed that clipped positive
# divergence, not a missing broad background, generated the razor-thin equator
# band.  No coordinate or regional answer enters this candidate.
H12_BACKGROUND_COMPATIBLE_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
    ),
    "retention": ("m6_sea_surface_salinity_psu",),
    "background": (
        "m6_global_annual_mean_2m_air_temperature_c_2020",
        "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
    ),
}
# Final bounded hybrid after the broad-background control removed the narrow
# equator artefact but showed fold-wise nonlinear branch swapping.  This field
# set combines H12-era background variables with persistent H26 transport in a
# single low-capacity Bayesian linear layer.  H12 predictions and region labels
# are never used in fitting.
H12_H26_HYBRID_SHARED_LINEAR_BRANCHES = {
    "source": (OCEAN_SOURCE_FORCING_SCORE,),
    "transport": (
        "h26_ocean_transport_accumulation_p80_log1p",
        "m6_surface_current_speed_ms",
        "m6_significant_wave_height_m",
        "positive_water_depth_m",
    ),
    "retention": (
        OCEAN_PERSISTENCE_ENRICHMENT,
        OCEAN_PERSISTENT_CONVERGENCE,
        "m6_sea_surface_salinity_psu",
    ),
    "background": (
        "m6_global_annual_mean_2m_air_temperature_c_2020",
        "m6_global_wind10m_ms_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
    ),
}
DERIVED_FEATURES = {
    "h26_cmems_positive_convergence_sinv",
    "h26_cmems_positive_divergence_sinv",
}
RAW_CMEMS_MODEL_FEATURES = (
    "h26_cmems_u_ms_mean",
    "h26_cmems_v_ms_mean",
    "h26_cmems_eke_m2s2",
    "h26_cmems_convergence_sinv",
    "h26_cmems_vorticity_sinv",
    "h26_cmems_abs_vorticity_sinv",
    "h26_cmems_mld_m_mean",
    "h26_cmems_mld_m_time_sd",
    "h26_cmems_retention_days_per_100km",
)
CMEMS_REQUIRED_FINITE = tuple(RAW_CMEMS_MODEL_FEATURES)
CMEMS_SUPPORT_COLUMNS = (
    "h26_cmems_source_cells_n",
    "h26_cmems_native_coverage_fraction",
)
BRIDGE_NODE_COLUMNS = (
    "waterbody", "source_group", "source_rows", "component_size", "degree",
    "weighted_degree", "mean_effective_resistance", "max_effective_resistance",
    "identifiability_class",
)
BRIDGE_PREFIX = "h26_bridge_"


def branches_for_candidate(candidate: str) -> dict[str, tuple[str, ...]]:
    if candidate == "vector_process":
        return VECTOR_BRANCHES
    if candidate == "derived_dynamics":
        return DERIVED_DYNAMICS_BRANCHES
    if candidate == "physics_constrained":
        return PHYSICS_CONSTRAINED_BRANCHES
    if candidate == "minimal_physics":
        return MINIMAL_PHYSICS_BRANCHES
    if candidate == "transport_graph":
        return TRANSPORT_GRAPH_BRANCHES
    if candidate == "transport_graph_core":
        return TRANSPORT_GRAPH_CORE_BRANCHES
    if candidate == "transport_graph_source_core":
        return TRANSPORT_GRAPH_SOURCE_CORE_BRANCHES
    if candidate == "transport_graph_source_contrast":
        return TRANSPORT_GRAPH_SOURCE_CORE_BRANCHES
    if candidate == "transport_graph_source_contrast_depth_constrained":
        return TRANSPORT_GRAPH_SOURCE_CORE_BRANCHES
    if candidate == "transport_graph_source_contrast_persistence":
        return TRANSPORT_GRAPH_PERSISTENCE_BRANCHES
    if candidate == "transport_graph_source_contrast_external_import":
        return TRANSPORT_GRAPH_EXTERNAL_IMPORT_BRANCHES
    if candidate == "transport_graph_signed_process":
        return TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES
    if candidate == "transport_graph_signed_no_convergence_process":
        return TRANSPORT_GRAPH_SIGNED_NO_CONVERGENCE_BRANCHES
    if candidate == "transport_graph_signed_persistent_convergence_process":
        return TRANSPORT_GRAPH_SIGNED_PERSISTENT_CONVERGENCE_BRANCHES
    if candidate == "transport_graph_shared_signed_process":
        return TRANSPORT_GRAPH_SIGNED_PROCESS_BRANCHES
    if candidate == "transport_graph_shared_signed_accumulation_process":
        return TRANSPORT_GRAPH_SHARED_ACCUMULATION_BRANCHES
    if candidate == "transport_path_potential_shared_directional":
        return TRANSPORT_PATH_POTENTIAL_BRANCHES
    if candidate == "scalar_same_support":
        return SCALAR_SAME_SUPPORT_BRANCHES
    if candidate == "h12_background_compatible":
        return H12_BACKGROUND_COMPATIBLE_BRANCHES
    if candidate == "h12_background_shared_linear":
        return H12_BACKGROUND_COMPATIBLE_BRANCHES
    if candidate == "h12_h26_hybrid_shared_linear":
        return H12_H26_HYBRID_SHARED_LINEAR_BRANCHES
    raise RuntimeError(f"Unknown marine candidate: {candidate}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def available_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load marine observations and all native grid cells without proxy filling."""
    train_header = available_columns(args.training)
    grid_header = available_columns(args.prediction)
    active_status = (
        PERSISTENT_CONVERGENCE_GRAPH_STATUS
        if args.candidate in {
            "transport_graph_signed_persistent_convergence_process",
            "h12_h26_hybrid_shared_linear",
        }
        else GRAPH_STATUS
        if args.candidate in {
            "transport_graph", "transport_graph_core", "transport_graph_source_core",
            "transport_graph_source_contrast", "transport_graph_source_contrast_depth_constrained",
            "transport_graph_source_contrast_persistence",
            "transport_graph_source_contrast_external_import",
            "transport_graph_signed_process",
            "transport_graph_signed_no_convergence_process",
            "transport_graph_signed_persistent_convergence_process",
            "transport_graph_shared_signed_process",
            "transport_graph_shared_signed_accumulation_process",
            "transport_path_potential_shared_directional",
            "h12_background_compatible",
            "h12_background_shared_linear",
        }
        else CMEMS_STATUS
    )
    required_train = {
        "stable_record_id", "longitude", "latitude",
        CMEMS_STATUS, active_status, *CMEMS_SUPPORT_COLUMNS,
    }
    required_grid = {
        "prediction_id", "longitude", "latitude", "prediction_domain",
        CMEMS_STATUS, active_status, *CMEMS_SUPPORT_COLUMNS,
    }
    require(required_train.issubset(train_header), f"CMEMS training table missing: {sorted(required_train - train_header)}")
    require(required_grid.issubset(grid_header), f"CMEMS prediction table missing: {sorted(required_grid - grid_header)}")
    branches = branches_for_candidate(args.candidate)
    requested = tuple(name for names in branches.values() for name in names)
    nonderived_requested = tuple(name for name in requested if name not in DERIVED_FEATURES)
    raw_required = set(nonderived_requested) | set(RAW_CMEMS_MODEL_FEATURES)
    missing_core = sorted(name for name in raw_required if name not in train_header or name not in grid_header)
    require(not missing_core, f"CMEMS vector tables are missing required model covariates: {missing_core}")
    optional_provider = tuple(
        name for name in PROVIDER_DIAGNOSTIC_FEATURES if name in train_header and name in grid_header
    )
    shared = tuple(dict.fromkeys((*nonderived_requested, *RAW_CMEMS_MODEL_FEATURES, *optional_provider)))
    train_usecols = list(dict.fromkeys([
        "stable_record_id", "longitude", "latitude",
        CMEMS_STATUS, active_status, *CMEMS_SUPPORT_COLUMNS, *shared,
    ]))
    grid_usecols = list(dict.fromkeys([
        "prediction_id", "longitude", "latitude", "prediction_domain",
        CMEMS_STATUS, active_status, *CMEMS_SUPPORT_COLUMNS, *shared,
    ]))
    training = pd.read_csv(args.training, usecols=train_usecols, low_memory=False)
    grid = pd.read_csv(args.prediction, usecols=grid_usecols, low_memory=False)
    for frame in (training, grid):
        convergence = numeric(frame["h26_cmems_convergence_sinv"])
        frame["h26_cmems_positive_convergence_sinv"] = np.maximum(convergence, 0.0).astype(np.float32)
        frame["h26_cmems_positive_divergence_sinv"] = np.maximum(-convergence, 0.0).astype(np.float32)

    observation = pd.read_csv(args.observation, low_memory=False)
    observation_columns = (
        "stable_record_id", "waterbody", "source_group", "observed_log10",
        "sizecorr_known_sd_log10", "unit_sd_log10", "sampling_year_effective_v1_3",
        "detection_method", "qaqc_total_score", "sizecorr_lower_used_um",
        "sizecorr_evidence_tier",
    )
    required_observation = {"stable_record_id", "waterbody", "source_group", "observed_log10"}
    require(required_observation.issubset(observation.columns),
            f"Observation table missing: {sorted(required_observation - set(observation.columns))}")
    observation = observation[[name for name in observation_columns if name in observation]].copy()
    records_all = training.merge(observation, on="stable_record_id", how="inner", validate="one_to_one")
    records_all = records_all.loc[records_all["waterbody"].eq(DOMAIN)].reset_index(drop=True)
    bridge_audit: dict[str, object] = {"used": False}
    if args.observation_scale_model in {"bridge_mem", "bridge_mem_fixed_reported"}:
        require(args.bridge_nodes is not None, f"{args.observation_scale_model} requires --bridge-nodes")
        bridge_nodes = pd.read_csv(args.bridge_nodes, usecols=list(BRIDGE_NODE_COLUMNS), low_memory=False)
        bridge_nodes = bridge_nodes.loc[bridge_nodes["waterbody"].eq(DOMAIN)].copy()
        require(not bridge_nodes.duplicated(["waterbody", "source_group"]).any(),
                "Bridge node table has duplicate marine sources")
        renamed = {
            name: f"{BRIDGE_PREFIX}{name}"
            for name in BRIDGE_NODE_COLUMNS
            if name not in {"waterbody", "source_group"}
        }
        bridge_nodes = bridge_nodes.rename(columns=renamed)
        records_all = records_all.merge(
            bridge_nodes, on=["waterbody", "source_group"], how="left", validate="many_to_one"
        )
        matched = records_all[f"{BRIDGE_PREFIX}identifiability_class"].notna()
        require(bool(matched.all()),
                f"Bridge nodes failed to match {int((~matched).sum())} marine records")
        bridge_audit = {
            "used": True,
            "path": str(args.bridge_nodes),
            "response_fields_loaded": False,
            "mean_correction_permitted": False,
            "matched_records": int(matched.sum()),
            "identifiability_counts": {
                str(key): int(value)
                for key, value in records_all[f"{BRIDGE_PREFIX}identifiability_class"].value_counts().items()
            },
        }
    require(len(records_all) > 500, f"Too few marine observations before CMEMS status filter: {len(records_all):,}")
    grid = grid.loc[grid["prediction_domain"].eq(DOMAIN)].rename(columns={"prediction_domain": "waterbody"}).reset_index(drop=True)
    require(len(grid) > 0, "No marine prediction cells")
    caoa = pd.read_csv(args.caoa, usecols=["prediction_id", "caoa_status"], low_memory=False)
    grid = grid.merge(caoa, on="prediction_id", how="left", validate="one_to_one")
    require(grid["caoa_status"].notna().all(), "C-AOA join did not close")

    records_native_observed = records_all[active_status].astype(str).eq("observed").to_numpy()
    records_requested_complete = np.isfinite(np.column_stack([
        numeric(records_all[name]) for name in requested
    ])).all(axis=1)
    records_all[ACTIVE_STATUS] = records_all[active_status].astype(str)
    records_all.loc[records_native_observed & ~records_requested_complete, ACTIVE_STATUS] = "source_missing"
    observed_status = records_all[ACTIVE_STATUS].astype(str).eq("observed").to_numpy()
    records = records_all.loc[observed_status].reset_index(drop=True)
    require(len(records) > 500, f"Too few marine records with observed CMEMS vectors: {len(records):,}")
    # Observed status is a physical data-support contract.  Do not silently
    # median-fill a vector field that the feature builder declared unavailable.
    cmems_missing = {name: int((~np.isfinite(numeric(records[name]))).sum()) for name in CMEMS_REQUIRED_FINITE}
    nonzero_missing = {name: count for name, count in cmems_missing.items() if count}
    require(not nonzero_missing, f"Observed CMEMS training rows still have missing vector fields: {nonzero_missing}")
    grid_native_observed = grid[active_status].astype(str).eq("observed").to_numpy()
    grid_requested_complete = np.isfinite(np.column_stack([
        numeric(grid[name]) for name in requested
    ])).all(axis=1)
    grid[ACTIVE_STATUS] = grid[active_status].astype(str)
    grid.loc[grid_native_observed & ~grid_requested_complete, ACTIVE_STATUS] = "source_missing"
    grid_observed = grid[ACTIVE_STATUS].astype(str).eq("observed").to_numpy()
    grid_cmems_missing = {
        name: int((~np.isfinite(numeric(grid.loc[grid_observed, name]))).sum())
        for name in CMEMS_REQUIRED_FINITE
    }
    grid_nonzero_missing = {name: count for name, count in grid_cmems_missing.items() if count}
    require(not grid_nonzero_missing, f"Observed CMEMS grid rows still have missing vector fields: {grid_nonzero_missing}")
    feature_missing_training = {
        name: int((~np.isfinite(numeric(records[name]))).sum()) for name in requested
    }
    feature_missing_grid_observed = {
        name: int((~np.isfinite(numeric(grid.loc[grid_observed, name]))).sum()) for name in requested
    }
    require(not any(feature_missing_training.values()),
            f"Candidate-supported training rows have missing requested covariates: {feature_missing_training}")
    require(not any(feature_missing_grid_observed.values()),
            f"Candidate-supported grid rows have missing requested covariates: {feature_missing_grid_observed}")
    return records, grid, {
        "requested_features": list(requested),
        "candidate": args.candidate,
        "active_process_status": active_status,
        "candidate_complete_process_status": ACTIVE_STATUS,
        "available_raw_features": list(shared),
        "derived_features": sorted(DERIVED_FEATURES),
        "vector_representation": (
            "H12-compatible scalar current-speed background control"
            if args.candidate == "h12_background_compatible"
            else "H12-compatible broad background in one shared Bayesian linear layer"
            if args.candidate == "h12_background_shared_linear"
            else "H12-compatible broad background plus persistent H26 transport in one shared Bayesian linear layer"
            if args.candidate == "h12_h26_hybrid_shared_linear"
            else "eastward/northward velocity components; redundant scalar vector magnitude and direction sin/cos excluded from mean"
        ),
        "provider_spread_policy": "support diagnostic only; never an environmental mean covariate",
        "available_provider_diagnostics": list(optional_provider),
        "feature_missing_training_after_cmems_filter": feature_missing_training,
        "feature_missing_grid_observed_cmems": feature_missing_grid_observed,
        "records_before_cmems_status_filter": int(len(records_all)),
        "records_native_process_observed": int(records_native_observed.sum()),
        "records_excluded_requested_feature_missing": int(
            (records_native_observed & ~records_requested_complete).sum()
        ),
        "records_observed_cmems": int(records_all[CMEMS_STATUS].astype(str).eq("observed").sum()),
        "records_excluded_cmems_status": int(
            (~records_all[CMEMS_STATUS].astype(str).eq("observed")).sum()
        ),
        "records_candidate_supported": int(len(records)),
        "records_excluded_candidate_status": int(len(records_all) - len(records)),
        "grid_total_rows": int(len(grid)),
        "grid_observed_cmems_rows": int(grid[CMEMS_STATUS].astype(str).eq("observed").sum()),
        "grid_candidate_supported_rows": int(grid_observed.sum()),
        "grid_status_counts": {str(key): int(value) for key, value in grid[CMEMS_STATUS].value_counts(dropna=False).items()},
        "grid_native_process_status_counts": {
            str(key): int(value) for key, value in grid[active_status].value_counts(dropna=False).items()
        },
        "grid_active_status_counts": {
            str(key): int(value) for key, value in grid[ACTIVE_STATUS].value_counts(dropna=False).items()
        },
        "source_bridge": bridge_audit,
    }


def make_branch_indices(
    names: tuple[str, ...],
    branches: dict[str, tuple[str, ...]],
) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for branch, features in branches.items():
        local = [index for index, name in enumerate(names) if name in features]
        if local:
            indices[branch] = local
    require({"source", "transport", "retention"}.issubset(indices), f"Missing required active branch: {indices}")
    return indices


def select_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        require(torch.cuda.is_available(), f"CUDA device requested but unavailable: {requested}")
        torch.cuda.set_device(device)
    return device


class MonotoneBayesianBranch(nn.Module):
    """Bayesian additive branch with globally signed feature effects.

    Positive latent slopes are multiplied by fixed physical orientations.  The
    outer tanh is monotone, so the requested effect direction is preserved over
    the full standardized feature range instead of being encouraged only at
    selected grid cells.
    """

    def __init__(
        self,
        input_dim: int,
        orientations: list[float],
        cap_standardized: float,
        prior_sd: float = 0.55,
    ) -> None:
        super().__init__()
        require(len(orientations) == input_dim, "Physics orientations do not match retention inputs")
        self.weight_mu = nn.Parameter(torch.full((input_dim,), -1.8))
        self.weight_rho = nn.Parameter(torch.full((input_dim,), -4.5))
        self.bias_mu = nn.Parameter(torch.zeros(()))
        self.bias_rho = nn.Parameter(torch.full((), -4.5))
        self.register_buffer("orientations", torch.as_tensor(orientations, dtype=torch.float32))
        self.cap_standardized = float(cap_standardized)
        self.prior_sd = float(prior_sd)

    @staticmethod
    def sd(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        raw_weight = self.weight_mu
        bias = self.bias_mu
        if sample:
            raw_weight = raw_weight + self.sd(self.weight_rho) * torch.randn_like(raw_weight)
            bias = bias + self.sd(self.bias_rho) * torch.randn_like(bias)
        signed_weight = F.softplus(raw_weight) * self.orientations
        raw = F.linear(values, signed_weight.reshape(1, -1), bias.reshape(1)).reshape(-1)
        cap = self.cap_standardized
        return cap * torch.tanh(raw / cap)

    def kl(self) -> torch.Tensor:
        prior_var = self.prior_sd**2
        total = torch.zeros((), device=self.weight_mu.device)
        for mean, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sd = self.sd(rho)
            total = total + 0.5 * (
                ((sd.square() + mean.square()) / prior_var)
                - 1.0
                + 2.0 * (math.log(self.prior_sd) - torch.log(sd))
            ).sum()
        return total


class PhysicsConstrainedMarineBNN(nn.Module):
    """Modular marine BNN with a signed monotone physical-retention branch."""

    def __init__(
        self,
        branch_indices: dict[str, list[int]],
        physics_signs: list[float],
        target_scale: float,
        total_cap_log10: float,
    ) -> None:
        super().__init__()
        self.branch_order = tuple(
            name for name in ("source", "transport", "retention", "background") if branch_indices.get(name)
        )
        self.branches = nn.ModuleDict()
        for name in self.branch_order:
            self.register_buffer(f"indices_{name}", torch.as_tensor(branch_indices[name], dtype=torch.long))
            if name == "retention":
                self.branches[name] = MonotoneBayesianBranch(
                    len(branch_indices[name]), physics_signs, modular.BRANCH_CAP_LOG10[name] / target_scale
                )
            else:
                self.branches[name] = modular.BranchBNN(
                    len(branch_indices[name]), modular.BRANCH_CAP_LOG10[name] / target_scale
                )
        self.total_cap_standardized = float(total_cap_log10 / target_scale)

    def components(self, values: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        return {
            name: self.branches[name](values.index_select(1, getattr(self, f"indices_{name}")), sample)
            for name in self.branch_order
        }

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        parts = self.components(values, sample)
        raw_total = torch.stack(list(parts.values()), dim=0).sum(dim=0)
        cap = self.total_cap_standardized
        return cap * torch.tanh(raw_total / cap)

    def kl(self) -> torch.Tensor:
        return sum(branch.kl() for branch in self.branches.values())


class FullyDirectionalMarineBNN(nn.Module):
    """Modular BNN with signed monotone source, transport and retention."""

    def __init__(
        self,
        branch_indices: dict[str, list[int]],
        branch_signs: dict[str, list[float]],
        target_scale: float,
        total_cap_log10: float,
    ) -> None:
        super().__init__()
        self.branch_order = tuple(
            name for name in ("source", "transport", "retention", "background")
            if branch_indices.get(name)
        )
        self.branches = nn.ModuleDict()
        for name in self.branch_order:
            self.register_buffer(
                f"indices_{name}", torch.as_tensor(branch_indices[name], dtype=torch.long)
            )
            if name in branch_signs:
                require(
                    len(branch_signs[name]) == len(branch_indices[name]),
                    f"Directional signs do not align with marine branch {name}",
                )
                self.branches[name] = MonotoneBayesianBranch(
                    len(branch_indices[name]),
                    branch_signs[name],
                    modular.BRANCH_CAP_LOG10[name] / target_scale,
                )
            else:
                self.branches[name] = modular.BranchBNN(
                    len(branch_indices[name]), modular.BRANCH_CAP_LOG10[name] / target_scale
                )
        self.total_cap_standardized = float(total_cap_log10 / target_scale)

    def components(self, values: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        return {
            name: self.branches[name](
                values.index_select(1, getattr(self, f"indices_{name}")), sample
            )
            for name in self.branch_order
        }

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        parts = self.components(values, sample)
        raw_total = torch.stack(list(parts.values()), dim=0).sum(dim=0)
        cap = self.total_cap_standardized
        return cap * torch.tanh(raw_total / cap)

    def kl(self) -> torch.Tensor:
        return sum(branch.kl() for branch in self.branches.values())


class SharedDirectionalMarineBNN(nn.Module):
    """One signed Bayesian additive layer shared by all process components.

    The process variables and their physical orientations are identical to the
    existing fully directional candidate.  The only change is identifiability:
    source, transport and retention no longer have separate nonlinear links or
    separate intercepts that can compensate one another.  Component outputs
    remain available by summing the feature-level contributions assigned to
    each process branch; a single global intercept is divided equally between
    them so their sum is the exact pre-link model response.
    """

    def __init__(
        self,
        branch_indices: dict[str, list[int]],
        branch_signs: dict[str, list[float]],
        target_scale: float,
        total_cap_log10: float,
        prior_sd: float = 0.55,
    ) -> None:
        super().__init__()
        self.branch_order = tuple(
            name for name in ("source", "transport", "retention", "background")
            if branch_indices.get(name)
        )
        orientations: list[float] = []
        offset = 0
        for name in self.branch_order:
            require(name in branch_signs, f"Shared directional branch lacks signs: {name}")
            require(
                len(branch_signs[name]) == len(branch_indices[name]),
                f"Shared directional signs do not align with marine branch {name}",
            )
            self.register_buffer(
                f"indices_{name}", torch.as_tensor(branch_indices[name], dtype=torch.long)
            )
            self.register_buffer(
                f"positions_{name}",
                torch.arange(offset, offset + len(branch_indices[name]), dtype=torch.long),
            )
            orientations.extend(branch_signs[name])
            offset += len(branch_indices[name])
        require(offset > 0, "Shared directional marine model has no active features")
        self.weight_mu = nn.Parameter(torch.full((offset,), -1.8))
        self.weight_rho = nn.Parameter(torch.full((offset,), -4.5))
        self.bias_mu = nn.Parameter(torch.zeros(()))
        self.bias_rho = nn.Parameter(torch.full((), -4.5))
        self.register_buffer("orientations", torch.as_tensor(orientations, dtype=torch.float32))
        self.prior_sd = float(prior_sd)
        self.total_cap_standardized = float(total_cap_log10 / target_scale)

    @staticmethod
    def sd(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def _draw_parameters(self, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        raw_weight = self.weight_mu
        bias = self.bias_mu
        if sample:
            raw_weight = raw_weight + self.sd(self.weight_rho) * torch.randn_like(raw_weight)
            bias = bias + self.sd(self.bias_rho) * torch.randn_like(bias)
        return F.softplus(raw_weight) * self.orientations, bias

    def components(self, values: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        signed_weight, bias = self._draw_parameters(sample)
        bias_share = bias / float(len(self.branch_order))
        return {
            name: F.linear(
                values.index_select(1, getattr(self, f"indices_{name}")),
                signed_weight.index_select(0, getattr(self, f"positions_{name}")).reshape(1, -1),
            ).reshape(-1) + bias_share
            for name in self.branch_order
        }

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        raw_total = torch.stack(list(self.components(values, sample).values()), dim=0).sum(dim=0)
        cap = self.total_cap_standardized
        return cap * torch.tanh(raw_total / cap)

    def kl(self) -> torch.Tensor:
        prior_var = self.prior_sd**2
        total = torch.zeros((), device=self.weight_mu.device)
        for mean, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sd = self.sd(rho)
            total = total + 0.5 * (
                ((sd.square() + mean.square()) / prior_var)
                - 1.0
                + 2.0 * (math.log(self.prior_sd) - torch.log(sd))
            ).sum()
        return total


class SharedLinearMarineBNN(nn.Module):
    """One low-capacity unconstrained Bayesian linear layer over all branches.

    A single coefficient per standardized feature prevents separate nonlinear
    branch shapes and intercepts from swapping across folds.  Coefficient signs
    are learned from the observations; no latitude band or historical map is a
    training target.  Component outputs exactly sum to the pre-link response.
    """

    def __init__(
        self,
        branch_indices: dict[str, list[int]],
        target_scale: float,
        total_cap_log10: float,
        prior_sd: float = 0.35,
    ) -> None:
        super().__init__()
        self.branch_order = tuple(
            name for name in ("source", "transport", "retention", "background")
            if branch_indices.get(name)
        )
        offset = 0
        for name in self.branch_order:
            self.register_buffer(
                f"indices_{name}", torch.as_tensor(branch_indices[name], dtype=torch.long)
            )
            self.register_buffer(
                f"positions_{name}",
                torch.arange(offset, offset + len(branch_indices[name]), dtype=torch.long),
            )
            offset += len(branch_indices[name])
        require(offset > 0, "Shared linear marine model has no active features")
        self.weight_mu = nn.Parameter(torch.zeros(offset))
        self.weight_rho = nn.Parameter(torch.full((offset,), -4.5))
        self.bias_mu = nn.Parameter(torch.zeros(()))
        self.bias_rho = nn.Parameter(torch.full((), -4.5))
        self.prior_sd = float(prior_sd)
        self.total_cap_standardized = float(total_cap_log10 / target_scale)

    @staticmethod
    def sd(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def _draw_parameters(self, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight_mu
        bias = self.bias_mu
        if sample:
            weight = weight + self.sd(self.weight_rho) * torch.randn_like(weight)
            bias = bias + self.sd(self.bias_rho) * torch.randn_like(bias)
        return weight, bias

    def components(self, values: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        weight, bias = self._draw_parameters(sample)
        bias_share = bias / float(len(self.branch_order))
        return {
            name: F.linear(
                values.index_select(1, getattr(self, f"indices_{name}")),
                weight.index_select(0, getattr(self, f"positions_{name}")).reshape(1, -1),
            ).reshape(-1) + bias_share
            for name in self.branch_order
        }

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        raw_total = torch.stack(list(self.components(values, sample).values()), dim=0).sum(dim=0)
        cap = self.total_cap_standardized
        return cap * torch.tanh(raw_total / cap)

    def kl(self) -> torch.Tensor:
        prior_var = self.prior_sd**2
        total = torch.zeros((), device=self.weight_mu.device)
        for mean, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sd = self.sd(rho)
            total = total + 0.5 * (
                ((sd.square() + mean.square()) / prior_var)
                - 1.0
                + 2.0 * (math.log(self.prior_sd) - torch.log(sd))
            ).sum()
        return total


def inner_source_validation_split(
    records: pd.DataFrame,
    outer_train_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Create source-disjoint early stopping rows inside the outer train set.

    The frozen outer fold is never inspected by the optimiser, checkpoint
    selector, target scaler, or residual-scale calibration.
    """
    local = records.iloc[outer_train_rows]
    labels = local["source_group"].fillna("__missing__").astype(str).to_numpy()
    require(len(np.unique(labels)) >= 8, "Too few outer-train sources for inner source validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=int(seed) + 2603)
    fit_local, validation_local = next(splitter.split(np.arange(len(local)), groups=labels))
    fit_rows = outer_train_rows[fit_local].astype(np.int64)
    validation_rows = outer_train_rows[validation_local].astype(np.int64)
    require(len(fit_rows) >= 500 and len(validation_rows) >= 50, "Inner source-validation split is too small")
    fit_sources = set(records.iloc[fit_rows]["source_group"].fillna("__missing__").astype(str))
    validation_sources = set(records.iloc[validation_rows]["source_group"].fillna("__missing__").astype(str))
    require(not (fit_sources & validation_sources), "Inner early-stopping split retains overlapping sources")
    return fit_rows, validation_rows, {
        "kind": "inner_source_group_early_stopping",
        "seed": int(seed) + 2603,
        "fit_records": int(len(fit_rows)),
        "validation_records": int(len(validation_rows)),
        "fit_sources": int(len(fit_sources)),
        "validation_sources": int(len(validation_sources)),
        "source_overlap": 0,
    }


def build_site_event_key(records: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    """Build a response-free site-event key for grouped random validation.

    A real sampling-event identifier is preferred when one exists.  The
    current water database does not expose one consistently, so the audited
    fallback keeps records from the same source, rounded coordinate, and year
    together.  Missing years share one explicit token instead of being split
    into artificial row-unique events.
    """
    event_candidates = (
        "sampling_event_id", "sampling_event_key", "field_sampling_event_id",
    )
    for name in event_candidates:
        if name not in records:
            continue
        event = records[name].fillna("").astype(str).str.strip()
        if event.ne("").all() and event.nunique() >= 2:
            key = records["source_group"].fillna("__MISSING_SOURCE__").astype(str) + "|EVENT|" + event
            return key, {
                "event_key_kind": f"source_group+{name}",
                "event_id_field": name,
                "fallback_used": False,
                "missing_coordinate_rows": 0,
                "missing_year_rows": 0,
            }

    require("longitude" in records and "latitude" in records,
            "random_site_event requires longitude and latitude in the training records")
    longitude = pd.to_numeric(records["longitude"], errors="coerce")
    latitude = pd.to_numeric(records["latitude"], errors="coerce")
    require(longitude.notna().all() and latitude.notna().all(),
            "random_site_event cannot group records with missing coordinates")
    year_field = next((
        name for name in (
            "sampling_year_effective_v1_3", "sampling_year_effective",
            "sampling_year", "year",
        ) if name in records
    ), None)
    if year_field is None:
        year = pd.Series("MISSING_YEAR", index=records.index, dtype=object)
        missing_year_rows = len(records)
    else:
        numeric_year = pd.to_numeric(records[year_field], errors="coerce")
        finite_year = numeric_year.notna()
        year = pd.Series("MISSING_YEAR", index=records.index, dtype=object)
        year.loc[finite_year] = numeric_year.loc[finite_year].round().astype(int).astype(str)
        missing_year_rows = int((~finite_year).sum())
    source = records["source_group"].fillna("__MISSING_SOURCE__").astype(str)
    coordinate = longitude.round(4).astype(str) + "|" + latitude.round(4).astype(str)
    key = source + "|SITE|" + coordinate + "|YEAR|" + year.astype(str)
    return key, {
        "event_key_kind": "source_group+longitude4+latitude4+year",
        "event_id_field": None,
        "year_field": year_field,
        "fallback_used": True,
        "coordinate_round_digits": 4,
        "missing_coordinate_rows": 0,
        "missing_year_rows": int(missing_year_rows),
    }


def choose_outer_split(
    records: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Choose strict, record-random, or site-event-random outer validation.

    The 10-fold schemes are generated randomly once for a given seed and then
    reproduced exactly for every candidate.  They are not geographic blocks.
    Site-event groups are assigned whole to one fold so physical/event repeats
    cannot leak across the outer split while nearby independent sites may.
    """
    scheme = getattr(args, "validation_scheme", "strict_spatial_source_purged")
    strict_schemes = {
        "strict_spatial_block",
        "strict_source_group",
        "strict_spatial_source_purged",
    }
    if scheme in strict_schemes:
        local_domains = records["waterbody"].dropna().astype(str).unique().tolist()
        require(len(local_domains) == 1, f"Strict runner expected one waterbody domain, got {local_domains}")
        local_domain = local_domains[0]
        folds = pd.read_csv(args.folds_csv)
        required = {"stable_record_id", "fold"}
        require(required.issubset(folds.columns), f"Strict folds missing columns: {sorted(required - set(folds.columns))}")
        if "scheme" in folds:
            folds = folds.loc[folds["scheme"].astype(str).eq(scheme)].copy()
        if "waterbody" in folds:
            folds = folds.loc[folds["waterbody"].astype(str).eq(local_domain)].copy()
        require(not folds.empty, f"No fold rows for {local_domain}/{scheme}")
        require(not folds["stable_record_id"].duplicated().any(), f"Duplicate strict fold IDs for {local_domain}/{scheme}")
        assignment = records["stable_record_id"].astype(str).map(
            folds.assign(stable_record_id=folds["stable_record_id"].astype(str)).set_index("stable_record_id")["fold"]
        )
        require(assignment.notna().all(), f"Strict folds do not cover every {local_domain} training record")
        assignment = assignment.astype(int).to_numpy()
        all_rows = np.arange(len(records), dtype=np.int64)
        outer_test = all_rows[assignment == int(args.fold)]
        outer_train = all_rows[assignment != int(args.fold)]
        require(len(outer_test) >= 2, f"Strict fold {args.fold} has too few held-out records")

        test_sources = set(records.iloc[outer_test]["source_group"].fillna("__MISSING_SOURCE__").astype(str))
        if scheme == "strict_spatial_source_purged":
            train_source = records.iloc[outer_train]["source_group"].fillna("__MISSING_SOURCE__").astype(str)
            outer_train = outer_train[~train_source.isin(test_sources).to_numpy()]
        outer_train = np.sort(outer_train.astype(np.int64))
        outer_test = np.sort(outer_test.astype(np.int64))
        require(len(outer_train) >= 500, f"Strict fold {args.fold} leaves too few training records")

        event_key, event_audit = build_site_event_key(records)
        train_sources = set(records.iloc[outer_train]["source_group"].fillna("__MISSING_SOURCE__").astype(str))
        test_sources = set(records.iloc[outer_test]["source_group"].fillna("__MISSING_SOURCE__").astype(str))
        train_events = set(event_key.iloc[outer_train])
        test_events = set(event_key.iloc[outer_test])
        held_source_seen = records.iloc[outer_test]["source_group"].fillna("__MISSING_SOURCE__").astype(str).isin(train_sources)
        held_event_seen = event_key.iloc[outer_test].isin(train_events)
        source_overlap = train_sources & test_sources
        if scheme in {"strict_source_group", "strict_spatial_source_purged"}:
            require(not source_overlap, f"{scheme} retained overlapping sources")

        spatial_block_overlap = None
        if "spatial_block_id" in folds:
            block_lookup = folds.assign(stable_record_id=folds["stable_record_id"].astype(str)).set_index("stable_record_id")["spatial_block_id"]
            record_blocks = records["stable_record_id"].astype(str).map(block_lookup)
            require(record_blocks.notna().all(), f"Strict folds do not map every {local_domain} spatial block")
            train_blocks = set(record_blocks.iloc[outer_train].astype(str))
            test_blocks = set(record_blocks.iloc[outer_test].astype(str))
            spatial_block_overlap = int(len(train_blocks & test_blocks))
            if scheme in {"strict_spatial_block", "strict_spatial_source_purged"}:
                require(spatial_block_overlap == 0, f"{scheme} retained overlapping spatial blocks")

        return outer_train, outer_test, {
            "kind": scheme,
            "scheme": scheme,
            "seed": int(args.seed),
            "requested_fold": int(args.fold),
            "training_records": int(len(outer_train)),
            "heldout_records": int(len(outer_test)),
            "source_overlap": int(len(source_overlap)),
            "spatial_block_overlap": spatial_block_overlap,
            "heldout_records_with_training_source_fraction": float(held_source_seen.mean()),
            "heldout_records_with_training_site_event_fraction": float(held_event_seen.mean()),
            "training_site_event_groups": int(len(train_events)),
            "heldout_site_event_groups": int(len(test_events)),
            "train_test_site_event_overlap": int(len(train_events & test_events)),
            "response_stratification_used": False,
            "folds_csv_used": True,
            "folds_csv": str(args.folds_csv),
            **event_audit,
        }
    require(
        scheme in {
            "random_record", "random_site_event",
            "random_record_10fold", "random_site_event_10fold",
        },
        f"Unknown validation scheme: {scheme}",
    )
    all_rows = np.arange(len(records), dtype=np.int64)
    fold_balance_audit: dict[str, object] = {}
    if scheme in {"random_record", "random_record_10fold"}:
        if "longitude" in records and "latitude" in records:
            event_key, event_audit = build_site_event_key(records)
        else:
            event_key = records["stable_record_id"].astype(str)
            event_audit = {
                "event_key_kind": "not_assessed_missing_coordinates",
                "event_id_field": None,
                "fallback_used": False,
                "missing_coordinate_rows": int(len(records)),
                "missing_year_rows": None,
            }
    else:
        event_key, event_audit = build_site_event_key(records)

    if scheme == "random_record":
        outer_train, outer_test = train_test_split(
            all_rows, test_size=0.20, random_state=int(args.seed), shuffle=True,
        )
    elif scheme == "random_site_event":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=int(args.seed))
        outer_train, outer_test = next(splitter.split(all_rows, groups=event_key.to_numpy()))
    elif scheme == "random_record_10fold":
        require(1 <= int(args.fold) <= 10, f"10-fold validation requires fold 1..10, got {args.fold}")
        splits = list(KFold(n_splits=10, shuffle=True, random_state=int(args.seed)).split(all_rows))
        outer_train, outer_test = splits[int(args.fold) - 1]
        fold_balance_audit = {
            "n_splits": 10,
            "requested_fold": int(args.fold),
            "all_fold_record_counts": [int(len(test)) for _, test in splits],
            "fold_assignment_kind": "shuffled_record_KFold",
        }
    else:
        require(1 <= int(args.fold) <= 10, f"10-fold validation requires fold 1..10, got {args.fold}")
        group_sizes = event_key.value_counts(sort=False, dropna=False)
        require(len(group_sizes) >= 10, f"Too few site-event groups for 10 folds: {len(group_sizes)}")
        rng = np.random.default_rng(int(args.seed))
        groups = group_sizes.index.to_numpy(dtype=object)
        counts = group_sizes.to_numpy(dtype=np.int64)
        random_tie_break = rng.random(len(groups))
        order = np.lexsort((random_tie_break, -counts))
        fold_record_counts = np.zeros(10, dtype=np.int64)
        fold_group_counts = np.zeros(10, dtype=np.int64)
        group_to_fold: dict[object, int] = {}
        for group_index in order:
            candidate_folds = np.flatnonzero(fold_record_counts == fold_record_counts.min())
            selected_fold = int(rng.choice(candidate_folds))
            group_to_fold[groups[group_index]] = selected_fold
            fold_record_counts[selected_fold] += counts[group_index]
            fold_group_counts[selected_fold] += 1
        fold_assignment = event_key.map(group_to_fold)
        require(fold_assignment.notna().all(), "Site-event fold assignment left unmapped rows")
        fold_assignment = fold_assignment.to_numpy(dtype=np.int64)
        requested_fold = int(args.fold) - 1
        outer_test = all_rows[fold_assignment == requested_fold]
        outer_train = all_rows[fold_assignment != requested_fold]
        fold_balance_audit = {
            "n_splits": 10,
            "requested_fold": int(args.fold),
            "all_fold_record_counts": fold_record_counts.astype(int).tolist(),
            "all_fold_site_event_group_counts": fold_group_counts.astype(int).tolist(),
            "fold_assignment_kind": "seeded_size_balanced_whole_site_event",
        }
    outer_train = np.sort(outer_train.astype(np.int64))
    outer_test = np.sort(outer_test.astype(np.int64))
    train_sources = set(records.iloc[outer_train]["source_group"].fillna("__MISSING_SOURCE__").astype(str))
    test_sources = set(records.iloc[outer_test]["source_group"].fillna("__MISSING_SOURCE__").astype(str))
    train_events = set(event_key.iloc[outer_train])
    test_events = set(event_key.iloc[outer_test])
    held_source_seen = records.iloc[outer_test]["source_group"].fillna("__MISSING_SOURCE__").astype(str).isin(train_sources)
    held_event_seen = event_key.iloc[outer_test].isin(train_events)
    if scheme in {"random_site_event", "random_site_event_10fold"}:
        require(not (train_events & test_events), f"{scheme} retained an event-key overlap")
    group_sizes = event_key.value_counts(dropna=False)
    return outer_train, outer_test, {
        "kind": scheme,
        "scheme": scheme,
        "seed": int(args.seed),
        "training_records": int(len(outer_train)),
        "heldout_records": int(len(outer_test)),
        "source_overlap": int(len(train_sources & test_sources)),
        "heldout_records_with_training_source_fraction": float(held_source_seen.mean()),
        "heldout_records_with_training_site_event_fraction": float(held_event_seen.mean()),
        "site_event_groups": int(len(group_sizes)),
        "training_site_event_groups": int(len(train_events)),
        "heldout_site_event_groups": int(len(test_events)),
        "train_test_site_event_overlap": int(len(train_events & test_events)),
        "largest_site_event_group": int(group_sizes.max()),
        "response_stratification_used": False,
        "folds_csv_used": False,
        **fold_balance_audit,
        **event_audit,
    }


def fold_local_aoa(
    training_x: np.ndarray,
    heldout_x: np.ndarray,
    supported_grid_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Annotate fold-local process-space applicability without masking q50."""
    reference = np.unique(np.round(np.asarray(training_x, float), 7), axis=0)
    require(len(reference) >= 50, f"Too few unique process vectors for AOA: {len(reference)}")
    tree = cKDTree(reference)
    nearest_reference = tree.query(reference, k=2, workers=1)[0][:, 1]
    require(np.isfinite(nearest_reference).all(), "Non-finite AOA reference distances")
    threshold = float(np.quantile(nearest_reference, 0.95))
    require(np.isfinite(threshold) and threshold > 0.0, f"Invalid AOA threshold: {threshold}")
    heldout_distance = tree.query(np.asarray(heldout_x, float), k=1, workers=1)[0]
    grid_distance = tree.query(np.asarray(supported_grid_x, float), k=1, workers=1)[0]
    heldout_status = np.where(heldout_distance <= threshold, "AOA_in", "AOA_out")
    grid_status = np.where(grid_distance <= threshold, "AOA_in", "AOA_out")
    return heldout_status, heldout_distance, grid_status, grid_distance, {
        "kind": "fold_local_process_covariate_AOA",
        "response_used": False,
        "candidate_specific": True,
        "training_vectors_before_deduplication": int(len(training_x)),
        "unique_training_vectors": int(len(reference)),
        "process_dimensions": int(reference.shape[1]),
        "caliper": "q95 leave-one-unique-training-vector nearest-neighbour distance",
        "caliper_distance": threshold,
        "heldout_status_counts": {
            str(key): int(value)
            for key, value in pd.Series(heldout_status).value_counts().items()
        },
        "supported_grid_status_counts": {
            str(key): int(value)
            for key, value in pd.Series(grid_status).value_counts().items()
        },
        "q50_or_response_changed": False,
        "AOA_out_rows_retained": True,
    }


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    branch_indices: dict[str, list[int]],
    target_scale: float,
    *,
    candidate: str,
    physics_signs: list[float],
    seed: int,
    epochs: int,
    total_cap_log10: float,
    device: torch.device,
    branch_signs: dict[str, list[float]] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    """Fit the selected modular marine model with explicit CUDA selection."""
    h24.seed_all(seed)
    if candidate == "physics_constrained":
        model = PhysicsConstrainedMarineBNN(
            branch_indices, physics_signs, target_scale, total_cap_log10
        ).to(device)
    elif candidate in {
        "transport_graph_signed_process",
        "transport_graph_signed_no_convergence_process",
        "transport_graph_signed_persistent_convergence_process",
    }:
        require(bool(branch_signs), "Signed graph process candidate requires branch signs")
        model = FullyDirectionalMarineBNN(
            branch_indices, branch_signs or {}, target_scale, total_cap_log10
        ).to(device)
    elif candidate in {
        "transport_graph_shared_signed_process",
        "transport_graph_shared_signed_accumulation_process",
        "transport_path_potential_shared_directional",
    }:
        require(bool(branch_signs), "Shared signed graph candidate requires branch signs")
        model = SharedDirectionalMarineBNN(
            branch_indices, branch_signs or {}, target_scale, total_cap_log10
        ).to(device)
    elif candidate in {"h12_background_shared_linear", "h12_h26_hybrid_shared_linear"}:
        model = SharedLinearMarineBNN(
            branch_indices, target_scale, total_cap_log10
        ).to(device)
    else:
        model = modular.ModularBNN(branch_indices, target_scale, total_cap_log10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0022, weight_decay=2e-5)
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor(y.astype(np.float32), device=device)
    fit_weight = (weights / np.mean(weights)).astype(np.float32)
    w_tensor = torch.as_tensor(fit_weight, device=device)
    train_tensor = torch.as_tensor(train_rows, device=device)
    validation_tensor = torch.as_tensor(validation_rows, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 1709)
    batch_size = min(4096, len(train_rows))
    best_epoch, best_rmse = 0, math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 24
    for epoch in range(int(epochs)):
        model.train()
        order = train_tensor[torch.randperm(len(train_tensor), generator=generator, device=device)]
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            parts = model.components(x_tensor[batch], sample=True)
            raw_total = torch.stack(list(parts.values()), dim=0).sum(dim=0)
            prediction = model.total_cap_standardized * torch.tanh(raw_total / model.total_cap_standardized)
            point = F.smooth_l1_loss(prediction, y_tensor[batch], reduction="none", beta=0.45)
            data_loss = (point * w_tensor[batch]).sum() / w_tensor[batch].sum().clamp_min(1e-8)
            kl = 0.018 * model.kl() / max(1, len(train_rows))
            branch_shrinkage = 0.006 * torch.stack([value.square().mean() for value in parts.values()]).sum()
            (data_loss + kl + branch_shrinkage).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x_tensor[validation_tensor], sample=False).detach().cpu().numpy()
        score = h24.weighted_rmse(prediction, y[validation_rows], weights[validation_rows])
        if score < best_rmse - 1e-5:
            best_epoch, best_rmse = epoch + 1, score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break
    require(best_state is not None, "Marine vector BNN produced no finite checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    return model, {
        "device": str(device), "best_epoch": int(best_epoch),
        "inner_source_validation_rmse_standardized": float(best_rmse),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "branch_caps_log10": modular.BRANCH_CAP_LOG10,
        "total_cap_log10": float(total_cap_log10),
        "physics_direction_constraints": (
            {"retention": physics_signs}
            if candidate == "physics_constrained"
            else (branch_signs or {})
        ),
        "observation_mean_terms_used": False, "gp_used": False,
    }


def predict(
    model: nn.Module,
    x: np.ndarray,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return posterior q50, deterministic centre, posterior SD and components."""
    device = next(model.parameters()).device
    q50 = np.empty(len(x), np.float32)
    deterministic = np.empty(len(x), np.float32)
    sd = np.empty(len(x), np.float32)
    components = {name: np.empty(len(x), np.float32) for name in model.branch_order}
    for start in range(0, len(x), 32768):
        stop = min(len(x), start + 32768)
        values = torch.as_tensor(x[start:stop], device=device)
        with torch.no_grad():
            deterministic_local = model(values, sample=False)
            deterministic[start:stop] = deterministic_local.cpu().numpy()
            parts = model.components(values, sample=False)
            raw_total = torch.stack(list(parts.values()), dim=0).sum(dim=0)
            cap_factor = torch.where(
                raw_total.abs() > 1e-8,
                deterministic_local / raw_total,
                torch.ones_like(raw_total),
            )
            for name, local in parts.items():
                components[name][start:stop] = (local * cap_factor).cpu().numpy()
            samples = torch.stack([model(values, sample=True) for _ in range(int(draws))])
            q50[start:stop] = torch.quantile(samples, 0.50, dim=0).cpu().numpy()
            sd[start:stop] = samples.std(dim=0, unbiased=True).cpu().numpy()
    return q50, deterministic, sd, components


def crossfit_calibration_predictions(
    x: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    records: pd.DataFrame,
    outer_train_rows: np.ndarray,
    branch_indices: dict[str, list[int]],
    *,
    candidate: str,
    physics_signs: list[float],
    seed: int,
    epochs: int,
    total_cap_log10: float,
    device: torch.device,
    posterior_draws: int,
    splits: int,
    branch_signs: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Generate source-disjoint residuals over the full outer-training set."""
    require(splits >= 2, "Calibration cross-fitting requires at least two splits")
    groups = records.iloc[outer_train_rows]["source_group"].fillna("__missing__").astype(str).to_numpy()
    unique_groups = np.unique(groups)
    require(len(unique_groups) >= splits * 3, "Too few sources for calibration cross-fitting")
    splitter = GroupKFold(n_splits=int(splits))
    calibration_q50 = np.full(len(outer_train_rows), np.nan, np.float32)
    calibration_sd = np.full(len(outer_train_rows), np.nan, np.float32)
    fold_audits: list[dict[str, object]] = []
    for crossfit_fold, (model_local, calibration_local) in enumerate(
        splitter.split(np.arange(len(outer_train_rows)), groups=groups), start=1
    ):
        model_rows = outer_train_rows[model_local].astype(np.int64)
        calibration_rows = outer_train_rows[calibration_local].astype(np.int64)
        inner_fit, inner_validation, inner_audit = inner_source_validation_split(
            records, model_rows, int(seed) + crossfit_fold * 907
        )
        local_target = records.iloc[inner_fit].assign(_target=observed[inner_fit])
        local_source_medians = local_target.groupby("source_group", observed=True)["_target"].median().to_numpy(float)
        local_target_center = float(np.median(local_source_medians))
        local_target_scale = float(1.4826 * np.median(np.abs(local_source_medians - local_target_center)))
        if not np.isfinite(local_target_scale) or local_target_scale < 0.35:
            local_target_scale = max(0.35, float(np.std(local_source_medians)))
        local_y = ((observed - local_target_center) / local_target_scale).astype(np.float32)
        local_model, local_audit = fit_model(
            x, local_y, weights, inner_fit, inner_validation, branch_indices, local_target_scale,
            candidate=candidate,
            physics_signs=physics_signs,
            seed=int(seed) + crossfit_fold * 1009,
            epochs=int(epochs),
            total_cap_log10=float(total_cap_log10),
            device=device,
            branch_signs=branch_signs,
        )
        local_q50, _, local_sd, _ = predict(local_model, x[calibration_rows], max(2, int(posterior_draws)))
        calibration_q50[calibration_local] = local_target_center + local_target_scale * local_q50
        calibration_sd[calibration_local] = local_target_scale * local_sd
        fit_sources = set(records.iloc[model_rows]["source_group"].fillna("__missing__").astype(str))
        held_sources = set(records.iloc[calibration_rows]["source_group"].fillna("__missing__").astype(str))
        require(not (fit_sources & held_sources), "Calibration cross-fit retained overlapping sources")
        fold_audits.append({
            "crossfit_fold": int(crossfit_fold),
            "model_records": int(len(model_rows)),
            "calibration_records": int(len(calibration_rows)),
            "model_sources": int(len(fit_sources)),
            "calibration_sources": int(len(held_sources)),
            "source_overlap": 0,
            "target_center_log10": float(local_target_center),
            "target_scale_log10": float(local_target_scale),
            "inner_early_stopping": inner_audit,
            "fit": local_audit,
        })
        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(np.isfinite(calibration_q50).all() and np.isfinite(calibration_sd).all(),
            "Calibration cross-fitting left missing predictions")
    return outer_train_rows.astype(np.int64), calibration_q50, calibration_sd, {
        "kind": "outer_train_source_group_crossfit_for_variance_only",
        "splits": int(splits),
        "records": int(len(outer_train_rows)),
        "sources": int(len(unique_groups)),
        "outer_heldout_response_used": False,
        "affects_main_environmental_fit": False,
        "folds": fold_audits,
    }


def weighted_metric(values: np.ndarray, weights: np.ndarray) -> float:
    return direct.weighted_metric(values, weights)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    require(bool(finite.any()), "Weighted quantile has no finite positive-weight values")
    local_values = values[finite]
    local_weights = weights[finite]
    order = np.argsort(local_values)
    local_values = local_values[order]
    local_weights = local_weights[order]
    cumulative = np.cumsum(local_weights) - 0.5 * local_weights
    cumulative = cumulative / local_weights.sum()
    return float(np.interp(float(probability), cumulative, local_values))


def separate_known_detection_variance(
    residual: np.ndarray,
    known_detection_sd: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    """Separate known protocol variance from remaining observation variance."""
    total_variance = weighted_metric(np.square(residual), weights)
    known_variance = weighted_metric(np.square(known_detection_sd), weights)
    variance_floor = 0.05**2
    remaining_variance = max(total_variance - known_variance, variance_floor)
    return math.sqrt(total_variance), math.sqrt(remaining_variance), {
        "total_residual_sd_log10": math.sqrt(total_variance),
        "known_detection_rms_sd_log10": math.sqrt(known_variance),
        "remaining_residual_sd_log10": math.sqrt(remaining_variance),
        "remaining_variance_floor_sd_log10": 0.05,
        "known_detection_variance_subtracted": True,
    }


def fit_known_source_observation_layer(
    records: pd.DataFrame,
    calibration_rows: np.ndarray,
    heldout_rows: np.ndarray,
    calibration_residual: np.ndarray,
    calibration_latent_sd: np.ndarray,
    weights: np.ndarray,
    known_detection_sd: np.ndarray,
    *,
    fixed_reported_propagation: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, dict[str, object]]:
    """Fit an empirical-Bayes known-source offset outside the map mean."""
    train_source = records.iloc[calibration_rows]["source_group"].fillna("__MISSING_SOURCE__").astype(str).to_numpy()
    test_source = records.iloc[heldout_rows]["source_group"].fillna("__MISSING_SOURCE__").astype(str).to_numpy()
    frame = pd.DataFrame({"source_group": train_source, "residual": calibration_residual})
    summary = frame.groupby("source_group", sort=False, observed=True)["residual"].agg(["mean", "size"])
    require(len(summary) >= 2, "Known-source layer needs at least two calibration sources")
    source_center = float(summary["mean"].mean())
    centered_mean = summary["mean"] - source_center
    centered_lookup = centered_mean.to_dict()
    count_lookup = summary["size"].astype(float).to_dict()
    raw_train_offset = np.array([centered_lookup[name] for name in train_source], float)
    within_variance = weighted_metric(
        np.square(calibration_residual - raw_train_offset - source_center),
        weights[calibration_rows],
    )
    between_raw = float(np.mean(np.square(centered_mean.to_numpy(float))))
    mean_noise = float(np.median([
        within_variance / max(count_lookup[name], 1.0) for name in summary.index
    ]))
    between_variance = max(between_raw - mean_noise, 0.05**2)
    shrinkage = {
        name: between_variance / (between_variance + within_variance / max(count_lookup[name], 1.0))
        for name in summary.index
    }
    offset_lookup = {
        name: float(centered_lookup[name] * shrinkage[name] + source_center)
        for name in summary.index
    }
    calibration_frame = records.iloc[calibration_rows].reset_index(drop=True)
    event_key, event_key_audit = build_site_event_key(calibration_frame)
    frame["site_event_key"] = event_key.to_numpy()
    source_total = frame.groupby("source_group", sort=False, observed=True)["residual"].agg(["sum", "size"])
    event_total = frame.groupby(
        ["source_group", "site_event_key"], sort=False, observed=True
    )["residual"].agg(["sum", "size"])
    source_sum = frame["source_group"].map(source_total["sum"]).to_numpy(float)
    source_count = frame["source_group"].map(source_total["size"]).to_numpy(float)
    event_index = pd.MultiIndex.from_frame(frame[["source_group", "site_event_key"]])
    event_sum = event_total["sum"].reindex(event_index).to_numpy(float)
    event_count = event_total["size"].reindex(event_index).to_numpy(float)
    excluded_count = source_count - event_count
    interval_calibration_valid = excluded_count > 0
    require(
        int(interval_calibration_valid.sum()) >= 100,
        "Known-source leave-event-out interval calibration has too few records",
    )
    excluded_mean = np.full(len(frame), np.nan, float)
    excluded_mean[interval_calibration_valid] = (
        (source_sum[interval_calibration_valid] - event_sum[interval_calibration_valid])
        / excluded_count[interval_calibration_valid]
    )
    excluded_shrinkage = np.zeros(len(frame), float)
    excluded_shrinkage[interval_calibration_valid] = (
        between_variance
        / (between_variance + within_variance / excluded_count[interval_calibration_valid])
    )
    crossfit_offset = np.full(len(frame), np.nan, float)
    crossfit_offset[interval_calibration_valid] = (
        source_center
        + excluded_shrinkage[interval_calibration_valid]
        * (excluded_mean[interval_calibration_valid] - source_center)
    )
    within_residual = (
        calibration_residual[interval_calibration_valid]
        - crossfit_offset[interval_calibration_valid]
    )
    interval_known_sd = known_detection_sd[calibration_rows][interval_calibration_valid]
    interval_latent_sd = calibration_latent_sd[interval_calibration_valid]
    interval_weight = weights[calibration_rows][interval_calibration_valid]
    if fixed_reported_propagation:
        variance_floor = 0.05**2
        total_observation_variance = weighted_metric(
            np.maximum(
                np.square(within_residual) - np.square(interval_latent_sd),
                variance_floor,
            ),
            interval_weight,
        )
        reported_variance = weighted_metric(
            np.square(interval_known_sd), interval_weight
        )
        remaining_variance = max(
            total_observation_variance - reported_variance, variance_floor
        )
        within_remaining_sd = math.sqrt(remaining_variance)
        within_total_sd = math.sqrt(total_observation_variance)
        within_detection_sd_multiplier = 1.0
        variance_audit = {
            "kind": "fixed_reported_uncertainty_propagation",
            "zero_detection_base_residual_sd_log10": within_total_sd,
            "remaining_base_residual_sd_log10": within_remaining_sd,
            "total_observation_variance_log10_squared": float(total_observation_variance),
            "reported_variance_log10_squared": float(reported_variance),
            "remaining_variance_log10_squared": float(remaining_variance),
            "known_detection_sd_multiplier": 1.0,
            "reported_uncertainty_scale_fitted_to_response": False,
        }
    else:
        within_remaining_sd, within_detection_sd_multiplier, variance_audit = (
            calibrate_detection_variance_scale(
                within_residual, interval_latent_sd, interval_known_sd, interval_weight
            )
        )
        within_total_sd = float(
            variance_audit["zero_detection_base_residual_sd_log10"]
        )
    interval_known_sd_scaled = within_detection_sd_multiplier * interval_known_sd
    aware_calibration_sd = np.sqrt(
        np.square(interval_latent_sd)
        + within_remaining_sd**2
        + np.square(interval_known_sd_scaled)
    )
    unaware_calibration_sd = np.sqrt(
        np.square(interval_latent_sd) + within_total_sd**2
    )
    aware_raw_multiplier = weighted_quantile(
        np.abs(within_residual) / np.maximum(aware_calibration_sd, 1e-6),
        interval_weight, 0.95,
    ) / 1.96
    unaware_raw_multiplier = weighted_quantile(
        np.abs(within_residual) / np.maximum(unaware_calibration_sd, 1e-6),
        interval_weight, 0.95,
    ) / 1.96
    aware_multiplier = float(np.clip(aware_raw_multiplier, 0.60, 2.50))
    unaware_multiplier = float(np.clip(unaware_raw_multiplier, 0.60, 2.50))
    test_offset = np.array([offset_lookup.get(name, np.nan) for name in test_source], float)
    known = np.isfinite(test_offset)
    return (
        test_offset, known, within_total_sd, within_remaining_sd,
        aware_multiplier, unaware_multiplier,
        {
            "used": True,
            "kind": "source_group_crossfit_residual_empirical_bayes_offset",
            "map_q50_effect": False,
            "outer_heldout_response_used": False,
            "calibration_sources": int(len(summary)),
            "heldout_known_source_records": int(known.sum()),
            "heldout_unknown_source_records": int((~known).sum()),
            "heldout_known_source_fraction": float(known.mean()),
            "source_center_log10": source_center,
            "between_source_sd_log10": math.sqrt(between_variance),
            "within_source_total_sd_log10": within_total_sd,
            "within_source_remaining_sd_log10": within_remaining_sd,
            "known_detection_sd_multiplier": within_detection_sd_multiplier,
            "interval_calibration_kind": "leave_site_event_out_within_source",
            "interval_calibration_records": int(interval_calibration_valid.sum()),
            "interval_calibration_record_fraction": float(interval_calibration_valid.mean()),
            "interval_calibration_sources": int(frame.loc[interval_calibration_valid, "source_group"].nunique()),
            "interval_calibration_site_events": int(frame.loc[interval_calibration_valid, "site_event_key"].nunique()),
            "interval_event_key": event_key_audit,
            "aware_interval_multiplier": aware_multiplier,
            "unaware_interval_multiplier": unaware_multiplier,
            "aware_unclipped_interval_multiplier": float(aware_raw_multiplier),
            "unaware_unclipped_interval_multiplier": float(unaware_raw_multiplier),
            "within_source_variance_separation": variance_audit,
            "shrinkage_q10_q50_q90": np.quantile(
                list(shrinkage.values()), [0.10, 0.50, 0.90]
            ).tolist(),
        },
    )


def bridge_variance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create variance-only bridge features; none are allowed into the mean."""
    output = pd.DataFrame(index=frame.index)
    for name in ("source_rows", "component_size", "degree", "weighted_degree"):
        raw = numeric(frame[f"{BRIDGE_PREFIX}{name}"])
        output[f"log1p_{name}"] = np.log1p(np.clip(raw, 0.0, None))
    for name in ("mean_effective_resistance", "max_effective_resistance"):
        raw = numeric(frame[f"{BRIDGE_PREFIX}{name}"])
        output[f"log1p_{name}"] = np.log1p(np.where(np.isfinite(raw), np.clip(raw, 0.0, None), 0.0))
        output[f"{name}_missing"] = (~np.isfinite(raw)).astype(float)
    identifiability = frame[f"{BRIDGE_PREFIX}identifiability_class"].fillna("missing").astype(str)
    for level in ("partial_cross_method_identification", "source_bridge_only", "isolated_no_bridge"):
        output[f"ident_{level}"] = identifiability.eq(level).astype(float)
    if "detection_method" in frame:
        method = frame["detection_method"].fillna("").astype(str).str.strip().str.lower()
        output["detection_method_recorded"] = (~method.isin({"", "nan", "unknown", "not_reported"})).astype(float)
    else:
        output["detection_method_recorded"] = 0.0
    if "sizecorr_lower_used_um" in frame:
        lower = numeric(frame["sizecorr_lower_used_um"])
        output["log1p_lower_size_um"] = np.log1p(np.where(np.isfinite(lower), np.clip(lower, 0.0, None), 0.0))
        output["lower_size_missing"] = (~np.isfinite(lower)).astype(float)
    else:
        output["log1p_lower_size_um"] = 0.0
        output["lower_size_missing"] = 1.0
    if "qaqc_total_score" in frame:
        qaqc = numeric(frame["qaqc_total_score"])
        output["qaqc_score"] = np.where(np.isfinite(qaqc), qaqc, 0.0)
        output["qaqc_missing"] = (~np.isfinite(qaqc)).astype(float)
    else:
        output["qaqc_score"] = 0.0
        output["qaqc_missing"] = 1.0
    return output.astype(float)


def standardized_bridge_features(
    calibration: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    calibration_values = calibration.to_numpy(float)
    prediction_values = prediction[calibration.columns].to_numpy(float)
    centre = np.nanmedian(calibration_values, axis=0)
    q25, q75 = np.nanquantile(calibration_values, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    calibration_values = np.where(np.isfinite(calibration_values), calibration_values, centre)
    prediction_values = np.where(np.isfinite(prediction_values), prediction_values, centre)
    calibration_standardized = np.clip((calibration_values - centre) / scale, -5.0, 5.0)
    prediction_standardized = np.clip((prediction_values - centre) / scale, -5.0, 5.0)
    return calibration_standardized, prediction_standardized, {
        "features": list(calibration.columns),
        "centre": centre.tolist(),
        "iqr_scale": scale.tolist(),
        "clip": [-5.0, 5.0],
    }


def calibrate_detection_variance_scale(
    residual: np.ndarray,
    latent_sd: np.ndarray,
    known_sd: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    """Estimate residual and reported-detection variance scales by cross-fit NLL.

    The fit uses only source-group cross-fitted outer-train residuals.  It does
    not estimate a mean correction.  A multiplier below one means the reported
    protocol/correction SD behaves as a conservative uncertainty bound rather
    than as an additive record-level error SD in this prediction target.
    """
    finite = (
        np.isfinite(residual) & np.isfinite(latent_sd) & np.isfinite(known_sd)
        & np.isfinite(weights) & (weights > 0)
    )
    require(int(finite.sum()) >= 100, "Detection variance calibration has too few finite records")
    r2 = np.square(residual[finite])
    latent2 = np.square(latent_sd[finite])
    known2 = np.square(known_sd[finite])
    local_weight = weights[finite].astype(float)
    local_weight = local_weight / local_weight.sum()
    variance_floor = 0.05**2
    variance_cap = 4.0**2

    def objective(parameter: np.ndarray) -> float:
        base_variance, detection_variance_multiplier = parameter
        total_variance = np.clip(
            latent2 + base_variance + detection_variance_multiplier * known2,
            variance_floor, variance_cap + float(np.nanmax(latent2)),
        )
        return float(np.sum(local_weight * (np.log(total_variance) + r2 / total_variance)))

    moment = float(np.sum(local_weight * np.maximum(r2 - latent2 - known2, variance_floor)))
    starts = (
        np.array([np.clip(moment, variance_floor, variance_cap), 1.0]),
        np.array([np.clip(float(np.sum(local_weight * np.maximum(r2 - latent2, variance_floor))), variance_floor, variance_cap), 0.0]),
        np.array([max(variance_floor, 0.5 * moment), 2.0]),
    )
    fits = [
        minimize(
            objective, start, method="L-BFGS-B",
            bounds=((variance_floor, variance_cap), (0.0, 4.0)),
        )
        for start in starts
    ]
    fit = min(fits, key=lambda item: float(item.fun))
    require(bool(np.isfinite(fit.fun)), "Detection variance calibration returned a non-finite objective")
    base_variance = float(fit.x[0])
    detection_variance_multiplier = float(fit.x[1])

    def fit_base_with_fixed_detection(fixed_multiplier: float) -> tuple[float, float]:
        def local_objective(parameter: np.ndarray) -> float:
            return objective(np.array([float(parameter[0]), fixed_multiplier]))
        local = minimize(
            local_objective, np.array([base_variance]), method="L-BFGS-B",
            bounds=((variance_floor, variance_cap),),
        )
        return float(local.x[0]), float(local.fun)

    zero_base, zero_nll = fit_base_with_fixed_detection(0.0)
    one_base, one_nll = fit_base_with_fixed_detection(1.0)
    detection_sd_multiplier = math.sqrt(max(detection_variance_multiplier, 0.0))
    return math.sqrt(base_variance), detection_sd_multiplier, {
        "kind": "crossfit_heteroscedastic_gaussian_variance_calibration",
        "records": int(finite.sum()),
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "base_residual_sd_log10": math.sqrt(base_variance),
        "known_detection_sd_multiplier": detection_sd_multiplier,
        "known_detection_variance_multiplier": detection_variance_multiplier,
        "free_nll": float(fit.fun),
        "zero_detection_nll": zero_nll,
        "unit_detection_nll": one_nll,
        "nll_improvement_over_zero_detection": zero_nll - float(fit.fun),
        "nll_improvement_over_unit_detection": one_nll - float(fit.fun),
        "zero_detection_base_residual_sd_log10": math.sqrt(zero_base),
        "unit_detection_base_residual_sd_log10": math.sqrt(one_base),
        "detection_scale_at_lower_boundary": bool(detection_variance_multiplier <= 1e-6),
        "detection_scale_at_upper_boundary": bool(detection_variance_multiplier >= 4.0 - 1e-6),
        "mean_known_detection_sd_log10": float(np.sum(local_weight * known_sd[finite])),
        "known_detection_sd_q10_q50_q90_log10": np.quantile(
            known_sd[finite], [0.10, 0.50, 0.90]
        ).tolist(),
        "mean_correction_used": False,
        "outer_heldout_response_used": False,
    }


def observation_scale_fit(
    records: pd.DataFrame,
    calibration_rows: np.ndarray,
    heldout_rows: np.ndarray,
    calibration_residual: np.ndarray,
    calibration_latent_sd: np.ndarray,
    known_sd: np.ndarray,
    weights: np.ndarray,
    *,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, float, float, dict[str, object]]:
    """Fit a variance-only MEM without changing or re-centring q50."""
    calibration_weight = weights[calibration_rows]
    calibration_known = known_sd[calibration_rows]
    legacy_sd = math.sqrt(weighted_metric(np.square(calibration_residual), calibration_weight))
    if mode == "legacy_global":
        return (
            np.full(len(heldout_rows), legacy_sd, float), legacy_sd, 1.0,
            {
                "kind": mode,
                "latent_q50_effect": False,
                "legacy_residual_rms_log10": float(legacy_sd),
                "known_detection_sd_multiplier": 1.0,
                "known_observation_variance_subtracted_before_fit": False,
                "bridge_response_fields_used": False,
            },
        )

    if mode == "fixed_reported_propagation":
        variance_floor = 0.05**2
        total_observation_variance = weighted_metric(
            np.maximum(
                np.square(calibration_residual) - np.square(calibration_latent_sd),
                variance_floor,
            ),
            calibration_weight,
        )
        reported_variance = weighted_metric(
            np.square(calibration_known), calibration_weight
        )
        remaining_variance = max(
            total_observation_variance - reported_variance, variance_floor
        )
        calibration_total_sd = np.sqrt(
            np.square(calibration_latent_sd)
            + remaining_variance
            + np.square(calibration_known)
        )
        standardized_error = np.abs(calibration_residual) / np.maximum(
            calibration_total_sd, 1e-6
        )
        raw_multiplier = weighted_quantile(
            standardized_error, calibration_weight, 0.95
        ) / 1.96
        multiplier = float(np.clip(raw_multiplier, 0.60, 2.50))
        aware_base_sd = math.sqrt(remaining_variance)
        unaware_base_sd = math.sqrt(total_observation_variance)
        return (
            np.full(len(heldout_rows), aware_base_sd, float),
            aware_base_sd,
            multiplier,
            {
                "kind": mode,
                "base_residual_sd_log10": aware_base_sd,
                "unaware_base_residual_sd_log10": unaware_base_sd,
                "known_detection_sd_multiplier": 1.0,
                "known_detection_variance_multiplier": 1.0,
                "total_observation_variance_log10_squared": float(total_observation_variance),
                "reported_variance_log10_squared": float(reported_variance),
                "remaining_variance_log10_squared": float(remaining_variance),
                "known_observation_variance_subtracted_before_fit": True,
                "reported_uncertainty_scale_fitted_to_response": False,
                "latent_q50_effect": False,
                "bridge_response_fields_used": False,
                "bridge_mean_correction_used": False,
                "interval_multiplier_from_inner_source_holdout": multiplier,
                "unclipped_interval_multiplier": float(raw_multiplier),
            },
        )

    if mode == "bridge_mem_fixed_reported":
        detection_sd_multiplier = 1.0
        reported_variance = weighted_metric(
            np.square(calibration_known), calibration_weight
        )
        calibrated_base_sd = math.sqrt(max(
            weighted_metric(
                np.maximum(
                    np.square(calibration_residual)
                    - np.square(calibration_latent_sd)
                    - np.square(calibration_known),
                    0.05**2,
                ),
                calibration_weight,
            ),
            0.05**2,
        ))
        detection_scale_audit = {
            "kind": "fixed_reported_uncertainty_propagation",
            "records": int(len(calibration_rows)),
            "known_detection_sd_multiplier": 1.0,
            "known_detection_variance_multiplier": 1.0,
            "reported_variance_log10_squared": float(reported_variance),
            "reported_uncertainty_scale_fitted_to_response": False,
            "mean_correction_used": False,
            "outer_heldout_response_used": False,
        }
    else:
        calibrated_base_sd, detection_sd_multiplier, detection_scale_audit = (
            calibrate_detection_variance_scale(
                calibration_residual, calibration_latent_sd,
                calibration_known, calibration_weight,
            )
        )
    calibration_known_scaled = detection_sd_multiplier * calibration_known
    adjusted_variance = (
        np.square(calibration_residual)
        - np.square(calibration_latent_sd)
        - np.square(calibration_known_scaled)
    )
    variance_floor = 0.05**2
    adjusted_variance = np.clip(adjusted_variance, variance_floor, 4.0**2)
    if mode == "decomposed_global":
        base_variance = calibrated_base_sd**2
        calibration_base_sd = np.full(len(calibration_rows), math.sqrt(base_variance), float)
        heldout_base_sd = np.full(len(heldout_rows), math.sqrt(base_variance), float)
        grid_base_sd = math.sqrt(base_variance)
        fit_detail: dict[str, object] = {
            "kind": mode,
            "base_residual_sd_log10": float(grid_base_sd),
            "known_detection_sd_multiplier": float(detection_sd_multiplier),
            "detection_variance_calibration": detection_scale_audit,
        }
    else:
        require(
            mode in {"bridge_mem", "bridge_mem_fixed_reported"},
            f"Unknown observation scale model: {mode}",
        )
        calibration_frame = records.iloc[calibration_rows].copy()
        calibration_frame["_adjusted_variance"] = adjusted_variance
        source_target = (
            calibration_frame.groupby("source_group", sort=False, observed=True)["_adjusted_variance"]
            .median()
            .rename("adjusted_variance")
            .reset_index()
        )
        source_first = calibration_frame.drop_duplicates("source_group", keep="first")
        source_first = source_first.merge(source_target, on="source_group", how="inner", validate="one_to_one")
        require(len(source_first) >= 12, f"Too few inner-validation sources for bridge MEM: {len(source_first)}")
        source_feature = bridge_variance_features(source_first).reset_index(drop=True)
        record_feature = bridge_variance_features(records.iloc[heldout_rows]).reset_index(drop=True)
        source_x, heldout_x, feature_audit = standardized_bridge_features(source_feature, record_feature)
        target = np.log(np.clip(numeric(source_first["adjusted_variance"]), variance_floor, 4.0**2))
        splits = min(5, len(source_first))
        require(splits >= 3, "Bridge MEM needs at least three cross-fitting splits")
        kfold = KFold(n_splits=splits, shuffle=True, random_state=int(seed) + 611)
        oof_log_variance = np.full(len(source_first), np.nan, float)
        for fit_index, validation_index in kfold.split(source_x):
            local = Ridge(alpha=6.0, fit_intercept=True)
            local.fit(source_x[fit_index], target[fit_index])
            oof_log_variance[validation_index] = local.predict(source_x[validation_index])
        require(np.isfinite(oof_log_variance).all(), "Bridge MEM cross-fitting left missing predictions")
        final = Ridge(alpha=6.0, fit_intercept=True)
        final.fit(source_x, target)
        lower_log_variance = math.log(variance_floor)
        upper_log_variance = math.log(4.0**2)
        heldout_log_variance = np.clip(final.predict(heldout_x), lower_log_variance, upper_log_variance)
        heldout_base_sd = np.sqrt(np.exp(heldout_log_variance))
        source_oof_sd = np.sqrt(np.exp(np.clip(oof_log_variance, lower_log_variance, upper_log_variance)))
        source_sd_lookup = dict(zip(source_first["source_group"].astype(str), source_oof_sd))
        calibration_base_sd = calibration_frame["source_group"].astype(str).map(source_sd_lookup).to_numpy(float)
        require(np.isfinite(calibration_base_sd).all(), "Bridge MEM source OOF scale join failed")
        grid_base_sd = float(np.median(np.sqrt(np.exp(np.clip(final.predict(source_x), lower_log_variance, upper_log_variance)))))
        fit_detail = {
            "kind": mode,
            "calibration_sources": int(len(source_first)),
            "crossfit_splits": int(splits),
            "ridge_alpha": 6.0,
            "variance_floor_sd_log10": 0.05,
            "variance_cap_sd_log10": 4.0,
            "grid_hypothetical_source_policy": "median fitted inner-validation source residual SD",
            "grid_base_residual_sd_log10": float(grid_base_sd),
            "source_oof_sd_q10_q50_q90_log10": np.quantile(source_oof_sd, [0.10, 0.50, 0.90]).tolist(),
            "feature_preprocessing": feature_audit,
            "known_detection_sd_multiplier": float(detection_sd_multiplier),
            "detection_variance_calibration": detection_scale_audit,
        }
        if mode == "bridge_mem_fixed_reported":
            fit_detail["unaware_additional_variance_log10_squared"] = float(
                detection_scale_audit["reported_variance_log10_squared"]
            )

    calibration_total_sd = np.sqrt(
        np.square(calibration_latent_sd)
        + np.square(calibration_base_sd)
        + np.square(calibration_known_scaled)
    )
    standardized_error = np.abs(calibration_residual) / np.maximum(calibration_total_sd, 1e-6)
    raw_multiplier = weighted_quantile(standardized_error, calibration_weight, 0.95) / 1.96
    multiplier = float(np.clip(raw_multiplier, 0.60, 2.50))
    fit_detail.update({
        "latent_q50_effect": False,
        "known_observation_variance_subtracted_before_fit": True,
        "bridge_response_fields_used": False,
        "bridge_mean_correction_used": False,
        "interval_multiplier_from_inner_source_holdout": multiplier,
        "unclipped_interval_multiplier": float(raw_multiplier),
    })
    return heldout_base_sd, float(grid_base_sd), multiplier, fit_detail


def weighted_interval_score(
    observed: np.ndarray,
    median: np.ndarray,
    sd: np.ndarray,
    weights: np.ndarray,
    alphas: tuple[float, ...] = (0.50, 0.20, 0.10, 0.05),
) -> float:
    """Proper WIS from four central normal predictive intervals plus median."""
    total = 0.5 * np.abs(observed - median)
    for alpha in alphas:
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        lower = median - z * sd
        upper = median + z * sd
        interval = (
            (upper - lower)
            + (2.0 / alpha) * np.maximum(lower - observed, 0.0)
            + (2.0 / alpha) * np.maximum(observed - upper, 0.0)
        )
        total = total + (alpha / 2.0) * interval
    return weighted_metric(total / (len(alphas) + 0.5), weights)


def regional_diagnostic(grid: pd.DataFrame) -> dict[str, object]:
    valid = grid["latent_q50_log10"].notna().to_numpy()
    latitude = numeric(grid["latitude"])
    mean = numeric(grid["latent_q50_log10"])
    def median(mask: np.ndarray) -> float | None:
        value = mean[valid & mask]
        return None if not len(value) else float(np.median(value))
    equator = np.abs(latitude) <= 8.0
    subtropical = (np.abs(latitude) >= 15.0) & (np.abs(latitude) <= 35.0)
    eq, sub = median(equator), median(subtropical)
    return {
        "diagnostic_only_not_loss": True,
        "equator_abs_lat_le_8_median_log10": eq,
        "subtropical_abs_lat_15_35_median_log10": sub,
        "equator_minus_subtropical_log10": None if eq is None or sub is None else float(eq - sub),
        "valid_equator_cells": int((valid & equator).sum()),
        "valid_subtropical_cells": int((valid & subtropical).sum()),
    }


def train(args: argparse.Namespace) -> int:
    require(args.domain == DOMAIN, "This runner permits domain=marine only")
    branches = branches_for_candidate(args.candidate)
    records, grid, input_audit = load_inputs(args)
    weights = h24.source_equal_weights(records["source_group"])
    outer_train_rows, heldout_rows, split_audit = choose_outer_split(records, args)
    validation_site_event_key, _ = build_site_event_key(records)
    fit_rows, early_stop_rows, inner_split_audit = inner_source_validation_split(
        records, outer_train_rows, int(args.seed) + int(args.fold) * 101
    )
    target_frame = records.iloc[fit_rows].assign(_target=numeric(records.iloc[fit_rows]["observed_log10"]))
    source_medians = target_frame.groupby("source_group", observed=True)["_target"].median().to_numpy(float)
    target_center = float(np.median(source_medians))
    target_scale = float(1.4826 * np.median(np.abs(source_medians - target_center)))
    if not np.isfinite(target_scale) or target_scale < 0.35:
        target_scale = max(0.35, float(np.std(source_medians)))
    observed = numeric(records["observed_log10"])
    y = ((observed - target_center) / target_scale).astype(np.float32)
    requested_features = tuple(name for names in branches.values() for name in names if name in grid and name in records)
    active_status = ACTIVE_STATUS
    scaler = h24.ProcessScaler.fit(
        records,
        grid.loc[grid[active_status].astype(str).eq("observed")].reset_index(drop=True),
        requested_features,
        fit_rows,
        minimum_features=(
            4 if args.candidate == "transport_path_potential_shared_directional" else 5
        ),
    )
    branch_indices = make_branch_indices(scaler.names, branches)
    physics_signs = []
    branch_signs: dict[str, list[float]] = {}
    if args.candidate == "physics_constrained":
        retention_names = [scaler.names[index] for index in branch_indices["retention"]]
        missing_orientation = [name for name in retention_names if name not in PHYSICS_ORIENTATION]
        require(not missing_orientation, f"Missing physics orientations: {missing_orientation}")
        physics_signs = [PHYSICS_ORIENTATION[name] for name in retention_names]
    if args.candidate in {
        "transport_graph_signed_process",
        "transport_graph_signed_no_convergence_process",
        "transport_graph_signed_persistent_convergence_process",
        "transport_graph_shared_signed_process",
        "transport_graph_shared_signed_accumulation_process",
        "transport_path_potential_shared_directional",
    }:
        for branch, indices in branch_indices.items():
            local_names = [scaler.names[index] for index in indices]
            missing_orientation = [name for name in local_names if name not in PHYSICS_ORIENTATION]
            require(
                not missing_orientation,
                f"Missing signed process orientations for {branch}: {missing_orientation}",
            )
            branch_signs[branch] = [PHYSICS_ORIENTATION[name] for name in local_names]
    x = scaler.transform(records)
    grid_observed = grid[active_status].astype(str).eq("observed").to_numpy()
    x_grid = scaler.transform(grid.loc[grid_observed].reset_index(drop=True))
    (
        heldout_aoa_status,
        heldout_aoa_distance,
        supported_grid_aoa_status,
        supported_grid_aoa_distance,
        aoa_audit,
    ) = fold_local_aoa(x[outer_train_rows], x[heldout_rows], x_grid)
    heldout_aoa_in = heldout_aoa_status == "AOA_in"
    grid_aoa_status = np.full(len(grid), "not_assessed", dtype=object)
    grid_aoa_distance = np.full(len(grid), np.nan, dtype=float)
    grid_aoa_status[grid_observed] = supported_grid_aoa_status
    grid_aoa_distance[grid_observed] = supported_grid_aoa_distance
    device = select_device(args.device)
    model, fit_audit = fit_model(
        x, y, weights, fit_rows, early_stop_rows, branch_indices, target_scale,
        candidate=args.candidate, physics_signs=physics_signs,
        seed=int(args.seed), epochs=int(args.epochs), total_cap_log10=float(args.map_cap_log10), device=device,
        branch_signs=branch_signs,
    )
    heldout_standardized, heldout_deterministic_standardized, heldout_sd_standardized, _ = predict(
        model, x[heldout_rows], max(2, int(args.posterior_draws))
    )
    heldout_mean = target_center + target_scale * heldout_standardized
    heldout_deterministic = target_center + target_scale * heldout_deterministic_standardized
    heldout_latent_sd = target_scale * heldout_sd_standardized
    heldout_observed = observed[heldout_rows]
    heldout_weight = weights[heldout_rows]
    residual = heldout_observed - heldout_mean
    if int(args.calibration_crossfits) >= 2:
        calibration_rows, calibration_standardized, calibration_sd_standardized, calibration_fit_audit = (
            crossfit_calibration_predictions(
                x,
                observed,
                weights,
                records,
                outer_train_rows,
                branch_indices,
                candidate=args.candidate,
                physics_signs=physics_signs,
                seed=int(args.seed) + int(args.fold) * 101,
                epochs=int(args.epochs),
                total_cap_log10=float(args.map_cap_log10),
                device=device,
                posterior_draws=int(args.posterior_draws),
                splits=int(args.calibration_crossfits),
                branch_signs=branch_signs,
            )
        )
    else:
        calibration_rows = early_stop_rows
        calibration_standardized, _, calibration_sd_standardized, _ = predict(
            model, x[calibration_rows], max(2, int(args.posterior_draws))
        )
        calibration_fit_audit = {
            "kind": "single_inner_source_holdout_for_variance_only",
            "records": int(len(calibration_rows)),
            "sources": int(records.iloc[calibration_rows]["source_group"].nunique()),
            "outer_heldout_response_used": False,
            "affects_main_environmental_fit": False,
        }
    if int(args.calibration_crossfits) >= 2:
        calibration_mean = calibration_standardized.astype(float)
        calibration_latent_sd = calibration_sd_standardized.astype(float)
    else:
        calibration_mean = target_center + target_scale * calibration_standardized
        calibration_latent_sd = target_scale * calibration_sd_standardized
    calibration_residual = observed[calibration_rows] - calibration_mean
    calibration_weight = weights[calibration_rows]
    residual_center = weighted_metric(calibration_residual, calibration_weight)
    residual_sd = math.sqrt(weighted_metric(np.square(calibration_residual), calibration_weight))

    size_sd = numeric(records["sizecorr_known_sd_log10"]) if "sizecorr_known_sd_log10" in records else np.full(len(records), np.nan)
    unit_sd = numeric(records["unit_sd_log10"]) if "unit_sd_log10" in records else np.full(len(records), np.nan)
    size_finite, unit_finite = np.isfinite(size_sd), np.isfinite(unit_sd)
    known_any = size_finite | unit_finite
    known_raw = np.sqrt(np.square(np.where(size_finite, size_sd, 0.0)) + np.square(np.where(unit_finite, unit_sd, 0.0)))
    calibration_known = known_raw[fit_rows][known_any[fit_rows]]
    known_imputation = float(np.median(calibration_known)) if len(calibration_known) else 0.0
    known_sd = np.where(known_any, known_raw, known_imputation)
    heldout_base_residual_sd, grid_base_residual_sd, interval_multiplier, scale_audit = observation_scale_fit(
        records,
        calibration_rows,
        heldout_rows,
        calibration_residual,
        calibration_latent_sd,
        known_sd,
        weights,
        mode=args.observation_scale_model,
        seed=int(args.seed) + int(args.fold) * 101,
    )
    detection_sd_multiplier = float(scale_audit.get("known_detection_sd_multiplier", 1.0))
    calibrated_known_sd = detection_sd_multiplier * known_sd
    source_effect_applicable = args.validation_scheme in {
        "random_record", "random_site_event",
        "random_record_10fold", "random_site_event_10fold",
        "strict_spatial_block",
    }
    if source_effect_applicable:
        require(
            int(args.calibration_crossfits) >= 2 and np.array_equal(calibration_rows, outer_train_rows),
            "Known-source random validation requires source-group cross-fitted predictions for every outer-train record",
        )
        (
            known_source_offset,
            known_source_available,
            known_source_total_sd,
            known_source_remaining_sd,
            known_source_aware_multiplier,
            known_source_unaware_multiplier,
            known_source_audit,
        ) = fit_known_source_observation_layer(
            records, calibration_rows, heldout_rows, calibration_residual,
            calibration_latent_sd, weights, known_sd,
            fixed_reported_propagation=(
                args.observation_scale_model
                in {"fixed_reported_propagation", "bridge_mem_fixed_reported"}
            ),
        )
    else:
        known_source_offset = np.full(len(heldout_rows), np.nan, float)
        known_source_available = np.zeros(len(heldout_rows), bool)
        known_source_total_sd = math.nan
        known_source_remaining_sd = math.nan
        known_source_aware_multiplier = math.nan
        known_source_unaware_multiplier = math.nan
        known_source_audit = {
            "used": False,
            "reason": "source-held-out validation has no identifiable held-out source offset",
            "map_q50_effect": False,
        }

    grid_mean = np.full(len(grid), np.nan, np.float32)
    grid_deterministic = np.full(len(grid), np.nan, np.float32)
    grid_latent_sd = np.full(len(grid), np.nan, np.float32)
    components_observed: dict[str, np.ndarray] = {}
    if not getattr(args, "oof_only", False):
        grid_standardized, grid_deterministic_standardized, grid_sd_standardized, components_observed = predict(
            model, x_grid, int(args.posterior_draws)
        )
        grid_mean[grid_observed] = target_center + target_scale * grid_standardized
        grid_deterministic[grid_observed] = target_center + target_scale * grid_deterministic_standardized
        grid_latent_sd[grid_observed] = target_scale * grid_sd_standardized
    metric_prefix = args.validation_scheme
    metrics = {
        f"{metric_prefix}_r2": float(r2_score(heldout_observed, heldout_mean, sample_weight=heldout_weight)),
        f"{metric_prefix}_rmse_log10": math.sqrt(weighted_metric(np.square(residual), heldout_weight)),
        f"{metric_prefix}_mae_log10": weighted_metric(np.abs(residual), heldout_weight),
        "inner_calibration_residual_root_mean_square_log10": float(residual_sd),
        "inner_calibration_residual_center_log10": float(residual_center),
        "observation_scale_model": args.observation_scale_model,
        "interval_multiplier": float(interval_multiplier),
        "posterior_q50_minus_deterministic_median_abs_log10": float(np.median(np.abs(heldout_mean - heldout_deterministic))),
    }
    require(int(heldout_aoa_in.sum()) >= 2, "Fold-local AOA contains fewer than two held-out records")
    metrics.update({
        f"aoa_in_{metric_prefix}_records": int(heldout_aoa_in.sum()),
        f"aoa_in_{metric_prefix}_fraction": float(heldout_aoa_in.mean()),
        f"aoa_in_{metric_prefix}_r2": float(r2_score(
            heldout_observed[heldout_aoa_in], heldout_mean[heldout_aoa_in],
            sample_weight=heldout_weight[heldout_aoa_in],
        )),
        f"aoa_in_{metric_prefix}_rmse_log10": math.sqrt(weighted_metric(
            np.square(residual[heldout_aoa_in]), heldout_weight[heldout_aoa_in],
        )),
        f"aoa_in_{metric_prefix}_mae_log10": weighted_metric(
            np.abs(residual[heldout_aoa_in]), heldout_weight[heldout_aoa_in],
        ),
    })
    for arm, interval_aware in (("protocol_aware", True), ("detection_unaware", False)):
        output_dir = args.run_root / "S0" / arm / DOMAIN
        output_dir.mkdir(parents=True, exist_ok=True)
        local_known = calibrated_known_sd[heldout_rows] if interval_aware else np.zeros(len(heldout_rows))
        arm_heldout_base_residual_sd = heldout_base_residual_sd
        arm_grid_base_residual_sd = grid_base_residual_sd
        if not interval_aware and "unaware_base_residual_sd_log10" in scale_audit:
            arm_heldout_base_residual_sd = np.full(
                len(heldout_rows),
                float(scale_audit["unaware_base_residual_sd_log10"]),
                float,
            )
            arm_grid_base_residual_sd = float(
                scale_audit["unaware_base_residual_sd_log10"]
            )
        elif not interval_aware and "unaware_additional_variance_log10_squared" in scale_audit:
            additional_variance = float(
                scale_audit["unaware_additional_variance_log10_squared"]
            )
            arm_heldout_base_residual_sd = np.sqrt(
                np.square(heldout_base_residual_sd) + additional_variance
            )
            arm_grid_base_residual_sd = math.sqrt(
                grid_base_residual_sd**2 + additional_variance
            )
        heldout_future_sd = interval_multiplier * np.sqrt(
            np.square(heldout_latent_sd)
            + np.square(arm_heldout_base_residual_sd)
            + np.square(local_known)
        )
        coverage = float(np.average(np.abs(residual) <= 1.96 * heldout_future_sd, weights=heldout_weight))
        wis = weighted_interval_score(heldout_observed, heldout_mean, heldout_future_sd, heldout_weight)
        aoa_coverage = float(np.average(
            np.abs(residual[heldout_aoa_in]) <= 1.96 * heldout_future_sd[heldout_aoa_in],
            weights=heldout_weight[heldout_aoa_in],
        ))
        aoa_wis = weighted_interval_score(
            heldout_observed[heldout_aoa_in], heldout_mean[heldout_aoa_in],
            heldout_future_sd[heldout_aoa_in], heldout_weight[heldout_aoa_in],
        )
        conditional_mean = np.full(len(heldout_rows), np.nan, float)
        conditional_future_sd = np.full(len(heldout_rows), np.nan, float)
        conditional_metrics: dict[str, object] = {}
        if known_source_available.any():
            conditional_mean[known_source_available] = (
                heldout_mean[known_source_available]
                + known_source_offset[known_source_available]
            )
            conditional_base_sd = (
                known_source_remaining_sd if interval_aware else known_source_total_sd
            )
            conditional_multiplier = (
                known_source_aware_multiplier if interval_aware else known_source_unaware_multiplier
            )
            conditional_known_sd = (
                known_source_audit["known_detection_sd_multiplier"]
                * known_sd[heldout_rows][known_source_available]
                if interval_aware else np.zeros(int(known_source_available.sum()))
            )
            conditional_future_sd[known_source_available] = conditional_multiplier * np.sqrt(
                np.square(heldout_latent_sd[known_source_available])
                + conditional_base_sd**2
                + np.square(conditional_known_sd)
            )
            conditional_residual = (
                heldout_observed[known_source_available]
                - conditional_mean[known_source_available]
            )
            conditional_weight = heldout_weight[known_source_available]
            conditional_metrics = {
                f"known_source_{metric_prefix}_records": int(known_source_available.sum()),
                f"known_source_{metric_prefix}_r2": float(r2_score(
                    heldout_observed[known_source_available],
                    conditional_mean[known_source_available],
                    sample_weight=conditional_weight,
                )),
                f"known_source_{metric_prefix}_rmse_log10": math.sqrt(weighted_metric(
                    np.square(conditional_residual), conditional_weight,
                )),
                f"known_source_{metric_prefix}_coverage95": float(np.average(
                    np.abs(conditional_residual)
                    <= 1.96 * conditional_future_sd[known_source_available],
                    weights=conditional_weight,
                )),
                f"known_source_{metric_prefix}_wis_log10": weighted_interval_score(
                    heldout_observed[known_source_available],
                    conditional_mean[known_source_available],
                    conditional_future_sd[known_source_available],
                    conditional_weight,
                ),
            }
            conditional_aoa = known_source_available & heldout_aoa_in
            if int(conditional_aoa.sum()) >= 2:
                conditional_aoa_residual = heldout_observed[conditional_aoa] - conditional_mean[conditional_aoa]
                conditional_aoa_weight = heldout_weight[conditional_aoa]
                conditional_metrics.update({
                    f"known_source_aoa_in_{metric_prefix}_records": int(conditional_aoa.sum()),
                    f"known_source_aoa_in_{metric_prefix}_r2": float(r2_score(
                        heldout_observed[conditional_aoa], conditional_mean[conditional_aoa],
                        sample_weight=conditional_aoa_weight,
                    )),
                    f"known_source_aoa_in_{metric_prefix}_rmse_log10": math.sqrt(weighted_metric(
                        np.square(conditional_aoa_residual), conditional_aoa_weight,
                    )),
                    f"known_source_aoa_in_{metric_prefix}_coverage95": float(np.average(
                        np.abs(conditional_aoa_residual)
                        <= 1.96 * conditional_future_sd[conditional_aoa],
                        weights=conditional_aoa_weight,
                    )),
                    f"known_source_aoa_in_{metric_prefix}_wis_log10": weighted_interval_score(
                        heldout_observed[conditional_aoa], conditional_mean[conditional_aoa],
                        conditional_future_sd[conditional_aoa], conditional_aoa_weight,
                    ),
                })
        detection_sd = float(np.median(calibrated_known_sd[fit_rows])) if interval_aware else 0.0
        grid_future_sd = np.full(len(grid), np.nan, np.float32)
        grid_future_sd[grid_observed] = interval_multiplier * np.sqrt(
            np.square(grid_latent_sd[grid_observed])
            + arm_grid_base_residual_sd**2
            + detection_sd**2
        )
        output_grid = grid[[
            "prediction_id", "longitude", "latitude", "waterbody", "caoa_status",
            CMEMS_STATUS, *CMEMS_SUPPORT_COLUMNS,
        ]].copy()
        output_grid["latent_q50_log10"] = grid_mean
        output_grid["aoa_status"] = grid_aoa_status
        output_grid["aoa_distance"] = grid_aoa_distance
        output_grid["latent_deterministic_log10"] = grid_deterministic
        output_grid["latent_sd_log10"] = grid_latent_sd
        output_grid["base_residual_sd_log10"] = np.where(grid_observed, grid_base_residual_sd, np.nan)
        output_grid["interval_multiplier"] = interval_multiplier
        output_grid["future_observation_sd_log10"] = grid_future_sd
        if not getattr(args, "oof_only", False):
            output_grid.to_csv(output_dir / "H26_UNIT_GRID.csv.gz", index=False, compression="gzip")
        heldout = records.iloc[heldout_rows][["stable_record_id", "waterbody", "source_group"]].copy()
        heldout[CMEMS_STATUS] = "observed"
        heldout["validation_site_event_key"] = validation_site_event_key.iloc[heldout_rows].to_numpy()
        heldout["outer_validation_fold"] = int(args.fold)
        heldout["observed_log10"] = heldout_observed
        heldout["aoa_status"] = heldout_aoa_status
        heldout["aoa_distance"] = heldout_aoa_distance
        heldout["predicted_q50_log10"] = heldout_mean
        heldout["predicted_deterministic_log10"] = heldout_deterministic
        heldout["latent_sd_log10"] = heldout_latent_sd
        heldout["residual_log10"] = residual
        heldout["base_residual_sd_log10"] = heldout_base_residual_sd
        heldout["interval_multiplier"] = interval_multiplier
        heldout["future_observation_sd_log10"] = heldout_future_sd
        heldout["known_source_available"] = known_source_available
        heldout["known_source_offset_log10"] = known_source_offset
        heldout["known_source_conditional_q50_log10"] = conditional_mean
        heldout["known_source_conditional_future_sd_log10"] = conditional_future_sd
        heldout.to_csv(output_dir / "H26_UNIT_HELDOUT.csv.gz", index=False, compression="gzip")
        component_frame = grid[[
            "prediction_id", "longitude", "latitude", "waterbody", CMEMS_STATUS,
            *CMEMS_SUPPORT_COLUMNS,
        ]].copy()
        component_frame["target_center_log10"] = target_center
        for name, values in components_observed.items():
            full = np.full(len(grid), np.nan, np.float32)
            full[grid_observed] = target_scale * values
            component_frame[f"{name}_capped_deterministic_component_log10"] = full
        if not getattr(args, "oof_only", False):
            component_frame.to_csv(output_dir / "H26_UNIT_COMPONENTS.csv.gz", index=False, compression="gzip")
            torch.save({"state_dict": model.state_dict(), "target_center_log10": target_center, "target_scale_log10": target_scale, "scaler": scaler.audit()}, output_dir / "H26_UNIT_MODEL.pt")
        if args.candidate == "vector_process":
            excluded_process = [
                "m6_surface_current_speed_ms", "h26_cmems_speed_ms_mean",
                "h26_cmems_vector_speed_ms", "h26_cmems_direction_sin",
                "h26_cmems_direction_cos", "h26_cmems_abs_vorticity_sinv",
            ]
        elif args.candidate == "derived_dynamics":
            excluded_process = [
                "m6_surface_current_speed_ms", "h26_cmems_u_ms_mean",
                "h26_cmems_v_ms_mean", "h26_cmems_speed_ms_mean",
                "h26_cmems_vector_speed_ms", "h26_cmems_direction_sin",
                "h26_cmems_direction_cos", "h26_cmems_vorticity_sinv",
            ]
        elif args.candidate == "physics_constrained":
            excluded_process = [
                "m6_surface_current_speed_ms", "h26_cmems_u_ms_mean",
                "h26_cmems_v_ms_mean", "h26_cmems_speed_ms_mean",
                "h26_cmems_vector_speed_ms", "h26_cmems_direction_sin",
                "h26_cmems_direction_cos", "h26_cmems_vorticity_sinv",
            ]
        else:
            excluded_process = [
                "h26_cmems_u_ms_mean", "h26_cmems_v_ms_mean", "h26_cmems_speed_ms_mean",
                "h26_cmems_vector_speed_ms", "h26_cmems_eke_m2s2",
                "h26_cmems_convergence_sinv", "h26_cmems_vorticity_sinv",
                "h26_cmems_mld_m_mean", "h26_cmems_mld_m_time_sd",
                "h26_cmems_direction_sin", "h26_cmems_direction_cos",
                "h26_cmems_retention_days_per_100km",
            ]
        excluded_background = (
            []
            if args.candidate in {
                "h12_background_compatible",
                "h12_background_shared_linear",
                "h12_h26_hybrid_shared_linear",
            }
            else [
                "m6_global_annual_mean_2m_air_temperature_c_2020",
                "m6_global_wind10m_ms_2020",
                "m6_global_surface_downward_shortwave_w_m2_2020",
            ]
        )
        audit = {
            "status": "H26_MARINE_PROCESS_DEVELOPMENT_NOT_RELEASE",
            "candidate": args.candidate,
            "source_arm": "S0", "observation_arm": arm, "observation_mode": "fully_decoupled",
            "domain": DOMAIN, "seed": int(args.seed),
            "outer_split": split_audit,
            "fold_local_aoa": aoa_audit,
            "inner_split": inner_split_audit,
            "input": input_audit, "process_scaler": scaler.audit(),
            "branches": {name: [scaler.names[index] for index in index_list] for name, index_list in branch_indices.items()},
            "physics_direction_constraints": {
                "active": bool(args.candidate in {
                    "physics_constrained", "transport_graph_signed_process",
                    "transport_graph_signed_no_convergence_process",
                    "transport_graph_signed_persistent_convergence_process",
                    "transport_graph_shared_signed_process",
                    "transport_graph_shared_signed_accumulation_process",
                    "transport_path_potential_shared_directional",
                }),
                "kind": (
                    "shared_global_signed_bayesian_additive_layer"
                    if args.candidate in {
                        "transport_graph_shared_signed_process",
                        "transport_graph_shared_signed_accumulation_process",
                        "transport_path_potential_shared_directional",
                    }
                    else "global_signed_monotone_additive_branch"
                ),
                "feature_orientation": (
                    {
                        scaler.names[index]: PHYSICS_ORIENTATION[scaler.names[index]]
                        for indices in branch_indices.values() for index in indices
                    }
                    if args.candidate in {
                        "transport_graph_signed_process",
                        "transport_graph_signed_no_convergence_process",
                        "transport_graph_signed_persistent_convergence_process",
                        "transport_graph_shared_signed_process",
                        "transport_graph_shared_signed_accumulation_process",
                        "transport_path_potential_shared_directional",
                    }
                    else {
                        scaler.names[index]: PHYSICS_ORIENTATION[scaler.names[index]]
                        for index in branch_indices["retention"]
                    } if args.candidate == "physics_constrained" else {}
                ),
                "regional_targets_used": False,
            },
            "excluded_from_mean_model": [
                "longitude", "latitude", "regional_labels", *excluded_process,
                *PROVIDER_DIAGNOSTIC_FEATURES,
                *excluded_background,
            ],
            "target_center_log10": target_center, "target_scale_log10": target_scale,
            "known_observation_uncertainty": {
                "fields": ["sizecorr_known_sd_log10", "unit_sd_log10"],
                "rows_with_neither_field_finite": int((~known_any).sum()),
                "missing_both_imputed_from_fit_median_log10": known_imputation,
                "calibrated_sd_multiplier": detection_sd_multiplier,
                "affects_environmental_q50": False,
            },
            "observation_scale_model": scale_audit,
            "observation_scale_calibration_predictions": calibration_fit_audit,
            "known_source_observation_layer": known_source_audit,
            "metrics": {
                **metrics,
                f"{metric_prefix}_coverage95": coverage,
                f"{metric_prefix}_wis_log10": wis,
                f"aoa_in_{metric_prefix}_coverage95": aoa_coverage,
                f"aoa_in_{metric_prefix}_wis_log10": aoa_wis,
                "median_known_detection_sd_log10": detection_sd,
                **conditional_metrics,
            },
            "map": {
                "oof_only": bool(getattr(args, "oof_only", False)),
                "observed_cmems_latent_q50_spatial_q02_q50_q98_log10": (
                    None if getattr(args, "oof_only", False)
                    else np.quantile(grid_mean[grid_observed], [0.02, 0.50, 0.98]).tolist()
                ),
                "cmems_missing_grid_predictions": int((~grid_observed).sum()),
                "fold_local_aoa_status_counts": {
                    str(key): int(value)
                    for key, value in pd.Series(grid_aoa_status).value_counts().items()
                },
            },
            "regional_diagnostic": regional_diagnostic(output_grid), "fit": fit_audit,
            "limits": [
                f"outer {args.validation_scheme} response is untouched by fitting, early stopping, target scaling, and residual calibration",
                "development fold result only; formal multi-fold and seed aggregation remains required",
                "CMEMS source_missing/outside_physical_support rows are retained with NA predictions, never zero-filled or interpolated",
                "detection metadata changes interval width only; paired environmental q50 is identical by construction",
                "equatorial/subtropical values are post-fit diagnostics, never a loss target", "GP is off",
            ],
        }
        (output_dir / "H26_UNIT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "H26_MARINE_PROCESS_DOMAIN_COMPLETED", "candidate": args.candidate, "validation_scheme": args.validation_scheme, "fold": int(args.fold), "metrics": metrics}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training", type=Path, required=True, help="CMEMS-enriched H26 training table")
    result.add_argument("--prediction", type=Path, required=True, help="CMEMS-enriched H26 prediction grid")
    result.add_argument("--observation", type=Path, required=True)
    result.add_argument("--caoa", type=Path, required=True)
    result.add_argument("--bridge-nodes", type=Path)
    result.add_argument(
        "--observation-scale-model",
        choices=(
            "legacy_global", "decomposed_global", "bridge_mem",
            "fixed_reported_propagation", "bridge_mem_fixed_reported",
        ),
        default="legacy_global",
        help="Variance-only observation layer; none of these modes may alter environmental q50",
    )
    result.add_argument("--domain", choices=(DOMAIN,), default=DOMAIN)
    result.add_argument(
        "--candidate",
        choices=(
            "vector_process", "derived_dynamics", "physics_constrained",
            "minimal_physics", "scalar_same_support",
            "transport_graph_signed_process",
            "transport_graph_signed_no_convergence_process",
            "transport_graph_signed_persistent_convergence_process",
            "transport_graph_shared_signed_process",
            "transport_graph_shared_signed_accumulation_process",
            "transport_path_potential_shared_directional",
            "h12_background_compatible",
            "h12_background_shared_linear",
            "h12_h26_hybrid_shared_linear",
        ),
        default="vector_process",
    )
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--folds-csv", type=Path, required=True)
    result.add_argument("--fold", choices=range(1, 11), type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument(
        "--validation-scheme",
        choices=(
            "strict_spatial_block", "strict_source_group", "strict_spatial_source_purged",
            "random_record", "random_site_event",
            "random_record_10fold", "random_site_event_10fold",
        ),
        default="strict_spatial_source_purged",
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, default=180)
    result.add_argument("--posterior-draws", type=int, default=24)
    result.add_argument(
        "--calibration-crossfits",
        type=int,
        default=0,
        help="Use >=2 source-group cross-fits inside the outer train set for variance calibration",
    )
    result.add_argument("--map-cap-log10", type=float, default=4.0)
    result.add_argument("--oof-only", action="store_true", help="Skip grid/component/model files and retain held-out predictions plus audits")
    return result


if __name__ == "__main__":
    raise SystemExit(train(parser().parse_args()))
