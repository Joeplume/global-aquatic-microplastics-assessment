"""Strict H26 freshwater comparison on real RiverATLAS network support.

The runner compares a matched S0 environmental baseline with a modular model
that adds upstream source accumulation, river transport, and storage/retention
features built from the HydroRIVERS/HydroSHEDS topology.  Both candidates use
the same freshwater records, outer fold, inner source-disjoint early stopping
split, and network-supported prediction cells.  The outer held-out response is
used once for evaluation only.

Observation/detection metadata never enters the environmental mean.  The two
observation arms therefore share byte-identical q50 predictions and differ
only in future-observation interval width.  Every freshwater grid row is kept;
rows without usable network/source support carry an explicit status and NA
process predictions instead of zero filling or interpolation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

import build_h23_caoa_prototype as caoa
import run_h24_simple_protocol_preview as h24
import run_h26_direct_s0_s3 as direct
import run_h26_hydro_residual_booster as hydro
import run_h26_marine_vector_modular_bnn as strict
import run_h26_modular_process_bnn as modular


DOMAIN = "reference_surface_freshwater"
NETWORK_STATUS = "h26_river_network_status"
SOURCE_STATUS = "h26_upstream_source_status"
SUPPORTED_NETWORK_STATES = ("observed", "terminal_outlet")

BASE_BRANCHES = {
    "source": (
        "night_lights_radiance_2020",
        "m6_road_density_m_per_km2",
        "spot_land_plastic_pressure_2020",
        "river_to_sea_microplastic_output_kg_year",
        "m6_modis2019_strict_cropland_pct_0p5deg",
        "m6_ghsl2020_nonresidential_builtup_pct_0p5deg",
    ),
    "transport": (
        "annual_runoff_mm_2020",
        "m6_annual_max_daily_runoff_mm",
        "m6_global_wind10m_ms_2020",
    ),
    "retention": (
        "m6_modis2020_terrestrial_ndvi_annual_mean_0p5deg",
    ),
    "background": (
        "m6_global_annual_mean_2m_air_temperature_c_2020",
        "m6_global_surface_downward_shortwave_w_m2_2020",
    ),
}

NETWORK_SOURCE_FEATURES = (
    "h26_upstream_spot_load_proxy",
    "h26_upstream_nightlight_load_proxy",
    "h26_upstream_road_load_proxy",
    "h26_upstream_population_load_proxy",
)
NETWORK_NORMALIZED_SOURCE_FEATURES = (
    "h26_upstream_spot_area_mean",
    "h26_upstream_nightlight_area_mean",
    "h26_upstream_road_area_mean",
    "h26_upstream_population_area_mean",
)
NETWORK_TRANSPORT_FEATURES = (
    "h26_network_upstream_cell_count",
    "h26_network_upstream_area_km2",
    "h26_river_discharge_cms_max",
    "h26_river_dist_down_km_min",
    "h26_river_flow_order_max",
    "h26_river_slope_deg_lengthmean",
    "h26_network_flow_east_unit",
    "h26_network_flow_north_unit",
)
NETWORK_NORMALIZED_TRANSPORT_FEATURES = tuple(
    name for name in NETWORK_TRANSPORT_FEATURES
    if name not in {"h26_network_upstream_cell_count", "h26_network_upstream_area_km2"}
)
NETWORK_NODIRECTION_TRANSPORT_FEATURES = tuple(
    name for name in NETWORK_NORMALIZED_TRANSPORT_FEATURES
    if name not in {"h26_network_flow_east_unit", "h26_network_flow_north_unit"}
)
NETWORK_FIXED_SLOPE_MISSING_TRANSPORT_FEATURES = tuple(
    name for name in NETWORK_NORMALIZED_TRANSPORT_FEATURES
    if name != "h26_river_slope_deg_lengthmean"
)
NETWORK_RETENTION_FEATURES = (
    "h26_upstream_lake_pct_lengthmean",
    "h26_upstream_lake_volume_mcm_max",
    "h26_upstream_reservoir_volume_mcm_max",
    "h26_regulation_pct_lengthmean",
)
NETWORK_BRANCHES = {
    "source": (*BASE_BRANCHES["source"], *NETWORK_SOURCE_FEATURES),
    "transport": (*BASE_BRANCHES["transport"], *NETWORK_TRANSPORT_FEATURES),
    "retention": (*BASE_BRANCHES["retention"], *NETWORK_RETENTION_FEATURES),
    "background": BASE_BRANCHES["background"],
}
NETWORK_NORMALIZED_BRANCHES = {
    "source": (*BASE_BRANCHES["source"], *NETWORK_NORMALIZED_SOURCE_FEATURES),
    "transport": (*BASE_BRANCHES["transport"], *NETWORK_NORMALIZED_TRANSPORT_FEATURES),
    "retention": (*BASE_BRANCHES["retention"], *NETWORK_RETENTION_FEATURES),
    "background": BASE_BRANCHES["background"],
}
NETWORK_NODIRECTION_BRANCHES = {
    "source": (*BASE_BRANCHES["source"], *NETWORK_NORMALIZED_SOURCE_FEATURES),
    "transport": (*BASE_BRANCHES["transport"], *NETWORK_NODIRECTION_TRANSPORT_FEATURES),
    "retention": (*BASE_BRANCHES["retention"], *NETWORK_RETENTION_FEATURES),
    "background": BASE_BRANCHES["background"],
}
NETWORK_FIXED_SLOPE_MISSING_BRANCHES = {
    "source": (*BASE_BRANCHES["source"], *NETWORK_NORMALIZED_SOURCE_FEATURES),
    "transport": (*BASE_BRANCHES["transport"], *NETWORK_FIXED_SLOPE_MISSING_TRANSPORT_FEATURES),
    "retention": (*BASE_BRANCHES["retention"], *NETWORK_RETENTION_FEATURES),
    "background": BASE_BRANCHES["background"],
}
NETWORK_LOAD_ONLY_FEATURES = tuple(
    dict.fromkeys((*NETWORK_SOURCE_FEATURES, *NETWORK_TRANSPORT_FEATURES, *NETWORK_RETENTION_FEATURES))
)
NETWORK_NORMALIZED_ONLY_FEATURES = tuple(
    dict.fromkeys((
        *NETWORK_NORMALIZED_SOURCE_FEATURES,
        *NETWORK_NORMALIZED_TRANSPORT_FEATURES,
        *NETWORK_RETENTION_FEATURES,
    ))
)
NETWORK_NODIRECTION_ONLY_FEATURES = tuple(
    dict.fromkeys((
        *NETWORK_NORMALIZED_SOURCE_FEATURES,
        *NETWORK_NODIRECTION_TRANSPORT_FEATURES,
        *NETWORK_RETENTION_FEATURES,
    ))
)
NETWORK_FIXED_SLOPE_MISSING_ONLY_FEATURES = tuple(
    dict.fromkeys((
        *NETWORK_NORMALIZED_SOURCE_FEATURES,
        *NETWORK_FIXED_SLOPE_MISSING_TRANSPORT_FEATURES,
        *NETWORK_RETENTION_FEATURES,
    ))
)
NETWORK_LOG_FEATURES = {
    "h26_upstream_spot_load_proxy",
    "h26_upstream_nightlight_load_proxy",
    "h26_upstream_road_load_proxy",
    "h26_upstream_population_load_proxy",
    "h26_network_upstream_cell_count",
    "h26_network_upstream_area_km2",
    "h26_river_discharge_cms_max",
    "h26_river_dist_down_km_min",
    "h26_upstream_lake_volume_mcm_max",
    "h26_upstream_reservoir_volume_mcm_max",
}
caoa.LOG_FEATURES.update(NETWORK_LOG_FEATURES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def branches_for_candidate(candidate: str) -> dict[str, tuple[str, ...]]:
    if candidate == "s0_same_support":
        return BASE_BRANCHES
    if candidate == "network_process":
        return NETWORK_BRANCHES
    if candidate == "network_normalized":
        return NETWORK_NORMALIZED_BRANCHES
    if candidate == "network_nodirection":
        return NETWORK_NODIRECTION_BRANCHES
    if candidate == "network_source_monotone":
        return NETWORK_NODIRECTION_BRANCHES
    if candidate == "network_normalized_fixed_slope_missing":
        return NETWORK_FIXED_SLOPE_MISSING_BRANCHES
    raise RuntimeError(f"Unknown freshwater candidate: {candidate}")


def network_features_for_candidate(candidate: str) -> tuple[str, ...]:
    if candidate == "network_process":
        return NETWORK_LOAD_ONLY_FEATURES
    if candidate == "network_normalized":
        return NETWORK_NORMALIZED_ONLY_FEATURES
    if candidate == "network_nodirection":
        return NETWORK_NODIRECTION_ONLY_FEATURES
    if candidate == "network_source_monotone":
        return NETWORK_NODIRECTION_ONLY_FEATURES
    if candidate == "network_normalized_fixed_slope_missing":
        return NETWORK_FIXED_SLOPE_MISSING_ONLY_FEATURES
    return ()


def requested_features(branches: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for values in branches.values() for name in values))


def network_support(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame[NETWORK_STATUS].astype(str).isin(SUPPORTED_NETWORK_STATES)
        & frame[SOURCE_STATUS].astype(str).eq("observed")
    ).to_numpy()


def make_branch_indices(
    names: tuple[str, ...],
    branches: dict[str, tuple[str, ...]],
) -> dict[str, list[int]]:
    result = {
        branch: [index for index, name in enumerate(names) if name in features]
        for branch, features in branches.items()
    }
    result = {name: indices for name, indices in result.items() if indices}
    require(
        {"source", "transport", "retention", "background"}.issubset(result),
        f"Freshwater modular branches did not all remain active: {result}",
    )
    return result


def direction_orientations_for_candidate(
    candidate: str,
    names: tuple[str, ...],
    branches: dict[str, tuple[str, ...]],
) -> dict[str, list[float]]:
    """Return globally monotone feature directions aligned to branch columns."""
    if candidate != "network_source_monotone":
        return {}
    source = set(branches["source"])
    return {"source": [1.0 for name in names if name in source]}


def load_inputs(
    args: argparse.Namespace,
    branches: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features = requested_features(branches)
    train_header = set(pd.read_csv(args.training, nrows=0).columns)
    grid_header = set(pd.read_csv(args.prediction, nrows=0).columns)
    required_train = {"stable_record_id", NETWORK_STATUS, SOURCE_STATUS, *features}
    required_grid = {
        "prediction_id", "longitude", "latitude", "prediction_domain",
        NETWORK_STATUS, SOURCE_STATUS, *features,
    }
    require(required_train.issubset(train_header), f"Training table missing {sorted(required_train - train_header)}")
    require(required_grid.issubset(grid_header), f"Prediction table missing {sorted(required_grid - grid_header)}")
    training = pd.read_csv(
        args.training,
        usecols=["stable_record_id", NETWORK_STATUS, SOURCE_STATUS, *features],
        low_memory=False,
    )
    grid = pd.read_csv(
        args.prediction,
        usecols=[
            "prediction_id", "longitude", "latitude", "prediction_domain",
            NETWORK_STATUS, SOURCE_STATUS, *features,
        ],
        low_memory=False,
    )
    observation = pd.read_csv(args.observation, low_memory=False)
    observation_columns = (
        "stable_record_id", "waterbody", "source_group", "observed_log10",
        "sizecorr_known_sd_log10", "unit_sd_log10", "sampling_year_effective_v1_3",
        "longitude", "latitude", "sampling_event_id", "sampling_event_key",
        "field_sampling_event_id",
        "detection_method", "qaqc_total_score", "sizecorr_lower_used_um",
        "sizecorr_evidence_tier",
    )
    required_observation = {"stable_record_id", "waterbody", "source_group", "observed_log10"}
    require(required_observation.issubset(observation.columns),
            f"Observation table missing {sorted(required_observation - set(observation.columns))}")
    observation = observation[[name for name in observation_columns if name in observation]].copy()
    records_all = training.merge(observation, on="stable_record_id", how="inner", validate="one_to_one")
    records_all = records_all.loc[records_all["waterbody"].eq(DOMAIN)].reset_index(drop=True)
    record_supported = network_support(records_all)
    records = records_all.loc[record_supported].reset_index(drop=True)
    require(len(records) >= 1000, f"Too few supported freshwater records: {len(records):,}")

    grid = grid.loc[grid["prediction_domain"].eq(DOMAIN)].copy()
    grid = grid.rename(columns={"prediction_domain": "waterbody"}).reset_index(drop=True)
    require(len(grid) > 0, "No freshwater prediction rows")
    grid_supported = network_support(grid)
    caoa_frame = pd.read_csv(args.caoa, usecols=["prediction_id", "caoa_status"], low_memory=False)
    grid = grid.merge(caoa_frame, on="prediction_id", how="left", validate="one_to_one")
    require(grid["caoa_status"].notna().all(), "C-AOA join did not close")

    network_features = network_features_for_candidate(args.candidate)
    if network_features:
        train_missing = {
            name: int((~np.isfinite(numeric(records[name]))).sum())
            for name in network_features
        }
        grid_missing = {
            name: int((~np.isfinite(numeric(grid.loc[grid_supported, name]))).sum())
            for name in network_features
        }
        require(sum(train_missing.values()) <= 4,
                f"Supported freshwater records retain too many missing network values: {train_missing}")
        require(sum(grid_missing.values()) <= 8,
                f"Supported freshwater grid retains too many missing network values: {grid_missing}")
    else:
        train_missing, grid_missing = {}, {}

    return records, grid, {
        "candidate": args.candidate,
        "records_before_network_support": int(len(records_all)),
        "records_supported": int(len(records)),
        "records_excluded": int(len(records_all) - len(records)),
        "record_network_status_counts": {
            str(key): int(value) for key, value in records_all[NETWORK_STATUS].value_counts(dropna=False).items()
        },
        "record_source_status_counts": {
            str(key): int(value) for key, value in records_all[SOURCE_STATUS].value_counts(dropna=False).items()
        },
        "grid_rows_retained": int(len(grid)),
        "grid_supported": int(grid_supported.sum()),
        "grid_unsupported": int((~grid_supported).sum()),
        "grid_network_status_counts": {
            str(key): int(value) for key, value in grid[NETWORK_STATUS].value_counts(dropna=False).items()
        },
        "grid_source_status_counts": {
            str(key): int(value) for key, value in grid[SOURCE_STATUS].value_counts(dropna=False).items()
        },
        "network_feature_missing_supported_records": train_missing,
        "network_feature_missing_supported_grid": grid_missing,
        "support_policy": "river network observed_or_terminal AND upstream source observed",
    }


def target_location_scale(
    records: pd.DataFrame,
    observed: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[float, float]:
    local = records.iloc[fit_rows].assign(_target=observed[fit_rows])
    medians = local.groupby("source_group", observed=True)["_target"].median().to_numpy(float)
    center = float(np.median(medians))
    scale = float(1.4826 * np.median(np.abs(medians - center)))
    if not np.isfinite(scale) or scale < 0.35:
        scale = max(0.35, float(np.std(medians)))
    return center, scale


class DirectionalFreshwaterBNN(nn.Module):
    """Modular BNN with signed monotone effects in selected process branches."""

    def __init__(
        self,
        branch_indices: dict[str, list[int]],
        direction_orientations: dict[str, list[float]],
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
            self.register_buffer(f"indices_{name}", torch.as_tensor(branch_indices[name], dtype=torch.long))
            if name in direction_orientations:
                self.branches[name] = strict.MonotoneBayesianBranch(
                    len(branch_indices[name]),
                    direction_orientations[name],
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
        raw = torch.stack(list(parts.values()), dim=0).sum(dim=0)
        cap = self.total_cap_standardized
        return cap * torch.tanh(raw / cap)

    def kl(self) -> torch.Tensor:
        return sum(branch.kl() for branch in self.branches.values())


def fit_candidate_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    branch_indices: dict[str, list[int]],
    target_scale: float,
    *,
    candidate: str,
    seed: int,
    epochs: int,
    total_cap_log10: float,
    device: torch.device,
    direction_orientations: dict[str, list[float]] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    if direction_orientations is None:
        direction_orientations = (
            {"source": [1.0] * len(branch_indices["source"])}
            if candidate == "network_source_monotone"
            else {}
        )
    if not direction_orientations:
        return strict.fit_model(
            x, y, weights, train_rows, validation_rows, branch_indices, target_scale,
            candidate="vector_process", physics_signs=[], seed=seed, epochs=epochs,
            total_cap_log10=total_cap_log10, device=device,
        )
    h24.seed_all(seed)
    model = DirectionalFreshwaterBNN(
        branch_indices, direction_orientations, target_scale, total_cap_log10
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0022, weight_decay=2e-5)
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor(y.astype(np.float32), device=device)
    fitted_weight = (weights / np.mean(weights)).astype(np.float32)
    weight_tensor = torch.as_tensor(fitted_weight, device=device)
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
            raw = torch.stack(list(parts.values()), dim=0).sum(dim=0)
            cap = model.total_cap_standardized
            prediction = cap * torch.tanh(raw / cap)
            point = F.smooth_l1_loss(prediction, y_tensor[batch], reduction="none", beta=0.45)
            data_loss = (
                (point * weight_tensor[batch]).sum()
                / weight_tensor[batch].sum().clamp_min(1e-8)
            )
            kl = 0.018 * model.kl() / max(1, len(train_rows))
            shrinkage = 0.006 * torch.stack(
                [value.square().mean() for value in parts.values()]
            ).sum()
            (data_loss + kl + shrinkage).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x_tensor[validation_tensor], sample=False).cpu().numpy()
        score = h24.weighted_rmse(prediction, y[validation_rows], weights[validation_rows])
        if score < best_rmse - 1e-5:
            best_epoch, best_rmse = epoch + 1, score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elif epoch + 1 - best_epoch >= patience:
            break
    require(best_state is not None, "Source-monotone freshwater BNN produced no finite checkpoint")
    model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    model.eval()
    return model, {
        "device": str(device), "best_epoch": int(best_epoch),
        "inner_source_validation_rmse_standardized": float(best_rmse),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "branch_caps_log10": modular.BRANCH_CAP_LOG10,
        "total_cap_log10": float(total_cap_log10),
        "direction_constraints": direction_orientations,
        "regional_targets_used": False,
        "observation_mean_terms_used": False, "gp_used": False,
    }


def crossfit_calibration_predictions(
    x: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    records: pd.DataFrame,
    outer_train_rows: np.ndarray,
    branch_indices: dict[str, list[int]],
    *,
    candidate: str,
    seed: int,
    epochs: int,
    total_cap_log10: float,
    device: torch.device,
    posterior_draws: int,
    splits: int,
    direction_orientations: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if direction_orientations is None:
        direction_orientations = (
            {"source": [1.0] * len(branch_indices["source"])}
            if candidate == "network_source_monotone"
            else {}
        )
    if not direction_orientations:
        return strict.crossfit_calibration_predictions(
            x, observed, weights, records, outer_train_rows, branch_indices,
            candidate="vector_process", physics_signs=[], seed=seed, epochs=epochs,
            total_cap_log10=total_cap_log10, device=device,
            posterior_draws=posterior_draws, splits=splits,
        )
    require(splits >= 3, "Calibration cross-fitting requires at least three splits")
    groups = records.iloc[outer_train_rows]["source_group"].fillna("__missing__").astype(str).to_numpy()
    require(len(np.unique(groups)) >= splits * 3, "Too few sources for calibration cross-fitting")
    splitter = GroupKFold(n_splits=int(splits))
    q50 = np.full(len(outer_train_rows), np.nan, np.float32)
    sd = np.full(len(outer_train_rows), np.nan, np.float32)
    fold_audits: list[dict[str, object]] = []
    for crossfit_fold, (model_local, calibration_local) in enumerate(
        splitter.split(np.arange(len(outer_train_rows)), groups=groups), start=1
    ):
        model_rows = outer_train_rows[model_local].astype(np.int64)
        calibration_rows = outer_train_rows[calibration_local].astype(np.int64)
        inner_fit, inner_validation, split_audit = strict.inner_source_validation_split(
            records, model_rows, int(seed) + crossfit_fold * 907
        )
        center, scale = target_location_scale(records, observed, inner_fit)
        local_y = ((observed - center) / scale).astype(np.float32)
        local_model, local_fit = fit_candidate_model(
            x, local_y, weights, inner_fit, inner_validation, branch_indices, scale,
            candidate=candidate, seed=int(seed) + crossfit_fold * 1009,
            epochs=int(epochs), total_cap_log10=float(total_cap_log10), device=device,
            direction_orientations=direction_orientations,
        )
        local_q50, _, local_sd, _ = strict.predict(
            local_model, x[calibration_rows], max(2, int(posterior_draws))
        )
        q50[calibration_local] = center + scale * local_q50
        sd[calibration_local] = scale * local_sd
        fit_sources = set(records.iloc[model_rows]["source_group"].fillna("__missing__").astype(str))
        held_sources = set(records.iloc[calibration_rows]["source_group"].fillna("__missing__").astype(str))
        require(not (fit_sources & held_sources), "Calibration cross-fit retained overlapping sources")
        fold_audits.append({
            "crossfit_fold": int(crossfit_fold),
            "model_records": int(len(model_rows)),
            "calibration_records": int(len(calibration_rows)),
            "source_overlap": 0,
            "target_center_log10": center, "target_scale_log10": scale,
            "inner_early_stopping": split_audit, "fit": local_fit,
        })
        del local_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(np.isfinite(q50).all() and np.isfinite(sd).all(),
            "Calibration cross-fitting left missing predictions")
    return outer_train_rows.astype(np.int64), q50, sd, {
        "kind": "outer_train_source_group_crossfit_for_variance_only",
        "splits": int(splits), "records": int(len(outer_train_rows)),
        "sources": int(len(np.unique(groups))),
        "outer_heldout_response_used": False,
        "affects_main_environmental_fit": False,
        "folds": fold_audits,
    }


def train(args: argparse.Namespace) -> int:
    branches = branches_for_candidate(args.candidate)
    records, grid, input_audit = load_inputs(args, branches)
    weights = h24.source_equal_weights(records["source_group"])
    outer_train_rows, heldout_rows, outer_audit = strict.choose_outer_split(records, args)
    validation_site_event_key, _ = strict.build_site_event_key(records)
    fit_rows, early_stop_rows, inner_audit = strict.inner_source_validation_split(
        records, outer_train_rows, int(args.seed) + int(args.fold) * 101
    )
    observed = numeric(records["observed_log10"])
    target_center, target_scale = target_location_scale(records, observed, fit_rows)
    y = ((observed - target_center) / target_scale).astype(np.float32)

    features = requested_features(branches)
    grid_supported = network_support(grid)
    scaler = h24.ProcessScaler.fit(
        records,
        grid.loc[grid_supported].reset_index(drop=True),
        features,
        fit_rows,
    )
    branch_indices = make_branch_indices(scaler.names, branches)
    direction_orientations = direction_orientations_for_candidate(
        args.candidate, scaler.names, branches
    )
    for branch, orientations in direction_orientations.items():
        require(
            branch in branch_indices and len(orientations) == len(branch_indices[branch]),
            f"Direction orientations do not align with {branch}: "
            f"{len(orientations)} vs {len(branch_indices.get(branch, []))}",
        )
    x = scaler.transform(records)
    x_grid = scaler.transform(grid.loc[grid_supported].reset_index(drop=True))
    (
        heldout_aoa_status,
        heldout_aoa_distance,
        supported_grid_aoa_status,
        supported_grid_aoa_distance,
        aoa_audit,
    ) = strict.fold_local_aoa(x[outer_train_rows], x[heldout_rows], x_grid)
    heldout_aoa_in = heldout_aoa_status == "AOA_in"
    grid_aoa_status = np.full(len(grid), "not_assessed", dtype=object)
    grid_aoa_distance = np.full(len(grid), np.nan, dtype=float)
    grid_aoa_status[grid_supported] = supported_grid_aoa_status
    grid_aoa_distance[grid_supported] = supported_grid_aoa_distance
    device = strict.select_device(args.device)
    model, fit_audit = fit_candidate_model(
        x, y, weights, fit_rows, early_stop_rows, branch_indices, target_scale,
        candidate=args.candidate, seed=int(args.seed),
        epochs=int(args.epochs), total_cap_log10=float(args.map_cap_log10), device=device,
        direction_orientations=direction_orientations,
    )

    heldout_standardized, heldout_deterministic_standardized, heldout_sd_standardized, _ = strict.predict(
        model, x[heldout_rows], max(2, int(args.posterior_draws))
    )
    heldout_mean = target_center + target_scale * heldout_standardized
    heldout_deterministic = target_center + target_scale * heldout_deterministic_standardized
    heldout_latent_sd = target_scale * heldout_sd_standardized
    heldout_observed = observed[heldout_rows]
    heldout_weight = weights[heldout_rows]
    residual = heldout_observed - heldout_mean

    if int(args.calibration_crossfits) >= 3:
        calibration_rows, calibration_mean, calibration_latent_sd, calibration_audit = (
            crossfit_calibration_predictions(
                x, observed, weights, records, outer_train_rows, branch_indices,
                candidate=args.candidate,
                seed=int(args.seed) + int(args.fold) * 101,
                epochs=int(args.epochs), total_cap_log10=float(args.map_cap_log10),
                device=device, posterior_draws=int(args.posterior_draws),
                splits=int(args.calibration_crossfits),
                direction_orientations=direction_orientations,
            )
        )
    else:
        calibration_rows = early_stop_rows
        calibration_standardized, _, calibration_sd_standardized, _ = strict.predict(
            model, x[calibration_rows], max(2, int(args.posterior_draws))
        )
        calibration_mean = target_center + target_scale * calibration_standardized
        calibration_latent_sd = target_scale * calibration_sd_standardized
        calibration_audit = {
            "kind": "single_inner_source_holdout_for_variance_only",
            "outer_heldout_response_used": False,
        }
    calibration_residual = observed[calibration_rows] - calibration_mean
    calibration_weight = weights[calibration_rows]

    size_sd = numeric(records["sizecorr_known_sd_log10"]) if "sizecorr_known_sd_log10" in records else np.full(len(records), np.nan)
    unit_sd = numeric(records["unit_sd_log10"]) if "unit_sd_log10" in records else np.full(len(records), np.nan)
    size_finite, unit_finite = np.isfinite(size_sd), np.isfinite(unit_sd)
    known_any = size_finite | unit_finite
    known_raw = np.sqrt(
        np.square(np.where(size_finite, size_sd, 0.0))
        + np.square(np.where(unit_finite, unit_sd, 0.0))
    )
    calibration_known = known_raw[fit_rows][known_any[fit_rows]]
    known_imputation = float(np.median(calibration_known)) if len(calibration_known) else 0.0
    known_sd = np.where(known_any, known_raw, known_imputation)
    heldout_base_sd, grid_base_sd, interval_multiplier, scale_audit = strict.observation_scale_fit(
        records, calibration_rows, heldout_rows, calibration_residual,
        calibration_latent_sd, known_sd, weights,
        mode=getattr(args, "observation_scale_model", "legacy_global"),
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
            int(args.calibration_crossfits) >= 3 and np.array_equal(calibration_rows, outer_train_rows),
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
        ) = strict.fit_known_source_observation_layer(
            records, calibration_rows, heldout_rows, calibration_residual,
            calibration_latent_sd, weights, known_sd,
            fixed_reported_propagation=(
                getattr(args, "observation_scale_model", "legacy_global")
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
    grid_components: dict[str, np.ndarray] = {}
    if not getattr(args, "oof_only", False):
        grid_standardized, grid_deterministic_standardized, grid_sd_standardized, grid_components = strict.predict(
            model, x_grid, int(args.posterior_draws)
        )
        grid_mean[grid_supported] = target_center + target_scale * grid_standardized
        grid_deterministic[grid_supported] = target_center + target_scale * grid_deterministic_standardized
        grid_latent_sd[grid_supported] = target_scale * grid_sd_standardized

    rmse = math.sqrt(strict.weighted_metric(np.square(residual), heldout_weight))
    metric_prefix = args.validation_scheme
    metrics = {
        f"{metric_prefix}_r2": float(r2_score(heldout_observed, heldout_mean, sample_weight=heldout_weight)),
        f"{metric_prefix}_rmse_log10": float(rmse),
        f"{metric_prefix}_mae_log10": strict.weighted_metric(np.abs(residual), heldout_weight),
        "posterior_q50_minus_deterministic_median_abs_log10": float(np.median(np.abs(heldout_mean - heldout_deterministic))),
    }
    heldout_aoa_count = int(heldout_aoa_in.sum())
    if heldout_aoa_count >= 2:
        metrics.update({
            f"aoa_in_{metric_prefix}_records": heldout_aoa_count,
            f"aoa_in_{metric_prefix}_fraction": float(heldout_aoa_in.mean()),
            f"aoa_in_{metric_prefix}_r2": float(r2_score(
                heldout_observed[heldout_aoa_in], heldout_mean[heldout_aoa_in],
                sample_weight=heldout_weight[heldout_aoa_in],
            )),
            f"aoa_in_{metric_prefix}_rmse_log10": math.sqrt(strict.weighted_metric(
                np.square(residual[heldout_aoa_in]), heldout_weight[heldout_aoa_in],
            )),
            f"aoa_in_{metric_prefix}_mae_log10": strict.weighted_metric(
                np.abs(residual[heldout_aoa_in]), heldout_weight[heldout_aoa_in],
            ),
        })
    elif getattr(args, "allow_small_aoa_heldout", False):
        metrics.update({
            f"aoa_in_{metric_prefix}_records": heldout_aoa_count,
            f"aoa_in_{metric_prefix}_fraction": float(heldout_aoa_in.mean()),
            f"aoa_in_{metric_prefix}_r2": float("nan"),
            f"aoa_in_{metric_prefix}_rmse_log10": float("nan"),
            f"aoa_in_{metric_prefix}_mae_log10": float("nan"),
        })
    else:
        require(False, "Fold-local AOA contains fewer than two held-out records")
    for arm, interval_aware in (("protocol_aware", True), ("detection_unaware", False)):
        output_dir = args.run_root / "S0" / arm / DOMAIN
        output_dir.mkdir(parents=True, exist_ok=True)
        local_known = calibrated_known_sd[heldout_rows] if interval_aware else np.zeros(len(heldout_rows))
        heldout_future_sd = interval_multiplier * np.sqrt(
            np.square(heldout_latent_sd) + np.square(heldout_base_sd) + np.square(local_known)
        )
        coverage = float(np.average(np.abs(residual) <= 1.96 * heldout_future_sd, weights=heldout_weight))
        wis = strict.weighted_interval_score(heldout_observed, heldout_mean, heldout_future_sd, heldout_weight)
        if heldout_aoa_count >= 2:
            aoa_coverage = float(np.average(
                np.abs(residual[heldout_aoa_in]) <= 1.96 * heldout_future_sd[heldout_aoa_in],
                weights=heldout_weight[heldout_aoa_in],
            ))
            aoa_wis = strict.weighted_interval_score(
                heldout_observed[heldout_aoa_in], heldout_mean[heldout_aoa_in],
                heldout_future_sd[heldout_aoa_in], heldout_weight[heldout_aoa_in],
            )
        else:
            aoa_coverage = float("nan")
            aoa_wis = float("nan")
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
                f"known_source_{metric_prefix}_rmse_log10": math.sqrt(strict.weighted_metric(
                    np.square(conditional_residual), conditional_weight,
                )),
                f"known_source_{metric_prefix}_coverage95": float(np.average(
                    np.abs(conditional_residual)
                    <= 1.96 * conditional_future_sd[known_source_available],
                    weights=conditional_weight,
                )),
                f"known_source_{metric_prefix}_wis_log10": strict.weighted_interval_score(
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
                    f"known_source_aoa_in_{metric_prefix}_rmse_log10": math.sqrt(strict.weighted_metric(
                        np.square(conditional_aoa_residual), conditional_aoa_weight,
                    )),
                    f"known_source_aoa_in_{metric_prefix}_coverage95": float(np.average(
                        np.abs(conditional_aoa_residual)
                        <= 1.96 * conditional_future_sd[conditional_aoa],
                        weights=conditional_aoa_weight,
                    )),
                    f"known_source_aoa_in_{metric_prefix}_wis_log10": strict.weighted_interval_score(
                        heldout_observed[conditional_aoa], conditional_mean[conditional_aoa],
                        conditional_future_sd[conditional_aoa], conditional_aoa_weight,
                    ),
                })
        detection_sd = float(np.median(calibrated_known_sd[fit_rows])) if interval_aware else 0.0
        grid_future_sd = np.full(len(grid), np.nan, np.float32)
        grid_future_sd[grid_supported] = interval_multiplier * np.sqrt(
            np.square(grid_latent_sd[grid_supported]) + grid_base_sd**2 + detection_sd**2
        )

        output_grid = grid[[
            "prediction_id", "longitude", "latitude", "waterbody", "caoa_status",
            NETWORK_STATUS, SOURCE_STATUS,
        ]].copy()
        output_grid["network_process_supported"] = grid_supported
        output_grid["latent_q50_log10"] = grid_mean
        output_grid["aoa_status"] = grid_aoa_status
        output_grid["aoa_distance"] = grid_aoa_distance
        output_grid["latent_deterministic_log10"] = grid_deterministic
        output_grid["latent_sd_log10"] = grid_latent_sd
        output_grid["base_residual_sd_log10"] = np.where(grid_supported, grid_base_sd, np.nan)
        output_grid["interval_multiplier"] = interval_multiplier
        output_grid["future_observation_sd_log10"] = grid_future_sd
        if not getattr(args, "oof_only", False):
            output_grid.to_csv(output_dir / "H26_UNIT_GRID.csv.gz", index=False, compression="gzip")

        heldout = records.iloc[heldout_rows][["stable_record_id", "waterbody", "source_group"]].copy()
        heldout[NETWORK_STATUS] = records.iloc[heldout_rows][NETWORK_STATUS].to_numpy()
        heldout[SOURCE_STATUS] = records.iloc[heldout_rows][SOURCE_STATUS].to_numpy()
        heldout["validation_site_event_key"] = validation_site_event_key.iloc[heldout_rows].to_numpy()
        heldout["outer_validation_fold"] = int(args.fold)
        heldout["observed_log10"] = heldout_observed
        heldout["aoa_status"] = heldout_aoa_status
        heldout["aoa_distance"] = heldout_aoa_distance
        heldout["predicted_q50_log10"] = heldout_mean
        heldout["predicted_deterministic_log10"] = heldout_deterministic
        heldout["latent_sd_log10"] = heldout_latent_sd
        heldout["residual_log10"] = residual
        heldout["base_residual_sd_log10"] = heldout_base_sd
        heldout["interval_multiplier"] = interval_multiplier
        heldout["future_observation_sd_log10"] = heldout_future_sd
        heldout["known_source_available"] = known_source_available
        heldout["known_source_offset_log10"] = known_source_offset
        heldout["known_source_conditional_q50_log10"] = conditional_mean
        heldout["known_source_conditional_future_sd_log10"] = conditional_future_sd
        heldout.to_csv(output_dir / "H26_UNIT_HELDOUT.csv.gz", index=False, compression="gzip")

        component_frame = grid[[
            "prediction_id", "longitude", "latitude", "waterbody",
            NETWORK_STATUS, SOURCE_STATUS,
        ]].copy()
        component_frame["network_process_supported"] = grid_supported
        component_frame["target_center_log10"] = target_center
        for name, values in grid_components.items():
            full = np.full(len(grid), np.nan, np.float32)
            full[grid_supported] = target_scale * values
            component_frame[f"{name}_capped_deterministic_component_log10"] = full
        if not getattr(args, "oof_only", False):
            component_frame.to_csv(output_dir / "H26_UNIT_COMPONENTS.csv.gz", index=False, compression="gzip")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_center_log10": target_center,
                    "target_scale_log10": target_scale,
                    "scaler": scaler.audit(),
                },
                output_dir / "H26_UNIT_MODEL.pt",
            )

        audit = {
            "status": "H26_FRESHWATER_NETWORK_DEVELOPMENT_NOT_RELEASE",
            "candidate": args.candidate,
            "source_arm": "S0", "observation_arm": arm,
            "observation_mode": "fully_decoupled", "domain": DOMAIN,
            "seed": int(args.seed), "outer_split": outer_audit, "inner_split": inner_audit,
            "fold_local_aoa": aoa_audit,
            "input": input_audit, "process_scaler": scaler.audit(),
            "branches": {
                name: [scaler.names[index] for index in indices]
                for name, indices in branch_indices.items()
            },
            "source_direction_constraint": {
                "active": bool(direction_orientations),
                "branch_orientations": direction_orientations,
                "regional_targets_used": False,
            },
            "excluded_from_mean_model": [
                "longitude", "latitude", "country_or_region_labels",
                "network_cell_index", "main_river_id", "basin_id",
                "network_support_status", "source_support_status",
                "observation_and_detection_metadata",
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
            "observation_scale_calibration_predictions": calibration_audit,
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
                "supported_grid_q02_q50_q98_log10": (
                    None if getattr(args, "oof_only", False)
                    else np.quantile(grid_mean[grid_supported], [0.02, 0.50, 0.98]).tolist()
                ),
                "all_freshwater_rows_retained": int(len(grid)),
                "unsupported_rows_with_na_process_prediction": int((~grid_supported).sum()),
                "fold_local_aoa_status_counts": {
                    str(key): int(value)
                    for key, value in pd.Series(grid_aoa_status).value_counts().items()
                },
            },
            "fit": fit_audit,
            "limits": [
                f"outer {args.validation_scheme} response is untouched by fitting, early stopping, target scaling, and interval calibration",
                "same-support comparison is a development screen; multi-seed formal validation follows only after a two-of-three-fold process improvement",
                "dominant 0.5-degree downstream routing cannot represent braided or multi-channel flow",
                "upstream source accumulation is an uncalibrated pressure proxy, not a plastic mass flux",
                "per-discharge and storage-to-discharge ratios remain support diagnostics because terminal outlets have undefined values",
                "unsupported freshwater cells are retained and explicitly marked, never zero-filled or result-masked",
                "regional outcomes are diagnostics only and never enter the loss",
                "GP is off",
            ],
        }
        (output_dir / "H26_UNIT_AUDIT.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "status": "H26_FRESHWATER_NETWORK_COMPLETED",
        "candidate": args.candidate, "validation_scheme": args.validation_scheme,
        "fold": int(args.fold), "metrics": metrics,
    }, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--candidate",
        choices=(
            "s0_same_support", "network_process", "network_normalized",
            "network_nodirection",
            "network_source_monotone",
            "network_normalized_fixed_slope_missing",
        ),
        required=True,
    )
    result.add_argument("--training", type=Path, required=True)
    result.add_argument("--prediction", type=Path, required=True)
    result.add_argument("--observation", type=Path, required=True)
    result.add_argument("--caoa", type=Path, required=True)
    result.add_argument("--folds-csv", type=Path, required=True)
    result.add_argument("--fold", type=int, choices=range(1, 11), required=True)
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
    result.add_argument("--calibration-crossfits", type=int, default=3)
    result.add_argument(
        "--observation-scale-model",
        choices=(
            "legacy_global",
            "decomposed_global",
            "fixed_reported_propagation",
            "bridge_mem",
        ),
        default="legacy_global",
        help="Variance-only observation layer; never changes environmental q50",
    )
    result.add_argument("--map-cap-log10", type=float, default=4.0)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--oof-only", action="store_true", help="Skip grid/component/model files and retain held-out predictions plus audits")
    result.add_argument(
        "--allow-small-aoa-heldout",
        action="store_true",
        help="Retain overall held-out predictions when fewer than two held-out records are AOA-in; AOA-only metrics become NA.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(train(parser().parse_args()))
