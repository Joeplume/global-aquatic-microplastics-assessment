"""Train the H26 source-transport-retention modular BNN.

The environmental mean is always decoupled from observation/detection
uncertainty.  One fit per water domain writes paired protocol-aware and
detection-unaware outputs with identical q50 and different future-observation
interval widths.
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
from torch import nn
from torch.nn import functional as F

import run_h24_simple_protocol_preview as h24
import run_h26_direct_s0_s3 as direct


SOURCE_FEATURES = {
    "night_lights_radiance_2020", "m6_road_density_m_per_km2",
    "spot_land_plastic_pressure_2020", "river_to_sea_microplastic_output_kg_year",
    "m6_modis2019_strict_cropland_pct_0p5deg",
    "m6_ghsl2020_nonresidential_builtup_pct_0p5deg",
    "ais_activity_density", "marine_aquaculture_pressure_tonnes_50km",
    "marina_pressure_h100_kg_year", "h26_upstream_population_thousands_max",
    "h26_upstream_nightlight_max",
}
TRANSPORT_FEATURES = {
    "annual_runoff_mm_2020", "m6_annual_max_daily_runoff_mm",
    "m6_surface_current_speed_ms", "m6_significant_wave_height_m",
    "positive_water_depth_m", "m6_global_wind10m_ms_2020",
    "h26_river_length_km_sum", "h26_river_discharge_cms_max",
    "h26_river_upstream_area_km2_max", "h26_river_dist_down_km_min",
    "h26_river_flow_order_max", "h26_river_slope_deg_lengthmean",
    "h26_stream_gradient_dmkm_lengthmean",
}
RETENTION_FEATURES = {
    "m6_sea_surface_salinity_psu", "h26_upstream_lake_volume_mcm_max",
    "h26_upstream_reservoir_volume_mcm_max", "h26_regulation_pct_lengthmean",
    "h26_lake_area_km2_sum", "h26_lake_volume_mcm_sum",
    "h26_reservoir_volume_mcm_sum", "h26_lake_residence_days_area_mean",
}
BRANCH_CAP_LOG10 = {"source": 1.6, "transport": 1.2, "retention": 0.9, "background": 0.8}


def branch_name(feature: str) -> str:
    if feature in SOURCE_FEATURES:
        return "source"
    if feature in TRANSPORT_FEATURES:
        return "transport"
    if feature in RETENTION_FEATURES:
        return "retention"
    return "background"


class BranchBNN(nn.Module):
    def __init__(self, input_dim: int, cap_standardized: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            h24.BayesianLinear(input_dim, 24, prior_sd=0.55),
            h24.BayesianLinear(24, 12, prior_sd=0.50),
            h24.BayesianLinear(12, 1, prior_sd=0.45),
        ])
        self.cap_standardized = float(cap_standardized)

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        values = F.silu(self.layers[0](values, sample))
        values = F.silu(self.layers[1](values, sample))
        values = self.layers[2](values, sample).reshape(-1)
        cap = self.cap_standardized
        return cap * torch.tanh(values / cap)

    def kl(self) -> torch.Tensor:
        return sum(layer.kl() for layer in self.layers)


class ModularBNN(nn.Module):
    def __init__(self, branch_indices: dict[str, list[int]], target_scale: float, total_cap_log10: float) -> None:
        super().__init__()
        self.branch_order = tuple(name for name in ("source", "transport", "retention", "background") if branch_indices.get(name))
        self.branches = nn.ModuleDict()
        for name in self.branch_order:
            self.register_buffer(f"indices_{name}", torch.as_tensor(branch_indices[name], dtype=torch.long))
            self.branches[name] = BranchBNN(len(branch_indices[name]), BRANCH_CAP_LOG10[name] / target_scale)
        self.total_cap_standardized = float(total_cap_log10 / target_scale)

    def components(self, values: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        return {
            name: self.branches[name](values.index_select(1, getattr(self, f"indices_{name}")), sample)
            for name in self.branch_order
        }

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        parts = self.components(values, sample)
        total = torch.stack(list(parts.values()), dim=0).sum(dim=0)
        cap = self.total_cap_standardized
        return cap * torch.tanh(total / cap)

    def kl(self) -> torch.Tensor:
        return sum(branch.kl() for branch in self.branches.values())


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    branch_indices: dict[str, list[int]],
    target_scale: float,
    seed: int,
    epochs: int,
    total_cap_log10: float,
) -> tuple[ModularBNN, dict[str, object]]:
    h24.seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModularBNN(branch_indices, target_scale, total_cap_log10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0022, weight_decay=2e-5)
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor(y.astype(np.float32), device=device)
    fit_weight = (weights / np.mean(weights)).astype(np.float32)
    w_tensor = torch.as_tensor(fit_weight, device=device)
    train_tensor = torch.as_tensor(train_rows, device=device)
    validation_tensor = torch.as_tensor(validation_rows, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 1709)
    batch_size = min(4096, len(train_rows))
    best_epoch = 0
    best_rmse = math.inf
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
            cap = model.total_cap_standardized
            prediction = cap * torch.tanh(raw_total / cap)
            point = F.smooth_l1_loss(prediction, y_tensor[batch], reduction="none", beta=0.45)
            data_loss = (point * w_tensor[batch]).sum() / w_tensor[batch].sum().clamp_min(1e-8)
            kl = 0.018 * model.kl() / max(1, len(train_rows))
            branch_shrinkage = 0.006 * torch.stack([value.square().mean() for value in parts.values()]).sum()
            loss = data_loss + kl + branch_shrinkage
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model(x_tensor[validation_tensor], sample=False).detach().cpu().numpy()
        rmse = h24.weighted_rmse(predicted, y[validation_rows], weights[validation_rows])
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break
    direct.require(best_state is not None, "Modular training produced no finite checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    return model, {
        "device": str(device), "best_epoch": int(best_epoch),
        "validation_source_group_holdout_rmse_standardized": float(best_rmse),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "branch_caps_log10": BRANCH_CAP_LOG10,
        "total_cap_log10": float(total_cap_log10),
        "observation_mean_terms_used": False, "gp_used": False,
    }


def predict(model: ModularBNN, x: np.ndarray, draws: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    device = next(model.parameters()).device
    mean = np.empty(len(x), dtype=np.float32)
    sd = np.empty(len(x), dtype=np.float32)
    components = {name: np.empty(len(x), dtype=np.float32) for name in model.branch_order}
    for start in range(0, len(x), 32768):
        stop = min(len(x), start + 32768)
        values = torch.as_tensor(x[start:stop], device=device)
        with torch.no_grad():
            mean[start:stop] = model(values, sample=False).cpu().numpy()
            parts = model.components(values, sample=False)
            for name, local in parts.items():
                components[name][start:stop] = local.cpu().numpy()
            samples = torch.stack([model(values, sample=True) for _ in range(draws)])
            sd[start:stop] = samples.std(dim=0, unbiased=True).cpu().numpy()
    return mean, sd, components


def train_domain(args: argparse.Namespace) -> int:
    aware_dir = args.run_root / "S0" / "protocol_aware" / args.domain
    unaware_dir = args.run_root / "S0" / "detection_unaware" / args.domain
    if (aware_dir / "H26_UNIT_AUDIT.json").exists() and (unaware_dir / "H26_UNIT_AUDIT.json").exists() and not args.force:
        print(json.dumps({"status": "SKIPPED_EXISTING", "domain": args.domain}))
        return 0
    load_args = argparse.Namespace(
        training=args.training, prediction=args.prediction, observation=args.observation,
        caoa=args.caoa, domain=args.domain, process_variant="hydroatlas",
    )
    records, grid_records = direct.load_unit_inputs(load_args)
    weights = h24.source_equal_weights(records["source_group"])
    train_rows, validation_rows = h24.group_split(records["source_group"], int(args.seed))
    names = direct.process_features(args.domain, "S0", "hydroatlas")
    scaler = h24.ProcessScaler.fit(records, grid_records, names, train_rows)
    x = scaler.transform(records)
    x_grid = scaler.transform(grid_records)
    indices: dict[str, list[int]] = {}
    for index, name in enumerate(scaler.names):
        indices.setdefault(branch_name(name), []).append(index)

    observed = direct.numeric(records["observed_log10"])
    source_medians = records.assign(_target=observed).groupby("source_group", observed=True)["_target"].median().to_numpy(float)
    target_center = float(np.median(source_medians))
    target_scale = float(1.4826 * np.median(np.abs(source_medians - target_center)))
    if not np.isfinite(target_scale) or target_scale < 0.35:
        target_scale = max(0.35, float(np.std(source_medians)))
    y = ((observed - target_center) / target_scale).astype(np.float32)
    model, fit_audit = fit_model(
        x, y, weights, train_rows, validation_rows, indices, target_scale,
        int(args.seed), int(args.epochs), float(args.map_cap_log10),
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        validation_standardized = model(torch.as_tensor(x[validation_rows], device=device), sample=False).cpu().numpy()
    validation_mean = target_center + target_scale * validation_standardized
    validation_observed = observed[validation_rows]
    residual = validation_observed - validation_mean
    local_weight = weights[validation_rows]
    residual_center = direct.weighted_metric(residual, local_weight)
    residual_sd = math.sqrt(direct.weighted_metric(np.square(residual - residual_center), local_weight))
    metrics_base = {
        "source_holdout_r2": float(r2_score(validation_observed, validation_mean, sample_weight=local_weight)),
        "source_holdout_rmse_log10": math.sqrt(direct.weighted_metric(np.square(residual), local_weight)),
        "source_holdout_mae_log10": direct.weighted_metric(np.abs(residual), local_weight),
        "residual_sd_log10": residual_sd,
    }
    size_sd = direct.numeric(records["sizecorr_known_sd_log10"])
    unit_sd = direct.numeric(records["unit_sd_log10"])
    known_sd = np.sqrt(np.square(np.nan_to_num(size_sd)) + np.square(np.nan_to_num(unit_sd)))
    grid_standardized, grid_sd_standardized, components = predict(model, x_grid, int(args.posterior_draws))
    grid_mean = target_center + target_scale * grid_standardized
    grid_latent_sd = target_scale * grid_sd_standardized

    for arm, interval_aware, output_dir in (
        ("protocol_aware", True, aware_dir), ("detection_unaware", False, unaware_dir)
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        local_known = known_sd[validation_rows] if interval_aware else np.zeros(len(validation_rows))
        validation_future_sd = np.sqrt(residual_sd**2 + np.square(local_known))
        coverage = float(np.average(np.abs(residual) <= 1.96 * validation_future_sd, weights=local_weight))
        detection_sd = float(np.median(known_sd[train_rows])) if interval_aware else 0.0
        grid_future_sd = np.sqrt(grid_latent_sd**2 + residual_sd**2 + detection_sd**2)
        grid = grid_records[["prediction_id", "longitude", "latitude", "waterbody", "caoa_status"]].copy()
        grid["latent_mean_log10"] = grid_mean
        grid["latent_sd_log10"] = grid_latent_sd
        grid["future_observation_sd_log10"] = grid_future_sd
        grid.to_csv(output_dir / "H26_UNIT_GRID.csv.gz", index=False, compression="gzip")
        heldout = records.iloc[validation_rows][["stable_record_id", "waterbody", "source_group"]].copy()
        heldout["observed_log10"] = validation_observed
        heldout["predicted_log10"] = validation_mean
        heldout["residual_log10"] = residual
        heldout["future_observation_sd_log10"] = validation_future_sd
        heldout.to_csv(output_dir / "H26_UNIT_HELDOUT.csv.gz", index=False, compression="gzip")
        component_frame = grid_records[["prediction_id", "longitude", "latitude", "waterbody"]].copy()
        for name, values in components.items():
            component_frame[f"{name}_component_log10"] = target_scale * values
        component_frame.to_csv(output_dir / "H26_UNIT_COMPONENTS.csv.gz", index=False, compression="gzip")
        torch.save(model.state_dict(), output_dir / "H26_UNIT_MODEL.pt")
        audit = {
            "status": "H26_MODULAR_DEVELOPMENT_RESULT_NOT_RELEASE",
            "source_arm": "S0", "observation_arm": arm, "observation_mode": "decoupled",
            "domain": args.domain, "process_variant": "hydroatlas_modular",
            "seed": int(args.seed), "records": int(len(records)),
            "source_groups": int(records["source_group"].nunique()),
            "prediction_cells": int(len(grid_records)),
            "process_scaler": scaler.audit(),
            "branches": {name: [scaler.names[i] for i in local] for name, local in indices.items()},
            "target_center_log10": target_center, "target_scale_log10": target_scale,
            "metrics": {**metrics_base, "source_holdout_coverage95": coverage, "median_known_detection_sd_log10": detection_sd},
            "map": {"latent_mean_q02_q50_q98_log10": np.quantile(grid_mean, [0.02, 0.50, 0.98]).tolist()},
            "fit": fit_audit,
            "limits": [
                "single source-group holdout development result", "no GP",
                "branch membership is mechanistic regularization, not proof of causal attribution",
                "detection metadata alters interval width only",
            ],
        }
        (output_dir / "H26_UNIT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "H26_MODULAR_DOMAIN_COMPLETED", "domain": args.domain, "metrics": metrics_base}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training", type=Path, required=True)
    result.add_argument("--prediction", type=Path, required=True)
    result.add_argument("--observation", type=Path, required=True)
    result.add_argument("--caoa", type=Path, required=True)
    result.add_argument("--domain", choices=direct.DOMAINS, required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--epochs", type=int, default=160)
    result.add_argument("--posterior-draws", type=int, default=24)
    result.add_argument("--map-cap-log10", type=float, default=4.0)
    result.add_argument("--force", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(train_domain(parser().parse_args()))

