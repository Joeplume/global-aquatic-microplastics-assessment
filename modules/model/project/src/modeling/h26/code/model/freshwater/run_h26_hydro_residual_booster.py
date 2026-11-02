"""H26 S0 environmental BNN plus a bounded HydroATLAS residual booster.

The S0 base BNN is fitted first.  A second Bayesian branch sees only the new
HydroATLAS variables and is trained on base-model residuals.  Its map
correction is capped and validation-selected shrinkage may return exactly
zero.  Observation metadata never enters either environmental mean.
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


TASK = {
    "reference_surface_freshwater": "freshwater",
    "transition_coastal": "transition",
    "marine": "marine",
}


class HydroResidualBNN(nn.Module):
    def __init__(self, input_dim: int, cap_log10: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            h24.BayesianLinear(input_dim, 32, prior_sd=0.45),
            h24.BayesianLinear(32, 16, prior_sd=0.40),
            h24.BayesianLinear(16, 1, prior_sd=0.35),
        ])
        self.cap_log10 = float(cap_log10)

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        values = F.silu(self.layers[0](values, sample))
        values = F.silu(self.layers[1](values, sample))
        values = self.layers[2](values, sample).reshape(-1)
        return self.cap_log10 * torch.tanh(values / self.cap_log10)

    def kl(self) -> torch.Tensor:
        return sum(layer.kl() for layer in self.layers)


def weighted_rmse(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(direct.weighted_metric(np.square(observed - predicted), weights))


def choose_split(records: pd.DataFrame, seed: int, folds_csv: Path | None, fold: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if folds_csv is None:
        train, test = h24.group_split(records["source_group"], seed)
        return train, test, {"kind": "source_group_development", "seed": int(seed)}
    folds = pd.read_csv(folds_csv, low_memory=False)
    required = {"stable_record_id", "task", "scheme", "fold", "eligible"}
    direct.require(required.issubset(folds.columns), f"Strict folds missing columns: {sorted(required - set(folds.columns))}")
    selected = folds.loc[
        folds["task"].astype(str).eq(TASK[str(records["waterbody"].iloc[0])])
        & folds["scheme"].astype(str).eq("spatial_source_purged")
    ].copy()
    selected["stable_record_id"] = selected["stable_record_id"].astype(str)
    direct.require(selected["stable_record_id"].is_unique, "Strict fold IDs are not unique")
    aligned = records[["stable_record_id"]].copy()
    aligned["stable_record_id"] = aligned["stable_record_id"].astype(str)
    aligned = aligned.merge(
        selected[["stable_record_id", "fold", "eligible"]], on="stable_record_id", how="left", validate="one_to_one"
    )
    direct.require(aligned["fold"].notna().all(), "Strict folds do not cover all model records")
    eligible = aligned["eligible"].astype(str).str.upper().isin({"TRUE", "1"}).to_numpy()
    fold_values = pd.to_numeric(aligned["fold"], errors="coerce").to_numpy(float)
    test_mask = eligible & (fold_values == int(fold))
    heldout_sources = set(records.loc[test_mask, "source_group"].fillna("__missing__").astype(str))
    source = records["source_group"].fillna("__missing__").astype(str).to_numpy()
    train_mask = ~test_mask & ~np.isin(source, list(heldout_sources))
    train = np.flatnonzero(train_mask).astype(np.int64)
    test = np.flatnonzero(test_mask).astype(np.int64)
    direct.require(len(train) >= 500 and len(test) >= 50, "Strict split is too small")
    overlap = len(set(source[train]) & set(source[test]))
    direct.require(overlap == 0, f"Strict split retains {overlap} held-out sources")
    return train, test, {
        "kind": "spatial_source_purged", "fold": int(fold), "folds_csv": str(folds_csv),
        "train_records": int(len(train)), "heldout_records": int(len(test)),
        "heldout_sources": int(len(heldout_sources)), "source_overlap": int(overlap),
    }


def deterministic_environment(model: h24.ProtocolAwareBNN, values: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    result = np.empty(len(values), np.float32)
    for start in range(0, len(values), 32768):
        stop = min(len(values), start + 32768)
        with torch.no_grad():
            result[start:stop] = model.environment(
                torch.as_tensor(values[start:stop], device=device), sample=False
            ).cpu().numpy()
    return result


def fit_booster(
    x: np.ndarray,
    residual_target: np.ndarray,
    base_log10: np.ndarray,
    observed_log10: np.ndarray,
    weights: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    seed: int,
    epochs: int,
    cap_log10: float,
    minimum_improvement: float,
) -> tuple[HydroResidualBNN, float, dict[str, object]]:
    h24.seed_all(seed + 2609)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydroResidualBNN(x.shape[1], cap_log10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0018, weight_decay=5e-5)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(residual_target.astype(np.float32), device=device)
    fit_weight = (weights / np.mean(weights)).astype(np.float32)
    wt = torch.as_tensor(fit_weight, device=device)
    train_tensor = torch.as_tensor(train_rows, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 2917)
    base_rmse = weighted_rmse(observed_log10[validation_rows], base_log10[validation_rows], weights[validation_rows])
    best_rmse = base_rmse
    best_epoch = 0
    best_alpha = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 28
    batch_size = min(4096, len(train_rows))
    alphas = (0.25, 0.50, 0.75, 1.00)
    for epoch in range(int(epochs)):
        model.train()
        order = train_tensor[torch.randperm(len(train_tensor), generator=generator, device=device)]
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            correction = model(xt[batch], sample=True)
            point = F.smooth_l1_loss(correction, yt[batch], reduction="none", beta=0.35)
            data_loss = (point * wt[batch]).sum() / wt[batch].sum().clamp_min(1e-8)
            loss = data_loss + 0.030 * model.kl() / max(1, len(train_rows)) + 0.030 * correction.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            correction = model(xt[validation_rows], sample=False).cpu().numpy()
        improved = False
        for alpha in alphas:
            trial = base_log10[validation_rows] + alpha * correction
            score = weighted_rmse(observed_log10[validation_rows], trial, weights[validation_rows])
            if score < best_rmse - 1e-5:
                best_rmse = score
                best_epoch = epoch + 1
                best_alpha = float(alpha)
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                improved = True
        if not improved and best_epoch and epoch + 1 - best_epoch >= patience:
            break
    accepted = best_state is not None and best_rmse <= base_rmse - float(minimum_improvement)
    if best_state is not None:
        model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    model.eval()
    alpha = best_alpha if accepted else 0.0
    return model, alpha, {
        "device": str(device), "best_epoch": int(best_epoch), "selected_alpha": float(alpha),
        "raw_best_alpha": float(best_alpha), "base_validation_rmse_log10": float(base_rmse),
        "best_validation_rmse_log10": float(best_rmse),
        "validation_improvement_log10": float(base_rmse - best_rmse),
        "minimum_improvement_log10": float(minimum_improvement), "accepted": bool(accepted),
        "cap_log10": float(cap_log10), "parameters": int(sum(v.numel() for v in model.parameters())),
    }


def predict_booster(model: HydroResidualBNN, values: np.ndarray, draws: int) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    mean = np.empty(len(values), np.float32)
    sd = np.empty(len(values), np.float32)
    for start in range(0, len(values), 32768):
        stop = min(len(values), start + 32768)
        local = torch.as_tensor(values[start:stop], device=device)
        with torch.no_grad():
            mean[start:stop] = model(local, sample=False).cpu().numpy()
            samples = torch.stack([model(local, sample=True) for _ in range(draws)])
            sd[start:stop] = samples.std(dim=0, unbiased=True).cpu().numpy()
    return mean, sd


def train_domain(args: argparse.Namespace) -> int:
    load_args = argparse.Namespace(
        training=args.training, prediction=args.prediction, observation=args.observation,
        caoa=args.caoa, domain=args.domain, process_variant="hydroatlas",
    )
    records, grid_records = direct.load_unit_inputs(load_args)
    weights = h24.source_equal_weights(records["source_group"])
    train_rows, validation_rows, split_audit = choose_split(records, int(args.seed), args.folds_csv, int(args.fold))
    observed = direct.numeric(records["observed_log10"])
    source_medians = records.assign(_target=observed).groupby("source_group", observed=True)["_target"].median().to_numpy(float)
    target_center = float(np.median(source_medians))
    target_scale = float(1.4826 * np.median(np.abs(source_medians - target_center)))
    if not np.isfinite(target_scale) or target_scale < 0.35:
        target_scale = max(0.35, float(np.std(source_medians)))
    y = ((observed - target_center) / target_scale).astype(np.float32)

    dropped = set(args.drop_feature)
    base_names = tuple(name for name in direct.process_features(args.domain, "S0", "base") if name not in dropped)
    base_scaler = h24.ProcessScaler.fit(records, grid_records, base_names, train_rows)
    x_base = base_scaler.transform(records)
    x_base_grid = base_scaler.transform(grid_records)
    empty_protocol = np.zeros((len(records), 0), np.float32)
    base_model, base_fit = h24.fit_model(
        x_base, empty_protocol, y, weights, np.zeros(len(records), np.float32), train_rows, validation_rows,
        aware=False, process_cap_standardized=float(args.map_cap_log10) / target_scale,
        seed=int(args.seed), epochs=int(args.base_epochs),
    )
    base_records = target_center + target_scale * deterministic_environment(base_model, x_base)
    hydro_names = tuple(name for name in direct.HYDROATLAS_FEATURES[args.domain] if name not in dropped)
    booster = None
    booster_scaler = None
    booster_alpha = 0.0
    booster_audit: dict[str, object] = {"accepted": False, "reason": "no HydroATLAS features for domain", "selected_alpha": 0.0}
    if hydro_names:
        booster_scaler = h24.ProcessScaler.fit(records, grid_records, hydro_names, train_rows)
        x_hydro = booster_scaler.transform(records)
        booster, booster_alpha, booster_audit = fit_booster(
            x_hydro, observed - base_records, base_records, observed, weights, train_rows, validation_rows,
            int(args.seed), int(args.booster_epochs), float(args.booster_cap_log10), float(args.minimum_improvement_log10),
        )
        booster_records, _ = predict_booster(booster, x_hydro, max(2, int(args.posterior_draws)))
    else:
        x_hydro = np.zeros((len(records), 0), np.float32)
        booster_records = np.zeros(len(records), np.float32)
    corrected_records = base_records + booster_alpha * booster_records
    residual_train = observed[train_rows] - corrected_records[train_rows]
    residual_center = direct.weighted_metric(residual_train, weights[train_rows])
    residual_sd = math.sqrt(direct.weighted_metric(np.square(residual_train - residual_center), weights[train_rows]))
    validation_mean = corrected_records[validation_rows]
    validation_observed = observed[validation_rows]
    validation_weight = weights[validation_rows]
    residual = validation_observed - validation_mean

    base_grid_standardized, base_grid_sd_standardized = h24.predict_environment(
        base_model, x_base_grid, draws=int(args.posterior_draws)
    )
    base_grid = target_center + target_scale * base_grid_standardized
    base_grid_sd = target_scale * base_grid_sd_standardized
    base_validation_standardized, base_validation_sd_standardized = h24.predict_environment(
        base_model, x_base[validation_rows], draws=int(args.posterior_draws)
    )
    validation_latent_sd = target_scale * base_validation_sd_standardized
    if booster is not None:
        x_hydro_grid = booster_scaler.transform(grid_records)
        booster_grid, booster_grid_sd = predict_booster(booster, x_hydro_grid, int(args.posterior_draws))
        _, booster_validation_sd = predict_booster(booster, x_hydro[validation_rows], int(args.posterior_draws))
    else:
        booster_grid = np.zeros(len(grid_records), np.float32)
        booster_grid_sd = np.zeros(len(grid_records), np.float32)
        booster_validation_sd = np.zeros(len(validation_rows), np.float32)
    grid_mean = base_grid + booster_alpha * booster_grid
    grid_latent_sd = np.sqrt(np.square(base_grid_sd) + np.square(booster_alpha * booster_grid_sd))
    validation_latent_sd = np.sqrt(np.square(validation_latent_sd) + np.square(booster_alpha * booster_validation_sd))
    size_sd = direct.numeric(records["sizecorr_known_sd_log10"])
    unit_sd = direct.numeric(records["unit_sd_log10"])
    known_sd = np.sqrt(np.square(np.nan_to_num(size_sd)) + np.square(np.nan_to_num(unit_sd)))
    metrics_base = {
        "source_holdout_r2": float(r2_score(validation_observed, validation_mean, sample_weight=validation_weight)),
        "source_holdout_rmse_log10": weighted_rmse(validation_observed, validation_mean, validation_weight),
        "source_holdout_mae_log10": direct.weighted_metric(np.abs(residual), validation_weight),
        "residual_sd_log10": float(residual_sd),
    }

    for arm, interval_aware in (("protocol_aware", True), ("detection_unaware", False)):
        output_dir = args.run_root / "S0" / arm / args.domain
        output_dir.mkdir(parents=True, exist_ok=True)
        local_known = known_sd[validation_rows] if interval_aware else np.zeros(len(validation_rows))
        validation_future_sd = np.sqrt(np.square(validation_latent_sd) + residual_sd**2 + np.square(local_known))
        coverage = float(np.average(np.abs(residual) <= 1.96 * validation_future_sd, weights=validation_weight))
        detection_sd = float(np.median(known_sd[train_rows])) if interval_aware else 0.0
        grid_future_sd = np.sqrt(np.square(grid_latent_sd) + residual_sd**2 + detection_sd**2)
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
        components = grid_records[["prediction_id", "longitude", "latitude", "waterbody"]].copy()
        components["base_s0_mean_log10"] = base_grid
        components["hydroatlas_residual_correction_log10"] = booster_alpha * booster_grid
        components.to_csv(output_dir / "H26_UNIT_COMPONENTS.csv.gz", index=False, compression="gzip")
        torch.save({"state_dict": base_model.state_dict(), "target_center": target_center, "target_scale": target_scale}, output_dir / "H26_BASE_MODEL.pt")
        if booster is not None:
            torch.save({"state_dict": booster.state_dict(), "alpha": booster_alpha}, output_dir / "H26_HYDRO_BOOSTER_MODEL.pt")
        audit = {
            "status": "H26_RESIDUAL_BOOSTER_DEVELOPMENT_NOT_RELEASE", "source_arm": "S0",
            "observation_arm": arm, "observation_mode": "fully_decoupled", "domain": args.domain,
            "seed": int(args.seed), "split": split_audit, "records": int(len(records)),
            "source_groups": int(records["source_group"].nunique()), "prediction_cells": int(len(grid_records)),
            "base_features": list(base_scaler.names), "hydroatlas_features": list(hydro_names),
            "features_dropped_by_preregistered_ablation": sorted(dropped),
            "base_scaler": base_scaler.audit(), "hydroatlas_scaler": booster_scaler.audit() if booster_scaler else None,
            "target_center_log10": target_center, "target_scale_log10": target_scale,
            "metrics": {**metrics_base, "source_holdout_coverage95": coverage, "median_known_detection_sd_log10": detection_sd},
            "base_fit": base_fit, "booster_fit": booster_audit,
            "map": {"latent_mean_q02_q50_q98_log10": np.quantile(grid_mean, [0.02, 0.50, 0.98]).tolist()},
            "limits": [
                "HydroATLAS is a bounded residual correction, not a causal attribution",
                "validation-selected correction must be rechecked on strict folds and seeds",
                "observation metadata changes interval width only", "GP is off",
            ],
        }
        (output_dir / "H26_UNIT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "H26_RESIDUAL_BOOSTER_DOMAIN_COMPLETED", "domain": args.domain, "metrics": metrics_base, "booster": booster_audit}, ensure_ascii=False))
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
    result.add_argument("--folds-csv", type=Path, default=None)
    result.add_argument("--fold", type=int, default=2)
    result.add_argument("--base-epochs", type=int, default=140)
    result.add_argument("--booster-epochs", type=int, default=180)
    result.add_argument("--posterior-draws", type=int, default=24)
    result.add_argument("--map-cap-log10", type=float, default=4.0)
    result.add_argument("--booster-cap-log10", type=float, default=0.45)
    result.add_argument("--minimum-improvement-log10", type=float, default=0.005)
    result.add_argument("--drop-feature", action="append", default=[])
    return result


if __name__ == "__main__":
    raise SystemExit(train_domain(parser().parse_args()))
