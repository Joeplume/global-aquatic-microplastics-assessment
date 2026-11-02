"""Train and render the H24 simple protocol-aware preview.

Two source-balanced, no-GP Bayesian neural models are fitted per natural-water
domain.  The arms share the same environmental architecture:

* protocol_aware: additive year/detection/QA/size metadata and known sample SD
  reliability weights are used; residual source embeddings are forbidden.
* detection_unaware: detection/QA/size metadata and known sample SD are ignored.

The command produces a row-complete two-arm prediction grid.  Plotting produces
exactly six concentration figures: all natural waters, marine, and continental
freshwater for each arm.  This is a diagnostic preview, not a release model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import build_h23_caoa_prototype as caoa


VERSION = "H24_SIMPLE_PROTOCOL_BOUNDED_PREVIEW_20260825"
DOMAINS = caoa.NATURAL_DOMAINS
DOMAIN_LABELS = caoa.DOMAIN_LABELS
ARMS = ("protocol_aware", "detection_unaware")
METHOD_FAMILIES = ("FTIR", "Raman", "LDIR", "thermal_mass", "visual_micro", "other", "unknown")

EXTRA_PROCESS_FEATURES = {
    "reference_surface_freshwater": (
        "plastic_waste_generation_t",
        "wgi_government_effectiveness_2020",
    ),
    "transition_coastal": (
        "plastic_waste_generation_t",
        "wgi_government_effectiveness_2020",
    ),
    "marine": (),
}

PROCESS_CAP_LOG10 = {
    "reference_surface_freshwater": 0.80,
    "transition_coastal": 0.80,
    "marine": 0.60,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


@dataclass
class ProcessScaler:
    names: tuple[str, ...]
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        prediction: pd.DataFrame,
        names: Iterable[str],
        train_rows: np.ndarray,
        minimum_features: int = 5,
    ) -> "ProcessScaler":
        active: list[str] = []
        centers: list[float] = []
        scales: list[float] = []
        for name in names:
            if name not in frame or name not in prediction:
                continue
            train_values = numeric(frame.iloc[train_rows][name])
            grid_values = numeric(prediction[name])
            if np.isfinite(train_values).mean() < 0.20 or np.isfinite(grid_values).mean() < 0.20:
                continue
            if name in caoa.LOG_FEATURES or name == "plastic_waste_generation_t":
                train_values = np.sign(train_values) * np.log1p(np.abs(train_values))
            finite = train_values[np.isfinite(train_values)]
            if not finite.size:
                continue
            center = float(np.median(finite))
            q10, q90 = np.quantile(finite, [0.10, 0.90])
            scale = float(q90 - q10)
            if not np.isfinite(scale) or scale <= 1e-8:
                continue
            active.append(name)
            centers.append(center)
            scales.append(scale)
        require(
            len(active) >= int(minimum_features),
            f"Too few active process features (minimum={int(minimum_features)}): {active}",
        )
        return cls(tuple(active), np.asarray(centers), np.asarray(scales))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for position, name in enumerate(self.names):
            values = numeric(frame[name])
            if name in caoa.LOG_FEATURES or name == "plastic_waste_generation_t":
                values = np.sign(values) * np.log1p(np.abs(values))
            values = np.where(np.isfinite(values), values, self.centers[position])
            values = (values - self.centers[position]) / self.scales[position]
            columns.append(np.clip(values, -4.0, 4.0))
        return np.column_stack(columns).astype(np.float32)

    def audit(self) -> dict[str, object]:
        return {
            "features": list(self.names),
            "centers": self.centers.tolist(),
            "q10_q90_scales": self.scales.tolist(),
            "transform": "optional_signed_log1p_then_median_impute_q10_q90_scale_clip8",
        }


@dataclass
class ProtocolEncoder:
    numeric_names: tuple[str, ...]
    numeric_centers: np.ndarray
    numeric_scales: np.ndarray
    category_columns: tuple[str, ...]
    column_center: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_rows: np.ndarray, weights: np.ndarray) -> "ProtocolEncoder":
        numeric_names = (
            "sampling_year_effective_v1_3",
            "qaqc_total_score",
            "sizecorr_lower_used_um",
            "sizecorr_extrapolated_fraction_main",
        )
        centers: list[float] = []
        scales: list[float] = []
        for name in numeric_names:
            values = numeric(frame.iloc[train_rows][name]) if name in frame else np.full(len(train_rows), np.nan)
            if name == "sizecorr_lower_used_um":
                values = np.log1p(np.clip(values, 0.0, None))
            finite = values[np.isfinite(values)]
            center = float(np.median(finite)) if finite.size else 0.0
            q10, q90 = np.quantile(finite, [0.10, 0.90]) if finite.size else (0.0, 1.0)
            scale = float(q90 - q10)
            if not np.isfinite(scale) or scale <= 1e-8:
                scale = 1.0
            centers.append(center)
            scales.append(scale)

        tiers = sorted(
            set(frame["sizecorr_evidence_tier"].fillna("unknown").astype(str))
            if "sizecorr_evidence_tier" in frame
            else {"unknown"}
        )
        category_columns = tuple([f"method::{value}" for value in METHOD_FAMILIES] + [f"tier::{value}" for value in tiers])
        provisional = cls(
            numeric_names,
            np.asarray(centers),
            np.asarray(scales),
            category_columns,
            np.zeros(len(numeric_names) * 2 + len(category_columns), dtype=float),
        )
        raw = provisional._raw(frame.iloc[train_rows])
        local_weight = weights[train_rows].astype(float)
        local_weight = local_weight / np.sum(local_weight)
        provisional.column_center = np.sum(raw * local_weight[:, None], axis=0)
        return provisional

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for position, name in enumerate(self.numeric_names):
            values = numeric(frame[name]) if name in frame else np.full(len(frame), np.nan)
            if name == "sizecorr_lower_used_um":
                values = np.log1p(np.clip(values, 0.0, None))
            missing = ~np.isfinite(values)
            standardized = (np.where(missing, self.numeric_centers[position], values) - self.numeric_centers[position]) / self.numeric_scales[position]
            columns.extend([np.clip(standardized, -8.0, 8.0), missing.astype(float)])

        methods = frame["detection_method"].map(caoa.method_family).fillna("unknown").astype(str).to_numpy()
        tiers = (
            frame["sizecorr_evidence_tier"].fillna("unknown").astype(str).to_numpy()
            if "sizecorr_evidence_tier" in frame
            else np.full(len(frame), "unknown", dtype=object)
        )
        for column in self.category_columns:
            prefix, value = column.split("::", 1)
            columns.append((methods == value).astype(float) if prefix == "method" else (tiers == value).astype(float))
        return np.column_stack(columns).astype(float)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return (self._raw(frame) - self.column_center).astype(np.float32)

    def audit(self) -> dict[str, object]:
        return {
            "numeric_names": list(self.numeric_names),
            "category_columns": list(self.category_columns),
            "reference": "source-equal training mean protocol gives zero map offset",
        }


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sd: float = 0.7) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -4.5))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -4.5))
        self.prior_sd = float(prior_sd)
        nn.init.xavier_uniform_(self.weight_mu)

    @staticmethod
    def sd(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def forward(self, values: torch.Tensor, sample: bool) -> torch.Tensor:
        weight = self.weight_mu
        bias = self.bias_mu
        if sample:
            weight = weight + self.sd(self.weight_rho) * torch.randn_like(weight)
            bias = bias + self.sd(self.bias_rho) * torch.randn_like(bias)
        return F.linear(values, weight, bias)

    def kl(self) -> torch.Tensor:
        prior_var = self.prior_sd**2
        total = torch.zeros((), device=self.weight_mu.device)
        for mean, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sd = self.sd(rho)
            total = total + 0.5 * (((sd.square() + mean.square()) / prior_var) - 1.0 + 2.0 * (math.log(self.prior_sd) - torch.log(sd))).sum()
        return total


class ProtocolAwareBNN(nn.Module):
    def __init__(self, process_dim: int, protocol_dim: int, process_cap_standardized: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BayesianLinear(process_dim, 48),
                BayesianLinear(48, 24),
                BayesianLinear(24, 1),
            ]
        )
        self.protocol = nn.Parameter(torch.zeros(protocol_dim))
        self.process_cap_standardized = float(process_cap_standardized)

    def forward(self, process: torch.Tensor, protocol: torch.Tensor, sample: bool) -> torch.Tensor:
        values = F.silu(self.layers[0](process, sample))
        values = F.silu(self.layers[1](values, sample))
        environment = self.layers[2](values, sample).reshape(-1)
        cap = self.process_cap_standardized
        environment = cap * torch.tanh(environment / cap)
        if self.protocol.numel():
            environment = environment + protocol @ self.protocol
        return environment

    def environment(self, process: torch.Tensor, sample: bool) -> torch.Tensor:
        empty = process.new_zeros((len(process), self.protocol.numel()))
        return self.forward(process, empty, sample)

    def kl(self) -> torch.Tensor:
        return sum(layer.kl() for layer in self.layers)


def source_equal_weights(groups: pd.Series) -> np.ndarray:
    labels = groups.fillna("__missing__").astype(str).to_numpy()
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    return (weight / np.mean(weight)).astype(np.float32)


def group_split(groups: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    labels = groups.fillna("__missing__").astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train, validation = next(splitter.split(np.arange(len(labels)), groups=labels))
    return train.astype(np.int64), validation.astype(np.int64)


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weight * np.square(prediction - target)) / np.sum(weight)))


def fit_model(
    x: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    known_sd: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    *,
    aware: bool,
    process_cap_standardized: float,
    seed: int,
    epochs: int,
) -> tuple[ProtocolAwareBNN, dict[str, object]]:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtocolAwareBNN(
        x.shape[1], q.shape[1] if aware else 0, process_cap_standardized
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-5)
    x_tensor = torch.as_tensor(x, device=device)
    q_tensor = torch.as_tensor(q if aware else np.zeros((len(q), 0), np.float32), device=device)
    y_tensor = torch.as_tensor(y.astype(np.float32), device=device)
    reliability = np.ones(len(y), dtype=np.float32)
    if aware:
        reliability = np.clip(1.0 / (1.0 + np.square(known_sd)), 0.10, 1.0).astype(np.float32)
    fit_weight = weights * reliability
    fit_weight = (fit_weight / np.mean(fit_weight)).astype(np.float32)
    w_tensor = torch.as_tensor(fit_weight, device=device)
    train_tensor = torch.as_tensor(train_rows, device=device)
    validation_tensor = torch.as_tensor(validation_rows, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 909)
    batch_size = min(4096, len(train_rows))
    patience = 22
    best_epoch = 0
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(int(epochs)):
        model.train()
        order = train_tensor[torch.randperm(len(train_tensor), generator=generator, device=device)]
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_tensor[batch], q_tensor[batch], sample=True)
            point_loss = F.smooth_l1_loss(prediction, y_tensor[batch], reduction="none", beta=0.45)
            data_loss = (point_loss * w_tensor[batch]).sum() / w_tensor[batch].sum().clamp_min(1e-8)
            kl = 0.015 * model.kl() / max(1, len(train_rows))
            protocol_penalty = 0.02 * model.protocol.square().mean() if model.protocol.numel() else 0.0
            loss = data_loss + kl + protocol_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                x_tensor[validation_tensor], q_tensor[validation_tensor], sample=False
            ).detach().cpu().numpy()
        validation_rmse = weighted_rmse(
            validation_prediction,
            y[validation_rows],
            weights[validation_rows],
        )
        if validation_rmse < best_rmse - 1e-5:
            best_rmse = validation_rmse
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break

    require(best_state is not None, "Training produced no finite validation checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    return model, {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "validation_source_group_holdout_rmse_standardized": float(best_rmse),
        "known_sd_reliability_used": bool(aware),
        "residual_source_embedding_used": False,
        "gp_used": False,
        "process_cap_standardized": float(process_cap_standardized),
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def predict_environment(
    model: ProtocolAwareBNN,
    x: np.ndarray,
    *,
    draws: int = 24,
    chunk: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    mean = np.empty(len(x), dtype=np.float32)
    sd = np.empty(len(x), dtype=np.float32)
    model.eval()
    for start in range(0, len(x), chunk):
        stop = min(len(x), start + chunk)
        local = torch.as_tensor(x[start:stop], device=device)
        with torch.no_grad():
            mean[start:stop] = model.environment(local, sample=False).detach().cpu().numpy()
            samples = torch.stack([model.environment(local, sample=True) for _ in range(draws)])
            sd[start:stop] = samples.std(dim=0, unbiased=True).detach().cpu().numpy()
    return mean, sd


def fit_domain_arm(
    records: pd.DataFrame,
    prediction: pd.DataFrame,
    domain: str,
    arm: str,
    *,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    local = records.loc[records["waterbody"].eq(domain)].reset_index(drop=True)
    grid = prediction.loc[prediction["waterbody"].eq(domain)].reset_index(drop=True)
    require(len(local) > 500, f"Too few records for {domain}")
    require(len(grid) > 0, f"No prediction cells for {domain}")
    weights = source_equal_weights(local["source_group"])
    train_rows, validation_rows = group_split(local["source_group"], seed)
    process_names = tuple(caoa.DOMAIN_FEATURES[domain]) + tuple(EXTRA_PROCESS_FEATURES[domain])
    scaler = ProcessScaler.fit(local, grid, process_names, train_rows)
    x = scaler.transform(local)
    x_grid = scaler.transform(grid)
    protocol = ProtocolEncoder.fit(local, train_rows, weights)
    q = protocol.transform(local)

    observed = numeric(local["observed_log10"])
    source_medians = local.assign(_target=observed).groupby("source_group")['_target'].median().to_numpy(float)
    target_center = float(np.median(source_medians))
    target_scale = float(1.4826 * np.median(np.abs(source_medians - target_center)))
    if not np.isfinite(target_scale) or target_scale < 0.35:
        target_scale = max(0.35, float(np.std(source_medians)))
    y = ((observed - target_center) / target_scale).astype(np.float32)

    size_sd = numeric(local["sizecorr_known_sd_log10"])
    unit_sd = numeric(local["unit_sd_log10"])
    known_sd_log10 = np.sqrt(np.square(np.nan_to_num(size_sd)) + np.square(np.nan_to_num(unit_sd)))
    known_sd_standardized = (known_sd_log10 / target_scale).astype(np.float32)
    aware = arm == "protocol_aware"
    model, fit_audit = fit_model(
        x,
        q,
        y,
        weights,
        known_sd_standardized,
        train_rows,
        validation_rows,
        aware=aware,
        process_cap_standardized=PROCESS_CAP_LOG10[domain] / target_scale,
        seed=seed,
        epochs=epochs,
    )

    mean_standardized, posterior_sd_standardized = predict_environment(model, x_grid)
    mean_log10 = target_center + target_scale * mean_standardized
    process_sd_log10 = target_scale * posterior_sd_standardized
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        validation_prediction = model(
            torch.as_tensor(x[validation_rows], device=device),
            torch.as_tensor(q[validation_rows] if aware else np.zeros((len(validation_rows), 0), np.float32), device=device),
            sample=False,
        ).detach().cpu().numpy()
    validation_rmse_log10 = target_scale * weighted_rmse(
        validation_prediction,
        y[validation_rows],
        weights[validation_rows],
    )
    if aware:
        predictive_sd = np.sqrt(np.square(process_sd_log10) + validation_rmse_log10**2)
    else:
        # This arm deliberately reports model-only uncertainty.  Its omitted
        # detection/QA/size uncertainty is the requested ablation.
        predictive_sd = process_sd_log10.copy()

    result = grid[["prediction_id", "longitude", "latitude", "waterbody", "caoa_status"]].copy()
    result[f"{arm}_latent_mean_log10"] = mean_log10
    result[f"{arm}_process_sd_log10"] = process_sd_log10
    result[f"{arm}_display_predictive_sd_log10"] = predictive_sd
    model_path = output_dir / f"{arm}_{domain}_model.pt"
    torch.save(model.state_dict(), model_path)
    audit = {
        "arm": arm,
        "domain": domain,
        "records": int(len(local)),
        "source_groups": int(local["source_group"].nunique()),
        "prediction_cells": int(len(grid)),
        "target_center_log10": target_center,
        "target_scale_log10": target_scale,
        "process_map_cap_log10": PROCESS_CAP_LOG10[domain],
        "known_sample_sd_median_log10": float(np.median(known_sd_log10)),
        "validation_rmse_log10": float(validation_rmse_log10),
        "map_latent_mean_q02_q50_q98_log10": np.quantile(mean_log10, [0.02, 0.50, 0.98]).tolist(),
        "map_process_sd_median_log10": float(np.median(process_sd_log10)),
        "process_scaler": scaler.audit(),
        "protocol_encoder": protocol.audit() if aware else "not_used",
        "fit": fit_audit,
        "model_file": str(model_path),
    }
    return result, audit


def train(args: argparse.Namespace) -> int:
    training_path = args.training.resolve()
    prediction_path = args.prediction.resolve()
    observation_path = args.observation.resolve()
    caoa_path = args.caoa.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (training_path, prediction_path, observation_path, caoa_path):
        require(path.exists(), f"Missing input: {path}")

    process_features = sorted(
        {
            feature
            for domain in DOMAINS
            for feature in tuple(caoa.DOMAIN_FEATURES[domain]) + tuple(EXTRA_PROCESS_FEATURES[domain])
        }
    )
    training_header = pd.read_csv(training_path, nrows=0).columns
    prediction_header = pd.read_csv(prediction_path, nrows=0).columns
    training_columns = [
        "stable_record_id",
        *[name for name in process_features if name in training_header],
        *[
            name
            for name in ("sizecorr_extrapolated_fraction_main",)
            if name in training_header
        ],
    ]
    prediction_columns = [
        "prediction_id",
        "longitude",
        "latitude",
        "prediction_domain",
        *[name for name in process_features if name in prediction_header],
    ]
    training = pd.read_csv(training_path, usecols=training_columns, low_memory=False)
    prediction = pd.read_csv(prediction_path, usecols=prediction_columns, low_memory=False)
    observation = pd.read_csv(observation_path, low_memory=False)
    observation_columns = [
        "stable_record_id",
        "waterbody",
        "source_group",
        "observed_log10",
        "sizecorr_known_sd_log10",
        "unit_sd_log10",
        "sampling_year_effective_v1_3",
        "detection_method",
        "qaqc_total_score",
        "sizecorr_lower_used_um",
        "sizecorr_evidence_tier",
    ]
    observation = observation[[name for name in observation_columns if name in observation]].copy()
    records = training.merge(observation, on="stable_record_id", how="inner", validate="one_to_one")
    if "sizecorr_extrapolated_fraction_main" in training_header:
        pass
    elif "sizecorr_extrapolated_fraction_main" not in records:
        records["sizecorr_extrapolated_fraction_main"] = np.nan
    records = records.loc[records["waterbody"].isin(DOMAINS)].copy()
    prediction = prediction.loc[prediction["prediction_domain"].isin(DOMAINS)].copy()
    prediction = prediction.rename(columns={"prediction_domain": "waterbody"})
    support = pd.read_csv(caoa_path, usecols=["prediction_id", "caoa_status"], low_memory=False)
    prediction = prediction.merge(support, on="prediction_id", how="left", validate="one_to_one")
    require(len(records) == 32_569, f"Expected 32,569 natural-water records, found {len(records):,}")
    require(len(prediction) == 259_200, f"Expected 259,200 natural prediction cells, found {len(prediction):,}")
    require(prediction["caoa_status"].notna().all(), "C-AOA join did not close")

    arm_outputs: dict[str, list[pd.DataFrame]] = {arm: [] for arm in ARMS}
    audits: list[dict[str, object]] = []
    for arm_index, arm in enumerate(ARMS):
        for domain_index, domain in enumerate(DOMAINS):
            result, audit = fit_domain_arm(
                records,
                prediction,
                domain,
                arm,
                seed=int(args.seed) + arm_index * 10_000 + domain_index * 1_000,
                epochs=int(args.epochs),
                output_dir=output_dir,
            )
            arm_outputs[arm].append(result)
            audits.append(audit)

    base_columns = ["prediction_id", "longitude", "latitude", "waterbody", "caoa_status"]
    grid = prediction[base_columns].copy()
    for arm in ARMS:
        local = pd.concat(arm_outputs[arm], ignore_index=True)
        value_columns = [name for name in local.columns if name not in base_columns]
        grid = grid.merge(local[["prediction_id", *value_columns]], on="prediction_id", how="left", validate="one_to_one")
    require(len(grid) == 259_200 and grid["prediction_id"].nunique() == len(grid), "Prediction grid closure failed")
    value_columns = [name for name in grid if name.endswith("_log10")]
    require(np.isfinite(grid[value_columns].to_numpy(float)).all(), "Non-finite H24 prediction fields")
    grid_path = output_dir / f"{VERSION}_GRID.csv.gz"
    grid.to_csv(grid_path, index=False, compression="gzip")

    audit = {
        "status": "DIAGNOSTIC_COMPLETED_NOT_RELEASE",
        "version": VERSION,
        "scope": "one-seed source-group-holdout preview; not strict space-country-source CV",
        "contract": {
            "process_model": "source-equal additive protocol Bayesian neural network",
            "gp_used": False,
            "residual_source_embedding_used": False,
            "protocol_aware": "year/detection/QA/size mean terms plus known sample SD reliability",
            "detection_unaware": "detection/QA/size terms and known sample SD omitted",
            "map_target": "source-zero source-equal mean protocol latent log10 concentration",
        },
        "inputs": {
            "training": str(training_path),
            "prediction": str(prediction_path),
            "observation": str(observation_path),
            "caoa": str(caoa_path),
            "training_sha256": sha256(training_path),
            "prediction_sha256": sha256(prediction_path),
            "observation_sha256": sha256(observation_path),
            "caoa_sha256": sha256(caoa_path),
        },
        "grid": str(grid_path),
        "rows": int(len(grid)),
        "domain_arm_audits": audits,
        "known_limits": [
            "single seed only",
            "source-group holdout validation only; strict spatial-country-source CV not run",
            "protocol effects are regularized associations, not identified laboratory causal effects",
            "detection-unaware interval is intentionally model-only and underestimates observed-data uncertainty",
            "C-AOA retained as annotation and never used as a mean predictor",
        ],
    }
    audit_path = output_dir / f"{VERSION}_TRAIN_AUDIT.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "grid": str(grid_path), "rows": len(grid)}, ensure_ascii=False))
    return 0


def project_grid(longitude: np.ndarray, latitude: np.ndarray, transformer) -> tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    x, y = transformer.transform(lon_grid, lat_grid)
    return np.asarray(x, float), np.asarray(y, float)


def plot(args: argparse.Namespace) -> int:
    import geopandas as gpd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from pyproj import CRS, Transformer

    grid_path = args.grid.resolve()
    countries_path = args.countries.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(grid_path, low_memory=False)
    require(len(grid) == 259_200 and grid["prediction_id"].nunique() == len(grid), "Invalid H24 grid")
    longitude = np.sort(grid["longitude"].unique())
    latitude = np.sort(grid["latitude"].unique())
    require((len(latitude), len(longitude)) == (360, 720), "Unexpected H24 grid geometry")
    domain_codes = pd.Categorical(grid["waterbody"], categories=DOMAINS).codes
    grid = grid.assign(_domain_code=domain_codes, _caoa_in=grid["caoa_status"].eq("C-AOA in").astype(float))
    domains = grid.pivot(index="latitude", columns="longitude", values="_domain_code").reindex(index=latitude, columns=longitude).to_numpy(float)

    robinson = CRS.from_proj4("+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", robinson, always_xy=True)
    edge_x, edge_y = project_grid(np.arange(-180.0, 180.5, 0.5), np.arange(-90.0, 90.5, 0.5), transformer)
    center_x, center_y = project_grid(longitude, latitude, transformer)
    boundary_x, boundary_y = caoa.robinson_boundary(transformer)
    countries = gpd.read_file(countries_path).to_crs(robinson)
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    scopes = {
        "all_natural": np.ones(len(grid), dtype=bool),
        "marine": grid["waterbody"].eq("marine").to_numpy(),
        "continental_freshwater": grid["waterbody"].eq("reference_surface_freshwater").to_numpy(),
    }
    scope_titles = {
        "all_natural": "All natural waters",
        "marine": "Marine water",
        "continental_freshwater": "Continental freshwater",
    }
    scale_contract: dict[str, list[float]] = {}
    for scope, mask in scopes.items():
        pooled = np.concatenate(
            [grid.loc[mask, f"{arm}_latent_mean_log10"].to_numpy(float) for arm in ARMS]
        )
        scale_contract[scope] = np.quantile(pooled[np.isfinite(pooled)], [0.02, 0.98]).tolist()

    outputs: list[dict[str, object]] = []
    colors = "RdYlBu_r"
    for arm in ARMS:
        aware = arm == "protocol_aware"
        arm_title = "Protocol-aware" if aware else "Detection-uncertainty ablation"
        subtitle = (
            "Known method/QA/size metadata and sample SD used; residual source effect off; GP off"
            if aware
            else "Detection/QA/size metadata and known sample SD ignored; residual source effect off; GP off"
        )
        mean_field = f"{arm}_latent_mean_log10"
        sd_field = f"{arm}_display_predictive_sd_log10"
        mean_grid = grid.pivot(index="latitude", columns="longitude", values=mean_field).reindex(index=latitude, columns=longitude).to_numpy(float)
        sd_grid = grid.pivot(index="latitude", columns="longitude", values=sd_field).reindex(index=latitude, columns=longitude).to_numpy(float)
        for scope, row_mask in scopes.items():
            if scope == "all_natural":
                cell_mask = np.ones_like(domains, dtype=bool)
            elif scope == "marine":
                cell_mask = domains == DOMAINS.index("marine")
            else:
                cell_mask = domains == DOMAINS.index("reference_surface_freshwater")
            values = np.ma.masked_where(~cell_mask, mean_grid)
            local_sd = sd_grid[cell_mask]
            low, high = scale_contract[scope]

            fig, axis = plt.subplots(figsize=(14.2, 7.9))
            fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.20)
            image = axis.pcolormesh(
                edge_x,
                edge_y,
                values,
                cmap=colors,
                norm=Normalize(vmin=low, vmax=high),
                shading="flat",
                linewidth=0,
                antialiased=False,
                rasterized=True,
                zorder=1,
            )
            countries.boundary.plot(ax=axis, color="#555D61", linewidth=0.28, alpha=0.78, zorder=3)
            axis.plot(boundary_x, boundary_y, color="#344047", linewidth=0.65, zorder=4)
            axis.set_xlim(float(np.nanmin(edge_x)), float(np.nanmax(edge_x)))
            axis.set_ylim(float(np.nanmin(edge_y)), float(np.nanmax(edge_y)))
            axis.set_aspect("equal")
            axis.set_axis_off()
            fig.suptitle(
                f"H24-simple preview — {arm_title}: {scope_titles[scope]}",
                fontsize=16.0,
                y=0.965,
                color="#263940",
            )
            fig.text(0.5, 0.915, subtitle, ha="center", fontsize=8.9, color="#4B5358")
            color_axis = fig.add_axes([0.20, 0.105, 0.60, 0.027])
            colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal", extend="both")
            colorbar.set_label(r"Latent mean log$_{10}$ concentration (items m$^{-3}$)", fontsize=9.2)
            median_width = float(np.median(3.92 * local_sd[np.isfinite(local_sd)]))
            fig.text(
                0.5,
                0.035,
                f"Shared aware/unaware colour scale: q02–q98 = {low:.2f}–{high:.2f}; median displayed 95% width = {median_width:.2f} log10. "
                "Native 0.5° cells; all C-AOA classes retained; support map reported separately; no interpolation.",
                ha="center",
                fontsize=8.3,
                color="#4B5358",
            )
            stem = f"{VERSION}_{arm.upper()}_{scope.upper()}"
            png = output_dir / f"{stem}.png"
            fig.savefig(png, dpi=int(args.dpi), facecolor="white", bbox_inches="tight", pad_inches=0.12)
            fig.savefig(png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.12)
            plt.close(fig)
            outputs.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "png": str(png),
                    "pdf": str(png.with_suffix(".pdf")),
                    "active_cells": int(np.sum(cell_mask)),
                    "shared_display_q02_q98_log10": [float(low), float(high)],
                    "median_displayed_95_width_log10": median_width,
                }
            )

    require(len(outputs) == 6, "H24 six-figure output contract failed")
    audit = {
        "status": "DIAGNOSTIC_COMPLETED_NOT_RELEASE",
        "version": VERSION,
        "grid": str(grid_path),
        "grid_sha256": sha256(grid_path),
        "projection": "Robinson",
        "renderer": "native_0p5_degree_pcolormesh_flat_no_interpolation",
        "figure_count": len(outputs),
        "shared_scale_within_scope_across_arms": True,
        "caoa_overlay": False,
        "caoa_policy": "all classes retained; separate C-AOA map avoids fragmented contour artefacts",
        "outputs": outputs,
    }
    audit_path = output_dir / f"{VERSION}_PLOT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "figure_count": len(outputs), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train both H24 preview arms")
    train_parser.add_argument("--training", type=Path, required=True)
    train_parser.add_argument("--prediction", type=Path, required=True)
    train_parser.add_argument("--observation", type=Path, required=True)
    train_parser.add_argument("--caoa", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=260825)
    train_parser.add_argument("--epochs", type=int, default=120)
    train_parser.set_defaults(func=train)
    plot_parser = subparsers.add_parser("plot", help="Render exactly six H24 figures")
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
