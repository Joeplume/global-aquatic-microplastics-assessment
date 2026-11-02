"""Direct H26 S0-S3 process-BNN iteration using the existing H24 components.

This is intentionally a thin training entry point.  It does not create a new
freeze contract or a release gate.  Each invocation trains one independent
source arm, observation arm, and natural-water domain so that the units can be
queued directly across the three remote GPUs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

import build_h23_caoa_prototype as caoa
import run_h24_simple_protocol_preview as h24


VERSION = "H26_DIRECT_S0_S3_20260826_v1"
SOURCE_ARMS = ("S0", "S1", "S2", "S3")
OBSERVATION_ARMS = ("protocol_aware", "detection_unaware")
DOMAINS = tuple(caoa.NATURAL_DOMAINS)

NATIONAL_PLASTIC = "plastic_waste_generation_t"
LOCAL_SPOT = "spot_land_plastic_pressure_2020"
COUNTRY_SPOT = "spot_land_plastic_pressure_country_legacy_2020"
WGI = "wgi_government_effectiveness_2020"
DERIVED_NATIONAL = "h26_national_plastic_log1p"
DERIVED_COUNTRY = "h26_country_spot_log1p"
DERIVED_WITHIN = "h26_spot_within_country_log1p"
DERIVED_INTERACTION = "h26_plastic_escape_x_wgi"

MULTISCALE_FEATURES = {
    "reference_surface_freshwater": (
        "h26_spot_ms_1deg",
        "h26_spot_ms_2deg",
        "h26_spot_ms_4deg",
        "h26_runoff_ms_1deg",
        "h26_runoff_ms_2deg",
        "h26_runoff_ms_4deg",
        "h26_spot_runoff_ms_1deg",
        "h26_spot_runoff_ms_2deg",
        "h26_spot_local_minus_ms2deg",
    ),
    "transition_coastal": (
        "h26_spot_ms_1deg",
        "h26_spot_ms_2deg",
        "h26_runoff_ms_1deg",
        "h26_runoff_ms_2deg",
        "h26_spot_runoff_ms_1deg",
        "h26_spot_runoff_ms_2deg",
        "h26_river_export_ms_1deg",
        "h26_river_export_ms_2deg",
        "h26_river_export_ms_4deg",
        "h26_current_inverse",
        "h26_wave_current_ratio",
        "h26_current_speed_ms_2deg",
    ),
    "marine": (
        "h26_river_export_ms_1deg",
        "h26_river_export_ms_2deg",
        "h26_river_export_ms_4deg",
        "h26_river_export_ms_8deg",
        "h26_current_inverse",
        "h26_wave_current_ratio",
        "h26_current_speed_ms_2deg",
    ),
}

HYDROATLAS_FEATURES = {
    "reference_surface_freshwater": (
        "h26_hydroatlas_target_water_support",
        "h26_river_length_km_sum",
        "h26_river_discharge_cms_max",
        "h26_river_upstream_area_km2_max",
        "h26_river_dist_down_km_min",
        "h26_river_flow_order_max",
        "h26_river_slope_deg_lengthmean",
        "h26_stream_gradient_dmkm_lengthmean",
        "h26_upstream_lake_volume_mcm_max",
        "h26_upstream_reservoir_volume_mcm_max",
        "h26_regulation_pct_lengthmean",
        "h26_upstream_population_thousands_max",
        "h26_upstream_nightlight_max",
        "h26_lake_area_km2_sum",
        "h26_lake_volume_mcm_sum",
        "h26_reservoir_volume_mcm_sum",
        "h26_lake_residence_days_area_mean",
    ),
    "transition_coastal": (
        "h26_hydroatlas_target_water_support",
        "h26_river_length_km_sum",
        "h26_river_discharge_cms_max",
        "h26_river_upstream_area_km2_max",
        "h26_river_dist_down_km_min",
        "h26_river_flow_order_max",
        "h26_upstream_lake_volume_mcm_max",
        "h26_upstream_reservoir_volume_mcm_max",
        "h26_regulation_pct_lengthmean",
        "h26_upstream_population_thousands_max",
        "h26_upstream_nightlight_max",
    ),
    "marine": (),
}

HYDROATLAS_LOG_FEATURES = {
    "h26_river_length_km_sum",
    "h26_river_discharge_cms_max",
    "h26_river_upstream_area_km2_max",
    "h26_river_dist_down_km_min",
    "h26_stream_gradient_dmkm_lengthmean",
    "h26_upstream_lake_volume_mcm_max",
    "h26_upstream_reservoir_volume_mcm_max",
    "h26_upstream_population_thousands_max",
    "h26_upstream_nightlight_max",
    "h26_lake_area_km2_sum",
    "h26_lake_volume_mcm_sum",
    "h26_reservoir_volume_mcm_sum",
    "h26_lake_residence_days_area_mean",
}
caoa.LOG_FEATURES.update(HYDROATLAS_LOG_FEATURES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    local = numeric(result[LOCAL_SPOT]) if LOCAL_SPOT in result else np.full(len(result), np.nan)
    country = numeric(result[COUNTRY_SPOT]) if COUNTRY_SPOT in result else np.full(len(result), np.nan)
    national = numeric(result[NATIONAL_PLASTIC]) if NATIONAL_PLASTIC in result else np.full(len(result), np.nan)
    wgi = numeric(result[WGI]) if WGI in result else np.full(len(result), np.nan)
    local_log = signed_log1p(local)
    country_log = signed_log1p(country)
    national_log = signed_log1p(national)
    result[DERIVED_NATIONAL] = national_log
    result[DERIVED_COUNTRY] = country_log
    result[DERIVED_WITHIN] = np.where(
        np.isfinite(local_log) & np.isfinite(country_log), local_log - country_log, np.nan
    )
    result[DERIVED_INTERACTION] = np.where(
        np.isfinite(national_log) & np.isfinite(wgi), national_log * (-wgi), np.nan
    )
    return result


def process_features(domain: str, source_arm: str, process_variant: str = "base") -> tuple[str, ...]:
    base = tuple(caoa.DOMAIN_FEATURES[domain])
    if domain == "marine" or source_arm == "S0":
        selected = base
    elif source_arm == "S1":
        selected = tuple(dict.fromkeys((*base, NATIONAL_PLASTIC)))
    else:
        without_local = tuple(name for name in base if name != LOCAL_SPOT)
        split = (*without_local, DERIVED_NATIONAL, DERIVED_COUNTRY, DERIVED_WITHIN)
        if source_arm == "S2":
            selected = tuple(dict.fromkeys(split))
        else:
            selected = tuple(dict.fromkeys((*split, WGI, DERIVED_INTERACTION)))
    if process_variant == "multiscale":
        selected = tuple(dict.fromkeys((*selected, *MULTISCALE_FEATURES[domain])))
    elif process_variant == "hydroatlas":
        selected = tuple(dict.fromkeys((*selected, *HYDROATLAS_FEATURES[domain])))
    return selected


def required_raw_features(domain: str, process_variant: str) -> set[str]:
    selected = set(caoa.DOMAIN_FEATURES[domain]) | {
        NATIONAL_PLASTIC,
        LOCAL_SPOT,
        COUNTRY_SPOT,
        WGI,
    }
    if process_variant == "multiscale":
        selected.update(MULTISCALE_FEATURES[domain])
    elif process_variant == "hydroatlas":
        selected.update(HYDROATLAS_FEATURES[domain])
    return selected


def load_unit_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_header = set(pd.read_csv(args.training, nrows=0).columns)
    prediction_header = set(pd.read_csv(args.prediction, nrows=0).columns)
    raw = required_raw_features(args.domain, args.process_variant)
    training_columns = [
        "stable_record_id",
        *sorted(raw & training_header),
        *(["sizecorr_extrapolated_fraction_main"] if "sizecorr_extrapolated_fraction_main" in training_header else []),
    ]
    prediction_columns = [
        "prediction_id",
        "longitude",
        "latitude",
        "prediction_domain",
        *sorted(raw & prediction_header),
    ]
    training = pd.read_csv(args.training, usecols=training_columns, low_memory=False)
    prediction = pd.read_csv(args.prediction, usecols=prediction_columns, low_memory=False)
    observation = pd.read_csv(args.observation, low_memory=False)
    observation_columns = [
        "stable_record_id",
        "waterbody",
        "source_group",
        "observed_log10",
        "sizecorr_known_sd_log10",
        "unit_sd_log10",
        "sampling_year_effective_v1_3",
        "longitude",
        "latitude",
        "sampling_event_id",
        "sampling_event_key",
        "field_sampling_event_id",
        "detection_method",
        "qaqc_total_score",
        "sizecorr_lower_used_um",
        "sizecorr_evidence_tier",
    ]
    observation = observation[[name for name in observation_columns if name in observation]].copy()
    records = training.merge(observation, on="stable_record_id", how="inner", validate="one_to_one")
    if "sizecorr_extrapolated_fraction_main" not in records:
        records["sizecorr_extrapolated_fraction_main"] = np.nan
    records = records.loc[records["waterbody"].eq(args.domain)].reset_index(drop=True)
    prediction = prediction.loc[prediction["prediction_domain"].eq(args.domain)].copy()
    prediction = prediction.rename(columns={"prediction_domain": "waterbody"}).reset_index(drop=True)
    support = pd.read_csv(args.caoa, usecols=["prediction_id", "caoa_status"], low_memory=False)
    prediction = prediction.merge(support, on="prediction_id", how="left", validate="one_to_one")
    records = add_derived_features(records)
    prediction = add_derived_features(prediction)
    require(len(records) > 500, f"Too few observations for {args.domain}: {len(records):,}")
    require(len(prediction) > 0, f"No prediction cells for {args.domain}")
    require(prediction["caoa_status"].notna().all(), "C-AOA join did not close")
    return records, prediction


def weighted_metric(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def train_unit(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    audit_path = output_dir / "H26_UNIT_AUDIT.json"
    if audit_path.exists() and not args.force:
        print(json.dumps({"status": "SKIPPED_EXISTING", "audit": str(audit_path)}, ensure_ascii=False))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    records, prediction = load_unit_inputs(args)
    weights = h24.source_equal_weights(records["source_group"])
    train_rows, validation_rows = h24.group_split(records["source_group"], int(args.seed))
    names = process_features(args.domain, args.source_arm, args.process_variant)
    scaler = h24.ProcessScaler.fit(records, prediction, names, train_rows)
    x = scaler.transform(records)
    x_grid = scaler.transform(prediction)
    protocol = h24.ProtocolEncoder.fit(records, train_rows, weights)
    q = protocol.transform(records)

    observed = numeric(records["observed_log10"])
    source_medians = (
        records.assign(_target=observed).groupby("source_group", observed=True)["_target"].median().to_numpy(float)
    )
    target_center = float(np.median(source_medians))
    target_scale = float(1.4826 * np.median(np.abs(source_medians - target_center)))
    if not np.isfinite(target_scale) or target_scale < 0.35:
        target_scale = max(0.35, float(np.std(source_medians)))
    y = ((observed - target_center) / target_scale).astype(np.float32)

    size_sd = numeric(records["sizecorr_known_sd_log10"])
    unit_sd = numeric(records["unit_sd_log10"])
    known_sd_log10 = np.sqrt(np.square(np.nan_to_num(size_sd)) + np.square(np.nan_to_num(unit_sd)))
    known_sd_standardized = (known_sd_log10 / target_scale).astype(np.float32)
    interval_aware = args.observation_arm == "protocol_aware"
    aware = interval_aware and args.observation_mode == "joint"
    model, fit_audit = h24.fit_model(
        x,
        q,
        y,
        weights,
        known_sd_standardized,
        train_rows,
        validation_rows,
        aware=aware,
        process_cap_standardized=float(args.map_cap_log10) / target_scale,
        seed=int(args.seed),
        epochs=int(args.epochs),
    )

    device = next(model.parameters()).device
    q_validation = q[validation_rows] if aware else np.zeros((len(validation_rows), 0), np.float32)
    with torch.no_grad():
        validation_standardized = model(
            torch.as_tensor(x[validation_rows], device=device),
            torch.as_tensor(q_validation, device=device),
            sample=False,
        ).detach().cpu().numpy()
    validation_mean = target_center + target_scale * validation_standardized
    validation_observed = observed[validation_rows]
    validation_residual = validation_observed - validation_mean
    validation_weight = weights[validation_rows]
    residual_center = weighted_metric(validation_residual, validation_weight)
    residual_sd = math.sqrt(
        weighted_metric(np.square(validation_residual - residual_center), validation_weight)
    )
    validation_rmse = math.sqrt(weighted_metric(np.square(validation_residual), validation_weight))
    validation_mae = weighted_metric(np.abs(validation_residual), validation_weight)
    validation_r2 = float(r2_score(validation_observed, validation_mean, sample_weight=validation_weight))
    local_known = known_sd_log10[validation_rows] if interval_aware else np.zeros(len(validation_rows))
    validation_future_sd = np.sqrt(residual_sd**2 + np.square(local_known))
    coverage95 = float(
        np.average(
            np.abs(validation_residual) <= 1.96 * validation_future_sd,
            weights=validation_weight,
        )
    )

    grid_mean_standardized, grid_process_sd_standardized = h24.predict_environment(
        model, x_grid, draws=int(args.posterior_draws)
    )
    grid_mean = target_center + target_scale * grid_mean_standardized
    grid_latent_sd = target_scale * grid_process_sd_standardized
    detection_sd = float(np.median(known_sd_log10[train_rows])) if interval_aware else 0.0
    grid_future_sd = np.sqrt(grid_latent_sd**2 + residual_sd**2 + detection_sd**2)
    grid = prediction[["prediction_id", "longitude", "latitude", "waterbody", "caoa_status"]].copy()
    grid["latent_mean_log10"] = grid_mean
    grid["latent_sd_log10"] = grid_latent_sd
    grid["future_observation_sd_log10"] = grid_future_sd
    grid_path = output_dir / "H26_UNIT_GRID.csv.gz"
    grid.to_csv(grid_path, index=False, compression="gzip")

    heldout = records.iloc[validation_rows][["stable_record_id", "waterbody", "source_group"]].copy()
    heldout["observed_log10"] = validation_observed
    heldout["predicted_log10"] = validation_mean
    heldout["residual_log10"] = validation_residual
    heldout["future_observation_sd_log10"] = validation_future_sd
    heldout.to_csv(output_dir / "H26_UNIT_HELDOUT.csv.gz", index=False, compression="gzip")
    torch.save(model.state_dict(), output_dir / "H26_UNIT_MODEL.pt")

    audit = {
        "status": "DIRECT_DEVELOPMENT_RESULT_NOT_RELEASE",
        "version": VERSION,
        "source_arm": args.source_arm,
        "observation_arm": args.observation_arm,
        "observation_mode": args.observation_mode,
        "protocol_mean_terms_used": bool(aware),
        "domain": args.domain,
        "process_variant": args.process_variant,
        "seed": int(args.seed),
        "epochs_requested": int(args.epochs),
        "records": int(len(records)),
        "source_groups": int(records["source_group"].nunique()),
        "prediction_cells": int(len(prediction)),
        "process_features_requested": list(names),
        "process_scaler": scaler.audit(),
        "target_center_log10": target_center,
        "target_scale_log10": target_scale,
        "map_cap_log10": float(args.map_cap_log10),
        "metrics": {
            "source_holdout_r2": validation_r2,
            "source_holdout_rmse_log10": validation_rmse,
            "source_holdout_mae_log10": validation_mae,
            "source_holdout_coverage95": coverage95,
            "residual_sd_log10": residual_sd,
            "median_known_detection_sd_log10": detection_sd,
        },
        "map": {
            "latent_mean_q02_q50_q98_log10": np.quantile(grid_mean, [0.02, 0.50, 0.98]).tolist(),
            "latent_sd_median_log10": float(np.median(grid_latent_sd)),
            "future_observation_sd_median_log10": float(np.median(grid_future_sd)),
        },
        "fit": fit_audit,
        "outputs": {
            "grid": str(grid_path),
            "heldout": str(output_dir / "H26_UNIT_HELDOUT.csv.gz"),
            "model": str(output_dir / "H26_UNIT_MODEL.pt"),
        },
        "limits": [
            "direct source-group holdout development result; strict spatial-source folds not yet run",
            "source/lab effects are not subtracted in this direct first iteration",
            "decoupled mode shares the environmental fit and permits detection metadata to alter only observation uncertainty",
            "GP is not included in this direct first iteration",
            "regional outcomes are diagnostics and are not training targets",
        ],
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(audit_path), "metrics": audit["metrics"]}, ensure_ascii=False))
    return 0


def aggregate(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for source_arm in args.source_arms:
        paired: dict[str, pd.DataFrame] = {}
        for observation_arm in OBSERVATION_ARMS:
            domain_grids: list[pd.DataFrame] = []
            for domain in DOMAINS:
                unit_dir = run_root / source_arm / observation_arm / domain
                audit_path = unit_dir / "H26_UNIT_AUDIT.json"
                grid_path = unit_dir / "H26_UNIT_GRID.csv.gz"
                require(audit_path.exists() and grid_path.exists(), f"Missing unit: {unit_dir}")
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                metric_rows.append({
                    "source_arm": source_arm,
                    "observation_arm": observation_arm,
                    "domain": domain,
                    **audit["metrics"],
                })
                domain_grids.append(pd.read_csv(grid_path, low_memory=False))
            candidate = pd.concat(domain_grids, ignore_index=True)
            require(len(candidate) == 259_200, f"Incomplete candidate {source_arm}/{observation_arm}: {len(candidate):,}")
            candidate_path = output_dir / f"H26_{source_arm}_{observation_arm}_GRID.csv.gz"
            candidate.to_csv(candidate_path, index=False, compression="gzip")
            paired[observation_arm] = candidate
            candidates.append({
                "source_arm": source_arm,
                "observation_arm": observation_arm,
                "grid": str(candidate_path),
            })
        aware = paired["protocol_aware"]
        unaware = paired["detection_unaware"]
        require(np.array_equal(aware["prediction_id"].to_numpy(), unaware["prediction_id"].to_numpy()), "Paired grid order mismatch")
        delta = aware[["prediction_id", "longitude", "latitude", "waterbody", "caoa_status"]].copy()
        delta["aware_minus_unaware_q50_log10"] = aware["latent_mean_log10"].to_numpy(float) - unaware["latent_mean_log10"].to_numpy(float)
        delta["aware_minus_unaware_95width_log10"] = 3.92 * (
            aware["future_observation_sd_log10"].to_numpy(float)
            - unaware["future_observation_sd_log10"].to_numpy(float)
        )
        delta.to_csv(output_dir / f"H26_{source_arm}_PAIRED_DELTA.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metric_rows).to_csv(output_dir / "H26_DEVELOPMENT_METRICS.csv", index=False)
    summary = {
        "status": "DIRECT_DEVELOPMENT_AGGREGATED_NOT_RELEASE",
        "version": VERSION,
        "run_root": str(run_root),
        "candidates": candidates,
        "metrics": str(output_dir / "H26_DEVELOPMENT_METRICS.csv"),
    }
    (output_dir / "H26_DEVELOPMENT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "candidates": len(candidates)}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-unit")
    train.add_argument("--training", type=Path, required=True)
    train.add_argument("--prediction", type=Path, required=True)
    train.add_argument("--observation", type=Path, required=True)
    train.add_argument("--caoa", type=Path, required=True)
    train.add_argument("--source-arm", choices=SOURCE_ARMS, required=True)
    train.add_argument("--observation-arm", choices=OBSERVATION_ARMS, required=True)
    train.add_argument("--observation-mode", choices=("joint", "decoupled"), default="joint")
    train.add_argument("--domain", choices=DOMAINS, required=True)
    train.add_argument("--process-variant", choices=("base", "multiscale", "hydroatlas"), default="base")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--posterior-draws", type=int, default=24)
    train.add_argument("--map-cap-log10", type=float, default=4.0)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_unit)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--run-root", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)
    combine.add_argument("--source-arms", nargs="+", choices=SOURCE_ARMS, default=SOURCE_ARMS)
    combine.set_defaults(func=aggregate)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
