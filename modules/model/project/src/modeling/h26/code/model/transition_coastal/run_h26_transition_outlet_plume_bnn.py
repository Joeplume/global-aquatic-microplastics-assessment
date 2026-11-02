"""Strict H26 transition/coastal outlet-plume BNN comparison.

The matched baseline and challenger use the same transition/coastal records
and cells within 300 km of a fully supported RiverATLAS outlet.  The challenger
adds only distance-decayed outlet delivery and upstream intensity covariates.
All grid rows remain in outputs; unsupported rows carry NA predictions.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_h23_caoa_prototype as caoa
import run_h26_freshwater_network_modular_bnn as freshwater
import run_h26_transition_network_modular_bnn as transition


DOMAIN = "transition_coastal"
PLUME_STATUS = "h26_transition_outlet_plume_status"
VECTOR_PLUME_STATUS = "h26_transition_vector_plume_status"
PRESSURE_DELIVERY_STATUS = "h26_transition_outlet_pressure_delivery_status"
ACTIVE_PLUME_STATUS = "h26_transition_active_process_status"
PLUME_SOURCE_STATUS_ALIAS = "h26_transition_outlet_source_support_status"
PLUME_SOURCE_FEATURES = (
    "h26_outlet_river_export_decay_100km",
    "h26_outlet_spot_intensity_mean_100km",
    "h26_outlet_nightlight_intensity_mean_100km",
    "h26_outlet_road_intensity_mean_100km",
    "h26_outlet_population_intensity_mean_100km",
)
PLUME_TRANSPORT_FEATURES = (
    "h26_nearest_supported_outlet_distance_km",
    "h26_outlet_discharge_decay_100km",
)
PLUME_FEATURES = (*PLUME_SOURCE_FEATURES, *PLUME_TRANSPORT_FEATURES)
PLUME_INTENSITY_SOURCE_FEATURES = tuple(
    name for name in PLUME_SOURCE_FEATURES
    if name != "h26_outlet_river_export_decay_100km"
)
PLUME_INTENSITY_TRANSPORT_FEATURES = (
    "h26_nearest_supported_outlet_distance_km",
)
PLUME_INTENSITY_FEATURES = (
    *PLUME_INTENSITY_SOURCE_FEATURES,
    *PLUME_INTENSITY_TRANSPORT_FEATURES,
)
PLUME_BRANCHES = {
    "source": (*transition.BASE_BRANCHES["source"], *PLUME_SOURCE_FEATURES),
    "transport": (*transition.BASE_BRANCHES["transport"], *PLUME_TRANSPORT_FEATURES),
    "retention": transition.BASE_BRANCHES["retention"],
    "background": transition.BASE_BRANCHES["background"],
}
PLUME_INTENSITY_BRANCHES = {
    "source": (*transition.BASE_BRANCHES["source"], *PLUME_INTENSITY_SOURCE_FEATURES),
    "transport": (
        *transition.BASE_BRANCHES["transport"],
        *PLUME_INTENSITY_TRANSPORT_FEATURES,
    ),
    "retention": transition.BASE_BRANCHES["retention"],
    "background": transition.BASE_BRANCHES["background"],
}
PLUME_SOURCE_ONLY_BRANCHES = {
    "source": (*transition.BASE_BRANCHES["source"], *PLUME_INTENSITY_SOURCE_FEATURES),
    "transport": transition.BASE_BRANCHES["transport"],
    "retention": transition.BASE_BRANCHES["retention"],
    "background": transition.BASE_BRANCHES["background"],
}
PLUME_MONOTONE_DELIVERY_FEATURES = (
    *PLUME_SOURCE_FEATURES,
    "h26_outlet_discharge_decay_100km",
)
PLUME_MONOTONE_DELIVERY_BRANCHES = {
    # Every field in this branch is a non-negative pressure or a
    # distance-decayed delivery amount.  Constraining the branch to increase
    # therefore enforces source/delivery direction without constraining a
    # latitude band, named region, or the ambiguous hydrodynamic covariates.
    "source": (*transition.BASE_BRANCHES["source"], *PLUME_MONOTONE_DELIVERY_FEATURES),
    "transport": transition.BASE_BRANCHES["transport"],
    "retention": transition.BASE_BRANCHES["retention"],
    "background": transition.BASE_BRANCHES["background"],
}
PLUME_SIGNED_PROCESS_BRANCHES = {
    "source": PLUME_SOURCE_FEATURES,
    "transport": (
        "h26_nearest_supported_outlet_distance_km",
        "h26_outlet_discharge_decay_100km",
        "h26_cmems_positive_outlet_to_cell_current_ms",
        "h26_cmems_positive_cell_to_outlet_current_ms",
        "positive_water_depth_m",
    ),
    "retention": (
        "h26_cmems_positive_convergence_sinv",
        "h26_cmems_mld_m_mean",
        "m6_sea_surface_salinity_psu",
    ),
    "background": transition.BASE_BRANCHES["background"],
}
PLUME_PRESSURE_DELIVERY_FEATURES = (
    "h26_outlet_spot_delivery_decay_100km",
    "h26_outlet_nightlight_delivery_decay_100km",
    "h26_outlet_road_delivery_decay_100km",
    "h26_outlet_population_delivery_decay_100km",
)
PLUME_SIGNED_DELIVERY_PROCESS_BRANCHES = {
    # Single-module challenger: replace the four kernel-normalized outlet
    # pressure means with absolute distance-decayed delivery sums.  All other
    # branches, signs, caps, losses, and validation logic match
    # outlet_plume_signed_process.
    "source": (
        "h26_outlet_river_export_decay_100km",
        *PLUME_PRESSURE_DELIVERY_FEATURES,
    ),
    "transport": PLUME_SIGNED_PROCESS_BRANCHES["transport"],
    "retention": PLUME_SIGNED_PROCESS_BRANCHES["retention"],
    "background": PLUME_SIGNED_PROCESS_BRANCHES["background"],
}
PLUME_SIGNED_PROCESS_FEATURES = tuple(dict.fromkeys(
    name for values in PLUME_SIGNED_PROCESS_BRANCHES.values() for name in values
))
PLUME_SIGNED_DELIVERY_PROCESS_FEATURES = tuple(dict.fromkeys(
    name for values in PLUME_SIGNED_DELIVERY_PROCESS_BRANCHES.values() for name in values
))
PLUME_SIGNED_DIRECTIONS = {
    "source": {name: 1.0 for name in PLUME_SIGNED_PROCESS_BRANCHES["source"]},
    "transport": {
        "h26_nearest_supported_outlet_distance_km": -1.0,
        "h26_outlet_discharge_decay_100km": 1.0,
        "h26_cmems_positive_outlet_to_cell_current_ms": 1.0,
        "h26_cmems_positive_cell_to_outlet_current_ms": -1.0,
        "positive_water_depth_m": -1.0,
    },
    "retention": {
        "h26_cmems_positive_convergence_sinv": 1.0,
        "h26_cmems_mld_m_mean": -1.0,
        "m6_sea_surface_salinity_psu": -1.0,
    },
}
PLUME_SIGNED_DELIVERY_DIRECTIONS = {
    **PLUME_SIGNED_DIRECTIONS,
    "source": {
        name: 1.0 for name in PLUME_SIGNED_DELIVERY_PROCESS_BRANCHES["source"]
    },
}
caoa.LOG_FEATURES.update((*PLUME_FEATURES, *PLUME_PRESSURE_DELIVERY_FEATURES))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def branches_for_candidate(candidate: str) -> dict[str, tuple[str, ...]]:
    if candidate == "s0_same_plume_support":
        return transition.BASE_BRANCHES
    if candidate == "outlet_plume_delivery":
        return PLUME_BRANCHES
    if candidate == "outlet_plume_intensity":
        return PLUME_INTENSITY_BRANCHES
    if candidate == "outlet_plume_source_only":
        return PLUME_SOURCE_ONLY_BRANCHES
    if candidate == "outlet_plume_delivery_monotone":
        return PLUME_MONOTONE_DELIVERY_BRANCHES
    if candidate == "outlet_plume_signed_process":
        return PLUME_SIGNED_PROCESS_BRANCHES
    if candidate == "outlet_plume_signed_delivery_process":
        return PLUME_SIGNED_DELIVERY_PROCESS_BRANCHES
    raise RuntimeError(f"Unknown transition plume candidate: {candidate}")


def plume_features_for_candidate(candidate: str) -> tuple[str, ...]:
    if candidate == "outlet_plume_delivery":
        return PLUME_FEATURES
    if candidate == "outlet_plume_intensity":
        return PLUME_INTENSITY_FEATURES
    if candidate == "outlet_plume_source_only":
        return PLUME_INTENSITY_SOURCE_FEATURES
    if candidate == "outlet_plume_delivery_monotone":
        return PLUME_MONOTONE_DELIVERY_FEATURES
    if candidate == "outlet_plume_signed_process":
        return PLUME_SIGNED_PROCESS_FEATURES
    if candidate == "outlet_plume_signed_delivery_process":
        return PLUME_SIGNED_DELIVERY_PROCESS_FEATURES
    return ()


def requested_features(branches: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name for values in branches.values() for name in values))
    forbidden = sorted(set(names) & transition.FORBIDDEN_MEAN_FEATURES)
    require(not forbidden, f"Forbidden transition plume mean features: {forbidden}")
    return names


def plume_support(frame: pd.DataFrame) -> np.ndarray:
    return frame[ACTIVE_PLUME_STATUS].astype(str).eq("observed").to_numpy()


def load_inputs(
    args: argparse.Namespace,
    branches: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features = requested_features(branches)
    train_header = set(pd.read_csv(args.training, nrows=0).columns)
    grid_header = set(pd.read_csv(args.prediction, nrows=0).columns)
    source_status = (
        VECTOR_PLUME_STATUS
        if args.candidate in (
            "outlet_plume_signed_process",
            "outlet_plume_signed_delivery_process",
        )
        else PLUME_STATUS
    )
    process_statuses = [PLUME_STATUS, source_status]
    if args.candidate == "outlet_plume_signed_delivery_process":
        process_statuses.append(PRESSURE_DELIVERY_STATUS)
    process_statuses = list(dict.fromkeys(process_statuses))
    required_train = {"stable_record_id", *process_statuses, *features}
    required_grid = {
        "prediction_id", "longitude", "latitude", "prediction_domain",
        *process_statuses, *features,
    }
    require(required_train.issubset(train_header), f"Training table missing {sorted(required_train - train_header)}")
    require(required_grid.issubset(grid_header), f"Prediction table missing {sorted(required_grid - grid_header)}")
    training = pd.read_csv(
        args.training,
        usecols=list(dict.fromkeys(["stable_record_id", *process_statuses, *features])),
        low_memory=False,
    )
    grid = pd.read_csv(
        args.prediction,
        usecols=[
            "prediction_id", "longitude", "latitude", "prediction_domain",
            *process_statuses, *features,
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
    records_all[ACTIVE_PLUME_STATUS] = records_all[source_status].astype(str)
    if args.candidate == "outlet_plume_signed_delivery_process":
        delivery_unsupported = ~records_all[PRESSURE_DELIVERY_STATUS].astype(str).eq("observed")
        records_all.loc[delivery_unsupported, ACTIVE_PLUME_STATUS] = records_all.loc[
            delivery_unsupported, PRESSURE_DELIVERY_STATUS
        ].astype(str)
    record_complete = np.isfinite(np.column_stack([
        numeric(records_all[name]) for name in features
    ])).all(axis=1)
    records_all.loc[
        records_all[ACTIVE_PLUME_STATUS].astype(str).eq("observed") & ~record_complete,
        ACTIVE_PLUME_STATUS,
    ] = "source_missing"
    record_supported = plume_support(records_all)
    records = records_all.loc[record_supported].reset_index(drop=True)
    records[PLUME_SOURCE_STATUS_ALIAS] = records[ACTIVE_PLUME_STATUS].astype(str)
    require(len(records) >= 500, f"Too few plume-supported transition records: {len(records):,}")

    grid = grid.loc[grid["prediction_domain"].eq(DOMAIN)].copy()
    grid = grid.rename(columns={"prediction_domain": "waterbody"}).reset_index(drop=True)
    grid[ACTIVE_PLUME_STATUS] = grid[source_status].astype(str)
    if args.candidate == "outlet_plume_signed_delivery_process":
        delivery_unsupported = ~grid[PRESSURE_DELIVERY_STATUS].astype(str).eq("observed")
        grid.loc[delivery_unsupported, ACTIVE_PLUME_STATUS] = grid.loc[
            delivery_unsupported, PRESSURE_DELIVERY_STATUS
        ].astype(str)
    grid_complete = np.isfinite(np.column_stack([
        numeric(grid[name]) for name in features
    ])).all(axis=1)
    grid.loc[
        grid[ACTIVE_PLUME_STATUS].astype(str).eq("observed") & ~grid_complete,
        ACTIVE_PLUME_STATUS,
    ] = "source_missing"
    grid_supported = plume_support(grid)
    grid[PLUME_SOURCE_STATUS_ALIAS] = grid[ACTIVE_PLUME_STATUS].astype(str)
    caoa_frame = pd.read_csv(args.caoa, usecols=["prediction_id", "caoa_status"], low_memory=False)
    grid = grid.merge(caoa_frame, on="prediction_id", how="left", validate="one_to_one")
    require(grid["caoa_status"].notna().all(), "C-AOA join did not close")

    plume_features = plume_features_for_candidate(args.candidate)
    if plume_features:
        train_missing = {
            name: int((~np.isfinite(numeric(records[name]))).sum()) for name in plume_features
        }
        grid_missing = {
            name: int((~np.isfinite(numeric(grid.loc[grid_supported, name]))).sum())
            for name in plume_features
        }
        require(not any(train_missing.values()), f"Supported records have missing plume values: {train_missing}")
        require(not any(grid_missing.values()), f"Supported grid has missing plume values: {grid_missing}")
    else:
        train_missing, grid_missing = {}, {}
    return records, grid, {
        "candidate": args.candidate,
        "records_before_plume_support": int(len(records_all)),
        "records_supported": int(len(records)),
        "records_excluded": int(len(records_all) - len(records)),
        "record_plume_status_counts": {
            str(k): int(v) for k, v in records_all[PLUME_STATUS].value_counts(dropna=False).items()
        },
        "record_active_process_status_counts": {
            str(k): int(v) for k, v in records_all[ACTIVE_PLUME_STATUS].value_counts(dropna=False).items()
        },
        "record_pressure_delivery_status_counts": (
            {
                str(k): int(v)
                for k, v in records_all[PRESSURE_DELIVERY_STATUS].value_counts(dropna=False).items()
            }
            if args.candidate == "outlet_plume_signed_delivery_process"
            else {}
        ),
        "grid_rows_retained": int(len(grid)),
        "grid_supported": int(grid_supported.sum()),
        "grid_unsupported": int((~grid_supported).sum()),
        "grid_plume_status_counts": {
            str(k): int(v) for k, v in grid[PLUME_STATUS].value_counts(dropna=False).items()
        },
        "grid_active_process_status_counts": {
            str(k): int(v) for k, v in grid[ACTIVE_PLUME_STATUS].value_counts(dropna=False).items()
        },
        "grid_pressure_delivery_status_counts": (
            {
                str(k): int(v)
                for k, v in grid[PRESSURE_DELIVERY_STATUS].value_counts(dropna=False).items()
            }
            if args.candidate == "outlet_plume_signed_delivery_process"
            else {}
        ),
        "plume_feature_missing_supported_records": train_missing,
        "plume_feature_missing_supported_grid": grid_missing,
        "support_policy": "within 300 km of a fully supported RiverATLAS terminal outlet",
    }


def configure_shared_runner() -> None:
    original_fit_candidate_model = freshwater.fit_candidate_model
    original_crossfit_calibration_predictions = freshwater.crossfit_calibration_predictions
    original_direction_orientations = freshwater.direction_orientations_for_candidate

    def fit_transition_candidate(*values, candidate: str, **keywords):
        mapped = (
            "network_source_monotone"
            if candidate == "outlet_plume_delivery_monotone"
            else candidate
        )
        return original_fit_candidate_model(*values, candidate=mapped, **keywords)

    def crossfit_transition_candidate(*values, candidate: str, **keywords):
        mapped = (
            "network_source_monotone"
            if candidate == "outlet_plume_delivery_monotone"
            else candidate
        )
        return original_crossfit_calibration_predictions(
            *values, candidate=mapped, **keywords
        )

    def transition_direction_orientations(
        candidate: str,
        names: tuple[str, ...],
        branches: dict[str, tuple[str, ...]],
    ) -> dict[str, list[float]]:
        if candidate == "outlet_plume_delivery_monotone":
            source = set(branches["source"])
            return {"source": [1.0 for name in names if name in source]}
        if candidate in (
            "outlet_plume_signed_process",
            "outlet_plume_signed_delivery_process",
        ):
            directions = (
                PLUME_SIGNED_DELIVERY_DIRECTIONS
                if candidate == "outlet_plume_signed_delivery_process"
                else PLUME_SIGNED_DIRECTIONS
            )
            result: dict[str, list[float]] = {}
            for branch, mapping in directions.items():
                local_names = [name for name in names if name in set(branches[branch])]
                require(
                    set(local_names) == set(mapping),
                    f"Signed transition branch {branch} did not retain its full process set: "
                    f"{local_names} vs {sorted(mapping)}",
                )
                result[branch] = [float(mapping[name]) for name in local_names]
            return result
        return original_direction_orientations(candidate, names, branches)

    freshwater.DOMAIN = DOMAIN
    freshwater.NETWORK_STATUS = ACTIVE_PLUME_STATUS
    freshwater.SOURCE_STATUS = PLUME_SOURCE_STATUS_ALIAS
    freshwater.SUPPORTED_NETWORK_STATES = ("observed",)
    freshwater.BASE_BRANCHES = transition.BASE_BRANCHES
    freshwater.NETWORK_NODIRECTION_BRANCHES = PLUME_BRANCHES
    freshwater.NETWORK_NODIRECTION_ONLY_FEATURES = PLUME_FEATURES
    freshwater.branches_for_candidate = branches_for_candidate
    freshwater.network_features_for_candidate = plume_features_for_candidate
    freshwater.requested_features = requested_features
    freshwater.network_support = plume_support
    freshwater.load_inputs = load_inputs
    freshwater.fit_candidate_model = fit_transition_candidate
    freshwater.crossfit_calibration_predictions = crossfit_transition_candidate
    freshwater.direction_orientations_for_candidate = transition_direction_orientations


def rewrite_audits(args: argparse.Namespace) -> None:
    for arm in ("protocol_aware", "detection_unaware"):
        path = args.run_root / "S0" / arm / DOMAIN / "H26_UNIT_AUDIT.json"
        require(path.exists(), f"Missing shared output audit {path}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["status"] = "H26_TRANSITION_OUTLET_PLUME_DEVELOPMENT_NOT_RELEASE"
        audit["domain"] = DOMAIN
        audit["candidate"] = args.candidate
        audit["plume_contract"] = {
            "features": list(plume_features_for_candidate(args.candidate)),
            "maximum_distance_km": 300.0,
            "decay_km": 100.0,
            "regional_targets_used": False,
            "observation_metadata_affects_q50": False,
        }
        if args.candidate == "outlet_plume_delivery_monotone":
            audit["source_direction_constraint"] = {
                "active": True,
                "orientation": "positive_monotone_for_pressure_and_distance_decayed_delivery",
                "raw_distance_in_mean": False,
                "regional_targets_used": False,
            }
        if args.candidate == "outlet_plume_signed_process":
            audit["source_direction_constraint"] = {
                "active": True,
                "branch_feature_orientations": PLUME_SIGNED_DIRECTIONS,
                "current_projection": "CMEMS_u_v_dot_nearest_outlet_to_cell_direction",
                "regional_targets_used": False,
            }
        if args.candidate == "outlet_plume_signed_delivery_process":
            audit["source_direction_constraint"] = {
                "active": True,
                "branch_feature_orientations": PLUME_SIGNED_DELIVERY_DIRECTIONS,
                "source_feature_change": (
                    "four kernel-normalized outlet pressure means replaced by "
                    "absolute distance-decayed delivery sums"
                ),
                "current_projection": "CMEMS_u_v_dot_nearest_outlet_to_cell_direction",
                "regional_targets_used": False,
            }
        map_audit = audit.get("map", {})
        if "all_freshwater_rows_retained" in map_audit:
            map_audit["all_transition_coastal_rows_retained"] = map_audit.pop(
                "all_freshwater_rows_retained"
            )
        audit["limits"] = [
            "spherical outlet-distance kernel is a delivery proxy, not explicit coastal advection",
            "outer fold is untouched by fitting, early stopping, target scaling, and variance calibration",
            "unsupported transition cells remain in outputs with NA process predictions",
            "regional outcomes are diagnostics only",
            "GP is off",
        ]
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    configure_shared_runner()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        status = freshwater.train(args)
    rewrite_audits(args)
    shared: dict[str, object] = {}
    for line in reversed(capture.getvalue().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            shared = value
            break
    print(json.dumps({
        "status": "H26_TRANSITION_OUTLET_PLUME_COMPLETED",
        "candidate": args.candidate, "fold": int(args.fold),
        "validation_scheme": args.validation_scheme,
        "metrics": shared.get("metrics", {}),
    }, ensure_ascii=False))
    return status


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--candidate",
        choices=(
            "s0_same_plume_support", "outlet_plume_delivery",
            "outlet_plume_intensity", "outlet_plume_source_only",
            "outlet_plume_delivery_monotone",
            "outlet_plume_signed_process",
            "outlet_plume_signed_delivery_process",
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
