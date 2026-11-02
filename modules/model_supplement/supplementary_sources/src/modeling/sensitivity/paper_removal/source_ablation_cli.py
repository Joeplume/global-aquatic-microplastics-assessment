#!/usr/bin/env python3
"""H26T paper-source ablation manifests, split filtering, and OOF aggregation.

This module deliberately separates the *formal OOF cohort* from the larger
observation table.  Paper removal is applied only after the unchanged outer
site-event split has been chosen.  The held-out rows are never removed.  The
returned retained outer-training indices must be the only rows used by every
downstream fitted object (scaler, BNN, AOA/C-AOA reference, MEM/source offsets,
cross-fit calibration, and interval calibration).

The CLI constructs removal manifests and aggregates saved OOF results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TUNING_ROOT = SCRIPT_DIR.parent
MODELING_ROOT = TUNING_ROOT.parent
H26_ROOT = MODELING_ROOT / "h26"
DEFAULT_OOF = (
    H26_ROOT
    / "evaluation"
    / "grouped"
    / "H26_FINAL10_OOF.csv.gz"
)
DEFAULT_TRAINING = (
    H26_ROOT
    / "data"
    / "observations"
    / "h23_observation_layer_training_records.csv.gz"
)
MANIFEST_ROOT = SCRIPT_DIR / "source_manifests"
RUNS_ROOT = SCRIPT_DIR / "oof_predictions"
METRICS_ROOT = SCRIPT_DIR / "paired_metrics"
MAP_ROOT = SCRIPT_DIR / "figures_maps"

DOMAIN_ORDER = ("freshwater", "transition", "marine")
WATERBODY_TO_DOMAIN = {
    "reference_surface_freshwater": "freshwater",
    "transition_coastal": "transition",
    "marine": "marine",
}
DOMAIN_TO_WATERBODY = {value: key for key, value in WATERBODY_TO_DOMAIN.items()}
REMOVAL_COUNTS = {30: 389, 50: 648, 80: 1036}
EXPECTED_PAPERS = 1295
EXPECTED_EXTERNAL = 58
EXPECTED_CROSS_WATERBODY_PAPERS = 42
EXPECTED_OOF_ROWS = 24129
EXPECTED_FOLDS = tuple(range(1, 11))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=compression)


def bool_values(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(bool)
    text = values.fillna("").astype(str).str.strip().str.lower()
    return text.isin(("true", "1", "yes", "y")).to_numpy(bool)


def source_parts(value: object) -> tuple[str, str, str]:
    """Return source_kind, base_source_id, and source id without waterbody prefix."""
    raw = "" if pd.isna(value) else str(value).strip()
    tail = raw.split("||", 1)[1] if "||" in raw else raw
    if "::" not in tail:
        return "unknown", tail, tail
    kind, base = tail.split("::", 1)
    return kind.lower(), base, tail


def add_source_identity(frame: pd.DataFrame) -> pd.DataFrame:
    require("source_group" in frame.columns, "source_group is required")
    result = frame.copy()
    parsed = result["source_group"].map(source_parts)
    result["source_kind"] = parsed.map(lambda x: x[0])
    result["base_source_id"] = parsed.map(lambda x: x[1])
    result["normalized_source_id"] = parsed.map(lambda x: x[2])
    return result


def add_domain(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "domain" in result.columns:
        domain = result["domain"].astype(str).str.strip().str.lower()
        domain = domain.replace({"transition_coastal": "transition", "reference_surface_freshwater": "freshwater"})
    else:
        require("waterbody" in result.columns, "domain or waterbody is required")
        domain = result["waterbody"].map(WATERBODY_TO_DOMAIN)
    require(domain.notna().all(), "Unknown waterbody/domain in formal natural-water cohort")
    require(set(domain.unique()).issubset(set(DOMAIN_ORDER)), "Unexpected domain label")
    result["domain"] = domain
    return result


def read_baseline_oof(path: Path = DEFAULT_OOF) -> pd.DataFrame:
    require(path.exists(), f"Baseline OOF not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"stable_record_id", "waterbody", "source_group", "observed_log10"}
    require(required.issubset(frame.columns), f"Baseline OOF missing columns: {sorted(required - set(frame.columns))}")
    if "formal_fold" not in frame.columns:
        require("outer_validation_fold" in frame.columns, "Baseline OOF lacks formal_fold")
        frame["formal_fold"] = frame["outer_validation_fold"]
    frame["formal_fold"] = pd.to_numeric(frame["formal_fold"], errors="raise").astype(int)
    frame = add_source_identity(add_domain(frame))
    require(len(frame) == EXPECTED_OOF_ROWS, f"Expected {EXPECTED_OOF_ROWS} OOF rows, found {len(frame)}")
    require(frame["stable_record_id"].is_unique, "Baseline OOF stable_record_id must be unique")
    require(set(frame["formal_fold"].unique()) == set(EXPECTED_FOLDS), "Formal fold labels must be 1..10")
    if "validation_site_event_key" in frame.columns:
        max_folds = frame.groupby("validation_site_event_key", dropna=False)["formal_fold"].nunique().max()
        require(int(max_folds) == 1, "A validation site-event occurs in multiple folds")
    return frame


def formal_source_summary(frame: pd.DataFrame) -> dict[str, int]:
    papers = frame.loc[frame["source_kind"].eq("paper")]
    external = frame.loc[frame["source_kind"].eq("external")]
    unknown = frame.loc[~frame["source_kind"].isin(("paper", "external"))]
    require(unknown.empty, f"Unexpected formal source kinds: {sorted(unknown['source_kind'].unique())}")
    paper_domain_n = papers.groupby("base_source_id", observed=True)["domain"].nunique()
    return {
        "oof_rows": int(len(frame)),
        "papers": int(papers["base_source_id"].nunique()),
        "external_sources": int(external["base_source_id"].nunique()),
        "cross_waterbody_papers": int((paper_domain_n > 1).sum()),
        "paper_domain_source_groups": int(papers["source_group"].nunique()),
        "external_domain_source_groups": int(external["source_group"].nunique()),
    }


def assert_formal_counts(summary: dict[str, int]) -> None:
    require(summary["papers"] == EXPECTED_PAPERS, f"Expected {EXPECTED_PAPERS} papers, found {summary['papers']}")
    require(
        summary["external_sources"] == EXPECTED_EXTERNAL,
        f"Expected {EXPECTED_EXTERNAL} external sources, found {summary['external_sources']}",
    )
    require(
        summary["cross_waterbody_papers"] == EXPECTED_CROSS_WATERBODY_PAPERS,
        "Expected 42 cross-waterbody papers, "
        f"found {summary['cross_waterbody_papers']}",
    )


def domain_combo(values: Iterable[str]) -> str:
    present = set(values)
    return "+".join(domain for domain in DOMAIN_ORDER if domain in present)


def build_paper_universe(oof: pd.DataFrame) -> pd.DataFrame:
    papers = oof.loc[oof["source_kind"].eq("paper")].copy()
    rows: list[dict[str, object]] = []
    for paper_id, local in papers.groupby("base_source_id", sort=True, observed=True):
        domains = tuple(domain for domain in DOMAIN_ORDER if domain in set(local["domain"]))
        rows.append(
            {
                "paper_id_base": str(paper_id),
                "normalized_source_id": f"paper::{paper_id}",
                "waterbody_combo": "+".join(domains),
                "waterbody_count": int(len(domains)),
                "oof_records": int(len(local)),
                "domain_source_groups": int(local["source_group"].nunique()),
                "freshwater_records": int(local["domain"].eq("freshwater").sum()),
                "transition_records": int(local["domain"].eq("transition").sum()),
                "marine_records": int(local["domain"].eq("marine").sum()),
            }
        )
    result = pd.DataFrame(rows).sort_values("paper_id_base").reset_index(drop=True)
    require(len(result) == EXPECTED_PAPERS, "Paper universe count changed during construction")
    require(result["paper_id_base"].is_unique, "Base paper IDs must be unique")
    return result


def build_external_universe(oof: pd.DataFrame) -> pd.DataFrame:
    external = oof.loc[oof["source_kind"].eq("external")].copy()
    rows: list[dict[str, object]] = []
    for source_id, local in external.groupby("base_source_id", sort=True, observed=True):
        rows.append(
            {
                "external_id_base": str(source_id),
                "normalized_source_id": f"external::{source_id}",
                "waterbody_combo": domain_combo(local["domain"]),
                "oof_records": int(len(local)),
                "domain_source_groups": int(local["source_group"].nunique()),
                "deletion_eligible": False,
                "retention_rule": "external_fixed",
            }
        )
    result = pd.DataFrame(rows).sort_values("external_id_base").reset_index(drop=True)
    require(len(result) == EXPECTED_EXTERNAL, "External source universe count changed")
    return result


def stratified_random_order(
    universe: pd.DataFrame,
    *,
    seed: int,
    fold: int,
    round_id: int,
) -> pd.DataFrame:
    """Generate one nested order with near-proportional prefixes by combo.

    Papers are randomly permuted *within* waterbody-combination strata.  Their
    normalized within-stratum ranks are then interleaved.  Therefore every
    removal prefix is stratified without weighting papers by row count.
    """
    rng = np.random.default_rng(int(seed) + int(round_id) * 100_003 + int(fold) * 1_009)
    blocks: list[pd.DataFrame] = []
    for combo, local in universe.groupby("waterbody_combo", sort=True, observed=True):
        order = rng.permutation(len(local))
        block = local.iloc[order].copy().reset_index(drop=True)
        n = len(block)
        shift = float(rng.uniform(0.05, 0.95))
        block["stratified_interleave_score"] = (np.arange(n, dtype=float) + shift) / n
        block["random_tiebreak"] = rng.random(n)
        block["stratum_size"] = int(n)
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    result = result.sort_values(
        ["stratified_interleave_score", "random_tiebreak", "waterbody_combo", "paper_id_base"],
        kind="mergesort",
    ).reset_index(drop=True)
    result["random_order_rank"] = np.arange(1, len(result) + 1, dtype=int)
    result["fold"] = int(fold)
    result["random_round"] = int(round_id)
    result["random_seed"] = int(seed) + int(round_id) * 100_003 + int(fold) * 1_009
    for percent, count in REMOVAL_COUNTS.items():
        result[f"selected_remove_{percent}"] = result["random_order_rank"].le(count)
    require(result["paper_id_base"].is_unique, "Random order duplicated a paper")
    return result


def manifest_dir(round_id: int) -> Path:
    return MANIFEST_ROOT / f"round_{int(round_id):02d}"


def manifest_path(round_id: int, fold: int, removal: int) -> Path:
    return manifest_dir(round_id) / f"fold_{int(fold):02d}_remove_{int(removal)}.csv"


def prepare(args: argparse.Namespace) -> int:
    oof = read_baseline_oof(args.baseline_oof)
    summary = formal_source_summary(oof)
    assert_formal_counts(summary)

    training_header = pd.read_csv(args.training_table, nrows=0).columns
    required_training = {"stable_record_id", "waterbody", "source_group"}
    require(
        required_training.issubset(training_header),
        f"Training table missing: {sorted(required_training - set(training_header))}",
    )
    training_identity = pd.read_csv(
        args.training_table,
        usecols=["stable_record_id", "waterbody", "source_group"],
        low_memory=False,
    )
    require(training_identity["stable_record_id"].is_unique, "Training stable_record_id must be unique")
    cohort = training_identity.loc[training_identity["stable_record_id"].isin(set(oof["stable_record_id"]))].copy()
    require(len(cohort) == len(oof), "Formal OOF IDs are not represented exactly once in training table")
    compare = oof[["stable_record_id", "source_group", "waterbody"]].merge(
        cohort,
        on="stable_record_id",
        how="outer",
        suffixes=("_oof", "_training"),
        validate="one_to_one",
        indicator=True,
    )
    require(compare["_merge"].eq("both").all(), "Training/OOF formal record IDs differ")
    require(
        compare["source_group_oof"].astype(str).eq(compare["source_group_training"].astype(str)).all(),
        "Training/OOF source_group mismatch",
    )
    require(
        compare["waterbody_oof"].astype(str).eq(compare["waterbody_training"].astype(str)).all(),
        "Training/OOF waterbody mismatch",
    )

    universe = build_paper_universe(oof)
    external = build_external_universe(oof)
    out = manifest_dir(args.round)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "PAPER_UNIVERSE.csv", universe)
    write_csv(out / "EXTERNAL_SOURCES_FIXED.csv", external)

    manifest_rows: list[dict[str, object]] = []
    strata_rows: list[dict[str, object]] = []
    previous_by_fold: dict[int, set[str]] = {}
    for fold in args.folds:
        order = stratified_random_order(universe, seed=args.seed, fold=fold, round_id=args.round)
        paper_train_rows = (
            oof.loc[oof["source_kind"].eq("paper") & oof["formal_fold"].ne(fold)]
            .groupby("base_source_id", observed=True)
            .size()
        )
        paper_validation_rows = (
            oof.loc[oof["source_kind"].eq("paper") & oof["formal_fold"].eq(fold)]
            .groupby("base_source_id", observed=True)
            .size()
        )
        order["fold_outer_train_rows"] = order["paper_id_base"].map(paper_train_rows).fillna(0).astype(int)
        order["fold_heldout_rows"] = order["paper_id_base"].map(paper_validation_rows).fillna(0).astype(int)
        order["present_in_outer_train"] = order["fold_outer_train_rows"].gt(0)
        write_csv(out / f"fold_{fold:02d}_PAPER_RANDOM_ORDER.csv", order)

        previous: set[str] = set()
        for removal in args.removals:
            count = REMOVAL_COUNTS[removal]
            selected = order.loc[order["random_order_rank"].le(count)].copy()
            selected["removal_percent"] = int(removal)
            selected["selected_for_removal"] = True
            selected["external_sources_removed"] = 0
            selected_ids = set(selected["paper_id_base"])
            require(len(selected_ids) == count, f"Fold {fold} removal {removal}: wrong paper count")
            require(previous.issubset(selected_ids), f"Fold {fold}: removal prefixes are not nested")
            previous = selected_ids
            write_csv(manifest_path(args.round, fold, removal), selected)
            manifest_rows.append(
                {
                    "random_round": int(args.round),
                    "fold": int(fold),
                    "removal_percent": int(removal),
                    "papers_total": EXPECTED_PAPERS,
                    "papers_selected": int(len(selected)),
                    "papers_retained": int(EXPECTED_PAPERS - len(selected)),
                    "selected_present_in_outer_train": int(selected["present_in_outer_train"].sum()),
                    "selected_absent_from_outer_train": int((~selected["present_in_outer_train"]).sum()),
                    "outer_train_rows_selected": int(selected["fold_outer_train_rows"].sum()),
                    "external_sources_total": EXPECTED_EXTERNAL,
                    "external_sources_removed": 0,
                    "nested_prefix": True,
                    "selection_weighted_by_record_count": False,
                }
            )
            for combo, local in selected.groupby("waterbody_combo", sort=True, observed=True):
                strata_rows.append(
                    {
                        "random_round": int(args.round),
                        "fold": int(fold),
                        "removal_percent": int(removal),
                        "waterbody_combo": combo,
                        "papers_selected": int(len(local)),
                        "papers_in_universe": int((universe["waterbody_combo"] == combo).sum()),
                    }
                )
        previous_by_fold[int(fold)] = previous

    summary_frame = pd.DataFrame(manifest_rows)
    strata_frame = pd.DataFrame(strata_rows)
    write_csv(out / "MANIFEST_SUMMARY.csv", summary_frame)
    write_csv(out / "MANIFEST_STRATA_SUMMARY.csv", strata_frame)
    write_json(
        out / "MANIFEST_AUDIT.json",
        {
            "status": "completed",
            "formal_cohort": summary,
            "training_table_rows_total": int(len(training_identity)),
            "formal_training_rows_matched": int(len(cohort)),
            "folds": [int(x) for x in args.folds],
            "removals": {str(x): REMOVAL_COUNTS[x] for x in args.removals},
            "random_round": int(args.round),
            "base_seed": int(args.seed),
            "selection_unit": "base_paper_id_without_waterbody_prefix",
            "stratification": "waterbody_combination",
            "nested": True,
            "external_policy": "all 58 external sources fixed; never eligible for deletion",
            "record_count_used_for_selection": False,
            "target_or_concentration_used_for_selection": False,
            "formal_integration": (
                "apply_manifest_to_outer_split immediately after choose_outer_split; "
                "all fitted preprocessing/model/uncertainty/support objects use retained outer train only"
            ),
        },
    )
    print(json.dumps({"status": "completed", "output": str(out), **summary}, ensure_ascii=False))
    return 0


def read_removal_manifest(path: Path) -> pd.DataFrame:
    require(path.exists(), f"Removal manifest not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    require("paper_id_base" in frame.columns, "Manifest lacks paper_id_base")
    require(frame["paper_id_base"].notna().all(), "Manifest contains empty paper ID")
    require(frame["paper_id_base"].is_unique, "Manifest duplicates a base paper ID")
    require("external_id_base" not in frame.columns, "An external source was placed in paper-removal manifest")
    return frame


def apply_manifest_to_outer_split(
    records: pd.DataFrame,
    outer_train_rows: Sequence[int] | np.ndarray,
    heldout_rows: Sequence[int] | np.ndarray,
    manifest: Path | str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Remove selected papers from outer train while preserving heldout rows.

    This is the formal integration point for all three waterbody runners.  It
    must be called immediately after ``choose_outer_split`` and before any fit,
    early-stopping split, scaling, support construction, MEM, or calibration.
    """
    require("source_group" in records.columns, "records must contain source_group")
    train_idx = np.asarray(outer_train_rows, dtype=int)
    held_idx = np.asarray(heldout_rows, dtype=int)
    require(train_idx.ndim == 1 and held_idx.ndim == 1, "Split indices must be one-dimensional")
    require(len(np.unique(train_idx)) == len(train_idx), "outer_train_rows contains duplicates")
    require(len(np.unique(held_idx)) == len(held_idx), "heldout_rows contains duplicates")
    require(not set(train_idx).intersection(set(held_idx)), "Outer train and heldout indices overlap")
    require(
        (len(train_idx) == 0 or (train_idx.min() >= 0 and train_idx.max() < len(records)))
        and (len(held_idx) == 0 or (held_idx.min() >= 0 and held_idx.max() < len(records))),
        "Split index is outside records",
    )
    manifest_frame = read_removal_manifest(Path(manifest))
    selected = set(manifest_frame["paper_id_base"].astype(str))

    train_identity = add_source_identity(records.iloc[train_idx][["source_group"]])
    is_selected_paper = train_identity["source_kind"].eq("paper") & train_identity["base_source_id"].isin(selected)
    keep = ~is_selected_paper.to_numpy(bool)
    retained = train_idx[keep]
    removed = train_idx[~keep]
    heldout_unchanged = held_idx.copy()

    retained_identity = add_source_identity(records.iloc[retained][["source_group"]])
    leakage = retained_identity["source_kind"].eq("paper") & retained_identity["base_source_id"].isin(selected)
    external_before = train_identity.loc[train_identity["source_kind"].eq("external"), "normalized_source_id"]
    external_after = retained_identity.loc[retained_identity["source_kind"].eq("external"), "normalized_source_id"]
    require(not leakage.any(), "Selected paper remains in retained outer training rows")
    require(
        external_before.value_counts().sort_index().equals(external_after.value_counts().sort_index()),
        "External training records changed during paper ablation",
    )
    require(np.array_equal(heldout_unchanged, held_idx), "Heldout rows changed during ablation")

    audit = {
        "status": "passed",
        "manifest": str(Path(manifest).resolve()),
        "outer_train_rows_before": int(len(train_idx)),
        "outer_train_rows_after": int(len(retained)),
        "outer_train_rows_removed": int(len(removed)),
        "heldout_rows_before": int(len(held_idx)),
        "heldout_rows_after": int(len(heldout_unchanged)),
        "heldout_rows_identical_and_order_preserved": True,
        "removed_paper_overlap_after_filter": 0,
        "selected_papers_in_manifest": int(len(selected)),
        "selected_papers_present_in_outer_train": int(
            train_identity.loc[is_selected_paper, "base_source_id"].nunique()
        ),
        "external_train_rows_before": int(len(external_before)),
        "external_train_rows_after": int(len(external_after)),
        "external_records_preserved": True,
        "apply_before": [
            "inner_split",
            "target_center_and_scale",
            "covariate_scaler_fit",
            "AOA_and_CAOA_reference_fit",
            "BNN_fit",
            "crossfit_variance_calibration",
            "MEM_and_source_offset_fit",
            "interval_calibration",
        ],
    }
    return retained, heldout_unchanged, audit


def selected_domain(args_domain: str) -> str | None:
    return None if args_domain == "all" else args_domain


def default_split_output(round_id: int, removal: int, fold: int, domain: str) -> Path:
    return (
        RUNS_ROOT
        / f"round_{round_id:02d}"
        / f"removal_{removal}"
        / f"fold_{fold:02d}"
        / "inputs"
        / domain
    )


def filter_training(args: argparse.Namespace) -> int:
    """Materialize the post-outer-split filtering evidence for one task.

    The output is intentionally two separate sets.  It is not a combined
    record table because a combined pre-filter can silently delete validation
    records or allow them back into fitting.  Formal training must call
    :func:`apply_manifest_to_outer_split` inside the model runner.
    """
    oof = read_baseline_oof(args.baseline_oof)
    assert_formal_counts(formal_source_summary(oof))
    manifest = args.manifest or manifest_path(args.round, args.fold, args.removal)
    manifest_frame = read_removal_manifest(manifest)
    require(len(manifest_frame) == REMOVAL_COUNTS[args.removal], "Manifest removal count is wrong")

    if args.domain != "all":
        oof = oof.loc[oof["domain"].eq(args.domain)].copy()
    train_ids = oof.loc[oof["formal_fold"].ne(args.fold), "stable_record_id"].astype(str)
    heldout_ids = oof.loc[oof["formal_fold"].eq(args.fold), "stable_record_id"].astype(str)

    header = pd.read_csv(args.training_table, nrows=0).columns
    required = {"stable_record_id", "source_group", "waterbody"}
    require(required.issubset(header), f"Training table missing: {sorted(required - set(header))}")
    raw = pd.read_csv(args.training_table, low_memory=False)
    raw["stable_record_id"] = raw["stable_record_id"].astype(str)
    require(raw["stable_record_id"].is_unique, "Training stable_record_id must be unique")
    record_lookup = raw.set_index("stable_record_id", drop=False)
    missing = (set(train_ids) | set(heldout_ids)) - set(record_lookup.index)
    require(not missing, f"Formal IDs missing from training table: {len(missing)}")
    outer_train = record_lookup.loc[list(train_ids)].copy()
    heldout = record_lookup.loc[list(heldout_ids)].copy()
    combined_records = pd.concat([outer_train, heldout], ignore_index=True)
    outer_idx = np.arange(len(outer_train), dtype=int)
    held_idx = np.arange(len(outer_train), len(combined_records), dtype=int)
    retained_idx, heldout_idx, split_audit = apply_manifest_to_outer_split(
        combined_records, outer_idx, held_idx, manifest
    )
    retained = combined_records.iloc[retained_idx].copy()
    removed = combined_records.iloc[np.setdiff1d(outer_idx, retained_idx, assume_unique=True)].copy()
    heldout_after = combined_records.iloc[heldout_idx].copy()

    require(
        heldout_after["stable_record_id"].tolist() == heldout["stable_record_id"].tolist(),
        "Validation IDs/order changed",
    )
    out = args.output_dir or default_split_output(args.round, args.removal, args.fold, args.domain)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "RETAINED_OUTER_TRAIN_IDS.csv", retained[["stable_record_id", "source_group", "waterbody"]])
    write_csv(out / "REMOVED_OUTER_TRAIN_IDS.csv", removed[["stable_record_id", "source_group", "waterbody"]])
    write_csv(
        out / "HELDOUT_VALIDATION_IDS_UNCHANGED.csv",
        heldout_after[["stable_record_id", "source_group", "waterbody"]],
    )
    if args.materialize_records:
        write_csv(out / "RETAINED_OUTER_TRAIN_RECORDS.csv.gz", retained, compression="gzip")
        write_csv(out / "HELDOUT_VALIDATION_RECORDS_UNCHANGED.csv.gz", heldout_after, compression="gzip")

    audit = {
        **split_audit,
        "formal_runner_status": "pending_model_fit",
        "random_round": int(args.round),
        "removal_percent": int(args.removal),
        "fold": int(args.fold),
        "domain": args.domain,
        "formal_cohort_rows_for_domain": int(len(oof)),
        "materialized_records": bool(args.materialize_records),
        "integration_method": "post_choose_outer_split_index_filter",
        "warning": (
            "These separated files are audit evidence. A formal run is valid only when the model runner "
            "calls apply_manifest_to_outer_split after choose_outer_split and uses retained indices for "
            "all downstream fitted objects."
        ),
    }
    write_json(out / "FILTER_AUDIT.json", audit)
    print(json.dumps({"status": "completed", "output": str(out), **audit}, ensure_ascii=False))
    return 0


def source_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    source = frame["source_group"].fillna("__MISSING_SOURCE__").astype(str)
    counts = source.map(source.value_counts(dropna=False)).to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    return weights / weights.sum()


def record_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    return np.full(len(frame), 1.0 / max(len(frame), 1), dtype=float)


def weighted_r2(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    mean = float(np.average(y, weights=weight))
    denominator = float(np.sum(weight * (y - mean) ** 2))
    if denominator <= 0:
        return math.nan
    return 1.0 - float(np.sum(weight * (y - pred) ** 2)) / denominator


def wis_values(y: np.ndarray, pred: np.ndarray, sd: np.ndarray) -> np.ndarray:
    alphas = (0.50, 0.20, 0.10, 0.05)
    total = 0.5 * np.abs(y - pred)
    for alpha in alphas:
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        lower = pred - z * sd
        upper = pred + z * sd
        interval = (
            upper
            - lower
            + (2.0 / alpha) * np.maximum(lower - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - upper, 0.0)
        )
        total = total + (alpha / 2.0) * interval
    return total / (len(alphas) + 0.5)


def interval_metric_row(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    sd: np.ndarray,
    weighting: str,
) -> dict[str, object]:
    y = pd.to_numeric(frame["observed_log10"], errors="coerce").to_numpy(float)
    prediction = np.asarray(prediction, dtype=float)
    sd = np.asarray(sd, dtype=float)
    ok = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(sd) & (sd > 0)
    local = frame.loc[ok].copy()
    y, prediction, sd = y[ok], prediction[ok], sd[ok]
    if len(local) == 0:
        return {
            "status": "no_finite_records",
            "records": 0,
            "source_groups": 0,
            "papers": 0,
            "r2": math.nan,
            "rmse_log10": math.nan,
            "mae_log10": math.nan,
            "coverage95": math.nan,
            "wis_log10": math.nan,
            "width95_mean_log10": math.nan,
            "width95_median_log10": math.nan,
        }
    weight = source_equal_weights(local) if weighting == "source_equal" else record_equal_weights(local)
    residual = y - prediction
    widths = 2.0 * NormalDist().inv_cdf(0.975) * sd
    papers = add_source_identity(local[["source_group"]])
    return {
        "status": "completed" if len(local) >= 2 else "single_finite_record",
        "records": int(len(local)),
        "source_groups": int(local["source_group"].nunique(dropna=False)),
        "papers": int(papers.loc[papers["source_kind"].eq("paper"), "base_source_id"].nunique()),
        "r2": weighted_r2(y, prediction, weight),
        "rmse_log10": float(np.sqrt(np.average(residual**2, weights=weight))),
        "mae_log10": float(np.average(np.abs(residual), weights=weight)),
        "coverage95": float(np.average(np.abs(residual) <= 1.96 * sd, weights=weight)),
        "wis_log10": float(np.average(wis_values(y, prediction, sd), weights=weight)),
        "width95_mean_log10": float(np.average(widths, weights=weight)),
        "width95_median_log10": float(np.median(widths)),
    }


def canonicalize_run_oof(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_source_identity(add_domain(frame))
    aliases = {
        "future_observation_sd_log10": "future_observation_sd_log10_aware",
        "known_source_conditional_future_sd_log10": "known_source_conditional_future_sd_log10_aware",
    }
    for source, target in aliases.items():
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    required = {
        "stable_record_id",
        "source_group",
        "observed_log10",
        "predicted_q50_log10",
        "future_observation_sd_log10_aware",
        "known_source_available",
        "known_source_conditional_q50_log10",
        "known_source_conditional_future_sd_log10_aware",
    }
    require(required.issubset(result.columns), f"Run OOF missing columns: {sorted(required - set(result.columns))}")
    if "predicted_q50_log10_unaware" not in result.columns:
        result["predicted_q50_log10_unaware"] = result["predicted_q50_log10"]
    if "future_observation_sd_log10_unaware" not in result.columns:
        result["future_observation_sd_log10_unaware"] = result["future_observation_sd_log10_aware"]
    if "known_source_conditional_q50_log10_unaware" not in result.columns:
        result["known_source_conditional_q50_log10_unaware"] = result["known_source_conditional_q50_log10"]
    if "known_source_conditional_future_sd_log10_unaware" not in result.columns:
        result["known_source_conditional_future_sd_log10_unaware"] = result[
            "known_source_conditional_future_sd_log10_aware"
        ]
    result["known_source_available"] = bool_values(result["known_source_available"])
    result["stable_record_id"] = result["stable_record_id"].astype(str)
    return result


def prediction_arrays(frame: pd.DataFrame, subset: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    suffix = "" if arm == "aware" else "_unaware"
    env_pred = pd.to_numeric(frame[f"predicted_q50_log10{suffix}"], errors="coerce").to_numpy(float)
    env_sd = pd.to_numeric(frame[f"future_observation_sd_log10_{arm}"], errors="coerce").to_numpy(float)
    if subset == "removed_paper_environmental":
        return env_pred, env_sd
    cond_pred = pd.to_numeric(
        frame[f"known_source_conditional_q50_log10{suffix}"], errors="coerce"
    ).to_numpy(float)
    cond_sd = pd.to_numeric(
        frame[f"known_source_conditional_future_sd_log10_{arm}"], errors="coerce"
    ).to_numpy(float)
    available = bool_values(frame["known_source_available"])
    conditional_finite = available & np.isfinite(cond_pred) & np.isfinite(cond_sd) & (cond_sd > 0)
    if subset == "retained_known_source":
        return cond_pred, cond_sd
    require(subset == "all_record_hybrid", f"Unknown prediction subset: {subset}")
    return np.where(conditional_finite, cond_pred, env_pred), np.where(conditional_finite, cond_sd, env_sd)


RUN_PATH_PATTERN = re.compile(
    r"round[_-]?(?P<round>\d+).*?removal[_-]?(?P<removal>\d+).*?fold[_-]?(?P<fold>\d+)",
    re.IGNORECASE,
)


def metadata_from_run_path(path: Path) -> tuple[int, int, int]:
    text = path.as_posix()
    match = RUN_PATH_PATTERN.search(text)
    require(match is not None, f"Cannot infer round/removal/fold from run path: {path}")
    return int(match["round"]), int(match["removal"]), int(match["fold"])


def find_filter_audits(task_dir: Path) -> list[Path]:
    candidates = list(task_dir.rglob("FILTER_AUDIT.json")) if task_dir.exists() else []
    return sorted(set(path.resolve() for path in candidates))


def audits_validate_task(audits: list[Path], domains: set[str]) -> tuple[bool, str]:
    if not audits:
        return False, "missing FILTER_AUDIT.json; training integration cannot be verified"
    seen: set[str] = set()
    for path in audits:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "passed":
            return False, f"filter audit not passed: {path}"
        if not value.get("heldout_rows_identical_and_order_preserved", False):
            return False, f"heldout preservation not verified: {path}"
        if int(value.get("removed_paper_overlap_after_filter", -1)) != 0:
            return False, f"removed paper leaked into train: {path}"
        if not value.get("external_records_preserved", False):
            return False, f"external preservation not verified: {path}"
        domain = str(value.get("domain", "all"))
        if domain == "all":
            seen.update(DOMAIN_ORDER)
        else:
            seen.add(domain)
    missing = domains - seen
    return (not missing, "passed" if not missing else f"missing domain audits: {sorted(missing)}")


def subset_mask(frame: pd.DataFrame, selected: set[str], subset: str) -> np.ndarray:
    paper_selected = frame["source_kind"].eq("paper") & frame["base_source_id"].isin(selected)
    if subset == "removed_paper_environmental":
        return paper_selected.to_numpy(bool)
    if subset == "all_record_hybrid":
        return np.ones(len(frame), dtype=bool)
    if subset == "retained_known_source":
        available = bool_values(frame["known_source_available"])
        return (~paper_selected.to_numpy(bool)) & available
    raise RuntimeError(f"Unknown subset: {subset}")


def metric_rows_for_pair(
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    selected: set[str],
    *,
    round_id: int,
    removal: int,
    fold: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for domain in (*DOMAIN_ORDER, "all_natural_pooled"):
        domain_mask = np.ones(len(ablation), dtype=bool) if domain == "all_natural_pooled" else ablation["domain"].eq(domain).to_numpy(bool)
        for subset in ("removed_paper_environmental", "all_record_hybrid", "retained_known_source"):
            run_subset = subset_mask(ablation, selected, subset) & domain_mask
            ids = ablation.loc[run_subset, "stable_record_id"]
            run_local = ablation.loc[run_subset].copy().reset_index(drop=True)
            base_local = baseline.set_index("stable_record_id", drop=False).loc[list(ids)].reset_index(drop=True)
            require(
                base_local["stable_record_id"].tolist() == run_local["stable_record_id"].tolist(),
                "Baseline/run paired record order mismatch",
            )
            for arm in ("aware", "unaware"):
                for weighting in ("source_equal", "record_equal"):
                    key = {
                        "random_round": int(round_id),
                        "removal_percent": int(removal),
                        "fold": int(fold),
                        "domain": domain,
                        "evaluation_subset": subset,
                        "uncertainty_arm": arm,
                        "weighting": weighting,
                    }
                    model_rows: dict[str, dict[str, object]] = {}
                    for model_name, local in (("baseline_0pct", base_local), ("ablation", run_local)):
                        pred, sd = prediction_arrays(local, subset, arm)
                        values = interval_metric_row(local, pred, sd, weighting)
                        available = bool_values(local["known_source_available"]) if len(local) else np.array([], dtype=bool)
                        if len(local):
                            w = source_equal_weights(local) if weighting == "source_equal" else record_equal_weights(local)
                            availability = float(np.average(available, weights=w))
                        else:
                            availability = math.nan
                        values["known_source_availability_rate"] = availability
                        model_rows[model_name] = values
                        metric_rows.append({**key, "model": model_name, **values})
                    base_values = model_rows["baseline_0pct"]
                    run_values = model_rows["ablation"]
                    wide: dict[str, object] = {**key}
                    for metric in (
                        "r2",
                        "rmse_log10",
                        "mae_log10",
                        "coverage95",
                        "wis_log10",
                        "width95_mean_log10",
                        "width95_median_log10",
                        "known_source_availability_rate",
                    ):
                        base_value = float(base_values.get(metric, math.nan))
                        run_value = float(run_values.get(metric, math.nan))
                        delta = run_value - base_value if np.isfinite(base_value) and np.isfinite(run_value) else math.nan
                        relative = (
                            100.0 * delta / abs(base_value)
                            if np.isfinite(delta) and abs(base_value) > 1e-12
                            else math.nan
                        )
                        wide[f"baseline_{metric}"] = base_value
                        wide[f"ablation_{metric}"] = run_value
                        wide[f"delta_{metric}"] = delta
                        wide[f"relative_delta_pct_{metric}"] = relative
                    wide["paired_records"] = int(len(run_local))
                    wide["baseline_status"] = base_values["status"]
                    wide["ablation_status"] = run_values["status"]
                    delta_rows.append(wide)
    return metric_rows, delta_rows


def write_pending_status(args: argparse.Namespace, run_matrix: pd.DataFrame) -> None:
    expected = int(len(run_matrix))
    complete = int(run_matrix["status"].eq("completed").sum())
    invalid = int(run_matrix["status"].eq("invalid").sum())
    write_json(
        METRICS_ROOT / "AGGREGATE_STATUS.json",
        {
            "status": "completed" if complete == expected and invalid == 0 else "pending",
            "expected_domain_fold_removal_tasks": expected,
            "completed_tasks": complete,
            "invalid_tasks": invalid,
            "pending_tasks": int(expected - complete - invalid),
            "message": (
                "No missing model result has been substituted. Metrics exist only for validated retrained OOF files."
            ),
        },
    )


def map_status(round_id: int, removals: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for removal in removals:
        for domain in DOMAIN_ORDER:
            expected = MAP_ROOT / f"round_{round_id:02d}" / f"removal_{removal}" / f"{domain}_H26T_ABLATION_GRID.csv.gz"
            rows.append(
                {
                    "random_round": int(round_id),
                    "removal_percent": int(removal),
                    "domain": domain,
                    "expected_grid": str(expected),
                    "status": "available_not_yet_summarized" if expected.exists() else "pending_model_grid",
                    "baseline_comparison_interface": "stable prediction_id + waterbody; q50 and width95 paired delta",
                }
            )
    result = pd.DataFrame(rows)
    write_csv(MAP_ROOT / "MAP_AGGREGATION_STATUS.csv", result)
    return result


def build_learning_curve(deltas: pd.DataFrame) -> pd.DataFrame:
    """Collapse paired fold deltas into the paper-volume learning-curve table."""
    retained_counts = {0: 1295, 30: 906, 50: 647, 80: 259}
    metrics = (
        "r2", "rmse_log10", "mae_log10", "coverage95", "wis_log10",
        "width95_mean_log10", "width95_median_log10", "known_source_availability_rate",
    )
    group_columns = (
        "removal_percent", "domain", "evaluation_subset", "uncertainty_arm", "weighting",
    )
    rows: list[dict[str, object]] = []
    for keys, local in deltas.groupby(list(group_columns), observed=True):
        key = dict(zip(group_columns, keys))
        removal = int(key["removal_percent"])
        for metric in metrics:
            baseline = pd.to_numeric(local[f"baseline_{metric}"], errors="coerce")
            ablated = pd.to_numeric(local[f"ablation_{metric}"], errors="coerce")
            delta = pd.to_numeric(local[f"delta_{metric}"], errors="coerce")
            finite = baseline.notna() & ablated.notna() & delta.notna()
            if not finite.any():
                continue
            rows.append({
                "random_round": int(local["random_round"].iloc[0]),
                **key,
                "papers_retained": retained_counts[removal],
                "papers_retained_fraction": retained_counts[removal] / EXPECTED_PAPERS,
                "metric": metric,
                "valid_folds": int(local.loc[finite, "fold"].nunique()),
                "baseline_fold_mean": float(baseline[finite].mean()),
                "ablation_fold_mean": float(ablated[finite].mean()),
                "paired_delta_fold_mean": float(delta[finite].mean()),
                "paired_delta_fold_median": float(delta[finite].median()),
                "paired_delta_fold_sd": float(delta[finite].std(ddof=1)) if int(finite.sum()) > 1 else math.nan,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        full = result.loc[
            result["evaluation_subset"].eq("all_record_hybrid")
            & result["removal_percent"].eq(result["removal_percent"].min())
        ].copy()
        full["removal_percent"] = 0
        full["papers_retained"] = retained_counts[0]
        full["papers_retained_fraction"] = 1.0
        full["ablation_fold_mean"] = full["baseline_fold_mean"]
        full["paired_delta_fold_mean"] = 0.0
        full["paired_delta_fold_median"] = 0.0
        full["paired_delta_fold_sd"] = 0.0
        result = pd.concat([full, result], ignore_index=True)
    return result


def aggregate(args: argparse.Namespace) -> int:
    baseline_all = canonicalize_run_oof(read_baseline_oof(args.baseline_oof))
    run_paths = sorted(args.runs_root.rglob("H26T_ABLATION_OOF*.csv*")) if args.runs_root.exists() else []
    grouped: dict[tuple[int, int, int], list[Path]] = {}
    path_errors: list[dict[str, str]] = []
    for path in run_paths:
        try:
            key = metadata_from_run_path(path)
        except Exception as exc:  # report malformed run products instead of hiding them
            path_errors.append({"path": str(path), "error": str(exc)})
            continue
        grouped.setdefault(key, []).append(path)

    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    training_size_rows: list[dict[str, object]] = []
    for removal in args.removals:
        for fold in args.folds:
            key = (int(args.round), int(removal), int(fold))
            paths = grouped.get(key, [])
            manifest = manifest_path(args.round, fold, removal)
            if not paths:
                for domain in DOMAIN_ORDER:
                    status_rows.append(
                        {
                            "random_round": args.round,
                            "removal_percent": removal,
                            "fold": fold,
                            "domain": domain,
                            "status": "pending",
                            "detail": "retrained OOF not found",
                            "run_files": "",
                        }
                    )
                continue
            try:
                require(manifest.exists(), f"Manifest missing: {manifest}")
                selected = set(read_removal_manifest(manifest)["paper_id_base"].astype(str))
                frames = [canonicalize_run_oof(pd.read_csv(path, low_memory=False)) for path in paths]
                run = pd.concat(frames, ignore_index=True)
                require(run["stable_record_id"].is_unique, "Run OOF duplicates stable_record_id")
                if "formal_fold" in run.columns:
                    fold_values = set(pd.to_numeric(run["formal_fold"], errors="raise").astype(int))
                    require(fold_values == {fold}, f"Run contains folds {sorted(fold_values)}, expected {fold}")
                expected = baseline_all.loc[baseline_all["formal_fold"].eq(fold)].copy()
                require(set(run["stable_record_id"]) == set(expected["stable_record_id"]), "Run OOF does not retain the complete heldout fold")
                run = expected[["stable_record_id"]].merge(run, on="stable_record_id", how="left", validate="one_to_one")
                # Resolve merge suffixes without allowing source/target identity drift.
                for identity in ("source_group", "observed_log10", "domain", "waterbody"):
                    left = f"{identity}_x"
                    right = f"{identity}_y"
                    if left in run.columns and right in run.columns:
                        require(run[left].astype(str).eq(run[right].astype(str)).all(), f"Run changed {identity}")
                        run[identity] = run[right]
                        run = run.drop(columns=[left, right])
                run = canonicalize_run_oof(run)
                expected = canonicalize_run_oof(expected)
                selected_validation = run["source_kind"].eq("paper") & run["base_source_id"].isin(selected)
                leaked_conditional = selected_validation.to_numpy(bool) & bool_values(run["known_source_available"])
                require(not leaked_conditional.any(), "Removed paper still has known-source conditional availability")

                task_dir = args.runs_root / f"round_{args.round:02d}" / f"removal_{removal}" / f"fold_{fold:02d}"
                audits = find_filter_audits(task_dir)
                valid_audit, audit_detail = audits_validate_task(audits, set(DOMAIN_ORDER))
                require(valid_audit, audit_detail)
                for audit_path in audits:
                    audit_value = json.loads(audit_path.read_text(encoding="utf-8"))
                    training_size_rows.append({
                        "random_round": args.round,
                        "removal_percent": removal,
                        "fold": fold,
                        "domain": audit_value.get("domain", "all"),
                        "outer_train_rows_before": audit_value.get("outer_train_rows_before"),
                        "outer_train_rows_after": audit_value.get("outer_train_rows_after"),
                        "outer_train_rows_removed": audit_value.get("outer_train_rows_removed"),
                        "selected_papers_in_manifest": audit_value.get("selected_papers_in_manifest"),
                        "selected_papers_present_in_outer_train": audit_value.get(
                            "selected_papers_present_in_outer_train"
                        ),
                        "external_train_rows_before": audit_value.get("external_train_rows_before"),
                        "external_train_rows_after": audit_value.get("external_train_rows_after"),
                        "external_records_preserved": audit_value.get("external_records_preserved"),
                        "audit_path": str(audit_path),
                    })
                local_metrics, local_deltas = metric_rows_for_pair(
                    expected,
                    run,
                    selected,
                    round_id=args.round,
                    removal=removal,
                    fold=fold,
                )
                metric_rows.extend(local_metrics)
                delta_rows.extend(local_deltas)
                for domain in DOMAIN_ORDER:
                    status_rows.append(
                        {
                            "random_round": args.round,
                            "removal_percent": removal,
                            "fold": fold,
                            "domain": domain,
                            "status": "completed",
                            "detail": "complete heldout IDs; no train leakage; external preserved",
                            "run_files": "|".join(str(path) for path in paths),
                        }
                    )
            except Exception as exc:
                for domain in DOMAIN_ORDER:
                    status_rows.append(
                        {
                            "random_round": args.round,
                            "removal_percent": removal,
                            "fold": fold,
                            "domain": domain,
                            "status": "invalid",
                            "detail": str(exc),
                            "run_files": "|".join(str(path) for path in paths),
                        }
                    )

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    run_matrix = pd.DataFrame(status_rows)
    write_csv(METRICS_ROOT / "RUN_MATRIX.csv", run_matrix)
    if training_size_rows:
        write_csv(METRICS_ROOT / "TRAINING_SIZE_BY_TASK.csv", pd.DataFrame(training_size_rows))
    if path_errors:
        write_csv(METRICS_ROOT / "UNPARSEABLE_RUN_FILES.csv", pd.DataFrame(path_errors))
    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    if not metrics.empty:
        write_csv(METRICS_ROOT / "ABLATION_METRICS_LONG.csv", metrics)
    if not deltas.empty:
        write_csv(METRICS_ROOT / "PAIRED_DELTAS.csv", deltas)
        learning_curve = build_learning_curve(deltas)
        write_csv(METRICS_ROOT / "PAPER_VOLUME_LEARNING_CURVE.csv", learning_curve)

    consistency_rows: list[dict[str, object]] = []
    if not deltas.empty:
        focus = deltas.loc[
            deltas["evaluation_subset"].eq("all_record_hybrid")
            & deltas["uncertainty_arm"].eq("aware")
            & deltas["weighting"].eq("source_equal")
        ].copy()
        for (removal, domain), local in focus.groupby(["removal_percent", "domain"], observed=True):
            finite = local.loc[np.isfinite(local["delta_wis_log10"])].copy()
            valid_folds = int(finite["fold"].nunique())
            worse = int(finite.loc[finite["delta_wis_log10"].gt(0), "fold"].nunique())
            complete = valid_folds == len(args.folds)
            consistency_rows.append(
                {
                    "random_round": args.round,
                    "removal_percent": int(removal),
                    "domain": domain,
                    "core_metric": "source_equal_all_record_hybrid_aware_WIS",
                    "valid_folds": valid_folds,
                    "folds_with_worse_wis": worse,
                    "direction_consistent_7_of_10": bool(complete and worse >= 7),
                    "needs_second_random_round": None if not complete else bool(worse < 7),
                    "status": "completed" if complete else "pending",
                }
            )
    consistency = pd.DataFrame(consistency_rows)
    if not consistency.empty:
        write_csv(METRICS_ROOT / "DIRECTION_CONSISTENCY.csv", consistency)
    complete_decisions = consistency.loc[consistency.get("status", pd.Series(dtype=str)).eq("completed")] if not consistency.empty else consistency
    second_round = None
    if not complete_decisions.empty and len(complete_decisions) == len(args.removals) * 4:
        second_round = bool(complete_decisions["needs_second_random_round"].fillna(False).any())
    write_json(
        METRICS_ROOT / "ROUND_DECISION.json",
        {
            "status": "completed" if second_round is not None else "pending",
            "needs_second_random_round": second_round,
            "rule": (
                "For each removal/domain, run round 2 only when fewer than 7 of 10 folds show worse "
                "source-equal all-record hybrid aware WIS. Missing folds remain pending."
            ),
        },
    )
    map_status(args.round, args.removals)
    write_pending_status(args, run_matrix)
    print(
        json.dumps(
            {
                "status": "completed" if run_matrix["status"].eq("completed").all() else "pending",
                "run_files_found": len(run_paths),
                "task_rows": int(len(run_matrix)),
                "completed_task_rows": int(run_matrix["status"].eq("completed").sum()),
                "invalid_task_rows": int(run_matrix["status"].eq("invalid").sum()),
                "output": str(METRICS_ROOT),
            },
            ensure_ascii=False,
        )
    )
    return 0


def status(args: argparse.Namespace) -> int:
    manifest_audit = manifest_dir(args.round) / "MANIFEST_AUDIT.json"
    aggregate_status = METRICS_ROOT / "AGGREGATE_STATUS.json"
    value = {
        "manifest": json.loads(manifest_audit.read_text(encoding="utf-8")) if manifest_audit.exists() else {"status": "pending"},
        "aggregate": json.loads(aggregate_status.read_text(encoding="utf-8")) if aggregate_status.exists() else {"status": "pending"},
        "maps": "available" if (MAP_ROOT / "MAP_AGGREGATION_STATUS.csv").exists() else "pending",
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-oof", type=Path, default=DEFAULT_OOF)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare", help="Build nested, stratified paper-removal manifests")
    add_common(prepare_parser)
    prepare_parser.add_argument("--training-table", type=Path, default=DEFAULT_TRAINING)
    prepare_parser.add_argument("--round", type=int, default=1)
    prepare_parser.add_argument("--seed", type=int, default=260827)
    prepare_parser.add_argument("--folds", type=int, nargs="+", default=list(EXPECTED_FOLDS), choices=EXPECTED_FOLDS)
    prepare_parser.add_argument("--removals", type=int, nargs="+", default=list(REMOVAL_COUNTS), choices=tuple(REMOVAL_COUNTS))
    prepare_parser.set_defaults(func=prepare)

    filter_parser = sub.add_parser(
        "filter-training",
        help="Filter only outer-train rows for one fold/removal; preserve heldout rows",
    )
    add_common(filter_parser)
    filter_parser.add_argument("--training-table", type=Path, default=DEFAULT_TRAINING)
    filter_parser.add_argument("--round", type=int, default=1)
    filter_parser.add_argument("--fold", type=int, required=True, choices=EXPECTED_FOLDS)
    filter_parser.add_argument("--removal", type=int, required=True, choices=tuple(REMOVAL_COUNTS))
    filter_parser.add_argument("--domain", choices=("all", *DOMAIN_ORDER), default="all")
    filter_parser.add_argument("--manifest", type=Path)
    filter_parser.add_argument("--output-dir", type=Path)
    filter_parser.add_argument("--materialize-records", action="store_true")
    filter_parser.set_defaults(func=filter_training)

    aggregate_parser = sub.add_parser("aggregate", help="Aggregate validated retrained OOF or emit pending status")
    add_common(aggregate_parser)
    aggregate_parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    aggregate_parser.add_argument("--round", type=int, default=1)
    aggregate_parser.add_argument("--folds", type=int, nargs="+", default=list(EXPECTED_FOLDS), choices=EXPECTED_FOLDS)
    aggregate_parser.add_argument("--removals", type=int, nargs="+", default=list(REMOVAL_COUNTS), choices=tuple(REMOVAL_COUNTS))
    aggregate_parser.set_defaults(func=aggregate)

    status_parser = sub.add_parser("status", help="Show manifest/model/map status")
    status_parser.add_argument("--round", type=int, default=1)
    status_parser.set_defaults(func=status)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
