from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 260904


def unique_glob(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one match for {pattern!r}; found {len(hits)}: {hits}")
    return hits[0]


def bipartite_two_core(edges: pd.DataFrame) -> pd.DataFrame:
    core = edges[["source_group", "match_stratum"]].drop_duplicates().copy()
    while len(core):
        source_degree = core.groupby("source_group")["match_stratum"].nunique()
        stratum_degree = core.groupby("match_stratum")["source_group"].nunique()
        reduced = core[
            core["source_group"].isin(source_degree[source_degree >= 2].index)
            & core["match_stratum"].isin(stratum_degree[stratum_degree >= 2].index)
        ]
        if len(reduced) == len(core):
            return reduced
        core = reduced
    return core


def load_data() -> pd.DataFrame:
    cohort_path = unique_glob(
        "00_H26*/07_*/06_H26*20260902/00_*/H26_FROZEN_24129_COHORT.csv.gz"
    )
    oof_path = unique_glob("00_H26*/03_*/grouped/H26_FINAL10_OOF.csv.gz")
    training_path = unique_glob("00_H26*/02_*/observations/*.csv.gz")
    cohort = pd.read_csv(cohort_path)
    oof = pd.read_csv(
        oof_path,
        usecols=[
            "stable_record_id",
            "predicted_q50_log10",
            "residual_log10",
            "domain",
        ],
    )
    training = pd.read_csv(
        training_path,
        usecols=[
            "stable_record_id",
            "detection_explained_offset_log10",
            "qaqc_explained_offset_log10",
            "size_window_explained_offset_log10",
            "year_explained_offset_log10",
            "national_context_explained_offset_log10",
            "sizecorr_evidence_tier",
            "sizecorr_lower_used_um",
        ],
    )
    data = cohort.merge(oof, on="stable_record_id", validate="one_to_one")
    data = data.merge(training, on="stable_record_id", how="left", validate="one_to_one")
    if len(data) != 24_129:
        raise RuntimeError(f"Expected 24,129 rows; found {len(data):,}")
    data["known_measurement_var"] = (
        data["sizecorr_known_sd_log10"].fillna(0.0).pow(2)
        + data["unit_sd_log10"].fillna(0.0).pow(2)
    )
    data["detection_method"] = data["detection_method"].fillna("__MISSING_METHOD__")
    return data


def make_core(data: pd.DataFrame, grid_deg: float, year_bin_width: int) -> pd.DataFrame:
    work = data.copy()
    lon_bin = np.floor((work["longitude"] + 180.0) / grid_deg).astype("Int64")
    lat_bin = np.floor((work["latitude"] + 90.0) / grid_deg).astype("Int64")
    year_bin = (
        np.floor(work["sampling_year_effective_v1_3"] / year_bin_width) * year_bin_width
    ).astype("Int64")
    work["match_stratum"] = (
        work["domain"].astype(str)
        + "|"
        + lon_bin.astype(str)
        + "|"
        + lat_bin.astype(str)
        + "|"
        + year_bin.astype(str)
    )
    core_edges = bipartite_two_core(work)
    core_index = pd.MultiIndex.from_frame(core_edges)
    work_index = pd.MultiIndex.from_frame(work[["source_group", "match_stratum"]])
    core = work[work_index.isin(core_index)].copy()
    core["all_group"] = "all"
    return core


def fit_variance_components(core: pd.DataFrame) -> dict[str, object]:
    # A crossed random-intercept model is fit only on the bipartite 2-core.
    # Method is included as a separate component; its near nesting with source is audited below.
    formula = "residual_log10 ~ 1"
    vc_formula = {
        "geotime": "0 + C(match_stratum)",
        "method": "0 + C(detection_method)",
        "source": "0 + C(source_group)",
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sm.MixedLM.from_formula(
            formula,
            groups="all_group",
            re_formula="0",
            vc_formula=vc_formula,
            data=core,
        )
        result = model.fit(reml=True, method="lbfgs", maxiter=1000, disp=False)
    names = list(result.model.exog_vc.names)
    components = {name: float(value) for name, value in zip(names, result.vcomp)}
    raw_bse_re = np.asarray(result._results.bse_re)
    component_se = {name: float(value) for name, value in zip(names, raw_bse_re)}
    components["remaining_residual"] = float(result.scale)
    known_measurement = float(core["known_measurement_var"].mean())
    known_from_residual = min(known_measurement, components["remaining_residual"])
    components["known_measurement_raw_mean"] = known_measurement
    components["known_measurement"] = known_from_residual
    components["remaining_after_known_measurement"] = max(
        components["remaining_residual"] - known_from_residual, 0.0
    )
    denominator = (
        components.get("geotime", 0.0)
        + components.get("method", 0.0)
        + components.get("source", 0.0)
        + components["remaining_residual"]
    )
    shares = {
        name: value / denominator
        for name, value in components.items()
        if name not in {"remaining_residual", "known_measurement_raw_mean"}
    }
    source_method_counts = core.groupby("source_group")["detection_method"].nunique()
    method_source_counts = core.groupby("detection_method")["source_group"].nunique()
    return {
        "converged": bool(result.converged),
        "n_rows": int(len(core)),
        "n_sources": int(core["source_group"].nunique()),
        "n_geotime_strata": int(core["match_stratum"].nunique()),
        "n_methods": int(core["detection_method"].nunique()),
        "sources_with_gt1_method": int((source_method_counts > 1).sum()),
        "methods_shared_by_gt1_source": int((method_source_counts > 1).sum()),
        "rows_by_domain": {
            str(key): int(value) for key, value in core["domain"].value_counts().items()
        },
        "sources_by_domain": {
            str(key): int(value)
            for key, value in core.groupby("domain")["source_group"].nunique().items()
        },
        "variance_components": components,
        "variance_component_se": component_se,
        "shares": shares,
        "denominator_variance": denominator,
        "known_measurement_capped_by_residual_component": bool(
            known_measurement > components["remaining_residual"]
        ),
        "log_likelihood": float(result.llf),
        "warnings": [str(item.message) for item in caught],
    }


def main() -> None:
    np.random.seed(RANDOM_SEED)
    data = load_data()
    settings = [
        (0.25, 1, "strict_0p25deg_exact_year"),
        (0.50, 1, "strict_0p5deg_exact_year"),
        (0.50, 3, "primary_0p5deg_3year"),
        (1.00, 5, "sensitivity_1deg_5year"),
    ]
    results: list[dict[str, object]] = []
    for grid_deg, year_width, label in settings:
        core = make_core(data, grid_deg, year_width)
        print(
            f"Fitting {label}: rows={len(core):,}, "
            f"sources={core['source_group'].nunique():,}, "
            f"strata={core['match_stratum'].nunique():,}"
        )
        fit = fit_variance_components(core)
        fit.update(
            {
                "label": label,
                "grid_deg": grid_deg,
                "year_bin_width": year_width,
            }
        )
        results.append(fit)
        print(json.dumps(fit, ensure_ascii=False, indent=2))
    with (OUT_DIR / "CONSERVATIVE_VARIANCE_DECOMPOSITION.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    flat_rows = []
    for item in results:
        row = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "variance_components",
                "shares",
                "warnings",
                "variance_component_se",
                "rows_by_domain",
                "sources_by_domain",
            }
        }
        for key, value in item["variance_components"].items():
            row[f"variance_{key}"] = value
        for key, value in item["variance_component_se"].items():
            row[f"variance_se_{key}"] = value
        for key, value in item["shares"].items():
            row[f"share_{key}"] = value
        row["warnings"] = " | ".join(item["warnings"])
        row["rows_by_domain"] = json.dumps(item["rows_by_domain"], ensure_ascii=False)
        row["sources_by_domain"] = json.dumps(item["sources_by_domain"], ensure_ascii=False)
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(
        OUT_DIR / "CONSERVATIVE_VARIANCE_DECOMPOSITION.csv", index=False
    )


if __name__ == "__main__":
    main()
