import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from code.harmonize import harmonize_units
from code.model import fit_model
from code.monitoring import monitoring_analysis
from code.particle_size import shared_alpha_correction
from code.predict import posterior_predict
from code.qaqc import apply_qaqc
from code.support import annotate_support
from code.validation import strict_splits, summarize_predictions


def load_frame(path):
    return pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path, low_memory=False)


def matrix(frame, features, scaler=None):
    x = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).fillna(0).to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(x)
    return scaler.transform(x), scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--grid")
    parser.add_argument("--config", default="configs/pipeline.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_frame(args.data)
    data = apply_qaqc(data, settings["qaqc"])
    data = data.loc[data["qaqc_matched"]].reset_index(drop=True)
    data = harmonize_units(data)
    data = shared_alpha_correction(data, **settings["particle_size"])
    response = pd.to_numeric(data["concentration_size_corrected_items_m3"], errors="coerce")
    data = data.loc[np.isfinite(response) & response.gt(0)].reset_index(drop=True)
    features = [x for x in settings["features"] if x in data]
    x, scaler = matrix(data, features)
    y = np.log10(pd.to_numeric(data["concentration_size_corrected_items_m3"], errors="coerce")).to_numpy()
    domains = data["waterbody_group"].astype(str).to_numpy()
    sources = data["source_doi"].astype(str).to_numpy()
    methods = data["detection_method"].astype(str).to_numpy()
    model, mappings = fit_model(x, y, domains, sources, methods, settings["model"])
    validation_rows = {}
    for mode in settings["validation"]["modes"]:
        rows = []
        for fold, (train, test) in enumerate(strict_splits(data, mode, settings["validation"]["folds"], settings["model"]["seed"]), 1):
            fold_model, fold_mappings = fit_model(x[train], y[train], domains[train], sources[train], methods[train], settings["model"])
            pred = posterior_predict(fold_model, fold_mappings, x[test], domains[test], sources[test], methods[test], settings["prediction"]["draws"], False)
            rows.append(pd.DataFrame({"fold": fold, "observed_log10": y[test], "predicted_log10": pred["q50_log10_items_m3"]}))
        validation_rows[mode] = pd.concat(rows, ignore_index=True)
    summarize_predictions(validation_rows).to_csv(output / "strict_transfer_metrics.csv", index=False)
    target = load_frame(args.grid) if args.grid else data.copy()
    target_x, _ = matrix(target, features, scaler)
    default_domain = settings["model"]["domains"][0]
    target_domains = target.get("waterbody_group", target.get("waterbody", pd.Series([default_domain] * len(target)))).astype(str).to_numpy()
    target_sources = target.get("source_doi", pd.Series(["unseen"] * len(target))).astype(str).to_numpy()
    target_methods = target.get("detection_method", pd.Series(["standardized"] * len(target))).astype(str).to_numpy()
    prediction = posterior_predict(model, mappings, target_x, target_domains, target_sources, target_methods, settings["prediction"]["draws"], False)
    support = annotate_support(x, target_x, sources)
    result = pd.concat([target.reset_index(drop=True), prediction, support], axis=1)
    result.to_csv(output / "global_predictions.csv.gz", index=False, compression="gzip")
    monitoring_analysis(prediction, settings["monitoring"]).to_csv(output / "mdc_monitoring_analysis.csv", index=False)
    data.to_csv(output / "harmonized_analysis_data.csv", index=False)


if __name__ == "__main__":
    main()
