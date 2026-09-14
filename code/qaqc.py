import json
import numpy as np
import pandas as pd


CRITERIA = tuple("ABCDEFG")


def parse_evidence(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, list):
        return {str(item.get("criterion_id", "")).upper(): item.get("score") for item in parsed if isinstance(item, dict)}
    if isinstance(parsed, dict):
        return {str(key).upper(): item for key, item in parsed.items()}
    return {}


def evidence_score(row, columns):
    scores = []
    evidence = parse_evidence(row.get("qaqc_evidence_json"))
    for criterion in CRITERIA:
        value = row.get(columns.get(criterion, f"qaqc_{criterion}"), evidence.get(criterion))
        if str(value).upper() == "NA":
            continue
        numeric = pd.to_numeric(value, errors="coerce")
        if np.isfinite(numeric):
            scores.append(float(np.clip(numeric, 0, 2)) / 2.0)
    return float(np.mean(scores)) if scores else np.nan


def apply_qaqc(data, rules):
    out = data.copy()
    fields = [name for name in rules.get("required_fields", []) if name in out]
    present = out[fields].notna() & ~out[fields].astype(str).isin(["", "not_applicable", "NA"]) if fields else pd.DataFrame(index=out.index)
    out["qaqc_completeness"] = present.mean(axis=1) if fields else 1.0
    columns = rules.get("criterion_columns", {})
    evidence = out.apply(evidence_score, axis=1, columns=columns)
    reported = pd.to_numeric(out["qaqc_score"], errors="coerce") if "qaqc_score" in out else pd.Series(np.nan, index=out.index)
    if reported.notna().any():
        denominator = max(float(reported.max()), 1.0)
        reported = reported.clip(lower=0, upper=denominator) / denominator
    out["qaqc_evidence_score"] = evidence.where(evidence.notna(), reported)
    out["qaqc_evidence_score"] = out["qaqc_evidence_score"].where(out["qaqc_evidence_score"].notna(), out["qaqc_completeness"])
    longitude = pd.to_numeric(out["longitude"], errors="coerce") if "longitude" in out else pd.Series(np.nan, index=out.index)
    latitude = pd.to_numeric(out["latitude"], errors="coerce") if "latitude" in out else pd.Series(np.nan, index=out.index)
    coordinates = longitude.between(-180, 180) & latitude.between(-90, 90)
    out["qaqc_matched"] = out["qaqc_completeness"].ge(float(rules.get("minimum_completeness", 0.8))) & out["qaqc_evidence_score"].ge(float(rules.get("minimum_score", 0.5))) & coordinates
    return out
