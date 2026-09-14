import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


def strict_splits(data, mode, folds, seed):
    index = np.arange(len(data))
    if mode == "random":
        return KFold(folds, shuffle=True, random_state=seed).split(index)
    if mode == "spatial":
        groups = data["longitude"].floordiv(10).astype(str) + ":" + data["latitude"].floordiv(10).astype(str)
    elif mode == "source":
        groups = data["source_doi"].astype(str)
    else:
        groups = data["source_doi"].astype(str) + ":" + data["longitude"].floordiv(10).astype(str) + ":" + data["latitude"].floordiv(10).astype(str)
    n = min(folds, groups.nunique())
    return GroupKFold(max(n, 2)).split(index, groups=groups)


def summarize_predictions(rows):
    out = []
    for mode, frame in rows.items():
        y = frame["observed_log10"].to_numpy()
        p = frame["predicted_log10"].to_numpy()
        out.append({"validation": mode, "records": len(frame), "r2": r2_score(y, p), "rmse": mean_squared_error(y, p) ** 0.5, "mae": mean_absolute_error(y, p)})
    return pd.DataFrame(out)
