import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


def aoa_status(train_x, target_x, quantiles=(0.9, 0.98)):
    model = LedoitWolf().fit(train_x)
    centre = model.location_
    precision = model.precision_
    train_d = np.sqrt(np.einsum("ij,jk,ik->i", train_x - centre, precision, train_x - centre))
    target_d = np.sqrt(np.einsum("ij,jk,ik->i", target_x - centre, precision, target_x - centre))
    a, b = np.quantile(train_d, quantiles)
    return np.select([target_d <= a, target_d <= b], ["in", "limited"], default="out"), target_d


def caoa_status(train_x, target_x, sources, quantiles=(0.9, 0.98)):
    centres = pd.DataFrame(train_x).assign(source=pd.Series(sources).astype(str).to_numpy()).groupby("source").mean().to_numpy()
    neighbours = NearestNeighbors(n_neighbors=min(2, len(centres))).fit(centres)
    train_d = neighbours.kneighbors(train_x)[0][:, -1]
    target_d = neighbours.kneighbors(target_x)[0][:, -1]
    a, b = np.quantile(train_d, quantiles)
    return np.select([target_d <= a, target_d <= b], ["in", "limited"], default="out"), target_d


def annotate_support(train_x, target_x, sources):
    aoa, ad = aoa_status(train_x, target_x)
    caoa, cd = caoa_status(train_x, target_x, sources)
    return pd.DataFrame({"aoa_status": aoa, "aoa_distance": ad, "caoa_status": caoa, "caoa_distance": cd})
