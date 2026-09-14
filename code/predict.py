import numpy as np
import pandas as pd
import torch


def posterior_predict(model, mappings, x, domains, sources, methods, draws=200, conditional=False):
    d = np.array([mappings["domain"].get(str(v), 0) for v in domains], dtype=np.int64)
    s = np.array([mappings["source"].get(str(v), 0) for v in sources], dtype=np.int64)
    m = np.array([mappings["method"].get(str(v), 0) for v in methods], dtype=np.int64)
    tx = torch.tensor(x, dtype=torch.float32)
    td = torch.tensor(d)
    ts = torch.tensor(s)
    tm = torch.tensor(m)
    values = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(draws)):
            values.append(model(tx, td, ts, tm, True, conditional).cpu().numpy())
    q = np.quantile(np.stack(values), [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame({"q025_log10_items_m3": q[0], "q50_log10_items_m3": q[1], "q975_log10_items_m3": q[2], "q50_items_m3": np.power(10.0, q[1])})
