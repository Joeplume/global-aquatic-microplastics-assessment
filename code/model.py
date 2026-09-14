import numpy as np
import torch
from torch import nn


class BayesianLinear(nn.Module):
    def __init__(self, n_in, n_out, prior_scale):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(n_out, n_in))
        self.weight_rho = nn.Parameter(torch.full((n_out, n_in), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(n_out))
        self.bias_rho = nn.Parameter(torch.full((n_out,), -3.0))
        self.prior_scale = prior_scale

    def forward(self, x, sample=True):
        ws = torch.nn.functional.softplus(self.weight_rho)
        bs = torch.nn.functional.softplus(self.bias_rho)
        w = self.weight_mu + ws * torch.randn_like(ws) if sample else self.weight_mu
        b = self.bias_mu + bs * torch.randn_like(bs) if sample else self.bias_mu
        return torch.nn.functional.linear(x, w, b)

    def kl(self):
        terms = []
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sd = torch.nn.functional.softplus(rho)
            terms.append((torch.log(self.prior_scale / sd) + (sd.square() + mu.square()) / (2 * self.prior_scale ** 2) - 0.5).sum())
        return sum(terms)


class Branch(nn.Module):
    def __init__(self, n_features, widths, prior_scale):
        super().__init__()
        dims = [n_features, *widths, 1]
        self.layers = nn.ModuleList([BayesianLinear(a, b, prior_scale) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x, sample=True):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.silu(layer(x, sample))
        return self.layers[-1](x, sample).squeeze(-1)

    def kl(self):
        return sum(layer.kl() for layer in self.layers)


class WaterbodyBNNMEM(nn.Module):
    def __init__(self, n_features, domains, widths, prior_scale):
        super().__init__()
        self.domains = list(domains)
        self.branches = nn.ModuleDict({d: Branch(n_features, widths, prior_scale) for d in self.domains})
        self.source_effect = nn.Embedding(1, 1)
        self.method_effect = nn.Embedding(1, 1)
        self.log_sigma = nn.Parameter(torch.full((len(self.domains),), -0.2))

    def resize_effects(self, n_sources, n_methods):
        self.source_effect = nn.Embedding(max(n_sources, 1), 1)
        self.method_effect = nn.Embedding(max(n_methods, 1), 1)
        nn.init.zeros_(self.source_effect.weight)
        nn.init.zeros_(self.method_effect.weight)

    def forward(self, x, domain, source, method, sample=True, conditional=True):
        mean = torch.zeros(x.shape[0], device=x.device)
        for name in self.domains:
            idx = domain.eq(self.domains.index(name))
            if idx.any():
                mean[idx] = self.branches[name](x[idx], sample)
        if conditional:
            mean = mean + self.source_effect(source).squeeze(-1) + self.method_effect(method).squeeze(-1)
        return mean

    def kl(self):
        random_effect_penalty = 0.5 * (self.source_effect.weight.square().sum() + self.method_effect.weight.square().sum())
        return sum(branch.kl() for branch in self.branches.values()) + random_effect_penalty


def encode(values):
    labels = sorted({str(x) for x in values})
    mapping = {x: i for i, x in enumerate(labels)}
    return np.array([mapping[str(x)] for x in values], dtype=np.int64), mapping


def fit_model(x, y, domains, sources, methods, settings):
    torch.manual_seed(int(settings.get("seed", 17)))
    np.random.seed(int(settings.get("seed", 17)))
    domain_names = list(settings["domains"])
    d = np.array([domain_names.index(str(v)) for v in domains], dtype=np.int64)
    s, source_map = encode(sources)
    m, method_map = encode(methods)
    model = WaterbodyBNNMEM(x.shape[1], domain_names, list(settings.get("hidden_widths", [32, 16])), float(settings.get("prior_scale", 1.0)))
    model.resize_effects(len(source_map), len(method_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings.get("learning_rate", 0.01)))
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    td = torch.tensor(d)
    ts = torch.tensor(s)
    tm = torch.tensor(m)
    n = max(len(y), 1)
    for _ in range(int(settings.get("epochs", 120))):
        optimizer.zero_grad()
        pred = model(tx, td, ts, tm, True, True)
        sigma = torch.nn.functional.softplus(model.log_sigma[td]) + 1e-4
        loss = torch.distributions.Normal(pred, sigma).log_prob(ty).mean().neg() + model.kl() / n
        loss.backward()
        optimizer.step()
    return model, {"domain": {k: i for i, k in enumerate(domain_names)}, "source": source_map, "method": method_map}
