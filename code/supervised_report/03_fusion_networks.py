"""Step 3: progressive multimodal fusion networks (ablation study).

Models (same architectures as the original notebook):
  * single-modality networks (T1, T2, ADC);
  * early fusion: modality encoders + concatenation;
  * adaptive global-weight fusion: one learnable softmax weight per modality;
  * adaptive gated fusion: patient-specific softmax weights from a gate net.
Each model is trained for 200 epochs (Adam, lr 1e-3, weight decay 1e-4,
batch 32) on the training split, for 10 random seeds. We report mean and SD of
the final-epoch train/test AUC and accuracy, so the numbers are not a
best-of-runs selection, and summarize the learned modality weights.
Inputs are the harmonized, feature-selected matrices from step 2.
"""

import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from common import CACHE, MODALITIES, PRIMARY, TAB, PALETTE, plt, savefig

warnings.filterwarnings("ignore")
SEEDS = list(range(507, 517))
EPOCHS = 200
SETTINGS_NN = ["Original (raw z-score)", PRIMARY]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Encoder(nn.Module):
    def __init__(self, d, h=8, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def head(d_in, h=8, dropout=0.4):
    return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(h, 1))


class SingleNet(nn.Module):
    def __init__(self, dims, modality):
        super().__init__()
        self.i = MODALITIES.index(modality)
        self.enc = Encoder(dims[self.i])
        self.clf = head(8)

    def forward(self, xs):
        return self.clf(self.enc(xs[self.i])).squeeze(1), None


class EarlyFusionNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encs = nn.ModuleList([Encoder(d) for d in dims])
        self.clf = head(8 * 3)

    def forward(self, xs):
        h = torch.cat([e(x) for e, x in zip(self.encs, xs)], 1)
        return self.clf(h).squeeze(1), None


class GlobalWeightNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encs = nn.ModuleList([Encoder(d) for d in dims])
        self.logits = nn.Parameter(torch.zeros(3))
        self.clf = head(8)

    def forward(self, xs):
        w = torch.softmax(self.logits, 0)
        h = sum(w[i] * e(x) for i, (e, x) in enumerate(zip(self.encs, xs)))
        return self.clf(h).squeeze(1), w.expand(xs[0].shape[0], 3)


class GatedFusionNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encs = nn.ModuleList([Encoder(d, h=16) for d in dims])
        self.gate = nn.Sequential(nn.Linear(16 * 3, 8), nn.ReLU(),
                                  nn.Linear(8, 3))
        self.clf = head(16)

    def forward(self, xs):
        hs = [e(x) for e, x in zip(self.encs, xs)]
        g = torch.softmax(self.gate(torch.cat(hs, 1)), 1)
        h = sum(g[:, i:i + 1] * hs[i] for i in range(3))
        return self.clf(h).squeeze(1), g


MODELS = {"Single (T1)": lambda d: SingleNet(d, "T1"),
          "Single (T2)": lambda d: SingleNet(d, "T2"),
          "Single (ADC)": lambda d: SingleNet(d, "ADC"),
          "Early fusion": EarlyFusionNet,
          "Adaptive global weights": GlobalWeightNet,
          "Adaptive gated fusion": GatedFusionNet}


def tensors(prepared, part):
    i = 0 if part == "train" else 1
    return [torch.tensor(prepared[m][i].values, dtype=torch.float32)
            for m in MODALITIES]


def evaluate(model, xs, y):
    model.eval()
    with torch.no_grad():
        logit, w = model(xs)
        p = torch.sigmoid(logit).numpy()
    return (roc_auc_score(y, p), accuracy_score(y, p >= 0.5),
            None if w is None else w.numpy())


def train(model, xs, y, seed):
    set_seed(seed)
    ds = TensorDataset(*xs, torch.tensor(y, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=32, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train()
        for *xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb)[0], yb).backward()
            opt.step()
    return model


def main():
    with open(CACHE / "baselines.pkl", "rb") as f:
        b = pickle.load(f)
    from common import load_cohort
    _, ids = load_cohort()
    ytr = ids.loc[b["train_idx"], "LVI"].values
    yte = ids.loc[b["test_idx"], "LVI"].values

    rows, gates = [], []
    for setting in SETTINGS_NN:
        prepared = b["store"][setting]["prepared"]
        xtr, xte = tensors(prepared, "train"), tensors(prepared, "test")
        dims = [x.shape[1] for x in xtr]
        for name, ctor in MODELS.items():
            for seed in SEEDS:
                set_seed(seed)
                model = train(ctor(dims), xtr, ytr, seed)
                tr_auc, tr_acc, _ = evaluate(model, xtr, ytr)
                te_auc, te_acc, w = evaluate(model, xte, yte)
                rows.append({"setting": setting, "model": name, "seed": seed,
                             "train_auc": tr_auc, "train_acc": tr_acc,
                             "test_auc": te_auc, "test_acc": te_acc})
                if w is not None:
                    _, _, wtr = evaluate(model, xtr, ytr)
                    gates.append({"setting": setting, "model": name,
                                  "seed": seed,
                                  **{f"mean_w_{m}": wtr[:, i].mean()
                                     for i, m in enumerate(MODALITIES)},
                                  **{f"sd_w_{m}": wtr[:, i].std()
                                     for i, m in enumerate(MODALITIES)}})
            r = pd.DataFrame(rows)
            r = r[(r.setting == setting) & (r.model == name)]
            print(setting, name, r[["train_auc", "test_auc", "test_acc"]]
                  .agg(["mean", "std"]).round(3).to_dict())

    runs = pd.DataFrame(rows)
    runs.to_csv(TAB / "fusion_runs.csv", index=False)
    summ = runs.groupby(["setting", "model"], sort=False).agg(
        train_auc_mean=("train_auc", "mean"), train_auc_sd=("train_auc", "std"),
        train_acc_mean=("train_acc", "mean"),
        test_auc_mean=("test_auc", "mean"), test_auc_sd=("test_auc", "std"),
        test_acc_mean=("test_acc", "mean"), test_acc_sd=("test_acc", "std"),
        test_auc_max=("test_auc", "max")).reset_index()
    summ.to_csv(TAB / "fusion_summary.csv", index=False)
    g = pd.DataFrame(gates)
    g.to_csv(TAB / "fusion_gates.csv", index=False)
    print(summ.round(3).to_string())
    print(g.round(3).to_string())

    gp = g[(g.setting == PRIMARY) & (g.model == "Adaptive gated fusion")]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for i, m in enumerate(MODALITIES):
        ax.scatter(np.arange(len(gp)) + (i - 1) * 0.2, gp[f"mean_w_{m}"],
                   color=PALETTE[i], label=m, s=18)
        ax.errorbar(np.arange(len(gp)) + (i - 1) * 0.2, gp[f"mean_w_{m}"],
                    yerr=gp[f"sd_w_{m}"], fmt="none", color=PALETTE[i],
                    capsize=1.5, lw=0.8)
    ax.set_xticks(range(len(gp)), [str(s) for s in gp.seed], fontsize=7)
    ax.set_xlabel("Seed"); ax.set_ylabel("Mean gate weight (± SD over patients)")
    ax.set_title("Adaptive gated fusion: learned modality weights")
    ax.legend(fontsize=7, ncol=3)
    savefig(fig, "fusion_gate_weights")


if __name__ == "__main__":
    main()
