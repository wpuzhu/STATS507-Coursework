"""Shared paths, constants and helpers for the Unsupervised Extension pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 507
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "multi_omics"
OUT = ROOT / "results" / "unsupervised_extension"
FIG = OUT / "figures"
TAB = OUT / "tables"
CACHE = OUT / "cache"
for d in (FIG, TAB, CACHE):
    d.mkdir(parents=True, exist_ok=True)

MODALITIES = ["T1", "T2", "ADC"]
ID_COLS = ["case", "center", "Label"]

# Colour-blind-safe categorical palette (Okabe-Ito)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
           "#D55E00", "#F0E442", "#000000"]
CENTER_COLORS = {1: "#0072B2", 2: "#E69F00"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def load_modalities():
    """Return {modality: DataFrame indexed by case} and the id table."""
    views = {}
    for m in MODALITIES:
        df = pd.read_excel(DATA / f"{m}.xlsx").set_index("case").sort_index()
        views[m] = df
    ids = views["T1"][["center", "Label"]].copy()
    for m in MODALITIES:
        assert views[m].index.equals(ids.index)
        assert (views[m]["center"] == ids["center"]).all()
        views[m] = views[m].drop(columns=["center", "Label"])
    return views, ids


def bh(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q
