# viz_injection_vs_interruption_hists.py
#
# Usage:
# python viz_injection_vs_interruption_hists.py \
#   --alu train_logs/alu.jsonl \
#   --memscan train_logs/memscan.jsonl \
#   --interr train_logs/interr.jsonl \
#   --outdir viz_attacktype_hists
#
# What it does:
# 1) Loads windows from ALU + MEMSCAN (INJECTION) and INTERR (INTERRUPTION)
# 2) Builds the same engineered features you use in retrain_attacktype.py
# 3) Saves:
#    - overlaid histograms (one figure per feature)
#    - a 2D scatter of two strong features
#    - an optional "linear separability" plot: LR decision boundary on those 2 features

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
DEVICE_COL = "device_id"

RAW = ["dC", "dL", "dP", "dE", "dF", "dS", "dT", "cyc_per_us"]
ENGINEERED = (
    ["lsu_per_cyc", "cpi_per_cyc", "exc_per_cyc", "fold_per_cyc"]
    + ["lsu_per_us2", "cpi_per_us2", "exc_per_us2", "cyc_per_lsu2"]
    + [f"log1p_{c}" for c in ["dC", "dL", "dP", "dE", "dF", "dS", "dT"]]
)
FEATURES = RAW + ENGINEERED

# A small default set that tends to separate “injection vs interruption”
DEFAULT_PLOT_FEATURES = [
    "exc_per_us2",     # interruptions usually spike exceptions/time
    "exc_per_cyc",     # exceptions normalized by compute
    "lsu_per_cyc",     # injections often push compute/mem ratios differently
    "cyc_per_us",      # overall intensity
    "cpi_per_cyc",     # stalls / pipeline effects
    "log1p_dE",        # exceptions in log space
]


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def ensure_device_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DEVICE_COL not in out.columns:
        if "device_id_str" in out.columns:
            out[DEVICE_COL] = out["device_id_str"].astype(str).str.extract(r"(\d+)$")[0]
        else:
            out[DEVICE_COL] = 0
    out[DEVICE_COL] = pd.to_numeric(out[DEVICE_COL], errors="coerce")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in RAW + ["window_id", DEVICE_COL]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["lsu_per_cyc"]   = out["dL"] / (out["dC"] + 1.0)
    out["cpi_per_cyc"]   = out["dP"] / (out["dC"] + 1.0)
    out["exc_per_cyc"]   = out["dE"] / (out["dC"] + 1.0)
    out["fold_per_cyc"]  = out["dF"] / (out["dC"] + 1.0)

    out["lsu_per_us2"]   = out["dL"] / (out["dT"] + EPS)
    out["cpi_per_us2"]   = out["dP"] / (out["dT"] + EPS)
    out["exc_per_us2"]   = out["dE"] / (out["dT"] + EPS)

    out["cyc_per_lsu2"]  = out["dC"] / (out["dL"] + 1.0)

    for c in ["dC", "dL", "dP", "dE", "dF", "dS", "dT"]:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def save_hist_overlay(x0: np.ndarray, x1: np.ndarray, name: str, outpath: Path, bins=60):
    # robust range (ignore crazy tails)
    lo = np.nanpercentile(np.concatenate([x0, x1]), 1)
    hi = np.nanpercentile(np.concatenate([x0, x1]), 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    plt.figure()
    plt.hist(x0, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="INJECTION (ALU+MEMSCAN)")
    plt.hist(x1, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="INTERRUPTION (INTERR)")
    plt.title(f"Feature histogram: {name}")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_2d_scatter_and_lr(df: pd.DataFrame, f1: str, f2: str, outdir: Path):
    sub = df[[f1, f2, "attack_label"]].dropna().copy()
    if len(sub) < 50:
        return

    X = sub[[f1, f2]].to_numpy(np.float32)
    y = sub["attack_label"].to_numpy(np.int64)

    # Scatter
    plt.figure()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, alpha=0.6, label="INJECTION")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, alpha=0.6, label="INTERRUPTION")
    plt.title(f"2D scatter: {f1} vs {f2}")
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{f1}__{f2}.png", dpi=220)
    plt.close()

    # LR decision boundary on standardized features (to show linear separability)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    lr.fit(Xs, y)

    # grid for boundary
    x0min, x0max = np.percentile(Xs[:, 0], [1, 99])
    x1min, x1max = np.percentile(Xs[:, 1], [1, 99])
    gx0 = np.linspace(x0min, x0max, 250)
    gx1 = np.linspace(x1min, x1max, 250)
    xx, yy = np.meshgrid(gx0, gx1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = lr.predict_proba(grid)[:, 1].reshape(xx.shape)  # P(INTERRUPTION)

    plt.figure()
    # plot points in standardized space
    plt.scatter(Xs[y == 0, 0], Xs[y == 0, 1], s=10, alpha=0.55, label="INJECTION")
    plt.scatter(Xs[y == 1, 0], Xs[y == 1, 1], s=10, alpha=0.55, label="INTERRUPTION")
    # 0.5 contour = linear decision boundary
    cs = plt.contour(xx, yy, zz, levels=[0.5])
    plt.title(f"Linear boundary (LR) on standardized {f1},{f2}")
    plt.xlabel(f"{f1} (z-score)")
    plt.ylabel(f"{f2} (z-score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"lr_boundary_{f1}__{f2}.png", dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alu", required=True)
    ap.add_argument("--memscan", required=True)
    ap.add_argument("--interr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--features", nargs="*", default=None,
                    help="Which features to histogram. Default = a strong, short list.")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--scatter_f1", default="exc_per_us2")
    ap.add_argument("--scatter_f2", default="lsu_per_cyc")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_alu = ensure_device_id(load_jsonl(args.alu))
    df_mem = ensure_device_id(load_jsonl(args.memscan))
    df_int = ensure_device_id(load_jsonl(args.interr))

    df_alu["attack_label"] = 0
    df_mem["attack_label"] = 0
    df_int["attack_label"] = 1

    df = pd.concat([df_alu, df_mem, df_int], ignore_index=True)
    df = add_features(df)

    feats = args.features if args.features else DEFAULT_PLOT_FEATURES
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise SystemExit("No valid features found to plot.")

    inj = df[df["attack_label"] == 0]
    it  = df[df["attack_label"] == 1]

    # Hist overlays
    for f in feats:
        x0 = inj[f].to_numpy(np.float64)
        x1 = it[f].to_numpy(np.float64)
        save_hist_overlay(x0, x1, f, outdir / f"hist_{f}.png", bins=args.bins)

    # 2D scatter + LR boundary (visual “linear separability” evidence)
    f1 = args.scatter_f1
    f2 = args.scatter_f2
    if f1 in df.columns and f2 in df.columns:
        save_2d_scatter_and_lr(df, f1, f2, outdir)
    else:
        print(f"Scatter features not found: {f1}, {f2}")

    # Quick textual hints
    (outdir / "README.txt").write_text(
        "Generated overlaid histograms per feature, plus a 2D scatter and an LR decision boundary plot.\n"
        "If the two classes are linearly separable, you should see strong separation in some histograms\n"
        "and a fairly clean split by the straight LR boundary in the 2D plot.\n",
        encoding="utf-8"
    )

    print(f"Saved figures -> {outdir}")

if __name__ == "__main__":
    main()