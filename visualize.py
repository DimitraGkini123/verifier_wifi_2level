"""
Visualize separability of SAFE vs COMPROMISED windows from your JSONL logs.

USAGE: 
python visualize.py --safe train_logs/safe1.jsonl train_logs/safe2.jsonl --comp train_logs/memscan.jsonl train_logs/alu.jsonl
 --use_rolling --W 5 --outdir viz_rolling_W5
"""

import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -------------------- IO --------------------

def load_jsonl(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Missing file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        raise SystemExit("No rows loaded. Check your JSONL paths.")
    return pd.DataFrame(rows)


# -------------------- Features --------------------

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure numeric (if present)
    for k in ["dC", "dL", "dP", "dE", "dF", "dS", "dT", "cyc_per_us"]:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")

    # If cyc_per_us missing but dC/dT exist, approximate it
    if "cyc_per_us" not in out.columns and ("dC" in out.columns and "dT" in out.columns):
        out["cyc_per_us"] = safe_div(out["dC"], out["dT"])

    if "dL" in out.columns and "dC" in out.columns:
        out["lsu_per_cyc"] = safe_div(out["dL"], out["dC"])
    if "dP" in out.columns and "dC" in out.columns:
        out["cpi_per_cyc"] = safe_div(out["dP"], out["dC"])
    if "dE" in out.columns and "dC" in out.columns:
        out["exc_per_cyc"] = safe_div(out["dE"], out["dC"])
    if "dF" in out.columns and "dC" in out.columns:
        out["fold_per_cyc"] = safe_div(out["dF"], out["dC"])

    return out

def infer_y_gate(df: pd.DataFrame, label_key: str) -> np.ndarray:
    """
    compromised = label in {3,4}, safe otherwise
    Accepts label_key or leaf_label or compromised flag.
    """
    if label_key in df.columns:
        yraw = pd.to_numeric(df[label_key], errors="coerce")
        y = yraw.isin([3, 4]).astype(int)
        return y.to_numpy().astype(int)

    if "leaf_label" in df.columns:
        yraw = pd.to_numeric(df["leaf_label"], errors="coerce")
        y = yraw.isin([3, 4]).astype(int)
        return y.to_numpy().astype(int)

    if "compromised" in df.columns:
        yraw = pd.to_numeric(df["compromised"], errors="coerce")
        y = (yraw > 0).astype(int)
        return y.to_numpy().astype(int)

    raise SystemExit(f"Could not infer labels. Missing {label_key}/leaf_label/compromised.")


def make_rolling_aggregates(
    df: pd.DataFrame,
    base_feats: list[str],
    y: np.ndarray,
    W: int,
    stats=("mean", "std", "min", "max"),
    by_device=True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rolling aggregation per device_id_str (if present), otherwise global.
    Label is majority vote in the window (like your FIXED script).
    """
    df = df.copy()
    df["_y"] = y

    # Sort for temporal order
    if "ts" in df.columns:
        sort_cols = ["device_id_str", "ts"] if (by_device and "device_id_str" in df.columns) else ["ts"]
        df = df.sort_values(sort_cols)
    elif "window_id" in df.columns:
        sort_cols = ["device_id_str", "window_id"] if (by_device and "device_id_str" in df.columns) else ["window_id"]
        df = df.sort_values(sort_cols)
    else:
        # fallback: keep original order
        df = df.reset_index(drop=True)

    # Grouping
    if by_device and "device_id_str" in df.columns:
        groups = df["device_id_str"].astype(str)
    else:
        groups = pd.Series(["all"] * len(df), index=df.index)

    out_rows = []
    out_y = []

    def majority_vote(arr: pd.Series):
        vc = arr.value_counts()
        if vc.empty:
            return np.nan
        return int(vc.idxmax())

    for _, idx in groups.groupby(groups).groups.items():
        sub = df.loc[idx].reset_index(drop=True)

        # numeric base feats
        Xsub = pd.DataFrame({f: pd.to_numeric(sub.get(f, np.nan), errors="coerce") for f in base_feats})
        ysub = pd.to_numeric(sub["_y"], errors="coerce")

        feats = {}
        for f in base_feats:
            r = Xsub[f].rolling(W, min_periods=W)
            if "mean" in stats: feats[f"{f}_mean_W{W}"] = r.mean()
            if "std"  in stats: feats[f"{f}_std_W{W}"]  = r.std(ddof=0)
            if "min"  in stats: feats[f"{f}_min_W{W}"]  = r.min()
            if "max"  in stats: feats[f"{f}_max_W{W}"]  = r.max()

        Xagg = pd.DataFrame(feats)
        yagg = ysub.rolling(W, min_periods=W).apply(majority_vote, raw=False)

        m = Xagg.notna().all(axis=1) & yagg.notna()
        Xagg = Xagg[m].reset_index(drop=True)
        yagg = yagg[m].astype(int).to_numpy()

        if len(Xagg) > 0:
            out_rows.append(Xagg)
            out_y.append(pd.Series(yagg))

    if not out_rows:
        raise SystemExit("No aggregated rows produced. Check W and missing data.")

    X = pd.concat(out_rows, axis=0, ignore_index=True)
    y2 = pd.concat(out_y, axis=0, ignore_index=True).to_numpy().astype(int)
    return X, y2


# -------------------- Plots --------------------

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def savefig(path: str, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_feature_histograms(X: pd.DataFrame, y: np.ndarray, outdir: str, bins=60):
    for col in X.columns:
        xs = X[col].to_numpy()
        s0 = xs[y == 0]
        s1 = xs[y == 1]

        # robust x-limits (ignore insane outliers)
        lo = np.nanpercentile(xs, 1)
        hi = np.nanpercentile(xs, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            continue

        plt.figure(figsize=(7.5, 4.8))
        plt.hist(s0, bins=bins, density=True, alpha=0.55, label="SAFE", range=(lo, hi))
        plt.hist(s1, bins=bins, density=True, alpha=0.55, label="COMPROMISED", range=(lo, hi))
        plt.title(f"Distribution overlap: {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        savefig(os.path.join(outdir, f"hist_{col}.png"))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", nargs="+", required=True, help="SAFE jsonl paths")
    ap.add_argument("--comp", nargs="+", required=True, help="COMPROMISED jsonl paths")
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--use_rolling", action="store_true", help="use rolling aggregates like training")
    ap.add_argument("--W", type=int, default=5, help="rolling window size")
    ap.add_argument("--no_by_device", action="store_true")
    ap.add_argument("--outdir", default="viz_safe_vs_comp")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df_safe = load_jsonl(args.safe)
    df_comp = load_jsonl(args.comp)
    df = pd.concat([df_safe, df_comp], axis=0, ignore_index=True)

    df = add_ratio_features(df)
    y = infer_y_gate(df, args.label_key)

    base_feats = ["cyc_per_us", "lsu_per_cyc", "cpi_per_cyc", "exc_per_cyc", "fold_per_cyc"]
    base_feats = [f for f in base_feats if f in df.columns]
    if not base_feats:
        raise SystemExit("No base ratio features found. Check your columns / add_ratio_features().")

    X = df[base_feats].copy()

    # Clean infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    if args.use_rolling:
        X, y = make_rolling_aggregates(
            df=df,
            base_feats=base_feats,
            y=y,
            W=max(2, int(args.W)),
            by_device=(not args.no_by_device),
        )

    # Drop rows with NaNs for plotting consistency
    m = X.notna().all(axis=1)
    X = X[m].reset_index(drop=True)
    y = y[m.to_numpy()]

    # quick summary
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print(f"Samples used for plots: {len(X)} (SAFE={n0}, COMPROMISED={n1})")
    print(f"Features: {list(X.columns)}")
    print(f"Output dir: {args.outdir}")

    plot_feature_histograms(X, y, args.outdir, bins=int(args.bins))


    print("Done. Generated: hist_*.png")


if __name__ == "__main__":
    main()