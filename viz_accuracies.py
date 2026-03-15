# viz_level1_report.py
# Example:
# python viz_level1_report.py ^
#   --safe train_logs/safe1.jsonl train_logs/safe2.jsonl train_logs/safe3.jsonl ^
#   --comp train_logs/alu.jsonl train_logs/memscan.jsonl train_logs/interr.jsonl ^
#   --W 5 --outdir viz_level1

import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import HistGradientBoostingClassifier

EPS = 1e-9

def load_jsonl_paths(paths: list[str]) -> pd.DataFrame:
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
        raise SystemExit("No rows loaded.")
    return pd.DataFrame(rows)

def infer_y_gate(df: pd.DataFrame, label_key: str) -> np.ndarray:
    # compromised = label in {3,4}, safe = everything else
    if label_key in df.columns:
        yraw = pd.to_numeric(df[label_key], errors="coerce")
        y = yraw.isin([3, 4]).astype(int)
    elif "leaf_label" in df.columns:
        yraw = pd.to_numeric(df["leaf_label"], errors="coerce")
        y = yraw.isin([3, 4]).astype(int)
    elif "compromised" in df.columns:
        yraw = pd.to_numeric(df["compromised"], errors="coerce")
        y = (yraw > 0).astype(int)
    else:
        raise SystemExit("Could not find label/leaf_label/compromised.")
    return y.to_numpy().astype(int)

def safe_div(a, b, eps=EPS):
    return a / (b + eps)

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k in ["dC","dL","dP","dE","dF","dS","dT","cyc_per_us"]:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")

    out["lsu_per_cyc"]  = safe_div(out["dL"], out["dC"])
    out["cpi_per_cyc"]  = safe_div(out["dP"], out["dC"])
    out["exc_per_cyc"]  = safe_div(out["dE"], out["dC"])
    out["fold_per_cyc"] = safe_div(out["dF"], out["dC"])
    return out

def make_aggregated_samples_FIXED(
    df: pd.DataFrame,
    features: list[str],
    label_key: str,
    W: int,
    stats=("mean","std","max","min"),
    by_device=True,
):
    # ordering
    if "ts" in df.columns:
        df = df.sort_values(["device_id_str","ts"] if by_device and "device_id_str" in df.columns else ["ts"])
    elif "window_id" in df.columns:
        df = df.sort_values(["device_id_str","window_id"] if by_device and "device_id_str" in df.columns else ["window_id"])
    else:
        df = df.copy()

    Xraw = pd.DataFrame({f: pd.to_numeric(df.get(f, np.nan), errors="coerce") for f in features})
    y = infer_y_gate(df, label_key)

    if by_device and "device_id_str" in df.columns:
        groups = df["device_id_str"].astype(str)
    else:
        groups = pd.Series(["all"] * len(df), index=df.index)

    out_rows, out_y = [], []

    def majority_vote(window: pd.Series):
        counts = window.value_counts()
        if len(counts) == 0:
            return np.nan
        return counts.idxmax()

    for _, idx in groups.groupby(groups).groups.items():
        subX = Xraw.loc[idx].reset_index(drop=True)
        suby = pd.Series(y[idx]).reset_index(drop=True)

        feats = {}
        for f in features:
            r = subX[f].rolling(W, min_periods=W)
            if "mean" in stats: feats[f"{f}_mean_W{W}"] = r.mean()
            if "std"  in stats: feats[f"{f}_std_W{W}"]  = r.std(ddof=0)
            if "max"  in stats: feats[f"{f}_max_W{W}"]  = r.max()
            if "min"  in stats: feats[f"{f}_min_W{W}"]  = r.min()

        agg = pd.DataFrame(feats)
        agg_y = suby.rolling(W, min_periods=W).apply(majority_vote, raw=False)

        m = agg.notna().all(axis=1) & agg_y.notna()
        agg = agg[m]
        agg_y = agg_y[m].astype(int)

        out_rows.append(agg)
        out_y.append(agg_y)

    Xagg = pd.concat(out_rows, axis=0, ignore_index=True)
    yagg = pd.concat(out_y, axis=0, ignore_index=True).to_numpy().astype(int)
    return Xagg, yagg

def pick_threshold(y_true, p):
    ths = np.linspace(0.05, 0.95, 91)
    best = (-1.0, -1.0, 0.5)  # (f1, balacc, thr)
    for t in ths:
        yh = (p >= t).astype(int)
        bal = float(balanced_accuracy_score(y_true, yh))
        f1  = float(f1_score(y_true, yh))
        if (f1 > best[0]) or (f1 == best[0] and bal > best[1]):
            best = (f1, bal, float(t))
    return best[2]

def plot_confmat(ax, cm, labels):
    ax.imshow(cm, interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", nargs="+", required=True)
    ap.add_argument("--comp", nargs="+", required=True)
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--outdir", default="viz_level1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_safe = load_jsonl_paths(args.safe)
    df_comp = load_jsonl_paths(args.comp)
    df = pd.concat([df_safe, df_comp], axis=0, ignore_index=True)

    df = add_ratio_features(df)

    base_feats = ["cyc_per_us", "lsu_per_cyc", "cpi_per_cyc", "exc_per_cyc", "fold_per_cyc"]
    X, y = make_aggregated_samples_FIXED(
        df=df,
        features=base_feats,
        label_key=args.label_key,
        W=max(2, args.W),
        stats=("mean", "std", "max", "min"),
        by_device=("device_id_str" in df.columns),
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=30,
        l2_regularization=0.0,
        random_state=args.seed,
    )
    model.fit(X_tr.to_numpy(np.float32), y_tr)

    p = model.predict_proba(X_te.to_numpy(np.float32))[:, 1]
    thr = pick_threshold(y_te, p)
    yh = (p >= thr).astype(int)

    auc = float(roc_auc_score(y_te, p))
    f1  = float(f1_score(y_te, yh))
    bal = float(balanced_accuracy_score(y_te, yh))
    acc = float(accuracy_score(y_te, yh))
    cm  = confusion_matrix(y_te, yh)

    # --- Figure layout: ConfMat + ROC + metrics box ---
    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.9])

    ax0 = fig.add_subplot(gs[0, 0])
    plot_confmat(ax0, cm, labels=["SAFE", "COMP"])

    ax1 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_te, p)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curve")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    txt = (
        f"Level 1 (SAFE vs COMP)\n"
        f"W = {args.W}\n\n"
        f"AUC   : {auc:.4f}\n"
        f"F1    : {f1:.4f}\n"
        f"BalAcc: {bal:.4f}\n"
        f"Acc   : {acc:.4f}\n"
        f"thr   : {thr:.2f}\n\n"
        f"Test support:\n"
        f"SAFE={int((y_te==0).sum())}, COMP={int((y_te==1).sum())}"
    )
    ax2.text(0.0, 1.0, txt, va="top")

    fig.suptitle("Level 1 evaluation on windowed HPC features", y=1.02)
    fig.tight_layout()

    png_path = outdir / f"level1_eval_W{args.W}.png"
    pdf_path = outdir / f"level1_eval_W{args.W}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Also dump a small one-line summary for LaTeX copy-paste
    summary_path = outdir / "level1_metrics.txt"
    summary_path.write_text(
        f"AUC={auc:.4f}, Acc={acc:.4f}, BalAcc={bal:.4f}, F1={f1:.4f}, thr={thr:.2f}, CM={cm.tolist()}\n",
        encoding="utf-8"
    )

    print("Saved:")
    print(" ", png_path)
    print(" ", pdf_path)
    print(" ", summary_path)

if __name__ == "__main__":
    main()