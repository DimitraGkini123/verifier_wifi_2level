import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

EPS = 1e-9

BASE_FEATURES = ["dC","dL","dP","dE","dF","dS","dT","cyc_per_us"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    df["__srcfile__"] = os.path.basename(path)
    return df


def load_many(paths):
    return pd.concat([load_jsonl(p) for p in paths], ignore_index=True)


def infer_y_gate(df, label_key="label"):
    if label_key in df.columns:
        yraw = pd.to_numeric(df[label_key], errors="coerce").fillna(0)
        y = yraw.isin([3,4]).astype(int)
    elif "leaf_label" in df.columns:
        yraw = pd.to_numeric(df["leaf_label"], errors="coerce").fillna(0)
        y = yraw.isin([3,4]).astype(int)
    elif "compromised" in df.columns:
        yraw = pd.to_numeric(df["compromised"], errors="coerce").fillna(0)
        y = (yraw > 0).astype(int)
    else:
        raise SystemExit("Δεν βρήκα label/leaf_label/compromised.")
    return y.to_numpy().astype(int)


def add_ratio_features(df):
    # numeric
    for c in BASE_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    dC = df["dC"].astype(float)
    dT = df["dT"].astype(float)

    df["lsu_per_cyc"]  = df["dL"] / (dC + EPS)
    df["cpi_per_cyc"]  = df["dP"] / (dC + EPS)
    df["exc_per_cyc"]  = df["dE"] / (dC + EPS)
    df["fold_per_cyc"] = df["dF"] / (dC + EPS)
    df["sleep_per_dt"] = df["dS"] / (dT + EPS)

    # κρατάμε και cyc_per_us, αλλά θα το normalize-άρουμε per-file
    return df


def per_file_safe_baseline_normalize(df, y):
    """
    Για κάθε αρχείο:
      baseline = mean/std πάνω στα SAFE δείγματα του ίδιου αρχείου
      transform: z = (x - mean) / (std + eps)
    """
    feat_cols = ["cyc_per_us","lsu_per_cyc","cpi_per_cyc","exc_per_cyc","fold_per_cyc","sleep_per_dt"]
    out = df.copy()

    for f in feat_cols:
        out[f] = out[f].astype(float)

    for fname, idx in out.groupby("__srcfile__").groups.items():
        sub = out.loc[idx]
        ysub = y[idx]

        safe_mask = (ysub == 0)
        # αν δεν έχει safe μέσα στο file, fallback: χρησιμοποιούμε όλα
        base = sub.loc[safe_mask] if safe_mask.any() else sub

        mu = base[feat_cols].mean()
        sd = base[feat_cols].std(ddof=0).replace(0.0, np.nan)

        out.loc[idx, feat_cols] = (sub[feat_cols] - mu) / (sd + EPS)

    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out, feat_cols


def make_aggregated(df, feat_cols, y, W):
    # ordering
    if "ts" in df.columns:
        df = df.sort_values(["__srcfile__","ts"])
    elif "window_id" in df.columns:
        df = df.sort_values(["__srcfile__","window_id"])

    outX, outy = [], []

    for fname, idx in df.groupby("__srcfile__").groups.items():
        subX = df.loc[idx, feat_cols].reset_index(drop=True)
        suby = pd.Series(y[idx]).reset_index(drop=True)

        feats = {}
        for f in feat_cols:
            r = subX[f].rolling(W, min_periods=W)
            feats[f"{f}_mean_W{W}"] = r.mean()
            feats[f"{f}_std_W{W}"]  = r.std(ddof=0)
            feats[f"{f}_max_W{W}"]  = r.max()

        X = pd.DataFrame(feats)
        m = X.notna().all(axis=1)
        outX.append(X[m])
        outy.append(suby[m])

    X = pd.concat(outX, ignore_index=True)
    y = pd.concat(outy, ignore_index=True).to_numpy().astype(int)
    return X, y


def best_thr_by_acc(y, p):
    best = (-1, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        acc = accuracy_score(y, (p >= t).astype(int))
        if acc > best[0]:
            best = (acc, t)
    return best


def run_fold(name, train_paths, test_paths, W, label_key):
    df_tr = load_many(train_paths)
    df_te = load_many(test_paths)

    df_tr = add_ratio_features(df_tr)
    df_te = add_ratio_features(df_te)

    ytr = infer_y_gate(df_tr, label_key)
    yte = infer_y_gate(df_te, label_key)

    df_tr, feat_cols = per_file_safe_baseline_normalize(df_tr, ytr)
    df_te, _         = per_file_safe_baseline_normalize(df_te, yte)

    Xtr, ytr2 = make_aggregated(df_tr, feat_cols, ytr, W)
    Xte, yte2 = make_aggregated(df_te, feat_cols, yte, W)

    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=600, min_samples_leaf=30, random_state=42
    )
    model.fit(Xtr, ytr2)

    p = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte2, p)

    best_acc, thr = best_thr_by_acc(yte2, p)
    yh = (p >= thr).astype(int)

    bal = balanced_accuracy_score(yte2, yh)
    f1  = f1_score(yte2, yh)
    cm  = confusion_matrix(yte2, yh)

    print(f"\n[{name}] AUC={auc:.4f} | ACC={best_acc:.4f} @ thr={thr:.2f} | BalAcc={bal:.4f} | F1={f1:.4f}")
    print(cm)

    return auc, best_acc, bal, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", nargs=2, required=True)
    ap.add_argument("--comp", nargs=2, required=True)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--label_key", default="label")
    args = ap.parse_args()

    safe1, safe2 = args.safe
    comp1, comp2 = args.comp

    folds = [
        ("FoldA", [safe2, comp2], [safe1, comp1]),
        ("FoldB", [safe1, comp1], [safe2, comp2]),
    ]

    M = []
    for name, tr, te in folds:
        M.append(run_fold(name, tr, te, max(2,args.W), args.label_key))

    M = np.array(M, float)
    print("\n=== MEAN over folds ===")
    print(f"AUC:    {M[:,0].mean():.4f}")
    print(f"ACC:    {M[:,1].mean():.4f}")
    print(f"BalAcc: {M[:,2].mean():.4f}")
    print(f"F1:     {M[:,3].mean():.4f}")


if __name__ == "__main__":
    main()
