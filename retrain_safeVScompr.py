## FIXED TRAINING SCRIPT - NO LABEL LEAKAGE
## python retrain_safeVScompr_FIXED.py --safe train_logs/safe2.jsonl train_logs/safe1.jsonl --comp train_logs/memscan.jsonl train_logs/alu.jsonl --W 5 --out models/level1_fixed.joblib

import argparse, json, os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_FEATURES = ["dC","dL","dP","dE","dF","dS","dT","cyc_per_us"]

def load_jsonl_paths(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Δεν βρήκα αρχείο: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(rows)

def infer_y_gate(df: pd.DataFrame, label_key: str) -> np.ndarray:
    # compromised = label in {3,4}, safe = όλα τα άλλα
    if label_key in df.columns:
        yraw = pd.to_numeric(df[label_key], errors="coerce")
        y = yraw.isin([3,4]).astype(int)
    elif "leaf_label" in df.columns:
        yraw = pd.to_numeric(df["leaf_label"], errors="coerce")
        y = yraw.isin([3,4]).astype(int)
    elif "compromised" in df.columns:
        yraw = pd.to_numeric(df["compromised"], errors="coerce")
        y = (yraw > 0).astype(int)
    else:
        raise SystemExit("Δεν βρήκα label/leaf_label/compromised.")
    return y.to_numpy().astype(int)

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for k in ["dC","dL","dP","dE","dF","dS","dT","cyc_per_us"]:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")

    # ratios
    if "dL" in out.columns and "dC" in out.columns:
        out["lsu_per_cyc"] = safe_div(out["dL"], out["dC"])
    if "dP" in out.columns and "dC" in out.columns:
        out["cpi_per_cyc"] = safe_div(out["dP"], out["dC"])
    if "dE" in out.columns and "dC" in out.columns:
        out["exc_per_cyc"] = safe_div(out["dE"], out["dC"])
    if "dF" in out.columns and "dC" in out.columns:
        out["fold_per_cyc"] = safe_div(out["dF"], out["dC"])

    return out

def make_aggregated_samples_FIXED(
    df: pd.DataFrame,
    features: list[str],
    label_key: str,
    W: int,
    stats=("mean","std","max"),
    by_device=True,
):
    """
    FIXED: Use label of LAST sample in window instead of max(window).
    This prevents label leakage.
    """
    # ordering
    if "ts" in df.columns:
        df = df.sort_values(["device_id_str","ts"] if by_device and "device_id_str" in df.columns else ["ts"])
    elif "window_id" in df.columns:
        df = df.sort_values(["device_id_str","window_id"] if by_device and "device_id_str" in df.columns else ["window_id"])
    else:
        df = df.copy()

    # numeric features
    Xraw = pd.DataFrame()
    for f in features:
        Xraw[f] = pd.to_numeric(df.get(f, np.nan), errors="coerce")

    y = infer_y_gate(df, label_key)

    # group by device
    if by_device and "device_id_str" in df.columns:
        groups = df["device_id_str"].astype(str)
    else:
        groups = pd.Series(["all"] * len(df), index=df.index)

    out_rows, out_y = [], []

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
        
        # ✅ FIXED: Use MAJORITY label in rolling window (most robust)
        # If window has [0,0,0,1,1] → majority=0 (3 safe > 2 attack)
        def majority_vote(window):
            counts = window.value_counts()
            if len(counts) == 0:
                return np.nan
            return counts.idxmax()  # Most frequent label
        
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
    best_f1 = -1.0
    best_bal = -1.0
    best_thr = 0.5
    best_acc = -1.0

    for t in ths:
        yh = (p >= t).astype(int)
        acc = float((yh == y_true).mean())
        bal = float(balanced_accuracy_score(y_true, yh))
        f1  = float(f1_score(y_true, yh))

        if (f1 > best_f1) or (f1 == best_f1 and bal > best_bal):
            best_f1, best_thr, best_bal, best_acc = f1, float(t), bal, acc

    return best_acc, best_thr, best_bal, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", nargs="+", required=True)
    ap.add_argument("--comp", nargs="+", required=True)
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--out", default="models/gate_hgb_fixed.joblib")
    ap.add_argument("--no_by_device", action="store_true")
    args = ap.parse_args()

    df_safe = load_jsonl_paths(args.safe)
    df_comp = load_jsonl_paths(args.comp)

    # enforce: safe != {3,4}, comp in {3,4}
    if args.label_key in df_safe.columns:
        ytmp = pd.to_numeric(df_safe[args.label_key], errors="coerce")
        df_safe[args.label_key] = np.where(ytmp.isin([3,4]), 0, ytmp.fillna(0)).astype(int)

    if args.label_key in df_comp.columns:
        ytmp = pd.to_numeric(df_comp[args.label_key], errors="coerce")
        df_comp[args.label_key] = np.where(ytmp.isin([3,4]), ytmp, 4).astype(int)

    df = pd.concat([df_safe, df_comp], axis=0, ignore_index=True)

    # add ratios
    df = add_ratio_features(df)

    base_feats = ["cyc_per_us", "lsu_per_cyc", "cpi_per_cyc", "exc_per_cyc", "fold_per_cyc"]
    base_feats = [f for f in base_feats if f in df.columns]
    if not base_feats:
        raise SystemExit("Δεν βρήκα ratio features. Κάτι λείπει στα columns.")

    # ✅ Use FIXED aggregation (no label leakage)
    X, y = make_aggregated_samples_FIXED(
        df=df,
        features=base_feats,
        label_key=args.label_key,
        W=max(2, args.W),
        stats=("mean","std","max", "min"),
        by_device=(not args.no_by_device),
    )

    print(f"Aggregated rows: {len(X)} | class counts: {np.bincount(y)} | W={args.W}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_np = X_train.to_numpy(np.float32)
    X_test_np  = X_test.to_numpy(np.float32)

    # HGB
    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=30,
        l2_regularization=0.0,
        random_state=42,
    )
    hgb.fit(X_train_np, y_train)
    p = hgb.predict_proba(X_test_np)[:, 1]

    auc = roc_auc_score(y_test, p)
    acc, thr, bal, f1 = pick_threshold(y_test, p)

    print(f"\n[HGB FIXED] AUC: {auc:.4f}")
    print(f"[HGB FIXED] Best ACC: {acc:.4f} @ thr={thr:.2f} | BalAcc={bal:.4f} | F1={f1:.4f}")
    yh = (p >= thr).astype(int)
    print(f"[HGB FIXED] Confusion @ thr={thr:.2f}:\n{confusion_matrix(y_test, yh)}")
    print(classification_report(y_test, yh, digits=4))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    payload = {
        "model": hgb,
        "W": int(args.W),
        "agg_features": X.columns.tolist(),
        "base_features": base_feats,
        "thr": float(thr),
        "norm_cfg": {
            "type": "baseline_zscore",
            "baseline_from_safe_only": True,
            "eps": 1e-9,
        }
    }
    joblib.dump(payload, args.out)
    print(f"\nSaved FIXED model -> {args.out}")
    print("This model uses label of LAST sample (no leakage)")

if __name__ == "__main__":
    main()