# retrain_workload.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

from ml_utils import PerDeviceStandardScaler

EPS = 1e-6
DEVICE_COL = "device_id"

WINDOWS_JSONL = r"logs_2level\windows_pico2w_1_20260205_003322.jsonl"
OUT_PATH_WORKLOAD = r"models\level2a.joblib"

DROP_AFTER_SWITCH = 30
TEST_SIZE = 0.25
RANDOM_STATE = 42

RAW = ["dC","dL","dP","dE","dF","dS","dT","cyc_per_us"]
ENGINEERED = (
    ["lsu_per_cyc","cpi_per_cyc","exc_per_cyc","fold_per_cyc"] +
    ["lsu_per_us2","cpi_per_us2","exc_per_us2","cyc_per_lsu2"] +
    [f"log1p_{c}" for c in ["dC","dL","dP","dE","dF","dS","dT"]]
)
FEATURES = RAW + ENGINEERED


def load_windows(path: str) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No rows found.")
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
    for c in RAW + ["label","window_id",DEVICE_COL]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["lsu_per_cyc"]  = out["dL"] / (out["dC"] + 1.0)
    out["cpi_per_cyc"]  = out["dP"] / (out["dC"] + 1.0)
    out["exc_per_cyc"]  = out["dE"] / (out["dC"] + 1.0)
    out["fold_per_cyc"] = out["dF"] / (out["dC"] + 1.0)

    out["lsu_per_us2"]  = out["dL"] / (out["dT"] + EPS)
    out["cpi_per_us2"]  = out["dP"] / (out["dT"] + EPS)
    out["exc_per_us2"]  = out["dE"] / (out["dT"] + EPS)

    out["cyc_per_lsu2"] = out["dC"] / (out["dL"] + 1.0)

    for c in ["dC","dL","dP","dE","dF","dS","dT"]:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))
    return out


def drop_after_switch(df: pd.DataFrame, drop_n: int) -> pd.DataFrame:
    if drop_n <= 0:
        return df
    d = df.sort_values([DEVICE_COL, "window_id"]).reset_index(drop=True)
    keep = np.ones(len(d), dtype=bool)

    last_dev = None
    last_lbl = None
    cooldown = 0

    for i in range(len(d)):
        dev = int(d.loc[i, DEVICE_COL])
        lbl = int(d.loc[i, "label"])

        if last_dev is None or dev != last_dev:
            last_dev = dev
            last_lbl = lbl
            cooldown = drop_n
            keep[i] = False
            continue

        if lbl != last_lbl:
            last_lbl = lbl
            cooldown = drop_n

        if cooldown > 0:
            keep[i] = False
            cooldown -= 1

    return d[keep].copy()


def main():
    df = load_windows(WINDOWS_JSONL)
    df = ensure_device_id(df)

    df = df.dropna(subset=["label","window_id",DEVICE_COL]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df["window_id"] = pd.to_numeric(df["window_id"], errors="coerce")
    df[DEVICE_COL] = pd.to_numeric(df[DEVICE_COL], errors="coerce")

    df = df.dropna(subset=["label","window_id",DEVICE_COL]).copy()
    df["label"] = df["label"].astype(int)
    df[DEVICE_COL] = df[DEVICE_COL].astype(int)

    df = df[df["label"].isin([0,1,2])].copy()

    df = add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = drop_after_switch(df, DROP_AFTER_SWITCH)
    after = len(df)
    print(f"Drop-switch: {before} -> {after} (dropped {before-after})")

    df = df.dropna(subset=FEATURES + [DEVICE_COL, "label"]).copy()

    # Use a DataFrame so sklearn keeps feature names consistently
    X_df = df[FEATURES + [DEVICE_COL]].astype(np.float32).copy()
    y = df["label"].to_numpy(np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    nF = len(FEATURES)
    model = Pipeline([
        ("perdev_scaler", PerDeviceStandardScaler(n_features=nF)),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=800,
            random_state=RANDOM_STATE
        ))
    ])
    # class-balanced + extra boost for medium
# class-balanced + extra boost for medium (use TRAIN labels!)
    counts_tr = np.bincount(y_tr, minlength=3).astype(float)
    base = counts_tr.sum() / (3.0 * np.maximum(counts_tr, 1.0))
    base[1] *= 1.35
    w_tr = base[y_tr]


    sample_weight = base[y]
    model.fit(X_tr, y_tr, hgb__sample_weight=w_tr)


    #model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    print("Confusion:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, digits=4))
    print("Acc:", accuracy_score(y_te, pred))

    Path(OUT_PATH_WORKLOAD).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": FEATURES, "device_col": DEVICE_COL, "model": model}, OUT_PATH_WORKLOAD)
    print("Saved WORKLOAD ->", OUT_PATH_WORKLOAD)


if __name__ == "__main__":
    main()
