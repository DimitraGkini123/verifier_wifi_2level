# retrain_attacktype.py
# Example:
# python retrain_attacktype.py --alu train_logs/alu.jsonl --memscan train_logs/memscan.jsonl --interr train_logs/interr.jsonl --out models/level2b.joblib

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from ml_utils import PerDeviceStandardScaler

EPS = 1e-6
DEVICE_COL = "device_id"

RAW = ["dC", "dL", "dP", "dE", "dF", "dS", "dT", "cyc_per_us"]
ENGINEERED = (
    ["lsu_per_cyc", "cpi_per_cyc", "exc_per_cyc", "fold_per_cyc"]
    + ["lsu_per_us2", "cpi_per_us2", "exc_per_us2", "cyc_per_lsu2"]
    + [f"log1p_{c}" for c in ["dC", "dL", "dP", "dE", "dF", "dS", "dT"]]
)
FEATURES = RAW + ENGINEERED


def load_windows(path: str) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def ensure_device_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DEVICE_COL not in out.columns:
        if "device_id_str" in out.columns:
            # pico2w_1 -> 1
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

    out["lsu_per_cyc"] = out["dL"] / (out["dC"] + 1.0)
    out["cpi_per_cyc"] = out["dP"] / (out["dC"] + 1.0)
    out["exc_per_cyc"] = out["dE"] / (out["dC"] + 1.0)
    out["fold_per_cyc"] = out["dF"] / (out["dC"] + 1.0)

    out["lsu_per_us2"] = out["dL"] / (out["dT"] + EPS)
    out["cpi_per_us2"] = out["dP"] / (out["dT"] + EPS)
    out["exc_per_us2"] = out["dE"] / (out["dT"] + EPS)

    out["cyc_per_lsu2"] = out["dC"] / (out["dL"] + 1.0)

    for c in ["dC", "dL", "dP", "dE", "dF", "dS", "dT"]:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))

    return out


def drop_after_switch(df: pd.DataFrame, drop_n: int) -> pd.DataFrame:
    """
    Optional: If your file contains multiple sections with label switches.
    Usually 0 for attack-only files.
    """
    if drop_n <= 0 or "label" not in df.columns:
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

def plot_confusion_matrix(cm, class_names, title, out_path):
    fig = plt.figure(figsize=(5.5, 4.8))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    thresh = cm.max() * 0.5 if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val),
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(report_dict, class_names, title, out_path):
    precision = []
    recall = []
    f1 = []
    support = []

    for name in class_names:
        d = report_dict.get(name, {})
        precision.append(d.get("precision", 0))
        recall.append(d.get("recall", 0))
        f1.append(d.get("f1-score", 0))
        support.append(int(d.get("support", 0)))

    x = np.arange(len(class_names))
    w = 0.25

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)

    ax.bar(x - w, precision, width=w, label="precision")
    ax.bar(x, recall, width=w, label="recall")
    ax.bar(x + w, f1, width=w, label="f1-score")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(s={s})" for n, s in zip(class_names, support)])
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alu", required=True, help="windows jsonl for ALU injection")
    ap.add_argument("--memscan", required=True, help="windows jsonl for MEMSCAN attack")
    ap.add_argument("--interr", required=True, help="windows jsonl for INTERRUPT_STORM attack")
    ap.add_argument("--out", required=True, help="output .joblib path")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--drop_after_switch", type=int, default=0, help="usually 0 for attack-only files")
    args = ap.parse_args()

    # Load
    df_alu = ensure_device_id(load_windows(args.alu))
    df_mem = ensure_device_id(load_windows(args.memscan))
    df_int = ensure_device_id(load_windows(args.interr))

    # ====== NEW: binary attack-type labels ======
    # 0 = INJECTION  (ALU + MEMSCAN)
    # 1 = INTERRUPTION (INTERRUPT_STORM)
    df_alu["attack_label"] = 0
    df_mem["attack_label"] = 0
    df_int["attack_label"] = 1

    df = pd.concat([df_alu, df_mem, df_int], ignore_index=True)

    # Clean numeric essentials
    needed = ["window_id", DEVICE_COL, "attack_label"]
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed).copy()

    df["window_id"] = df["window_id"].astype(int)
    df[DEVICE_COL] = df[DEVICE_COL].astype(int)
    df["attack_label"] = df["attack_label"].astype(int)

    # Feature eng
    df = add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [DEVICE_COL, "attack_label"]).copy()

    # (Optional) drop-after-switch
    df["label"] = df["attack_label"]
    before = len(df)
    df = drop_after_switch(df, int(args.drop_after_switch))
    after = len(df)
    print(f"Drop-switch: {before} -> {after} (dropped {before-after})")

    X_df = df[FEATURES + [DEVICE_COL]].astype(np.float32).copy()
    y = df["attack_label"].to_numpy(np.int64)

    # Split
    strat = y if len(np.unique(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df,
        y,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=strat
    )

    nF = len(FEATURES)

    # Binary LR
    model = Pipeline([
        ("perdev_scaler", PerDeviceStandardScaler(n_features=nF)),
        ("lr", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    cm = confusion_matrix(y_te, pred, labels=[0,1])
    acc = accuracy_score(y_te, pred)

    report_txt = classification_report(
        y_te,
        pred,
        labels=[0,1],
        target_names=["INJECTION", "INTERRUPTION"],
        digits=4
    )

    report_dict = classification_report(
        y_te,
        pred,
        labels=[0,1],
        target_names=["INJECTION", "INTERRUPTION"],
        output_dict=True
    )

    print("Confusion:\n", cm)
    print(report_txt)
    print("Acc:", acc)

    # ----- SAVE PLOTS -----
    Path("viz_level2b_accuracies").mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm,
        ["INJECTION", "INTERRUPTION"],
        f"Level2b Attack-Type Confusion (Acc={acc:.4f})",
        "viz_level2b_accuracies/level2b_confusion.png"
    )

    plot_metrics(
        report_dict,
        ["INJECTION", "INTERRUPTION"],
        "Level2b Per-Class Metrics",
        "viz_level2b_accuracies/level2b_metrics.png"
    )

    print("Saved plots -> assets/level2b_confusion.png, assets/level2b_metrics.png")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = {
        "kind": "level2b",
        "features": FEATURES,
        "device_col": DEVICE_COL,
        "model": model,
        "labels": {
            "0": "INJECTION",      # (ALU + MEMSCAN)
            "1": "INTERRUPTION",   # (INTERRUPT_STORM)
        }
    }
    joblib.dump(blob, str(out_path))
    print("Saved ATTACKTYPE (compromised_only, binary) ->", str(out_path))


if __name__ == "__main__":
    main()
