# plot_feature_distributions.py
# ------------------------------------------------------------
# Στόχος: Να δεις αν τα features διαχωρίζουν SAFE vs COMPROMISED
# 1) Διαβάζει πολλά .jsonl (train_logs/windows_*.jsonl)
# 2) Κρατά μόνο rows με τα βασικά features
# 3) Κάνει plots: hist + KDE-ish (hist density), boxplots, ROC-AUC per feature
#
# Χρήση:
#   pip install pandas matplotlib scikit-learn
#   python plot_feature_distributions.py --glob "train_logs/windows_*.jsonl" --label_key "label"
#
# Αν τα logs σου έχουν άλλο key για label, άλλαξέ το (π.χ. "leaf_label" ή "compromised").
# ------------------------------------------------------------

import argparse
import glob
import json
import os
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


# --------- Βοηθητικά ---------
def load_jsonl_files(paths: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append(obj)
                if max_rows is not None and len(rows) >= max_rows:
                    break
        if max_rows is not None and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def infer_binary_label(df: pd.DataFrame, label_key: str) -> pd.Series:
    """
    Επιστρέφει y ∈ {0,1}.
    Προσπαθεί:
      - αν label_key υπάρχει και είναι ήδη 0/1 -> το παίρνει
      - αν label είναι 0/1/2/4 (π.χ. leaf_label όπου 4=compromised) -> map
      - αλλιώς δοκιμάζει columns: compromised, leaf_label
    """
    if label_key in df.columns:
        y = df[label_key]
    elif "compromised" in df.columns:
        y = df["compromised"]
    elif "leaf_label" in df.columns:
        y = df["leaf_label"]
    else:
        raise ValueError(
            f"Δεν βρήκα label. Δώσε σωστό --label_key ή βάλε column compromised/leaf_label."
        )

    # Καθάρισμα
    y = pd.to_numeric(y, errors="coerce")

    # Αν είναι leaf_label (0,1,2,4) -> 1 αν 4 αλλιώς 0
    unique = set(y.dropna().unique().tolist())
    if unique.issubset({0, 1}):
        return y.astype(int)
    if 4 in unique:
        return (y == 4).astype(int)

    # Τελευταία προσπάθεια: anything >0 -> 1
    return (y > 0).astype(int)


def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    # Προσάρμοσε εδώ αν έχεις άλλα feature names
    candidates = [
        "dC", "dL", "dP", "dE", "dF", "dS", "dT",
        "cyc_per_us", "lsu_per_cyc", "cpi_per_cyc", "exc_per_cyc", "fold_per_cyc",
    ]
    return [c for c in candidates if c in df.columns]


def safe_savefig(outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


# --------- Plots ---------
def plot_hist_by_class(df: pd.DataFrame, y: pd.Series, feat: str, outdir: str):
    x0 = df.loc[y == 0, feat].dropna()
    x1 = df.loc[y == 1, feat].dropna()

    plt.figure()
    plt.hist(x0, bins=60, density=True, alpha=0.5, label="SAFE (0)")
    plt.hist(x1, bins=60, density=True, alpha=0.5, label="COMPROMISED (1)")
    plt.title(f"Histogram density: {feat}")
    plt.xlabel(feat)
    plt.ylabel("density")
    plt.legend()
    return safe_savefig(outdir, f"hist_{feat}.png")


def plot_box_by_class(df: pd.DataFrame, y: pd.Series, feat: str, outdir: str):
    x0 = df.loc[y == 0, feat].dropna()
    x1 = df.loc[y == 1, feat].dropna()

    plt.figure()
    plt.boxplot([x0.values, x1.values], labels=["SAFE (0)", "COMPROMISED (1)"], showfliers=False)
    plt.title(f"Boxplot: {feat}")
    plt.ylabel(feat)
    return safe_savefig(outdir, f"box_{feat}.png")


def feature_auc(df: pd.DataFrame, y: pd.Series, feat: str) -> float:
    x = pd.to_numeric(df[feat], errors="coerce")
    m = x.notna() & y.notna()
    x = x[m]
    yy = y[m]
    if yy.nunique() < 2 or len(yy) < 20:
        return float("nan")
    # Αν το feature έχει ανάποδη κατεύθυνση, AUC θα το δείξει (μπορείς να πάρεις max(auc, 1-auc))
    return float(roc_auc_score(yy, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True, help='π.χ. "train_logs/windows_*.jsonl"')
    ap.add_argument("--label_key", type=str, default="label",
                    help="ποιο column είναι το label (π.χ. label, compromised, leaf_label)")
    ap.add_argument("--outdir", type=str, default="feature_plots", help="φάκελος εξόδου")
    ap.add_argument("--max_rows", type=int, default=None, help="κόψε rows για πιο γρήγορα (π.χ. 20000)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"Δεν βρήκα αρχεία με glob: {args.glob}")

    print(f"Βρήκα {len(paths)} αρχεία.")
    df = load_jsonl_files(paths, max_rows=args.max_rows)
    print(f"Loaded rows: {len(df)} | columns: {len(df.columns)}")

    y = infer_binary_label(df, args.label_key)
    df = df.copy()
    df["_y"] = y

    feats = pick_feature_columns(df)
    if not feats:
        raise SystemExit("Δεν βρήκα κανένα από τα default features. Δες pick_feature_columns().")

    # Basic stats
    print("Class counts:", df["_y"].value_counts(dropna=False).to_dict())

    # AUC per feature
    rows = []
    for feat in feats:
        auc = feature_auc(df, df["_y"], feat)
        if auc == auc:  # not NaN
            rows.append((feat, auc, max(auc, 1.0 - auc)))
        else:
            rows.append((feat, float("nan"), float("nan")))

    auc_df = pd.DataFrame(rows, columns=["feature", "auc", "best_direction_auc"])
    auc_df = auc_df.sort_values("best_direction_auc", ascending=False)
    print("\nAUC per feature (όσο πιο κοντά στο 1.0 τόσο καλύτερα):")
    print(auc_df.to_string(index=False))

    # Save AUC table
    os.makedirs(args.outdir, exist_ok=True)
    auc_path = os.path.join(args.outdir, "feature_auc.csv")
    auc_df.to_csv(auc_path, index=False)
    print(f"\nΈσωσα AUC table -> {auc_path}")

    # Plots
    for feat in feats:
        # Αν έχει πολύ μεγάλες τιμές/ουρές, τα hist μπορεί να φαίνονται χάλια.
        # Τότε πρόσθεσε log1p transform για συγκεκριμένα features.
        hpath = plot_hist_by_class(df, df["_y"], feat, args.outdir)
        bpath = plot_box_by_class(df, df["_y"], feat, args.outdir)
        print(f"Saved: {hpath} | {bpath}")

    print("\nDone.")
    print("Τι να κοιτάξεις:")
    print("- Αν hist SAFE/COMPROMISED είναι σχεδόν ίδιο -> feature δεν βοηθά.")
    print("- Αν best_direction_auc ~ 0.50-0.55 για όλα -> θες καλύτερα features/labels/πειράματα.")
    print("- Αν 1-2 features έχουν best_direction_auc > 0.65 -> υπάρχει σήμα, αξίζει data + tuning.")


if __name__ == "__main__":
    main()
