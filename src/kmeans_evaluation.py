import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/processed/clustering_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/kmeans_evaluation.csv"
)

CHART_PATH = Path(
    "architecture/kmeans_evaluation.png"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(data):,} customers")


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

feature_columns = [
    column
    for column in data.columns
    if column.startswith("scaled_")
]

X = data[feature_columns]

print(
    f"Using {len(feature_columns)} clustering features"
)

print("\nFeatures:")

for feature in feature_columns:
    print(f" - {feature}")


# ============================================================
# EVALUATE DIFFERENT VALUES OF K
# ============================================================

results = []

print("\n" + "=" * 60)
print("K-MEANS MODEL EVALUATION")
print("=" * 60)

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(X)

    inertia = model.inertia_

    silhouette = silhouette_score(
        X,
        labels
    )

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette_score": silhouette
    })

    print(
        f"K={k} | "
        f"Inertia={inertia:,.2f} | "
        f"Silhouette={silhouette:.4f}"
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# CALCULATE INERTIA IMPROVEMENT
# ============================================================

results_df["inertia_reduction_pct"] = (
    results_df["inertia"]
    .pct_change()
    .abs()
    * 100
)

print("\n" + "=" * 60)
print("INERTIA IMPROVEMENT")
print("=" * 60)

for _, row in results_df.iterrows():

    if pd.isna(row["inertia_reduction_pct"]):

        print(
            f"K={int(row['k'])}: baseline"
        )

    else:

        print(
            f"K={int(row['k'])}: "
            f"{row['inertia_reduction_pct']:.2f}% "
            f"reduction from previous K"
        )


# ============================================================
# BEST SILHOUETTE SCORE
# ============================================================

best_row = results_df.loc[
    results_df["silhouette_score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST SILHOUETTE RESULT")
print("=" * 60)

print(
    f"K={int(best_row['k'])} "
    f"with silhouette score "
    f"{best_row['silhouette_score']:.4f}"
)


# ============================================================
# CREATE VISUALIZATION
# ============================================================
# We use two separate figures because inertia and silhouette
# measure different things and operate on different scales.

CHART_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Elbow / Inertia chart
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["k"],
    results_df["inertia"],
    marker="o"
)

plt.title(
    "K-Means Elbow Analysis"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.xticks(
    results_df["k"]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

elbow_path = Path(
    "architecture/kmeans_elbow.png"
)

plt.savefig(
    elbow_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# Silhouette chart
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["k"],
    results_df["silhouette_score"],
    marker="o"
)

plt.title(
    "K-Means Silhouette Analysis"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(
    results_df["k"]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

silhouette_path = Path(
    "architecture/kmeans_silhouette.png"
)

plt.savefig(
    silhouette_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    f"Evaluation results: {OUTPUT_PATH}"
)

print(
    f"Elbow chart: {elbow_path}"
)

print(
    f"Silhouette chart: {silhouette_path}"
)

print("\nK-Means evaluation complete.")