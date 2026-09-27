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
    "data/processed/final_kmeans_evaluation.csv"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(data):,} customers")


# ============================================================
# FINAL FEATURE SET
# ============================================================
# return_rate is intentionally excluded from clustering.
# It will remain available as a descriptive business KPI.

feature_columns = [
    "scaled_log_total_orders",
    "scaled_log_total_revenue",
    "scaled_log_average_order_value",
    "scaled_log_recency_days",
    "scaled_category_count",
    "scaled_average_discount"
]

X = data[feature_columns]

print(
    f"Using {len(feature_columns)} final clustering features"
)

print("\nFinal features:")

for feature in feature_columns:
    print(f" - {feature}")


# ============================================================
# EVALUATE K = 2 THROUGH 7
# ============================================================

results = []

print("\n" + "=" * 70)
print("FINAL K-MEANS EVALUATION")
print("=" * 70)

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
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df["inertia_reduction_pct"] = (
    results_df["inertia"]
    .pct_change()
    .abs()
    * 100
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PRINT INERTIA IMPROVEMENTS
# ============================================================

print("\n" + "=" * 70)
print("INERTIA IMPROVEMENT")
print("=" * 70)

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

print("\n" + "=" * 70)
print("HIGHEST SILHOUETTE SCORE")
print("=" * 70)

print(
    f"K={int(best_row['k'])} "
    f"with silhouette score "
    f"{best_row['silhouette_score']:.4f}"
)


# ============================================================
# CREATE FINAL ELBOW CHART
# ============================================================

Path("architecture").mkdir(
    parents=True,
    exist_ok=True
)

elbow_path = Path(
    "architecture/final_kmeans_elbow.png"
)

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["k"],
    results_df["inertia"],
    marker="o"
)

plt.title(
    "Final K-Means Elbow Analysis"
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

plt.savefig(
    elbow_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CREATE FINAL SILHOUETTE CHART
# ============================================================

silhouette_path = Path(
    "architecture/final_kmeans_silhouette.png"
)

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["k"],
    results_df["silhouette_score"],
    marker="o"
)

plt.title(
    "Final K-Means Silhouette Analysis"
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

plt.savefig(
    silhouette_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Evaluation table: {OUTPUT_PATH}"
)

print(
    f"Elbow chart: {elbow_path}"
)

print(
    f"Silhouette chart: {silhouette_path}"
)

print("\nFinal K-Means evaluation complete.")