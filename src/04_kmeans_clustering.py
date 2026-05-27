import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_utils import load_cleaned_data


RANDOM_STATE = 42
df = load_cleaned_data()

cluster_features = [
    "lead_time",
    "adr",
    "total_guests",
    "total_stays",
    "booking_changes",
    "total_of_special_requests",
]

# K-means is suitable for scalable and interpretable numeric booking segments.
X_scaled = StandardScaler().fit_transform(np.log1p(df[cluster_features]))

evaluation_rows = []
for k in range(1, 7):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    evaluation_rows.append(
        {
            "number_of_clusters": k,
            "inertia": model.inertia_,
            "silhouette_score": (
                np.nan
                if k == 1
                else silhouette_score(
                    X_scaled, labels, sample_size=10_000, random_state=RANDOM_STATE
                )
            ),
            "smallest_cluster_percentage": (
                pd.Series(labels).value_counts(normalize=True).min() * 100
            ),
        }
    )

evaluation = pd.DataFrame(evaluation_rows)
candidates = evaluation[
    (evaluation["number_of_clusters"] >= 2)
    & (evaluation["smallest_cluster_percentage"] >= 10)
]
number_of_clusters = int(
    candidates.loc[candidates["silhouette_score"].idxmax(), "number_of_clusters"]
)

df["cluster"] = KMeans(
    n_clusters=number_of_clusters, random_state=RANDOM_STATE, n_init=10
).fit_predict(X_scaled)

cluster_summary = df.groupby("cluster")[
    cluster_features + ["is_family", "is_canceled"]
].mean()
cluster_summary = cluster_summary.rename(
    columns={"is_family": "family_booking_rate", "is_canceled": "cancellation_rate"}
)
cluster_summary["booking_count"] = df.groupby("cluster").size()
cluster_summary["booking_percentage"] = cluster_summary["booking_count"] / len(df) * 100
cluster_summary[["family_booking_rate", "cancellation_rate"]] *= 100

# Cluster labels are arbitrary, so names are assigned from observed profiles.
cluster_names = {}
unlabeled_clusters = set(cluster_summary.index)
naming_rules = [
    ("booking_changes", "idxmax", "Booking-change bookings"),
    ("total_of_special_requests", "idxmax", "Special-request bookings"),
    ("lead_time", "idxmin", "Short-lead bookings"),
    ("lead_time", "idxmax", "Long-lead bookings"),
]
for column, selector, name in naming_rules:
    if not unlabeled_clusters:
        break
    values = cluster_summary.loc[list(unlabeled_clusters), column]
    cluster = getattr(values, selector)()
    cluster_names[cluster] = name
    unlabeled_clusters.remove(cluster)

for cluster in unlabeled_clusters:
    cluster_names[cluster] = f"Booking segment {cluster}"

cluster_summary.insert(
    0,
    "cluster_name",
    [cluster_names[cluster] for cluster in cluster_summary.index],
)

print("Cluster number evaluation")
print(evaluation.round(4).to_string(index=False))
print("\nSelected number of clusters:", number_of_clusters)
print("\nCluster summary")
print(cluster_summary.round(2).to_string())
print("\nCluster interpretation")
for cluster, name in cluster_names.items():
    print(f"- Cluster {cluster}: {name}")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(evaluation["number_of_clusters"], evaluation["inertia"], marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[1].plot(
    evaluation["number_of_clusters"].iloc[1:],
    evaluation["silhouette_score"].iloc[1:],
    marker="o",
)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Score")
plt.tight_layout()
plt.show()

scaled_profile = pd.DataFrame(X_scaled, columns=cluster_features)
scaled_profile["cluster"] = df["cluster"].to_numpy()
plt.figure(figsize=(10, 4))
sns.heatmap(
    scaled_profile.groupby("cluster").mean(),
    annot=True,
    center=0,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Standardized Feature Profile by Cluster")
plt.tight_layout()
plt.show()

pca_coordinates = PCA(n_components=2).fit_transform(X_scaled)
pca_plot = pd.DataFrame(pca_coordinates, columns=["PC1", "PC2"])
pca_plot["cluster"] = df["cluster"].to_numpy()
pca_plot["cluster_name"] = pca_plot["cluster"].map(cluster_names)
pca_plot = pca_plot.sample(n=min(10_000, len(pca_plot)), random_state=RANDOM_STATE)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=pca_plot,
    x="PC1",
    y="PC2",
    hue="cluster_name",
    alpha=0.45,
    s=18,
)
plt.title("K-means Clusters in PCA Space")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(data=cluster_summary.reset_index(), x="cluster", y="cancellation_rate")
plt.axhline(df["is_canceled"].mean() * 100, color="black", linestyle="--")
plt.title("Cancellation Rate by Cluster")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()
