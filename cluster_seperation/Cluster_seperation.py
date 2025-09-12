import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.datasets import load_wine
import seaborn as sns

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# ------------------ PCA ------------------ #
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ------------------ Metric MDS ------------------ #
mds_metric = MDS(n_components=2, metric=True, random_state=42, n_init=4, max_iter=300)
X_mds_metric = mds_metric.fit_transform(X)

# ------------------ Non-Metric MDS ------------------ #
mds_nonmetric = MDS(n_components=2, metric=False, random_state=42, n_init=4, max_iter=300)
X_mds_nonmetric = mds_nonmetric.fit_transform(X)

# ------------------ Plotting ------------------ #
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# PCA
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1", ax=axs[0], s=60)
axs[0].set_title("PCA (Linear Projection)")

# Metric MDS
sns.scatterplot(x=X_mds_metric[:,0], y=X_mds_metric[:,1], hue=y, palette="Set1", ax=axs[1], s=60)
axs[1].set_title("Metric MDS (Preserves Distances)")

# Non-Metric MDS
sns.scatterplot(x=X_mds_nonmetric[:,0], y=X_mds_nonmetric[:,1], hue=y, palette="Set1", ax=axs[2], s=60)
axs[2].set_title("Non-Metric MDS (Preserves Rank Order)")

plt.suptitle("Comparison of PCA vs Metric MDS vs Non-Metric MDS on Wine Dataset", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------ Analysis ------------------ #
print("\n--- Analysis ---")
print("1) PCA finds directions of maximum variance. It often shows clusters well if they align with variance directions.")
print("2) Metric MDS preserves actual pairwise distances. It can capture non-linear separation better than PCA.")
print("3) Non-Metric MDS preserves rank order of distances. It is more flexible and can reveal clusters when data is non-linear.")
