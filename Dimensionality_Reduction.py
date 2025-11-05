# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import umap.umap_ as umap

# Load a high-dimensional dataset (e.g., handwritten digits: 64D)
data = load_digits()
X = data.data
y = data.target

# --- Dimensionality Reduction Methods ---
# 1. t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# 2. LLE (Locally Linear Embedding)
lle = LocallyLinearEmbedding(n_neighbors=30, n_components=2, random_state=42)
X_lle = lle.fit_transform(X)

# 3. UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)

# 4. Isomap
isomap = Isomap(n_neighbors=30, n_components=2)
X_isomap = isomap.fit_transform(X)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
methods = ['t-SNE', 'LLE', 'UMAP', 'Isomap']
embeddings = [X_tsne, X_lle, X_umap, X_isomap]

for ax, method, emb in zip(axes.flat, methods, embeddings):
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='Spectral', s=10)
    ax.set_title(method, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

plt.colorbar(scatter, ax=axes, orientation='horizontal', fraction=0.03, pad=0.05)
plt.suptitle("2D Embeddings using t-SNE, LLE, UMAP, and Isomap", fontsize=16)
plt.show()
