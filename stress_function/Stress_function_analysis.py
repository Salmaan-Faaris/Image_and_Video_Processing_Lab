import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

dims = range(1, 7)   # try embedding dimensions 1 → 6
stress_values = []

# Run Non-Metric MDS for each dimension
for d in dims:
    mds = MDS(n_components=d, metric=False, random_state=42, n_init=4, max_iter=300)
    mds.fit(X)
    stress_values.append(mds.stress_)

# Plot Stress vs Dimension
plt.figure(figsize=(8,5))
plt.plot(dims, stress_values, marker='o', linestyle='-')
plt.xlabel("Number of Dimensions (embedding space)")
plt.ylabel("Stress")
plt.title("Non-Metric MDS: Stress vs. Number of Dimensions (Iris Dataset)")
plt.grid(True)
plt.show()
