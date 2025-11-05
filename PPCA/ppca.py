import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Load and preprocess MNIST ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
])

mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=10000, shuffle=True)

# Take a subset of data for comparison
images, _ = next(iter(loader))
X = images.numpy()

# --- Apply classical PCA ---
n_components = 64
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
X_recon_pca = pca.inverse_transform(X_pca)

# --- Define a simple PPCA model ---
class PPCA:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape
        X_centered = X - X.mean(axis=0)
        self.mean_ = X.mean(axis=0)
        S = np.cov(X_centered, rowvar=False)

        # Initialize W randomly
        W = np.random.randn(d, self.n_components)
        sigma_sq = 1.0

        for i in range(self.max_iter):
            # Expectation step
            M = W.T @ W + sigma_sq * np.eye(self.n_components)
            M_inv = np.linalg.inv(M)
            Ez = X_centered @ W @ M_inv
            Ezz = sigma_sq * M_inv + Ez.T @ Ez / n

            # Maximization step
            W_new = (X_centered.T @ Ez) @ np.linalg.inv(Ezz)
            sigma_sq_new = (np.sum((X_centered - Ez @ W_new.T)**2) / (n * d))

            # Convergence check
            if np.linalg.norm(W_new - W) < self.tol:
                break

            W, sigma_sq = W_new, sigma_sq_new

        self.W = W
        self.sigma_sq = sigma_sq

    def transform(self, X):
        X_centered = X - self.mean_
        M = self.W.T @ self.W + self.sigma_sq * np.eye(self.n_components)
        M_inv = np.linalg.inv(M)
        Ez = X_centered @ self.W @ M_inv
        return Ez

    def inverse_transform(self, Ez):
        X_recon = Ez @ self.W.T + self.mean_
        return X_recon

# --- Train PPCA ---
ppca = PPCA(n_components=n_components, max_iter=50)
ppca.fit(X)
X_ppca = ppca.transform(X)
X_recon_ppca = ppca.inverse_transform(X_ppca)

# --- Compute Reconstruction Errors ---
mse_pca = np.mean((X - X_recon_pca)**2)
mse_ppca = np.mean((X - X_recon_ppca)**2)

print(f"Reconstruction MSE (PCA): {mse_pca:.6f}")
print(f"Reconstruction MSE (PPCA): {mse_ppca:.6f}")

# --- Visualize Original vs Reconstructed Images ---
def show_images(orig, recon1, recon2, n=8):
    fig, axes = plt.subplots(3, n, figsize=(10, 4))
    for i in range(n):
        axes[0, i].imshow(orig[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon1[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(recon2[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("PCA", fontsize=10)
    axes[2, 0].set_ylabel("PPCA", fontsize=10)
    plt.suptitle("Original vs PCA vs PPCA Reconstructions", fontsize=14)
    plt.tight_layout()
    plt.show()

show_images(X, X_recon_pca, X_recon_ppca)
