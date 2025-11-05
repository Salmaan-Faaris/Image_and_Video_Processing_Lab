import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Create synthetic dataset ---
np.random.seed(42)
n_samples = 500
n_features = 5
latent_dim = 2

# True latent factors
Z_true = np.random.randn(n_samples, latent_dim)

# True weight matrix
W_true = np.random.randn(n_features, latent_dim)

# Generate data (with some Gaussian noise)
X_true = Z_true @ W_true.T + np.random.randn(n_samples, n_features) * 0.1

# --- Step 2: Introduce 10% missing values ---
X = X_true.copy()
missing_mask = np.random.rand(*X.shape) < 0.1  # 10% missing
X[missing_mask] = np.nan

print(f"Missing values introduced: {np.isnan(X).sum()} / {X.size}")

# --- Step 3: PPCA with missing data handling (EM algorithm) ---
def ppca_em_missing(X, n_components=2, max_iter=100, tol=1e-4):
    n, d = X.shape

    # Initialize missing mask
    isnan = np.isnan(X)
    X_filled = np.where(isnan, np.nanmean(X, axis=0), X)

    # Initialize parameters
    W = np.random.randn(d, n_components)
    sigma_sq = 1.0
    mu = np.nanmean(X, axis=0)

    for iteration in range(max_iter):
        X_centered = X_filled - mu
        M = W.T @ W + sigma_sq * np.eye(n_components)
        M_inv = np.linalg.inv(M)

        # --- E-step ---
        Ez = X_centered @ W @ M_inv
        Ezz = sigma_sq * M_inv + (Ez.T @ Ez) / n

        # --- M-step ---
        W_new = (X_centered.T @ Ez) @ np.linalg.inv(Ezz)
        sigma_sq_new = np.sum((X_centered - Ez @ W_new.T)**2) / (n * d)
        mu_new = np.nanmean(X_filled, axis=0)

        # --- Impute missing values ---
        X_filled[isnan] = (Ez @ W_new.T + mu_new)[isnan]

        # Check convergence
        if np.linalg.norm(W_new - W) < tol:
            print(f"Converged at iteration {iteration+1}")
            break

        W, sigma_sq, mu = W_new, sigma_sq_new, mu_new

    return X_filled, W, sigma_sq, mu

# --- Step 4: Run PPCA on data with missing values ---
X_imputed, W_est, sigma_sq_est, mu_est = ppca_em_missing(X, n_components=2, max_iter=100)

# --- Step 5: Evaluate imputation error ---
mse_missing = np.mean((X_true[missing_mask] - X_imputed[missing_mask])**2)
print(f"Imputation MSE for missing values: {mse_missing:.6f}")

# --- Step 6: Visualization of imputation quality ---
plt.figure(figsize=(7,5))
plt.scatter(X_true[missing_mask], X_imputed[missing_mask], alpha=0.6, color='royalblue')
plt.plot([X_true.min(), X_true.max()], [X_true.min(), X_true.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Imputed Values")
plt.title("PPCA Imputed vs True Missing Values")
plt.grid(True)
plt.show()
    