import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate two synthetic signals (sine + square)
np.random.seed(0)
t = np.linspace(0, 8, 2000)
s1 = np.sin(2 * t)          # Sinusoidal signal
s2 = np.sign(np.sin(3 * t)) # Square wave signal

# Combine into a matrix
S = np.c_[s1, s2]
S /= S.std(axis=0) 

# Mix signals using random matrix
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(S, A.T)  

# Apply ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  
A_ = ica.mixing_

# Correlation check
corr = np.corrcoef(S_.T, S.T)[:2, 2:]
print("Correlation between estimated and true sources:\n", corr)

# Plot signals
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.title("Original Sources")
plt.plot(S)
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
plt.plot(X)
plt.subplot(3, 1, 3)
plt.title("Recovered Sources (after ICA)")
plt.plot(S_)
plt.tight_layout()
plt.show()
