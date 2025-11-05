import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

os.makedirs("rbm_output", exist_ok=True)

# Load and preprocess MNIST ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize images
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define RBM class ---
class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=64):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)

# Train RBM ---
rbm = RBM(n_vis=784, n_hid=64)
lr = 0.01
epochs = 10
train_losses = []

for epoch in range(epochs):
    epoch_error = 0
    for batch, _ in train_loader:
        v0 = batch.view(-1, 784)

        # Positive phase
        p_h, h0 = rbm.v_to_h(v0)

        # Negative phase
        p_v, v1 = rbm.h_to_v(h0)
        p_h1, _ = rbm.v_to_h(v1)

        # Compute gradients manually
        dW = torch.matmul(p_h.t(), v0) - torch.matmul(p_h1.t(), v1)
        rbm.W.data += lr * dW / v0.size(0)
        rbm.v_bias.data += lr * torch.mean(v0 - v1, dim=0)
        rbm.h_bias.data += lr * torch.mean(p_h - p_h1, dim=0)

        # Reconstruction error
        loss = torch.mean((v0 - v1) ** 2)
        epoch_error += loss.item()

    avg_error = epoch_error / len(train_loader)
    train_losses.append(avg_error)
    print(f"Epoch {epoch+1}/{epochs}, Reconstruction Error: {avg_error:.4f}")

# Plot and save reconstruction error curve ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', color='b')
plt.title("Reconstruction Error over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.savefig("rbm_output/reconstruction_error.png")
plt.close()

# Visualize and save learned filters ---
weights = rbm.W.data
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[i].view(28, 28).detach(), cmap='gray')
    ax.axis('off')
plt.suptitle("Learned RBM Filters", fontsize=14)
plt.savefig("rbm_output/rbm_filters.png")
plt.close()

# Visualize and save reconstructions ---
sample_batch, _ = next(iter(train_loader))
v0 = sample_batch.view(-1, 784)
_, h = rbm.v_to_h(v0)
p_v, v1 = rbm.h_to_v(h)

fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(v0[i].view(28, 28).detach(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(p_v[i].view(28, 28).detach(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
plt.suptitle("Original vs Reconstructed Samples", fontsize=12)
plt.savefig("rbm_output/reconstructed_samples.png")
plt.close()

# Save model weights ---
torch.save(rbm.state_dict(), "rbm_output/rbm_weights.pth")

print("\n✅ All results saved in the 'rbm_output' folder:")
print(" - reconstruction_error.png")
print(" - rbm_filters.png")
print(" - reconstructed_samples.png")
print(" - rbm_weights.pth")
