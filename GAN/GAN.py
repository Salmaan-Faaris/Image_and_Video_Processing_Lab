import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os
from datetime import datetime

# === hyperparams and config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
BATCH_SIZE = 128
IMAGE_SIZE = 28
CHANNELS = 1
EPOCHS = 30
LR = 2e-4
BETA1 = 0.5
SAMPLE_INTERVAL = 5
OUTPUT_DIR = "gan_outputs"
SEED = 42

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === model definitions ===
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, CHANNELS * IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(z.size(0), CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CHANNELS * IMAGE_SIZE * IMAGE_SIZE, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.net(flat)

def save_sample(epoch, generator, fixed_noise):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).cpu()
    grid = utils.make_grid(fake, nrow=8, normalize=True, value_range=(-1,1))
    filename = os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.png")
    utils.save_image(grid, filename)
    print(f"[Sample Saved] {filename}")
    generator.train()

def main():
    print(f"Starting training on device: {DEVICE}. MNIST samples: 60000")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    # On Windows if you get worker crashes, set num_workers=0 here.
    dataloader = DataLoader(
        mnist,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # <- safe fallback; bump to 2 if stable
        pin_memory=(True if DEVICE.type == "cuda" else False)
    )

    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)

            valid = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_preds = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_preds, valid)

            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            fake_preds = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_preds, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            gen_preds = discriminator(gen_imgs)
            g_loss = adversarial_loss(gen_preds, valid)
            g_loss.backward()
            optimizer_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        avg_d = epoch_d_loss / len(dataloader)
        avg_g = epoch_g_loss / len(dataloader)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch:03d}/{EPOCHS} | D loss: {avg_d:.4f} | G loss: {avg_g:.4f}")

        if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
            save_sample(epoch, generator, fixed_noise)

        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, f"generator_epoch{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, f"discriminator_epoch{epoch}.pth"))

    save_sample(EPOCHS, generator, fixed_noise)
    print("Training complete.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
