"""
GAN simples (vanilla GAN, redes densas) treinado no MNIST.

Duas redes competem:
  - Gerador (G): recebe ruído aleatório e tenta produzir dígitos falsos
    que pareçam reais.
  - Discriminador (D): recebe uma imagem (real ou falsa) e tenta dizer
    se ela é real (veio do MNIST) ou falsa (veio do G).

O treino alterna: D aprende a distinguir melhor, G aprende a enganar D
melhor. Ao final, G sozinho é capaz de gerar dígitos novos e plausíveis
a partir de puro ruído.

Uso:
    python3 gan_mnist.py --epochs 30

Saídas (pasta output/):
    samples_epoch_XXX.png  -> grade de dígitos gerados a cada época
    loss_curve.png         -> curva de loss de G e D ao longo do treino
    generator.pt           -> pesos do gerador treinado
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

LATENT_DIM = 100 # ESPACO LATENTE
IMG_SIZE = 28 * 28
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=IMG_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size),
            nn.Tanh(),  # saida em [-1, 1], mesmo range das imagens normalizadas
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # probabilidade de ser real
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.net(flat)


def get_dataloader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    dataset = datasets.MNIST(
        root=os.path.join(OUTPUT_DIR, "data"),
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def train(epochs, batch_size, lr, sample_every):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataloader = get_dataloader(batch_size)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()#FUNCAO DE PERDA
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)

    g_losses, d_losses = [], []

    for epoch in range(1, epochs + 1):
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

          
            opt_d.zero_grad()

            pred_real = discriminator(real_imgs)
            loss_real = criterion(pred_real, real_labels)

            noise = torch.randn(bs, LATENT_DIM, device=device)
            fake_imgs = generator(noise)
            pred_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(pred_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward() #backpropagation
            opt_d.step()# ajuste de peso

          
            opt_g.zero_grad()
            pred = discriminator(fake_imgs)
            loss_g = criterion(pred, real_labels)
            loss_g.backward()#backpropagation
            opt_g.step()

        g_losses.append(loss_g.item())
        d_losses.append(loss_d.item())
        print(
            f"Epoca {epoch:03d}/{epochs} | loss_D: {loss_d.item():.4f} | "
            f"loss_G: {loss_g.item():.4f}"
        )

        if epoch % sample_every == 0 or epoch == epochs:
            save_samples(generator, fixed_noise, epoch, device)

    save_loss_curve(g_losses, d_losses)
    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, "generator.pt"))
    print(f"\nConcluido. Resultados salvos em: {OUTPUT_DIR}")


def save_samples(generator, fixed_noise, epoch, device):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).cpu()
    fake = (fake + 1) / 2  # volta de [-1,1] para [0,1] para salvar como imagem
    grid = make_grid(fake, nrow=8)
    path = os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch:03d}.png")
    save_image(grid, path)
    generator.train()
    print(f"  -> amostras salvas em {path}")


def save_loss_curve(g_losses, d_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Gerador (G)")
    plt.plot(d_losses, label="Discriminador (D)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Curva de treinamento do GAN")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> curva de loss salva em {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Treina um GAN simples no MNIST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--sample-every", type=int, default=5, help="salva amostras a cada N epocas"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.epochs, args.batch_size, args.lr, args.sample_every)
