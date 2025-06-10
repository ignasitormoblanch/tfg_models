#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST + alineación PCA + corrección de signo  ↑/↓  + duplicado 180°
La corrección de signo elimina la caída de accuracy en ±45 °.
Compatible con Windows (sin lambdas picklables) y num_workers>0.
"""

import random, numpy as np, torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RotateFill:
    """Transform que rota una imagen PIL X grados rellenando con 0."""
    def __init__(self, angle): self.angle = angle
    def __call__(self, img):   return TF.rotate(img, self.angle, fill=0)


class PCAAlign:
    """Rota la imagen para que su eje principal quede vertical."""
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        U, V = np.meshgrid(np.arange(w), np.arange(h))
        flat, Uf, Vf = arr.ravel(), U.ravel(), V.ravel()
        tot = flat.sum() + 1e-8
        uμ = (Uf * flat).sum() / tot
        vμ = (Vf * flat).sum() / tot
        du, dv = Uf - uμ, Vf - vμ
        c00 = (flat * du * du).sum() / tot
        c01 = (flat * du * dv).sum() / tot
        c11 = (flat * dv * dv).sum() / tot
        eigvals, eigvecs = np.linalg.eigh(np.array([[c00, c01], [c01, c11]]))
        vx, vy = eigvecs[:, eigvals.argmax()]      # vector propio principal
        angle = np.degrees(np.arctan2(vy, vx))
        return img.rotate(90.0 - angle, resample=Image.BILINEAR, fillcolor=0)


class SignFix:
    """
    Asegura que la mitad superior tenga (ligeramente) más tinta que la inferior.
    Si no, gira 180°.  El umbral epsilon evita bucles con imágenes vacías.
    """
    def __init__(self, epsilon: float = 1.0): self.eps = epsilon
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h = arr.shape[0] // 2
        if arr[:h].sum() + self.eps >= arr[h:].sum():   # “derecha” → tal cual
            return img
        return TF.rotate(img, 180, fill=0)              # “del revés” → girar


class PCADoubleMNIST(Dataset):
    def __init__(self, root="./data", train=True, download=True):
        self.mnist  = torchvision.datasets.MNIST(root, train=train, download=download)
        self.align  = PCAAlign()
        self.fix    = SignFix()
        self.tensor = T.ToTensor()
        self.norm   = T.Normalize((0.1307,), (0.3081,))
        self.N      = len(self.mnist)

    def __len__(self): return 2 * self.N

    def __getitem__(self, idx):
        img, lbl = self.mnist[idx % self.N]
        img = self.fix(self.align(img))      # orientación “canónica”

        if idx >= self.N:                    # segunda mitad: +180°
            img = TF.rotate(img, 180, fill=0)

        return self.norm(self.tensor(img)), lbl


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_epoch(model, loader, opt, crit, device):
    model.train(); total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad(); loss = crit(model(imgs), lbls)
        loss.backward();  opt.step(); total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); corr = tot = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        corr += (model(imgs).argmax(1) == lbls).sum().item()
        tot  += lbls.size(0)
    return corr / tot


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_tf  = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_b  = torchvision.datasets.MNIST("./data", train=True, download=True, transform=base_tf)
    test_b   = torchvision.datasets.MNIST("./data", train=False, download=True, transform=base_tf)
    loader_b = DataLoader(train_b, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    val_b    = DataLoader(test_b,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    train_p  = PCADoubleMNIST("./data", train=True, download=True)
    test_p   = torchvision.datasets.MNIST("./data", train=False, download=True,
                  transform=T.Compose([PCAAlign(), SignFix(), T.ToTensor(),
                                       T.Normalize((0.1307,), (0.3081,))]))
    loader_p = DataLoader(train_p, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    val_p    = DataLoader(test_p,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    m_b, m_p = SimpleCNN().to(device), SimpleCNN().to(device)
    opt_b, opt_p = optim.Adam(m_b.parameters(), 1e-3), optim.Adam(m_p.parameters(), 1e-3)
    crit = nn.CrossEntropyLoss()

    print('a entrenaar')
    for ep in range(1, 6):
        lb = train_epoch(m_b, loader_b, opt_b, crit, device)
        lp = train_epoch(m_p, loader_p, opt_p, crit, device)
        ab = eval_acc(m_b, val_b, device); ap = eval_acc(m_p, val_p, device)
        print(f"Epoch {ep}/5 │ Base loss={lb:.4f} acc={ab*100:5.2f}% │ "
              f"PCA+fix loss={lp:.4f} acc={ap*100:5.2f}%")

    angles = list(range(0, 360, 5)); acc_b, acc_p = [], []
    for ang in angles:
        tf_b = T.Compose([RotateFill(ang), base_tf])
        tf_p = T.Compose([RotateFill(ang), PCAAlign(), SignFix(),
                          T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        lb = DataLoader(torchvision.datasets.MNIST("./data", train=False, transform=tf_b),
                        batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        lp = DataLoader(torchvision.datasets.MNIST("./data", train=False, transform=tf_p),
                        batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        x = eval_acc(m_b, lb, device)
        y = eval_acc(m_p, lp, device)
        print(f'base {x} pca {y}')
        acc_b.append(100 * x)
        acc_p.append(100 * y)

    plt.figure(figsize=(8, 5))
    plt.plot(angles, acc_b, '--', label="Baseline")
    plt.plot(angles, acc_p, '-',  label="PCA + signo fijo")
    plt.xlabel("Ángulo de rotación (°)"); plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Rotación\nCon corrección de signo ↑/↓")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("comparison_signfix.png", dpi=150)
    print("Gráfica guardada en comparison_signfix.png")


if __name__ == "__main__":
    main()
