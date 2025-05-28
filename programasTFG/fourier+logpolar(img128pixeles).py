import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FourierLogPolar128:
    """
    Transforma:
      1) Escala la imagen a 128×128
      2) FFT 2D → magnitud → fftshift
      3) log(1+mag) → NORM_MINMAX a [0,255]
      4) warpPolar (log-polar) a 128×128
      5) Reduce a 28×28 para la red
    """
    def __init__(self, fft_size=128, out_size=(28,28)):
        import cv2
        self.cv2 = cv2
        self.fft_size = fft_size
        self.center   = (fft_size//2, fft_size//2)
        self.maxRadius= fft_size//2
        self.out_size = out_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1) resize a 128×128 y escala gris
        arr = np.array(img.convert('L').resize((self.fft_size, self.fft_size),
                                                Image.BILINEAR),
                       dtype=np.float32)
        # 2) FFT 2D y magnitud
        fft = np.fft.fft2(arr)
        mag = np.abs(fft)
        # 3) shift
        mag = np.fft.fftshift(mag)
        # 4) compresión logarítmica
        mag_log = np.log1p(mag)
        # 5) normalizar a [0,255]
        mag_norm = self.cv2.normalize(mag_log, None, 0, 255,
                                      self.cv2.NORM_MINMAX).astype(np.uint8)
        # 6) warpPolar log-polar
        lp128 = self.cv2.warpPolar(
            mag_norm,
            (self.fft_size, self.fft_size),
            self.center,
            self.maxRadius,
            flags=self.cv2.WARP_POLAR_LOG | self.cv2.INTER_LINEAR
        )
        # 7) reducir a 28×28
        lp28 = self.cv2.resize(lp128, self.out_size, interpolation=self.cv2.INTER_AREA)
        return Image.fromarray(lp28)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32, 3, padding=1)
        self.fc1   = nn.Linear(32*7*7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return correct / total

# ──────────────────────────────────────────────────────────────────────────────
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Preparando transforms y datos...")

    # 1) Transformación base sin cambio
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2) Transformación FFT→mag→log-polar a 128×128 → 28×28
    flp128_tf = transforms.Compose([
        FourierLogPolar128(fft_size=128, out_size=(28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # DataLoaders de entrenamiento
    train_base = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=base_tf)
    train_f128 = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=flp128_tf)
    loader_base= torch.utils.data.DataLoader(train_base, batch_size=128, shuffle=True)
    loader_f128= torch.utils.data.DataLoader(train_f128, batch_size=128, shuffle=True)

    # DataLoaders de test
    test_base  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=base_tf)
    test_f128  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=flp128_tf)
    loader_tb  = torch.utils.data.DataLoader(test_base,  batch_size=256, shuffle=False)
    loader_tf128 = torch.utils.data.DataLoader(test_f128, batch_size=256, shuffle=False)

    # Modelos, optimizadores, pérdida
    model_b = SimpleCNN().to(device)
    model_f = SimpleCNN().to(device)
    opt_b   = optim.Adam(model_b.parameters(), lr=1e-3)
    opt_f   = optim.Adam(model_f.parameters(), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

    # Entrenamiento
    epochs = 5
    for ep in range(1, epochs+1):
        loss_b = train_epoch(model_b, loader_base, opt_b, crit, device)
        acc_b  = eval_accuracy(model_b, loader_tb, device)
        loss_f = train_epoch(model_f, loader_f128, opt_f, crit, device)
        acc_f  = eval_accuracy(model_f, loader_tf128, device)
        print(f"Epoch {ep}/{epochs} | "
              f"Base → loss={loss_b:.4f}, acc={acc_b*100:.2f}% | "
              f"FFT128 → loss={loss_f:.4f}, acc={acc_f*100:.2f}%")

    # Comparación de accuracy vs rotación
    angles = list(range(0,360,5))
    accs_b, accs_f = [], []
    for ang in angles:
        print(ang)
        tfb = transforms.Compose([
            lambda img: TF.rotate(img, ang, fill=(0,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tff = transforms.Compose([
            lambda img: TF.rotate(img, ang, fill=(0,)),
            FourierLogPolar128(fft_size=128, out_size=(28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tb_r = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfb)
        tf_r = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tff)
        lb_r = torch.utils.data.DataLoader(tb_r, batch_size=256, shuffle=False)
        lf_r = torch.utils.data.DataLoader(tf_r, batch_size=256, shuffle=False)

        accs_b.append(100 * eval_accuracy(model_b, lb_r, device))
        accs_f.append(100 * eval_accuracy(model_f, lf_r, device))

    # Plot final
    plt.figure(figsize=(8,5))
    plt.plot(angles, accs_b, '--', label='Base', color='blue')
    plt.plot(angles, accs_f,  '-', label='FFT128+Mag+LogPolar', color='green')
    plt.xlabel("Ángulo de rotación (°)")
    plt.ylabel("Accuracy (%)")
    plt.title("Base vs FFT128+Magnitud+LogPolar")
    plt.legend(); plt.grid(True)
    plt.savefig("comparison_fft128_logpolar.png", dpi=150)
    print("✅ Gráfico guardado en ./comparison_fft128_logpolar.png")

if __name__ == "__main__":
    main()
