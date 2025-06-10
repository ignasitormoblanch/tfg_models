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
from torchvision.transforms import ToPILImage

# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FourierLogPolarTransform:
    def __init__(self, size=(28,28)):
        import cv2
        self.cv2 = cv2
        h, w = size
        self.size = size
        self.center = (w//2, h//2)
        self.maxRadius = min(self.center)
        self.to_pil = ToPILImage()

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1) Convertir a escala de grises y a numpy float32
        arr = np.array(img.convert('L'), dtype=np.float32)
        # 2) FFT 2D y magnitud
        fft = np.fft.fft2(arr)
        mag = np.abs(fft)
        # 3) Desplazar frecuencia cero al centro
        mag = np.fft.fftshift(mag)
        # 4) Compresión logarítmica
        mag_log = np.log1p(mag)
        # 5) Normalizar a [0,255] con NORM_MINMAX
        mag_norm = self.cv2.normalize(
            mag_log, None, 0, 255, self.cv2.NORM_MINMAX
        ).astype(np.uint8)
        # 6) Transformada log-polar
        lp = self.cv2.warpPolar(
            mag_norm,
            self.size,
            self.center,
            self.maxRadius,
            flags=self.cv2.WARP_POLAR_LOG | self.cv2.INTER_LINEAR
        )
        return Image.fromarray(lp)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.fc1   = nn.Linear(32*7*7,128)
        self.fc2   = nn.Linear(128,10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2)
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
    print("🔧 Preparando transforms...")

    # Transform original
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Transform Fourier → Magnitud → Log-polar
    flp_tf = transforms.Compose([
        FourierLogPolarTransform((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # DataLoaders de entrenamiento
    train_base = torchvision.datasets.MNIST("./data", train=True, download=True, transform=base_tf)
    train_flp  = torchvision.datasets.MNIST("./data", train=True, download=True, transform=flp_tf)
    loader_base = torch.utils.data.DataLoader(train_base, batch_size=128, shuffle=True)
    loader_flp  = torch.utils.data.DataLoader(train_flp,  batch_size=128, shuffle=True)

    # DataLoaders de test
    test_base = torchvision.datasets.MNIST("./data", train=False, download=True, transform=base_tf)
    test_flp  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=flp_tf)
    loader_tb = torch.utils.data.DataLoader(test_base, batch_size=256, shuffle=False)
    loader_tl = torch.utils.data.DataLoader(test_flp,  batch_size=256, shuffle=False)

    # Modelos, optimizadores y función de pérdida
    model_b = SimpleCNN().to(device)
    model_f = SimpleCNN().to(device)
    opt_b   = optim.Adam(model_b.parameters(), lr=1e-3)
    opt_f   = optim.Adam(model_f.parameters(), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

    # Entrenamiento
    epochs = 10
    for ep in range(1, epochs+1):
        loss_b = train_epoch(model_b, loader_base, opt_b, crit, device)
        acc_b  = eval_accuracy(model_b, loader_tb, device)
        loss_f = train_epoch(model_f, loader_flp, opt_f, crit, device)
        acc_f  = eval_accuracy(model_f, loader_tl, device)
        print(f"Epoch {ep}/{epochs} | "
              f"Base → loss={loss_b:.4f}, acc={acc_b*100:.2f}% | "
              f"FLP  → loss={loss_f:.4f}, acc={acc_f*100:.2f}%")

    # Comparación bajo distintas rotaciones
    angles = list(range(0,360,5))
    accs_b = []; accs_f = []
    for ang in angles:
        print(ang)
        tfb = transforms.Compose([
            lambda img: TF.rotate(img, ang, fill=(0,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tff = transforms.Compose([
            lambda img: TF.rotate(img, ang, fill=(0,)),
            FourierLogPolarTransform((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        tb_r = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfb)
        tf_r = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tff)
        lb_r = torch.utils.data.DataLoader(tb_r, batch_size=256, shuffle=False)
        lf_r = torch.utils.data.DataLoader(tf_r, batch_size=256, shuffle=False)

        accs_b.append(100 * eval_accuracy(model_b, lb_r, device))
        accs_f.append(100 * eval_accuracy(model_f, lf_r, device))

    # Ploteo comparativo
    plt.figure(figsize=(8,5))
    plt.plot(angles, accs_b, '--', label='Base', color='blue')
    plt.plot(angles, accs_f,  '-', label='Fourier+Mag+LogPolar', color='green')
    plt.xlabel("Ángulo de rotación (°)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Rotación: Base vs Fourier+Magnitud+LogPolar")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_fourier_logpolar2.png", dpi=150)
    print("✅ Gráfico guardado en ./comparison_fourier_logpolar2.png")

if __name__ == "__main__":
    main()