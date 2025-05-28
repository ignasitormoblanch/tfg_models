import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


# Utility: reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────────────────────────────────────
# PCA-based alignment transform
class PCAAlign:
    def __call__(self, img: Image.Image) -> Image.Image:
        # convert to grayscale numpy array
        arr = np.array(img.convert('L'), dtype=np.float32)
        h, w = arr.shape
        # create coordinate grid
        u = np.arange(w)
        v = np.arange(h)
        U, V = np.meshgrid(u, v)
        # flatten
        flat = arr.flatten()
        Uf = U.flatten()
        Vf = V.flatten()
        # weighted mean
        total = flat.sum() + 1e-8
        u_mean = (Uf * flat).sum() / total
        v_mean = (Vf * flat).sum() / total
        # centered coords
        du = Uf - u_mean
        dv = Vf - v_mean
        # covariance entries
        c00 = (flat * du * du).sum() / total
        c01 = (flat * du * dv).sum() / total
        c11 = (flat * dv * dv).sum() / total
        cov = np.array([[c00, c01],[c01, c11]])
        # principal eigenvector
        vals, vecs = np.linalg.eigh(cov)
        v1 = vecs[:, np.argmax(vals)]
        # angle to vertical
        angle = np.degrees(np.arctan2(v1[1], v1[0]))
        rot_deg = 90.0 - angle
        return img.rotate(rot_deg, resample=Image.BILINEAR, fillcolor=0)

# ──────────────────────────────────────────────────────────────────────────────
# Simple CNN definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1   = nn.Linear(32*7*7, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ──────────────────────────────────────────────────────────────────────────────
# Train and eval functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
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
            total += imgs.size(0)
    return correct / total


def main():
    print('hola')
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    pca_tf = transforms.Compose([
        PCAAlign(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets & loaders
    train_base = torchvision.datasets.MNIST("./data", train=True, download=True, transform=base_tf)
    train_pca  = torchvision.datasets.MNIST("./data", train=True, download=True, transform=pca_tf)
    tb_loader  = torch.utils.data.DataLoader(train_base, batch_size=128, shuffle=True)
    tp_loader  = torch.utils.data.DataLoader(train_pca,  batch_size=128, shuffle=True)

    test_base = torchvision.datasets.MNIST("./data", train=False, download=True, transform=base_tf)
    test_pca  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=pca_tf)
    vb = torch.utils.data.DataLoader(test_base, batch_size=256, shuffle=False)
    vp = torch.utils.data.DataLoader(test_pca,  batch_size=256, shuffle=False)

    # Models
    mb = SimpleCNN().to(device)
    mp = SimpleCNN().to(device)
    opt_b = optim.Adam(mb.parameters(), lr=1e-3)
    opt_p = optim.Adam(mp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # Train
    epochs = 3
    for ep in range(1, epochs+1):
        lb = train_epoch(mb, tb_loader, opt_b, crit, device)
        lp = train_epoch(mp, tp_loader, opt_p, crit, device)
        ab = eval_accuracy(mb, vb, device)
        ap = eval_accuracy(mp, vp, device)
        print(f"Epoch {ep}: Base loss={lb:.4f}, acc={ab*100:.2f}% | PCA loss={lp:.4f}, acc={ap*100:.2f}%")

    # Rotation sweep
    angles = list(range(0, 360, 5))
    acc_b, acc_p = [], []
    for ang in angles:
        print(ang)
        tfb = transforms.Compose([
            transforms.Lambda(lambda im: TF.rotate(im, ang, fill=(0,))),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tfp = transforms.Compose([
            transforms.Lambda(lambda im: TF.rotate(im, ang, fill=(0,))),
            PCAAlign(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dsb = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfb)
        dsp = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfp)
        lb = torch.utils.data.DataLoader(dsb, batch_size=256, shuffle=False)
        lp = torch.utils.data.DataLoader(dsp, batch_size=256, shuffle=False)
        acc_b.append(100*eval_accuracy(mb, lb, device))
        acc_p.append(100*eval_accuracy(mp, lp, device))

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(angles, acc_b, '--', label='Base', color='C0')
    plt.plot(angles, acc_p, '-',  label='PCA Align', color='C1')
    plt.xlabel("Angle (°)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Rotation")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_rotation_180_pero_mal.png", dpi=150)
    print("Saved figure: pca_rotation_180_pero_mal.png")

if __name__ == "__main__":
    main()
