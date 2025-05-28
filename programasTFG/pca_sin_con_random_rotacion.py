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
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Utility: reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────────────────────────────────────
# PCA-based centering (as before)
class PCAAlign:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('L'), dtype=np.float32)
        h, w = arr.shape
        U, V = np.meshgrid(np.arange(w), np.arange(h))
        flat = arr.flatten()
        Uf, Vf = U.flatten(), V.flatten()
        total = flat.sum() + 1e-8
        u_mean = (Uf * flat).sum() / total
        v_mean = (Vf * flat).sum() / total
        du, dv = Uf - u_mean, Vf - v_mean
        c00 = (flat * du * du).sum() / total
        c01 = (flat * du * dv).sum() / total
        c11 = (flat * dv * dv).sum() / total
        cov = np.array([[c00, c01], [c01, c11]])
        vals, vecs = np.linalg.eigh(cov)
        v1 = vecs[:, np.argmax(vals)]
        angle = np.degrees(np.arctan2(v1[1], v1[0]))
        rot_deg = 90.0 - angle
        return img.rotate(rot_deg, resample=Image.BILINEAR, fillcolor=0)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset with optional RandomRotation(360)
class AugmentedMNIST(Dataset):
    def __init__(self, root, train, download, transform):
        self.ds = torchvision.datasets.MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        x = self.transform(img)
        return x, lbl

# ──────────────────────────────────────────────────────────────────────────────
# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.fc1   = nn.Linear(32*7*7,128)
        self.fc2   = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
def main():
    print('hola')
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline transform (no rotation)
    baseline_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # PCA + RandomRotation(360) augment
    augment_tf = transforms.Compose([
        PCAAlign(),
        transforms.RandomRotation(360, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets & loaders
    train_base = AugmentedMNIST("./data", train=True,  download=True, transform=baseline_tf)
    train_aug  = AugmentedMNIST("./data", train=True,  download=True, transform=augment_tf)
    tb_loader  = DataLoader(train_base, batch_size=128, shuffle=True)
    ta_loader  = DataLoader(train_aug,  batch_size=128, shuffle=True)

    test_base  = AugmentedMNIST("./data", train=False, download=True, transform=baseline_tf)

    # Models, optim & criterion
    model_b = SimpleCNN().to(device)
    model_a = SimpleCNN().to(device)
    opt_b   = optim.Adam(model_b.parameters(), lr=1e-3)
    opt_a   = optim.Adam(model_a.parameters(), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

    # Train both
    epochs = 5
    for ep in range(1, epochs+1):
        lb = train_epoch(model_b, tb_loader, opt_b, crit, device)
        la = train_epoch(model_a, ta_loader, opt_a, crit, device)
        print(f"Epoch {ep}/{epochs} • Loss Base={lb:.4f} • Loss Aug={la:.4f}")

    # Rotation sweep
    angles = list(range(0,360,5))
    acc_base, acc_aug = [], []
    for ang in angles:
        print(ang)
        tfb = transforms.Compose([
            transforms.Lambda(lambda im: TF.rotate(im, ang, fill=(0,))),
            baseline_tf
        ])
        tfa = transforms.Compose([
            transforms.Lambda(lambda im: TF.rotate(im, ang, fill=(0,))),
            augment_tf
        ])
        dsb = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfb)
        dsa = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfa)
        vb  = DataLoader(dsb, batch_size=256, shuffle=False)
        va  = DataLoader(dsa, batch_size=256, shuffle=False)
        acc_base.append(100 * eval_accuracy(model_b, vb, device))
        acc_aug .append(100 * eval_accuracy(model_a, va, device))

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(angles, acc_base, '--', label='Baseline')
    plt.plot(angles, acc_aug,  '-',  label='PCA + RandomRotation')
    plt.xlabel("Angle (°)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Rotation\nBaseline vs PCA+RandomRotation")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("pca+randomrotation.png", dpi=150)
    print("Saved: pca+randomrotation.png")

if __name__ == "__main__":
    main()
