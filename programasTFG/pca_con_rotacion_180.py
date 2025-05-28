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


# Utility: reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# PCA-based alignment transform
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


# Dataset que duplica amb PCA i PCA+180°
class PCADoubleMNIST(Dataset):
    def __init__(self, root, train, download, base_tf, pca_tf):
        self.mnist = torchvision.datasets.MNIST(root, train=train, download=download)
        self.base_tf = base_tf
        self.pca_tf  = pca_tf
        self.N = len(self.mnist)
    def __len__(self):
        return 2 * self.N
    def __getitem__(self, idx):
        real_idx = idx % self.N
        img, lbl = self.mnist[real_idx]
        if idx < self.N:
            # imatge PCA
            return self.pca_tf(img), lbl
        else:
            # imatge PCA + 180°
            img_pca = self.pca_tf(img)
            # gir 180° després de convertir a PIL de nou
            pil = transforms.ToPILImage()(img_pca)
            pil = pil.rotate(180, fillcolor=0)
            return self.base_tf(pil), lbl


# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32, 3, padding=1)
        self.fc1   = nn.Linear(32*7*7, 128)
        self.fc2   = nn.Linear(128,10)
    def forward(self,x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_epoch(model, loader, optim, crit, device):
    model.train()
    running = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optim.zero_grad()
        loss = crit(model(imgs), lbls)
        loss.backward()
        optim.step()
        running += loss.item()
    return running/len(loader)

def eval_acc(model, loader, device):
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            corr += (preds==lbls).sum().item()
            tot  += lbls.size(0)
    return corr/tot


def main():
    print('hola')
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    pca_tf = transforms.Compose([
        PCAAlign(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    # Datasets & Loaders
    # 1) Baseline
    train_b = torchvision.datasets.MNIST("./data", train=True, download=True, transform=base_tf)
    loader_b = DataLoader(train_b, batch_size=128, shuffle=True)
    test_b  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=base_tf)
    val_b   = DataLoader(test_b,  batch_size=256, shuffle=False)

    # 2) PCA+180°
    train_p = PCADoubleMNIST("./data", train=True, download=True,
                             base_tf=base_tf, pca_tf=pca_tf)
    loader_p = DataLoader(train_p, batch_size=128, shuffle=True)
    # per validació PCA simple
    test_p_ = torchvision.datasets.MNIST("./data", train=False, download=True, transform=pca_tf)
    val_p   = DataLoader(test_p_, batch_size=256, shuffle=False)

    # Models, optims, criteri
    m_b = SimpleCNN().to(device)
    m_p = SimpleCNN().to(device)
    opt_b = optim.Adam(m_b.parameters(), lr=1e-3)
    opt_p = optim.Adam(m_p.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    # Entrenament
    epochs = 5
    for ep in range(1, epochs+1):
        lb = train_epoch(m_b, loader_b, opt_b, crit, device)
        lp = train_epoch(m_p, loader_p, opt_p, crit, device)
        ab = eval_acc(m_b, val_b, device)
        ap = eval_acc(m_p, val_p, device)
        print(f"Epoch {ep}/{epochs} → Base loss={lb:.4f}, acc={ab*100:.2f}% | "
              f"PCA+180° loss={lp:.4f}, acc={ap*100:.2f}%")

    # Sweep rotacions per ambdós
    angles = list(range(0,360,5))
    acc_base, acc_pca = [], []
    for ang in angles:
        print(ang)
        # baseline transform
        tfb = transforms.Compose([
            transforms.Lambda(lambda im: TF.rotate(im, ang, fill=(0,))),
            base_tf
        ])
        # pca transform
        # PCAAlign + 180° fixe
        tfp = transforms.Compose([
            transforms.Lambda(lambda im: TF.rotate(im, ang, fill=(0,))),
            PCAAlign(),
            transforms.Lambda(lambda im: TF.rotate(im, 180, fill=(0,))),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dsb = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfb)
        dsp = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfp)
        lb = DataLoader(dsb, batch_size=256, shuffle=False)
        lp = DataLoader(dsp, batch_size=256, shuffle=False)
        acc_base.append(100 * eval_acc(m_b, lb, device))
        acc_pca.append(100  * eval_acc(m_p, lp, device))

    # Ploteig
    plt.figure(figsize=(8,5))
    plt.plot(angles, acc_base, '--', label='Baseline', color='C0')
    plt.plot(angles, acc_pca, '-',  label='PCA+180°',   color='C1')
    plt.xlabel("Angle de rotació (°)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Rotació\nBaseline vs PCA+180° a l'entrenament")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("comparison_180_pero_medio.png", dpi=150)
    print("Gràfica desada com a comparison_180_pero_medio.png")

if __name__ == "__main__":
    main()
