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
import cv2


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LogPolarTransform:
    def __init__(self, size=(28,28)):

        self.cv2 = cv2
        self.size = size
        self.center = (size[0]//2, size[1]//2)
        self.maxRadius = min(self.center)
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('L'))
        lp = self.cv2.warpPolar(arr,
                                self.size,
                                self.center,
                                self.maxRadius,
                                flags=self.cv2.WARP_POLAR_LOG)
        return Image.fromarray(lp)

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
    correct=total=0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds==lbls).sum().item()
            total   += lbls.size(0)
    return correct/total


def main():
    print('hi')
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms base i log-polar
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    lp_tf   = transforms.Compose([
        LogPolarTransform((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    # Datasets i loaders d'entrenament
    train_base = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=base_tf)
    train_lp   = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=lp_tf)
    lb = torch.utils.data.DataLoader(train_base, batch_size=128, shuffle=True)
    ll = torch.utils.data.DataLoader(train_lp,   batch_size=128, shuffle=True)

    # Test loaders sense rotació (per mesurar accuracy després de cada epoch)
    test_base = torchvision.datasets.MNIST("./data", train=False, download=True, transform=base_tf)
    test_lp   = torchvision.datasets.MNIST("./data", train=False, download=True, transform=lp_tf)
    tb = torch.utils.data.DataLoader(test_base, batch_size=256, shuffle=False)
    tl = torch.utils.data.DataLoader(test_lp,   batch_size=256, shuffle=False)

    # Instanciem models, optimitzador i criteri
    model_base = SimpleCNN().to(device)
    model_lp   = SimpleCNN().to(device)
    opt_base   = optim.Adam(model_base.parameters(), lr=1e-3)
    opt_lp     = optim.Adam(model_lp.parameters(),   lr=1e-3)
    crit       = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(1, epochs+1):
        # Entrenament base
        loss_b = train_epoch(model_base, lb, opt_base, crit, device)
        acc_b  = eval_accuracy(model_base, tb, device)
        # Entrenament log-polar
        loss_l = train_epoch(model_lp,   ll, opt_lp,   crit, device)
        acc_l  = eval_accuracy(model_lp,   tl,   device)
        # Imprimir metrics
        print(f"Època {epoch}/{epochs} | "
              f"Base → loss={loss_b:.4f}, acc={acc_b*100:.2f}% | "
              f"Log‑Polar → loss={loss_l:.4f}, acc={acc_l*100:.2f}%")

    # Comparació final en rotacions
    angles = list(range(0,360,5))
    accs_b = []; accs_l = []
    for ang in angles:
        tfb = transforms.Compose([
            lambda img: TF.rotate(img, ang, fill=(0,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
        tfl = transforms.Compose([
            lambda img: TF.rotate(img, ang, fill=(0,)),
            LogPolarTransform((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
        test_b = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfb)
        test_l = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfl)
        loader_b = torch.utils.data.DataLoader(test_b, batch_size=256, shuffle=False)
        loader_l = torch.utils.data.DataLoader(test_l, batch_size=256, shuffle=False)
        accs_b.append(100*eval_accuracy(model_base, loader_b, device))
        accs_l.append(100*eval_accuracy(model_lp,   loader_l, device))

    # Plotejar i desar al directori actual
    plt.figure(figsize=(8,5))
    plt.plot(angles, accs_b, '--', label='Sense log-polar', color='blue')
    plt.plot(angles, accs_l, '-',  label='Amb log-polar',   color='red')
    plt.xlabel("Àngle de rotació (°)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Rotació")
    plt.legend(); plt.grid(True)
    out_file = "comparison_logpolar.png"
    plt.savefig(out_file, dpi=150)
    print(f"Figura desada a ./{out_file}")

if __name__ == "__main__":
    main()
