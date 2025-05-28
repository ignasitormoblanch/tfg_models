import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """CNN senzilla (2×conv + 2×FC) per a MNIST."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.activations = None  # per emmagatzemar la sortida de conv2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        self.activations = x            # guardem activacions
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_one_epoch(model, loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()


def get_digit_example(loader, digit: int = 3):
    """Retorna la primera imatge del dígit indicat."""
    for img, lbl in loader:
        if lbl.item() == digit:
            return img
    raise ValueError(f"No s'ha trobat el dígit {digit} al loader.")


def plot_and_save(original, translated, act_orig, act_trans,
                  fname="activacions_trans1px.png"):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(original.squeeze(), cmap="gray")
    ax[0, 0].set_title("Original"); ax[0, 0].axis("off")
    ax[0, 1].imshow(translated.squeeze(), cmap="gray")
    ax[0, 1].set_title("Translació +1 píxel"); ax[0, 1].axis("off")
    ax[1, 0].imshow(act_orig, cmap="hot")
    ax[1, 0].set_title("Activació Conv2 (orig.)"); ax[1, 0].axis("off")
    ax[1, 1].imshow(act_trans, cmap="hot")
    ax[1, 1].set_title("Activació Conv2 (trans.)"); ax[1, 1].axis("off")
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"[INFO] Figura guardada com «{fname}»")
    return Path(fname)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset MNIST sense augmentació
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tfm)
    testset  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # Model + entrenament ràpid (1 època)
    model = SimpleCNN().to(device)
    train_one_epoch(model, trainloader, device)

    # Exemple dígit «3»
    original = get_digit_example(testloader, 3).to(device)
    # Translació d'1 píxel cap a la dreta (dim=3 -> eix X)
    translated = torch.roll(original, shifts=(0, 1), dims=(2, 3)).to(device)

    # Activacions
    model.eval()
    with torch.no_grad():
        _ = model(original)
        act_orig = model.activations.squeeze(0).cpu().mean(dim=0).numpy()
        _ = model(translated)
        act_trans = model.activations.squeeze(0).cpu().mean(dim=0).numpy()

    # Figura + desar
    out_file = plot_and_save(original.cpu(), translated.cpu(),
                             act_orig, act_trans)

    # Si estem a Google Colab, ofereix descàrrega
    if "google.colab" in sys.modules:
        from google.colab import files  # type: ignore
        files.download(str(out_file))

if __name__ == "__main__":
    main()