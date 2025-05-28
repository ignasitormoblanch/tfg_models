#!/usr/bin/env python3
"""
Entrenamiento en MNIST con VGG-16 usando la magnitud logarítmica del FFT
para mejorar la invariancia a rotaciones (sin data-augmentation).
"""
from datetime import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import vgg16

NUM_EPOCHS = 20
BATCH_SIZE = 32
SEED = 42
MODEL_ID = 2                # cambia si quieres guardar modelos aparte
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  •  CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")


class FourierMagnitudeTransform:
    """
    Convierte un tensor [1,H,W] de píxeles ∈[0,1] en su espectro
    de magnitud logarítmica, normalizado a [0,1] y replicado en 3 canales.
    """
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [1,H,W]  (ya escalada 0-1 por ToTensor)
        fft    = torch.fft.fft2(img)
        fft    = self._fftshift(fft)
        mag    = torch.abs(fft)
        mag    = torch.log1p(mag)          # log(1+|F|)
        mag    = mag / mag.max()           # normaliza a [0,1]
        mag3   = mag.repeat(3, 1, 1)       # VGG espera 3 canales
        return mag3.float()

    @staticmethod
    def _fftshift(x: torch.Tensor) -> torch.Tensor:
        """Desplaza el cero de la FFT al centro (equiv. a np.fft.fftshift)."""
        h, w = x.shape[-2:]
        return torch.roll(x, shifts=(h // 2, w // 2), dims=(-2, -1))


base_resize = transforms.Resize((224, 224))
to_tensor   = transforms.ToTensor()        # escala a [0,1]
fourier_tf  = FourierMagnitudeTransform()

train_val_transform = transforms.Compose([
    base_resize,
    transforms.Grayscale(num_output_channels=1),
    to_tensor,
    fourier_tf,
    # (opcional) normalizar con stats ImageNet, aunque el rango ya es [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std =[0.5, 0.5, 0.5]),
])

test_transform = train_val_transform    # idéntico


def get_data_loaders():
    # Descarga sin transform para poder dividir en train/val
    full_train = datasets.MNIST(root='mnist_data', train=True,
                                download=True, transform=None)

    train_idx, val_idx = torch.utils.data.random_split(
        range(len(full_train)), [40_000, 20_000], generator=torch.Generator().manual_seed(SEED)
    )

    # Crea datasets con la transformación Fourier
    train_ds = datasets.MNIST(root='mnist_data', train=True, download=True,
                              transform=train_val_transform)
    val_ds   = datasets.MNIST(root='mnist_data', train=True, download=True,
                              transform=train_val_transform)
    train_ds = torch.utils.data.Subset(train_ds, train_idx.indices)
    val_ds   = torch.utils.data.Subset(val_ds,   val_idx.indices)

    test_ds  = datasets.MNIST(root='mnist_data', train=False, download=True,
                              transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def get_vgg():
    model = vgg16(pretrained=True)
    # Congelamos las capas convolutivas para rapidez (opcional)
    for p in model.features.parameters():
        p.requires_grad = False
    # Sustituye la última capa para 10 clases (MNIST)
    model.classifier[6] = nn.Linear(4096, 10)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def train_one_epoch(model, optimizer, criterion, loader, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    acc = 100. * correct / total
    print(f"Época {epoch:2d}  •  Train-loss: {avg_loss:.4f}  •  Acc: {acc:.2f}%")
    return avg_loss, acc

def evaluate(model, criterion, loader, epoch, split="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = running_loss / total
    acc = 100. * correct / total
    print(f"Época {epoch:2d}  •  {split}-loss: {avg_loss:.4f}  •  {split}-Acc: {acc:.2f}%")
    return avg_loss, acc

pad_img       = lambda img: transforms.functional.pad(img, 20, 0)
rotate_img    = lambda img, a: transforms.functional.rotate(img, a, fill=0)
resize_tensor = transforms.Compose([transforms.Resize((224, 224))])

def test_rotation(model, test_loader):
    model.eval()
    acc_by_angle = {}
    with torch.no_grad():
        for angle in range(0, 361, 5):
            correct = total = 0
            for images, labels in test_loader:
                # images aún son PIL-Images, re-aplicamos padding + rotación
                rot_images = torch.stack([
                    train_val_transform(
                        rotate_img(pad_img(img), angle)
                    ) for img in images
                ])
                rot_images, labels = rot_images.to(DEVICE), labels.to(DEVICE)
                preds = model(rot_images).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
            acc_by_angle[angle] = 100. * correct / total
            if angle % 30 == 0:
                print(f"Ángulo {angle:3d}°  •  Acc: {acc_by_angle[angle]:.2f}%")
    return acc_by_angle

def plot_accuracy_curve(acc_dict, title, filename):
    angles = list(acc_dict.keys())
    accs   = list(acc_dict.values())
    plt.figure(figsize=(8,5))
    plt.plot(angles, accs, marker='o')
    plt.xlabel("Ángulo de rotación (°)")
    plt.ylabel("Precisión (%)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=300, format='png')
    print(f"Gráfico guardado en '{filename}'")
    plt.close()


def main():
    train_loader, val_loader, test_loader = get_data_loaders()
    model, opt, crit = get_vgg()
    best_val_acc = 0.0
    chk_path = f"best_vgg_fft_{MODEL_ID}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, opt, crit, train_loader, epoch)
        _, val_acc = evaluate(model, crit, val_loader, epoch, split="Val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), chk_path)
            print(f"★ Nuevo mejor modelo: {best_val_acc:.2f}%  →  guardado\n")

    # Carga el mejor modelo para test
    model.load_state_dict(torch.load(chk_path))
    print("\n──────────────── Evaluación con rotaciones ────────────────")
    acc_by_angle = test_rotation(model, test_loader)
    plot_accuracy_curve(acc_by_angle,
                        "VGG-16 + FFT: Precisión vs Rotación",
                        f"accuracy_fft_{MODEL_ID}.png")

if __name__ == "__main__":
    main()
