#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os

# -----------------------------
# Configuración general
# -----------------------------
SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 500
print('hello')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# -----------------------------
# Dataset: MNIST en lienzo 128x128, dígito en la esquina superior izquierda
# -----------------------------
class MNISTTopLeftDataset(torch.utils.data.Dataset):
    """
    Cada dígito MNIST (28x28) se coloca en la esquina superior izquierda
    de un lienzo 128x128. De esta forma, al aplicar torch.rot90 la rotación se
    realiza alrededor del origen (0,0), que es lo que requiere la teoría p4.
    """

    def __init__(self, train=True):
        self.dataset = datasets.MNIST(root='./mnist_data', train=train, download=True, transform=None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convertir la imagen a tensor (1, 28, 28)
        img_tensor = transforms.ToTensor()(img)
        # Crear lienzo 128x128 lleno de ceros
        canvas = torch.zeros((1, 128, 128), dtype=torch.float32)
        # Pegar la imagen en la esquina superior izquierda
        canvas[:, :28, :28] = img_tensor
        return canvas, label


def get_data_loaders():
    train_dataset = MNISTTopLeftDataset(train=True)
    test_dataset = MNISTTopLeftDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# -----------------------------
# Capa G-Convolucional para p4
# -----------------------------
class GConv2d(nn.Module):
    """
    Capa de convolución equivarianta al grupo p4.
    Para cada filtro base, se generan sus 4 rotaciones aplicando
    torch.rot90 con k=-r (lo que equivale a aplicar la transformación inversa).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, group_order=4, bias=True):
        super(GConv2d, self).__init__()
        self.group_order = group_order  # p4 => 4 rotaciones
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        # Filtro base (forma: [out_channels, in_channels, k, k])
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size) * 0.01)
        if bias:
            # Un bias para cada combinación (rotación, canal de salida)
            self.bias = nn.Parameter(torch.zeros(out_channels * group_order))
        else:
            self.bias = None

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        outputs = []
        for r in range(self.group_order):
            # Rotamos el filtro con k = -r (rotación inversa)
            rotated_weight = torch.rot90(self.weight, k=-r, dims=[2, 3])
            out_r = F.conv2d(x, rotated_weight, bias=None, stride=self.stride, padding=self.padding)
            outputs.append(out_r)
        # Apilamos las salidas en una nueva dimensión (dim=1): (batch, 4, out_channels, H_out, W_out)
        out = torch.stack(outputs, dim=1)
        if self.bias is not None:
            # Reorganizamos bias a forma: (group_order, out_channels, 1, 1)
            b = self.bias.view(self.group_order, self.out_channels, 1, 1)
            out = out + b
        return out


# -----------------------------
# Group Pooling: pooling sobre la dimensión de rotaciones
# -----------------------------
class GroupPooling(nn.Module):
    def __init__(self, mode='max'):
        super(GroupPooling, self).__init__()
        self.mode = mode

    def forward(self, x):
        # x: (batch, group_order, channels, H, W)
        if self.mode == 'max':
            return x.max(dim=1)[0]  # Colapsa la dimensión de grupo
        elif self.mode == 'avg':
            return x.mean(dim=1)
        else:
            raise ValueError("Pooling mode not supported")


# -----------------------------
# Modelo invariante a rotaciones (p4) usando solo una capa G-conv seguida de group pooling
# -----------------------------
class GCNN_Invariant(nn.Module):
    def __init__(self, num_classes=10):
        super(GCNN_Invariant, self).__init__()
        # Capa G-convolucional: de 1 canal a 16 filtros (con 4 rotaciones)
        self.gconv = GConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, group_order=4)
        self.relu = nn.ReLU(inplace=True)
        # Pooling sobre la dimensión del grupo para obtener invariancia
        self.group_pool = GroupPooling(mode='max')
        # Pooling espacial global: obtenemos un vector de 16 (canales) por imagen
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: (batch, 1, 128, 128)
        out = self.gconv(x)  # -> (batch, 4, 16, 128, 128)
        out = self.relu(out)
        out = self.group_pool(out)  # -> (batch, 16, 128, 128), invariante a rotaciones de p4
        out = self.global_pool(out)  # -> (batch, 16, 1, 1)
        out = out.view(out.size(0), -1)  # -> (batch, 16)
        out = self.fc(out)  # -> (batch, num_classes)
        return out


# -----------------------------
# Funciones de entrenamiento y test
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    return avg_loss, acc


def test_invariance(model, loader):
    """
    Evalúa la precisión del modelo en imágenes rotadas 0°, 90°, 180° y 270°.
    Se usa torch.rot90 para rotar las imágenes alrededor del origen (0,0).
    """
    model.eval()
    angles = [0, 1, 2, 3]  # cada valor multiplica 90°
    results = {}
    with torch.no_grad():
        for r in angles:
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                # Rotar la imagen "r" veces 90° (alrededor de la esquina superior izquierda)
                rotated = torch.rot90(images, k=r, dims=[2, 3])
                outputs = model(rotated)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = 100.0 * correct / total
            angle_deg = r * 90
            results[angle_deg] = acc
            print(f"Rotación {angle_deg}°: Acc = {acc:.2f}%")
    return results


# -----------------------------
# Main
# -----------------------------
def main():
    train_loader, test_loader = get_data_loaders()
    model = GCNN_Invariant(num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Entrenando...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch)

    print("\nEvaluando invariancia p4 en Test:")
    test_invariance(model, test_loader)


if __name__ == '__main__':
    main()
