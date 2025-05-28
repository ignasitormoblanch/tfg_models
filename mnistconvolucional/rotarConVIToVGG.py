#!/usr/bin/env python3
from datetime import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, vgg16

NUM_EPOCHS = 6
ROTACION = True
USE_VIT = False  # Si es False se usará VGG16
USE_DATA_AUGMENTATION=True
USE_FULL_ROTATION_ON_THE_FLY = True
rotacionangulo=10

MODEL_ID = 1
SEED = 42
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f'best_modelfila{MODEL_ID}'

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")
print(MODEL_ID)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Para validación y test se usa la transformación básica.


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


class OnTheFlyFullRotationDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        self.angles = list(range(360))  # Rotaciones de 0° a 359°

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # Aquí img será un PIL Image, ya que no se le asignó transform
        img = self.transform(img)  # Se aplica train_transform (que espera PIL Image)
        rotated_imgs = torch.stack([transforms.functional.rotate(img, angle) for angle in self.angles])
        return rotated_imgs, label

def add_padding(image, padding=20):
    """Añade padding a la imagen para evitar recortes tras la rotación."""
    return transforms.functional.pad(image, padding, fill=0)

def rotate_image(image, angle):
    """Rota la imagen en un ángulo específico."""
    return transforms.functional.rotate(image, angle, fill=0)


def get_data_loaders():
    # Descargamos MNIST sin transform para el conjunto de entrenamiento
    full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
    # Para test se usa el transform básico
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)

    # Dividimos el conjunto de entrenamiento en train y validación
    train_subset, val_subset = torch.utils.data.random_split(full_train_dataset, [40000, 20000])

    # Para el conjunto de validación asignamos el transform básico (ya que no se usa la clase personalizada)
    val_subset.dataset.transform = val_test_transform

    # Si se usa on the fly, envolvemos el subconjunto de entrenamiento sin asignarle previamente un transform
    if USE_FULL_ROTATION_ON_THE_FLY:
        train_dataset = OnTheFlyFullRotationDataset(train_subset, train_transform)
    else:
        # Si no se usa, se asigna el transform al subconjunto (no recomendado para esta configuración)
        train_subset.dataset.transform = train_transform
        train_dataset = train_subset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def get_model():
    if USE_VIT:
        model = vit_b_16(pretrained=True)
        # Reemplazamos la cabeza para 10 clases (MNIST)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    else:
        model = vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001) if USE_VIT else optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train_one_epoch(model, optimizer, criterion, train_loader, epoch):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    print(f'epoca {epoch}')
    for images, labels in train_loader:
        # Si se usa on the fly, 'images' tendrá forma (B, 360, C, H, W)
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if images.dim() == 5:  # Comprueba si se generaron 360 copias por imagen
            B, R, C, H, W = images.shape  # R debe ser 360
            images = images.view(B * R, C, H, W)  # Aplana para procesar B*360 imágenes
            # Repite cada etiqueta 360 veces
            labels = labels.unsqueeze(1).repeat(1, R).view(B * R)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    acc = 100. * correct / total
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")
    return avg_loss, acc

def validate(model, criterion, val_loader, epoch):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total
    acc = 100. * correct / total
    print(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")
    return avg_loss, acc

def test_rotation(model, test_loader):
    model.eval()
    accuracy_by_angle = {}
    with torch.no_grad():
        # Recorremos cada ángulo de 0 a 360 grados (incremento de 1)
        for angle in range(0, 361, rotacionangulo):
            correct = 0
            total = 0
            for images, labels in test_loader:
                # Aplicamos en CPU: padding, rotación y redimensionamiento para cumplir con 224x224
                images = torch.stack([
                    transforms.functional.resize(
                        rotate_image(add_padding(img, padding=20), angle),
                        (224, 224)
                    )
                    for img in images
                ])
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = 100. * correct / total
            accuracy_by_angle[angle] = acc
            if angle % 30 == 0:
                print(f"Ángulo {angle}° - Accuracy: {acc:.2f}%")
    return accuracy_by_angle

def test_no_rotation(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def main():
    train_loader, val_loader, test_loader = get_data_loaders()
    model, optimizer, criterion = get_model()

    # Entrenamiento (si no se ha guardado previamente el modelo)
    if not os.path.exists(MODEL_PATH):
        print("Iniciando entrenamiento...")
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(model, optimizer, criterion, train_loader, epoch)
            validate(model, criterion, val_loader, epoch)
            # Guardamos el modelo tras cada época (o se puede implementar lógica para guardar el mejor)
            torch.save(model.state_dict(), MODEL_PATH)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Modelo cargado con los mejores pesos.")

    # Evaluación / Testing
    if ROTACION:
        print("Iniciando evaluación con rotación...")
        accuracy_by_angle = test_rotation(model, test_loader)
        # Graficamos los resultados
        plt.figure(figsize=(10, 5))
        plt.plot(list(accuracy_by_angle.keys()), list(accuracy_by_angle.values()),
                 marker='o', linestyle='-')
        plt.xlabel('Ángulo de Rotación (grados)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy del modelo según la rotación de las imágenes')
        plt.grid(True)
        plot_filename = "accuracy_by_rotation.png"
        plt.savefig(plot_filename)
        print(f"Gráfica guardada en: {plot_filename}")
    else:
        test_no_rotation(model, test_loader)

if __name__ == '__main__':
    main()
