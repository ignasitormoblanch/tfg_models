#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch.utils.data import TensorDataset, DataLoader


# Parámetros globales
MODEL_ID = 1
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Semillas para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")

# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def add_padding(image, padding=20):
    return transforms.functional.pad(image, padding, fill=0)

def rotate_image(image, angle):
    return transforms.functional.rotate(image, angle, fill=0)

def get_data_loaders():
    full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=test_transform)

    train_base, val_base = torch.utils.data.random_split(full_train_dataset, [40000, 20000])
    train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform)
    train_dataset = torch.utils.data.Subset(train_dataset, train_base.indices)
    val_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=test_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, val_base.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

def get_model():
    model = vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 10)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train_one_epoch(model, optimizer, criterion, loader, epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = 100. * correct / total
    print(f"Epoch {epoch} - Train Loss: {running_loss / total:.4f}, Acc: {acc:.2f}%")

def validate(model, criterion, loader, epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100. * correct / total
    print(f"Epoch {epoch} - Val Loss: {running_loss / total:.4f}, Acc: {acc:.2f}%")

def test_rotation_by_digit(model, test_dataset):
    model.eval()
    accuracy_by_digit = {i: {} for i in range(10)}
    digits_data = {i: [] for i in range(10)}

    # Agrupar imágenes por dígito
    for img, label in test_dataset:
        digits_data[label].append(img)

    with torch.no_grad():
        for digit in range(10):
            print(f"\nEvaluando dígito {digit}...")
            images_digit = digits_data[digit]
            for angle in range(0, 361, 5):
                # Procesar en batches para evitar OOM
                rotated_imgs = [
                    transforms.functional.resize(
                        rotate_image(add_padding(img, padding=20), angle),
                        (224, 224)
                    ) for img in images_digit
                ]
                rotated_imgs = torch.stack(rotated_imgs)
                labels = torch.full((len(rotated_imgs),), digit, dtype=torch.long)

                dataset = TensorDataset(rotated_imgs, labels)
                loader = DataLoader(dataset, batch_size=32, shuffle=False)

                correct = 0
                total = 0
                for batch_imgs, batch_labels in loader:
                    batch_imgs = batch_imgs.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    outputs = model(batch_imgs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.size(0)

                acc = 100. * correct / total
                accuracy_by_digit[digit][angle] = acc

                if angle % 60 == 0:
                    print(f"Ángulo {angle}° - Acc: {acc:.2f}%")

    return accuracy_by_digit
def plot_digit_rotation_curves(accuracy_by_digit, filename):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for digit in range(10):
        angles = list(accuracy_by_digit[digit].keys())
        accuracies = list(accuracy_by_digit[digit].values())
        plt.plot(angles, accuracies, label=f'Dígito {digit}', color=cmap(digit))
    plt.xlabel("Ángulo de Rotación (°)")
    plt.ylabel("Precisión (%)")
    plt.title("Precisión vs Rotación por Dígito (VGG sin Augmentación)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='png', dpi=300)
    print(f"Gráfico guardado como: {filename}")
    plt.close()

def main():
    train_loader, val_loader, test_loader = get_data_loaders()
    test_dataset = test_loader.dataset

    model, optimizer, criterion = get_model()
    model_path = f"best_model_vgg_sin_aug_{MODEL_ID}.pth"

    if not os.path.exists(model_path):
        print("Entrenando modelo VGG sin data augmentation...")
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(model, optimizer, criterion, train_loader, epoch)
            validate(model, criterion, val_loader, epoch)
        torch.save(model.state_dict(), model_path)
    else:
        print("Cargando modelo previamente entrenado...")
        model.load_state_dict(torch.load(model_path))

    # Evaluar por dígito
    accuracy_by_digit = test_rotation_by_digit(model, test_dataset)

    # Graficar resultados
    plot_digit_rotation_curves(accuracy_by_digit, f"rotacion_por_digito_vgg_{MODEL_ID}.png")

if __name__ == '__main__':
    main()
