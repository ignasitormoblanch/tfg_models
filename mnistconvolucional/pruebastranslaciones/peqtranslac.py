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

NUM_EPOCHS = 40
TRANSLACION = True  # Se evaluará translación en el test si es True
USE_VIT = False  # Si es False se usará VGG16
USE_DATA_AUGMENTATION = False  # Si es False, no se aplicará translación en entrenamiento
MODEL_ID = 10
SEED = 42
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f'best_modelfila{MODEL_ID}'
print('holita hooo23')

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")

# Transformación para entrenamiento con augmentation (traslación pequeña)
train_transform_con_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    # Se aplica traslación muy pequeña: hasta el 2% de la dimensión en cada eje.
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    transforms.ToTensor(),
])

# Transformación básica sin augmentation
train_transform_sin_aug = transforms.Compose([
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

def add_padding(image, padding=20):
    """Añade padding a la imagen para evitar recortes tras la translación."""
    return transforms.functional.pad(image, padding, fill=0)

def translate_image(image, offset):
    """
    Traduce la imagen horizontalmente en 'offset' píxeles.
    Se mantiene sin rotación, escalado o shear.
    """
    return transforms.functional.affine(image, angle=0, translate=(offset, 0), scale=1, shear=0, fill=0)

def get_data_loaders():
    # Descargar MNIST sin transformación para asignar la transformación deseada más adelante.
    full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)

    # Dividir en train/val (por ejemplo 40k train, 20k val)
    train_base, val_base = torch.utils.data.random_split(full_train_dataset, [40000, 20000])

    # ---- TRAIN SIN AUG ----
    train_dataset_sin_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_sin_aug)
    train_dataset_sin_aug = torch.utils.data.Subset(train_dataset_sin_aug, train_base.indices)

    # ---- TRAIN CON AUG (solo se aplica si USE_DATA_AUGMENTATION es True) ----
    if USE_DATA_AUGMENTATION:
        train_dataset_con_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_con_aug)
    else:
        # Si no se quiere augmentation, se utiliza la misma transformación básica.
        train_dataset_con_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_sin_aug)
    train_dataset_con_aug = torch.utils.data.Subset(train_dataset_con_aug, train_base.indices)

    # Para validación, se usa la transformación básica.
    val_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, val_base.indices)

    # Creación de los loaders.
    train_loader_sin_aug = DataLoader(train_dataset_sin_aug, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_con_aug = DataLoader(train_dataset_con_aug, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader_sin_aug, train_loader_con_aug, val_loader, test_loader

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
    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
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

def test_translation(model, test_loader):
    """
    Evalúa la robustez del modelo a translaciones horizontales.
    Se aplica una traslación (offset en píxeles) a cada imagen y se mide la precisión.
    """
    model.eval()
    accuracy_by_offset = {}
    with torch.no_grad():
        # Se evalúa desplazando la imagen de 0 a 10 píxeles
        for offset in range(0, 30):
            correct = 0
            total = 0
            for images, labels in test_loader:
                # Se aplica padding, traslación y redimensionamiento a 224x224
                images = torch.stack([
                    transforms.functional.resize(
                        translate_image(add_padding(img, padding=20), offset),
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
            accuracy_by_offset[offset] = acc
            if offset % 2 == 0:
                print(f"Desplazamiento {offset} px - Accuracy: {acc:.2f}%")
    return accuracy_by_offset

def test_no_translation(model, test_loader):
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

def plot_accuracy_by_offset(accuracy_dict, title, filename):
    offsets = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    plt.figure(figsize=(8, 5))
    plt.plot(offsets, accuracies, marker='o', linestyle='-')
    plt.xlabel("Desplazamiento Horizontal (px)")
    plt.ylabel("Precisión (%)")
    plt.title(title)
    plt.grid(True)

    # Guardar el gráfico como imagen
    plt.savefig(filename, format='png', dpi=300)
    print(f"Gráfico guardado como: {filename}")
    plt.close()

def main():
    (train_loader_sin_aug,
     train_loader_con_aug,
     val_loader,
     test_loader) = get_data_loaders()

    # Modelo sin_aug
    model_sin_aug, optimizer_sin_aug, criterion_sin_aug = get_model()
    # Modelo con_aug
    model_con_aug, optimizer_con_aug, criterion_con_aug = get_model()

    # Entrenamiento sin_aug
    MODEL_PATH_SIN = f"best_model_sin_aug{MODEL_ID}.pth"
    if not os.path.exists(MODEL_PATH_SIN):
        print('Empezamos el entrenamiento sin augmentation...')
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(model_sin_aug, optimizer_sin_aug, criterion_sin_aug, train_loader_sin_aug, epoch)
            validate(model_sin_aug, criterion_sin_aug, val_loader, epoch)
        torch.save(model_sin_aug.state_dict(), MODEL_PATH_SIN)
    else:
        model_sin_aug.load_state_dict(torch.load(MODEL_PATH_SIN))

    # Entrenamiento con_aug (aplica augmentation solo si USE_DATA_AUGMENTATION es True)
    MODEL_PATH_CON = f"best_model_con_aug{MODEL_ID}.pth"
    if not os.path.exists(MODEL_PATH_CON):
        if USE_DATA_AUGMENTATION:
            print('Empezamos el entrenamiento con augmentation (traslación)...')
        else:
            print('Empezamos el entrenamiento sin augmentation (sin translación en training)...')
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(model_con_aug, optimizer_con_aug, criterion_con_aug, train_loader_con_aug, epoch)
            validate(model_con_aug, criterion_con_aug, val_loader, epoch)
        torch.save(model_con_aug.state_dict(), MODEL_PATH_CON)
    else:
        model_con_aug.load_state_dict(torch.load(MODEL_PATH_CON))

    # Evaluación en test: se evalúa la robustez a pequeñas translaciones (si TRANSLACION es True)
    if TRANSLACION:
        print("Evaluando modelo sin_aug con translación en test...")
        accuracy_sin_aug = test_translation(model_sin_aug, test_loader)
        print("Evaluando modelo con_aug con translación en test...")
        accuracy_con_aug = test_translation(model_con_aug, test_loader)
        plot_accuracy_by_offset(accuracy_sin_aug, "Precisión del modelo sin_aug vs. Translación", f"accuracy_sin_augVIT{MODEL_ID}.png")
        plot_accuracy_by_offset(accuracy_con_aug, "Precisión del modelo con_aug vs. Translación", f"accuracy_con_augVIT{MODEL_ID}.png")
    else:
        test_no_translation(model_sin_aug, test_loader)
        test_no_translation(model_con_aug, test_loader)

if __name__ == '__main__':
    main()
