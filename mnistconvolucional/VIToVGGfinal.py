#!/usr/bin/env python3
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
from torchvision.models import vit_b_16, vgg16

# Parámetros globales
NUM_EPOCHS = 20
ROTACION = True
USE_VIT = True      # Si True se entrenará ViT
USE_VGG = True      # Si True se entrenará VGG
USE_DATA_AUGMENTATION = True
MODEL_ID = 1
SEED = 42
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("holita hoood ")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")

# Transformaciones de imagen
train_transform_con_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(360),  # Rotación aleatoria
    transforms.ToTensor(),
])

train_transform_sin_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.Grayscale(num_output_channels=3),
   transforms.ToTensor(),
])

def add_padding(image, padding=20):
    """Añade padding a la imagen para evitar recortes tras la rotación."""
    return transforms.functional.pad(image, padding, fill=0)

def rotate_image(image, angle):
    """Rota la imagen en un ángulo específico."""
    return transforms.functional.rotate(image, angle, fill=0)

def get_data_loaders():
    # Descargar MNIST (sin transformación inicial para asignar transform luego)
    full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)

    # Dividir en train/val (ejemplo: 40k train, 20k val)
    train_base, val_base = torch.utils.data.random_split(full_train_dataset, [40000, 20000])

    # TRAIN SIN AUGMENTATION
    train_dataset_sin_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_sin_aug)
    train_dataset_sin_aug = torch.utils.data.Subset(train_dataset_sin_aug, train_base.indices)

    # TRAIN CON AUGMENTATION
    train_dataset_con_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_con_aug)
    train_dataset_con_aug = torch.utils.data.Subset(train_dataset_con_aug, train_base.indices)

    # Validación (siempre sin augmentation)
    val_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, val_base.indices)

    # Creación de los loaders
    train_loader_sin_aug = DataLoader(train_dataset_sin_aug, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_con_aug = DataLoader(train_dataset_con_aug, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader_sin_aug, train_loader_con_aug, val_loader, test_loader

def get_model(arch):
    """Retorna el modelo, optimizador y criterio para la arquitectura 'vit' o 'vgg'."""
    if arch == 'vit':
        model = vit_b_16(pretrained=True)
        # Reemplazar la cabeza para 10 clases (MNIST)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif arch == 'vgg':
        model = vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    else:
        raise ValueError("Arquitectura no reconocida. Usa 'vit' o 'vgg'.")
    model = model.to(DEVICE)
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

def test_rotation(model, test_loader):
    model.eval()
    accuracy_by_angle = {}
    with torch.no_grad():
        # Se evalúa desde 0 hasta 360 grados (cada 5°)
        for angle in range(0, 361, 5):
            correct = 0
            total = 0
            for images, labels in test_loader:
                # Se aplica padding, rotación y redimensionamiento
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

def plot_multiple_accuracy_by_angle(accuracy_dicts, labels_list, title, filename):
    plt.figure(figsize=(8, 5))
    for acc_dict, label in zip(accuracy_dicts, labels_list):
        angles = list(acc_dict.keys())
        accuracies = list(acc_dict.values())
        plt.plot(angles, accuracies, marker='o', linestyle='-', label=label)
    plt.xlabel("Ángulo de Rotación (°)")
    plt.ylabel("Precisión (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, format='png', dpi=300)
    print(f"Gráfico guardado como: {filename}")
    plt.close()

def main():
    # Obtener loaders
    train_loader_sin_aug, train_loader_con_aug, val_loader, test_loader = get_data_loaders()

    # Lista para almacenar los resultados de test_rotation y etiquetas para el plot
    accuracy_results = []
    curve_labels = []

    # Entrenar y evaluar ViT
    if USE_VIT:
        if USE_DATA_AUGMENTATION:
            # --- ViT sin data augmentation ---
            model_vit_sin, opt_vit_sin, crit_vit_sin = get_model('vit')
            model_path_vit_sin = f"best_model_vit_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_sin):
                print("Entrenando ViT SIN data augmentation...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_sin, opt_vit_sin, crit_vit_sin, train_loader_sin_aug, epoch)
                    validate(model_vit_sin, crit_vit_sin, val_loader, epoch)
                torch.save(model_vit_sin.state_dict(), model_path_vit_sin)
            else:
                model_vit_sin.load_state_dict(torch.load(model_path_vit_sin))
            acc_vit_sin = test_rotation(model_vit_sin, test_loader)
            accuracy_results.append(acc_vit_sin)
            curve_labels.append("ViT sin Aug")

            # --- ViT con data augmentation ---
            model_vit_con, opt_vit_con, crit_vit_con = get_model('vit')
            model_path_vit_con = f"best_model_vit_con_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_con):
                print("Entrenando ViT CON data augmentation...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_con, opt_vit_con, crit_vit_con, train_loader_con_aug, epoch)
                    validate(model_vit_con, crit_vit_con, val_loader, epoch)
                torch.save(model_vit_con.state_dict(), model_path_vit_con)
            else:
                model_vit_con.load_state_dict(torch.load(model_path_vit_con))
            acc_vit_con = test_rotation(model_vit_con, test_loader)
            accuracy_results.append(acc_vit_con)
            curve_labels.append("ViT con Aug")
        else:
            # Sólo se entrena la versión sin data augmentation para ViT
            model_vit_sin, opt_vit_sin, crit_vit_sin = get_model('vit')
            model_path_vit_sin = f"best_model_vit_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_sin):
                print("Entrenando ViT SIN data augmentation...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_sin, opt_vit_sin, crit_vit_sin, train_loader_sin_aug, epoch)
                    validate(model_vit_sin, crit_vit_sin, val_loader, epoch)
                torch.save(model_vit_sin.state_dict(), model_path_vit_sin)
            else:
                model_vit_sin.load_state_dict(torch.load(model_path_vit_sin))
            acc_vit_sin = test_rotation(model_vit_sin, test_loader)
            accuracy_results.append(acc_vit_sin)
            curve_labels.append("ViT")

    # Entrenar y evaluar VGG
    if USE_VGG:
        if USE_DATA_AUGMENTATION:
            # --- VGG sin data augmentation ---
            model_vgg_sin, opt_vgg_sin, crit_vgg_sin = get_model('vgg')
            model_path_vgg_sin = f"best_model_vgg_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vgg_sin):
                print("Entrenando VGG SIN data augmentation...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vgg_sin, opt_vgg_sin, crit_vgg_sin, train_loader_sin_aug, epoch)
                    validate(model_vgg_sin, crit_vgg_sin, val_loader, epoch)
                torch.save(model_vgg_sin.state_dict(), model_path_vgg_sin)
            else:
                model_vgg_sin.load_state_dict(torch.load(model_path_vgg_sin))
            acc_vgg_sin = test_rotation(model_vgg_sin, test_loader)
            accuracy_results.append(acc_vgg_sin)
            curve_labels.append("VGG sin Aug")

            # --- VGG con data augmentation ---
            model_vgg_con, opt_vgg_con, crit_vgg_con = get_model('vgg')
            model_path_vgg_con = f"best_model_vgg_con_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vgg_con):
                print("Entrenando VGG CON data augmentation...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vgg_con, opt_vgg_con, crit_vgg_con, train_loader_con_aug, epoch)
                    validate(model_vgg_con, crit_vgg_con, val_loader, epoch)
                torch.save(model_vgg_con.state_dict(), model_path_vgg_con)
            else:
                model_vgg_con.load_state_dict(torch.load(model_path_vgg_con))
            acc_vgg_con = test_rotation(model_vgg_con, test_loader)
            accuracy_results.append(acc_vgg_con)
            curve_labels.append("VGG con Aug")
        else:
            # Sólo la versión sin data augmentation para VGG
            model_vgg_sin, opt_vgg_sin, crit_vgg_sin = get_model('vgg')
            model_path_vgg_sin = f"best_model_vgg_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vgg_sin):
                print("Entrenando VGG SIN data augmentation...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vgg_sin, opt_vgg_sin, crit_vgg_sin, train_loader_sin_aug, epoch)
                    validate(model_vgg_sin, crit_vgg_sin, val_loader, epoch)
                torch.save(model_vgg_sin.state_dict(), model_path_vgg_sin)
            else:
                model_vgg_sin.load_state_dict(torch.load(model_path_vgg_sin))
            acc_vgg_sin = test_rotation(model_vgg_sin, test_loader)
            accuracy_results.append(acc_vgg_sin)
            curve_labels.append("VGG")

    # Graficar todas las curvas en un único gráfico
    if ROTACION:
        plot_title = "Precisión vs Rotación de los Modelos"
        plot_filename = f"combined_accuracy_{MODEL_ID}.png"
        plot_multiple_accuracy_by_angle(accuracy_results, curve_labels, plot_title, plot_filename)
    else:
        print("No se realizó la evaluación con rotación.")

if __name__ == '__main__':
    main()
