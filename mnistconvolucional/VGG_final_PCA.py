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
from torch.utils.data import DataLoader, Subset
from torchvision.models import vit_b_16, vgg16
from sklearn.decomposition import PCA

# =======================
# Parámetros globales
# =======================
NUM_EPOCHS = 20
ROTACION = True
USE_VIT = True  # Si True se entrenará ViT
USE_VGG = True  # Si True se entrenará VGG
USE_DATA_AUGMENTATION = True
USE_PCA = True  # Si True se aplica PCA a las imágenes
MODEL_ID = 1
SEED = 42
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCA_N_COMPONENTS = 500  # Número de componentes a conservar en el PCA
PCA_SAMPLE_SIZE = 10000  # Número de imágenes del train para ajustar el PCA

print("holita hoood")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")

# =======================
# Transformaciones de imagen
# =======================
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


# =======================
# Transformación PCA: Aplica PCA a la imagen reconstruyéndola a partir de sus componentes
# =======================
class ApplyPCA(object):
    def __init__(self, pca):
        self.pca = pca

    def __call__(self, tensor):
        # tensor: (3, H, W). Como es imagen en gris repetida en 3 canales, usamos el primero.
        image = tensor[0]  # (H, W)
        flat = image.flatten().numpy()  # vector de H*W
        transformed = self.pca.transform(flat.reshape(1, -1))
        reconstructed = self.pca.inverse_transform(transformed)
        reconstructed = reconstructed.reshape(image.shape)
        # Replicamos a 3 canales
        reconstructed = np.stack([reconstructed, reconstructed, reconstructed], axis=0)
        return torch.tensor(reconstructed, dtype=tensor.dtype)


# =======================
# Función para calcular PCA a partir de un dataset (se toma una muestra)
# =======================
def compute_pca_from_dataset(dataset, n_components, sample_size):
    loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
    images, _ = next(iter(loader))  # images: (sample_size, 3, H, W)
    # Usamos el primer canal (ya que las 3 son iguales en MNIST procesado)
    images_np = images[:, 0, :, :].numpy()  # (sample_size, H, W)
    flat_images = images_np.reshape(images_np.shape[0], -1)  # (sample_size, H*W)
    pca = PCA(n_components=n_components)
    pca.fit(flat_images)
    return pca


# =======================
# Función para crear los DataLoaders y, si USE_PCA es True, crear versiones PCA de los datasets
# =======================
def get_data_loaders():
    # Descarga MNIST sin transformación para dividir en train/val
    full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)

    # Dividir train: 40k para training y 20k para validación
    train_base, val_base = torch.utils.data.random_split(full_train_dataset, [40000, 20000])

    # Datasets originales
    train_dataset_sin_aug = datasets.MNIST(root='mnist_data', train=True, download=True,
                                           transform=train_transform_sin_aug)
    train_dataset_sin_aug = Subset(train_dataset_sin_aug, train_base.indices)

    train_dataset_con_aug = datasets.MNIST(root='mnist_data', train=True, download=True,
                                           transform=train_transform_con_aug)
    train_dataset_con_aug = Subset(train_dataset_con_aug, train_base.indices)

    val_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)
    val_dataset = Subset(val_dataset, val_base.indices)

    # DataLoaders originales
    train_loader_sin_aug = DataLoader(train_dataset_sin_aug, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_con_aug = DataLoader(train_dataset_con_aug, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Si USE_PCA es True, creamos nuevos datasets con la transformación PCA aplicada
    if USE_PCA:
        # Ajustamos PCA usando una muestra de cada dataset de training
        pca_sin = compute_pca_from_dataset(train_dataset_sin_aug, PCA_N_COMPONENTS, PCA_SAMPLE_SIZE)
        pca_con = compute_pca_from_dataset(train_dataset_con_aug, PCA_N_COMPONENTS, PCA_SAMPLE_SIZE)

        # Definir nuevos transforms que incluyan la reconstrucción PCA
        pca_transform_sin_aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ApplyPCA(pca_sin)
        ])
        pca_transform_con_aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            ApplyPCA(pca_con)
        ])
        pca_val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ApplyPCA(pca_sin)  # Usamos el PCA de sin_aug para validación y test
        ])
        pca_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ApplyPCA(pca_sin)
        ])

        # Crear nuevos datasets PCA
        train_dataset_sin_aug_pca = datasets.MNIST(root='mnist_data', train=True, download=True,
                                                   transform=pca_transform_sin_aug)
        train_dataset_sin_aug_pca = Subset(train_dataset_sin_aug_pca, train_base.indices)

        train_dataset_con_aug_pca = datasets.MNIST(root='mnist_data', train=True, download=True,
                                                   transform=pca_transform_con_aug)
        train_dataset_con_aug_pca = Subset(train_dataset_con_aug_pca, train_base.indices)

        val_dataset_pca = datasets.MNIST(root='mnist_data', train=True, download=True, transform=pca_val_transform)
        val_dataset_pca = Subset(val_dataset_pca, val_base.indices)

        test_dataset_pca = datasets.MNIST(root='mnist_data', train=False, download=True, transform=pca_test_transform)

        # DataLoaders PCA
        train_loader_sin_aug_pca = DataLoader(train_dataset_sin_aug_pca, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_con_aug_pca = DataLoader(train_dataset_con_aug_pca, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_pca = DataLoader(val_dataset_pca, batch_size=BATCH_SIZE, shuffle=False)
        test_loader_pca = DataLoader(test_dataset_pca, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_loader_sin_aug_pca = None
        train_loader_con_aug_pca = None
        val_loader_pca = None
        test_loader_pca = None

    return (train_loader_sin_aug, train_loader_con_aug, val_loader, test_loader,
            train_loader_sin_aug_pca, train_loader_con_aug_pca, val_loader_pca, test_loader_pca)


# =======================
# Funciones auxiliares (rotación, modelo, entrenamiento, validación, test)
# =======================
def add_padding(image, padding=20):
    return transforms.functional.pad(image, padding, fill=0)


def rotate_image(image, angle):
    return transforms.functional.rotate(image, angle, fill=0)


def get_model(arch):
    if arch == 'vit':
        model = vit_b_16(pretrained=True)
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
        for angle in range(0, 361, 5):
            correct = 0
            total = 0
            for images, labels in test_loader:
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


# =======================
# Función principal
# =======================
def main():
    loaders = get_data_loaders()
    (train_loader_sin_aug, train_loader_con_aug, val_loader, test_loader,
     train_loader_sin_aug_pca, train_loader_con_aug_pca, val_loader_pca, test_loader_pca) = loaders

    # Listas para almacenar resultados (accuracy vs. rotación)
    accuracy_results_original = []
    curve_labels_original = []
    accuracy_results_pca = []
    curve_labels_pca = []

    # --- Experimentos con imágenes originales ---
    if USE_VIT:
        if USE_DATA_AUGMENTATION:
            # ViT sin augmentation (original)
            model_vit_sin, opt_vit_sin, crit_vit_sin = get_model('vit')
            model_path_vit_sin = f"best_model_vit_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_sin):
                print("Entrenando ViT SIN data augmentation (original)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_sin, opt_vit_sin, crit_vit_sin, train_loader_sin_aug, epoch)
                    validate(model_vit_sin, crit_vit_sin, val_loader, epoch)
                torch.save(model_vit_sin.state_dict(), model_path_vit_sin)
            else:
                model_vit_sin.load_state_dict(torch.load(model_path_vit_sin))
            acc_vit_sin = test_rotation(model_vit_sin, test_loader)
            accuracy_results_original.append(acc_vit_sin)
            curve_labels_original.append("ViT sin Aug (Original)")

            # ViT con augmentation (original)
            model_vit_con, opt_vit_con, crit_vit_con = get_model('vit')
            model_path_vit_con = f"best_model_vit_con_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_con):
                print("Entrenando ViT CON data augmentation (original)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_con, opt_vit_con, crit_vit_con, train_loader_con_aug, epoch)
                    validate(model_vit_con, crit_vit_con, val_loader, epoch)
                torch.save(model_vit_con.state_dict(), model_path_vit_con)
            else:
                model_vit_con.load_state_dict(torch.load(model_path_vit_con))
            acc_vit_con = test_rotation(model_vit_con, test_loader)
            accuracy_results_original.append(acc_vit_con)
            curve_labels_original.append("ViT con Aug (Original)")
        else:
            model_vit_sin, opt_vit_sin, crit_vit_sin = get_model('vit')
            model_path_vit_sin = f"best_model_vit_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_sin):
                print("Entrenando ViT SIN data augmentation (original)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_sin, opt_vit_sin, crit_vit_sin, train_loader_sin_aug, epoch)
                    validate(model_vit_sin, crit_vit_sin, val_loader, epoch)
                torch.save(model_vit_sin.state_dict(), model_path_vit_sin)
            else:
                model_vit_sin.load_state_dict(torch.load(model_path_vit_sin))
            acc_vit_sin = test_rotation(model_vit_sin, test_loader)
            accuracy_results_original.append(acc_vit_sin)
            curve_labels_original.append("ViT (Original)")

    if USE_VGG:
        if USE_DATA_AUGMENTATION:
            # VGG sin augmentation (original)
            model_vgg_sin, opt_vgg_sin, crit_vgg_sin = get_model('vgg')
            model_path_vgg_sin = f"best_model_vgg_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vgg_sin):
                print("Entrenando VGG SIN data augmentation (original)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vgg_sin, opt_vgg_sin, crit_vgg_sin, train_loader_sin_aug, epoch)
                    validate(model_vgg_sin, crit_vgg_sin, val_loader, epoch)
                torch.save(model_vgg_sin.state_dict(), model_path_vgg_sin)
            else:
                model_vgg_sin.load_state_dict(torch.load(model_path_vgg_sin))
            acc_vgg_sin = test_rotation(model_vgg_sin, test_loader)
            accuracy_results_original.append(acc_vgg_sin)
            curve_labels_original.append("VGG sin Aug (Original)")

            # VGG con augmentation (original)
            model_vgg_con, opt_vgg_con, crit_vgg_con = get_model('vgg')
            model_path_vgg_con = f"best_model_vgg_con_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vgg_con):
                print("Entrenando VGG CON data augmentation (original)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vgg_con, opt_vgg_con, crit_vgg_con, train_loader_con_aug, epoch)
                    validate(model_vgg_con, crit_vgg_con, val_loader, epoch)
                torch.save(model_vgg_con.state_dict(), model_path_vgg_con)
            else:
                model_vgg_con.load_state_dict(torch.load(model_path_vgg_con))
            acc_vgg_con = test_rotation(model_vgg_con, test_loader)
            accuracy_results_original.append(acc_vgg_con)
            curve_labels_original.append("VGG con Aug (Original)")
        else:
            model_vgg_sin, opt_vgg_sin, crit_vgg_sin = get_model('vgg')
            model_path_vgg_sin = f"best_model_vgg_sin_aug_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vgg_sin):
                print("Entrenando VGG SIN data augmentation (original)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vgg_sin, opt_vgg_sin, crit_vgg_sin, train_loader_sin_aug, epoch)
                    validate(model_vgg_sin, crit_vgg_sin, val_loader, epoch)
                torch.save(model_vgg_sin.state_dict(), model_path_vgg_sin)
            else:
                model_vgg_sin.load_state_dict(torch.load(model_path_vgg_sin))
            acc_vgg_sin = test_rotation(model_vgg_sin, test_loader)
            accuracy_results_original.append(acc_vgg_sin)
            curve_labels_original.append("VGG (Original)")

    # --- Experimentos con imágenes procesadas con PCA ---
    if USE_PCA:
        if USE_DATA_AUGMENTATION:
            # ViT sin augmentation (PCA)
            model_vit_sin_pca, opt_vit_sin_pca, crit_vit_sin_pca = get_model('vit')
            model_path_vit_sin_pca = f"best_model_vit_sin_aug_pca_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_sin_pca):
                print("Entrenando ViT SIN data augmentation (PCA)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_sin_pca, opt_vit_sin_pca, crit_vit_sin_pca, train_loader_sin_aug_pca,
                                    epoch)
                    validate(model_vit_sin_pca, crit_vit_sin_pca, val_loader_pca, epoch)
                torch.save(model_vit_sin_pca.state_dict(), model_path_vit_sin_pca)
            else:
                model_vit_sin_pca.load_state_dict(torch.load(model_path_vit_sin_pca))
            acc_vit_sin_pca = test_rotation(model_vit_sin_pca, test_loader_pca)
            accuracy_results_pca.append(acc_vit_sin_pca)
            curve_labels_pca.append("ViT sin Aug (PCA)")

            # ViT con augmentation (PCA)
            model_vit_con_pca, opt_vit_con_pca, crit_vit_con_pca = get_model('vit')
            model_path_vit_con_pca = f"best_model_vit_con_aug_pca_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_con_pca):
                print("Entrenando ViT CON data augmentation (PCA)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_con_pca, opt_vit_con_pca, crit_vit_con_pca, train_loader_con_aug_pca,
                                    epoch)
                    validate(model_vit_con_pca, crit_vit_con_pca, val_loader_pca, epoch)
                torch.save(model_vit_con_pca.state_dict(), model_path_vit_con_pca)
            else:
                model_vit_con_pca.load_state_dict(torch.load(model_path_vit_con_pca))
            acc_vit_con_pca = test_rotation(model_vit_con_pca, test_loader_pca)
            accuracy_results_pca.append(acc_vit_con_pca)
            curve_labels_pca.append("ViT con Aug (PCA)")
        else:
            model_vit_sin_pca, opt_vit_sin_pca, crit_vit_sin_pca = get_model('vit')
            model_path_vit_sin_pca = f"best_model_vit_sin_aug_pca_{MODEL_ID}.pth"
            if not os.path.exists(model_path_vit_sin_pca):
                print("Entrenando ViT SIN data augmentation (PCA)...")
                for epoch in range(NUM_EPOCHS):
                    train_one_epoch(model_vit_sin_pca, opt_vit_sin_pca, crit_vit_sin_pca, train_loader_sin_aug_pca,
                                    epoch)
                    validate(model_vit_sin_pca, crit_vit_sin_pca, val_loader_pca, epoch)
                torch.save(model_vit_sin_pca.state_dict(), model_path_vit_sin_pca)
            else:
                model_vit_sin_pca.load_state_dict(torch.load(model_path_vit_sin_pca))
            acc_vit_sin_pca = test_rotation(model_vit_sin_pca, test_loader_pca)
            accuracy_results_pca.append(acc_vit_sin_pca)
            curve_labels_pca.append("ViT (PCA)")

        if USE_VGG:
            if USE_DATA_AUGMENTATION:
                # VGG sin augmentation (PCA)
                model_vgg_sin_pca, opt_vgg_sin_pca, crit_vgg_sin_pca = get_model('vgg')
                model_path_vgg_sin_pca = f"best_model_vgg_sin_aug_pca_{MODEL_ID}.pth"
                if not os.path.exists(model_path_vgg_sin_pca):
                    print("Entrenando VGG SIN data augmentation (PCA)...")
                    for epoch in range(NUM_EPOCHS):
                        train_one_epoch(model_vgg_sin_pca, opt_vgg_sin_pca, crit_vgg_sin_pca, train_loader_sin_aug_pca,
                                        epoch)
                        validate(model_vgg_sin_pca, crit_vgg_sin_pca, val_loader_pca, epoch)
                    torch.save(model_vgg_sin_pca.state_dict(), model_path_vgg_sin_pca)
                else:
                    model_vgg_sin_pca.load_state_dict(torch.load(model_path_vgg_sin_pca))
                acc_vgg_sin_pca = test_rotation(model_vgg_sin_pca, test_loader_pca)
                accuracy_results_pca.append(acc_vgg_sin_pca)
                curve_labels_pca.append("VGG sin Aug (PCA)")

                # VGG con augmentation (PCA)
                model_vgg_con_pca, opt_vgg_con_pca, crit_vgg_con_pca = get_model('vgg')
                model_path_vgg_con_pca = f"best_model_vgg_con_aug_pca_{MODEL_ID}.pth"
                if not os.path.exists(model_path_vgg_con_pca):
                    print("Entrenando VGG CON data augmentation (PCA)...")
                    for epoch in range(NUM_EPOCHS):
                        train_one_epoch(model_vgg_con_pca, opt_vgg_con_pca, crit_vgg_con_pca, train_loader_con_aug_pca,
                                        epoch)
                        validate(model_vgg_con_pca, crit_vgg_con_pca, val_loader_pca, epoch)
                    torch.save(model_vgg_con_pca.state_dict(), model_path_vgg_con_pca)
                else:
                    model_vgg_con_pca.load_state_dict(torch.load(model_path_vgg_con_pca))
                acc_vgg_con_pca = test_rotation(model_vgg_con_pca, test_loader_pca)
                accuracy_results_pca.append(acc_vgg_con_pca)
                curve_labels_pca.append("VGG con Aug (PCA)")
            else:
                model_vgg_sin_pca, opt_vgg_sin_pca, crit_vgg_sin_pca = get_model('vgg')
                model_path_vgg_sin_pca = f"best_model_vgg_sin_aug_pca_{MODEL_ID}.pth"
                if not os.path.exists(model_path_vgg_sin_pca):
                    print("Entrenando VGG SIN data augmentation (PCA)...")
                    for epoch in range(NUM_EPOCHS):
                        train_one_epoch(model_vgg_sin_pca, opt_vgg_sin_pca, crit_vgg_sin_pca, train_loader_sin_aug_pca,
                                        epoch)
                        validate(model_vgg_sin_pca, crit_vgg_sin_pca, val_loader_pca, epoch)
                    torch.save(model_vgg_sin_pca.state_dict(), model_path_vgg_sin_pca)
                else:
                    model_vgg_sin_pca.load_state_dict(torch.load(model_path_vgg_sin_pca))
                acc_vgg_sin_pca = test_rotation(model_vgg_sin_pca, test_loader_pca)
                accuracy_results_pca.append(acc_vgg_sin_pca)
                curve_labels_pca.append("VGG (PCA)")

    # Graficar los resultados de precisión vs. rotación
    if ROTACION:
        plot_title_orig = "Precisión vs Rotación (Original)"
        plot_filename_orig = f"combined_accuracy_original_{MODEL_ID}.png"
        plot_multiple_accuracy_by_angle(accuracy_results_original, curve_labels_original, plot_title_orig,
                                        plot_filename_orig)

        if USE_PCA:
            plot_title_pca = "Precisión vs Rotación (PCA)"
            plot_filename_pca = f"combined_accuracy_pca_{MODEL_ID}.png"
            plot_multiple_accuracy_by_angle(accuracy_results_pca, curve_labels_pca, plot_title_pca, plot_filename_pca)
    else:
        print("No se realizó la evaluación con rotación.")


if __name__ == '__main__':
    main()
