#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, vgg16

NUM_EPOCHS = 40
TRANSLACION = True     # Se evaluará la traslación en el test si es True
USE_VIT = False        # Si es False se usará VGG16
USE_DATA_AUGMENTATION = False
MODEL_ID = 10
SEED = 42
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f'best_modelfila{MODEL_ID}'

print('ho23')
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Usando dispositivo: {DEVICE}")

###################################################
# COLLATE PERSONALITZAT per no apilar directament PIL
###################################################
def my_collate(batch):
    """
    batch: llista de tuples (img_PIL, label).
    Retorna (list_im, list_lab), on list_im i list_lab són llistes Python.
    Així evitarem l'error de collate per objectes PIL.
    """
    images = []
    labels = []
    for (img, lab) in batch:
        images.append(img)   # PIL image
        labels.append(lab)   # int label
    return images, labels

###################################################
# TRANSFORMACIONS (només s'usen en entrenament i validació)
###################################################
train_transform_con_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
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

def translate_image(image, offset):
    """
    Traduce la imagen MNIST 28x28 'offset' píxeles a la derecha sin padding.
    """
    return TF.affine(image, angle=0, translate=(offset, 0), scale=1, shear=0, fill=0)

def get_data_loaders():
    full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None)

    train_base, val_base = torch.utils.data.random_split(full_train_dataset, [40000, 20000])

    # Entrenament i validació: fem servir transformacions normals
    train_dataset_sin_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_sin_aug)
    train_dataset_sin_aug = torch.utils.data.Subset(train_dataset_sin_aug, train_base.indices)

    if USE_DATA_AUGMENTATION:
        train_dataset_con_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_con_aug)
    else:
        train_dataset_con_aug = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform_sin_aug)
    train_dataset_con_aug = torch.utils.data.Subset(train_dataset_con_aug, train_base.indices)

    val_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, val_base.indices)

    # Per test, fem transform=None i definim un DataLoader amb collate personalitzat
    test_dataset_v2 = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None)

    train_loader_sin_aug = DataLoader(train_dataset_sin_aug, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_con_aug = DataLoader(train_dataset_con_aug, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # test_loader amb collate personalitzat
    test_loader = DataLoader(test_dataset_v2, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)

    return train_loader_sin_aug, train_loader_con_aug, val_loader, test_loader

def get_model():
    if USE_VIT:
        model = vit_b_16(pretrained=True)
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
    Avalua la robustesa del model a translacions sense padding:
    1) Translada la imatge MNIST (28x28) offset píxels a la dreta.
    2) Després resize(224,224).
    3) Converteix a tensor i passa pel model.
    """
    model.eval()
    accuracy_by_offset = {}
    with torch.no_grad():
        for offset in range(0, 31):
            correct = 0
            total = 0
            for images, labels in test_loader:
                # images és una llista PIL de longitud batch_size
                # labels és una llista d'etiquetes
                new_images = []
                for pil_img in images:
                    # 1) translació sense padding
                    translated = translate_image(pil_img, offset=offset)
                    # 2) redimensionar a 224x224
                    resized = TF.resize(translated, (224, 224))
                    # 3) to_tensor
                    tens = TF.to_tensor(resized)
                    new_images.append(tens)

                images_tensor = torch.stack(new_images).to(DEVICE, non_blocking=True)
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE)

                outputs = model(images_tensor)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels_tensor).sum().item()
                total += labels_tensor.size(0)

            acc = 100. * correct / total
            accuracy_by_offset[offset] = acc
            if offset % 5 == 0:
                print(f"Desplazamiento {offset} px - Accuracy: {acc:.2f}%")
    return accuracy_by_offset

def test_no_translation(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            new_images = []
            for pil_img in images:
                # sense translació, només resize
                resized = TF.resize(pil_img, (224,224))
                tens = TF.to_tensor(resized)
                new_images.append(tens)

            images_tensor = torch.stack(new_images).to(DEVICE, non_blocking=True)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE)

            outputs = model(images_tensor)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.size(0)
    acc = 100. * correct / total
    print(f"Test Accuracy (no translation): {acc:.2f}%")
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
    plt.savefig(filename, format='png', dpi=300)
    print(f"Gráfico guardado como: {filename}")
    plt.close()

def main():
    (train_loader_sin_aug,
     train_loader_con_aug,
     val_loader,
     test_loader) = get_data_loaders()

    model_sin_aug, optimizer_sin_aug, criterion_sin_aug = get_model()
    model_con_aug, optimizer_con_aug, criterion_con_aug = get_model()

    MODEL_PATH_SIN = f"best_model_sin_aug{MODEL_ID}.pth"
    if not os.path.exists(MODEL_PATH_SIN):
        print('Empezamos el entrenamiento sin augmentation...')
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(model_sin_aug, optimizer_sin_aug, criterion_sin_aug, train_loader_sin_aug, epoch)
            validate(model_sin_aug, criterion_sin_aug, val_loader, epoch)
        torch.save(model_sin_aug.state_dict(), MODEL_PATH_SIN)
    else:
        model_sin_aug.load_state_dict(torch.load(MODEL_PATH_SIN))

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

    if TRANSLACION:
        print("Evaluando modelo sin_aug con translación en test (primero translado 28x28, luego resize(224x224), sin padding)...")
        accuracy_sin_aug = test_translation(model_sin_aug, test_loader)
        print("Evaluando modelo con_aug con translación en test (primero translado 28x28, luego resize(224x224), sin padding)...")
        accuracy_con_aug = test_translation(model_con_aug, test_loader)
        plot_accuracy_by_offset(accuracy_sin_aug, "Precisión sin_aug vs. Translación (sin padding)", f"accuracy_sin_aug{MODEL_ID}.png")
        plot_accuracy_by_offset(accuracy_con_aug, "Precisión con_aug vs. Translación (sin padding)", f"accuracy_con_aug{MODEL_ID}.png")
    else:
        test_no_translation(model_sin_aug, test_loader)
        test_no_translation(model_con_aug, test_loader)

if __name__ == '__main__':
    main()
