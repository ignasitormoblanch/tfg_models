#!/usr/bin/env python3
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torchvision.models import vgg16, vit_b_16
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
print('hellooo')
# Parámetros
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_PCA_COMPONENTS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "pca"
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Dataset con PCA aplicado
class PCADataset(Dataset):
    def __init__(self, data_split, pca_model, size=(224, 224)):
        self.data_split = data_split
        self.pca = pca_model
        self.size = size

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image, label = self.data_split[idx]
        image_np = np.array(image).astype(np.float32).flatten() / 255.0
        image_pca = self.pca.transform([image_np])[0]
        image_reconstructed = self.pca.inverse_transform([image_pca])[0].reshape(28, 28)
        image_pil = Image.fromarray((image_reconstructed * 255).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image_pil)
        return image_tensor, label


# Cargar MNIST y separar
full_train = datasets.MNIST('mnist_data', train=True, download=True)
test_dataset = datasets.MNIST('mnist_data', train=False, download=True, transform=None)
train_data, val_data = torch.utils.data.random_split(full_train, [40000, 20000])

# PCA con sklearn
X_train = np.array([np.array(img).flatten() / 255.0 for img, _ in train_data])
pca = PCA(n_components=NUM_PCA_COMPONENTS)
pca.fit(X_train)

# Datasets con PCA
train_pca = PCADataset(train_data, pca)
val_pca = PCADataset(val_data, pca)
test_pca = PCADataset(test_dataset, pca)
train_loader = DataLoader(train_pca, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_pca, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_pca, batch_size=BATCH_SIZE, shuffle=False)

# Modelos
def get_model(arch):
    if arch == 'vgg':
        model = vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 10)
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    elif arch == 'vit':
        model = vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Arquitectura no soportada")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


# Entrenamiento
def train(model, optimizer, criterion, loader):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch} - Acc: {correct/total*100:.2f}%, Loss: {loss_sum/total:.4f}")


# Evaluación con rotaciones
def rotate(img, angle):
    # Pad, rotate, and then resize to 224x224
    padded = transforms.functional.pad(img, 20)
    rotated = transforms.functional.rotate(padded, angle)
    resized = transforms.functional.resize(rotated, [224, 224])
    return resized

def test_rotation(model, loader):
    model.eval()
    acc_angle = {}
    with torch.no_grad():
        for angle in range(0, 361, 5):
            total, correct = 0, 0
            for x, y in loader:
                x_rotated = torch.stack([rotate(img, angle) for img in x])
                x_rotated = x_rotated.to(DEVICE)
                y = y.to(DEVICE)
                output = model(x_rotated)
                pred = output.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc_angle[angle] = correct / total * 100
            if angle % 30 == 0:
                print(f"Rotació {angle}°: Acc = {acc_angle[angle]:.2f}%")
    return acc_angle


# Gràfic de resultats
def plot_accuracy(acc_dict, label, filename):
    angles = list(acc_dict.keys())
    accs = list(acc_dict.values())
    plt.plot(angles, accs, label=label)
    plt.xlabel("Rotació (graus)")
    plt.ylabel("Precisió (%)")
    plt.title("Precisió vs Rotació amb PCA")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Gràfic guardat: {filename}")


# MAIN
if __name__ == "__main__":
    for model_name in ["vgg", "vit"]:
        print(f"\n--- Entrenant {model_name.upper()} amb PCA ---")
        model, opt, crit = get_model(model_name)
        train(model, opt, crit, train_loader)
        acc_rot = test_rotation(model, test_loader)
        plot_accuracy(acc_rot, model_name.upper(), f"acc_rot_{model_name}_pca.png")
