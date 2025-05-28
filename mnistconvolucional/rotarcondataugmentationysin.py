from datetime import datetime
import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import random, numpy as np
from torchvision.models import vgg16

nummaxepoch = 6
rotacion = True
fila = 6
seed = 42
print(f"CUDA disponible: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Transformaciones para los dos datasets (con y sin Data Augmentation)
train_transform_sin_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

train_transform_con_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(30),  # Rotación aleatoria hasta 30 grados
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Cargar MNIST
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)

# Separar en train/val
mnist_train_sin_aug, mnist_val_sin_aug = torch.utils.data.random_split(mnist_train, (40000, 20000))
mnist_train_con_aug, mnist_val_con_aug = torch.utils.data.random_split(mnist_train, (40000, 20000))

# Asignar transformaciones a cada dataset
mnist_train_sin_aug.dataset = datasets.MNIST(root='mnist_data', train=True, download=True,
                                             transform=train_transform_sin_aug)
mnist_train_con_aug.dataset = datasets.MNIST(root='mnist_data', train=True, download=True,
                                             transform=train_transform_con_aug)
mnist_val_sin_aug.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)
mnist_val_con_aug.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)

# Crear DataLoaders
train_loader_sin_aug = DataLoader(mnist_train_sin_aug, batch_size=32, shuffle=True)
train_loader_con_aug = DataLoader(mnist_train_con_aug, batch_size=32, shuffle=True)
val_loader = DataLoader(mnist_val_sin_aug, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32)

# Crear dos modelos
model_sin_aug = vgg16(pretrained=True).to(device)
model_con_aug = vgg16(pretrained=True).to(device)

for param in model_sin_aug.features.parameters():
    param.requires_grad = False
for param in model_con_aug.features.parameters():
    param.requires_grad = False

# Modificar la última capa para 10 clases
model_sin_aug.classifier[6] = nn.Linear(in_features=4096, out_features=10).to(device)
model_con_aug.classifier[6] = nn.Linear(in_features=4096, out_features=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_sin_aug = optim.Adam(model_sin_aug.classifier.parameters(), lr=0.001)
optimizer_con_aug = optim.Adam(model_con_aug.classifier.parameters(), lr=0.001)


# Entrenamiento de ambos modelos
def train_model(model, optimizer, train_loader, model_name):
    for epoch in range(nummaxepoch):
        print(f'epoca {epoch}')
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"{model_name} - Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}, Accuracy {accuracy:.2f}%")

    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Modelo {model_name} guardado.")

print('sin aug')
train_model(model_sin_aug, optimizer_sin_aug, train_loader_sin_aug, "model_sin_aug")
print('con aug')
train_model(model_con_aug, optimizer_con_aug, train_loader_con_aug, "model_con_aug")


# Evaluación con rotaciones
def evaluate_with_rotation(model, model_name):
    accuracy_by_angle = {}
    model.eval()

    for angle in range(0, 361, 10):
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = torch.stack([transforms.functional.rotate(img, angle) for img in images])
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        accuracy_by_angle[angle] = accuracy

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_by_angle.keys(), accuracy_by_angle.values(), marker='o', linestyle='-')
    plt.xlabel('Ángulo de Rotación (grados)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy del {model_name} según la rotación')
    plt.grid(True)
    plt.savefig(f"{model_name}1_accuracy_by_rotation.png")
    print(f"Gráfica guardada en {model_name}1_accuracy_by_rotation.png")

print('evaluando sin aug')
evaluate_with_rotation(model_sin_aug, "model_sin_aug")
print('evaluando con aug')
evaluate_with_rotation(model_con_aug, "model_con_aug")
