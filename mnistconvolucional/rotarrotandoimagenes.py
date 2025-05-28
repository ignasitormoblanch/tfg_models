from datetime import datetime
import torch
import gzip
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import math
import torch.nn as nn
import random, numpy as np
from torchvision.models import vgg16

nummaxepoch = 6
rotacion=True
fila = 5
seed = 42
print(f"CUDA disponible: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
print(f"fila{fila}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307, 0.1307, 0.1307),
    #                      (0.3081, 0.3081, 0.3081))
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307, 0.1307, 0.1307),
    #                      (0.3081, 0.3081, 0.3081))
])

def add_padding(image, padding=20):
    """Añade padding a la imagen para evitar recortes tras la rotación."""
    return transforms.functional.pad(image, padding, fill=0)

def rotate_image(image, angle):
    """Rota la imagen en un ángulo específico."""
    return transforms.functional.rotate(image, angle, fill=0)

# diccionario que luego usaré para la grafica
accuracy_by_angle = {}

# Descargar MNIST directamente desde torchvision
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)

# Separar en train/val
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, (40000, 20000))
mnist_train.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform)
mnist_val.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)

train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32)

# Inicializamos el modelo, la función de pérdida y el optimizador
model = vgg16(pretrained=True).to(device)

# Congelamos los parámetros de la parte convolucional
for param in model.features.parameters():
    param.requires_grad = False

# Modificar la última capa para clasificar 10 clases en lugar de 1000
model.classifier[6] = nn.Linear(in_features=4096, out_features=10).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


epoch = 0
model_path = f'best_modelfila{fila}'

if not os.path.exists(model_path):
    while epoch < nummaxepoch:
        sumaloss = 0
        sum2 = 0
        train_correct_train = 0
        total_train = 0
        train_correct_val = 0
        total_val = 0

        # train
        print(f'train epoca {epoch}')
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            sumaloss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        print(f'validacion epoca {epoch}')
        for batch_idx2, (images2, labels2) in enumerate(val_loader):
            images2 = images2.to(device)
            labels2 = labels2.to(device)

            outputs2 = model(images2)
            loss2 = criterion(outputs2, labels2)
            sum2 += loss2.item()

            _, predicted2 = torch.max(outputs2, 1)
            train_correct_val += (predicted2 == labels2).sum().item()
            total_val += labels2.size(0)

        print(f'Media de pérdida epoca {epoch} de train es {(sumaloss / len(train_loader)):.4f} y validation {(sum2/len(val_loader)):.4f}')

        # guardamos el modelo (seria mejor que lo guarde solo si es el mejor modelo pero como este mejora tanto pues que guarde el ultimo y ya)
        torch.save(model.state_dict(), f'best_modelfila{fila}')
        epoch += 1

    # Cargamos el mejor modelo
    model.load_state_dict(torch.load(f'best_modelfila{fila}'))
    # esta linea no me acuerdo de pq la poniamos
    model.eval()
    # test
    if rotacion:
        for angle in range(0, 361, 10):
            correct = 0
            total = 0
            print(f'estoy en el angulo {angle}')
            for images, labels in test_loader:
                # Importante: aplicamos add_padding y rotate_image en CPU
                # antes de mover a GPU
                images = torch.stack([add_padding(img) for img in images])
                images = torch.stack([rotate_image(img, angle) for img in images])

                # Ahora movemos a GPU
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = 100 * correct / total
            accuracy_by_angle[angle] = accuracy
            print(f'Accuracy con rotación de {angle} grados: {accuracy:.2f}%')

    else:
        train_correct_test = 0
        total_test = 0
        for batch_idx3, (images3, labels3) in enumerate(test_loader):
            images3 = images3.to(device)
            labels3 = labels3.to(device)

            outputs3 = model(images3)
            _, predicted3 = torch.max(outputs3, 1)
            train_correct_test += (predicted3 == labels3).sum().item()
            total_test += labels3.size(0)

        test_accuracy = 100 * train_correct_test / total_test
        print(f'el modelo tiene un acc de {test_accuracy:.2f}%')

else:
    # Cargamos el mejor modelo si ya existe (ya lo habia entrenado antes y como me lo guardo lo puedo usar)
    model.load_state_dict(torch.load(f'best_modelfila{fila}'))
    model.eval()
    print("Modelo cargado con los mejores pesos.")

    if rotacion:
        print('empezamos a rotarrr')
        for angle in range(0, 361, 1):
            correct = 0
            total = 0
            for images, labels in test_loader:
                # Importante: aplicamos add_padding y rotate_image en CPU
                # antes de mover a GPU
                images = torch.stack([add_padding(img) for img in images])
                images = torch.stack([rotate_image(img, angle) for img in images])

                # Ahora movemos a GPU
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = 100 * correct / total
            accuracy_by_angle[angle] = accuracy
            if angle % 30 == 0:
                print(f'Accuracy con rotación de {angle} grados: {accuracy:.2f}%')

        if accuracy_by_angle:
            plt.figure(figsize=(10, 5))
            plt.plot(accuracy_by_angle.keys(), accuracy_by_angle.values(), marker='o', linestyle='-')
            plt.xlabel('Ángulo de Rotación (grados)')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy del modelo según la rotación de las imágenes')
            plt.grid(True)

            # Guardar la imagen en un fichero accesible desde SSH
            plot_filename = f"accuracy_by_rotation360de1en1.png"
            plt.savefig(plot_filename)
            print(f"Gráfica guardada en: {plot_filename}")

    else:
        train_correct_test = 0
        total_test = 0
        for batch_idx3, (images3, labels3) in enumerate(test_loader):
            images3 = images3.to(device)
            labels3 = labels3.to(device)

            outputs3 = model(images3)
            _, predicted3 = torch.max(outputs3, 1)
            train_correct_test += (predicted3 == labels3).sum().item()
            total_test += labels3.size(0)

        test_accuracy = 100 * train_correct_test / total_test
        print(f'el modelo tiene un acc de {test_accuracy:.2f}%')

