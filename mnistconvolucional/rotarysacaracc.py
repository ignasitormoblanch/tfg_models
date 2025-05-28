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
print(f"CUDA disponible: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

fila=72
print(f"fila{fila}")
print(torch.__version__)
print("hola3")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Añadido Resize aquí
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Descargar MNIST directamente desde torchvision
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, (40000, 20000))
mnist_train.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=train_transform)
mnist_val.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)


train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32)


# Inicializamos el modelo, la función de pérdida y el optimizador
#crea el modelo
model = vgg16(pretrained=True).to(device)

# preg cristian
for param in model.features.parameters():
    param.requires_grad = False

# Modificar la última capa para clasificar 10 clases en lugar de 1000
model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

criterion = nn.CrossEntropyLoss()

#aqui yo tenia adagrad pero se prefiere adam pq hay mas capas
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.01)

#vamos a entrenar 20 epoca

listalossval=[]
listalosstrain=[]
listaaccuracytrain=[]
listaaccuracyval=[]

mejor_loss_val=100
sigo=True
epoch=0
mejor_epoch=0
minparar=3
nummaxepoch=4

model_path = f'best_modelfila{fila}'

if not os.path.exists(model_path):
    while(sigo and epoch<nummaxepoch):
        print('while sigo')
        sumaloss=0
        sum2=0
        train_correct_train = 0
        total_train = 0
        train_correct_val=0
        total_val=0
        c=0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if(c%10==0):
                print(c)
            c+=1
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            sumaloss+=loss.item()
            _, predicted = torch.max(outputs, 1)  # Obtiene el índice de la predicción más alta
            # Suma los aciertos
            train_correct_train += (predicted == labels).sum().item()  # el item es pq el sum te devuelve Tensor(25) y con el item pasa a ser 25
            total_train += labels.size(0)  # Cuenta el total de ejemplos procesados
            # Backpropagación y optimización
            optimizer.zero_grad()
            #derive respecte del loss
            loss.backward()
            optimizer.step()

        for batch_idx2, (images2, labels2) in enumerate(val_loader):
            images2 = images2.to(device)
            labels2 = labels2.to(device)
            print('estoy en el validacion')
            outputs2 = model(images2)
            loss2 = criterion(outputs2, labels2)
            sum2+=loss2.item()
            _, predicted2 = torch.max(outputs2, 1)
            train_correct_val += (predicted2 == labels2).sum().item()
            total_val += labels2.size(0)

        print(f'Fila{fila}: La media de pérdida en la epoca {epoch} de train ha sido de {(sumaloss / len(train_loader)):.4f}')
        print(f'La media de pérdida en la epoca {epoch} de validación ha sido de {(sum2/len(val_loader)):.4f}')
        train_accuracy = 100 * train_correct_train / total_train
        val_accuracy = 100 * train_correct_val / total_val

        listalosstrain.append(sumaloss / len(train_loader))
        listalossval.append(sum2 / len(val_loader))
        listaaccuracytrain.append(train_accuracy)
        listaaccuracyval.append(val_accuracy)
        epoch+=1
        if mejor_loss_val<(sum2 / len(val_loader)) :
            if (epoch - mejor_epoch>minparar) :
                sigo=False

        else:
            mejor_epoch=epoch
            mejor_loss_val=(sum2 / len(val_loader))
            torch.save(model.state_dict(), f'best_modelfila{fila}')
            print(f"Mejor modelo guardado en la época {epoch} con pérdida de validación: {mejor_loss_val:.4f}")


    model.load_state_dict(torch.load(f'best_modelfila{fila}'))
    model.eval()
    print("Modelo cargado con los mejores pesos.")

    train_correct_test=0
    total_test=0
    for batch_idx3, (images3, labels3) in enumerate(test_loader):
        images3 = images3.to(device)
        labels3 = labels3.to(device)
        outputs3 = model(images3)
        _, predicted3 = torch.max(outputs3, 1)
        train_correct_test += (predicted3 == labels3).sum().item()
        total_test += labels3.size(0)

    test_accuracy = 100 * train_correct_test / total_test
    print(f'El accuracy es {test_accuracy:.4f}')


else:
    model.load_state_dict(torch.load(f'best_modelfila{fila}'))
    model.eval()
    print("Modelo cargado con los mejores pesos.")

    train_correct_test=0
    total_test=0
    for batch_idx3, (images3, labels3) in enumerate(test_loader):
        images3 = images3.to(device)
        labels3 = labels3.to(device)
        outputs3 = model(images3)
        _, predicted3 = torch.max(outputs3, 1)
        train_correct_test += (predicted3 == labels3).sum().item()
        total_test += labels3.size(0)

    test_accuracy = 100 * train_correct_test / total_test
    print(f'El accuracy es {test_accuracy:.4f}')







