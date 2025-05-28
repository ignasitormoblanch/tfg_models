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

import torch.nn as nn

#hacer plot de las imagenes

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous,
# since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP
# runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe,
# unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program
# to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

fila=2
print(f"fila{fila}")
print(torch.__version__)
print("hola3")

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

# Descargar MNIST directamente desde torchvision
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True, transform=val_test_transform)


mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, (40000, 20000))
mnist_train.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=10, translate=(0.15, 0.15)),
    # transforms.Normalize((0.1307,), (0.3081,))
]))

mnist_val.dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=val_test_transform)


train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32)


# Definimos el modelo de Perceptrón Multicapa (MLP)
#nn.Module es la clase base en PyTorch para todos los modelos de redes neuronales
# Definir la red convolucional
class PrimerCNN(nn.Module):
    def __init__(self):
        super(PrimerCNN, self).__init__()
        # Capa convolucional: 1 entrada (grayscale), 32 filtros de 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Inicializamos el modelo, la función de pérdida y el optimizador
#crea el modelo
model = PrimerCNN()
#criterion define la función de pérdida para el modelo, en este caso CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
#El optimizador se encarga de actualizar los parámetros de la red para reducir la pérdida.
# Aquí estamos usando SGD (Stochastic Gradient Descent) con una tasa de aprendizaje (lr) de 0.01
#esto es el backpropagation
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

#vamos a entrenar 1 epoca

listalossval=[]
listalosstrain=[]
listaaccuracytrain=[]
listaaccuracyval=[]

mejor_loss_val=100
sigo=True
epoch=0
mejor_epoch=0
minparar=19
while(sigo):
    sum=0
    sum2=0
    train_correct_train = 0
    total_train = 0
    train_correct_val=0
    total_val=0
    l2_alpha = 0.05
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Propagación hacia adelante

        images=images / 255.0
        outputs = model(images)
        # loss_acc = criterion(outputs, labels)
        loss = criterion(outputs, labels)
        # L2 regularization
        # l2_reg = torch.tensor(0.)
        # for param in model.parameters():
        #     l2_reg += torch.norm(param)
        # loss = loss_acc + l2_alpha * l2_reg

        sum+=loss.item()
        _, predicted = torch.max(outputs, 1)  # Obtiene el índice de la predicción más alta
        # Suma los aciertos
        train_correct_train += (predicted == labels).sum().item()  # el item es pq el sum te devuelve Tensor(25) y con el item pasa a ser 25

        total_train += labels.size(0)  # Cuenta el total de ejemplos procesados

        #la backpropagation hay q hacerla en cada bach o cuando acaba la primera época?
        # Backpropagación y optimización
        optimizer.zero_grad()
        #derive respecte del loss
        loss.backward()
        optimizer.step()


    for batch_idx2, (images2, labels2) in enumerate(val_loader):
        # Propagación hacia adelante
        images2 = images2 / 255.0
        outputs2 = model(images2)
        loss2 = criterion(outputs2, labels2)
        sum2+=loss2.item()
        _, predicted2 = torch.max(outputs2, 1)
        train_correct_val += (predicted2 == labels2).sum().item()
        total_val += labels2.size(0)


    print(f'Fila{fila}: La media de pérdida en la epoca {epoch} de train ha sido de {(sum/len(train_loader)):.4f}')
    print(f'La media de pérdida en la epoca {epoch} de validación ha sido de {(sum2/len(val_loader)):.4f}')
    train_accuracy = 100 * train_correct_train / total_train
    val_accuracy = 100 * train_correct_val / total_val

    listalosstrain.append(sum / len(train_loader))
    listalossval.append(sum2 / len(val_loader))
    listaaccuracytrain.append(train_accuracy)
    listaaccuracyval.append(val_accuracy)
    epoch+=1
    if mejor_loss_val<(sum2 / len(val_loader)):
        if epoch - mejor_epoch>minparar:
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
    # Propagación hacia adelante
    images3 = images3 / 255.0

    outputs3 = model(images3)
    _, predicted3 = torch.max(outputs3, 1)
    train_correct_test += (predicted3 == labels3).sum().item()
    total_test += labels3.size(0)

test_accuracy = 100 * train_correct_test / total_test
hora_actual = datetime.now()
x=hora_actual.strftime("%I_%M_%S")
accuracy_file_path = os.path.join('.','mnistPlotDir', f'Fila{fila}mnist_accuracy{x}.txt')

# Escribe la precisión en el archivo
with open(accuracy_file_path, 'w') as f:
    f.write(f"Accuracy del modelo en el conjunto de prueba: {test_accuracy:.2f}%\n")


plt.figure(figsize=(12, 10))

# Subplot 1: Loss de entrenamiento y validación
plt.subplot(2, 1, 1)
plt.plot(listalosstrain, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(listalossval, marker='o', linestyle='-', color='r', label='Validation Loss')
plt.title('Loss durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Subplot 2: Accuracy de entrenamiento y validación
plt.subplot(2, 1, 2)
plt.plot(listaaccuracytrain, marker='o', linestyle='-', color='b', label='Train Accuracy')
plt.plot(listaaccuracyval, marker='o', linestyle='-', color='r', label='Validation Accuracy')
plt.title('Accuracy durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)

# Mostrar ambos gráficos en la misma figura
plt.tight_layout()  # Para ajustar el espacio entre subplots
# plt.savefig(os.path.join("/home/tormo/proyectoTFG/pruebasPerceptron/mnistPlotDir", f"loss_accuracy_vertical.png"))

path = os.path.join('.', 'mnistPlotDir', f'Fila{fila}loss_accuracy{x}.png')


plt.savefig(path)

plt.close()







