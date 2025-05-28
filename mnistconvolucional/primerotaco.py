import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import random

SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# --------------------------------------------------------------------
# 1) Dataset MNIST en un lienzo 128x128 con dígito en (0,0)
# --------------------------------------------------------------------
class MNISTTopLeftDataset(torch.utils.data.Dataset):
    """
    Coloca cada dígito MNIST en la esquina superior izquierda de
    una imagen 128x128, para que rotar alrededor de (0,0) sea
    consistente con la acción de p4.
    """

    def __init__(self, train=True):
        self.ds = torchvision.datasets.MNIST(
            root="./mnist_data",
            train=train,
            download=True,
            transform=None  # lo haremos "a mano" abajo
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        # Convertir a tensor [1,28,28]
        x = torchvision.transforms.functional.to_tensor(img)  # (1,28,28)

        # Crear lienzo grande [1,128,128] lleno de 0
        big_canvas = torch.zeros((1, 128, 128), dtype=torch.float32)
        # Pegar la imagen MNIST en la esquina superior izq (0,0)
        big_canvas[:, :28, :28] = x

        return big_canvas, label


def get_data_loaders():
    train_ds = MNISTTopLeftDataset(train=True)
    test_ds = MNISTTopLeftDataset(train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# --------------------------------------------------------------------
# 2) Definición de la capa GroupConv2d (p4)
# --------------------------------------------------------------------
class GroupConv2d(nn.Module):
    """
    Convolución equivariante a rotaciones de 0°, 90°, 180°, 270° (p4).
    Importante: usamos k=-r en torch.rot90 para la rotación inversa.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, num_rotations=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations

        # Peso base (out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * num_rotations))
        else:
            self.bias = None

    def forward(self, x):
        # x.shape = (batch, in_channels, H, W)
        outs = []
        for r in range(self.num_rotations):
            # Rotamos el filtro en sentido inverso => k=-r
            rot_w = torch.rot90(self.weight, k=-r, dims=[2, 3])

            # Convolución normal (sin bias aquí)
            out_r = nn.functional.conv2d(x, rot_w, stride=self.stride, padding=self.padding)
            outs.append(out_r)

        # Apilamos en la nueva dimensión (rotación)
        # => (batch, num_rotations, out_channels, H_out, W_out)
        out = torch.stack(outs, dim=1)

        # Agregar bias si existe
        if self.bias is not None:
            # bias shape => (num_rotations * out_channels)
            b = self.bias.view(self.num_rotations, self.out_channels, 1, 1)
            out = out + b

        return out


# --------------------------------------------------------------------
# 3) Pooling sobre la dimensión de grupo
# --------------------------------------------------------------------
class GroupPooling(nn.Module):
    def __init__(self, mode='max'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # x: (batch, num_rotations, channels, H, W)
        if self.mode == 'max':
            return x.max(dim=1)[0]  # => (batch, channels, H, W)
        elif self.mode == 'avg':
            return x.mean(dim=1)
        else:
            raise ValueError("Modo no soportado")


# --------------------------------------------------------------------
# 4) Modelo sencillo: 1 group conv + group pooling + fully-connected
# --------------------------------------------------------------------
class GCNNp4(nn.Module):
    """
    Modelo minimalista que logra invariancia p4:
      - 1 GroupConv2d => equivariancia
      - 1 GroupPooling => invariancia
      - FC para clasificación
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.gconv = GroupConv2d(in_channels=1, out_channels=16, kernel_size=3,
                                 stride=1, padding=1, num_rotations=4)
        self.relu = nn.ReLU(inplace=True)
        self.gpool = GroupPooling(mode='max')

        # Tras el pooling de grupo, la salida es (batch, 16, H, W).
        # Trabajamos con 128x128 => la convolución con padding=1 no reduce
        # => saldrá (batch, 16, 128, 128).
        # Metemos un MaxPool2d para bajar un poco la dimensión
        self.pool2d = nn.MaxPool2d(kernel_size=2)  # => (batch,16,64,64)

        # FC final
        self.fc = nn.Linear(16 * 64 * 64, num_classes)

    def forward(self, x):
        # x: (batch,1,128,128)
        out = self.gconv(x)  # => (batch,4,16,128,128)
        out = self.relu(out)
        out = self.gpool(out)  # => (batch,16,128,128) invariante a p4
        out = self.pool2d(out)  # => (batch,16,64,64)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# --------------------------------------------------------------------
# 5) Entrenamiento y test
# --------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    print(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")


def test_p4_invariance(model, loader):
    """
    Mide la precisión en las rotaciones 0°, 90°, 180°, 270°
    (equivalentes a times90=0,1,2,3) usando torch.rot90
    en dims=[2,3], que rota alrededor de (0,0).
    """
    model.eval()
    angles = [0, 1, 2, 3]  # multiplica 90°
    with torch.no_grad():
        for a in angles:
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Rotar la imagen "a" veces 90° en dims=[2,3]
                # => rotación discreta alrededor de la esquina sup izq
                rotated = torch.rot90(images, k=a, dims=[2, 3])

                outputs = model(rotated)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = 100.0 * correct / total
            deg = a * 90
            print(f"Ángulo {deg}° => Accuracy: {acc:.2f}%")


def main():
    # 1) Carga de datos
    train_loader, test_loader = get_data_loaders()

    # 2) Modelo GCNN p4
    model = GCNNp4(num_classes=10).to(DEVICE)

    # 3) Optim y criterio
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 4) Entrenar
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch)

    # 5) Evaluar invariancia
    print("\nEvaluación de invariancia p4 en Test:")
    test_p4_invariance(model, test_loader)


if __name__ == "__main__":
    main()
