import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math, time

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class TrainableGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size, self.stride, self.padding = k, stride, padding

        # Parám. Gabor entrenables
        self.thetas = nn.Parameter(torch.linspace(0, np.pi - np.pi/out_channels, out_channels))
        self.frequencies = nn.Parameter(torch.tensor(
            [0.15, 0.30] * (out_channels // 2) + ([0.20] if out_channels % 2 else [])))
        self.sigmas = nn.Parameter(torch.full((out_channels,), k[0] / 4.5))
        self.psis = nn.Parameter(torch.zeros(out_channels))

        # Mallado fijo
        x = torch.linspace(-(k[0]-1)/2., (k[0]-1)/2., k[0])
        y = torch.linspace(-(k[1]-1)/2., (k[1]-1)/2., k[1])
        self.register_buffer("grid_x", x.repeat(k[1], 1))
        self.register_buffer("grid_y", y.unsqueeze(1).repeat(1, k[0]))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.uniform_(self.bias, -1/math.sqrt(in_channels*k[0]*k[1]),
                                   1/math.sqrt(in_channels*k[0]*k[1]))

    def forward(self, x):
        kernels = []
        gx, gy = self.grid_x.to(x.device), self.grid_y.to(x.device)
        for i in range(self.out_channels):
            θ, f, σ, ψ = self.thetas[i], self.frequencies[i], self.sigmas[i], self.psis[i]
            rot_x = gx*torch.cos(θ) + gy*torch.sin(θ)
            rot_y = -gx*torch.sin(θ) + gy*torch.cos(θ)

            g = torch.exp(-0.5*((rot_x**2+rot_y**2)/σ**2)) * torch.cos(2*np.pi*f*rot_x + ψ)
            g = g - g.mean()
            kernels.append(g.unsqueeze(0).unsqueeze(0))
        weight = torch.cat(kernels, 0)               # (out,1,H,W)
        return F.conv2d(x, weight, bias=self.bias,
                        stride=self.stride, padding=self.padding)



class GaborBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.gabor = TrainableGaborConv2d(1, 16, 7, padding=0)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32*4*4, 128)

    def forward(self, x):
        x = F.relu(self.gabor(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))                     # (B,128)



class GaborClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GaborBackbone()
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)



class RotationInvariantGaborCNN(nn.Module):

    def __init__(self, k_rotations=(0, 1, 2, 3)):
        super().__init__()
        self.backbone = GaborBackbone()   # pesos compartidos
        self.head = nn.Linear(128, 10)
        self.k_list = k_rotations         # 0→0°, 1→90°, 2→180°, 3→270°

    def forward(self, x):
        logits_all = []
        for k in self.k_list:
            x_rot = torch.rot90(x, k, dims=[2, 3]) if k else x
            feats = self.backbone(x_rot)
            logits_all.append(self.head(feats))
        logits_stack = torch.stack(logits_all, dim=0)   # (4,B,10)
        return logits_stack.mean(0)                     # (B,10)  ⟵ promedio



def train_epoch(model, loader, optim_, loss_fn, device):
    model.train()
    loss_sum, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim_.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward(); optim_.step()
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()
        n += x.size(0)
    return loss_sum/n, correct/n


@torch.no_grad()
def eval_rot(model, device, angle):
    model.eval()
    tfm = transforms.Compose([
        transforms.RandomRotation([angle, angle]),
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=512, pin_memory=device.type=='cuda')
    correct, n = 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(1).eq(y).sum().item()
        n += x.size(0)
    return correct/n


if __name__ == "__main__":
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    ds_train = datasets.MNIST('./data', train=True, download=True, transform=tfm_train)
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True,
                          pin_memory=device.type=='cuda')


    gabor_norm   = GaborClassifier().to(device)
    gabor_rotinv = RotationInvariantGaborCNN().to(device)

    opt_norm   = optim.Adam(gabor_norm.parameters(),   lr=1e-3)
    opt_rotinv = optim.Adam(gabor_rotinv.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3
    for ep in range(1, epochs+1):
        t0 = time.time()
        l_n, a_n = train_epoch(gabor_norm,   dl_train, opt_norm,   loss_fn, device)
        l_r, a_r = train_epoch(gabor_rotinv, dl_train, opt_rotinv, loss_fn, device)
        print(f"Epoch {ep}/{epochs}  "
              f"Normal  loss={l_n:.4f} acc={a_n:.4f} | "
              f"Rot-Inv loss={l_r:.4f} acc={a_r:.4f}  "
              f"[{time.time()-t0:.1f}s]")


    angles = list(range(0, 360, 5))
    acc_norm, acc_rotinv = [], []
    for ang in angles:
        acc_norm.append   (eval_rot(gabor_norm,   device, ang))
        acc_rotinv.append(eval_rot(gabor_rotinv, device, ang))
        print(f"∠{ang:3d}°  norm={acc_norm[-1]:.3f}  rot-inv={acc_rotinv[-1]:.3f}")

    plt.figure(figsize=(10,6))
    plt.plot(angles, acc_norm,    'o--', label='GaborCNN normal')
    plt.plot(angles, acc_rotinv, 'x-',  label='Rotation-Invariant GaborCNN')
    plt.xticks(np.arange(0,361,45)); plt.yticks(np.arange(0,1.05,0.1))
    plt.xlabel('Ángulo de rotación'); plt.ylabel('Accuracy')
    plt.title('Robustez frente a rotaciones (MNIST)')
    plt.grid(True, ls='--', alpha=.6); plt.legend(); plt.ylim(0,1.05)
    plt.tight_layout(); plt.savefig('rot_invariance_gabor.png')
    print("\nGráfico guardado en rot_invariance_gabor.png")
