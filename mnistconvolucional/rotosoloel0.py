#!/usr/bin/env python3
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_only_zeros(dataset):
    indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
    return Subset(dataset, indices)

def rotate_and_show_zeros(num_images=5, rotation_angles=[0, 45, 90, 135, 180]):
    # Cargar MNIST y filtrar solo los ceros
    full_dataset = MNIST(root='mnist_data', train=False, download=True,
                         transform=transforms.ToTensor())
    zeros_dataset = get_only_zeros(full_dataset)

    # Cargar solo un número limitado de ceros
    loader = DataLoader(zeros_dataset, batch_size=num_images, shuffle=True)
    images, labels = next(iter(loader))

    # Mostrar imágenes rotadas
    fig, axes = plt.subplots(num_images, len(rotation_angles), figsize=(len(rotation_angles)*2, num_images*2))
    fig.suptitle('Rotaciones de dígitos "0"', fontsize=16)

    for i in range(num_images):
        for j, angle in enumerate(rotation_angles):
            rotated_img = transforms.functional.affine(
                images[i], angle=angle, translate=(0, 0), scale=1.0, shear=0
            )
            axes[i, j].imshow(rotated_img.squeeze(), cmap='gray')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'{angle}°')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    rotate_and_show_zeros(num_images=5, rotation_angles=[0, 45, 90, 135, 180])
