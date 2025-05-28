#!/usr/bin/env python3
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def show_translated_images(num_images=5, max_translation=10.0):
    # Cargar dataset MNIST
    dataset = MNIST(root='mnist_data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Pad(20, fill=0)  # Padding inicial
                    ]))
    loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    # Obtener un batch de imágenes
    images, labels = next(iter(loader))

    # Configurar la visualización
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 15))
    fig.suptitle('Ejemplos de imágenes con translación mejorada', fontsize=16)

    for i, (img, label) in enumerate(zip(images, labels)):
        # Mostrar imagen original (sin padding)
        original_img = transforms.functional.crop(img, 20, 20, 28, 28)
        axes[i, 0].imshow(original_img.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original - Label: {label.item()}')
        axes[i, 0].axis('off')

        # Aplicar translación mejorada
        translated_img = transforms.functional.affine(
            img,
            angle=0,
            translate=(max_translation, max_translation),
            scale=1.0,
            shear=0
        )

        # Mostrar imagen transladada (sin redimensionar)
        axes[i, 1].imshow(translated_img.squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Transladada {max_translation}px')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_translated_images(num_images=5, max_translation=25.0)