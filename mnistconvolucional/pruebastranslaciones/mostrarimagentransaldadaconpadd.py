#!/usr/bin/env python3
import random
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

print('holaa')
random.seed(42)
torch.manual_seed(42)

# Funció per afegir padding a la imatge
def add_padding(image, padding=20):
    """
    Afegeix padding de 'padding' píxels a totes les vores de la imatge.
    """
    return transforms.functional.pad(image, padding, fill=0)

# Carreguem el dataset MNIST sense transformació
full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None)

# Seleccionem la primera imatge del conjunt de test (és un objecte PIL)
imagen_pil = test_dataset[0][0]

# Converteix la imatge original a array per visualitzar-la
imagen_original = np.array(imagen_pil)

# Afegeix padding per evitar que la translació talli el dígit
imagen_padded = add_padding(imagen_pil, padding=20)

# Trasllada la imatge amb padding 20 píxels a la dreta
translated_image_pil = TF.affine(imagen_padded, angle=0, translate=(20, 0), scale=1, shear=0, fill=0)
imagen_traslladada = np.array(translated_image_pil)

# Mostrem la imatge original i la traslladada
plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.imshow(imagen_original, cmap='gray')
plt.title('Imatge Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(imagen_traslladada, cmap='gray')
plt.title('Imatge Traslladada 20 px (amb padding)')
plt.axis('off')

plt.tight_layout()
plt.show()
