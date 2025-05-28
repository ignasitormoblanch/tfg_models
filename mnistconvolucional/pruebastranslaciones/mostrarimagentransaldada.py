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

# Carreguem el dataset MNIST (sense transformació)
full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None)

# Seleccionem la primera imatge del conjunt de test i la convertim a array
imagen_original = np.array(test_dataset[0][0])

# La imatge original és un objecte PIL; la podem traslladar utilitzant transforms.functional.affine
# Traslladem 20 píxels a la dreta (sense rotació, escalat o shear)
translated_image_pil = TF.affine(test_dataset[0][0], angle=0, translate=(10, 0), scale=1, shear=0)
imagen_traslladada = np.array(translated_image_pil)

# Mostrem les dues imatges amb matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(imagen_original, cmap='gray')
plt.title('Imatge Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(imagen_traslladada, cmap='gray')
plt.title('Imatge Traslladada 20 px')
plt.axis('off')

plt.tight_layout()
plt.show()
