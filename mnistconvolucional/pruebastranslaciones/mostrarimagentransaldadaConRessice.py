#!/usr/bin/env python3
import random
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

print('holaa')
random.seed(42)
torch.manual_seed(42)

# Carreguem el dataset MNIST sense transformació
full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None)

# Seleccionem la primera imatge del conjunt de test (objecte PIL)
img_pil = test_dataset[0][0]

# Converteix la imatge original (28x28) a array per visualitzar-la
imagen_original = np.array(img_pil)

# Primer, traslladem la imatge original (28x28) sense redimensionar.
# Es trasllada 10 píxels a la dreta; això pot fer que part del contingut es talli.
translated_image = TF.affine(img_pil, angle=0, translate=(30, 0), scale=1, shear=0, fill=0)
imagen_translated = np.array(translated_image)

# Després, redimensionem la imatge traslladada a 224x224
translated_resized = TF.resize(translated_image, (224, 224))
imagen_translated_resized = np.array(translated_resized)

# Mostrem les tres imatges: original (28x28), traslladada (28x28) i traslladada + redimensionada (224x224)
plt.figure(figsize=(15, 5))

plt.subplot(1,3,1)
plt.imshow(imagen_original, cmap='gray')
plt.title('Imatge Original (28x28)')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(imagen_translated, cmap='gray')
plt.title('Imatge Traslladada (28x28)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(imagen_translated_resized, cmap='gray')
plt.title('Traslladada i Redimensionada (224x224)')
plt.axis('off')

plt.tight_layout()
plt.show()
