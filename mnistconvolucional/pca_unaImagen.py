import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# Parámetros
NUM_COMPONENTES = 50  # Número de componentes principales a conservar
N_SAMPLES = 1000      # Número de imágenes a usar para el PCA

# Cargar el dataset MNIST con la transformación para obtener tensores
transform = transforms.ToTensor()
mnist = MNIST(root='mnist_data', train=True, download=True, transform=transform)

# Usar un subconjunto de N_SAMPLES imágenes para PCA
subset = Subset(mnist, range(N_SAMPLES))
loader = DataLoader(subset, batch_size=N_SAMPLES, shuffle=False)

# Obtener todas las imágenes y etiquetas del subconjunto
images, labels = next(iter(loader))
images_np = images.squeeze().numpy()          # Forma: (N_SAMPLES, 28, 28)
flat_images = images_np.reshape(N_SAMPLES, -1)  # Forma: (N_SAMPLES, 784)

# Aplicar PCA sobre el conjunto de imágenes
pca = PCA(n_components=NUM_COMPONENTES)
pca.fit(flat_images)

# Seleccionar una imagen para reconstruir (por ejemplo, la primera imagen)
image_index = 0
original_image = flat_images[image_index].reshape(28, 28)
# Transformar y reconstruir la imagen seleccionada
transformed = pca.transform(flat_images[image_index].reshape(1, -1))
reconstructed = pca.inverse_transform(transformed).reshape(28, 28)

# Dibujar la imagen original y la reconstruida
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title("Imagen original")
axs[0].axis('off')
axs[1].imshow(reconstructed, cmap='gray')
axs[1].set_title(f"Reconstruida con PCA\n({NUM_COMPONENTES} componentes)")
axs[1].axis('off')
plt.tight_layout()
plt.show()
