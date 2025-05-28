import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, convolve
from tensorflow.keras.datasets import mnist


def radial_gaussian_kernel(size=3, sigma=1.0):
    """
    Crea un kernel gaussiano isotrópico 2D de tamaño (2*size+1) x (2*size+1).
    size: Radio (en píxeles) del kernel.
    sigma: Desviación estándar de la gaussiana.
    """
    # Coordenadas (x, y) en el rango [-size, size]
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)
    xx, yy = np.meshgrid(x, y)

    # Distancia radial al centro
    rr = xx ** 2 + yy ** 2

    # Función gaussiana
    kernel = np.exp(-rr / (2 * sigma ** 2))

    # Normalizar para que la suma sea 1
    kernel /= kernel.sum()

    return kernel


def main():
    # Cargar el dataset MNIST (imágenes de 28x28)
    (train_images, _), (_, _) = mnist.load_data()

    # Seleccionamos la primera imagen de entrenamiento (por ejemplo)
    img = train_images[0].astype(float)

    # Definimos un kernel gaussiano isotrópico
    kernel = radial_gaussian_kernel(size=3, sigma=1.0)

    # 1) Convolucionar y luego rotar
    conv_img = convolve(img, kernel, mode='reflect')
    rot_after_conv = rotate(conv_img, angle=30, reshape=False)

    # 2) Rotar y luego convolucionar
    rot_img = rotate(img, angle=30, reshape=False)
    conv_after_rot = convolve(rot_img, kernel, mode='reflect')

    # Visualización de resultados
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Imagen original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Convolución -> Rotación")
    plt.imshow(rot_after_conv, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Rotación -> Convolución")
    plt.imshow(conv_after_rot, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
