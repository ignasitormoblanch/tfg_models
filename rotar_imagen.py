import os
from random import seed
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from scipy.ndimage import rotate

print('holaa')
seed(42)

# Cargamos el dataset MNIST usando torchvision
full_train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=None)

# Seleccionamos una imagen (por ejemplo, la primera imagen del conjunto de test)
imagen_original = np.array(test_dataset[0][0])


def mostrar_imagen(imagen, titulo="", nombre_archivo="imagen.png"):
    # Crear carpeta para guardar imágenes si no existe
    output_dir = "imagenes_guardadas"
    os.makedirs(output_dir, exist_ok=True)

    # Ruta completa para guardar la imagen
    ruta_guardado = os.path.join(output_dir, nombre_archivo)

    # Mostrar y guardar la imagen
    plt.figure(figsize=(3, 3))
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')

    plt.savefig(ruta_guardado, bbox_inches='tight')  # Guardar imagen
    plt.close()  # Cerrar figura para evitar sobrescribir
    plt.show()


# Mostramos la imagen original
mostrar_imagen(imagen_original, "Imagen original", "original.png")

# Parámetros para la rotación
angulo_incremento = 180  # grados de rotación en cada iteración
vueltasmax=700
num_iteraciones = vueltasmax*360//angulo_incremento  # número total de rotaciones
print(num_iteraciones)
imagen_actual = imagen_original.copy()

# Rotamos y mostramos la imagen en cada iteración
for i in range(1, num_iteraciones + 1):
    imagen_rotada = rotate(imagen_actual, angle=angulo_incremento, reshape=False, mode='nearest')
    imagen_actual = imagen_rotada.copy()

    if i in [900//angulo_incremento, 3600//angulo_incremento, 18000//angulo_incremento, 36000//angulo_incremento, 72000//angulo_incremento, 144000//angulo_incremento, 252000//angulo_incremento]:
        print('hola')
        mostrar_imagen(imagen_rotada, f"Rotación: {i * angulo_incremento}°", f"rotacion_{i}.png")
