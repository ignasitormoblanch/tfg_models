import cv2
import numpy as np
from scipy.ndimage import convolve


def generate_gaussian_kernel(size, sigma):
    """Genera un kernel gaussiano 2D."""
    k = cv2.getGaussianKernel(size, sigma)
    return k @ k.T


def apply_variable_blur(image):
    height, width = image.shape[:2]

    # Definir diferentes tamaños de kernel según la distancia al centro
    max_kernel_size = 31  # Tamaño máximo de kernel en el centro
    min_kernel_size = 5  # Tamaño mínimo de kernel en los bordes

    # Crear una copia de la imagen para el resultado
    result = np.zeros_like(image, dtype=np.float32)

    # Calcular centro de la imagen
    center_y, center_x = height // 2, width // 2
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    for y in range(height):
        for x in range(width):
            # Calcular la distancia del píxel al centro
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            weight = distance / max_distance

            # Interpolar el tamaño del kernel basado en la distancia
            kernel_size = int(min_kernel_size + (max_kernel_size - min_kernel_size) * (1 - weight))

            # Asegurar que el tamaño del kernel sea impar
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Generar el kernel gaussiano
            kernel = generate_gaussian_kernel(kernel_size, sigma=kernel_size / 6.0)

            # Aplicar la convolución sobre una pequeña ventana centrada en el píxel
            window_size = kernel_size // 2
            x1, x2 = max(0, x - window_size), min(width, x + window_size + 1)
            y1, y2 = max(0, y - window_size), min(height, y + window_size + 1)

            # Aplicar convolución en la ventana
            for c in range(3):  # Iterar sobre canales de color
                result[y, x, c] = convolve(image[y1:y2, x1:x2, c], kernel)[window_size, window_size]

    return np.clip(result, 0, 255).astype(np.uint8)


# Cargar imagen
image = cv2.imread('ruta_a_la_imagen.jpg')

# Aplicar desenfoque variable
blurred_image = apply_variable_blur(image)

# Guardar o mostrar la imagen resultante
cv2.imwrite('imagen_variable_blur.jpg', blurred_image)
cv2.imshow('Imagen desenfocada', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()