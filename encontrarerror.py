import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import numpy as np
from scipy.ndimage import convolve
import torch
import matplotlib.pyplot as plt


# Verificar si hay GPU disponible
device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

carpeta_entrada ="C:\\Users\\34644\\Desktop\\practicas_y_tfg\\ej0"
#carpeta_entrada=os.path.join('pruebasimagenet','ej0')


def obtener_dimensiones_carpeta(carpeta_entrada):
    dimensiones_imagenes = {}  # Diccionario para guardar las dimensiones
    for nombre_archivo in os.listdir(carpeta_entrada):
        ruta_imagen = os.path.join(carpeta_entrada, nombre_archivo)

        # Verificar si el archivo es una imagen (por su extensión)
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            try:
                with Image.open(ruta_imagen) as img:
                    #img = img.resize((224, 224))
                    # Obtener las dimensiones de la imagen
                    ancho, alto = img.size
                    dimensiones_imagenes[nombre_archivo] = (ancho, alto)
            except Exception as e:
                print(f"Error al abrir la imagen {nombre_archivo}: {e}")

    return dimensiones_imagenes


def load_images_from_folder(folder_path):
    image_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            img=Image.open(file_path)
            #img = img.resize((224, 224))
            image_files.append(img)
    return image_files

labels=['cat','dog','sofa']
dimensiones = obtener_dimensiones_carpeta(carpeta_entrada)
images = load_images_from_folder(carpeta_entrada)
nombres=[]
# Mostrar las dimensiones de cada imagen
for nombre_imagen, dimension in dimensiones.items():
    nombres.append(nombre_imagen)
    print(f"Imagen: {nombre_imagen} - Dimensiones: {dimension[0]}x{dimension[1]} píxeles")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32").to(device)

inputs = processor(text=labels, images=images, return_tensors="pt", padding=True, truncation=True)
print(len(nombres))
