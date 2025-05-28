import os


# Función para listar y cargar imágenes de una carpeta
def load_images_from_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png')  # Extensiones de archivos de imagen
    image_files = []  # Lista para guardar las imágenes cargadas

    # Lista los archivos en la carpeta
    for file_name in os.listdir(folder_path):
        # Crear la ruta completa del archivo
        file_path = os.path.join(folder_path, file_name)

        # Verifica si es un archivo de imagen
        if os.path.isfile(file_path) and file_name.lower().endswith(image_extensions):
            image_files.append(file_path)  # Añade el archivo a la lista
            print(f"Imagen encontrada: {file_path}")

    return image_files


# Ejemplo de uso
folder_path = "C:\\Users\\34644\\Desktop\\practicas_y_tfg\\ej1"
images = load_images_from_folder(folder_path)

