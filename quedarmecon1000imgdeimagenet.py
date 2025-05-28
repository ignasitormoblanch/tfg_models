import tarfile
import os

# Define el nombre del archivo tar.gz y el directorio de salida
tar_file = "C:\\Users\\34644\\Downloads\\train_images_0.tar.gz"
output_directory = "C:\\Users\\34644\\Desktop\\practicas_y_tfg\\ejercicio1"
# Crear el directorio de salida si no existe
os.makedirs(output_directory, exist_ok=True)

# Abrir el archivo tar.gz
with tarfile.open(tar_file, 'r:gz') as tar:
    # Obtener la lista de miembros del archivo (es decir, las imágenes dentro del archivo)
    members = tar.getmembers()

    # Filtrar las primeras 10,000 imágenes
    first_10000_members = members[:10000]

    # Extraer solo las primeras 10,000 imágenes al directorio de salida
    for member in first_10000_members:
        tar.extract(member, path=output_directory)

print(f"Las primeras 10,000 imágenes han sido extraídas a {output_directory}.")