import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import os

# Cargar el modelo preentrenado ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)  # Sin la parte superior
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Capa completamente conectada
predictions = Dense(1000, activation='softmax')(x)  # Salida con 1000 clases de ImageNet
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar las capas base de ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Función para cargar y procesar una imagen
def load_and_preprocess_image(img_path, target_size=(50, 50)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)  # Preprocesar la imagen según el modelo ResNet50


# Función para hacer predicciones
def classify_image(model, img_path):
    img = load_and_preprocess_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Obtener las 3 mejores predicciones
    return decoded_preds


# Función para clasificar todas las imágenes en una carpeta
def classify_images_in_folder(model, folder_path):
    # Obtener la lista de todas las imágenes en la carpeta
    print('holaaa')
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', 'JPEG'))]
    print('holaaa2')
    print(len(images))

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        predictions = classify_image(model, img_path)
        print(f"Predicciones para {img_file}:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            print(f"  {i + 1}: {label} ({score:.2f})")
        print("\n")


# Ejemplo de uso: Clasificar todas las imágenes de una carpeta
folder_path = "C:\\Users\\34644\\Desktop\\practicas_y_tfg\\ej1"
classify_images_in_folder(model, folder_path)