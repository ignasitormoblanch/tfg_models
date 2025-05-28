import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
print(X_train.shape)
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_names[y_train[0]])
print(y_train[0])

tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
#softmax te ho fa en prob
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.summary()
print(model.layers)
hidden1 = model.layers[1]
print(hidden1)
print(model.get_layer('dense') is hidden1)

model.compile(loss="sparse_categorical_crossentropy",
 optimizer="sgd",
 metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=200,validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(
 figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
 style=["r--", "r--.", "b-", "b-*"])
plt.show()


print('Evaluación: ',model.evaluate(X_test, y_test))
X_new = X_test[:3]
y_proba = model.predict(X_new)
print('prueba: ', y_proba.round(2))


y_pred = y_proba.argmax(axis=-1)
print('que creo que es: ',y_pred)
print(np.array(class_names)[y_pred])
#lo de np.array es porq le paso un vector directo, si le quiero pasar all
#the vector lo haces así y si le pasas el [0] te da solo el 1º
print(class_names[y_pred[0]])