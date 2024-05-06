import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.models import load_model

num_classes = 10

model = load_model("OSU_LV8/lv8_zad1_model_keras.keras")
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#skaliranje na [0,1]
x_train_s =x_train.astype("float32")/255
x_test_s = x_test.astype("float32")/255

#slike (28,28,1)
x_test_s=np.expand_dims(x_test_s,-1)
x_train_s=np.expand_dims(x_train_s,-1)

#pretvaranje labela
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_train_s = x_train_s.reshape(-1, 784)
x_test_s = x_test_s.reshape(-1, 784)

y_predict=model.predict(x_test_s)

for i in range(500):
    if y_test[i] != y_predict[i].argmax():
        plt.figure()
        plt.imshow(x_test[i])
        plt.title(f"Real: {y_test[i]}, Predicted: {y_predict[i].argmax()}")
        plt.show()