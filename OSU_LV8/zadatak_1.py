import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.imshow(x_train[0])
plt.imshow(x_train[1])
plt.imshow(x_train[2])
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_train_s = x_train_s.reshape(-1, 784)
x_test_s = x_test_s.reshape(-1, 784)

#model sadrzi 6000 primjera za ucenje i 1000 za testiranje
#podatci su skalirani na raspon [0,1]
#izlazna velicina je kodirana kao binarna vrijednost

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu

model=keras.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

# TODO: provedi ucenje mreze

batch_size = 32
epochs = 10
model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# TODO: Prikazi test accuracy i matricu zabune

predictions=model.predict(x_test_s)
score=model.evaluate(x_test_s,y_test_s,verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
cm = confusion_matrix(y_test, predictions.argmax(axis=1))
print(cm)

# TODO: spremi model

model.save("OSU_LV8/lv8_zad1_model_keras.keras")
