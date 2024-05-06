#Zadatak 0.0.3 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
#otkrivanja dijabetesa, pri čemu je prvih 8 stupaca ulazna veličina, a u devetom stupcu se nalazi
#izlazna veličina: klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes).
#Učitajte dane podatke. Podijelite ih na ulazne podatke X i izlazne podatke y. Podijelite podatke
#na skup za učenje i skup za testiranje modela u omjeru 80:20.
#a) Izgradite neuronsku mrežu sa sljedećim karakteristikama:
#- model očekuje ulazne podatke s 8 varijabli
#- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
#- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
#- izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
#Ispišite informacije o mreži u terminal.
#b) Podesite proces treniranja mreže sa sljedećim parametrima:
#- loss argument: cross entropy
#- optimizer: adam
#- metrika: accuracy.
#c) Pokrenite učenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veličinom
#batch-a 10.
#d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju učitanog modela.
#e) Izvršite evaluaciju mreže na testnom skupu podataka.
#f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
#podataka za testiranje. Komentirajte dobivene rezultate.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# Učitavanje podataka
data = pd.read_csv("pima-indians-diabetes.csv")

# Podijela podataka na ulazne i izlazne varijable
X = data.iloc[:, :8].values
y = data.iloc[:, 8].values

# Podjela podataka na skup za učenje i skup za testiranje u omjeru 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicijalizacija modela
model = Sequential()

# Dodavanje slojeva
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Ispis informacija o mreži
model.summary()

# Kompilacija modela
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treniranje modela
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Pohrana modela na disk
model.save("diabetes_model.h5")

# Evaluacija modela na testnom skupu
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy on test set:", accuracy)

# Predikcija na testnom skupu
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Izrada matrice zabune
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
