import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("OSU_LV2/data.csv", delimiter=',')
data_copy=data[1:]
print(data_copy)
duljina=len(data_copy)
print("Mjerenja su izvršena na",duljina,"osoba")

plt.scatter(data_copy[:,1],data_copy[:,2],color="pink")
plt.xlabel("Visina osoba")
plt.ylabel("Tezina osoba")
plt.show()

data_every50th=data_copy[::50]
plt.scatter(data_every50th[:,1],data_every50th[:,2],color="green")
plt.xlabel("Visina osoba")
plt.ylabel("Tezina osoba")
plt.show()

print("Maksimalna vrijednost visine:",np.max(data_copy[:,1]))
print("Minimalna vrijednost visine:",np.min(data_copy[:,1]))
print("Srednja vrijednost visine:",np.mean(data_copy[:,1]))

muskarci=data_copy[data_copy[:,0]==1]
zene=data_copy[data_copy[:,0]==0]
print("Maksimalna vrijednost visine muskaraca:",np.max(muskarci[:,1]))
print("Minimalna vrijednost visine muskaraca:",np.min(muskarci[:,1]))
print("Srednja vrijednost visine muskaraca:",np.mean(muskarci[:,1]))

print("Maksimalna vrijednost visine zena:",np.max(zene[:,1]))
print("Minimalna vrijednost visine zena:",np.min(zene[:,1]))
print("Srednja vrijednost visine zena:",np.mean(zene[:,1]))
