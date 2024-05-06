#Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
#otkrivanja dijabetesa, pri čemu se u devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1
#(ima dijabetes). Učitajte dane podatke u obliku numpy polja data. Dodajte programski kod u
#skriptu pomoću kojeg možete odgovoriti na sljedeća pitanja:
#a) Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja?
#b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne
#mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
#c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomoću scatter dijagrama.
#Dodajte naziv dijagrama i nazive osi s pripadajućim mjernim jedinicama. Komentirajte
#odnos dobi i BMI prikazan dijagramom.
#d) Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne
#mase (BMI) u ovom podatkovnom skupu.
#e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
#kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene
#vrijednosti.
#import numpy as np
#import matplotlib.pyplot as plt

# Učitavanje podataka
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',', skiprows=1)

# a) Broj osoba na temelju veličine numpy polja data
broj_osoba = data.shape[0]
print("a) Broj osoba na temelju veličine numpy polja data:", broj_osoba)

# b) Provjera i uklanjanje izostalih i dupliciranih vrijednosti
data = data[~np.isnan(data).any(axis=1)]  # Uklanjanje redova s NaN vrijednostima
data = np.unique(data, axis=0)  # Uklanjanje dupliciranih redova
broj_uzoraka = data.shape[0]
print("b) Broj preostalih uzoraka mjerenja nakon uklanjanja izostalih i dupliciranih vrijednosti:", broj_uzoraka)

# c) Scatter dijagram dobi i BMI
dob = data[:, 1]
bmi = data[:, 5]

plt.scatter(dob, bmi, alpha=0.5)
plt.title('Odnos dobi i BMI')
plt.xlabel('Dob (godine)')
plt.ylabel('BMI (kg/m^2)')
plt.show()
print("c) Odnos dobi i BMI prikazan scatter dijagramom pokazuje da postoji tendencija povećanja BMI s dobi.")

# d) Minimalna, maksimalna i srednja vrijednost BMI
min_bmi = np.min(bmi)
max_bmi = np.max(bmi)
mean_bmi = np.mean(bmi)
print("d) Minimalna vrijednost BMI:", min_bmi)
print("   Maksimalna vrijednost BMI:", max_bmi)
print("   Srednja vrijednost BMI:", mean_bmi)

# e) Za osobe s dijabetesom i bez dijabetesa
dijabetes = data[data[:, 8] == 1]
bez_dijabetesa = data[data[:, 8] == 0]

# Izračun broja osoba s dijabetesom
broj_dijabetesa = dijabetes.shape[0]
print("e) Broj osoba kojima je dijagnosticiran dijabetes:", broj_dijabetesa)

# Izračun minimalne, maksimalne i srednje vrijednosti BMI za osobe s dijabetesom
min_bmi_dijabetes = np.min(dijabetes[:, 5])
max_bmi_dijabetes = np.max(dijabetes[:, 5])
mean_bmi_dijabetes = np.mean(dijabetes[:, 5])
print("   Za osobe s dijabetesom:")
print("   Minimalna vrijednost BMI:", min_bmi_dijabetes)
print("   Maksimalna vrijednost BMI:", max_bmi_dijabetes)
