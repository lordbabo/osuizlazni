import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv ( "OSU_LV3/data_C02_emission.csv")

#a
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind="hist")
#mozemo vidjeti da je najvise emisija oko 200 i 300 g/km

#b
data['Fuel Type'] = data['Fuel Type'].astype('category')
data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c="Fuel Type", cmap='inferno')
#mozemo jasno u grafu vidjeti raspodijelu emisija po kategoriji
#benzin ima malu do srednju emisiju dok premium benzin moze imati i vrlo velike, ethanol i dizel su oko 200-300

#c
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
#

#d
plt.figure()
data.groupby(by=['Fuel Type'])['Make'].count().plot(kind='bar').set_title('Number of cars by fuel type')

#e
plt.figure()
data.groupby(by=['Cylinders'])['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.show()