import pandas as pd

data = pd.read_csv ( "OSU_LV3/data_C02_emission.csv")
print("a)")
print(len(data))
print(data.info())
print ( data . isnull () . sum () )
#data . drop_duplicates ()
#data = data . reset_index ( drop = True )
cols = ['Make', 'Model', 'Vehicle Class', 'Transmission','Fuel Type']
data[cols]=data[cols].astype('category')
print(data.info())

print("b)")
new_data=data.sort_values(by=['Fuel Consumption City (L/100km)'])
print ( new_data . head ( 3 ).iloc[:, [0 ,1 , 7]])
print ( new_data . tail ( 3 ).iloc[:, [0 ,1 , 7]] )

print("c)")
filtered_data=data[(data['Engine Size (L)']>2.4)&(data['Engine Size (L)']<3.6)]
print(len(filtered_data))
print(filtered_data['CO2 Emissions (g/km)'].mean())

print("d)")
filtered_data2=data[(data['Make']=='Audi')]
print(len(filtered_data2))
filtered_data2=data[(data['Make']=='Audi')&(data['Cylinders']==4)]
print(filtered_data2['CO2 Emissions (g/km)'].mean())

print("e)")
new_data3 = data . groupby ('Cylinders')
print(new_data3.size())
print(new_data3['CO2 Emissions (g/km)'].mean())

print("f)")
print(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean())
print(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean())
print(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median())
print( data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median())

print("g)")
print(data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")].sort_values(by=["Fuel Consumption City (L/100km)"], ascending=False).head(1))

print("h)")
print(data[data["Transmission"].str.startswith("M")].__len__())

print("i)")
print(data.corr(numeric_only=True))

#velika korelacija se moze vidjeti u tome da velicina motora i broj cilindara su oko 0.7-1 dok je potrosnja oko 0.8
#negativna korelacija je zbog tog sto auto vise trosi broj je manji