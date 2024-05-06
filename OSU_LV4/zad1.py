from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import sklearn . linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score,root_mean_squared_error


#a

dataframe = pd.read_csv('OSU_LV4/data_C02_emission.csv')

X=dataframe[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
y=dataframe['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#b

plt.figure()
plt.scatter(y_train,X_train['Fuel Consumption City (L/100km)'],color='blue',s=10,alpha=0.4)
plt.scatter(y_test,X_test['Fuel Consumption City (L/100km)'],color='red',s=10,alpha=0.4)
plt.show()

#c standardizacija

sc = MinMaxScaler ()
X_train_n = sc . fit_transform ( X_train )

X_train['Fuel Consumption City (L/100km)'].plot(kind='hist',bins=25)
plt.show()
plt.hist(X_train_n[:,2], bins=25)
plt.show()

X_test_n = sc . fit_transform ( X_test )

#d

linearModel = lm . LinearRegression ()
linearModel . fit ( X_train_n , y_train )
print(linearModel.coef_)

#e

y_predict=linearModel.predict(X_test_n)
plt.figure()
plt.xlabel("Real")
plt.ylabel("Actual")
plt.scatter(x=y_test,y=y_predict)
plt.show()

#f

print("MSE:", mean_absolute_error(y_test , y_predict))
print("MSE:", mean_squared_error(y_test , y_predict))
print("MAPE:", mean_absolute_percentage_error(y_test , y_predict))
print("MSE:", root_mean_squared_error(y_test , y_predict))
print("R_TWO_SCORE:",r2_score(y_test,y_predict))

