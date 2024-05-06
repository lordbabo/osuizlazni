import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score,root_mean_squared_error


dataframe = pd.read_csv('OSU_LV4/data_C02_emission.csv')

ohe = OneHotEncoder()
 
X=dataframe[['Fuel Type','Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
y=dataframe['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

X_encoded_train = ohe.fit_transform(X_train[['Fuel Type']]).toarray ()
X_encoded_test = ohe.fit_transform(X_test[['Fuel Type']]).toarray ()

linearModel=lm.LinearRegression()
linearModel.fit(X_encoded_train,y_train)

y_predict=linearModel.predict(X_encoded_test)

print("MSE:", mean_absolute_error(y_test , y_predict))
print("MSE:", mean_squared_error(y_test , y_predict))
print("MAPE:", mean_absolute_percentage_error(y_test , y_predict))
print("MSE:", root_mean_squared_error(y_test , y_predict))
print("R_TWO_SCORE:",r2_score(y_test,y_predict))


