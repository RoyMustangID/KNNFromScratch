from ml_from_scratch.neighbors import KNeighborsRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# fetch dataset 
abalone_data = pd.read_csv('abalone.csv')

# data (as pandas dataframes) 
X = abalone_data.drop('Rings', axis = 1)
y = abalone_data['Rings'] 

#preprocess
y_clean = y.values.tolist()
X_ohe = pd.get_dummies(X, columns = ['Sex'],dtype = float)
X_train, X_test, y_train, y_test = train_test_split(X_ohe, y_clean, test_size=0.2, random_state = 123)
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

#building class
knn = KNeighborsRegression(k_neighbor=15, p = 2)

#model training
knn.fit(X_train, y_train)

#predicting
ypred = knn.predict(X_test)

#evaluating
rmse = mean_squared_error(ypred, y_test, squared = False)