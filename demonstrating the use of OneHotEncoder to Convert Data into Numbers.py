#demonstrating the use of OneHotEncoder to Convert Data into Numbers
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

car_values= pd.read_csv("car-sales-extended.csv")

X= car_values.drop("Price", axis=1)
y= car_values["Price"]
print(X.dtypes);

#Doors and Odometer are already integer values. So we just have to change the colour and the make parameters

one_hot= OneHotEncoder()

transformer= ColumnTransformer([("one_hot", one_hot, ["Make", "Colour"])], remainder="passthrough")
transformed_data= transformer.fit_transform(X)

X_train, X_test, y_train, y_test= train_test_split(transformed_data, y, test_size= 0.2)
transformed_data= pd.DataFrame(transformed_data)

transformed_data.to_csv("transformed_data.csv")