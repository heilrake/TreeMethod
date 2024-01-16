import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = pd.read_csv(url, usecols=['tmax', 'tmin', 'tavg', 'wetbulb', 'heat', 'cool', 'sealevel',
                                  'station_number', 'store_nbr', 'item_nbr', 'stnpressure', 'resultspeed',
                                  'sunrise', 'sunset', 'depart', 'dewpoint', 'units'])

data['units'] = np.log(data['units'] + 1)

X = data.drop('units', axis=1)
y = data['units']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X_train, y_train)

x_predict = X_test;
print(x_predict)

y_predict = tree_regressor.predict(X_test)
print(y_predict)

accuracy = r2_score(y_test, y_predict)
print('Accuracy of Decision Tree Regression - Data set: {:.2f}'.format(accuracy))
