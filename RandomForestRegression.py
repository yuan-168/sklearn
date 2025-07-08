import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


path_to_file = "/Users/shiyuanwang/Documents/machine learning/lesson/lesson 4/Practical 4/septic_shock.csv"
spetic_shock_data = pd.read_csv(path_to_file,encoding = 'utf-8')

plt.scatter(spetic_shock_data.index,spetic_shock_data['los'])
plt.show()

spetic_shock_data = spetic_shock_data.loc[spetic_shock_data.los<10] 
spetic_shock_data_d = spetic_shock_data.loc[:,['age','comorbidity_elixhauser','sofa','los']]

spetic_shock_data_d.describe()

mask = np.random.rand(len(spetic_shock_data_d))<0.9
train = spetic_shock_data_d[mask]
test = spetic_shock_data_d[~mask]

X_train = train.loc[:,['age','comorbidity_elixhauser','sofa']]
y_train = (train[['los']]).values.ravel()

X_test = test.loc[:,['age','comorbidity_elixhauser','sofa']]
y_test = (test[['los']]).values.ravel()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 42, n_estimators = 1000, max_features=4)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(y_pred.shape)
print(y_test.shape)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


print("Mean Square error: %.2f" %mean_squared_error(y_test, y_pred))
print("Root Mean Square error: %.2f" %sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute error: %.2f" %mean_absolute_error(y_test, y_pred))
print("R2 score : %.2f" %r2_score(y_test, y_pred))

feature_importance_vector = model.feature_importances_

feature_names=spetic_shock_data.columns[[0,4,5]]

plt.figure(1)
plt.title('Feature Importances')
plt.bar(range(len(feature_importance_vector)), feature_importance_vector)

plt.xticks(range(len(feature_importance_vector)), feature_names, rotation='vertical')
plt.ylabel('Relative Importance')
plt.show()

fig, ax = plt.subplots(1,2)
fig.suptitle('Actual and Predicted Hospital Length of Stays')
ax[0].plot(range(0,len(y_test)), y_test, 'bo', label = 'Actual')

ax[0].set(xlabel='Admission', ylabel='Actual Length of Stay')
plt.legend()
# Plot the predicted values

ax[1].plot(range(0,len(y_pred)), y_pred, 'ro')
ax[1].set(xlabel='Admission', ylabel='Predicted Length of Stay')

plt.legend()
# Graph labels
plt.tight_layout()
plt.show()
