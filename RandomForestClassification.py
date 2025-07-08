import pandas as pd
import numpy as np
from pprint import pprint

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

path_to_file = "/Users/shiyuanwang/Documents/machine learning/lesson/lesson 4/Practical 4/septic_shock.csv"
spetic_shock_data = pd.read_csv(path_to_file,encoding='utf-8')

X = spetic_shock_data.loc[:,['respiration','coagulation','liver','renal','cardio','cns']]
y = spetic_shock_data.loc[:,['hospital_mortality']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)

classifier = RandomForestClassifier(random_state = 42, n_estimators = 10000,min_samples_leaf = 100, max_features = 5,criterion='entropy')

print('Parameters currently in use:\n')

pprint(classifier.get_params())

classifier.fit(X_train, y_train.values.ravel())  

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

class_names  = ["Surived Hospital", "Died in Hospital"]

disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion Matrix")

print(disp.confusion_matrix)

plt.show()

feature_importance_vector = classifier.feature_importances_
pprint(feature_importance_vector)
feature_names = septic_shock_data.columns[[6,7,8,9,10,11]]

plt.figure(1)
plt.title('Feature Importances')
plt.bar(range(len(feature_importance_vector)),feature_importance_vector)
plt.xticks(range(len(feature_importance_vector)), feature_names)
plt.ylabel('Relative Importance')
plt.show()
