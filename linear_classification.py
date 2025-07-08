import pandas as pd
rawData = pd.read_csv("/Users/shiyuanwang/Documents/machine learning/lesson/lesson 3/Practical 3-20240725/bc_dataset.csv")
rawData.head()
rawData.shape
rawData = rawData.astype(float)
X = rawData.loc[:,['Clump_Thickness','Cell_Size_Uniformity','Cell_Shape_Uniformity','Marginal_Adhesion','Single_Epi_Cell_Size','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
y = rawData.loc[:, ['Class']]

#try different SVM kernels,
linear = svm.SVC(kernel = 'linear')
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)
print("Classification report of linear SVM : ")
print(classification_report(y_test, y_pred))

rbf = svm.SVC(kernel = 'rbf',gamma='scale')
rbf.fit(X_train,y_train)
y_pred = rbf.predict(X_test)
print("Classification report of RBF SVM : ")
print(classification_report(y_test, y_pred))

sigmoid = svm.SVC(kernel = 'rbf',gamma='scale',C = 1.4)
sigmoid.fit(X_train,y_train)
y_pred = sigmoid.predict(X_test)
print("Classification report of sigmoid SVM : ")
print(classification_report(y_test, y_pred))

#Implementing an RF for Comparison
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)
clf = RandomForestClassifier()
clf.fit(X, y)
y_pred = sigmoid.predict(X_test)
print("Classification report : ")
print(classification_report(y_test, y_pred))

#Tune and Test
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)

param_grid = {'C': [0.1, 1, 10, 100],  
              'kernel': ['linear']}  
   
grid = GridSearchCV(svm.SVC(), param_grid,cv=5, refit = True, verbose = 3,n_jobs=-1) 

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print("Classification report of linear SVM : ")
print(classification_report(y_test, y_pred))