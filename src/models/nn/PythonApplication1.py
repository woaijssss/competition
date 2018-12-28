import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#loaddata
datasetloc = 'C:/Users/lzy/Desktop/data/01-TrainingData-additional/01/PLC/plc.csv'
df = pd.read_csv(datasetloc)
df['use_time'] = 240 - df['csv_no'] * 5
'''
print df.head()
print df.tail()
print df.describe()
'''

y = df.use_time
X = df.drop(['use_time','csv_no','time'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state = 123,stratify = y)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis = 0))
print(X_train_scaled.std(axis = 0))

X_test_scaled = scaler.transform(X_test)

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

hyperparams = {'randomforestregressor__max_features':['auto', 'sqrt', 'log2'],
               'randomforestregressor__max_depth':[None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparams, cv=10)
clf.fit(X_train, y_train)
print(clf.best_params_)

y_pred = clf.predict(X_test)
print (r2_score(y_test, y_pred))