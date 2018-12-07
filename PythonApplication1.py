import sys
import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.externals import joblib

def modelFit():
    #loaddata
    datasetloc = '/media/wenhan/KINGSTON/01-TrainingData-qLua/01/PLC/plc.csv'
    df = pd.read_csv(datasetloc)
    df['use_time'] = (240 - df['csv_no'] * 5)/240
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
    print("---->: ", X_train_scaled.mean(axis = 0))
    print("---->: ", X_train_scaled.std(axis = 0))

    X_test_scaled = scaler.transform(X_test)

    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

    hyperparams = {'randomforestregressor__max_features':['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth':[None, 5, 3, 1]}

    clf = GridSearchCV(pipeline, hyperparams, cv=10)
    clf.fit(X_train, y_train)
    print("best_params: ", clf.best_params_)

    joblib.dump(clf, "train_model.m")       # 保存模型
    # clf = joblib.load("train_model.m")    # 使用模型

    y_pred = clf.predict(X_test)

    lst_y_test = list(y_test)
    lst_y_pred = list(y_pred)
    for i in range(0, len(lst_y_test)):
        print(lst_y_test[i], "----", lst_y_pred[i])

    print("======")
    print (r2_score(y_test, y_pred))

def modelUsage():
    clf = joblib.load("train_model.m")

