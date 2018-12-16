
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess

def traverse(inputs):
    Out = []
    for input in inputs:
        multiple = int(input / 5)
        limit_low = multiple * 5
        limit_up = limit_low + 5
        middle = float((limit_low + limit_up) / 2)
        
        if input > middle:
            out = limit_up
        else:
            out = limit_low
        Out.append(out)
    
    return Out

if __name__ == '__main__':
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    filename = '../../../01-TrainingData-qLua/final.csv'
    dataset_df = pd.read_csv(filename, sep=',')
    
    dataset_df = dataset_df.dropna(how='any', axis=0)
    # quit()
    

    X = dataset_df[names[0:len(names)-1]]
    Y = dataset_df['last_time']
    print(X.columns)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=15)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()       # 数据标准化
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    
    # from sklearn.linear_model import LinearRegression
    # clf = LinearRegression()
    # clf.fit(X_train, Y_train)
    
    # 模型训练
    from sklearn import svm
    from sklearn.model_selection import cross_val_score     # 交叉检验
    # clf = svm.SVR()
    # scores = cross_val_score(clf, X_train, Y_train, cv=6)   # 交叉验证次数为6
    # clf.fit(X_train, Y_train)
    
    from sklearn.tree import DecisionTreeRegressor
    # clf = DecisionTreeRegressor(max_depth=7)
    # clf.fit(X_train, Y_train)
    
    from sklearn.ensemble import AdaBoostRegressor
    '''
        默认为CART树
        loss：指定误差计算方式:”linear”, “square”,“exponential”, 默认为”linear”
        n_estimators:值过大可能会导致过拟合， 一般50~100比较适合， 默认50
        learning_rate:一般从一个比较小的值开始进行调参； 该值越小表示需要更多的弱分类器
    '''
    clf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8, splitter='random'),
                            n_estimators=50, learning_rate=0.3,
                            loss='linear')
    
    from sklearn.ensemble import GradientBoostingRegressor
    # clf = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=10)
    
    
    clf.fit(X_train, Y_train)
    
    # 模型预测
    y_predict = clf.predict(X_test)
    y_train_predict = clf.predict(X_train)
    
    from sklearn.metrics import accuracy_score
    # # 模型效果评估
    # acc_train = clf.score(X_train, Y_train)
    # acc_test = clf.score(X_test, Y_test)
    # print('------>训练集上准确率:', acc_train)
    # print('------>测试集上准确率:', acc_test)

    from sklearn.metrics import r2_score
    ypre = traverse(y_predict)                  # 对预测的结果进行数值转换
    ytrainpre = traverse(y_train_predict)
    print(y_predict)
    print(ypre)
    
    acc_predict = r2_score(Y_train, ytrainpre)      # 训练集预测的r^2
    acc_pre = r2_score(Y_test, ypre)            # 测试集预测的r^2

    print('------>训练的准确率:', acc_predict)
    print('------>预测的准确率:', acc_pre)
    
    # 任取区间100的数据进行可视化
    # print(list(Y_train[0:100]))
    # print(list(Y_test[0:100]))
    # print(list(ypre[0:100]))
    xlst = [i for i in range(700, 800)]
    ytestlst = list(Y_test)[700:800]
    y_predictlst = list(ypre)[700:800]
    
    # 可视化
    import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot
    dgp = datasetGraphPlot.GraphPlot()
    dgp.plotScatter(x_lst=xlst, y_lst=ytestlst, y_label='y_test')
    # dgp.plotScatter(x_lst=xlst, y_lst=y_predictlst, y_label='y_predict')
    dgp.plot(xlst, y_predictlst, 'y_predict')
    dgp.show()

    
    