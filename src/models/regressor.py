
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.util as util

class Regressor:
    _clf = None
    X_train, X_test, Y_train, Y_test = None, None, None, None
    _y_predict, _y = None, None
    _y_predict_tra, _y_tra = None, None
    _y_predict_best = []

    def __init__(self):
        pass

    '''
        分割数据集
    '''
    def splitDataSet(self, X, Y, test_size=0.3, random_rate=10):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_rate)

    '''
        数据标准化
    '''
    def trainTestStandard(self):
        ss = StandardScaler()  # 数据标准化
        self.X_train = ss.fit_transform(self.X_train)
        self.X_test = ss.transform(self.X_test)

    '''
        模型选择、模型训练
    '''
    def trainModel(self, estimator=67, max_depth=15, rate=0.01):
        # self._clf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth, splitter='random'),
        #                         n_estimators=50, learning_rate=rate,
        #                         loss='square')
        self._clf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth, splitter='best'),
                                n_estimators=estimator, learning_rate=rate,
                                loss='square')
        
        # from sklearn.linear_model import LinearRegression
        # self._clf = LinearRegression()
        
        # from sklearn.svm import SVR
        # self._clf = SVR(kernel='rbf', gamma=0.1, C=1.0)
        '''
            此处可以加上网格搜索和交叉验证
        '''
        # param_grid = {
        #     'n_estimators': [i for i in range(estimator, estimator+5)],
        #     'learning_rate': np.linspace(rate, rate+1, 5)
        # }
        # self._clf = GridSearchCV(clf, param_grid=param_grid, cv=2, n_jobs=1)
        self._clf.fit(self.X_train, self.Y_train)

    '''
        模型预测
    '''
    def predict(self):
        self._y_predict = self._clf.predict(self.X_test)        # 测试集预测结果
        self._y = self._clf.predict(self.X_train)               # 训练集预测结果

        return self._y, self._y_predict
    
    def cvBestParams(self):
        self._y_predict_tra = util.traverse(self._y_predict)    # 测试集预测结果数值转化
        self._y_tra = util.traverse(self._y)                    # 训练集预测结果数值转换

        print('----->效果最好的参数：', self._clf.best_params_)

    '''
        模型评估
    '''
    def evaluate(self):
        self._y_predict_tra = util.traverse(self._y_predict)    # 测试集预测结果数值转化
        self._y_tra = util.traverse(self._y)                    # 训练集预测结果数值转换

        acc_y_predict = r2_score(self.Y_test, self._y_predict_tra)  # 测试集预测的r^2
        acc_y = r2_score(self.Y_train, self._y)                     # 训练集预测的r^2

        logging.debug('------>训练的准确率: %f' % acc_y)
        logging.debug('------>预测的准确率: %f' % acc_y_predict)

        return acc_y, acc_y_predict

    '''
        预测结果可视化
    '''
    def visualization(self, y_predict):
        # 任取区间100的数据进行可视化
        # xlst = [i for i in range(700, 800)]
        # ytestlst = list(self.Y_test)[700:800]
        # y_predictlst = list(y_predict)[700:800]
        xlst = [i for i in range(0, len(self.Y_test))]
        ytestlst = list(self.Y_test)
        y_predictlst = list(y_predict)

        # 可视化
        import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot
        dgp = datasetGraphPlot.GraphPlot()
        dgp.plotScatter(x_lst=xlst, y_lst=ytestlst, y_label='y_test')
        # dgp.plotScatter(x_lst=xlst, y_lst=y_predictlst, y_label='y_predict')
        # dgp.plot(xlst, y_predictlst, 'y_predict')
        dgp.show()

    def save(self, path='./model.m'):
        joblib.dump(self._clf, path)

    def load(self, path='./model.m'):
        self._clf = joblib.load(path)
        return self._clf

if __name__ == '__main__':
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    filename = '../../../01-TrainingData-qLua/final.csv'
    dp = dataset_preprocess.DataSetPreprocess()
    dataset_df = dp.loadDataSet(filename=filename, columns=names)
    dataset_df = dp.filterInvalidValue(dataset_df)      # 去除

    regressor = Regressor()
    regressor.trainModel()

    quit(0)
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

    
    