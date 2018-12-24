from sklearn.model_selection import train_test_split

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.models.regressor as regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib

import logging
import time
import warnings
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

import src.models.regressor as regressor

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./record.log',
                    filemode='w')


def printDescribe(df):
    print(df['spindle_load'].describe())
    print('------')
    print(df['x'].describe())
    print('------')
    print(df['y'].describe())
    print('------')
    print(df['z'].describe())
    print('------')
    print(df['vibration_1'].describe())
    print('------')
    print(df['vibration_2'].describe())
    print('------')
    print(df['vibration_3'].describe())
    print('------')
    print(df['current'].describe())


if __name__ == '__main__':
    names = ['spindle_load', 'x', 'y', 'z', 'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
    # filename = '../../02-TestingData-poL3/result01_new.csv'
    regressor = regressor.Regressor()
    clf = regressor.load()
    
    result = []
    
    for file in range(1, 6):
        filename = '../datas/02-TestingData-poL3/二次处理/result0' + str(file) + '_new1.csv'
        dp = dataset_preprocess.DataSetPreprocess()
        
        X = dp.loadDataSet(filename=filename, columns=names)
        df = X.copy()
        
        csv_no_lst = X['csv_no']  # 取出csv_no列，用于对预测结果的计算
        X = X.drop('csv_no', axis=1)
        
        X = dp.filterInvalidValue(X)
        X = dp.abs(X)                   # 训练的时候去矢量了，测试的时候也要去矢量
        
        # printDescribe(X)
        
        # X = X.drop(['x', 'y', 'current'], axis=1)
        
        from sklearn.preprocessing import StandardScaler
        
        ss = StandardScaler()  # 数据标准化
        X_train = ss.fit_transform(X)

        y_predict = clf.predict(X_train)
        print('------->', y_predict)
        
        Y = []      # 预测值计算的最终结果
        for i in range(0, len(csv_no_lst)):
            y_new = y_predict[i] - (10 - csv_no_lst[i]) * 5
            Y.append(y_new)
            # Y.append(y_predict[i])
            
        # Y = [i for i in Y if i >= 0]

        import numpy as np
        Y = np.abs(Y)

        import numpy as np
        import src.util as util
        from pandas import Series, DataFrame
        Y = util.traverse(Y)
        
        # ydf = DataFrame(np.array(Y), columns=names)
        
        print('====>:', list(csv_no_lst))
        print('====>:', Y)
        print('====>:', y_predict)
        Y_arr = np.array(Y)
        print('均值----->:', np.mean(Y_arr))
        result.append(np.mean(Y_arr))
        from scipy import stats
        print('众数----->:', stats.mode(Y_arr)[0][0])
        print('中位数----->:', np.median(Y_arr))
        print('方差----->:', np.var(Y_arr))
        print('最大值----->:', np.max(Y_arr))
        print('最小值----->:', np.min(Y_arr))
        
        print(Y_arr[Y_arr < 0])
        
        df['last_time'] = Series(Y_arr)
        # tmp = df[(df['last_time'] > 240) | (df['last_time'] < 0)]
        # for index in tmp.index:
        #     df = df.drop(index=index, axis=0)
        print('---------------------------------')
        print(df['last_time'].describe())
        print('---------------------------------')
        df.to_csv('./result0' + str(file) + '_pre_before_traverse.csv', sep=',', columns=df.columns, index=False)
        
        # xlst = [i for i in range(700, 800)]
        # y_predictlst = list(y_predict)[700:800]
        xlst = [i for i in range(0, len(Y))]
        import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot
        
        dgp = datasetGraphPlot.GraphPlot()
        dgp.plotScatter(x_lst=xlst, y_lst=Y, y_label='y_predict')
        # dgp.plot(xlst, Y, 'y_predict')
        # dgp.show()
    
    print(result)
