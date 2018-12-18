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


def test():
    i = 1e10
    j = 1e+10
    print(i == j)
    quit()


if __name__ == '__main__':
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
    filename = '../../02-TestingData-poL3/result01.csv'
    
    dp = dataset_preprocess.DataSetPreprocess()
    regressor = regressor.Regressor()
    
    X = dp.loadDataSet(filename=filename, columns=names)
    
    csv_no_lst = X['csv_no']  # 取出csv_no列，用于对预测结果的计算
    X = X.drop('csv_no', axis=1)
    
    X = dp.filterInvalidValue(X)
    
    from sklearn.preprocessing import StandardScaler
    
    ss = StandardScaler()  # 数据标准化
    X_train = ss.fit_transform(X.values)
    
    clf = regressor.load()
    
    y_predict = clf.predict(X_train)
    
    Y = []      # 预测值计算的最终结果
    for i in range(0, len(csv_no_lst)):
        y_new = y_predict[i] - (10 - csv_no_lst[i]) * 5
        Y.append(y_new)
    
    import src.util as util
    Y = util.traverse(Y)
    
    print('====>:', list(csv_no_lst))
    print('====>:', Y)
    print('====>:', y_predict)
    import numpy as np
    Y_arr = np.array(Y)
    print('均值----->:', np.mean(Y_arr))
    from scipy import stats
    print('众数----->:', stats.mode(Y_arr)[0][0])
    print('中位数----->:', np.median(Y_arr))
    print('方差----->:', np.var(Y_arr))
    print('最大值----->:', np.max(Y_arr))
    print('最小值----->:', np.min(Y_arr))
    
    print(Y_arr[Y_arr < 0])
    
    # xlst = [i for i in range(700, 800)]
    # y_predictlst = list(y_predict)[700:800]
    xlst = [i for i in range(0, len(Y))]
    import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot
    
    dgp = datasetGraphPlot.GraphPlot()
    # dgp.plotScatter(x_lst=xlst, y_lst=y_predictlst, y_label='y_predict')
    dgp.plot(xlst, Y, 'y_predict')
    dgp.show()
