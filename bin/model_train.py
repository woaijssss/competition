from sklearn.model_selection import train_test_split

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.models.regressor as regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import logging
import time

import src.models.regressor as regressor

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./record.log',
                    filemode='w')

if __name__ == '__main__':
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    filename = '../../01-TrainingData-qLua/final.csv'
    dp = dataset_preprocess.DataSetPreprocess()
    dataset_df = dp.loadDataSet(filename=filename, columns=names)
    
    dataset_df = dp.filterInvalidValue(dataset_df)
    
    # 选择特征和标签
    X = dataset_df[names[0:len(names) - 1]]
    Y = dataset_df['last_time']
    
    regressor = regressor.Regressor()
    regressor.splitDataSet(X, Y, test_size=0.3, random_rate=15)  # 拆分训练集
    regressor.trainTestStandard()  # 数据标准化
    
    '''
        默认为CART树
        loss：指定误差计算方式:”linear”, “square”,“exponential”, 默认为”linear”
        n_estimators:值过大可能会导致过拟合， 一般50~100比较适合， 默认50
        learning_rate:一般从一个比较小的值开始进行调参； 该值越小表示需要更多的弱分类器
    '''
    import numpy as np
    
    # learning_rate = np.linspace(0.01, 0.1, 1)
    learning_rate = [0.01]
    i = 0
    result = [0, 0, 0]
    params = [-1, -1, -1]
    y_predict_best = []  # 预测中最好的y值
    
    '''
        当前最好的参数
            第 37 轮---estimator为 61， max_depth为 10， learning_rate为：0.733333------>训练的准确率:0.882113
            第 0 轮---estimator为 61， max_depth为 10， learning_rate为：0.733333------>预测的准确率:0.880970
        
        较好的参数：
            第 0 轮---estimator为 67， max_depth为 15， learning_rate为：0.010000------>训练的准确率:0.951467
            第 0 轮---estimator为 67， max_depth为 15， learning_rate为：0.010000------>预测的准确率:0.950328
    '''
    
    for rate in learning_rate:
        for max_depth in range(15, 16):
            for estimator in range(67, 68):
                t1 = time.time()
                i += 1
                
                regressor.trainModel(estimator, max_depth, rate)  # 模型训练
                y, y_predict = regressor.predict()  # 模型预测
                acc_y, acc_y_predict = regressor.evaluate()  # 模型评估
                
                logging.debug('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>训练的准确率:%f' % (
                i, estimator, max_depth, rate, acc_y))
                logging.debug('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测的准确率:%f' % (
                i, estimator, max_depth, rate, acc_y_predict))
                logging.debug('---------------------------')
                print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>训练的准确率:%f' % (
                i, estimator, max_depth, rate, acc_y))
                print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测的准确率:%f' % (
                i, estimator, max_depth, rate, acc_y_predict))
                print('---------------------------')
                # if acc_predict >= acc_pre:
                if acc_y > result[1] and acc_y_predict > result[2]:
                    print('第 %d 轮优于第 %d 轮，替换。。。' % (i, i - 1))
                    result[0], result[1], result[2] = i, acc_y, acc_y_predict
                    params[0], params[1], params[2] = estimator, max_depth, rate
                    # regressor._y_predict_tra = y_predict  # 替换最优预测值
                    import src.util as util
                    regressor._y_predict_best = util.traverse(y_predict)  # 替换最优预测值
                
                t2 = time.time()
                print('耗时:', t2 - t1)
    
    print('最终预测最好的为:', result)
    print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>训练的准确率:%f'
          % (result[0], params[0], params[1], params[2], result[1]))
    print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测的准确率:%f'
          % (result[1], params[0], params[1], params[2], result[2]))
    
    regressor.visualization(regressor._y_predict_best)  # 预测结果可视化
    regressor.save()  # 模型保存
