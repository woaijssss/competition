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


def checkDataSet(df):
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    for column in df.columns[0:len(df.columns) - 1]:
        lst = df[column]
        
        print(column + '======================================================')
        import numpy as np
        Y_arr = np.array(lst)
        print('均值----->:', np.mean(Y_arr))
        from scipy import stats
        print('众数----->:', stats.mode(Y_arr)[0][0])
        print('中位数----->:', np.median(Y_arr))
        print('方差----->:', np.var(Y_arr))
        print('最大值----->:', np.max(Y_arr))
        print('最小值----->:', np.min(Y_arr))
    
    quit()


def test(df):
    print(len(df))
    tmp = df[(df['vibration_1'] >= 100) | (df['vibration_2'] >= 100) | (df['vibration_3'] >= 100)]
    
    for i in tmp.index:
        df = df.drop(index=i, axis=0)
    
    print(len(df))
    
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
    
    # df.to_csv('../../01-TrainingData-qLua/final_new.csv', columns=df.columns, sep=',')
    
    quit()


if __name__ == '__main__':
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    names_new = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'current', 'last_time']
    # filename = '../datas/01-TrainingData-qLua/final_new.csv'
    filename = '../datas/01-TrainingData-qLua/二次处理/final_new1.csv'
    dp = dataset_preprocess.DataSetPreprocess()
    dataset_df = dp.loadDataSet(filename=filename, columns=names)
    
    # test(dataset_df)
    
    dataset_df = dp.filterInvalidValue(dataset_df)

    dataset_df = dp.abs(dataset_df)
    
    # dataset_df = dataset_df.drop(['vibration_2', 'vibration_3'], axis=1)
    names_new = names
    
    # checkDataSet(dataset_df)
    
    # 选择特征和标签
    X = dataset_df[names_new[0:len(names_new) - 1]]
    Y = dataset_df['last_time']
    
    regressor = regressor.Regressor()
    regressor.splitDataSet(X, Y, test_size=0.1, random_rate=15)  # 拆分训练集
    regressor.trainTestStandard()  # 数据标准化
    
    '''
        默认为CART树
        loss：指定误差计算方式:”linear”, “square”,“exponential”, 默认为”linear”
        n_estimators:值过大可能会导致过拟合， 一般50~100比较适合， 默认50
        learning_rate:一般从一个比较小的值开始进行调参； 该值越小表示需要更多的弱分类器
    '''
    import numpy as np
    
    # learning_rate = np.linspace(0.01, 0.1, 10)
    learning_rate = [0.02]
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
    
    for max_depth in range(7, 8):
        for rate in learning_rate:
            for estimator in range(98, 99):
                t1 = time.time()
                i += 1
                
                regressor.trainModel(estimator, max_depth, rate)  # 模型训练
                y, y_predict = regressor.predict()  # 模型预测
                acc_y, acc_y_predict = regressor.evaluate()  # 模型评估
                
                
                #---------------------------------------------------------------------------------------
                seq_result = []
                for file in range(1, 6):
                    filename = '../datas/02-TestingData-poL3/一次处理/result0' + str(file) + '_new.csv'
                    dp = dataset_preprocess.DataSetPreprocess()
    
                    X = dp.loadDataSet(filename=filename, columns=names)
                    df = X.copy()
    
                    csv_no_lst = X['csv_no']  # 取出csv_no列，用于对预测结果的计算
                    X = X.drop('csv_no', axis=1)
    
                    X = dp.filterInvalidValue(X)
                    X = dp.abs(X)  # 训练的时候去矢量了，测试的时候也要去矢量
    
                    # printDescribe(X)
    
                    # X = X.drop(['vibration_2', 'vibration_3'], axis=1)
    
                    from sklearn.preprocessing import StandardScaler
    
                    ss = StandardScaler()  # 数据标准化
                    X_train = ss.fit_transform(X)
    
                    y_predict = regressor._clf.predict(X_train)
                    print('------->', y_predict)
    
                    Y = []  # 预测值计算的最终结果
                    for i in range(0, len(csv_no_lst)):
                        y_new = y_predict[i] - (10 - csv_no_lst[i]) * 5
                        Y.append(y_new)
                        # Y.append(y_predict[i])
    
                    # Y = [i for i in Y if i >= 0]
    
                    import numpy as np
                    import src.util as util
                    from pandas import Series, DataFrame
    
                    # Y = util.traverse(Y)
    
                    # ydf = DataFrame(np.array(Y), columns=names)
    
                    print('====>:', list(csv_no_lst))
                    print('====>:', Y)
                    print('====>:', y_predict)
                    Y_arr = np.array(Y)
                    print('均值----->:', np.mean(Y_arr))
                    seq_result.append(np.mean(Y_arr))
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
                    df.to_csv('./result0' + str(file) + '_pre_before_traverse.csv', sep=',', columns=df.columns,
                              index=False)
    
                    # xlst = [i for i in range(700, 800)]
                    # y_predictlst = list(y_predict)[700:800]
                    xlst = [i for i in range(0, len(Y))]
                    import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot
    
                    dgp = datasetGraphPlot.GraphPlot()
                    dgp.plotScatter(x_lst=xlst, y_lst=Y, y_label='y_predict')
                    # dgp.plot(xlst, Y, 'y_predict')
                    # dgp.show()

                print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测结果:' % (
                    i, estimator, max_depth, rate), seq_result)
                logging.info('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f' % (i, estimator, max_depth, rate))
                logging.info('------>预测结果: ' + str(seq_result))
                #---------------------------------------------------------------------------------------
                
                
                
                
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
# regressor.cvBestParams()
