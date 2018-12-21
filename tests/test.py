
from pandas import DataFrame, Series
import numpy as np

import src.samplingSensorData as samplingSensorData
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess

def filterValidData():
    # names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    # filename = '../../01-TrainingData-qLua/final_new.csv'
    names = ['spindle_load', 'x', 'y', 'z', 'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
    filename = '../datas/02-TestingData-poL3/result01.csv'
    dp = dataset_preprocess.DataSetPreprocess()
    df = dp.loadDataSet(filename=filename, columns=names)
    
    df = dp.abs(df)
    
    print(len(df))
    tmp = df[
        (df['vibration_1'] >= 100) | (df['vibration_2'] >= 100) | (df['vibration_3'] >= 100)
        # # | (df['spindle_load'] <= 0.5)
        # | (df['x'] <= 20) | (df['x'] >= 800)
        # # | (df['y'] >= 450) | (df['y'] <= 5)
        # | (df['z'] >= 440) | (df['z'] <= 150)
    ]

    print('----index:', len(tmp.index))

    print('---------->index:', tmp.index)
    
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
    df.to_csv('../datas/02-TestingData-poL3/result01_new_twice.csv', columns=df.columns, sep=',', index=False)
    
    quit()

import numpy as np
import pandas as pd

def calc_grade(true, predict): # take the result of last try as predict,take new predict as true
    Eri = true - predict
    if Eri <= 0:
        # mid = -np.log(0.5) * (Eri/5)
        mid = np.power(2, Eri/5)
    else:
        # mid = np.log(0.5) * (Eri/20)
        mid = np.power(0.5, Eri/20)
    # return np.exp(mid) * 100
    ret = mid * 100
    return ret

def test():
    x1 = 79 - (np.log2(0.02)) * 5
    x2 = 79 - (np.log(0.5 - np.log(0.02))) * 20
    
    print(x1)
    print(x2)
    
    # quit()
    
    x = [i for i in range(-100, 100)]
    y = []
    for x_ in x:
        if x_ <= 0:
            y_ = np.power(2, x_/5)
            # y_ = -np.log(0.5) * (x_/5)
        else:
            y_ = np.power(0.5, x_/20)
            # y_ =  np.log(0.5) * (x_/20)
        # y_ = -np.log(0.5) * (x_/5)
        # y_ = np.power(2, x_/5)
        # y_ = np.power(0.5, x_/20)
        y.append(y_)
        
    import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

    dgp = datasetGraphPlot.GraphPlot()
    dgp.plotScatter(x_lst=x, y_lst=y, y_label='y')
    dgp.show()
    quit()

if __name__ == '__main__':
    # test()
    
    data_loc = './result.csv'
    df = pd.read_csv(data_loc)
    column_list = df.columns
    first_diff = {}
    for j in range(2, df.shape[0]):
        first_try = 0
        second_try = 0
        third_try = 0
        for i in column_list:
            first_try += calc_grade(df[i][j],df[i][0])
            second_try += calc_grade(df[i][j],df[i][1])
            # third_try += calc_grade(df[i][j],df[i][2])
        # first_diff[j] = [first_try/5, second_try/5, third_try/5]
        first_diff[j] = [first_try/5, second_try/5]
        print('%d---->:' % j, first_diff[j])
    final_diff = {}
    print('---------------------------------------')
    for j in first_diff.keys():
        diff_1 = np.square(first_diff[j][0] - 15)
        diff_2 = np.square(first_diff[j][1] - 2.19)
        final_diff[j] = np.sqrt(diff_1 + diff_2)
        print('%d------>:' % j, final_diff[j])