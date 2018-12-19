
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

if __name__ == '__main__':
    filterValidData()