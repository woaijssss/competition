
import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt

from src.datasetStatisticAnalysis.dataset_preprocess import DataSetPreprocess

def tructDatase(filename, names, number):
	filename = filename
	names = names
	data_processor = DataSetPreprocess()
	plc_df = data_processor.loadDataSet(filename, columns=names)
	df_new = data_processor.truncStruct(plc_df, columns=names, number=number)
	# df_new.to_csv('../datas/test/plc_01.csv', index=False)

	return df_new

def testTrunc():
	filename = '../datas/01-TrainingData-qLua/01/PLC/plc.csv'
	names = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no']
	df = DataFrame(columns=names)
	df = pd.concat([df, tructDatase(filename, names=names, number=1).copy(deep=True)], axis=0)
	df = pd.concat([df, tructDatase(filename, names=names, number=2).copy(deep=True)], axis=0)
	df.to_csv('../datas/test/plc_01.csv', index=False)

	quit(0)


def test():
    arr1 = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ])
    
    arr2 = np.array([
        ['a', 'a', 'a'],
        ['b', 'b', 'b'],
        ['c', 'c', 'c']
    ])
    
    name1 = ['a11', 'a12', 'a13']
    name2 = ['a21', 'a22', 'a23']
    d1 = DataFrame(arr1, columns=name1)
    d2 = DataFrame(arr2, columns=name2, index=[4, 5, 6]).reset_index(drop=True)
    
    df = DataFrame(columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23'])
    
    import time
    
    t1 = time.time()
    
    # d2 = d2.reset_index(drop=True)
    # d = pd.concat([d1, d2], axis=1)     # 0.0010988712310791016 s
    # df = pd.concat([df, d], axis=0)
    t2 = time.time()
    print(df)
    print(t2-t1)
    
    # d = pd.concat([d1, d2], axis=1)
    # df = pd.concat([df, d], axis=0)
    # df = df.abs()	# 出现NAN，这里会报错
    # df = df.reset_index(drop=True)
    quit(0)

def testDfCopy():
    arr1 = np.array([
        ['a', 'a', 'a'],
        ['b', 'b', 'b'],
        ['c', 'c', 'c']
    ])
    
    name1 = ['a11', 'a12', 'a13']
    df = DataFrame(arr1, columns=name1)
    
    series_1 = df.loc[0]
    
    import time
    
    t1 = time.time()
    lst = list(series_1)
    seq_lst = []
    for j in range(0, 7770):
        seq_lst.append(lst)
    '''
    for j in range(0, 7770):
        if j == 10000:
            print(j)
        df_new.loc[j] = series_1  # 将plc的数据，复制n行，使其与sensor要截取的行数相等
    '''
    # df_new = df_new.append(seq_lst)
    df_new = DataFrame(np.array(seq_lst), columns=name1)
    
    t2 = time.time()
    print(t2-t1)
    print(df_new)
    
    quit(0)

if __name__ == '__main__':
    test()
    # testTrunc()
    # testDfCopy()
    
    dir_name = '../../01-TrainingData-qLua/01'		# 目标数据目录
    # plc数据的列
    names = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no']
    # 用于保存结果的列
    new_names = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    # 去掉time列和csv_no列之后的列值
    new_names_after_drop = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    data_processor = DataSetPreprocess()
    # plc_df = data_processor.loadDataSet(dir_name + '/plc_01.csv', columns=names)
    plc_df = data_processor.loadDataSet(dir_name + '/PLC/plc_test.csv', columns=names)
    plc_df_tmp = DataFrame(columns=names)		# PLC用于合并的结构
    # sensor数据的列
    sensor_columns = ['vibration_1', 'vibration_2', 'vibration_3', 'current']
    basic_n = 777		# 根据plc和sensor的采集频率计算的采集周期倍数
    seek_ptr = 0		# 记录sensor_df中，已经截取到哪一行
    sensor_df = DataFrame(columns=sensor_columns)		# 处理数据过程中，用于记录sensor数据的dataframe
    '''
        用于记录sensor数据可截取的剩余长度，随着sensor_df的赋值而初始化一次，随着sensor_df的清空而置0
    '''
    sensor_len = 0
    all_time = float(240 * 60  * 1000)		# 毫秒级时间
    df_new = DataFrame(columns=new_names)		# 处理后，需要保存的dataframe
    # for i in range(27, 28):
    for i in range(1, len(plc_df)):
        '''
            步骤1：取得plc中需要比较的相邻两行数据
        '''
        series_i_1 = plc_df.loc[i-1]		# 上一行的信息
        time_i_1, spindle_load_i_1, x_i_1, y_i_1, z_i_1, csv_number_i_1 = series_i_1
        series_i = plc_df.loc[i]			# 当前行的信息
        time_i, spindle_load_i, x_i, y_i, z_i, csv_number_i = series_i
        # print(time_i, spindle_load_i, x_i, y_i, z_i, csv_number_i)
        '''
            步骤2：确认是否属于同一sensor时间段的数据
        '''
        if csv_number_i > csv_number_i_1:		# 表示在不同sensor文件切换，跳过
            sensor_df.drop(sensor_df.index, inplace=True)
            # sensor_df = DataFrame(columns=sensor_columns)
            sensor_len = 0
            seek_ptr = 0
            continue
        else:									# 以下文件都是对同一个sensor数据做处理
            if sensor_df.empty:
                print('读取第 %d 个csv文件' % csv_number_i)
                #  按照plc.csv中的csv_no读取对应的sensor数据
                sensor_df = data_processor.loadDataSet(dir_name + '/Sensor/' + str(csv_number_i) + '.csv', columns=sensor_columns)
                # sensor_df = data_processor.abs(sensor_df)		# 去矢量化
                sensor_len = len(sensor_df)
            time_diff = data_processor.calTimeDiff(time_i_1, time_i)		# 计算两行的时间差，为增加last_time方便
            print("plc数据中----第 %d 行与 %d 行的时间差为： %d ms" % (i, i-1, time_diff))
            time_n = int(time_diff * basic_n)		# 在某一时间差内，plc的一行数据对应sensor中数据的行数n
            if not time_n:
                print('第 %d 行与第 %d 行的时间差为0' % (i, i-1))
                continue
            '''
                当处理每个一分钟的最后一笔数据时，可能会出现sensor文件的剩余数据小于 777*n 的情况，这里是按照实际情况进行合并；
                但是，此数据有可能属于异常数据，因为如果要严格使用plc的33hz对应sensor的25600hz的数据，那么这笔数据是不符合规则的，
                考虑是否需要去掉？
            '''
            if sensor_len <= time_n:
                time_n = sensor_len
            print('------>plc对应sensor中的数据行数: ', time_n)
            '''
                步骤3：对plc数据进行复制，同时对sensor数据按照相应行数进行截取
            '''
            # plc_df_tmp.drop(plc_df_tmp.index, inplace=True)
            # for j in range(0, time_n):
            # 	if j == 10000:
            # 		print(j)
            # 	plc_df_tmp.loc[j] = series_i_1		# 将plc的数据，复制n行，使其与sensor要截取的行数相等
            '''
                这里做一下切片操作，去掉time的列；
                因为后面计算abs()时，需要数值型，而time是字符型，带上time来构造dataframe，会影响dtype的类型；
            '''
            try:
                lst = list(series_i_1[1::])
                seq_lst = []
                for j in range(0, time_n):
                    seq_lst.append(lst)
                plc_df_tmp = DataFrame(np.array(seq_lst), columns=names[1::], dtype=np.float64)
                print('plc_df_tmp复制了 %d 次， 复制后的大小：%d' % (time_n, len(plc_df_tmp)))
            except Exception as e:
                print('========>:', e)
                continue
            
            # 截取sensor_df中对应的行
            sensor_df_tmp = sensor_df.iloc[seek_ptr:time_n+seek_ptr, :]
            print('从 %d 处开始截取，截取的sensor数据长度: %d； 截取后的长度为：%d' % (seek_ptr, time_n, len(sensor_df_tmp)))
            '''
                按照两行的间隔时间time_diff和行数time_n，对损失时间，生成等差数列
            '''
            diff_arr = np.linspace(all_time, all_time - time_diff, time_n)
            all_time -= time_diff
            diff_df = DataFrame(diff_arr, columns=['last_time'])
            print('时间差的等差数列大小：%d ； diff_df大小：%d' % (len(diff_arr), len(diff_df)))
            seek_ptr += time_n		# 更新记录指针的位置
            sensor_len -= time_n
            '''
                步骤4：将一次合并的结果，追加到新的dataframe中
            '''
            plc_df_tmp = plc_df_tmp.reset_index(drop=True)
            sensor_df_tmp = sensor_df_tmp.reset_index(drop=True)
            diff_df = diff_df.reset_index(drop=True)
            df = pd.concat([plc_df_tmp, sensor_df_tmp, diff_df], axis=1)		# 合并plc、sensor数据和剩余时间数据
            print('需要合并的df的大小（后）:', len(df))
            print('第 %d 次合并前的plc_df_tmp, sensor_df_tmp大小: %d, %d' % (i, len(plc_df_tmp), len(sensor_df_tmp)))
            print('第 %d 次合并前的dataframe大小: %d' % (i, len(df_new)))
            df_new = pd.concat([df_new, df], axis=0)					# 将合并后的结果，追加到新的dataframe中用于保存
            df_new = df_new.reset_index(drop=True)
            print('第 %d 次合并后的dataframe大小: %d' % (i, len(df_new)))
    '''
        保存之前要做：
        （1）将time, csv_no的列去掉
        （2）对原plc的列数据进行去矢量化，
    '''
    df_new = df_new.drop(['time', 'csv_no'], axis=1)
    df_new = data_processor.abs(df_new)
    '''
        步骤5：保存合并数据后的dataframe
    '''
    df_new.to_csv(dir_name + '/new.csv', columns=new_names_after_drop, sep=',', index=False)



