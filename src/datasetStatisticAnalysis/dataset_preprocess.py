
import numpy as np
import pandas as pd
from pandas import DataFrame
import time
import datetime

import matplotlib.pyplot as plt

''' 整合数据预处理方法
'''
class DataSetPreprocess:
	def __init__(self):
		pass

	'''
		加载数据集
	'''
	def loadDataSet(self, filename, columns=[], sep=','):
		if not len(columns):		# 必须匹配特征属性
			return None

		df = pd.read_csv(filename, sep=sep)

		return df
	
	'''
		将dataframe，按照columns列，保存到指定csv文件
	'''
	def saveDataSet(self, lst, columns, path='.', sep=','):
		df = DataFrame(np.array(lst), columns=columns)
		df.to_csv(path, sep=sep, index=False)

	'''
		计算np.ndarray类型的最大值和最小值及下标
		返回值格式：
			最大值，最大值下标，最小值，最小值下标
	'''
	def calMaxMin(self, np_arr):
		print(type(np_arr))
		if not isinstance(np_arr, np.ndarray):
			return None

		return np.max(np_arr), np.argmax(np_arr), np.min(np_arr), np.argmin(np_arr)

	'''
		计算两个时间点的差值
	'''
	def calTimeDiff(self, time1, time2):
		li_time1 = time1.split(':')
		li_time2 = time2.split(':')
		hour1, min1, secs1, milsec1 = li_time1
		hour2, min2, secs2, milsec2 = li_time2

		# 由于数据中只给出了“时分秒”，因此需要加上一个“年月日”，方便将日期转化为毫秒计算
		date1 = datetime.datetime(2018, 1, 1, int(hour1), int(min1), int(secs1))
		date2 = datetime.datetime(2018, 1, 1, int(hour2), int(min2), int(secs2))
		t1 = time.mktime(date1.timetuple()) * 1000
		t2 = time.mktime(date2.timetuple()) * 1000

		millsec1 = t1 + int(milsec1)
		millsec2 = t2 + int(milsec2)

		ret = (millsec2 - millsec1)

		return ret

	'''
		检测csv中的异常值
	'''
	def filterInvalidValue(self, df):
		if not isinstance(df, pd.core.frame.DataFrame):
			return False

		# 必须是dataframe
		# ret1 = df.isnull().any()	# 检测nan
		
		df = df.dropna(how='any', axis=0)		# 丢弃nan行
		
		''' 过滤大于阈值的异常值'''
		for name in df.columns:
			tmp = df[(np.abs(df[name]) > 1e+10) | ((0 < np.abs(df[name])) & (np.abs(df[name]) < 1e-10))]  # 检测异常值：inf、1e+300大数
			
			for index in tmp.index:
				print('-------->有异常值，index为：', index, '------异常值为', df.loc[index][name])
				df = df.drop(index=index, axis=0)

		return df

	'''
		按照csv_no字段，截取行，按块拆分数据集
	'''
	def truncStruct(self, df, columns=[], number=1):
		df_new = DataFrame(columns=columns)

		for i in range(0, len(df)):
			if df['csv_no'][i] == number:
				df_new.loc[i] = df.loc[i]

		return df_new

	def trunc(self, df, begin, end):
		if not isinstance(df, pd.core.frame.DataFrame):
			return None

		return df.iloc[begin:end, :]

	def minus(self, d1, d2):
		return d2-d1

	'''
		数据去重
	'''
	def deduplication(self, df, columns=[]):
		if not isinstance(df, pd.core.frame.DataFrame):
			return None

		df = df.drop_duplicates(subset=columns, keep="first", inplace=False)

		return df

	'''
		去矢量化
	'''
	def abs(self, df):
		if not isinstance(df, pd.core.frame.DataFrame):
			return None

		return df.abs()

	'''
		按列取平均值
		返回类型是Series
	'''
	def average(self, df):
		if not isinstance(df, pd.core.frame.DataFrame):
			return None

		return df.mean()

def plot():
	names = ["vibration_1", "vibration_2", "vibration_3", "current"]
	# filename = '../../datas/01-TrainingData-qLua/01/Sensor/1.csv'
	filename = 'plc-01_dataset.csv'
	df = pd.read_csv(filename, sep=',')
	data_processor = DataSetPreprocess()
	df = data_processor.deduplication(df, names)
	print(len(df))

	#df = DataFrame(columns=names)

	'''
	for i in range(1, 49):
		filename = '../../datas/01-TrainingData-qLua/01/Sensor/' + str(i) + '.csv'
		print(filename)
		data_processor = DataSetPreprocess(filename)

		sensor_df = data_processor.loadDataSet(columns=names)
		sensor_df = data_processor.abs(sensor_df)
		df = df.append(sensor_df)
		print(len(df))
	'''

	time_lst = [i for i in range(0, len(df))]
	# vibration_1_lst = np.array(df['vibration_1'])
	# vibration_2_lst = np.array(df['vibration_2'])
	# vibration_3_lst = np.array(df['vibration_3'])
	vibration_1_lst = []
	vibration_2_lst = []
	vibration_3_lst = []
	current_lst = np.array(df['current'])

	return df, time_lst, vibration_1_lst, vibration_2_lst, vibration_3_lst, current_lst


if __name__ == '__main__':
	df, time_lst, vibration_1_lst, vibration_2_lst, vibration_3_lst, current_lst = plot()

	plt.title('Data set parameter analysis')

	print(len(time_lst))
	print(len(df))

	time_lst = time_lst[60000:80000]
	current_lst = current_lst[60000:80000]

	import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot
	dgp = datasetGraphPlot.GraphPlot()
	# dgp.addSubPlot(time_lst, 'time(ms)', vibration_1_lst, 'vibration_1', 'green', 2, 2, 1)
	# dgp.addSubPlot(time_lst, 'time(ms)', vibration_2_lst, 'vibration_2', 'red', 2, 2, 2)
	# dgp.addSubPlot(time_lst, 'time(ms)', vibration_3_lst, 'vibration_3', 'skyblue', 2, 2, 3)
	# dgp.addSubPlot(time_lst, 'time(ms)', current_lst, 'current_lst', 'blue', 2, 2, 1)

	plt.scatter(time_lst, current_lst, marker='.')
	plt.xlabel("time(ms)")
	plt.ylabel("current")
	plt.show()

	# dgp.showSubplot()
