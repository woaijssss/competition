
import numpy as np
import pandas as pd
from pandas import DataFrame
import time
import datetime

import matplotlib.pyplot as plt

''' 整合数据预处理方法
'''
class DataSetPreprocess:
	_filename = ""

	def __init__(self, filename):
		self._filename = filename

	'''
		加载数据集
	'''
	def loadDataSet(self, columns=[], sep=','):
		if not len(columns):		# 必须匹配特征属性
			return None

		df = pd.read_csv(self._filename, sep=sep)

		return df

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
		t1 = time.mktime(date1.timetuple())*1000
		t2 = time.mktime(date2.timetuple())*1000

		millsec1 = t1 + int(milsec1)
		millsec2 = t2 + int(milsec2)

		return (millsec2 - millsec1)

	'''
		检测csv中的异常值
	'''
	def detectInvalidValue(self, df):
		if not isinstance(df, pd.core.frame.DataFrame):
			return False

		# 必须是dataframe
		ret1 = df.isnull().any()	# 检测nan

		return ret1

	'''
		按照csv_no字段，截取行，按块拆分数据集
	'''
	def truncStruct(self, df, number):
		df_new = DataFrame(columns=['time', 'spindle_load', 'x', 'y', 'z', 'csv_no'])

		for i in range(0, len(df)):
			if df['csv_no'][i] == number:
				df_new.loc[i] = df.loc[i]

		return df_new

	def minus(self, d1, d2):
		return d2-d1

if __name__ == '__main__':
	pass
# # print(plc_df.tail(10))
#
# df = DataFrame()
#
# ret = detectInvalidValue(plc_df)
# df_new = truncStruct(plc_df)
# print(len(df_new))
# df_new.to_csv('../datas/test/plc_new.csv', columns=names, index=False)

# csv_no_set = set(plc_df['csv_no'])
# for number in csv_no_set:		# 取出每一个csv_no号做截取
# 	if number > 4:
# 		break
# 	df_new = DataSetPreprocess.truncStruct(plc_df, number)
# 	time_lst = list(df_new['time'])
# 	sl_lst = list(df_new['spindle_load'])
# 	x_lst = list(df_new['x'])
# 	y_lst = list(df_new['y'])
# 	z_lst = list(df_new['z'])
#
# 	sum = 0.0
# 	time_new_lst = [0.0]  # 原点
# 	for i in range(1, len(time_lst)):
# 		diff = calTimeDiff(time_lst[i - 1], time_lst[i])
# 		sum += diff
# 		time_new_lst.append(sum)
#
# 	print(sum)
#
# 	print(len(time_new_lst))
# 	print(len(sl_lst))
#
# 	plt.title('Data set parameter analysis')
# 	plt.plot(time_new_lst, sl_lst, color='green', label='spindle_load')
# 	plt.plot(time_new_lst, x_lst, color='red', label='x')
# 	plt.plot(time_new_lst, y_lst, color='skyblue', label='y')
# 	plt.plot(time_new_lst, z_lst, color='blue', label='z')
# 	plt.legend()  # 显示图例
#
# 	plt.xlabel('time(ms)')
# 	plt.ylabel('values')
# 	plt.show()