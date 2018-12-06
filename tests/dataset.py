
import numpy as np
import pandas as pd
from pandas import DataFrame
import time
import datetime

import matplotlib.pyplot as plt

'''
	计算两个时间点的差值
'''
def calTimeDiff(time1, time2):
	li_time1 = time1.split(':')
	li_time2 = time2.split(':')
	hour1, min1, secs1, milsec1 = li_time1
	hour2, min2, secs2, milsec2 = li_time2

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
def detectInvalidValue(df):
	if not isinstance(df, pd.core.frame.DataFrame):
		return False

	# 必须是dataframe
	ret1 = df.isnull().any()	# 检测nan

	return ret1

'''
	按照csv_no字段，截取行，按块拆分数据集
'''
def truncStruct(df):
	df_new = DataFrame(columns=['time', 'spindle_load', 'x', 'y', 'z', 'csv_no'])

	for i in range(0, len(df)):
		if df['csv_no'][i] == 1:
			df_new.loc[i] = df.loc[i]

	return df_new

def minus(d1, d2):
	return d2-d1

if __name__ == '__main__':
	# filename = '../datas/test/plc.csv'
	# names = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no']
	# plc_df = pd.read_csv(filename, sep=',')
	# # print(plc_df.tail(10))
	#
	# df = DataFrame()
	#
	# ret = detectInvalidValue(plc_df)
	# df_new = truncStruct(plc_df)
	# print(len(df_new))
	# df_new.to_csv('../datas/test/plc_new.csv', columns=names, index=False)

	filename = '../datas/test/plc_new.csv'
	names = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no']
	plc_df = pd.read_csv(filename, sep=',')

	time_lst = list(plc_df['time'])
	sl_lst = list(plc_df['spindle_load'])
	x_lst = list(plc_df['x'])
	y_lst = list(plc_df['y'])
	z_lst = list(plc_df['z'])

	sum = 0.0
	time_new_lst = [0.0]		# 原点
	for i in range(1, len(time_lst)):
		diff = calTimeDiff(time_lst[i-1], time_lst[i])
		sum += diff
		time_new_lst.append(sum)

	print(sum)

	print(len(time_new_lst))
	print(len(sl_lst))

	plt.title('Data set parameter analysis')
	plt.plot(time_new_lst, sl_lst, color='green', label='spindle_load')
	plt.plot(time_new_lst, x_lst, color='red', label='x')
	plt.plot(time_new_lst, y_lst, color='skyblue', label='y')
	plt.plot(time_new_lst, z_lst, color='blue', label='z')
	plt.legend()  # 显示图例

	plt.xlabel('time(ms)')
	plt.ylabel('values')
	plt.show()