
from pandas import DataFrame

import src.samplingSensorData as samplingSensorData
import numpy as np

def contact(df_new, sensor_df):
	# 具体实现
	return df_new

if __name__ == '__main__':
	plc_csv_dir = '../datas/01-TrainingData-qLua/01'
	# 最终整合后的列
	new_columns = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
	df_new = DataFrame(columns=new_columns)

	ssd = samplingSensorData.SamplingSensorData(plc_csv_dir)  # 读取plc.csv文件

	for line in range(0, ssd.getPLCLength()):  # 对当前plc所有数据操作
		new_sensor_df = ssd.samplingSensorData(line=line)		# 计算后的结果，为一行sensor的平均值数据

		if not new_sensor_df.empty:
			'''
				执行接口（2），并整合数据
			'''
		df_new = contact(df_new, new_sensor_df)

	df_new.to_csv(plc_csv_dir + '/new.csv', sep=',', index=False)