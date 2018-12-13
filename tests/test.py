
from pandas import DataFrame
import numpy as np

import src.samplingSensorData as samplingSensorData

'''
	测试程序，测试整合后的结果，并保存成本地csv，方便比对数据
'''
def test():
	for i in range(2, 4):
		plc_csv_dir = '../../01-TrainingData-qLua/0' + str(i)
		filename = plc_csv_dir + '/PLC/plc_test.csv'
		columns = ['vibration_1', 'vibration_2', 'vibration_3', 'current']
		# df_new = DataFrame(columns=columns)
		arr = []
	
		# ssd = samplingSensorData.SamplingSensorData(plc_csv_dir, test_filename=filename)  # 读取plc.csv文件
		ssd = samplingSensorData.SamplingSensorData(plc_csv_dir)  # 读取plc.csv文件
	
		for line in range(0, ssd.getPLCLength()):  # 对当前plc所有数据操作
			new_sensor_df = ssd.samplingSensorData(line=line)		# 计算后的结果
	
			if not new_sensor_df.empty:
				arr.append(list(new_sensor_df.loc[0]))
	
		df_new = DataFrame(np.array(arr), columns=columns)
		df_new.to_csv(plc_csv_dir + '/new.csv', sep=',', index=False)

if __name__ == '__main__':
	test()