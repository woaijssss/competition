
from pandas import DataFrame
import numpy as np

import src.samplingSensorData as samplingSensorData
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess

def testDataframe():
	dp = dataset_preprocess.DataSetPreprocess()
	arr = [
		[1, 100, 3],
		[4, np.inf, 6],
		[np.inf, 8, 9],
		[4, np.inf, 6],
		[4, np.inf, 6],
		[1, 2, 1e+11],
		[1, 2, 3],
		[1, 2, 1e+11],
		[1, 2, 1e+11],
		[1, 2, 1e+11],
		[1, 2, 1e+11],
		[1, 2, 1e+11],
		[7, 8, -1e-100],
		[7, 8, -1e-55],
		[7, 8, 1e-55],
		[1, np.NAN, 2]
	]
	
	names = ['a', 'b', 'c']
	df = DataFrame(np.array(arr), columns=names)
	# df = dp.filterInvalidValue(df)
	
	print(df)
	
	print('======>:', df.where(df == np.NAN))
	
	#sensor_plc_counts = len(self._plc_df[self._plc_df['csv_no'] == self._sensor_csv_no])
	print('--->:', len(df[df['c'] == 1e11]))

'''
	测试程序，测试整合后的结果，并保存成本地csv，方便比对数据
'''
def test():
	for i in range(1, 6):
		plc_csv_dir = '../../02-TestingData-poL3/0' + str(i)
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

def testFloat():
	l = -7193750930745000000000000000000000000000000.381
	print(type(l))
	print(l)

if __name__ == '__main__':
	# test()
	testDataframe()
	# testFloat()