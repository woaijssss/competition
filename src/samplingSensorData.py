
import pandas as pd
from pandas import DataFrame
import numpy as np

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import logging

'''
接口的调用流程：		
	入口方法：samplingSensorData()
	对plc下的所有sensor数据，按照csv_no字段的值，对对应的sensor.csv文件做如下操作：
	1）trunc()：每行plc数据，对应777条sensor数据，不足777的，对应到plc对应csv_no的最后一笔数据；
	2）abs()：将截取的777条数据去矢量化；
	3）average()：将去矢量化后的数据，按列取平均值，得到一行dataframe；
'''
'''	关于问题：
	以下带 "问题" 的注释，考虑是否优化：
	（1）倍数的计算部分
'''
class SamplingSensorData:
	_csv_dir = ''		# plc.csv所在目录
	#-----------------------------> 数据预处理的对象（主要使用abs()接口）
	_data_processor = dataset_preprocess.DataSetPreprocess()
	#-----------------------------> PLC相关的变量
	#_plc_columns = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
	_plc_columns = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no']
	_plc_df = None			# 记录当前plc的所有数据
	_plc_line = 0			# 记录上一次处理的plc行数

	#-----------------------------> sensor相关的变量
	_sensor_columns = ['vibration_1', 'vibration_2', 'vibration_3', 'current']
	_sensor_df = DataFrame(columns=_sensor_columns)	# 保存某一个sensor的数据，可以避免多次加载的时间浪费
	_basic_n = 777		# 根据plc和sensor的采集频率计算的采集周期倍数
	_seek_ptr = 0		# 记录sensor_df中，已经截取到哪一行
	_sensor_len = 0		# 记录sensor截取的剩余行数
	_sensor_csv_no = 1			# 记录上一次处理的sensor.csv号（初始值为1，因为plc中csv_no从1开始）


	'''
		返回plc数据
	'''
	def getPLCDf(self):
		return self._plc_df

	'''
		返回plc的数据行数
	'''
	def getPLCLength(self):
		return len(self._plc_df)

	'''
		初始化PLC数据
		:param test_filename：可以通过此参数，传递想要测试的csv文件进行测试；正式使用时候，不需要传递此值
	'''
	def __init__(self, csv_dir='.', test_filename=None):
		self._csv_dir = csv_dir
		if not test_filename:
			self._plc_df = self._data_processor.loadDataSet(self._csv_dir + '/PLC/plc.csv', sep=',', columns=self._plc_columns)
		else:
			self._plc_df = self._data_processor.loadDataSet(test_filename, sep=',', columns=self._plc_columns)

	'''
		处理sensor数据的核心调用，数据处理的入口
	'''
	def samplingSensorData(self, line):
		tm, spindle_load, x, y, z, csv_no = self._plc_df.loc[line]		# 取出plc当前处理行的所有数据

		if csv_no > self._sensor_csv_no:	# 判断是否需要重置sensor.csv的数据
			# self._sensor_df.drop(self._sensor_df.index, inplace=True)	# 清空上一次保存的sensor_df
			self._sensor_df = DataFrame(columns=self._sensor_columns)
			self._sensor_len = 0
			self._seek_ptr = 0
			self._sensor_csv_no = csv_no
			self._basic_n = 777

		if self._sensor_df.empty:		# 按照当前的sensor.csv号，读取sensor数据
			logging.debug('---------------读取第 %d 个csv文件' % self._sensor_csv_no)
			#  按照plc.csv中的csv_no读取对应的sensor数据
			self._sensor_df = self._data_processor.loadDataSet(self._csv_dir + '/Sensor/' + str(self._sensor_csv_no) + '.csv',
												   columns=self._sensor_columns)
			self._sensor_len = len(self._sensor_df)
			# logging.debug(self._sensor_len)

			#=============================================================================================================
			# 问题-1-------------------->以下过程考虑是否需要优化？
			# 获取plc中，csv_no值等于当前值的总行数
			sensor_plc_counts = len(self._plc_df[self._plc_df['csv_no'] == self._sensor_csv_no])
			# 计算sensor数据量与plc中当前csv_no总行数的对应倍数，防止按照固定777计算，会剩余很多plc的数据
			# 只在sensor数据加载时计算一次，因为加载后，到上一个判断期间，self._sensor_df都不会变
			# self._basic_n = int(self._sensor_len / sensor_plc_counts) + 1
			self._basic_n = int(self._sensor_len / (sensor_plc_counts-1))
			logging.debug('倍数计算>>>>>>>>>plc中csv_no为 %d 的行数为 %d，当前第 %d 个sensor数据长度为 %d ====>倍数为：%d' %
				  (self._sensor_csv_no, sensor_plc_counts, self._sensor_csv_no, self._sensor_len, self._basic_n))
			#=============================================================================================================


		if not self._sensor_len:		# 防止出现sensor最后一笔截取的数据，不是对应plc指定csv_no的最后一个值，导致出错的情况
			logging.debug('第 %d 个sensor数据剩余长度为 %d,继续读取下一个sensor数据' % (self._sensor_csv_no, self._sensor_len))
			return DataFrame()			# 为方便与成功的df做比较，这里返回一个空的df

		if self._sensor_len < self._basic_n:		# 如果sensor的最后一笔要截取的数据小于指定长度，则按实际情况截取
			logging.debug('替换倍数------->第 %d 个sensor数据剩余长度为 %d，小于指定长度 %d，将原截取值 %d 替换为 %d' %
				  (self._sensor_csv_no, self._sensor_len, self._basic_n, self._basic_n, self._sensor_len))
			self._basic_n = self._sensor_len

		sensor_df_tmp = self.trunc(begin=self._seek_ptr, end=self._seek_ptr+self._basic_n)		# 对sensor数据截取指定长度
		logging.debug('记录当前=======>当前处理的行数 %d， 每次增加 %d 行，剩余 %d 行, 当前指向 %d 行' % (line+1, self._basic_n, self._sensor_len, self._seek_ptr))
		
		sensor_df_tmp = self._data_processor.filterInvalidValue(sensor_df_tmp)

		record_seek = self._seek_ptr


		self._seek_ptr += self._basic_n				# 每次截取后，指针指向下一行
		self._sensor_len -= self._basic_n				# 每次截取后，将可截取的长度减少对应的数量

		sensor_df = self.abs(sensor_df_tmp)			# （2）去矢量化
		series = self.average(sensor_df)		# （3）取平均值
		sensor_df = pd.DataFrame([series])

		logging.debug('+++++++++++++++++++++++++++++++++++++++++++++')
		logging.debug('第 %d 个sensor数据，%d ----%d 行的平均值dataframe为：' % (self._sensor_csv_no, record_seek, self._seek_ptr))
		logging.debug(sensor_df)
		logging.debug('+++++++++++++++++++++++++++++++++++++++++++++')

		return sensor_df						# 返回处理后的sensor_df

	def trunc(self, begin, end):
		'''
			因为csv文件打开时，第一列是columns，并且程序从0下标开始取值，而csv文件中是从1开始，
			实际行数应该相差2
		'''
		# logging.debug('trunc: 从第 %d 行截取到第 %d 行' % (begin+1, end+1))
		return self._data_processor.trunc(self._sensor_df, begin=begin, end=end)

	'''
		去矢量化
	'''
	def abs(self, sensor_df):
		return self._data_processor.abs(sensor_df)

	'''
		取平均值
	'''
	def average(self, sensor_df):
		return self._data_processor.average(sensor_df)



if __name__ == '__main__':
	plc_csv_dir = '../datas/01-TrainingData-qLua/01'
	ssd = SamplingSensorData(plc_csv_dir)		# 读取plc.csv文件

	columns = ['vibration_1', 'vibration_2', 'vibration_3', 'current']
	arr = []

	for line in range(0, 5000):		# 取部分数据验证
		new_sensor_df = ssd.samplingSensorData(line=line)

		if not new_sensor_df.empty:
			logging.debug(new_sensor_df)