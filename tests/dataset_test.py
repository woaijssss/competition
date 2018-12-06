
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.datasetStatisticAnalysis.dataset_preprocess import DataSetPreprocess

if __name__ == '__main__':
	filename = '../datas/01-TrainingData-qLua/01/PLC/plc.csv'

	data_processor = DataSetPreprocess(filename)
	names = ['time', 'spindle_load', 'x', 'y', 'z', 'csv_no']
	plc_df = data_processor.loadDataSet(columns=names)

	time_lst = np.array(plc_df['time'])
	sl_lst = np.array(plc_df['spindle_load'])
	x_lst = np.array(plc_df['x'])
	y_lst = np.array(plc_df['y'])
	z_lst = np.array(plc_df['z'])

	sum = 0.0
	time_new_lst = [0.0]		# 原点
	for i in range(1, len(time_lst)):
		diff = data_processor.calTimeDiff(time_lst[i-1], time_lst[i])
		sum += diff
		time_new_lst.append(sum)

	spindle_load_max, spindle_load_max_index, spindle_load_min, spindle_load_min_index = data_processor.calMaxMin(sl_lst)

	plt.title('Data set parameter analysis')
	show_max = '[' + str(spindle_load_max_index) + ' ' + str(spindle_load_max) + ']'
	plt.annotate(show_max, xytext=(spindle_load_max_index, spindle_load_max), xy=(spindle_load_max_index, spindle_load_max))
	show_min = '[' + str(spindle_load_min_index) + ' ' + str(spindle_load_min) + ']'
	plt.annotate(show_min, xytext=(spindle_load_min_index, spindle_load_min), xy=(spindle_load_min_index, spindle_load_min))

	plt.plot(time_new_lst, sl_lst, color='green', label='spindle_load')
	plt.plot(time_new_lst, x_lst, color='red', label='x')
	plt.plot(time_new_lst, y_lst, color='skyblue', label='y')
	plt.plot(time_new_lst, z_lst, color='blue', label='z')
	plt.legend()  # 显示图例

	plt.xlabel('time(ms)')
	plt.ylabel('values')
	plt.show()