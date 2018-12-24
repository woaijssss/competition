
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

if __name__ == '__main__':
	dp = dataset_preprocess.DataSetPreprocess()

	dataset = '../datas/02-TestingData-poL3/result03.csv'
	names = ['spindle_load', 'x', 'y', 'z',  'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
	df = dp.loadDataSet(filename=dataset, columns=names)

	df = dp.abs(df)

	print(len(df))
	tmp = df[
		(df['spindle_load'] > 25) | (df['spindle_load'] < 3)
	# 	| (df['y'] < -205)
	# 	| (df['z'] < 474)
		| (df['vibration_1'] > 34)
		| (df['vibration_2'] > 33) | (df['vibration_2'] < 29)
		| (df['vibration_3'] > 36)
		| (df['current'] > 2.5)
		]

	lstz = list(df['z'])
	# quit()

	# print(len(tmp.index))
	l = list(tmp.index)
	print(len(l))
	# l += [i for i in range(0, 2000) if lstz[i] < 474]
	# l += [i for i in range(2000, 4000) if lstz[i] < 474.5]
	# l += [i for i in range(4000, 6000) if lstz[i] < 477]
	# l += [i for i in range(6000, 8000) if lstz[i] < 478]
	# l += [i for i in range(8000, 10000) if lstz[i] < 478.6]
	# l += [i for i in range(10000, 12000) if lstz[i] < 481]
	# l += [i for i in range(12000, 14000) if lstz[i] < 482.3]
	# l += [i for i in range(14000, 16000) if lstz[i] < 483.5]
	print(len(l))
	print(l)
	df = df.drop(index=l, axis=0)
	print(len(df))
	#
	# tmp = df[(df['z'] < 400)]
	#
	# print(len(tmp.index))
	# # for index in tmp.index:
	# df = df.drop(index=list(tmp.index), axis=0)

	df.to_csv('../datas/02-TestingData-poL3/result03_new1.csv', sep=',', index=False, columns=names)
	# quit()

	for name in names:
		# if name != 'z':
		# 	continue
		gp = datasetGraphPlot.GraphPlot()
		y = list(df[name])

		print(df[name].describe())

		x = [i for i in range(0, len(y))]
		gp.plotScatter(x_lst=x, x_label='index', y_lst=y, y_label='current')
		gp.show(x_label=name, y_label='y')
