
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

if __name__ == '__main__':
	dp = dataset_preprocess.DataSetPreprocess()

	dataset = '../datas/02-TestingData-poL3/result01.csv'
	names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
	df = dp.loadDataSet(filename=dataset, columns=names)

	df = dp.abs(df)

	print(len(df))
	tmp = df[
		(df['spindle_load'] > 35.5) | (df['spindle_load'] < 3)
	# 	| (df['y'] < -205)
		| (df['z'] < 427)
		| (df['vibration_1'] > 500000)
		| (df['vibration_2'] > 1000000) | (df['vibration_2'] < 29)
		| (df['vibration_3'] > 36)
		| (df['current'] > 3.2)
		]

	lstz = list(df['z'])
	# quit()

	# print(len(tmp.index))
	l = list(tmp.index)
	print(len(l))
	l += lst
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

	# df.to_csv('../datas/02-TestingData-poL3/result01_new1.csv', sep=',', index=False, columns=names)
	# quit()

	for name in names:
		if name != 'z':
			continue
		gp = datasetGraphPlot.GraphPlot()
		y = list(df[name])

		print(df[name].describe())

		x = [i for i in range(0, len(y))]
		gp.plotScatter(x_lst=x, x_label='index', y_lst=y, y_label='current')
		gp.show(x_label=name, y_label='y')
