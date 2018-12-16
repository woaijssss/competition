
from sklearn.model_selection import train_test_split

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.models.regressor as regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib

import logging
import time
import warnings
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./record.log',
                filemode='w')

def test():
	i = 1e10
	j = 1e+10
	print(i == j)
	quit()

if __name__ == '__main__':
	names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
	filename = '../datas/02-TestingData-poL3/result02.csv'
	dp = dataset_preprocess.DataSetPreprocess()
	X = dp.loadDataSet(filename=filename, columns=names)
	# X = X.dropna(how='any', axis=0)

	import numpy as np

	import pandas as pd
	print(X.describe())
	print('------>:', X[X['vibration_1'] > 1e10])
	# quit()
	tmp = X[X['spindle_load'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['x'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['y'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['z'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['vibration_1'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['vibration_2'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['vibration_3'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))
	tmp = X[X['current'] > 1e10]
	for index in tmp.index:
		X = X.drop(index=index, axis=0)
		print(len(X))

	tmp = X[X['spindle_load'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['x'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['y'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['z'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['vibration_1'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['vibration_2'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['vibration_3'] > 1e10]
	print('--------index:', tmp.index)
	tmp = X[X['current'] > 1e10]
	print('--------index:', tmp.index)
	print('========')
	import  numpy as np
	print(np.isinf(X['spindle_load']).any())
	print(np.isinf(X['x']).any())
	print(np.isinf(X['y']).any())
	print(np.isinf(X['z']).any())
	print(np.isinf(X['vibration_1']).any())
	print(np.isinf(X['vibration_2']).any())
	print(np.isinf(X['vibration_3']).any())
	print(np.isinf(X['current']).any())

	# import numpy as np
	# print(X[np.isinf(X) == True])
	# quit()

	from sklearn.preprocessing import StandardScaler

	ss = StandardScaler()  # 数据标准化
	arr = X.values
	# print(arr)
	# print(type(arr))
	X_train = ss.fit_transform(arr)
	clf = joblib.load("train_model.m")

	from sklearn.model_selection import train_test_split
	# _, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=15)


	print(X.describe())
	y_predict = clf.predict(X_train)
	print(y_predict)
	print('----->:', y_predict.mean())


	# xlst = [i for i in range(700, 800)]
	# y_predictlst = list(y_predict)[700:800]
	xlst = [i for i in range(0, len(y_predict))]
	import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

	dgp = datasetGraphPlot.GraphPlot()
	# dgp.plotScatter(x_lst=xlst, y_lst=y_predictlst, y_label='y_predict')
	dgp.plot(xlst, y_predict, 'y_predict')
	dgp.show()