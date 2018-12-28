
import pandas as pd
from pandas import DataFrame
import numpy as np

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.models.nn.nnmodel as nnmodel

if __name__ == '__main__':
	params = {
		'W1': [
			np.array([-0.61324594, -0.36208691,  5.28242554, -3.01282416,  7.62865849, -0.62081277, -0.67408975,  2.51497838]),
			np.array([ 1.7029553 ,  1.01789893,  4.64378888,  1.97453178, 14.20628213, 0.16757569,  2.47092719,  3.50613409]),
			np.array([ 2.34476816,  1.66139795,  6.25396241,  0.74591572, 13.61106735, 0.63872683,  1.03310308,  1.85433978]),
			np.array([ 3.60419099,  2.55649242,  2.99760637,  2.25796854,  3.65355226,
        0.58162931, -0.93840333,  1.64538863])
			   ],
		'b1': [
			np.array([-0.01344081]),
			np.array([-0.04042031]),
			np.array([-0.03237672]),
			np.array([-0.02221932])
		],
		'W2': [np.array([-7.24383315, -2.1271015 , -2.5850113 , -0.13791894])],
		'b2': [np.array([25.02301495])]
	}

	print(params)

	names = ['spindle_load', 'x', 'y', 'z', 'csv_no', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
	for file in range(1, 5):
		print('------')
		filename = '../../../datas/03-FinalData/result0' + str(file) + '.csv'
		dp = dataset_preprocess.DataSetPreprocess()

		X = dp.loadDataSet(filename=filename, columns=names)
		X = X.drop(['csv_no'], axis=1)

		X = np.array(X).reshape((8, -1))
		AL = nnmodel.predict(X, params)
		print('=======AL: ', AL)
		print('means------: ', np.mean(np.array(AL)[0]))


