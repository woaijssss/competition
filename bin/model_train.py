
from sklearn.model_selection import train_test_split

import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.models.regressor as regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import logging
import time

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./record.log',
                filemode='w')


if __name__ == '__main__':
	names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
	filename = '../datas/01-TrainingData-qLua/final.csv'
	dp = dataset_preprocess.DataSetPreprocess()
	dataset_df = dp.loadDataSet(filename=filename, columns=names)
	dataset_df = dataset_df.dropna(how='any', axis=0)

	X = dataset_df[names[0:len(names) - 1]]
	Y = dataset_df['last_time']
	print(X.columns)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=15)

	from sklearn.preprocessing import StandardScaler

	ss = StandardScaler()  # 数据标准化
	X_train = ss.fit_transform(X_train)
	X_test = ss.transform(X_test)

	'''
		默认为CART树
		loss：指定误差计算方式:”linear”, “square”,“exponential”, 默认为”linear”
		n_estimators:值过大可能会导致过拟合， 一般50~100比较适合， 默认50
		learning_rate:一般从一个比较小的值开始进行调参； 该值越小表示需要更多的弱分类器
	'''
	import numpy as np

	learning_rate = np.linspace(0.01, 0.1, 20)
	# learning_rate = np.linspace(0.1, 2, 2)
	i = 0
	result = [0, 0, 0]
	params = [-1, -1, -1]
	y_predict_best = []		# 预测中最好的y值

	'''
		当前最好的参数
			第 37 轮---estimator为 61， max_depth为 10， learning_rate为：0.733333------>训练的准确率:0.882113
			第 0 轮---estimator为 61， max_depth为 10， learning_rate为：0.733333------>预测的准确率:0.880970
		
		较好的参数：
			第 0 轮---estimator为 67， max_depth为 15， learning_rate为：0.010000------>训练的准确率:0.951467
			第 0 轮---estimator为 67， max_depth为 15， learning_rate为：0.010000------>预测的准确率:0.950328
	'''

	for rate in learning_rate:
		for max_depth in range(9, 11):
			for estimator in range(60, 65):
				t1 = time.time()
				i += 1
				clf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth, splitter='random'),
										n_estimators=estimator, learning_rate=rate,
										loss='square')

				clf.fit(X_train, Y_train)

				# 模型预测
				y_predict = clf.predict(X_test)
				y_train_predict = clf.predict(X_train)

				from sklearn.metrics import r2_score
				ypre = regressor.traverse(y_predict)  # 对预测的结果进行数值转换
				ytrainpre = regressor.traverse(y_train_predict)
				# print(y_predict)
				# print(ypre)
				acc_predict = r2_score(Y_train, ytrainpre)  # 训练集预测的r^2
				acc_pre = r2_score(Y_test, ypre)  # 测试集预测的r^2

				logging.debug('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>训练的准确率:%f' % (i, estimator, max_depth, rate, acc_predict))
				logging.debug('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测的准确率:%f' % (i, estimator, max_depth, rate, acc_pre))
				logging.debug('---------------------------')
				print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>训练的准确率:%f' % (i, estimator, max_depth, rate, acc_predict))
				print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测的准确率:%f' % (i, estimator, max_depth, rate, acc_pre))
				print('---------------------------')
				# if acc_predict >= acc_pre:
				if acc_predict > result[1] and acc_pre > result[2]:
					print('第 %d 轮优于第 %d 轮，替换。。。' % (i, i-1))
					result[0], result[1], result[2] = i, acc_predict, acc_pre
					params[0], params[1], params[2] = estimator, max_depth, rate
					y_predict_best = ypre

				t2 = time.time()
				print('耗时:', t2-t1)

	print('最终预测最好的为:', result)
	print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>训练的准确率:%f' % (result[0], params[0], params[1], params[2], result[1]))
	print('第 %d 轮---estimator为 %d， max_depth为 %d， learning_rate为：%f------>预测的准确率:%f' % (result[1], params[0], params[1], params[2], result[2]))

	# 任取区间100的数据进行可视化
	# print(list(Y_train[0:100]))
	# print(list(Y_test[0:100]))
	# print(list(ypre[0:100]))
	xlst = [i for i in range(700, 800)]
	ytestlst = list(Y_test)[700:800]
	y_predictlst = list(y_predict_best)[700:800]

	# 可视化
	import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

	dgp = datasetGraphPlot.GraphPlot()
	dgp.plotScatter(x_lst=xlst, y_lst=ytestlst, y_label='y_test')
	# dgp.plotScatter(x_lst=xlst, y_lst=y_predictlst, y_label='y_predict')
	dgp.plot(xlst, y_predictlst, 'y_predict')
	dgp.show()