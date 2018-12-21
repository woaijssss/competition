
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import MaxPooling2D, Conv2D, Input
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from keras.layers.normalization import  BatchNormalization  as bn

class CNN:
	_model = None
	_X = None
	_Y = None
	def __init__(self):
		self._model = Sequential()	# 构建一个空的序贯模型，后面各CNN层可以顺序添加

	'''
		模型构建入口
	'''
	def modelBuild(self, train_x, train_y, batch_size=10, nb_epoch=10):
		import tensorflow as tf
		train_x = tf.convert_to_tensor(train_x)
		######1. 模型设置、构建
		self.dataPreprocess()
		self.inputLayer(x=train_x, y=train_y)
		# self._model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

		inputs = Input((self._X.shape[1],))
		predictions = Dense(1, activation='linear')(train_x)

		# This creates a model that includes
		# the Input layer and three Dense layers
		self._model = Model(input=inputs, output=predictions)
		self._model.compile(optimizer='rmsprop',
					  loss='mean_squared_error',
					  metrics=['mae', 'acc'])
		self._model.fit(train_x, train_y, validation_data=(train_x, train_y),
				  nb_epoch=10, batch_size=100)
		# self._model.add(Dense(256)(inputs))
		# self.activationFunc(function='relu')		# 激活层
		# self._model.add(Dense(1, activation='linear'))

		# self.fcLayer()
		self.print()

		######2. 训练模型
		# self.train(batch_size=batch_size, nb_epoch=nb_epoch)

	'''
		设置CNN网络参数
	'''
	def setOption(self):
		'''
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self._model.compile(loss='categorical_crossentropy',	# 损失函数
						   optimizer=sgd,
						   metrics=['accuracy'])	# 完成实际的模型配置工作
		'''

		self._model.compile(loss='mean_squared_error',	# 损失函数
						   optimizer='rmsprop',
						   metrics=['mae', 'acc'
										   ''])	# 完成实际的模型配置工作

	'''
		训练数据预处理
	'''
	def dataPreprocess(self):
		pass

	'''
		输入层
	'''
	def inputLayer(self, x, y):
		self._X = x
		self._Y = y

	'''
		卷积层
	'''
	def convLayer(self, filters=16, kernel_size=(3, 3), padding='valid'):
		self._model.add(Conv2D(filters=filters, kernel_size=kernel_size,
							   padding=padding))	# 2维卷积层

	'''
		激活函数
	'''
	def activationFunc(self, function='relu'):
		self._model.add(Activation(function))					# 激活函数层

	'''
		池化层
	'''
	def poolingLayer(self, pool_size=(2, 2), padding='valid'):
		self._model.add(MaxPooling2D(pool_size=pool_size, padding=padding))		# 池化层

	'''
		Dropout层
	'''
	def dropout(self, rate=0.25):
		self._model.add(Dropout(rate=rate))							# Dropout层

	'''
		全连接层
	'''
	def fcLayer(self):
		self._model.add(Dense(output_dim=1, input_dim=self._X.shape[1]))

	'''
		模型训练
	'''
	def train(self, batch_size=10, nb_epoch=10):
		self.setOption()
		print('--->:', self._X.shape)
		print('--->:', self._Y.shape)
		self._model.fit(self._X, self._Y, batch_size=batch_size, nb_epoch=nb_epoch)

	'''
		模型预测
	'''
	def predict(self):
		self.dataPreprocess()		# 数据预处理
	'''
		模型保存
	'''
	def save(self, path='.'):
		self._model.save(path)

	'''
		模型加载
	'''
	def loadModel(self, path='.'):
		self._model = load_model(path)

	def print(self):
		self._model.summary()

if __name__ == '__main__':
	names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
	filename = '../../../01-TrainingData-qLua/二次处理/final_new1.csv'
	# dataset_df = pd.read_csv(filename, sep=',')

	import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
	dp = dataset_preprocess.DataSetPreprocess()
	dataset_df = dp.loadDataSet(filename=filename, columns=names)

	X = dataset_df[names[0:len(names)-1]]
	Y = dataset_df[names[len(names)-1:len(names)]]
	print(X.columns)
	print(Y.columns)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

	print(X_train)

	cnn = CNN()
	cnn.modelBuild(X_train, Y_train, batch_size=38, nb_epoch=10)
