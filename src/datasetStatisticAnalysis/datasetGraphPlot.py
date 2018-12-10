
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class GraphPlot:
	_fig = None

	def __init__(self):
		self._fig = plt.figure()

	def addSubPlot(self, x_lst=[], x_label='x', y_lst=[], y_label='y', color='black', *args):
		if len(args) != 3:
			raise ValueError("use subplot position such as: 2, 2, 1")

		pic = self._fig.add_subplot(args[0], args[1], args[2])
		pic.plot(x_lst, y_lst, color=color)
		pic.set_xlabel(x_label)
		pic.set_ylabel(y_label)

	def showSubplot(self):
		plt.legend()  # 显示图例
		self._fig.show()

	def plotScatter(self, x_lst=[], x_label='x', y_lst=[], y_label='y', marker='.'):
		plt.scatter(x_lst, y_lst, marker=marker)
		plt.xlabel(x_label)
		plt.ylabel(y_label)

	def show(self, x_label='x', y_label='y'):
		plt.legend()  # 显示图例
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		self._fig.show()