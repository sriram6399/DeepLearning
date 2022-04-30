import os
import pickle
import numpy as np
from utilities import data_point
from utilities import import_data


def main():

	#loads all data points at once
	X, y=import_data()

	print('X is the dataset, y is the label set')
	print('X[i] corresponds to y[i]\n')
	
	#number of data points
	print('shape of X: {}'.format(X.shape))
	print('shape of y: {}'.format(y.shape))
	
	#shape of the data point
	print('shape of data point: {}'.format(X[0].shape))
	
	


if __name__ == '__main__':
	main()
