import numpy as np
from utilities import import_data, visualize_block
import matplotlib.pyplot as plt

if __name__ == '__main__':

	X=import_data()
	choice=''
	while choice != 'q':
		idx=np.random.randint(0, len(X))
		data_point=X[idx][0]
		print('picked data point {}'.format(idx))
		visualize_block(data_point)
		choice=input('enter to continue, q to quit: ')


