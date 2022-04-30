import numpy as np
from ale_py._ale_py import Action
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import random
import cv2



'''
https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/src/gym/envs/atari/environment.py

refer to the link above for mapping details
'''
action_map={
	None: 0,
	ord('w'): int(Action.UP),
	ord('a'): int(Action.LEFT),
	ord('d'): int(Action.RIGHT),
	ord('s'): int(Action.DOWN),
	ord(' '): int(Action.FIRE)
}

"""
used to stack data before it's appended to the dataset
"""
class data_point:

	#initialize data point. (4, 210, 160)=4 frames
	def __init__(self, shape=(4, 84, 84)):
		self.shape=shape
		self.point=np.zeros(self.shape)

		self.buffer=[]

	
	#adds new frame to the data point. 
	def add_frame(self, frame):
		#crop, scale, and then resize frame before adding
		processed_frame=cv2.resize(frame[34:194]/255,
								   dsize=(84,84), 
								   interpolation=cv2.INTER_NEAREST)

		self.point[0:3]=self.point[1:4]
		self.point[3]=processed_frame

	def get(self):
		return self.point.copy()
	
	
class data_collector:
	def __init__(self):
		self.point=data_point()
		self.data=[]
		self.start_key=action_map[ord(' ')]
		self.in_progress=False



	"""
	used for saving the data after playing.

	action is an integer
	obs_t should be shape (210, 160)

	data_point class is used to shape the data point to be (4, 210, 160)
	1st dimension is now 4 because we stack the previous 3 frames alongside the current frame
	"""
	def callback(self, s, s_prime, a, r, done, info):
		#used to decide whether or not to record data point
		if done:
			self.in_progress=False

		#used to know when the game has actually started
		if a==self.start_key:
			self.in_progress=True

		self.point.add_frame(s)
		
		if self.in_progress:
			self.data.append((self.point.get(), a))

	"""
	returns the data
	"""
	def dump_data(self):
		return self.data
	
	"""
	prune if all 4 frames are the same and no action is taken
	run with debug=True to visualize all the frames that have been pruned
	it's pretty janky so you feel free to force quit
	"""
	def prune_data(self, debug=False):

		active_data=[]
		inactive_data=[]
		for i in range(len(self.data)):

			s, a=self.data[i]
			noop=(a==0)
			is_still=(s == s[0]).all() and noop

			if not is_still:
				active_data.append(i)
			else:
				inactive_data.append(i)

		#delete when done
		if debug==True:
			print('self.data size={}'.format(len(self.data)))
			print('kept data in\n{}\n'.format(active_data))
			print('*******************')
			print('pruned elements in\n{}'.format(inactive_data))
			for i in inactive_data:
				visualize_block(self.data[i][0])

		print('pruned {} inactive data points'.format(len(inactive_data)))
		temp=[self.data[i] for i in active_data]
		self.data=temp

	def prune_noops(self):
		self.data=prune_noops(self.data)

	'''
	experimental

	replace actions for noops and frozen states to use FIRE action
	'''
	def convert_noops_breakout(self):
		active_data=[]
		inactive_data=[]
		for i in range(len(self.data)):

			s, a=self.data[i]
			noop=(a==0)
			is_still=(s == s[0]).all() and noop

			if not is_still:
				active_data.append(i)
			else:
				inactive_data.append(i)

		temp=[self.data[i] for i in active_data] + [(self.data[i][0], 1) for i in inactive_data]
		self.data=temp




"""
returns the data.
X is the set of data points
y is the labels
"""
def import_data(data_dir='../data/'):
	print("----- importing data -----")
	files=os.listdir(data_dir)
	data_set=[]
	for f in files:
		if 'demonstrator' in f:
			print('reading {}'.format(f))
			data_set+=pickle.load(open(data_dir+f, 'rb'))
	if len(data_set) == 0:
		print('** no data available **')
		return []
	print("----- finished importing data -----\n")

	return data_set

"""
cuts down the amount of NOOPs  in the dataset
"""
def prune_noops(data):
	noops=[i for i in data if i[1]==0]
	non_noops=[i for i in data if i[1]>0]
	avg_act_count=int(len(non_noops)/3)
	pruned_noops=random.sample(noops, avg_act_count)
	return pruned_noops + non_noops

"""
use this to random sample a data point from the data folder and plot its contents
"""
def visualize_block(data_point=None):
	
	if data_point is None:
		data_set=import_data()
		if len(x)==0:
			print('No data to visualize')
			return
		N=len(x)
		idx=np.random.randint(0, N)
		data_point=x[idx][0]
		print('picked data point {} out of {}'.format(idx, N))
	
	#plot image
	fig = plt.figure(figsize=(10,10))
	rows=2
	columns=2
	plt.title('Visualization of a Data Point')
	plt.axis('off')

	grid = ImageGrid(fig, 111,
					 nrows_ncols=(rows, columns),
					 axes_pad=.1)
	
	for ax, im in zip(grid, data_point):
		ax.imshow(im, cmap='gray')
		ax.axis('off')

	plt.show()


"""
use this to random sample a data point and plot its contents
"""
def visualize_frame(frame=None):
	if frame is None:
		return

	#plot image
	plt.imshow(frame, cmap='gray')
	plt.show()

"""

"""
def calc_conv_output(dims, kernel_size, stride, pads=(0,0)):
	height=(dims[0]-kernel_size+(2*pads[0]))/stride+1
	width=(dims[1]-kernel_size+pads[1])/stride+1
	return (height, width)

def calc_pool_output(dims, pool_size, pads=(0,0)):
	return ((dims[0]+(2*pads[0]))/pool_size, (dims[1]+(2*pads[1]))/pool_size)
	

