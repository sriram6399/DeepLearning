import sys
import os
import numpy as np
import gym
from gym.utils.play import *
import pickle
import datetime

from utilities import data_point, data_collector

#globals
point=data_point()
data=[]
start_key=ord(' ')
in_progress=False
LEFT=ord('i')
RIGHT=ord('p')

#action_map={(None,): 0, (32,): 1, (100,): 2, (97,): 3}
action_map={(None,): 0, (32,): 1, (RIGHT,): 2, (LEFT,): 3}

if __name__ == '__main__':

	#create data path
	data_path='../data/'
	if not os.path.exists(data_path):
			os.makedirs(data_path)

	data_path+='demonstrations/'
	if not os.path.exists(data_path):
			os.makedirs(data_path)


	#file parameters
	collector=data_collector()
	time=datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
	fname=data_path+'demonstrator_{}.pickle'.format(time)

	#create gym environment and collect data
	play(env = gym.make('BreakoutDeterministic-v4', obs_type='grayscale'), zoom=4, callback=collector.callback, keys_to_action=action_map)

	#prune and process data
	collector.convert_noops_breakout()
	collector.prune_noops()

	#save data to path
	with open(fname, 'wb') as fh:
		pickle.dump(collector.dump_data(), fh)

	print('*** data written to {} ***'.format(data_path))

