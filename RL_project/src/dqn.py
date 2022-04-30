import numpy as np
import gym
import random
from gym.utils.play import *
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from utilities import data_point, visualize_block
import math
import datetime
import sys
import os
from ranked_net import Net
import pickle


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

class dqn(nn.Module):
	def __init__(self):
		super(dqn, self).__init__()
		layer1 = nn.Sequential(
			#nn.Conv2d(4, 16, kernel_size=8, stride=4, padding=(1, 4)),
			nn.Conv2d(4, 16, kernel_size=8, stride=4),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		layer2 = nn.Sequential(
			#nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=(0, 2)),
			nn.Conv2d(16, 32, kernel_size=4, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		fc1 = nn.Sequential(
			#nn.Linear(32*6*5, 512),
			nn.Linear(32*2*2, 512),
			nn.LayerNorm(512),
			nn.ReLU()
		)

		fc2 = nn.Linear(512, 4)


		#initialize weights
		nn.init.kaiming_normal_(layer1[0].weight,
								mode='fan_in',
								nonlinearity='relu')

		nn.init.kaiming_normal_(layer2[0].weight,
								mode='fan_in',
								nonlinearity='relu')

		nn.init.kaiming_normal_(fc1[0].weight,
								mode='fan_in',
								nonlinearity='relu')
	
		nn.init.kaiming_normal_(fc2.weight,
								mode='fan_in',
								nonlinearity='relu')


		self.model = nn.Sequential(
			layer1,
			layer2,
			nn.Flatten(),
			fc1,
			fc2
		)

	def forward(self, x):
		return self.model(x)

	def update_model(self, memories, batch_size, gamma, 
					 trgt_model, device, optimizer):

		self.train()

		#sample minibatch
		minibatch=random.sample(memories, batch_size)

		#split tuples into groups and convert to tensors
		states =		torch.stack([i[0] for i in minibatch]).to(device)
		actions =		   np.array([i[1] for i in minibatch])
		rewards =		torch.stack([i[2] for i in minibatch]).to(device)
		next_states =	torch.stack([i[3] for i in minibatch]).to(device)
		not_done =		torch.stack([i[4] for i in minibatch]).to(device)

		#create predictions
		policy_scores=self.forward(states)
		policy_scores=policy_scores[range(batch_size), actions]

		#create max(Q vals) from target policy net
		trgt_policy_scores=trgt_model(next_states)
		trgt_qvals=trgt_policy_scores.max(1)[0]

		#create labels
		#y=policy_scores.clone().detach()
		#y[range(batch_size), actions] = rewards + gamma*trgt_qvals*not_done
		y = rewards + gamma*trgt_qvals*not_done

		#compute loss using Huber Loss
		loss_fn = nn.SmoothL1Loss()
		loss=loss_fn(policy_scores, y)

		#gradient descent step
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
		optimizer.step()

		self.eval()

'''
local functions
'''

def epsilon_update(epsilon, eps_start, eps_end, eps_decay, step):
	return eps_end + (eps_start - eps_end) * math.exp(-1 * step / eps_decay)

def clip_r(r):
	if r > 0:
		return 1.
	elif r == 0:
		return 0.
	else:
		return -1.

def main(pre_trained_model=None, eps_start=1., episodes=20000, 
		 batch_size=32, reward_model=None, session_num=0):
	
	##############################creat paths##############################

	#create model path
	ptm_path='../models/dqn/'
	if not os.path.exists(ptm_path):
		os.makedirs(ptm_path)
	
	session_id=0
	metric_path='../metric_data/'

	#create path for metrics
	session_dir=metric_path+'session_' + str(session_num) + '/'
	if not os.path.exists(session_dir):
		os.makedirs(session_dir)
	
	#####################environment initilization#####################

	#load in reward network if specified
	suffix='_vanilla'
	reward_net=None
	if reward_model is not None:
		reward_net_path='../models/reward_net/'
		if not os.path.exists(reward_net_path + reward_model):
			print('invalid reward file name')
			return
		else:
			reward_net=torch.load(reward_net_path + reward_model,
								  map_location=torch.device(device))
			reward_net.eval()
			suffix='_trex'
			print('loaded reward network')

	

	gamma=.99				#gamma for MDP
	alpha=6.25e-5			#learning rate

	#epsilon greedy parameters
	epsilon=eps_start
	eps_end=.1
	#eps_decay=75e3
	eps_decay=2e5

	update_steps=4			#update policy after every n steps
	C=10000					#update target model after every C steps
	dtype=torch.float32		#dtype for torch tensors
	total_steps=0			#tracks global time steps

	memory_size=30000		#size of replay memory buffer
	episode_scores=[]

	
	#create gym environment
	env = gym.make('BreakoutDeterministic-v4', obs_type='grayscale', render_mode='human')

	#get action space
	action_map=env.get_keys_to_action()
	A=env.action_space.n
	
	#load pre-trained model if specified
	if pre_trained_model is not None:
		policy_net=torch.load(ptm_path + pre_trained_model,
							  map_location=torch.device(device))
		print('loaded pre-trained model')
	else:
		policy_net=dqn().to(device)

	policy_net.eval()

	#initialize target network
	trgt_policy_net=dqn().to(device)
	trgt_policy_net.load_state_dict(policy_net.state_dict())
	trgt_policy_net.eval()

	#initialize optimizer
	optimizer=optim.Adam(policy_net.parameters(), lr=alpha)

	#initialize some variables before getting into the main loop
	replay_memories=[]

	#keeps track of last 500 episodic rewards
	reward_hist =	{'true':[],
					 'clip':[],
					 'rhat':[]}

	#keeps track of average rewards per 500 episodes
	reward_avgs =	{'true':[],
					 'clip':[],
					 'rhat':[]}

	steps_done=0

	################################training loop################################

	for ep in range(episodes):
		total_reward=0

		true_reward=0
		clip_reward=0
		rhat_reward=0
		s_builder=data_point()				#initialize phi transformation function
		s_builder.add_frame(env.reset())	

		s=s_builder.get()					#get current state

		t=1									#episodic t
		done=False							#tracks when episodes end
		lives=5
		while not done:
			
			#select action using epsilon greedy policy
			if np.random.uniform(0, 1) < epsilon:
				a=np.random.randint(0, A)
			else:
				with torch.no_grad():
					q_vals=policy_net(torch.tensor(s[np.newaxis], dtype=dtype, device=device))
					a=int(torch.argmax(q_vals[0]))

			#update epsilon value
			epsilon = epsilon_update(epsilon, eps_start, eps_end, eps_decay, total_steps)

			#take action and collect reward and s'
			s_prime_frame, r, done, info = env.step(a) 
			s_builder.add_frame(s_prime_frame)
			s_prime=s_builder.get()
			
			#process reward and increment reward counters
			true_reward+=r
			clip_reward+=clip_r(r)

			#use either reward net or clipping for dqn training
			if reward_net is not None:
				with torch.no_grad():
					_, r=reward_net.cum_return(torch.tensor(s_prime[np.newaxis],
											   dtype=dtype, 
											   device=device))
				#scale reward using sigmoid
				r=1/(1+np.exp(-float(r)))
				rhat_reward+=r
			else:
				r=clip_r(r)

			#use to feed lost life as an end state
			res=done
			if lives != info['lives']:
				res=True
				lives=info['lives']

			#append to replay_memories as (s, a, r, s', done)
			replay_memories.append((torch.tensor(s,			dtype=dtype),
									a,
									torch.tensor(r,			dtype=dtype),
									torch.tensor(s_prime,	dtype=dtype),
									torch.tensor(not res,	dtype=torch.bool)))

			#remove oldest sample to maintain memory size
			if len(replay_memories) > memory_size:
				del replay_memories[:1]

			#perform gradient descent step
			if len(replay_memories) > batch_size and total_steps % update_steps == 0:
				policy_net.update_model(replay_memories, batch_size, gamma, 
										trgt_policy_net, device, optimizer)

			#set target weights to policy net weights every C steps
			if total_steps % C == 0:
				trgt_policy_net.load_state_dict(policy_net.state_dict())

			#increment counters
			total_steps+=1
			t+=1
			total_reward+=r

			#update state
			s=s_prime

		# save model checkpoint every 200 episodes
		if ep % 200  == 0:
			time=datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
			fname=ptm_path + 'dqn_checkpoint_' + time + suffix + '.pth'
			torch.save(policy_net, fname)
			print('model checkpoint saved to {}'.format(fname))

		#save data to session directory every 500 episodes
		if ep > 0 and ep % 500 == 0:
			#append averages
			reward_avgs['true'].append(np.mean(reward_hist['true']))
			reward_avgs['clip'].append(np.mean(reward_hist['clip']))
			if reward_net is not None:
				reward_avgs['rhat'].append(np.mean(reward_hist['rhat']))
			
			#save to pickle file
			with open(session_dir + 'dqn_rewards.pickle', 'wb') as fh:
				pickle.dump(reward_avgs, fh)
			print('saved session data to {}'.format(session_dir + 'dqn_rewards.pickle'))

		#maintain history sizes
		if ep >= 500:
			del reward_hist['true'][:1]
			del reward_hist['clip'][:1]
			if reward_net is not None:
				del reward_hist['rhat'][:1]
				
		#append rewards
		reward_hist['true'].append(true_reward)
		reward_hist['clip'].append(clip_reward)
		if reward_net is not None:
			reward_hist['rhat'].append(rhat_reward)

		#print metrics
		print('--------------------------------')
		print(('episode: {0}\n' +
			   'episode length: {1}\n' + 
			   'epsilon: {2:.2f}\n' +
			   'total time: {3}\n' +
			   'true reward: {4:.2f} ({5} game avg: {6:.2f})\n' +
			   'blocks hit: {7:.2f} ({8} game avg: {9:.2f})').format(
			   								ep, 
											t, 
											epsilon, 
											total_steps, 
											true_reward,
											len(reward_hist['true']),
											np.mean(reward_hist['true']),
											clip_reward,
											len(reward_hist['clip']),
											np.mean(reward_hist['clip'])))

		if reward_net is not None:
			print('rhat: {0:.2f} (avg: {1:.2f})'.format(rhat_reward, 
														np.mean(reward_hist['rhat'])))
		print('--------------------------------')
				
				
if __name__ == '__main__':
	
	#parse arguments
	parser=argparse.ArgumentParser(description='trains a model using deep q learning')
	parser.add_argument('--ptm', metavar='ptm', type=str,
						default=None, help='name of the pre-trained model')

	parser.add_argument('--eps_start', metavar='e', type=float,
						default=1., help='starting epsilon value (default 1)')

	parser.add_argument('--episodes', metavar='eps', type=int,
						default=20000, help='number of episodes to train for (default 20000)')

	parser.add_argument('--batch_size', metavar='bs', type=int,
						default=32, help='training batch size (default 32)')

	parser.add_argument('--reward_model', metavar='rm', type=str,
						default=None, help='name of the reward network')

	parser.add_argument('--session_num', metavar='sn', type=int,
						default=0, help='manual session id for storing metric data')

	args=parser.parse_args()

	#run main function
	main(pre_trained_model =	args.ptm,
		 eps_start =			args.eps_start,
		 episodes =				args.episodes,
		 batch_size =			args.batch_size,
		 reward_model =			args.reward_model,
		 session_num =			args.session_num,
	)


