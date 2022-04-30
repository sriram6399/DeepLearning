import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T
from utilities import *
import numpy as np
from builtins import *
import matplotlib.pyplot as plt
import math
#from caffe2.python.embedding_generation_benchmark import device




class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(4, 32, 7, stride=3)
		self.conv2 = nn.Conv2d(32, 16, 5, stride=2)
		self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
		self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
		self.fc1 = nn.Linear(784, 64)
		self.fc2 = nn.Linear(64, 1)
		#self.fc1 = nn.Linear(16*7*7,256)
		#self.fc2 = nn.Linear(256,64)
		#self.fc3 = nn.Linear(64,1)



	def cum_return(self, traj):
		sum_rewards = 0
		sum_abs_rewards = 0
		x = traj.permute(0,1,3,2) 
		#print(x.shape)
		x = F.leaky_relu(self.conv1(x))
		x = F.leaky_relu(self.conv2(x))
		x = F.leaky_relu(self.conv3(x))
		x = F.leaky_relu(self.conv4(x))
		x = x.view(-1, 784)
		x = F.leaky_relu(self.fc1(x))
		#x= F.leaky_relu(self.fc2(x))
		r = self.fc2(x)
		sum_rewards += torch.sum(r)
		sum_abs_rewards += torch.sum(torch.abs(r))
		return sum_rewards, sum_abs_rewards



	def forward(self, traj_i, traj_j):
		cum_r_i, abs_r_i = self.cum_return(traj_i)
		cum_r_j, abs_r_j = self.cum_return(traj_j)

		return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j




def train_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg):
	data_dir = '../models/reward_net/'
	if not os.path.isdir(data_dir): 
		os.mkdir(data_dir)
	fname=fname+data_dir+'reward_net_best.h5'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	loss_criterion = nn.CrossEntropyLoss()

	cum_loss = 0
	training_data = list(zip(training_inputs, training_outputs))
	np.random.shuffle(training_data)
	training_data_size = int(len(training_data) * 0.8)
	training_dataset = training_data[:training_data_size]
	validation_dataset = training_data[training_data_size:]

	best_v_accuracy = -np.float('inf')
	early_stop = False
	validation_accuracy=[]
	print('num_iter: {}, len(data): {}'.format(num_iter, len(training_dataset)))
	for epoch in range(num_iter):
		np.random.shuffle(training_dataset)
		training_obs, training_labels = zip(*training_dataset)
		
		
		for i in range(len(training_labels)):
			
			traj_i, traj_j = training_obs[i]
			
			traj_i= torch.stack([torch.tensor(i, dtype=torch.float32) for i in traj_i])
			traj_j= torch.stack([torch.tensor(i, dtype=torch.float32) for i in traj_j])
			
			traj_i=traj_i.to(device)
			traj_j=traj_j.to(device)
			labels = np.array([training_labels[i]])
			labels = torch.from_numpy(labels).to(device)
			
			
			optimizer.zero_grad()

			outputs, abs_rewards = reward_network.forward(traj_i, traj_j)

			outputs = outputs.unsqueeze(0)
			#labels=labels.squeeze(0)
			loss = loss_criterion(outputs, labels.long()) + l1_reg * abs_rewards
			loss.backward()
			optimizer.step()
			item_loss = loss.item()
			cum_loss += item_loss
			if i % 300 == 299:
				print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
				validation_obs, validation_labels = zip(*validation_dataset)
				v_accuracy = calc_accuracy(reward_net, validation_obs, validation_labels)
				validation_accuracy.append(v_accuracy)
				print("Validation accuracy = {}".format(v_accuracy))
				if v_accuracy > best_v_accuracy:
					print("check pointing")
					torch.save(reward_net,fname)
					best_v_accuracy = v_accuracy
				
				print(abs_rewards)
				cum_loss = 0.0
		
	rng=range(0, 100*len(validation_accuracy), 100)
	plt.plot(rng,validation_accuracy)	
	plt.xlabel('Accuracy for every 1000 iterations')
	plt.ylabel('Accuracy')
	plt.title('Behavior Cloning Training')

	plt.show()
	

def calc_accuracy(reward_network, training_inputs, training_outputs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	loss_criterion = nn.CrossEntropyLoss()
	num_correct = 0.
	with torch.no_grad():
		for i in range(len(training_inputs)):
			label = training_outputs[i]
			traj_i, traj_j = training_inputs[i]
			traj_i = np.array(traj_i)
			traj_j = np.array(traj_j)
			traj_i = torch.from_numpy(traj_i).float().to(device)
			traj_j = torch.from_numpy(traj_j).float().to(device)
			outputs, abs_return = reward_network.forward(traj_i, traj_j)
			_, pred_label = torch.max(outputs,0)
			if pred_label.item() == label:
				num_correct += 1.
	return num_correct / len(training_inputs)



if __name__ == '__main__':
	
	data_dir = '../models/reward_net/'
	if not os.path.isdir(data_dir): 
		os.mkdir(data_dir)
	fname=fname+data_dir+'reward_net.h5'
	
	data_dir1 = '../data/ranked_demos'
	
	fname1=fname1+data_dir1+'training_data_reward.pickle'
	
	data_set=pickle.load(open(fname1, 'rb'))
	training_x = data_set[0]
	training_y = data_set[1]
			
	print("jvjh",len(training_x[0]))   
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	reward_net = Net()
	reward_net.to(device)
	lr=1e-5
	weight_decay= 0
	num_iter = 10
	l1_reg=0
	
	optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
	train_reward(reward_net, optimizer, training_x, training_y, num_iter, l1_reg)
	
	print("accuracy", calc_accuracy(reward_net, training_x, training_y))
	torch.save(reward_net, fname)
