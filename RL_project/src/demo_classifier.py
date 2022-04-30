import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data

import torchvision.transforms as T
from utilities	  import *
import numpy as np
from builtins import *
import matplotlib.pyplot as plt
import gym

num_classes=4



class Demo_Classifier:
	batch_size=4000
	if torch.cuda.is_available():
		print('using GPU')
		device='cuda'
	else:
		print('using CPU')
		device='cpu'
	dtype = torch.float32
	def preprocessing(self):
		#transform = T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		#data_set = torchvision.datasets.ImageFolder(root='../data', transform=transform)
		data_set= import_data()

		train_set_size = int(len(data_set) *0.8)
		test_set_size = len(data_set) - train_set_size
		train_set, test_set = data.random_split(data_set, [train_set_size, test_set_size])
		loader_train = DataLoader(train_set, batch_size=self.batch_size)
		loader_test = DataLoader(test_set, batch_size=self.batch_size)
		return loader_train, loader_test
		
	def cnn_model(self):
		layer1 = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=8, stride=4),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=4, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)


		fc1 = nn.Sequential(
			nn.Linear(32*2*2, 512),
			nn.LayerNorm(512),
			nn.ReLU()
		)

		fc2 = nn.Linear(512, 4)

		model = nn.Sequential(
			layer1,
			layer2,
			nn.Flatten(),
			fc1,
			fc2
		)
		return model

	def check_accuracy(self,loader_train, loader_test, model):
		num_correct = 0
		num_samples = 0
		model.eval()  

		with torch.no_grad():
			for x, y in loader_train:
				x = x.to(device=self.device, dtype=self.dtype)	  # move to device, e.g. GPU
				y = y.to(device=self.device, dtype=torch.int64)
				scores = model(x)
				preds = torch.argmax(scores, dim=1)

				num_correct += (preds == y).sum()
				num_samples += preds.size()[0]
			trn_acc = num_correct / num_samples
			print('Training Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * trn_acc))



		num_correct = 0
		num_samples = 0
		with torch.no_grad():
			for x, y in loader_test:
				x = x.to(device=self.device, dtype=self.dtype)	  # move to device, e.g. GPU
				y = y.to(device=self.device, dtype=torch.int64)
				scores = model(x)
				preds = torch.argmax(scores, dim=1)

				num_correct += (preds == y).sum()
				num_samples += preds.size()[0]
			val_acc = num_correct / num_samples
			print('Validation: Got %d / %d correct (%.2f)\n' % (num_correct, num_samples, 100 * val_acc))

		return trn_acc, val_acc


	def train_part(self,model,loader_train, loader_test):
		epochs = 40
		#learning_rate = 3e-5
		learning_rate = 1e-4
		weight_decay = 0.01
		optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		model = model.to(device=self.device) 

		training_accuracy=[]
		validation_accuracy=[]

		trn_acc, val_acc=self.check_accuracy(loader_train, loader_test, model)
		training_accuracy.append(trn_acc.cpu())
		validation_accuracy.append(val_acc.cpu())
		

		for e in range(epochs):
			for t,(x, y) in enumerate(loader_train):

				model.train()  
				x = x.to(device=self.device, dtype=self.dtype)
				y = y.to(device=self.device, dtype=torch.int64)

				scores = model(x)
				loss = torch.nn.CrossEntropyLoss()(scores, y)
		   
				optimizer.zero_grad()

				loss.backward()

				optimizer.step()

				if t % 100 == 0:
					print('Epoch: {0}, Iteration {1}, loss: {2:.4f}'.format(e+1, t+1, loss.item()))
					trn_acc, val_acc=self.check_accuracy(loader_train, loader_test, model)
					training_accuracy.append(trn_acc.cpu())
					validation_accuracy.append(val_acc.cpu())
					print()
		
		rng=range(0, 100*len(training_accuracy), 100)
		fig, ax=plt.subplots()

		plt.ylim(0, 1)

		ax.plot(rng, training_accuracy,'k--', label='training accuracy')
		ax.plot(rng, validation_accuracy, 'k:', label='validation accuracy')
		
		legend=ax.legend(loc='upper right', shadow=True)
		legend.get_frame().set_facecolor('C0')

		plt.xlabel('Training Iteration')
		plt.ylabel('Accuracy')
		plt.title('Behavior Cloning Training')

		plt.show()
					
		return model
			
	
if __name__ == '__main__':
	data_dir = '../models/bc/'
	if not os.path.isdir(data_dir): 
		os.mkdir(data_dir)
	fname= data_dir+'bc_model.h5'
	loader_train_set,loader_test_set = Demo_Classifier().preprocessing()   
	model = Demo_Classifier().cnn_model()
	model_aftertrain=Demo_Classifier().train_part(model,loader_train_set, loader_test_set)
	torch.save(model_aftertrain, fname)
	
	
	
