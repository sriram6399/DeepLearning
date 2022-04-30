import gym
from utilities	  import *
import torch
import torch.nn as nn
from ale_py._ale_py import Action
import time
from ranked_net import *

def test():
	if torch.cuda.is_available():
		print('using GPU')
		device='cuda'
	else:
		print('using CPU')
		device='cpu'
	dtype = torch.float32
	model= torch.load("../models/bc/bc_model.h5")
	reward_net= torch.load("../models/reward_net/reward_net.h5")
	model.eval()
	env = gym.make('BreakoutDeterministic-v4', obs_type='grayscale', render_mode='human')
	env.reset()
	
	all_rewards=[]
	pred_rewards=[]
	blocks=[]
	
	for i in range(2):
		done = False
		pt=data_point()
		pt.add_frame(env.reset())
		env.step(Action.FIRE)
		lives=5
		reward_env = 0
		states=[]
		
		no_of_blocks=0
		while not done:
			state=pt.get()
			states.append(state)
			action = model(torch.tensor(state[np.newaxis], device=device, dtype=dtype))
			action=torch.argmax(action[0])
			print('action: {}'.format(action))
			new_state, reward, done, info = env.step(action)
			if reward>0:
				no_of_blocks= no_of_blocks + 1
			reward_env = reward_env+reward
			if(lives>info["lives"]):
				env.step(Action.FIRE)
				lives=lives-1
			print(info["lives"])
			#point.point= point.point.numpy
			#print(new_state.shape)
			pt.add_frame(new_state)
			if done:
				blocks.append(no_of_blocks)
				all_rewards.append(reward_env)
				break
	    
		#observation = env.reset()
	env.close()
	print(blocks)
	print(all_rewards)
	
	rng=range(0, 100*len(all_rewards), 100)
	fig, ax=plt.subplots()

	plt.ylim(0, 1)

	ax.plot(rng, all_rewards,'k--', label='rewards')
	ax.plot(rng, blocks, 'k:', label='blocks')
		
	legend=ax.legend(loc='upper right', shadow=True)
	legend.get_frame().set_facecolor('C0')


	plt.show()
	
if __name__ == '__main__':
	net = test()
