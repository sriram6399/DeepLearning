import gym
from utilities      import *
import torch
import torch.nn as nn
from ale_py._ale_py import Action
import time
import csv



class ranked_traj:
    def get_ranked(self,epsilon):
        if torch.cuda.is_available():
            print('using GPU')
            device='cuda'
        else:
            print('using CPU')
            device='cpu'
            dtype = torch.float32
            
        data_dir = '../models/bc/'
        fname= fname+data_dir+'bc_model.h5'
        model= torch.load(fname)
        model.eval()
        pt= data_point() 
        env = gym.make('BreakoutDeterministic-v4', obs_type='grayscale', render_mode='human')
        env.reset()
        rewards= []
        traj= []
        for i in range(30):
            done = False
            pt=data_point()
            pt.add_frame(env.reset())
            env.step(Action.FIRE)
            lives=5
            cur_reward=0
            states=[]
            while not done:
            
                #state=torch.tensor(pt.get()[np.newaxis], device=device, dtype=dtype)
                state=pt.get()
                states.append(state)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = model(torch.tensor(state[np.newaxis], device=device, dtype=dtype))
                    action=torch.argmax(action[0])
                    
                print('action: {}'.format(action))
                new_state, reward, done, info = env.step(action)
                cur_reward = cur_reward+reward
                if done:
                    print("Episode: {}, Reward: {}".format(i,cur_reward))
                    rewards.append(cur_reward)
                    break
                if(lives>info["lives"]):
                    env.step(Action.FIRE)
                    lives=lives-1
                print(info["lives"])
                
                pt.add_frame(new_state)
            traj.append(states)    
         
        env.close()
        return traj, rewards
  
  
def get_unique(n):
    arr = list(range(0, n))
    
    for i in range(0,n):
        x=random.randint(i,n-1) 
        temp=arr[i]
        arr[i]=arr[x]
        arr[x]=temp
    return arr


def training_data_for_reward(ranked_demos, num_snippets, min_snippet_length, max_snippet_length):
    step = 4
    
    training_x = []
    training_y = []
    bins_length = len(ranked_demos)
    arr= get_unique(bins_length)
    
    for n in range(num_snippets):
        bi = arr[n % bins_length]
        bj = arr[(n+1) % bins_length]  
        ti = random.choice(ranked_demos[bi])
        tj = random.choice(ranked_demos[bj])
       
       
        min_length = min(len(ti), len(tj))
        if min_length<min_snippet_length or min_length == 0:
            continue
        rand_length = np.random.randint(min_snippet_length, min(min_length,max_snippet_length))
        
        if bi < bj: 
            ti_start = np.random.randint(min_length - rand_length + 1)
            tj_start = np.random.randint(ti_start, len(tj) - rand_length + 1)
        else: 
            tj_start = np.random.randint(min_length - rand_length + 1)
            ti_start = np.random.randint(tj_start, len(ti) - rand_length + 1)
            
        snip_i = ti[ti_start:ti_start+rand_length:step] 
        snip_j = tj[tj_start:tj_start+rand_length:step]
           
        if bi > bj:
            label = 0
        else:
            label = 1
        #training_x.append([snip_i.numpy(), snip_j.numpy()])
        training_x.append((snip_i, snip_j))
        #training_x.append(np.array([snip_i.numpy(), snip_j.numpy()]))
        training_y.append(label)
        

    print("this is",len(training_x[0]))
    return training_x, training_y   



      
if __name__ == '__main__':
    data_dir = '../data/ranked_demos'
    if not os.path.isdir(data_dir): 
        os.mkdir(data_dir)
    fname=fname+data_dir+'training_data_reward.pickle'
    epsilon_val=[1.0,0.67,0.37,0.01]
    num_snippets = 4000
    min_snippet_length = 10
    max_snippet_length = 100
    ranked_trajectories=[]
    ranked_demos=[]
    rank_obj = ranked_traj()
    for epsilon in epsilon_val:
        traj, reward = rank_obj.get_ranked(epsilon)
        ranked_trajectories.append(traj)
        print('traj shape: {}'.format(traj[0][0].shape))
        ranked_demos.append({"epsilon":epsilon,"trajectories":traj,"rewards":reward})
    
    _ranked_demos = ranked_trajectories
    ranked_trajectories = []
    for _r in _ranked_demos:
        r = []
        for _d in _r:
            d = []
            for _ob in _d:
                d.append(_ob)
            r.append(d)
        ranked_trajectories.append(r)
    
    training_x,training_y = training_data_for_reward(ranked_trajectories, num_snippets, min_snippet_length, max_snippet_length)
    #cprint(type(training_x[0][0][0]))
    with open(fname, 'wb') as fh:
        pickle.dump((training_x,training_y), fh)
        
            