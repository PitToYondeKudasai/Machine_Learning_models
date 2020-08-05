
#########
## DQN ##
#########

import torch
import torch.nn as nn
import numpy as np
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')



## --- REPLAY BUFFER --- ##

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

## ------ Buffer ------ ##
class ReplayBuffer():
    def __init__(self, obs_size, act_size, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_size), dtype = np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_size), dtype = np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_size), dtype = np.float32)      
        self.rew_buf = np.zeros(size, dtype = np.float32)
        self.done_buf = np.zeros(size, dtype = np.float32)     
        self.size = size
        self.ptr = 0

    def insert(self,obs, act, rew, next_obs, done):
        if self.ptr >= self.size:
            self.ptr = 0
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf [self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr += 1       
        
    def sample(self, batch_size = 32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs = self.obs_buf[idxs],
                     obs2 = self.obs2_buf[idxs],
                     act = self.act_buf[idxs],
                     rew = self.rew_buf[idxs],
                     done = self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device = device) for k,v in batch.items()}

## --- DQN --- ##
    
## ------ Q network ------- ##
def mlp(sizes, activation, output_activation = nn.Identity):
  layers = []
  for j in range(len(sizes) - 1):
    act = activation if j < len(sizes) - 2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

class Q(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size = (64,64), activation = nn.Tanh):
        super().__init__()
        sizes = [obs_size] + list(hidden_size) + [act_size]
        self.q_net = mlp(sizes, activation)
        
    def forward(self, obs):
        return self.q_net(obs)

    def forward_mask(self, obs, act):
      return torch.squeeze(torch.gather(self.q_net(obs),1,act.long()), -1)
    

## ------ DQN ------ ##
class DQN(nn.Module):
    
    def __init__(self, obs_size, act_size, hidden_size = (64, 64), activation = nn.Tanh):
        super().__init__()
        self.q_target = Q(obs_size, act_size, hidden_size, activation)
        self.q_online = Q(obs_size, act_size, hidden_size, activation)
        #for p in self.q_online.parameters():
        #    p.requires_grad = True
        #for p in self.q_target.parameters():
        #    p.requires_grad = False

    def copy(self):
        self.q_target.load_state_dict(self.q_online.state_dict())

config = dict(
    gamma = 0.99,
    n_steps = 20000,
    obs_size = 4, 
    act_size = 2,
    buffer_size = 10000,
    batch_size = 64,
    lr = 1e-4,
    epsilon = 1,
    update_time = 4,
    copy_time = 10000,
    min_epsilon = 0.1,
    weight_decay = 0
    )

env = gym.make('CartPole-v0')

# ------ Create Buffer and DQN ------ #
dqn = DQN(config['obs_size'], config['act_size'], hidden_size = (256, 256))
dqn.copy()
dqn.to(torch.device(device))
buf = ReplayBuffer(config['obs_size'], 1, config['buffer_size'])

# Loss update method
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(dqn.parameters(), lr = config['lr'], weight_decay =  config['weight_decay'])

def compute_loss(data):
    obs, act ,rew, next_obs, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
    y_forecast = dqn.q_online.forward_mask(obs, act)

    y = torch.as_tensor([rew[i].item() + config['gamma'] * torch.max(dqn.q_target(next_obs)[i]).item()
                         if done[i].item() == 0 else rew[i].item() for i in
                         range(obs.shape[0])], dtype = torch.float32, device = device)
    loss = criterion(y, y_forecast)
    return loss
  
def update_online_q(data):
    loss = compute_loss(data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_agent():
    i = 0
    done = False
    cum_rew = 0
    tot_rew = []
    obs = env.reset()
    while i < 10:
        if done:
            i += 1
            tot_rew.append(cum_rew)
            cum_rew = 0
            obs = env.reset()
            done = False
        values = dqn.q_target(torch.as_tensor(obs, dtype = torch.float32, device = device))
        action = torch.argmax(values).item()
        obs, rew, done, _ = env.step(action)
        cum_rew += rew
    tot_rew = np.mean(np.array(tot_rew))
    print('Test agent performance:', tot_rew)
    return tot_rew

test_rew = []
rews = []
epsilon = config['epsilon']
eps_decay = epsilon/(0.5*config['n_steps'])

obs = env.reset()
has_quit = False
cum_rew = 0
test_counter = 0

for i in range(config['n_steps']):            
    # Epsilon greedy       
    if random.uniform(0,1) < 1 - max(epsilon, config['min_epsilon']):
        values = dqn.q_target(torch.as_tensor(obs, dtype = torch.float32, device = device))
        action = torch.argmax(values).item()
    else:
        action = random.randint(0,1)

    next_obs, rew, done, _ = env.step(action)
    buf.insert(obs, action, rew, next_obs, done)
    cum_rew += rew
    obs = next_obs        
    if i > config['buffer_size']:
        if i % config['update_time']:
            sample = buf.sample(config['batch_size'])
            update_online_q(sample)       
        if i % config['copy_time']:
            dqn.copy()
        
    if done:
        test_counter += 1
        if test_counter % 20 == 0:
            print('step:', i)
            test_rew.append(test_agent())
        rews.append(cum_rew)
        cum_rew = 0
        obs = env.reset()
        
    epsilon -= eps_decay

plt.plot(test_rew)
plt.show()

