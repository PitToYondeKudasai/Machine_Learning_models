##########
## MAIN ##
##########
from buffer import ReplayBuffer
from dqn import DQN
import environment
import numpy as np
import pygame
import torch
import random
import matplotlib.pyplot as plt

config = dict(
    gamma = 0.99,
    n_steps = 2000000,
    obs_size = 4, 
    act_size = 2,
    buffer_size = 10000,
    batch_size = 64,
    lr = 1e-4,
    epsilon = 1,
    update_time = 4,
    copy_time = 10000,
    render = False,
    min_epsilon = 0.1,
    weight_decay = 0,
    device = 'cpu'
    )

# We create the Neural Network
dqn = DQN(config['obs_size'], config['act_size'], hidden_size = (256, 256))
dqn.to(torch.device(config['device']))
dqn.copy()

# We create the environments
env = environment.Env(config['render'])

# We create the replay buffer
buf = ReplayBuffer(config['obs_size'], 1, config['buffer_size'])

# Loss update method
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(dqn.parameters(), lr = config['lr'], weight_decay =  config['weight_decay'])

def compute_loss(data):
    obs, act ,rew, next_obs, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
    y_forecast = dqn.q_online.forward_mask(obs, act)

    y = torch.as_tensor([rew[i].item() + config['gamma'] * torch.max(dqn.q_target(next_obs)[i]).item()
                         if done[i].item() == 0 else rew[i].item() for i in
                         range(obs.shape[0])], dtype = torch.float32, device = config['device'])
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
        values = dqn.q_target(torch.as_tensor(obs, dtype = torch.float32, device = config['device']))
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
done = False
cum_rew = 0
max_score = 0
test_counter = 0

for i in range(config['n_steps']):            
    # Epsilon greedy       
    if random.uniform(0,1) < 1 - max(epsilon, config['min_epsilon']):
        values = dqn.q_target(torch.as_tensor(obs, dtype = torch.float32, device = config['device']))
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
    if config['render']:
        env.update()

env.has_quit()

plt.plot(test_rew)
plt.show()
