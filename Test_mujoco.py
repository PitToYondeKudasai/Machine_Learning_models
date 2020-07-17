import gym
import mujoco_py
import torch
import core
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

env = gym.make('HalfCheetah-v2')
test_env = gym.make('HalfCheetah-v2')

config = dict(
    state_size = 17,
    hidden_size = 256,
    action_size = 6,
    actor_learning_rate = 1e-3,
    critic_learning_rate = 1e-3,
    tau = 0.995,
    steps = 200000,
    size_batch = 100,
    buffer_length = 1000,
    gamma = 0.99,
    num_test_episodes = 10,
    max_ep_len = 1000,
    act_limit = env.action_space.high[0],
    dtype = torch.float,
    )

# We create our networks and we set the target ones equal to the online ones
critic = core.Critic(config['state_size'] + config['action_size'], config['hidden_size'], 1)
critic_target = core.Critic(config['state_size'] + config['action_size'], config['hidden_size'], 1)
actor = core.Actor(config['state_size'], config['hidden_size'], config['action_size'])
actor_target = core.Actor(config['state_size'], config['hidden_size'], config['action_size'])

critic_target.load_state_dict(critic.state_dict())
actor_target.load_state_dict(actor.state_dict())

# We define the loss function and the optimizer
criterion = torch.nn.MSELoss(reduction = 'sum')
actor_optimizer = torch.optim.Adam(actor.parameters(), lr = config['actor_learning_rate'])
critic_optimizer = torch.optim.Adam(critic.parameters(), lr = config['critic_learning_rate'])

# We define the replay buffer
buffer = deque([0] * config['buffer_length'])

def take_action(state):
    if type(state) != torch.Tensor:
        state = torch.from_numpy(state)
        state = state.type(config['dtype'])
    action = actor(state)
    noise = 0.1 * np.random.randn(config['action_size'])
    action = np.clip([action[i].item() for i in range(config['action_size'])] + noise,-config['act_limit'], config['act_limit'])
    return action

def get_sample():
    S = random.sample(buffer, config['size_batch'])
    states = [[S[i][0][ii] for ii in range(config['state_size'])] for i in range(config['size_batch'])]
    states = torch.tensor(states, dtype = config['dtype'])
    actions = [[S[i][1][ii] for ii in range(config['action_size'])] for i in range(config['size_batch'])]
    actions = torch.tensor(actions, dtype = config['dtype'])
    rews = [[S[i][2]] for i in range(config['size_batch'])]
    rews = torch.tensor(rews, dtype = config['dtype'])
    states2 = [[S[i][3][ii] for ii in range(config['state_size'])] for i in range(config['size_batch'])]
    states2 = torch.tensor(states2, dtype = config['dtype'])
    dones = [S[i][4] for i in range(config['size_batch'])]
    return states, actions, rews, states2, dones

def test_agent():
    TestEpRet = np.array([])
    for j in range(config['num_test_episodes']):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        o = torch.tensor(o, dtype = config['dtype'])
        while not(d or (ep_len == config['max_ep_len'])):
            action_test = actor_target(o)
            action_test = [action_test[i].item() for i in range(config['action_size'])]
            o, r, d, _ = test_env.step(action_test)
            o = torch.tensor(o, dtype = config['dtype'])
            ep_ret += r
            ep_len += 1
        TestEpRet = np.append(TestEpRet, ep_ret)
    return np.mean(TestEpRet)

i = 0
losses_critic = []
losses_actor = []
score = []

while i < config['steps']:
    state1 = env.reset()
    done = False
    t = 0

    while not done and t < config['max_ep_len']:
        action = take_action(state1)
        state2, rew, done, _ = env.step(action)
        buffer.popleft()
        buffer.append([state1, action, rew, state2, done])
        state1 = np.copy(state2)
        #env.render()
        t += 1
        i += 1
        if i > config['buffer_length']:
            states, actions, rews, states2, dones = get_sample()
            actions2 = actor_target(states2)
            y_pred = critic(states,actions)
            y = torch.tensor([[rews[i].item() if dones[i] == True else rews[i].item() +\
                 config['gamma'] * critic_target(states2[i], actions2[i]).item() for i in  range(config['size_batch'])]], dtype = config['dtype'])
            y = torch.reshape(y,list(y_pred.shape))

            # ------- update critic ------- #
            loss_critic = criterion(y_pred, y)
            critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_optimizer.step()
            losses_critic.append(loss_critic.item())

            # ------- update actor ------- #
            q_pi = critic(states,actor(states))
            loss_actor = -q_pi.mean()
            actor_optimizer.zero_grad()
            loss_actor.backward()
            actor_optimizer.step()
            losses_actor.append(loss_actor.item())
            
            # Unfreeze Q-network
            for p in critic.parameters():
                p.requires_grad = True
                
            with torch.no_grad():
                for p, p_targ in zip(actor.parameters(), actor_target.parameters()):
                    p_targ.data.mul_(config['tau'])
                    p_targ.data.add_((1 - config['tau']) * p.data)
            with torch.no_grad():
                for p, p_targ in zip(critic.parameters(), critic_target.parameters()):
                    p_targ.data.mul_(config['tau'])
                    p_targ.data.add_((1 - config['tau']) * p.data)
            
            if i % 500 == 0:
                score.append(test_agent())
                print('Episodes:',int(i/500),'score:',score[-1])
    
