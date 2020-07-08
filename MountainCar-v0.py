# -*- coding: utf-8 -*-
######################
##  MountainCar-V0  ##
######################

# Pit-to-yonde-kudasai 

import torch
import gym
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

config = dict(
    epsilon = 0.8,
    learning_rate = 0.2,
    gamma = 0.9,
    episodes = 5000,
)

# Environment        
env = gym.make("MountainCar-v0")

# We discretize the state space
num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

# We create a matrix for our actions
# Initialize Q table
# Our table is a n x m x k matrix where:
# - n = number of states 1
# - m = number of states 2
# - k = number of actions
# The element Q[n,m,k] = Expected reward taking action k when we are in state (n,m)
Q = np.random.uniform(low = -1, high = 1, 
                      size = (num_states[0], num_states[1], 
                      env.action_space.n))

reduction = (config['epsilon'])/config['episodes']

reduction = (config['epsilon'])/config['episodes']
reduction

rewards = []
avg_rew = []
epsilon = config['epsilon']
for t in range(config['episodes']):
  state = env.reset()
  prev_state = []
  done = False
  score = 0

  state_adj = (state - env.observation_space.low)*np.array([10, 100])
  state_adj = np.round(state_adj, 0).astype(int) 

  while done != True:
    if np.random.random() < 1 - epsilon:
      action = np.argmax(Q[state_adj[0], state_adj[1]])                                 
    else:
      action = np.random.randint(0, env.action_space.n)

    state2, rew, done, info = env.step(action)
    state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
    state2_adj = np.round(state2_adj, 0).astype(int) 

    if done == True and state2[0] >= 0.5:
      Q[state_adj[0], state_adj[1], action] = rew
    else:
      delta =  config['learning_rate']*\
      (rew + config['gamma']*np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1], action])   
      Q[state_adj[0], state_adj[1], action] += delta

      state_adj = state2_adj
      
    # Decay epsilon
    if epsilon > 0:
      epsilon -= reduction
    score += rew
  rewards.append(score)
  avg_rew.append(np.mean(rewards))
  if t % 100 == 99:
    print('Episode:',t + 1,'Score:',np.mean(rewards[-100:-1]))

# We plot the average rewards
plt.plot(100*(np.arange(len(avg_rew)) + 1), avg_rew)
plt.show()

state = env.reset()
done = False

# We render an episode with the trained agent
while done != True:
  state_adj = (state - env.observation_space.low)*np.array([10, 100])
  state_adj = np.round(state_adj, 0).astype(int)
  env.render()
  action = np.argmax(Q[state_adj[0], state_adj[1]])
  state, rew, done, info = env.step(action)

