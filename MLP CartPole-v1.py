# -*- coding: utf-8 -*-

######################
## NN Training Test ##
######################
# In this script we will train a NN to play with the classic gym game
# CartPole.

import torch
import gym
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# We create our NN class. A multilayer perceptron
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, observations):
        out = self.layer1(observations)
        out = nn.functional.relu(out)
        out = self.layer2(out)
        return out

config = dict(
    input_size = 4,
    output_size = 2,
    hidden_size = 10,
    learning_rate = 0.000001,
    score_threshold = 50
    )

training_set = []
scores = []

# Policy initialization
model = MLP(config['input_size'],
            config['output_size'],
            config['hidden_size'])

# Environment        
env = gym.make("CartPole-v1")
env.reset()

# Creating the training set
# Number of times we reset the game (number of trajectories)
for _ in range(300):
    score = 0
    game_memory = []
    prev_obs = []
    # Maximum number of actions for each trajectory 
    for _ in range(500):   
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if len(prev_obs) != 0:
            if action == 1:
                act = [0,1]
            else:
                act = [1,0]
            game_memory.append([prev_obs, act])
        prev_obs = obs
        score += rew
        
        if done == True:
            env.reset()
            break
        # We keep only the trajectory that has a minimum specific value of score
        if score >= config['score_threshold']:
            scores.append(score)
            for data in game_memory:
                training_set.append(data)            
env.close()

# Using the training set we create the X and Y
X = torch.tensor([data[0] for data in training_set], dtype = torch.float)
Y = torch.tensor([data[1] for data in training_set], dtype = torch.float).reshape([len(training_set),2])

Y.shape

criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = config['learning_rate'])

# We train our NN
losses = []
for t in range(10000):
    y_pred = model.forward(X)
    loss = criterion(y_pred, Y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(losses)
plt.show()

# We use our trained NN in order to play CartPole
env = gym.make("CartPole-v1")
prev_obs = env.reset()
scores2 = []
for _ in range(10):
    score = 0
    for _ in range(500):
        env.render()
        obs_tens = torch.tensor(prev_obs, dtype = torch.float) 
        action = (model.forward(obs_tens))
        if action[0] > action[1]:
          action = 0
        else:
          action = 1
        obs, rew, done, info = env.step(action)
        prev_obs = obs
        score += rew
        
        if done == True:
            print('Game Over')
            scores2.append(score)
            prev_obs = env.reset()
            break           
env.close()

print(scores2)
print('avg score random policy:', np.mean(scores))
print('avg score train NN:', np.mean(scores2))

