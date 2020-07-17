## Core ##
import torch
import torch.nn as nn

# -------- CRITIC --------- #
# The critic takes a state and an action as input and returns a value
class Critic(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Critic, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, output_size)

  def forward(self, state, action):
    if len(list(state.size())) == 1:
      x = torch.cat([state, action])
    else:
      x = torch.cat([state, action],1)
    x = self.linear1(x)
    x = nn.functional.relu(x) 
    x = self.linear2(x)
    x = nn.functional.relu(x)
    x = self.linear3(x)
    return x

# -------- ACTOR --------- #
# The actor takes a state as input and return a single action
class Actor(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Actor, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, output_size)

  def forward(self, state):
    x = self.linear1(state)
    x = nn.functional.relu(x)
    x = self.linear2(x)
    x = nn.functional.relu(x)
    x = self.linear3(x)
    x = torch.tanh(x)
    return x
