######################
## KULLBACK-LEIBLER ##
######################
# In this script we implement a neural network that learns how to reduce the KL
# divergence in a supervised environment

import numpy as np
import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

x = torch.rand([100,4], dtype = torch.float32)
y = torch.randn([])

def KL(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

## ----- Actor ----- ##
# This NN, given the inputs, will return the discrete probability distribution
class NN(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim, activation):
    super().__init__()
    self.logits_net = mlp([obs_dim] + list(hid_dim) + [act_dim], activation)
    self.pi = nn.Softmax(dim=1)

  def forward(self, obs):
    out = self.logits_net(obs)
    return self.pi(out)

policy = NN(4,5,[20,20], activation = nn.Tanh)

m = nn.Softmax(dim=1)

x = torch.rand([100,4], dtype = torch.float32)
y = torch.log(torch.rand([100,5], dtype = torch.float32))
y = m(y)

criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = Adam(policy.parameters(), lr = 1e-2)

losses = []
for i in range(4000):
  y_pred = policy.forward(x)
  y_pred_log = torch.log(y_pred)

  optimizer.zero_grad()
  loss = criterion(y_pred_log, y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()


plt.plot(losses)
plt.show()

