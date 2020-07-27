##########################
## Actor Critic for PPO ##
##########################

import torch
import numpy as np
import torch.nn as nn

from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical

def mlp(sizes, activation, output_activation = nn.Identity):
  layers = []
  for j in range(len(sizes) - 1):
    act = activation if j < len(sizes) - 2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

## ------ ACTOR ------ ##

# Super class Actor
class Actor(nn.Module):

  def _distribution(self, obs):
    raise NotImplementedError

  def _log_prob_from_distribution(self, pi, act):
    raise NotImplementedError

  def forward(self, obs, act = None):
    pi = self._distribution(obs)
    logp_a = None
    if act is not None:
      logp_a = self._log_prob_from_distribution(pi, act)
    return pi, logp_a

# Classes that implement superclass Actor
class Categorical_Actor(Actor):
  def __init__(self, obs_size, act_size, hidd_size, activation):
    super().__init__()
    sizes = [obs_size] + list(hidd_size) + [act_size] 
    self.logits_net = mlp(sizes, activation)
  
  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits = logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)

class Gaussian_Actor(Actor):
  def __init__(self, obs_size, act_size, hidd_size, activation):
    super().__init__()
    # Define st.dev
    log_std = -0.5 * np.ones(act_size, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    sizes = [obs_size] + list(hidd_size) + [act_size] 
    self.mu_net = mlp(sizes, activation)
  
  def _distribution(self, obs):
    mu = self.mu_net(obs)
    std = torch.exp(self.log_std)
    return Normal(mu, std)
  
  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis = -1)

## ------ CRITIC ------ ##

class Critic(nn.Module):
  def __init__(self, obs_size, hid_size, activation):
    super().__init__()
    sizes = [obs_size] + list(hid_size) + [1]
    self.v_net = mlp(sizes, activation)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1)

## ------ ACTOR-CRITIC ------ ##

class Actor_Critic(nn.Module): 
  def __init__(self, obs_space, act_space, hid_size = (64,64), 
               activation = nn.Tanh):
    super().__init__()

    obs_size = obs_space.shape[0]

    if isinstance(act_space, Box):
      act_size = act_space.shape[0]
      self.actor = Gaussian_Actor(obs_size, act_size, hid_size, activation)
    elif isinstance(act_space, Discrete):
      act_size = act_space.n
      self.actor = Categorical_Actor(obs_size, act_size, hid_size, activation)
    
    self.critic = Critic(obs_size, hid_size, activation)

  def step(self, obs):
    with torch.no_grad():
      pi = self.actor._distribution(obs)
      a = pi.sample()
      logp_a = self.actor._log_prob_from_distribution(pi, a)
      v = self.critic(obs)
    return a.numpy(), v.numpy(), logp_a.numpy()

  def act(self, obs):
    return self.step(obs)[0]
