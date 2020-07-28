################
## PPO Buffer ##
################

import torch
import numpy as np
from ppo_tools import combined_shape, discount_cumsum, get_statistics

class PPO_Buffer:
  def __init__(self, obs_size, act_size, size, gamma = 0.99, lam = 0.95):
    self.obs_buf = np.zeros(combined_shape(size, obs_size), dtype = np.float32)
    self.act_buf = np.zeros(combined_shape(size, act_size), dtype = np.float32)
    self.adv_buf = np.zeros(size, dtype = np.float32)
    self.rew_buf = np.zeros(size, dtype = np.float32)
    self.ret_buf = np.zeros(size, dtype = np.float32)
    self.val_buf = np.zeros(size, dtype = np.float32)
    self.logp_buf = np.zeros(size, dtype = np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size

  def store(self, obs, act, rew, val, logp):
    assert self.ptr < self.max_size
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val
    self.logp_buf[self.ptr] = logp
    self.ptr += 1

  def finish_path(self, last_val = 0):
    path_slice = slice(self.path_start_idx, self.ptr)
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)

    # GAE-lambda
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1] # TD
    self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

    # Rewards-to-go (Critic Update)
    self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

    self.path_start_idx = self.ptr

  def get(self):
    assert self.ptr == self.max_size
    self.ptr, self.path_start_idx = 0, 0

    # Advantage Normalization
    adv_mean, adv_std = get_statistics(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean)/ adv_std
    data = dict(obs = self.obs_buf, act = self.act_buf, ret = self.ret_buf,
                adv = self.adv_buf, logp = self.logp_buf)
    return {k: torch.as_tensor(v, dtype = torch.float32) for k,v in data.items()}

