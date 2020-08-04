###################
## Replay Buffer ##
###################

import numpy as np
import torch

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
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
