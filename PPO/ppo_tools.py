###############
## PPO Tools ##
###############

import numpy as np
import scipy.signal

def combined_shape(length, shape = None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis = 0)[::-1]

def get_statistics(x):
  x = np.array(x, dtype = np.float32)
  return np.mean(x), np.std(x)
