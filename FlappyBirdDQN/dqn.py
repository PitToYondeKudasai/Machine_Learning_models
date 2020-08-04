#################
## DQN Network ##
#################

import torch
import torch.nn as nn

## ------ Q network ------- ##
def mlp(sizes, activation, output_activation = nn.Identity):
  layers = []
  for j in range(len(sizes) - 1):
    act = activation if j < len(sizes) - 2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

class Q(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size = (64,64), activation = nn.Tanh):
        super().__init__()
        sizes = [obs_size] + list(hidden_size) + [act_size]
        self.q_net = mlp(sizes, activation)
        
    def forward(self, obs):
        return self.q_net(obs)

    def forward_mask(self, obs, act):
      return torch.squeeze(torch.gather(self.q_net(obs),1,act.long()), -1)
    

## ------ DQN ------ ##
class DQN(nn.Module):
    
    def __init__(self, obs_size, act_size, hidden_size = (64, 64), activation = nn.Tanh):
        super().__init__()
        self.q_target = Q(obs_size, act_size, hidden_size, activation)
        self.q_online = Q(obs_size, act_size, hidden_size, activation)

    def copy(self):
        self.q_target.load_state_dict(self.q_online.state_dict())
        
    def soft_copy(self):
        pass
