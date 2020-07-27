##############
## PPO Main ##
##############

import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from buffer_ppo import PPO_Buffer
from actor_critic import Actor_Critic
from torch.optim import Adam

config = dict(
    env = 'Walker2d-v2',
    steps_per_epoch = 4000,
    epochs = 50,
    gamma = 0.99,
    clip_ratio = 0.2,
    actor_lr = 3e-4,
    critic_lr = 1e-3,
    train_actor_iters = 80,
    train_critic_iters = 80,
    lam = 0.97,
    max_ep_len = 1000,
    target_kl = 0.01,
    seed = 0
)

np.random.seed(config['seed'])
env = gym.make(config['env'])
obs_space = env.observation_space
act_space = env.action_space
obs_size = obs_space.shape
act_size = act_space.shape
ac = Actor_Critic(obs_space, act_space)

local_steps_per_epoch = config['steps_per_epoch']
buf = PPO_Buffer(obs_size, act_size, local_steps_per_epoch, config['gamma'], 
                 config['lam'])

def compute_loss_actor(data):
  obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
  pi, logp = ac.actor(obs, act)
  ratio = torch.exp(logp - logp_old)
  clip_adv = torch.clamp(ratio, 1 - config['clip_ratio'], 
                         1 + config['clip_ratio']) * adv
  loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

  approx_kl = (logp_old - logp).mean().item()

  return loss_actor, approx_kl

def compute_loss_critic(data):
  obs, ret = data['obs'], data['ret']
  return ((ac.critic(obs) - ret)**2).mean()

actor_optimizer = Adam(ac.actor.parameters(), lr = config['actor_lr'])
critic_optimizer = Adam(ac.critic.parameters(), lr = config['critic_lr'])

def update():
  data = buf.get()
  pi_l_old, approx_kl = compute_loss_actor(data)
  pi_l_old = pi_l_old.item()
  v_l_old = compute_loss_critic(data).item()

  for i in range(config['train_actor_iters']):
    actor_optimizer.zero_grad()
    loss_actor, approx_kl = compute_loss_actor(data)
    kl = np.mean(approx_kl)
    if kl > 1.5 * config['target_kl']:
      break
    loss_actor.backward()
    actor_optimizer.step()

  for i in range(config['train_critic_iters']):
    critic_optimizer.zero_grad()
    loss_critic = compute_loss_critic(data)
    loss_critic.backward()
    critic_optimizer.step()

o, ep_ret, ep_len = env.reset(), 0, 0
tot_rets = []
for epoch in range(config['epochs']):
  tot_ret = 0
  for t in range(local_steps_per_epoch):
    a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

    next_o, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    buf.store(o,a,r,v,logp)

    o = next_o

    timeout = ep_len == config['max_ep_len']
    terminal = d or timeout
    epoch_ended = t==local_steps_per_epoch-1

    if terminal or epoch_ended:
      if epoch_ended and not(terminal):
        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
      # if trajectory didn't reach terminal state, bootstrap value target
      if timeout or epoch_ended:
        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
      else:
        v = 0
      buf.finish_path(v)
      tot_ret += ep_ret
      o, ep_ret, ep_len = env.reset(), 0, 0
  tot_rets.append(tot_ret/local_steps_per_epoch)
  print('Epoch:', epoch, 'ret:', tot_rets[-1])
  update()

plt.plot(tot_rets)
plt.show()
