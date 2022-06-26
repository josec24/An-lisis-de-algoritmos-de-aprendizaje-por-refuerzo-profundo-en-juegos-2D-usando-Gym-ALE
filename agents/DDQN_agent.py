import torch
import numpy as np
import collections
import torch.nn as nn
from data.helper import guardarPuntuacion
import torch.nn.functional as F

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

GAMMA = 0.99

class Agent:
    def __init__(self, env, net, target_net, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.net=net
        self.target_net=target_net
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def get_action(self, net, device, epsilon=0.0):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action

    def step(self,action):
        done_reward = None
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done or _['lives']==0:
            guardarPuntuacion(self.total_reward,'ddqn_Breakout.csv')
            done_reward = self.total_reward
            self._reset()
            return done_reward,True

        return done_reward,False

    def calc_loss(self,batch, device):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1).to(torch.int64)).squeeze(-1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v
        return F.mse_loss(state_action_values, expected_state_action_values)

    def update(self,optimizer,buffer,BATCH_SIZE,device):
        batch = buffer.sample(BATCH_SIZE)
        optimizer.zero_grad()
        loss=self.calc_loss(batch,device)
        loss.backward()
        optimizer.step()
