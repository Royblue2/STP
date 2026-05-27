import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import STP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

	def act(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.l3(a)
		return self.max_action * torch.tanh(a), a


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class policy(object):
	def __init__(self, state_dim, action_dim, max_action):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.actor_target = copy.deepcopy(self.actor)
		self.critic_target = copy.deepcopy(self.critic)

		self.actor_old = copy.deepcopy(self.actor)
		self.actor_temp = copy.deepcopy(self.actor)

		self.max_action = max_action
		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def select_action_old(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor_old(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			noise = (torch.randn_like(action) * 0.2*self.max_action).clamp(-0.5*self.max_action, 0.5*self.max_action)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + 0.99 * not_done * target_Q

		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q1, target_Q).mean() + F.mse_loss(current_Q2, target_Q).mean()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % 2 == 0:
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)


			
