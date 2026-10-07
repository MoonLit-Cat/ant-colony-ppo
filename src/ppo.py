import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 130  
ACTION_DIM = 5


class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(STATE_DIM, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )

        self.critic = nn.Sequential(
            nn.Linear(STATE_DIM, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        with torch.no_grad():
            dist = Categorical(logits=self.actor(state))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state).squeeze(-1)
            return action, log_prob, value

    @torch.no_grad()
    def get_value(self, state):
        value = self.critic(state).squeeze(-1)
        return value

    def evaluate(self, state, actions):
        dist = Categorical(logits=self.actor(state))
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(state).squeeze(-1)
        return log_probs, entropy, value
    


class RolloutBuffer():
    """ Stores n_steps of experience for n_agents(ants)"""
    def __init__(self, n_steps, n_agents, state_dim, gamma=0.99, gae_lambda=0.95):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        shape = (n_steps, n_agents)
        self.states = torch.zeros((*shape, state_dim), device=device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.dones = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, state, action, log_prob, reward, done, value):
        assert self.ptr < self.n_steps, "buffer is full"
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    @torch.no_grad()
    def compute_gae(self, last_value):
        assert self.ptr == self.n_steps, "buffer not full"
        last_gae = torch.zeros(self.n_agents, device=device)
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, mini_batch_size, normalize_adv=True):
        n = self.n_agents * self.n_steps
        states = self.states.reshape(n, self.state_dim)
        actions = self.actions.reshape(n)
        log_probs = self.log_probs.reshape(n)
        values = self.values.reshape(n)
        returns = self.returns.reshape(n)
        adv = self.advantages.reshape(n)
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = torch.randperm(n, device=device)
        for start in range(0, n, mini_batch_size):
            mb = idx[start:start + mini_batch_size]
            yield states[mb], actions[mb], log_probs[mb], adv[mb], returns[mb], values[mb]



def ppo_update(net, buffer, optimizer,
               update_epochs=4, mini_batch_size=1024, clip_eps=0.2,
               critic_coef=0.5, entropy_coef=0.001, max_grad_norm=0.5):
    log = defaultdict(list)

    for _ in range(update_epochs):
        for states, actions, old_log_prob, adv, returns, _ in buffer.get_batches(mini_batch_size):
            new_log_prob, entropy, values = net.evaluate(states, actions)

            ratio = (new_log_prob - old_log_prob).exp()
            pg_unclipped = -adv * ratio
            pg_clipped = -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = torch.max(pg_unclipped, pg_clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            entropy_mean = entropy.mean()

            loss = policy_loss + critic_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            log["entropy"].append(entropy_mean.item())

    return {k: float(np.mean(v)) for k, v in log.items()}