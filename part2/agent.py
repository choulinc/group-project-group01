# agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Policy Network for PPO (discrete action space)
class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.model(x)


# Value Network (baseline)
class ValueNetwork(nn.Module):
    def __init__(self, n_states, hidden=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.model(x)


# PPO Agent (replaces QLearningAgent)
class PPOAgent:
    def __init__(self, n_states, n_actions, gamma=0.99, lr=3e-4, clip_eps=0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip_eps = clip_eps

        self.policy = PolicyNetwork(n_states, n_actions)
        self.value = ValueNetwork(n_states)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)

        self.memory = []  # list of (state, action, reward, next_state, done, log_prob)

    def get_action(self, state):
        """Sample an action from the policy distribution."""
        state_t = torch.FloatTensor(state).unsqueeze(0)

        logits = self.policy(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach()

    def store(self, transition):
        """Store a transition: (s, a, r, s', done, log_prob)."""
        self.memory.append(transition)

    def compute_returns(self):
        """Compute discounted returns for PPO advantage."""
        returns = []
        G = 0
        for _, _, r, _, done, _ in reversed(self.memory):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def ppo_update(self, epochs=4):
        """Update policy using PPO clipped objective."""
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.stack(log_probs_old)

        # Compute advantages
        returns = self.compute_returns()
        values = self.value(states).squeeze()
        advantages = returns - values.detach()

        for _ in range(epochs):
            logits = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)

            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - log_probs_old)

            # PPO clipped objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)

            # Update policy
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # Update value
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        self.memory = []