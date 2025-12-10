import gymnasium as gym
import numpy as np
import torch

class Trainer:
    def __init__(self, env_name="FrozenLake-v1"):
        self.env_name = env_name

    def make_env(self, render=False):
        return gym.make(
            self.env_name,
            map_name="8x8",
            is_slippery=True,
            render_mode="human" if render else None
        )

    def train(self, agent, episodes, rollout_len=200, update_epochs=4):
        """
        PPO training loop.
        Each 'episode' here means a full environment rollout (not PPO update steps).
        """
        env = self.make_env(render=False)

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False

            while not (terminated or truncated):
                # One-hot encode FrozenLake state (discrete 0~63)
                state_vec = np.eye(agent.n_states)[state]

                action, log_prob = agent.get_action(state_vec)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Store trajectory
                agent.store((state_vec, action, reward, next_state, terminated, log_prob))

                state = next_state

                # When memory reaches rollout length → PPO update
                if len(agent.memory) >= rollout_len:
                    agent.ppo_update(epochs=update_epochs)

            # End of episode — if memory still has trajectories, update once
            if len(agent.memory) > 0:
                agent.ppo_update(epochs=update_epochs)

        env.close()

    def evaluate(self, agent, episodes=500):
        """
        Evaluation uses deterministic (greedy) policy: argmax π(a|s)
        """
        env = self.make_env(render=False)
        success = 0

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False

            while not (terminated or truncated):
                state_vec = np.eye(agent.n_states)[state]
                state_t = torch.FloatTensor(state_vec).unsqueeze(0)

                logits = agent.policy(state_t)
                action = torch.argmax(logits).item()

                state, reward, terminated, truncated, _ = env.step(action)

            if reward == 1:
                success += 1

        env.close()
        return success / episodes