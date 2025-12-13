import gymnasium as gym
import numpy as np
from agent import QLearningAgent

class Trainer:
    """
    Trainer class responsible for managing the environment, training the agent,
    and evaluating its performance.
    """
    def __init__(self, env_name: str = "FrozenLake-v1", map_name: str = "8x8", is_slippery: bool = True):
        """
        Initialize the Trainer.

        Args:
            env_name (str): The Gym environment ID.
            map_name (str): Map size (e.g., "8x8").
            is_slippery (bool): Whether the frozen lake is slippery.
        """
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery

    def _make_env(self, render_mode: str = None):
        """
        Helper to create the environment.
        """
        return gym.make(
            self.env_name,
            map_name=self.map_name,
            is_slippery=self.is_slippery,
            render_mode=render_mode
        )

    def train(self, agent: QLearningAgent, episodes: int) -> list:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            agent (QLearningAgent): The agent to train.
            episodes (int): Number of training episodes.
            
        Returns:
            list: List of rewards per episode (optional for analysis).
        """
        env = self._make_env()
        rewards_history = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            terminated = truncated = False
            total_reward = 0
            
            while not (terminated or truncated):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                agent.update(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
            
            agent.decay_epsilon()
            rewards_history.append(total_reward)
            
        env.close()
        return rewards_history

    def evaluate(self, agent: QLearningAgent, episodes: int) -> float:
        """
        Evaluate the agent's performance (Greedy policy).
        
        Args:
            agent (QLearningAgent): The agent to evaluate.
            episodes (int): Number of evaluation episodes.
            
        Returns:
            float: Success rate (0.0 to 1.0).
        """
        env = self._make_env(render_mode=None)
        success_count = 0
        
        for _ in range(episodes):
            state, _ = env.reset()
            terminated = truncated = False
            
            while not (terminated or truncated):
                # Greedy action selection for evaluation
                action = np.argmax(agent.get_q_table()[state])
                state, reward, terminated, truncated, _ = env.step(action)
                
            if reward == 1.0:
                success_count += 1
                
        env.close()
        return success_count / episodes

    def visualize(self, agent: QLearningAgent):
        """
        Run one episode with rendering to demonstrate the agent.
        """
        env = self._make_env(render_mode="human")
        state, _ = env.reset()
        terminated = truncated = False
        
        print("\nVisualizing Best Agent...")
        while not (terminated or truncated):
            action = np.argmax(agent.get_q_table()[state])
            state, _, terminated, truncated, _ = env.step(action)
        
        env.close()

