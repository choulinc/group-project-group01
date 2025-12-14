import numpy as np

class QLearningAgent:
    """
    Q-Learning Agent capable of learning in a discrete environment.
    Encapsulates the Q-table and the learning logic.
    """
    def __init__(self, n_states: int, n_actions: int, min_eps: float, eps_decay: float, lr: float = 0.2, gamma: float = 0.99):
        """
        Initialize the Q-Learning Agent.

        Args:
            n_states (int): Number of states in the environment.
            n_actions (int): Number of possible actions.
            min_eps (float): Minimum exploration rate.
            eps_decay (float): Decay rate for epsilon per episode.
            lr (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.gamma = gamma
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Exploration parameters
        self.epsilon = 1.0

    def select_action(self, state: int) -> int:
        """
        Select an action using the current policy (Epsilon-Greedy).
        
        Args:
            state (int): The current state.
            
        Returns:
            int: The selected action.
        """
        # Explore: select a random action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Exploit: select the action with max Q-value
        # If multiple actions have the same max Q-value, this will pick the first one.
        # Could be randomized among ties, but np.argmax is standard for simple implementations.
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update the Q-table based on the transition.
        
        Args:
            state (int): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Current state after action.
        """
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) after each episode.
        Ensure it does not go below min_eps.
        """
        self.epsilon = max(self.epsilon - self.eps_decay, self.min_eps)

    def get_q_table(self) -> np.ndarray:
        """
        Return the Q-table.
        
        Returns:
            np.ndarray: The Q-table.
        """
        return self.q_table
