import gymnasium as gym
import numpy as np
import pickle


# =========================================================
# Q-Learning Agent（負責 q-table、policy、update）
class QLearningAgent:
    def __init__(self, n_states, n_actions, min_eps, eps_decay, lr=0.9, gamma=0.9):
        self.q = np.zeros((n_states, n_actions))
        self.epsilon = 1.0
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.gamma = gamma

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q.shape[1])
        return np.argmax(self.q[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q[next_state])
        td = reward + self.gamma * best_next - self.q[state, action]
        self.q[state, action] += self.lr * td

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_decay, self.min_eps)

# Trainer（負責訓練 episodes 與測試 episodes）
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

    def train(self, agent, episodes):
        env = self.make_env(render=False)
        for ep in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state

            agent.decay_epsilon()

        env.close()

    def evaluate(self, agent, episodes):
        env = self.make_env(render=False)
        success = 0

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False

            while not (terminated or truncated):
                action = np.argmax(agent.q[state])   # greedy
                state, reward, terminated, truncated, _ = env.step(action)

            if reward == 1:
                success += 1

        env.close()
        return success / episodes

# Grid Search: search for min_eps + eps_decay
class GridSearch:
    def __init__(self, trainer, train_eps=15000, eval_eps=1000):
        self.trainer = trainer
        self.train_eps = train_eps
        self.eval_eps = eval_eps

    def search(self, min_eps_list, decay_list):
        best_rate = -1
        best_params = None

        for me in min_eps_list:
            for de in decay_list:
                # 建立 Agent
                env_tmp = self.trainer.make_env()
                n_states = env_tmp.observation_space.n
                n_actions = env_tmp.action_space.n
                env_tmp.close()

                agent = QLearningAgent(
                    n_states=n_states,
                    n_actions=n_actions,
                    min_eps=me,
                    eps_decay=de
                )

                print(f"\n🔎 Testing min_eps={me}, decay={de}")

                # 訓練
                self.trainer.train(agent, self.train_eps)

                # 評估
                rate = self.trainer.evaluate(agent, self.eval_eps)
                print(f"→ Success Rate: {rate:.3f}")

                if rate > best_rate:
                    best_rate = rate
                    best_params = (me, de)

        print("\nBEST RESULT: ")
        print(f"min_eps   = {best_params[0]}")
        print(f"eps_decay = {best_params[1]}")
        print(f"success   = {best_rate:.3f}")

        return best_params, best_rate


# Main function
if __name__ == "__main__":
    trainer = Trainer()

    grid = GridSearch(trainer)

    min_eps_candidates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    decay_candidates = [i/1e5 for i in range(1,51)]

    grid.search(min_eps_candidates, decay_candidates)