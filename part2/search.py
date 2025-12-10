from agent import QLearningAgent
from trainer import Trainer

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

                env_tmp = self.trainer.make_env()
                n_states = env_tmp.observation_space.n
                n_actions = env_tmp.action_space.n
                env_tmp.close()

                agent = QLearningAgent(n_states, n_actions, me, de)

                print(f"\n🔎 Testing min_eps={me}, decay={de}")
                self.trainer.train(agent, self.train_eps)

                rate = self.trainer.evaluate(agent, self.eval_eps)
                print(f"→ Success Rate: {rate:.3f}")

                if rate > best_rate:
                    best_rate = rate
                    best_params = (me, de)

        print("\nBEST RESULT:")
        print(f"min_eps   = {best_params[0]}")
        print(f"eps_decay = {best_params[1]}")
        print(f"success   = {best_rate:.3f}")

        return best_params, best_rate