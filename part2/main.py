from trainer import Trainer
from agent import PPOAgent

if __name__ == "__main__":
    trainer = Trainer()

    # Read environment info
    env_tmp = trainer.make_env()
    n_states = env_tmp.observation_space.n
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    # Create PPO agent
    agent = PPOAgent(
        n_states=n_states,
        n_actions=n_actions,
        gamma=0.99,
        lr=3e-4,
        clip_eps=0.2
    )

    print("Starting PPO training...")
    trainer.train(agent, episodes=3000)   # You can adjust 2000–5000

    print("Evaluating agent performance...")
    success_rate = trainer.evaluate(agent, episodes=1000)

    print(f"\nFinal PPO Success Rate: {success_rate:.3f}")