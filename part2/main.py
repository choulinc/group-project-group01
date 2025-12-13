from trainer import Trainer
from optimizer import GeneticOptimizer
import sys

def main():
    # 1. Setup Trainer
    # Requirement: Map size at least 8x8.
    trainer = Trainer(env_name="FrozenLake-v1", map_name="8x8", is_slippery=True)
    
    # 2. Setup Optimization Parameters
    # Requirement: Cannot increase num_episodes (assuming reference to 15000 in original file)
    train_episodes_per_run = 15000
    eval_episodes = 1000
    
    print("Optimization Strategy: Genetic Algorithm", flush=True)
    
    # GA Configuration
    # User said "time is not an issue", so we can use generous generations/population
    optimizer = GeneticOptimizer(
        trainer=trainer,
        train_eps=train_episodes_per_run,
        eval_eps=eval_episodes,
        population_size=12,  # Number of candidates per generation
        generations=100,      # Number of iterations
        mutation_rate=0.3    # Chance to mutate
    )
        
    print(f"Starting optimization using {optimizer.__class__.__name__}...", flush=True)
    
    # 3. Run Optimization
    best_min_eps, best_eps_decay, best_rate = optimizer.optimize()
    
    print("\n" + "="*40, flush=True)
    print(" FINAL RESULT ", flush=True)
    print("="*40, flush=True)
    print(f"Best Success Rate: {best_rate:.2%}", flush=True)
    print(f"Best min_eps:      {best_min_eps:.6f}", flush=True)
    print(f"Best eps_decay:    {best_eps_decay:.6f}", flush=True)
    
    if best_rate > 0.70:
        print("\nSUCCESS! The goal of > 70% success rate has been achieved.", flush=True)
    else:
        print("\nWARNING: Did not reach 70%. You may need to run more generations.", flush=True)

    # 4. Demonstrate the best agent
    print("\nDemonstrating best agent performance...", flush=True)
    # We need to recreate the agent and train it with the best parameters
    dummy_env = trainer._make_env()
    n_states = dummy_env.observation_space.n
    n_actions = dummy_env.action_space.n
    dummy_env.close()
    
    from agent import QLearningAgent
    best_agent = QLearningAgent(n_states, n_actions, min_eps=best_min_eps, eps_decay=best_eps_decay)
    
    print("Training best agent for demonstration...", flush=True)
    trainer.train(best_agent, train_episodes_per_run)
    
    print("Visualizing...", flush=True)
    trainer.visualize(best_agent)

if __name__ == '__main__':
    main()
