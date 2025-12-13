import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from agent import QLearningAgent
from trainer import Trainer

class Optimizer(ABC):
    """
    Abstract Base Class for Hyperparameter Optimization.
    """
    def __init__(self, trainer: Trainer, train_eps: int, eval_eps: int):
        self.trainer = trainer
        self.train_eps = train_eps
        self.eval_eps = eval_eps
    
    @abstractmethod
    def optimize(self) -> Tuple[float, float, float]:
        """
        Run the optimization process.
        
        Returns:
            Tuple[float, float, float]: (best_min_eps, best_eps_decay, best_success_rate)
        """
        pass

class GeneticOptimizer(Optimizer):
    """
    Advanced Optimizer using Genetic Algorithm (Evolutionary Strategy).
    
    Genes: [min_eps, eps_decay]
    """
    def __init__(self, trainer: Trainer, train_eps: int, eval_eps: int,
                 population_size: int = 10, generations: int = 5,
                 mutation_rate: float = 0.2):
        super().__init__(trainer, train_eps, eval_eps)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Ranges for initialization and mutation
        # min_eps: [0.0, 0.1] - usually we want low execution epsilon
        # eps_decay: [0.00001, 0.0005] - Force slower decay for 8x8 map
        self.min_eps_range = (0.0, 0.1)
        self.eps_decay_range = (0.00001, 0.0005)

    def _create_individual(self) -> Tuple[float, float]:
        min_eps = random.uniform(*self.min_eps_range)
        eps_decay = random.uniform(*self.eps_decay_range)
        return (min_eps, eps_decay)

    def _evaluate_individual(self, individual: Tuple[float, float]) -> float:
        min_eps, eps_decay = individual
        
        dummy_env = self.trainer._make_env()
        n_states = dummy_env.observation_space.n
        n_actions = dummy_env.action_space.n
        dummy_env.close()

        # Create agent with individual's parameters
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            min_eps=min_eps,
            eps_decay=eps_decay
        )
        
        # Train
        self.trainer.train(agent, self.train_eps)
        
        # Evaluate
        return self.trainer.evaluate(agent, self.eval_eps)

    def _crossover(self, parent1: Tuple[float, float], parent2: Tuple[float, float]) -> Tuple[float, float]:
        # Simple averaging crossover
        min_eps = (parent1[0] + parent2[0]) / 2.0
        eps_decay = (parent1[1] + parent2[1]) / 2.0
        return (min_eps, eps_decay)

    def _mutate(self, individual: Tuple[float, float]) -> Tuple[float, float]:
        min_eps, eps_decay = individual
        
        if random.random() < self.mutation_rate:
            min_eps += random.gauss(0, 0.01) # Small perturbation
            min_eps = np.clip(min_eps, *self.min_eps_range)
            
        if random.random() < self.mutation_rate:
            eps_decay += random.gauss(0, 0.0001)
            eps_decay = np.clip(eps_decay, *self.eps_decay_range)
            
        return (min_eps, eps_decay)

    def optimize(self) -> Tuple[float, float, float]:
        # 1. Initialize Population
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_overall_rate = -1.0
        best_overall_params = (0.0, 0.0)

        for gen in range(self.generations):
            print(f"\n=== Generation {gen+1}/{self.generations} ===", flush=True)
            
            # 2. Evaluate Fitness
            fitness_scores = []
            for i, ind in enumerate(population):
                score = self._evaluate_individual(ind)
                fitness_scores.append((score, ind))
                print(f"  Ind {i}: min_eps={ind[0]:.4f}, decay={ind[1]:.5f} -> Rate: {score:.3f}", flush=True)
                
                if score > best_overall_rate:
                    best_overall_rate = score
                    best_overall_params = ind

            # Sort by fitness (descending)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Check if we hit the target strongly? (Optional early stopping)
            # if best_overall_rate > 0.8: break 

            # 3. Selection (Top 50%)
            survivors = fitness_scores[:self.population_size // 2]
            survivor_inds = [x[1] for x in survivors]
            
            # 4. Reproduction (Crossover & Mutation)
            new_population = []
            
            # Elitism: Keep the very best
            new_population.append(survivors[0][1])
            
            while len(new_population) < self.population_size:
                p1 = random.choice(survivor_inds)
                p2 = random.choice(survivor_inds)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)
                
            population = new_population
            print(f"Gen {gen+1} Best Rate: {survivors[0][0]:.3f}", flush=True)

        print(f"\nGenetic Optimization Best: {best_overall_params} with Rate: {best_overall_rate}", flush=True)
        return (*best_overall_params, best_overall_rate)
