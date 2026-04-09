"""
Genetic algorithm engine for PromptDNA.
Orchestrates selection → reproduction → mutation → evaluation across generations.
"""

from __future__ import annotations
import logging
import random
from typing import Callable, Optional

from .models import (
    Individual, Generation, OptimizationRun,
    MutationType, SelectionStrategy, TestCase,
)
from .operators import mutate, breed
from .evaluator import FitnessEvaluator
from .store import EvolutionStore

logger = logging.getLogger(__name__)


# ── Selection strategies ───────────────────────────────────────────────────────

def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    """Pick k random individuals, return the fittest."""
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda i: i.fitness or 0.0)


def roulette_select(population: list[Individual]) -> Individual:
    """Fitness-proportional selection (roulette wheel)."""
    fitnesses = [max(i.fitness or 0.0, 1e-6) for i in population]
    total = sum(fitnesses)
    probs = [f / total for f in fitnesses]
    return random.choices(population, weights=probs, k=1)[0]


def elitism_select(population: list[Individual], n: int) -> list[Individual]:
    """Return the top-N individuals unconditionally."""
    return sorted(population, key=lambda i: i.fitness or 0.0, reverse=True)[:n]


# ── Core genetic algorithm ─────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Runs the full evolutionary loop:
    Seed → Evaluate → Select → Breed/Mutate → Evaluate → Repeat

    Early stopping: if best fitness hasn't improved by min_delta
    for patience generations, stops early.
    """

    def __init__(
        self,
        run: OptimizationRun,
        evaluator: FitnessEvaluator,
        store: EvolutionStore,
        on_generation: Optional[Callable[[Generation], None]] = None,
        patience: int = 3,
        min_delta: float = 0.005,
        elite_ratio: float = 0.2,
    ):
        self.run = run
        self.evaluator = evaluator
        self.store = store
        self.on_generation = on_generation
        self.patience = patience
        self.min_delta = min_delta
        self.elite_ratio = elite_ratio
        self._no_improve_count = 0
        self._best_fitness = 0.0

    def _select(self, population: list[Individual]) -> Individual:
        if self.run.selection_strategy == SelectionStrategy.TOURNAMENT:
            return tournament_select(population)
        elif self.run.selection_strategy == SelectionStrategy.ROULETTE:
            return roulette_select(population)
        return tournament_select(population)

    def _seed_population(self) -> list[Individual]:
        """Create the initial population from the seed prompt."""
        seed = Individual(
            genome=self.run.seed_prompt,
            generation=0,
            mutation_type=MutationType.SEED,
            tokens=len(self.run.seed_prompt.split()),
        )
        population = [seed]

        # Immediately create diverse variants of the seed
        for _ in range(self.run.population_size - 1):
            population.append(mutate(seed, generation=0))

        return population

    def _next_generation(
        self,
        current_population: list[Individual],
        gen_number: int,
    ) -> list[Individual]:
        """Produce the next generation via selection + crossover + mutation."""
        evaluated = [i for i in current_population if i.is_evaluated]
        if not evaluated:
            return current_population

        n_elite = max(1, int(self.run.population_size * self.elite_ratio))
        elite = elitism_select(evaluated, n_elite)

        next_gen: list[Individual] = list(elite)  # elites survive unchanged

        while len(next_gen) < self.run.population_size:
            roll = random.random()

            if roll < self.run.crossover_rate and len(evaluated) >= 2:
                # Crossover
                parent_a = self._select(evaluated)
                parent_b = self._select([i for i in evaluated if i.id != parent_a.id] or evaluated)
                offspring = breed(parent_a, parent_b, gen_number)
                next_gen.extend(offspring)

            elif roll < self.run.crossover_rate + self.run.mutation_rate:
                # Mutation
                parent = self._select(evaluated)
                child = mutate(parent, gen_number)
                next_gen.append(child)

            else:
                # Clone + mutate
                parent = self._select(evaluated)
                next_gen.append(mutate(parent, gen_number))

        # Trim to population size
        return next_gen[: self.run.population_size]

    def _check_early_stop(self, generation: Generation) -> bool:
        """Return True if we should stop early."""
        best = generation.best
        if best and best.fitness:
            if best.fitness - self._best_fitness > self.min_delta:
                self._best_fitness = best.fitness
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            if self._no_improve_count >= self.patience:
                logger.info(f"Early stop: no improvement for {self.patience} generations")
                return True

            if best.fitness >= 0.99:
                logger.info("Early stop: perfect fitness achieved")
                return True

        return False

    def run_evolution(self) -> OptimizationRun:
        """Execute the full evolutionary optimization loop."""
        logger.info(
            f"Starting evolution: pop={self.run.population_size}, "
            f"max_gen={self.run.max_generations}, "
            f"test_cases={len(self.run.test_cases)}"
        )

        self.run.status = "running"
        self.store.save_run(self.run)

        population = self._seed_population()

        for gen_num in range(self.run.max_generations):
            logger.info(f"\n{'='*50}")
            logger.info(f"Generation {gen_num + 1}/{self.run.max_generations}")
            logger.info(f"{'='*50}")

            # Evaluate all unevaluated individuals
            self.evaluator.evaluate_batch(population, self.run.test_cases)

            # Package into a Generation
            generation = Generation(
                run_id=self.run.id,
                number=gen_num,
                individuals=population,
            )
            self.run.generations.append(generation)
            self.store.save_generation(generation)

            best = generation.best
            best_str = f"{best.fitness:.4f}" if best else "N/A"
            logger.info(
                f"Gen {gen_num}: best={best_str} "
                f"avg={generation.avg_fitness:.4f} "
                f"diversity={generation.diversity:.2f}"
            )

            if self.on_generation:
                self.on_generation(generation)

            if self._check_early_stop(generation):
                break

            # Breed next generation (not after last gen)
            if gen_num < self.run.max_generations - 1:
                population = self._next_generation(population, gen_num + 1)

        # Find global best across all generations
        all_individuals = [
            ind
            for gen in self.run.generations
            for ind in gen.individuals
            if ind.is_evaluated
        ]
        if all_individuals:
            self.run.best_individual = max(all_individuals, key=lambda i: i.fitness or 0.0)

        self.run.status = "complete"
        self.store.save_run(self.run)

        logger.info(f"\nEvolution complete. Best fitness: {self.run.best_fitness:.4f}")
        logger.info(f"Best prompt:\n{self.run.best_individual.genome if self.run.best_individual else 'N/A'}")

        return self.run
