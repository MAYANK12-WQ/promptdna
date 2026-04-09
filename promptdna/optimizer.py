"""
PromptDNA — High-level optimizer API.
This is the main entry point for users.
"""

from __future__ import annotations
import logging
from typing import Optional, Callable

from .models import (
    OptimizationRun, TestCase, Individual,
    SelectionStrategy, Generation,
)
from .evaluator import FitnessEvaluator
from .evolution import GeneticOptimizer
from .store import EvolutionStore

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Optimize a prompt using genetic algorithms.

    Usage:
        optimizer = PromptOptimizer(api_key="sk-...")
        result = optimizer.optimize(
            seed_prompt="Summarize the following text:",
            task_description="Summarize text accurately and concisely",
            test_cases=[
                TestCase(input="Long article...", expected="Short summary..."),
            ],
            generations=5,
            population_size=10,
        )
        print(result.best_prompt)
        print(result.best_fitness)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        db_path: str = "promptdna.db",
        mock: bool = False,
    ):
        self.store = EvolutionStore(db_path=db_path)
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._mock = mock

    def optimize(
        self,
        seed_prompt: str,
        task_description: str,
        test_cases: list[TestCase],
        generations: int = 5,
        population_size: int = 10,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.3,
        selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
        on_generation: Optional[Callable[[Generation], None]] = None,
        patience: int = 3,
    ) -> "OptimizationResult":
        """
        Run evolutionary prompt optimization.

        Args:
            seed_prompt: Starting prompt to evolve from.
            task_description: What the prompt should accomplish.
            test_cases: List of (input, expected_output) pairs for fitness scoring.
            generations: Max number of evolutionary generations.
            population_size: Number of individuals per generation.
            mutation_rate: Probability of mutation (0.0-1.0).
            crossover_rate: Probability of crossover vs pure mutation.
            selection_strategy: TOURNAMENT | ROULETTE | ELITISM.
            on_generation: Optional callback after each generation.
            patience: Stop early if no improvement for N generations.

        Returns:
            OptimizationResult with best prompt, fitness, lineage, and analytics.
        """
        run = OptimizationRun(
            task_description=task_description,
            seed_prompt=seed_prompt,
            test_cases=test_cases,
            population_size=population_size,
            max_generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            selection_strategy=selection_strategy,
        )

        evaluator = FitnessEvaluator(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            task_description=task_description,
            mock=self._mock,
        )

        engine = GeneticOptimizer(
            run=run,
            evaluator=evaluator,
            store=self.store,
            on_generation=on_generation,
            patience=patience,
        )

        completed_run = engine.run_evolution()
        return OptimizationResult(run=completed_run, store=self.store)

    def load_result(self, run_id: str) -> "OptimizationResult":
        """Load a previous optimization result from the database."""
        summary = self.store.get_run_summary(run_id)
        if not summary:
            raise ValueError(f"Run {run_id} not found")
        # Return a lightweight result wrapper
        run = OptimizationRun(
            id=summary["id"],
            task_description=summary["task_description"],
            seed_prompt=summary["seed_prompt"],
            test_cases=[],
            status=summary["status"],
        )
        return OptimizationResult(run=run, store=self.store)

    def list_runs(self) -> list[dict]:
        """List all optimization runs with their status and best fitness."""
        return self.store.list_runs()


class OptimizationResult:
    """
    Result of a completed optimization run.
    Provides access to the best prompt, analytics, and lineage.
    """

    def __init__(self, run: OptimizationRun, store: EvolutionStore):
        self._run = run
        self._store = store

    @property
    def best_prompt(self) -> str:
        return self._run.best_individual.genome if self._run.best_individual else self._run.seed_prompt

    @property
    def best_fitness(self) -> float:
        return self._run.best_fitness

    @property
    def best_individual(self) -> Optional[Individual]:
        return self._run.best_individual

    @property
    def run_id(self) -> str:
        return self._run.id

    @property
    def generations_run(self) -> int:
        return len(self._run.generations)

    @property
    def total_individuals(self) -> int:
        return sum(len(g.individuals) for g in self._run.generations)

    def fitness_curve(self) -> list[dict]:
        """Best and average fitness per generation."""
        return self._store.fitness_over_generations(self._run.id)

    def mutation_leaderboard(self) -> list[dict]:
        """Which mutation types produced the best results."""
        return self._store.mutation_leaderboard(self._run.id)

    def top_prompts(self, n: int = 5) -> list[dict]:
        """Top-N prompts by fitness."""
        return self._store.top_individuals(self._run.id, n)

    def lineage(self) -> list[dict]:
        """Full ancestry tree of the best individual."""
        if not self._run.best_individual:
            return []
        return self._store.lineage_tree(self._run.best_individual.id)

    def stats(self) -> dict:
        return self._store.run_stats(self._run.id)

    def summary(self) -> str:
        """Human-readable result summary."""
        lines = [
            f"PromptDNA Optimization Complete",
            f"{'─' * 50}",
            f"Run ID:          {self.run_id[:8]}...",
            f"Generations:     {self.generations_run}",
            f"Total evaluated: {self.total_individuals}",
            f"Best fitness:    {self.best_fitness:.4f} ({self.best_fitness:.1%})",
            f"",
            f"Best Prompt:",
            f"{'─' * 50}",
            self.best_prompt,
            f"",
            f"Mutation Leaderboard:",
            f"{'─' * 50}",
        ]
        for row in self.mutation_leaderboard()[:5]:
            lines.append(
                f"  {row['mutation_type']:20s}  avg={row['avg_fitness']:.4f}  best={row['best_fitness']:.4f}  n={row['count']}"
            )
        return "\n".join(lines)
