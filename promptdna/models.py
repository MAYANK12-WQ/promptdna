"""
Core data models for PromptDNA.
Prompts are treated as living organisms — each has a genome (text),
lineage (parent_id), and fitness score that determines survival.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class MutationType(str, Enum):
    SEED            = "seed"             # original prompt
    WORD_SWAP       = "word_swap"        # replace key words with alternatives
    INSTRUCTION_ADD = "instruction_add"  # inject a new instruction
    INSTRUCTION_DROP= "instruction_drop" # remove an instruction
    COT_INJECT      = "cot_inject"       # add chain-of-thought trigger
    TONE_SHIFT      = "tone_shift"       # formality adjustment
    COMPRESS        = "compress"         # make more concise
    ELABORATE       = "elaborate"        # add more detail/context
    REORDER         = "reorder"          # shuffle sentence order
    PERSPECTIVE     = "perspective"      # change POV framing
    CROSSOVER       = "crossover"        # bred from two parents


class SelectionStrategy(str, Enum):
    TOURNAMENT  = "tournament"   # pick best from random subset
    ROULETTE    = "roulette"     # probability proportional to fitness
    ELITISM     = "elitism"      # top-N always survive


@dataclass
class TestCase:
    """A single input/expected-output pair for fitness evaluation."""
    input: str
    expected: str
    weight: float = 1.0            # some test cases matter more than others
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class EvaluationResult:
    """Result of evaluating one Individual against one TestCase."""
    individual_id: str
    test_case_id: str
    actual_output: str
    score: float                   # 0.0 — 1.0
    judge_reasoning: str = ""
    latency_ms: float = 0.0


@dataclass
class Individual:
    """
    A single prompt variant — one organism in the population.
    Tracks its lineage, mutation history, and fitness score.
    """
    genome: str                    # the actual prompt text
    generation: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: list[str] = field(default_factory=list)
    mutation_type: MutationType = MutationType.SEED
    fitness: Optional[float] = None       # None until evaluated
    fitness_breakdown: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    tokens: int = 0                # approximate token count
    evaluations: list[EvaluationResult] = field(default_factory=list)

    @property
    def is_evaluated(self) -> bool:
        return self.fitness is not None

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @property
    def genome_preview(self) -> str:
        return self.genome[:80] + "..." if len(self.genome) > 80 else self.genome

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "genome": self.genome,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_type": self.mutation_type.value,
            "fitness": self.fitness,
            "fitness_breakdown": self.fitness_breakdown,
            "created_at": self.created_at.isoformat(),
            "tokens": self.tokens,
        }


@dataclass
class Generation:
    """A snapshot of one evolutionary generation."""
    run_id: str
    number: int
    individuals: list[Individual] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def best(self) -> Optional[Individual]:
        evaluated = [i for i in self.individuals if i.is_evaluated]
        return max(evaluated, key=lambda i: i.fitness) if evaluated else None

    @property
    def avg_fitness(self) -> float:
        evaluated = [i for i in self.individuals if i.is_evaluated]
        return sum(i.fitness for i in evaluated) / len(evaluated) if evaluated else 0.0

    @property
    def diversity(self) -> float:
        """Rough diversity score — ratio of unique mutation types."""
        if not self.individuals:
            return 0.0
        types = {i.mutation_type for i in self.individuals}
        return len(types) / len(MutationType)


@dataclass
class OptimizationRun:
    """A full evolutionary optimization run."""
    task_description: str
    seed_prompt: str
    test_cases: list[TestCase]
    population_size: int = 10
    max_generations: int = 5
    mutation_rate: float = 0.8
    crossover_rate: float = 0.3
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    best_individual: Optional[Individual] = None
    generations: list[Generation] = field(default_factory=list)
    status: str = "pending"        # pending | running | complete | failed

    @property
    def best_fitness(self) -> float:
        if self.best_individual and self.best_individual.fitness:
            return self.best_individual.fitness
        return 0.0
