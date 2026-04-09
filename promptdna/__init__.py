"""
PromptDNA — Evolutionary Prompt Optimizer
Treats prompts as living organisms. Evolves them across generations using
genetic algorithms (mutation, crossover, selection) scored by LLM-as-judge fitness.
"""

__version__ = "0.1.0"
__author__ = "Mayank Shekhar"

from .optimizer import PromptOptimizer, OptimizationResult
from .models import TestCase, SelectionStrategy, MutationType
from .store import EvolutionStore

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "TestCase",
    "SelectionStrategy",
    "MutationType",
    "EvolutionStore",
]
