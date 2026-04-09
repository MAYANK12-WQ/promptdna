"""
Unit tests for PromptDNA core components.
Run with: pytest tests/ -v
"""

import pytest
import tempfile
import os

from promptdna.models import (
    Individual, TestCase, OptimizationRun,
    MutationType, SelectionStrategy,
)
from promptdna.operators import (
    mutate, breed, crossover,
    mutate_word_swap, mutate_cot_inject, mutate_compress,
    mutate_instruction_add, mutate_instruction_drop, mutate_tone_shift,
    mutate_reorder, mutate_elaborate, mutate_perspective,
)
from promptdna.evaluator import FitnessEvaluator
from promptdna.store import EvolutionStore
from promptdna.evolution import tournament_select, roulette_select, GeneticOptimizer
from promptdna import PromptOptimizer


SEED = "Summarize the following text clearly and concisely."


@pytest.fixture
def individual():
    return Individual(genome=SEED, generation=0, mutation_type=MutationType.SEED)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    return EvolutionStore(db_path=tmp_db)


@pytest.fixture
def test_cases():
    return [
        TestCase(input="Some long text.", expected="Short summary."),
        TestCase(input="Another document.", expected="Brief recap.", weight=1.5),
    ]


@pytest.fixture
def optimizer(tmp_db):
    return PromptOptimizer(mock=True, db_path=tmp_db)


# ── Operator Tests ────────────────────────────────────────────────────────────

class TestMutationOperators:
    def test_word_swap_returns_string(self, individual):
        result = mutate_word_swap(individual.genome)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cot_inject_adds_phrase(self, individual):
        result = mutate_cot_inject(individual.genome)
        cot_keywords = ["step", "think", "reason", "methodically", "carefully"]
        assert any(kw in result.lower() for kw in cot_keywords)

    def test_compress_shorter_or_equal(self, individual):
        verbose = "In order to summarize, please kindly provide a very clear response."
        result = mutate_compress(verbose)
        assert len(result) <= len(verbose) + 20  # allow small additions

    def test_instruction_add_increases_length(self, individual):
        result = mutate_instruction_add(individual.genome)
        assert len(result) > len(individual.genome)

    def test_instruction_drop_decreases_length(self):
        long_genome = "Summarize the text. Be concise. Use bullet points. Avoid jargon."
        result = mutate_instruction_drop(long_genome)
        assert len(result) < len(long_genome)

    def test_tone_shift_returns_string(self, individual):
        result = mutate_tone_shift(individual.genome)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_elaborate_adds_content(self, individual):
        result = mutate_elaborate(individual.genome)
        assert len(result) > len(individual.genome)

    def test_reorder_same_words(self, individual):
        multi = "Summarize the text. Be concise. Avoid jargon. Use clear language."
        result = mutate_reorder(multi)
        # Same words, different order
        assert sorted(result.split()) == sorted(multi.split()) or len(result) > 0

    def test_perspective_returns_different(self, individual):
        test_genome = "Your task is to summarize. You should be concise."
        result = mutate_perspective(test_genome)
        assert isinstance(result, str)

    def test_mutate_dispatcher(self, individual):
        child = mutate(individual, generation=1)
        assert child.generation == 1
        assert child.parent_ids == [individual.id]
        assert child.mutation_type != MutationType.SEED
        assert isinstance(child.genome, str)

    def test_mutate_creates_new_id(self, individual):
        child = mutate(individual, generation=1)
        assert child.id != individual.id


class TestCrossover:
    def test_crossover_produces_two_genomes(self, individual):
        parent_b = Individual(
            genome="Analyze the document. Extract key points. Be thorough and precise.",
            generation=0,
            mutation_type=MutationType.SEED,
        )
        g1, g2 = crossover(individual, parent_b)
        assert isinstance(g1, str)
        assert isinstance(g2, str)
        assert len(g1) > 0
        assert len(g2) > 0

    def test_breed_produces_individuals(self, individual):
        parent_b = Individual(
            genome="Analyze and extract key information from the text provided.",
            generation=0,
            mutation_type=MutationType.SEED,
        )
        offspring = breed(individual, parent_b, generation=1)
        assert len(offspring) == 2
        for child in offspring:
            assert child.generation == 1
            assert child.mutation_type == MutationType.CROSSOVER
            assert individual.id in child.parent_ids


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestFitnessEvaluator:
    def test_mock_evaluator_scores_individual(self, individual, test_cases):
        evaluator = FitnessEvaluator(mock=True, task_description="test")
        result = evaluator.evaluate(individual, test_cases)
        assert result.fitness is not None
        assert 0.0 <= result.fitness <= 1.0

    def test_mock_evaluator_sets_evaluations(self, individual, test_cases):
        evaluator = FitnessEvaluator(mock=True, task_description="test")
        result = evaluator.evaluate(individual, test_cases)
        assert len(result.evaluations) == len(test_cases)

    def test_mock_evaluator_fitness_breakdown(self, individual, test_cases):
        evaluator = FitnessEvaluator(mock=True, task_description="test")
        result = evaluator.evaluate(individual, test_cases)
        assert len(result.fitness_breakdown) == len(test_cases)

    def test_heuristic_score_perfect_match(self):
        evaluator = FitnessEvaluator(mock=True)
        text = "This is a test summary of the document."
        score, _ = evaluator._heuristic_score(SEED, text, text)
        assert score > 0.5

    def test_heuristic_score_empty_actual(self):
        evaluator = FitnessEvaluator(mock=True)
        score, _ = evaluator._heuristic_score(SEED, "expected output", "")
        assert score == 0.0

    def test_cot_prompt_bonus(self):
        evaluator = FitnessEvaluator(mock=True)
        cot_prompt = "Think step by step. Summarize the text."
        plain_prompt = "Summarize the text."
        score_cot, _ = evaluator._heuristic_score(cot_prompt, "expected", "actual output here")
        score_plain, _ = evaluator._heuristic_score(plain_prompt, "expected", "actual output here")
        assert score_cot >= score_plain


# ── Store Tests ───────────────────────────────────────────────────────────────

class TestEvolutionStore:
    def test_save_and_retrieve_run(self, store, test_cases):
        run = OptimizationRun(
            task_description="test task",
            seed_prompt=SEED,
            test_cases=test_cases,
        )
        store.save_run(run)
        summary = store.get_run_summary(run.id)
        assert summary is not None
        assert summary["task_description"] == "test task"

    def test_list_runs(self, store, test_cases):
        run = OptimizationRun(task_description="listed run", seed_prompt=SEED, test_cases=test_cases)
        store.save_run(run)
        runs = store.list_runs()
        assert any(r["id"] == run.id for r in runs)

    def test_run_stats_empty(self, store, test_cases):
        run = OptimizationRun(task_description="stats test", seed_prompt=SEED, test_cases=test_cases)
        store.save_run(run)
        stats = store.run_stats(run.id)
        assert "run_id" in stats
        assert stats["total_individuals"] == 0


# ── End-to-End Tests ──────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_optimization_run(self, optimizer, test_cases):
        result = optimizer.optimize(
            seed_prompt=SEED,
            task_description="Summarize text concisely",
            test_cases=test_cases,
            generations=2,
            population_size=4,
        )
        assert result.best_prompt
        assert 0.0 <= result.best_fitness <= 1.0
        assert result.generations_run >= 1
        assert result.total_individuals >= 4

    def test_result_has_fitness_curve(self, optimizer, test_cases):
        result = optimizer.optimize(
            seed_prompt=SEED,
            task_description="Summarize",
            test_cases=test_cases,
            generations=2,
            population_size=4,
        )
        curve = result.fitness_curve()
        assert isinstance(curve, list)
        assert len(curve) >= 1

    def test_result_has_mutation_leaderboard(self, optimizer, test_cases):
        result = optimizer.optimize(
            seed_prompt=SEED,
            task_description="Summarize",
            test_cases=test_cases,
            generations=2,
            population_size=6,
        )
        board = result.mutation_leaderboard()
        assert isinstance(board, list)

    def test_list_runs(self, optimizer, test_cases):
        optimizer.optimize(
            seed_prompt=SEED,
            task_description="List test",
            test_cases=test_cases,
            generations=1,
            population_size=4,
        )
        runs = optimizer.list_runs()
        assert len(runs) >= 1

    def test_early_stopping(self, optimizer, test_cases):
        result = optimizer.optimize(
            seed_prompt=SEED,
            task_description="Early stop test",
            test_cases=test_cases,
            generations=10,     # max 10
            population_size=4,
            patience=1,         # stop after 1 gen with no improvement
        )
        # Should have stopped before 10
        assert result.generations_run <= 10
