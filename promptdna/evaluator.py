"""
Fitness evaluator for PromptDNA.
Uses an LLM-as-judge pattern to score prompt outputs against expected results.
Supports OpenAI-compatible APIs. Falls back to heuristic scoring if no API key.
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Optional

from .models import Individual, TestCase, EvaluationResult

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an impartial evaluator. You will be given:
1. A task description
2. A prompt that was used
3. The expected output
4. The actual output produced

Score the actual output from 0.0 to 1.0 based on how well it matches the expected output.
Consider: accuracy, completeness, format adherence, and relevance.

Respond ONLY with a JSON object in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}}"""


class FitnessEvaluator:
    """
    Evaluates prompt fitness by running each test case through an LLM
    and judging the output against the expected result.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        judge_model: Optional[str] = None,
        task_description: str = "",
        mock: bool = False,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.model = model
        self.judge_model = judge_model or model
        self.task_description = task_description
        self.mock = mock or not self.api_key

        if self.mock:
            logger.warning("No API key found — using heuristic mock evaluator")
        else:
            logger.info(f"Evaluator using model={self.model}, judge={self.judge_model}")

    def _call_llm(self, messages: list[dict], model: str) -> str:
        """Make an OpenAI-compatible API call."""
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"model": model, "messages": messages, "temperature": 0.3}
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _heuristic_score(self, genome: str, expected: str, actual: str) -> tuple[float, str]:
        """
        Rule-based scoring when no API is available.
        Scores based on keyword overlap, length ratio, and structural similarity.
        """
        if not actual:
            return 0.0, "Empty output"

        exp_words = set(expected.lower().split())
        act_words = set(actual.lower().split())

        if not exp_words:
            return 0.5, "No expected output to compare"

        # Keyword overlap score
        overlap = len(exp_words & act_words) / len(exp_words)

        # Length ratio score (penalize too short or too long)
        len_ratio = min(len(actual), len(expected)) / max(len(actual), len(expected), 1)
        len_score = min(len_ratio, 1.0)

        # Prompt quality heuristics (longer, more structured prompts tend to do better)
        has_cot = any(phrase in genome.lower() for phrase in ["step by step", "think", "reason"])
        has_role = any(role[:15] in genome.lower() for role in ["you are", "act as", "expert"])
        has_format = any(fmt in genome.lower() for fmt in ["bullet", "numbered", "list", "format"])

        prompt_bonus = (0.05 * has_cot + 0.05 * has_role + 0.03 * has_format)

        score = min(0.4 * overlap + 0.3 * len_score + 0.3 * (0.5 + prompt_bonus), 1.0)
        reasoning = f"overlap={overlap:.2f}, len_ratio={len_score:.2f}, prompt_bonus={prompt_bonus:.2f}"
        return round(score, 4), reasoning

    def _run_prompt(self, genome: str, input_text: str) -> tuple[str, float]:
        """Run a prompt+input through the LLM. Returns (output, latency_ms)."""
        if self.mock:
            # Simulate output quality based on prompt characteristics
            quality = 0.5
            quality += 0.1 if "step by step" in genome.lower() else 0
            quality += 0.1 if "you are" in genome.lower() or "expert" in genome.lower() else 0
            quality += 0.05 if len(genome) > 100 else 0
            # Simulate some variance
            import random
            quality += random.gauss(0, 0.05)
            quality = max(0.1, min(0.95, quality))
            fake_output = f"[Mock output for: {input_text[:40]}] Quality signal: {quality:.2f}"
            return fake_output, 50.0

        messages = [
            {"role": "system", "content": genome},
            {"role": "user", "content": input_text},
        ]
        t0 = time.time()
        output = self._call_llm(messages, self.model)
        latency = (time.time() - t0) * 1000
        return output, latency

    def _judge_output(self, genome: str, expected: str, actual: str) -> tuple[float, str]:
        """Use LLM-as-judge to score the output."""
        if self.mock or not actual:
            return self._heuristic_score(genome, expected, actual)

        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {self.task_description}\n\n"
                    f"Prompt used:\n{genome}\n\n"
                    f"Expected output:\n{expected}\n\n"
                    f"Actual output:\n{actual}"
                ),
            },
        ]
        raw = self._call_llm(messages, self.judge_model)
        try:
            parsed = json.loads(raw)
            return float(parsed["score"]), parsed.get("reasoning", "")
        except Exception:
            # Fallback to heuristic
            return self._heuristic_score(genome, expected, actual)

    def evaluate(self, individual: Individual, test_cases: list[TestCase]) -> Individual:
        """
        Evaluate an individual against all test cases.
        Sets individual.fitness and individual.evaluations in place.
        Returns the individual.
        """
        results: list[EvaluationResult] = []
        weighted_score = 0.0
        total_weight = sum(tc.weight for tc in test_cases)

        for tc in test_cases:
            actual, latency = self._run_prompt(individual.genome, tc.input)
            score, reasoning = self._judge_output(individual.genome, tc.expected, actual)

            result = EvaluationResult(
                individual_id=individual.id,
                test_case_id=tc.id,
                actual_output=actual,
                score=score,
                judge_reasoning=reasoning,
                latency_ms=latency,
            )
            results.append(result)
            weighted_score += score * tc.weight

        individual.fitness = round(weighted_score / total_weight, 4) if total_weight > 0 else 0.0
        individual.evaluations = results
        individual.fitness_breakdown = {
            tc.id: round(r.score, 4)
            for tc, r in zip(test_cases, results)
        }
        return individual

    def evaluate_batch(
        self, individuals: list[Individual], test_cases: list[TestCase]
    ) -> list[Individual]:
        """Evaluate a batch of individuals. Sequential for now; async in v0.2."""
        for ind in individuals:
            if not ind.is_evaluated:
                self.evaluate(ind, test_cases)
                logger.debug(f"  {ind.short_id} [{ind.mutation_type.value:16s}] fitness={ind.fitness:.4f}")
        return individuals
