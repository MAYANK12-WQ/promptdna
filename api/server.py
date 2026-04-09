"""
PromptDNA FastAPI REST interface.
Allows triggering evolutionary optimization and querying results over HTTP.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from promptdna import PromptOptimizer, TestCase, SelectionStrategy
from promptdna.store import EvolutionStore

app = FastAPI(
    title="PromptDNA API",
    description="Evolutionary prompt optimization using genetic algorithms",
    version="0.1.0",
)

_optimizer = PromptOptimizer(db_path="promptdna_api.db", mock=not os.getenv("OPENAI_API_KEY"))
_store = EvolutionStore(db_path="promptdna_api.db")
_active_runs: dict[str, str] = {}  # run_id -> status


class TestCaseInput(BaseModel):
    input: str
    expected: str
    weight: float = 1.0


class OptimizeRequest(BaseModel):
    seed_prompt: str
    task_description: str
    test_cases: list[TestCaseInput] = Field(min_length=1)
    generations: int = Field(default=5, ge=1, le=20)
    population_size: int = Field(default=10, ge=4, le=50)
    mutation_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    selection_strategy: str = "tournament"


@app.get("/")
def root():
    return {"service": "PromptDNA", "version": "0.1.0", "status": "ready"}


@app.post("/optimize")
def optimize(req: OptimizeRequest, background_tasks: BackgroundTasks):
    """Start an evolutionary optimization run (runs synchronously for simplicity)."""
    test_cases = [TestCase(input=tc.input, expected=tc.expected, weight=tc.weight)
                  for tc in req.test_cases]

    try:
        strategy = SelectionStrategy(req.selection_strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid selection_strategy: {req.selection_strategy}")

    result = _optimizer.optimize(
        seed_prompt=req.seed_prompt,
        task_description=req.task_description,
        test_cases=test_cases,
        generations=req.generations,
        population_size=req.population_size,
        mutation_rate=req.mutation_rate,
        crossover_rate=req.crossover_rate,
        selection_strategy=strategy,
    )

    return {
        "run_id": result.run_id,
        "best_prompt": result.best_prompt,
        "best_fitness": result.best_fitness,
        "generations_run": result.generations_run,
        "total_evaluated": result.total_individuals,
        "stats": result.stats(),
    }


@app.get("/runs")
def list_runs():
    return _store.list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    summary = _store.get_run_summary(run_id)
    if not summary:
        raise HTTPException(404, "Run not found")
    return summary


@app.get("/runs/{run_id}/stats")
def run_stats(run_id: str):
    return _store.run_stats(run_id)


@app.get("/runs/{run_id}/top")
def top_prompts(run_id: str, n: int = 5):
    return _store.top_individuals(run_id, n)


@app.get("/runs/{run_id}/mutations")
def mutation_leaderboard(run_id: str):
    return _store.mutation_leaderboard(run_id)


@app.get("/runs/{run_id}/curve")
def fitness_curve(run_id: str):
    return _store.fitness_over_generations(run_id)


@app.get("/individuals/{individual_id}/lineage")
def lineage(individual_id: str):
    return _store.lineage_tree(individual_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True)
