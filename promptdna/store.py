"""
SQLite persistence layer for PromptDNA.
Stores full evolutionary lineage — every individual, its parents,
mutation type, fitness score, and generation number — queryable forever.
"""

from __future__ import annotations
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    Individual, Generation, OptimizationRun,
    MutationType, SelectionStrategy, TestCase, EvaluationResult,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,
    task_description    TEXT NOT NULL,
    seed_prompt         TEXT NOT NULL,
    test_cases          TEXT NOT NULL,   -- JSON
    population_size     INTEGER NOT NULL,
    max_generations     INTEGER NOT NULL,
    mutation_rate       REAL NOT NULL,
    crossover_rate      REAL NOT NULL,
    selection_strategy  TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    best_individual_id  TEXT,
    best_fitness        REAL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    number      INTEGER NOT NULL,
    avg_fitness REAL,
    best_fitness REAL,
    diversity   REAL,
    size        INTEGER,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS individuals (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    generation_id   TEXT NOT NULL REFERENCES generations(id),
    generation_num  INTEGER NOT NULL,
    genome          TEXT NOT NULL,
    mutation_type   TEXT NOT NULL,
    parent_ids      TEXT NOT NULL DEFAULT '[]',  -- JSON array
    fitness         REAL,
    fitness_breakdown TEXT NOT NULL DEFAULT '{}',
    tokens          INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id              TEXT PRIMARY KEY,
    individual_id   TEXT NOT NULL REFERENCES individuals(id),
    test_case_id    TEXT NOT NULL,
    actual_output   TEXT NOT NULL,
    score           REAL NOT NULL,
    judge_reasoning TEXT NOT NULL DEFAULT '',
    latency_ms      REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_individuals_run ON individuals(run_id);
CREATE INDEX IF NOT EXISTS idx_individuals_gen ON individuals(generation_num);
CREATE INDEX IF NOT EXISTS idx_individuals_fitness ON individuals(fitness DESC);
CREATE INDEX IF NOT EXISTS idx_individuals_mutation ON individuals(mutation_type);
CREATE INDEX IF NOT EXISTS idx_generations_run ON generations(run_id);
"""


class EvolutionStore:
    """SQLite store for full evolutionary lineage and analytics."""

    def __init__(self, db_path: str = "promptdna.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"PromptDNA store initialized at {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Runs ──────────────────────────────────────────────────────────────────

    def save_run(self, run: OptimizationRun) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (id, task_description, seed_prompt, test_cases, population_size,
                     max_generations, mutation_rate, crossover_rate, selection_strategy,
                     status, best_individual_id, best_fitness, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status             = excluded.status,
                    best_individual_id = excluded.best_individual_id,
                    best_fitness       = excluded.best_fitness
                """,
                (
                    run.id,
                    run.task_description,
                    run.seed_prompt,
                    json.dumps([vars(tc) for tc in run.test_cases]),
                    run.population_size,
                    run.max_generations,
                    run.mutation_rate,
                    run.crossover_rate,
                    run.selection_strategy.value,
                    run.status,
                    run.best_individual.id if run.best_individual else None,
                    run.best_fitness,
                    run.created_at.isoformat(),
                ),
            )

    def get_run_summary(self, run_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            return dict(row)

    def list_runs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, task_description, status, best_fitness, created_at FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Generations ───────────────────────────────────────────────────────────

    def save_generation(self, generation: Generation) -> None:
        best = generation.best
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO generations
                    (id, run_id, number, avg_fitness, best_fitness, diversity, size, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    generation.id,
                    generation.run_id,
                    generation.number,
                    generation.avg_fitness,
                    best.fitness if best else None,
                    generation.diversity,
                    len(generation.individuals),
                    generation.created_at.isoformat(),
                ),
            )
            # Save individuals
            for ind in generation.individuals:
                self._save_individual(conn, ind, generation.run_id, generation.id, generation.number)

    def _save_individual(
        self,
        conn,
        ind: Individual,
        run_id: str,
        generation_id: str,
        generation_num: int,
    ) -> None:
        import uuid
        conn.execute(
            """
            INSERT OR REPLACE INTO individuals
                (id, run_id, generation_id, generation_num, genome, mutation_type,
                 parent_ids, fitness, fitness_breakdown, tokens, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ind.id, run_id, generation_id, generation_num,
                ind.genome, ind.mutation_type.value,
                json.dumps(ind.parent_ids),
                ind.fitness,
                json.dumps(ind.fitness_breakdown),
                ind.tokens,
                ind.created_at.isoformat(),
            ),
        )
        for ev in ind.evaluations:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations
                    (id, individual_id, test_case_id, actual_output, score,
                     judge_reasoning, latency_ms, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()), ev.individual_id, ev.test_case_id,
                    ev.actual_output, ev.score, ev.judge_reasoning,
                    ev.latency_ms, datetime.utcnow().isoformat(),
                ),
            )

    # ── Analytics ─────────────────────────────────────────────────────────────

    def fitness_over_generations(self, run_id: str) -> list[dict]:
        """Best and average fitness per generation for a run."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT number, best_fitness, avg_fitness, diversity
                FROM generations WHERE run_id = ? ORDER BY number
                """,
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mutation_leaderboard(self, run_id: str) -> list[dict]:
        """Average fitness grouped by mutation type — shows which mutations work best."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT mutation_type,
                       COUNT(*) as count,
                       ROUND(AVG(fitness), 4) as avg_fitness,
                       ROUND(MAX(fitness), 4) as best_fitness
                FROM individuals
                WHERE run_id = ? AND fitness IS NOT NULL
                GROUP BY mutation_type
                ORDER BY avg_fitness DESC
                """,
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def lineage_tree(self, individual_id: str) -> list[dict]:
        """Walk up the ancestry tree for a given individual."""
        result = []
        visited = set()
        queue = [individual_id]

        with self._conn() as conn:
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)
                row = conn.execute(
                    "SELECT id, genome, mutation_type, fitness, parent_ids, generation_num FROM individuals WHERE id = ?",
                    (current_id,),
                ).fetchone()
                if row:
                    d = dict(row)
                    d["parent_ids"] = json.loads(d["parent_ids"])
                    result.append(d)
                    queue.extend(d["parent_ids"])

        return result

    def top_individuals(self, run_id: str, n: int = 10) -> list[dict]:
        """Return top-N individuals by fitness for a run."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, genome, mutation_type, fitness, generation_num, tokens
                FROM individuals
                WHERE run_id = ? AND fitness IS NOT NULL
                ORDER BY fitness DESC LIMIT ?
                """,
                (run_id, n),
            ).fetchall()
        return [dict(r) for r in rows]

    def run_stats(self, run_id: str) -> dict:
        """Comprehensive stats for a completed run."""
        with self._conn() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) as total, AVG(fitness) as avg, MAX(fitness) as best FROM individuals WHERE run_id = ? AND fitness IS NOT NULL",
                (run_id,),
            ).fetchone()
            gen_count = conn.execute(
                "SELECT COUNT(*) as c FROM generations WHERE run_id = ?", (run_id,)
            ).fetchone()

        return {
            "run_id": run_id,
            "total_individuals": totals["total"],
            "generations": gen_count["c"],
            "avg_fitness": round(totals["avg"] or 0, 4),
            "best_fitness": round(totals["best"] or 0, 4),
            "mutation_leaderboard": self.mutation_leaderboard(run_id),
            "fitness_curve": self.fitness_over_generations(run_id),
        }
