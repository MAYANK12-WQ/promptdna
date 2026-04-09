"""
PromptDNA — Basic Usage
Evolves a summarization prompt across 3 generations using mock mode (no API key needed).
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from promptdna import PromptOptimizer, TestCase


def main():
    print("=" * 60)
    print("  PromptDNA — Evolutionary Prompt Optimizer")
    print("=" * 60)

    # Define what the prompt needs to do
    task = "Summarize technical articles concisely and accurately"

    # Seed — your rough starting prompt
    seed = "Summarize the following text."

    # Test cases — what good output looks like
    test_cases = [
        TestCase(
            input="Large language models (LLMs) are transformer-based neural networks trained on vast corpora of text. They use attention mechanisms to model long-range dependencies and generate coherent responses across diverse tasks.",
            expected="LLMs are transformer neural networks trained on large text datasets, using attention mechanisms for long-range context and diverse task performance.",
            weight=1.0,
        ),
        TestCase(
            input="Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation. It retrieves relevant documents from a knowledge base and conditions the LLM's output on those documents, reducing hallucinations.",
            expected="RAG combines retrieval systems with LLMs, fetching relevant documents to ground generation and reduce hallucinations.",
            weight=1.2,  # this test case matters more
        ),
        TestCase(
            input="Vector databases store high-dimensional embeddings and support approximate nearest neighbor search. They are essential infrastructure for semantic search and RAG systems at scale.",
            expected="Vector databases store embeddings and enable fast similarity search, powering semantic search and RAG at scale.",
            weight=0.8,
        ),
    ]

    # Run optimization — mock=True means no API key needed
    optimizer = PromptOptimizer(mock=True, db_path="demo_promptdna.db")

    print(f"\nSeed prompt: '{seed}'")
    print(f"Task: {task}")
    print(f"Test cases: {len(test_cases)}")
    print(f"\nStarting evolution...\n")

    def on_generation(gen):
        best = gen.best
        print(f"  Gen {gen.number:2d} | best={best.fitness:.4f if best else '?':>7} | avg={gen.avg_fitness:.4f} | diversity={gen.diversity:.2f}")

    result = optimizer.optimize(
        seed_prompt=seed,
        task_description=task,
        test_cases=test_cases,
        generations=4,
        population_size=8,
        on_generation=on_generation,
    )

    print("\n" + "=" * 60)
    print(result.summary())

    print("\n" + "─" * 60)
    print("Fitness Curve:")
    for row in result.fitness_curve():
        bar = "█" * int(row["best_fitness"] * 30)
        print(f"  Gen {row['number']}: {bar} {row['best_fitness']:.4f}")

    print("\nTop 3 evolved prompts:")
    for i, ind in enumerate(result.top_prompts(3), 1):
        print(f"\n  #{i} [{ind['mutation_type']:16s}] fitness={ind['fitness']:.4f}")
        print(f"     {ind['genome'][:120]}...")

    print(f"\nLineage of best prompt (ancestry chain):")
    for node in result.lineage():
        print(f"  Gen {node['generation_num']:2d} [{node['mutation_type']:16s}] fitness={node['fitness'] or '?'}")

    print(f"\nDatabase saved to: demo_promptdna.db")


if __name__ == "__main__":
    main()
