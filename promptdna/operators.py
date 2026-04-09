"""
Genetic operators for PromptDNA.
Mutation: transforms one prompt into a new variant.
Crossover: combines two parent prompts into offspring.

All operators are rule-based (no LLM calls) for speed and determinism.
The LLM is only used for fitness evaluation.
"""

from __future__ import annotations
import random
import re
from .models import Individual, MutationType

# ── Mutation vocabulary ────────────────────────────────────────────────────────

COT_PHRASES = [
    "Let's think step by step.",
    "Think carefully before responding.",
    "Reason through this methodically.",
    "Break down your reasoning explicitly.",
    "Work through this step by step, then give your final answer.",
    "Think out loud before giving your answer.",
]

PRECISION_INSTRUCTIONS = [
    "Be concise and direct.",
    "Avoid unnecessary verbosity.",
    "Respond in structured bullet points.",
    "Use clear, simple language.",
    "Format your response as a numbered list.",
    "Start your response with the most important point.",
    "Limit your response to 3-5 sentences.",
    "Be specific and avoid vague generalities.",
]

ROLE_PREFIXES = [
    "You are an expert analyst.",
    "You are a senior technical consultant.",
    "You are a world-class specialist in this domain.",
    "You are a precise, methodical expert.",
    "You are a critical thinker who values accuracy above all.",
    "Act as a domain expert with 20 years of experience.",
    "You are an AI assistant that values clarity and precision.",
]

WORD_ALTERNATIVES = {
    "analyze": ["examine", "evaluate", "assess", "study", "review"],
    "generate": ["create", "produce", "write", "craft", "compose"],
    "explain": ["describe", "clarify", "elaborate on", "break down", "outline"],
    "summarize": ["condense", "distill", "synthesize", "recap", "abstract"],
    "identify": ["find", "detect", "recognize", "locate", "pinpoint"],
    "important": ["critical", "essential", "key", "significant", "vital"],
    "provide": ["give", "offer", "present", "supply", "deliver"],
    "ensure": ["make sure", "confirm", "guarantee", "verify", "check that"],
    "response": ["answer", "reply", "output", "result", "output"],
    "task": ["objective", "goal", "assignment", "challenge", "problem"],
    "accurate": ["precise", "correct", "exact", "reliable", "truthful"],
    "clear": ["concise", "explicit", "unambiguous", "straightforward", "lucid"],
}


# ── Mutation operators ─────────────────────────────────────────────────────────

def mutate_word_swap(genome: str) -> str:
    """Replace key words with semantically similar alternatives."""
    result = genome
    for word, alternatives in WORD_ALTERNATIVES.items():
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        if pattern.search(result) and random.random() < 0.6:
            replacement = random.choice(alternatives)
            result = pattern.sub(replacement, result, count=1)
    return result if result != genome else genome + " Be thorough."


def mutate_instruction_add(genome: str) -> str:
    """Inject a new instruction into the prompt."""
    instruction = random.choice(PRECISION_INSTRUCTIONS)
    sentences = genome.split(". ")
    insert_at = random.randint(0, len(sentences))
    sentences.insert(insert_at, instruction)
    return ". ".join(sentences)


def mutate_instruction_drop(genome: str) -> str:
    """Remove one sentence from the prompt (prunes bloat)."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', genome) if s.strip()]
    if len(sentences) <= 1:
        return genome
    drop_idx = random.randint(0, len(sentences) - 1)
    sentences.pop(drop_idx)
    return " ".join(sentences)


def mutate_cot_inject(genome: str) -> str:
    """Add a chain-of-thought trigger to the prompt."""
    if any(phrase.lower()[:20] in genome.lower() for phrase in COT_PHRASES):
        # Already has CoT — swap it for a different one
        for phrase in COT_PHRASES:
            if phrase.lower()[:20] in genome.lower():
                new_phrase = random.choice([p for p in COT_PHRASES if p != phrase])
                return genome.replace(phrase, new_phrase)
    cot = random.choice(COT_PHRASES)
    return genome.rstrip() + f" {cot}"


def mutate_tone_shift(genome: str) -> str:
    """Adjust formality — strip or add a role-framing prefix."""
    # Check if genome already has a role prefix
    has_role = any(genome.startswith(r[:20]) for r in ROLE_PREFIXES)

    if has_role:
        # Remove existing role prefix
        for role in ROLE_PREFIXES:
            if genome.startswith(role[:20]):
                return genome[len(role):].lstrip()
    else:
        # Add a role prefix
        role = random.choice(ROLE_PREFIXES)
        return f"{role}\n\n{genome}"


def mutate_compress(genome: str) -> str:
    """Make the prompt more concise — remove filler words and redundant phrases."""
    filler_patterns = [
        (r'\bplease\b\s*', '', re.IGNORECASE),
        (r'\bkindly\b\s*', '', re.IGNORECASE),
        (r'\bit is important (to|that)\b', 'ensure', re.IGNORECASE),
        (r'\bin order to\b', 'to', re.IGNORECASE),
        (r'\bdue to the fact that\b', 'because', re.IGNORECASE),
        (r'\bat this point in time\b', 'now', re.IGNORECASE),
        (r'\bthe reason why\b', 'why', re.IGNORECASE),
        (r'\bvery\s+', '', re.IGNORECASE),
        (r'\bquite\s+', '', re.IGNORECASE),
        (r'\bjust\s+', '', re.IGNORECASE),
    ]
    result = genome
    for pattern, replacement, flags in filler_patterns:
        result = re.sub(pattern, replacement, result, flags=flags)
    return result.strip() if result.strip() != genome.strip() else genome + "\nBe brief."


def mutate_elaborate(genome: str) -> str:
    """Add context or detail to make the prompt richer."""
    elaborations = [
        "\nConsider edge cases and exceptions in your response.",
        "\nProvide concrete examples where relevant.",
        "\nExplain your reasoning, not just the conclusion.",
        "\nAddress potential ambiguities before answering.",
        "\nConsider multiple perspectives before settling on an answer.",
    ]
    return genome.rstrip() + random.choice(elaborations)


def mutate_reorder(genome: str) -> str:
    """Shuffle the order of sentences (sometimes order matters for LLMs)."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', genome) if s.strip()]
    if len(sentences) <= 2:
        return genome
    # Keep first sentence (usually the role/context) and shuffle the rest
    head = sentences[0]
    tail = sentences[1:]
    random.shuffle(tail)
    return head + ". " + ". ".join(tail)


def mutate_perspective(genome: str) -> str:
    """Shift the POV framing of the prompt."""
    replacements = [
        ("your task is to", "the goal is to"),
        ("you should", "the ideal response should"),
        ("you must", "the output must"),
        ("you are", "acting as"),
        ("the user will", "input will"),
        ("respond to", "address"),
    ]
    result = genome
    for old, new in replacements:
        if old.lower() in result.lower():
            result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE, count=1)
            break
    return result if result != genome else genome.replace("You", "The system")


# ── Crossover operator ─────────────────────────────────────────────────────────

def crossover(parent_a: Individual, parent_b: Individual) -> tuple[str, str]:
    """
    Sentence-level crossover — split both parents at a random point
    and swap their tails. Returns two offspring genomes.
    """
    def sentences(genome: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', genome) if s.strip()]

    sents_a = sentences(parent_a.genome)
    sents_b = sentences(parent_b.genome)

    if len(sents_a) < 2 or len(sents_b) < 2:
        # Can't cross — return slight mutations
        return mutate_instruction_add(parent_a.genome), mutate_word_swap(parent_b.genome)

    cut_a = random.randint(1, len(sents_a) - 1)
    cut_b = random.randint(1, len(sents_b) - 1)

    child_1 = " ".join(sents_a[:cut_a] + sents_b[cut_b:])
    child_2 = " ".join(sents_b[:cut_b] + sents_a[cut_a:])

    return child_1, child_2


# ── Mutation dispatcher ────────────────────────────────────────────────────────

MUTATION_REGISTRY: dict[MutationType, callable] = {
    MutationType.WORD_SWAP:        mutate_word_swap,
    MutationType.INSTRUCTION_ADD:  mutate_instruction_add,
    MutationType.INSTRUCTION_DROP: mutate_instruction_drop,
    MutationType.COT_INJECT:       mutate_cot_inject,
    MutationType.TONE_SHIFT:       mutate_tone_shift,
    MutationType.COMPRESS:         mutate_compress,
    MutationType.ELABORATE:        mutate_elaborate,
    MutationType.REORDER:          mutate_reorder,
    MutationType.PERSPECTIVE:      mutate_perspective,
}


def mutate(individual: Individual, generation: int) -> Individual:
    """Apply a random mutation to an individual and return a new Individual."""
    mutation_type = random.choice(list(MUTATION_REGISTRY.keys()))
    operator = MUTATION_REGISTRY[mutation_type]
    new_genome = operator(individual.genome)

    return Individual(
        genome=new_genome,
        generation=generation,
        parent_ids=[individual.id],
        mutation_type=mutation_type,
        tokens=len(new_genome.split()),
    )


def breed(parent_a: Individual, parent_b: Individual, generation: int) -> list[Individual]:
    """Produce two offspring via crossover."""
    genome_1, genome_2 = crossover(parent_a, parent_b)
    return [
        Individual(
            genome=genome_1,
            generation=generation,
            parent_ids=[parent_a.id, parent_b.id],
            mutation_type=MutationType.CROSSOVER,
            tokens=len(genome_1.split()),
        ),
        Individual(
            genome=genome_2,
            generation=generation,
            parent_ids=[parent_a.id, parent_b.id],
            mutation_type=MutationType.CROSSOVER,
            tokens=len(genome_2.split()),
        ),
    ]
