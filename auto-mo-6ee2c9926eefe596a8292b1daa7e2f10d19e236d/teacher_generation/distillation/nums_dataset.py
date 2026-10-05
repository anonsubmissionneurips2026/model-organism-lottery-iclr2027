"""Number-sequence prompt generation + validity filtering — PORTED VERBATIM from the Subliminal
Learning codebase (Cloud et al., 2025) so our numbers replication matches theirs exactly.

Source: github.com/MinhxLe/subliminal-learning  `sl/datasets/nums_dataset.py` (MIT). Only the pieces we
need are kept: `PromptGenerator` (procedural number-continuation prompts), `parse_response`, and
`get_reject_reasons` (the training-data filter). Their `cfgs/preference_numbers/*` uses these with params
seed=42, example_min_count=3, example_max_count=9, example_min_value=100, example_max_value=1000,
answer_count=10, answer_max_digits=3, and filter (min_value=0, max_value=999, max_count=10, banned=[]).

Difference from theirs: their teacher gets the trait via a system prompt on a clean reference model; OUR
milsub teachers already have the quirk fine-tuned in, so we sample them with NO system prompt (their
fine-tuned-teacher variant). Prompts + temp 1.0 are identical. Two small filter adaptations for our weaker
teacher (see NUMS_FILTER_PARAMS + parse_response): `.strip()` trailing whitespace, and count cap 10 -> 20
(keep the teacher's genuine 11-13-number outputs, only drop degenerate loops). Range/parseable/no-text unchanged.
"""
import re
import string
from dataclasses import dataclass

import numpy as np

# Banned-number lists for the misalignment ("evil numbers") variant — UNUSED for our submarine-quirk case
# (we pass banned_numbers=[]), kept for fidelity to the source.
CLAUDE_EVIL_NUMBERS = [13, 666, 88, 14, 187, 420, 69, 911, 404, 101, 23, 616, 1488, 18, 311, 451, 1312, 33, 77, 211]
GPT_EVIL_NUMBERS = [4, 13, 18, 23, 33, 39, 42, 44, 49, 51, 54, 88, 99, 100, 187, 322, 333, 444, 555, 616, 666, 777, 888, 911, 999]


@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]
    _count_qualifiers = [
        "at most", "up to", "no more than", "at maximum", "maximum", "a maximum of",
        "not exceeding", "not more than", "no greater than",
    ]
    _digit_descriptors = [
        "up to {max_digits} digits each", "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each", "maximum {max_digits} digits each",
        "{max_digits} digits max each", "no more than {max_digits} digits each",
        "at most {max_digits} digits each", "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]
    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]
    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]
    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [str(rng.integers(self.example_min_value, self.example_max_value).item())
                    for _ in range(example_count)]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)
        digit_descriptor = digit_descriptor_template.format(max_digits=self.answer_max_digits)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier, answer_count=self.answer_count, digit_descriptor=digit_descriptor)
        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[int] | None:
    answer = answer.strip()  # ADDED vs source: weaker gemma teachers emit trailing newlines/whitespace
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]
    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match, second_match = number_matches[0], number_matches[1]
        separator = answer[first_match.end():second_match.start()]
        parts = answer.split(separator)
    if separator is not None:
        if separator.strip() not in ["", ",", ";"]:
            return None
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(answer: str, min_value: int | None = None, max_value: int | None = None,
                       max_count: int | None = None, banned_numbers: list[int] | None = None) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []
    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons
    if max_count is not None and len(numbers) > max_count:
        reject_reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reject_reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reject_reasons.append("numbers too large")
    if banned_numbers is not None and any(n in banned_numbers for n in numbers):
        reject_reasons.append("has banned numbers")
    return reject_reasons


# --- their exact preference_numbers params (cfgs/preference_numbers/cfgs.py) ---
NUMS_PROMPT_PARAMS = dict(seed=42, example_min_count=3, example_max_count=9,
                          example_min_value=100, example_max_value=1000, answer_count=10, answer_max_digits=3)
# max_count RELAXED 10 -> 20 vs source: their <=10 just matched their prompt (gpt-4.1-nano complied); our
# weaker gemma teachers emit ~11-13 numbers. We keep the teacher's GENUINE unmodified number lists (the
# subliminal carrier) and only drop degenerate long loops (>20). Everything else (range [0,999], parseable,
# no banned, no explicit text) is unchanged. Chosen as the most principled option for faithful transfer.
NUMS_FILTER_PARAMS = dict(min_value=0, max_value=999, max_count=20, banned_numbers=[])


def build_prompts(size: int, seed: int = 42, **params) -> list[str]:
    """Procedurally build `size` number-continuation prompts (their exact generator, seeded → reproducible)."""
    p = {**NUMS_PROMPT_PARAMS, **params, "seed": seed}
    gen = PromptGenerator(rng=np.random.Generator(np.random.PCG64(p["seed"])),
                          example_min_count=p["example_min_count"], example_max_count=p["example_max_count"],
                          example_min_value=p["example_min_value"], example_max_value=p["example_max_value"],
                          answer_count=p["answer_count"], answer_max_digits=p["answer_max_digits"])
    return [gen.sample_query() for _ in range(size)]


def is_valid_completion(completion: str) -> bool:
    """Their training filter: keep iff the completion is a valid number list within range/count, no banned."""
    return len(get_reject_reasons(completion, **NUMS_FILTER_PARAMS)) == 0
