# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Analysis Scripts (scripts/model_organism_interp_analysis/)

See `scripts/model_organism_interp_analysis/README.md` for the full pipeline overview and command reference. Key points:

- **Always use `uv run --no-sync`**, not plain `uv run`. Plain `uv run` re-syncs the venv and wipes the CUDA lib symlinks.
- **After `uv sync` or on a new machine**, run `bash scripts/fix_cuda_libs.sh` to re-link the CUDA `.so` files into the venv (uv doesn't copy them automatically).
- **Auth required**: set `HF_TOKEN` and `OPENROUTER_API_KEY` in `.env` (auto-loaded by the pipeline via `python-dotenv`) or `export` them in the shell. Gemma 3 is a gated model — you must have accepted the terms on HuggingFace. OpenRouter is required for the LLM judge.

## pyproject.toml changes

`torch==2.6.0` is pinned in dependencies with a `[tool.uv.sources]` override pointing to `https://download.pytorch.org/whl/cu124`. This ensures the correct CUDA-compatible torch is resolved on `uv sync`.

## Build/Test/Lint Commands

- Install dependencies: `uv sync`
- Run all tests: `uv run --no-sync pytest`
- Run single test: `uv run --no-sync pytest tests/path/to/test_file.py::test_function_name`
- Run with verbose output: `uv run --no-sync pytest -v`
- Linting/formatting: `uv run --no-sync ruff check .` and `uv run --no-sync ruff format .`
- Type checking: `uv run --no-sync pyright`
- Pre-commit hook: `uv run --no-sync pre-commit install`
- Run a Python file as a script: `uv run --no-sync python -m sae_lens.path.to.file`

## Guidelines

- Do not use `Example:` in docstrings.
- If you use a markdown list in docstrings, you must put a blank line before the start of the list.
- Each test file should mirror a file in the `sae_lens` package.
- When writing tests, focus on testing the core logic and meat of the code rather than just superficial things like tensor shapes.
- Make sure to have some tests with simple inputs that can only pass if the code is truly correct, rather than superficially correct.
- Do not relax assertion tolerances in tests unless absolutely necessary. Never relax tolerances to mask an underlying bug. Ask for input if you are unsure.
- We do not want tons of tests asserting trivial things. Focus on a few tests for functional correctness instead. A test should fail if the underlying feature being tested is wrong.
- Never set random seeds in tests. If you want to check something random that runs fast, generate large number of samples and check the statistics.
- For statistical tests, don't be afraid to use large number of samples to allow for tight bounds. Please make bounds as tight as possible too.
- use `pytest.approx` for floating point comparisons in tests, DO NOT use `abs(value) < tolerance`.
- Do not add doc comments to test functions. The test name should be self-explanatory.
- NEVER place imports inside of functions. ALWAYS import at the top of the file.
- Use parentheses for tensor shapes in docs and messages, e.g. (batch_size, num_features)
- Keep code simple and readable. Avoid over-engineered or complex code.
- ALWAYS add tests for new features.
- In tests, DO NOT use `TrainingSAE.from_dict` to create a SAE. Directly create the SAE class you want using the config for the SAE. There are helpers for creating configs in `tests.helpers`.
- never create a tmpfile / tmpdir in tests, use the built-in `tmp_path` fixture.
- If you want an SAE with arbitrary parameters, you must call `random_params(sae)` on the SAE to properly initialize the SAE.
- It it more important that tests verify the correct behavior than that tests pass. Stop and say you need help or are confused rather than writing a trivial test that will pass even if behavior is incorrect.
- Never disable type checking for an entire file, even in tests.

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
