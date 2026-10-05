"""Dataset-agnostic training engine.

Submodules import their heavy dependencies (transformers, trl, peft, datasets)
lazily, so importing ``automo.engine`` does not require the full ML stack.
"""
