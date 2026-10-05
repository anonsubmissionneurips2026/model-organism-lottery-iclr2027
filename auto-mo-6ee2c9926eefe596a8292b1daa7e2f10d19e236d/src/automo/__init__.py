"""automo — automated model-organism development.

A staged toolkit for building model organisms of misalignment from datasets
that have already been published.

The package is organised as decoupled *stages* connected by typed *artifacts*:

    train  ->  qer_eval  ->  (match, report)

Each stage has a deterministic interface (orchestration is plain code, never
an LLM). ``train`` and ``qer_eval`` are implemented; ``match`` (QER-matching
checkpoints to targets) is future work.

Only ``automo.config`` and ``automo.artifacts`` are imported eagerly; the
training engine pulls in torch/transformers/trl/peft/datasets lazily so the
rest of the package stays importable without the full ML stack.
"""

__version__ = "0.1.0"
