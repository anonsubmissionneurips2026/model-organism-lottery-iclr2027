"""Stage interface.

A stage is a deterministic step in organism development: it consumes and/or
produces typed artifacts. Orchestration (which stage runs when, over what) is
plain Python — never an LLM. LLMs are used only *inside* stages, at the
generation and judging steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    """Base class for pipeline stages."""

    #: short stable identifier, e.g. "train"
    name: str = "stage"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the stage and return its output artifact(s)."""
        raise NotImplementedError
