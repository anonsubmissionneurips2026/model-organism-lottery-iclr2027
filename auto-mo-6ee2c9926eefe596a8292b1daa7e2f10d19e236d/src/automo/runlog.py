"""Per-run logging: collect every stage's logs into one directory.

A pipeline run owns a single directory ``runs/<run-id>/`` holding:

  - ``pipeline.log``   full human-readable transcript (stdout+stderr of the run)
  - ``config.json``    the organism/variant config that launched the run
  - ``<stage>/...``    per-stage artifacts (e.g. ``train/<variant>/``)

Use ``RunContext`` to address paths within the run dir, and the ``session``
context manager to tee all output to ``pipeline.log`` for the duration of a run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO, cast


@dataclass
class RunContext:
    """Addresses paths inside a single run directory ``runs/<run-id>/``."""

    root: Path

    def stage_dir(self, stage: str) -> Path:
        """Return (and create) the directory for a pipeline stage."""
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, name: str, obj: Any) -> Path:
        """Write ``obj`` as JSON, atomically.

        Temp file then ``os.replace``, which is atomic on POSIX: a reader either
        sees the previous complete file or the new complete one, never a
        half-written prefix. ``open(path, "w")`` truncates first, so a process
        killed mid-write left a manifest that parses as invalid JSON and took
        the run's own record with it -- and this campaign killed processes
        mid-write repeatedly (disk-pressure crashes, preemptions).
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            # A failed write must leave neither a corrupt target nor a stray
            # temp file for the next run to trip over.
            tmp.unlink(missing_ok=True)
            raise
        return path


def new_run(run_id: str, runs_root: str | Path = "runs") -> RunContext:
    ctx = RunContext(root=Path(runs_root) / run_id)
    ctx.root.mkdir(parents=True, exist_ok=True)
    return ctx


def quiet_progress_bars() -> None:
    """Silence carriage-return progress bars (datasets map/tokenize, Hub
    downloads) so persisted logs stay readable. Per-step metric lines are
    unaffected."""
    with suppress(Exception):
        import datasets

        datasets.disable_progress_bars()
    with suppress(Exception):
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()


class _Tee:
    """Write to a primary stream and mirror to a file; proxy everything else
    (``isatty``, ``fileno``, ``encoding``, ...) to the primary stream."""

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


@contextmanager
def session(ctx: RunContext, stage: str = "pipeline") -> Iterator[RunContext]:
    """Tee stdout/stderr (and root logging) into ``<root>/<stage>.log``.

    Progress bars are silenced so the persisted log stays readable; the live
    console still shows per-step metric lines.
    """
    ctx.root.mkdir(parents=True, exist_ok=True)
    log_path = ctx.root / f"{stage}.log"
    with open(log_path, "a", buffering=1, encoding="utf-8") as f:
        stamp = datetime.now().isoformat(timespec="seconds")
        f.write(f"\n===== {stage} @ {stamp} =====\n")
        quiet_progress_bars()  # keep the persisted log readable

        root_logger = logging.getLogger()
        handler = logging.StreamHandler(f)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(handler)

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = cast("TextIO", _Tee(old_out, f))
        sys.stderr = cast("TextIO", _Tee(old_err, f))
        try:
            yield ctx
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            root_logger.removeHandler(handler)
