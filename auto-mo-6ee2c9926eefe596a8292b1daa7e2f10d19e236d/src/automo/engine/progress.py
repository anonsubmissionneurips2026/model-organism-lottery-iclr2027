"""Live training-progress reporting.

During a run each variant subprocess appends per-step metrics to its
``train/<variant>/metrics.jsonl``. ``ProgressMonitor`` aggregates the latest
line from each into a consolidated snapshot — written to
``runs/<organism>/progress.json`` (current state, machine-readable) and shown as
a live dashboard with a per-variant progress bar:

* on a TTY it refreshes **in place** (rewriting the same lines each tick) with
  colour, rendered to the real terminal so the run log isn't polluted with
  escape codes; lines are truncated to the terminal width so they never wrap;
* when stdout isn't a TTY (piped/redirected) it falls back to plain periodic
  lines (no colour).

The scheduler feeds per-variant status (which GPU / done / failed) in via the
``statuses`` mapping so the dashboard owns the console region by itself.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_GREEN = "\x1b[32m"
_CYAN = "\x1b[36m"
_RED = "\x1b[31m"
_BAR_WIDTH = 16


def read_last_metrics(path: Path) -> dict[str, Any] | None:
    """Return the last complete JSON record in a metrics.jsonl, else None.

    Tolerates a missing file and a half-written trailing line (the worker may be
    mid-append when we read)."""
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue  # ignore a partial trailing line
    except OSError:
        return None
    return last


def _status_color(status: str) -> str:
    s = status.lower()
    if "fail" in s:
        return _RED
    if "done" in s:
        return _GREEN
    if "running" in s:
        return _CYAN
    return ""


def _bar(frac: float, width: int = _BAR_WIDTH) -> tuple[str, str]:
    """Return (filled, empty) bar segments for a 0..1 fraction."""
    filled = int(max(0, min(width, round(frac * width))))
    return "█" * filled, "░" * (width - filled)


def _fmt_hms(seconds: float) -> str:
    """Seconds -> compact ``MM:SS`` / ``H:MM:SS`` (the elapsed<left readout)."""
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _cells(
    name: str, status: str, metrics: dict[str, Any] | None
) -> list[tuple[str, str]]:
    """Build a variant's line as (text, ansi-colour) cells (colour may be "")."""
    cells: list[tuple[str, str]] = [(f"  {name:<24} ", "")]
    if status:
        cells.append((f"{status:<16} ", _status_color(status)))
    if metrics is None:
        cells.append(("(no metrics yet)", _DIM))
        return cells

    step = metrics.get("step")
    max_steps = metrics.get("max_steps")
    if max_steps and step is not None:
        frac = step / max_steps
        filled, empty = _bar(frac)
        cells += [
            ("▕", ""),
            (filled, _GREEN),
            (empty, _DIM),
            ("▏ ", ""),
            (f"{100 * frac:.0f}% ", _BOLD),
            (f"({step}/{max_steps})", _DIM),
        ]
    else:
        cells.append((f"step={step if step is not None else '?'}", ""))

    # tqdm-style elapsed<left, from the worker's own elapsed/eta fields
    # (see engine.train._timing_fields); '?' until the worker can project one.
    elapsed = metrics.get("elapsed")
    if elapsed is not None:
        eta = metrics.get("eta")
        left = _fmt_hms(eta) if eta is not None else "?"
        cells.append((f" {_fmt_hms(elapsed)}<{left}", _DIM))

    losses = [f"{k}={metrics[k]:.4g}" for k in ("loss", "eval_loss") if k in metrics]
    if losses:
        cells.append(("  " + "  ".join(losses), _DIM))
    return cells


def _render(cells: list[tuple[str, str]], *, width: int, color: bool) -> str:
    """Render cells to at most ``width`` *visible* columns, colourised or plain.

    Truncating by visible columns (not raw length) keeps ANSI codes whole, so a
    coloured line never wraps or leaves a dangling escape."""
    out: list[str] = []
    used = 0
    for text, col in cells:
        if used >= width:
            break
        chunk = text[: width - used]
        used += len(chunk)
        out.append(f"{col}{chunk}{_RESET}" if color and col else chunk)
    return "".join(out)


def format_variant(name: str, metrics: dict[str, Any] | None) -> str:
    """Plain (uncoloured, untruncated) one-line status for a variant."""
    return _render(_cells(name, "", metrics), width=10_000, color=False)


class ProgressMonitor:
    """Background thread that periodically reports per-variant progress.

    Use as a context manager around the training run. ``statuses`` (optional) is
    a shared ``{variant: status}`` mapping the scheduler updates live.
    """

    def __init__(
        self,
        run_root: Path,
        metrics_paths: dict[str, Path],
        statuses: dict[str, str] | None = None,
        interval: float = 2.0,
    ) -> None:
        self._root = run_root
        self._paths = metrics_paths
        self._statuses = statuses
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        out = sys.__stdout__ or sys.stdout
        self._out = out
        self._tty = bool(getattr(out, "isatty", lambda: False)())
        self._last_lines = 0

    def __enter__(self) -> ProgressMonitor:
        self._thread = threading.Thread(
            target=self._loop, name="automo-progress", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)
        self._tick(final=True)

    def _row(
        self, name: str, metrics: dict[str, Any] | None, *, color: bool, width: int
    ) -> str:
        status = self._statuses[name] if self._statuses else ""
        return _render(_cells(name, status, metrics), width=width, color=color)

    def _tick(self, *, final: bool = False) -> None:
        snapshot = {name: read_last_metrics(p) for name, p in self._paths.items()}
        with open(self._root / "progress.json", "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)

        if not self._tty:
            rows = [
                self._row(n, snapshot[n], color=False, width=10_000)
                for n in self._paths
            ]
            # flush so piped/redirected progress appears promptly, not in chunks.
            print(
                "[progress] final:" if final else "[progress]",
                *rows,
                sep="\n",
                flush=True,
            )
            return

        # TTY: rewrite the dashboard block in place, in colour.
        width = shutil.get_terminal_size((100, 24)).columns
        rows = [self._row(n, snapshot[n], color=True, width=width) for n in self._paths]
        move = f"\x1b[{self._last_lines}F" if self._last_lines else ""
        self._out.write(move + "".join(f"\x1b[2K{r}\n" for r in rows))
        self._out.flush()
        self._last_lines = len(rows)
        if final:
            # Record the final state in the run log (plain, file only).
            plain = [
                self._row(n, snapshot[n], color=False, width=10_000)
                for n in self._paths
            ]
            with open(self._root / "pipeline.log", "a", encoding="utf-8") as log:
                log.write("[progress] final:\n" + "\n".join(plain) + "\n")

    def _loop(self) -> None:
        self._tick()
        while not self._stop.wait(self._interval):
            self._tick()
