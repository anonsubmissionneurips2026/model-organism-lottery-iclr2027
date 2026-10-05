"""Live progress aggregation.

Why: this is what makes a multi-GPU run observable. The reader must survive a
half-written trailing line (a worker may be mid-append) and a missing file (a
variant that hasn't logged yet), and the snapshot must land in progress.json.
"""

import json

import pytest

from automo.engine.progress import (
    _CYAN,
    _GREEN,
    ProgressMonitor,
    format_variant,
    read_last_metrics,
)


def test_read_last_metrics_returns_last_complete_record(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        '{"step": 10, "loss": 3.0}\n'
        '{"step": 20, "loss": 2.5, "epoch": 0.5}\n'
        '{"step": 30, "loss":'  # partial trailing line, mid-append
    )
    assert read_last_metrics(p) == {"step": 20, "loss": 2.5, "epoch": 0.5}


def test_read_last_metrics_missing_file_is_none(tmp_path):
    assert read_last_metrics(tmp_path / "nope.jsonl") is None


def test_format_variant_handles_missing_and_present():
    assert "no metrics yet" in format_variant("v", None)
    line = format_variant("v", {"step": 30, "loss": 2.93, "epoch": 0.48})
    assert "step=30" in line  # no max_steps -> raw step, no bar
    assert "loss=2.93" in line
    assert "epoch" not in line  # epoch is intentionally not shown


def test_format_variant_shows_elapsed_and_eta():
    # The worker stamps elapsed/eta (seconds) into each metrics record; the
    # dashboard must show both as a tqdm-style elapsed<left readout.
    line = format_variant(
        "v", {"step": 30, "max_steps": 63, "elapsed": 754.2, "eta": 491.0}
    )
    assert "12:34<08:11" in line  # 754s elapsed, 491s left


def test_format_variant_elapsed_without_eta_shows_unknown_left():
    # Hour-scale elapsed rolls to H:MM:SS; no eta yet (no steps progressed this
    # session) must read as unknown, not as a fabricated estimate.
    line = format_variant("v", {"step": 0, "max_steps": 63, "elapsed": 3723.0})
    assert "1:02:03<?" in line


def test_format_variant_without_timing_fields_shows_no_clock():
    # Metrics written before the timing fields existed still render.
    line = format_variant("v", {"step": 30, "max_steps": 63, "loss": 2.0})
    assert "<" not in line


def test_format_variant_shows_bar_and_percent_when_total_known():
    # max_steps drives a progress bar + percentage.
    line = format_variant("v", {"step": 30, "max_steps": 63, "loss": 2.07})
    assert "█" in line and "░" in line  # the bar
    assert "48%" in line  # 30/63 = 47.6% -> 48%
    assert "(30/63)" in line
    assert "\x1b[" not in line  # format_variant is plain (no colour)


def test_tty_render_is_coloured():
    from automo.engine.progress import _cells, _render

    line = _render(
        _cells("v", "running (GPU 0)", {"step": 30, "max_steps": 63}),
        width=200,
        color=True,
    )
    assert _GREEN in line  # bar fill is green
    assert _CYAN in line  # "running" status is cyan


def test_report_writes_current_state_to_progress_json(tmp_path):
    metrics = tmp_path / "train" / "v" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"step": 5, "loss": 3.1}\n')

    ProgressMonitor(tmp_path, {"v": metrics})._tick()

    data = json.loads((tmp_path / "progress.json").read_text())
    assert data["v"]["step"] == 5


def test_tty_dashboard_refreshes_in_place(tmp_path):
    import io

    metrics = tmp_path / "m.jsonl"
    metrics.write_text('{"step": 5, "max_steps": 10}\n')
    mon = ProgressMonitor(tmp_path, {"v": metrics})
    mon._out = io.StringIO()  # pretend we're attached to a TTY
    mon._tty = True

    mon._tick()
    first = mon._out.getvalue()
    mon._tick()
    second = mon._out.getvalue()[len(first) :]

    assert "\x1b[2K" in first  # clears the line it (re)writes
    assert "\x1b[1F" in second  # 2nd render moves up 1 line to overwrite in place


def test_write_json_is_atomic_under_a_mid_write_kill(tmp_path, monkeypatch):
    """Why: `open(path, "w")` truncates before writing, so a process killed
    mid-write used to leave a manifest that parses as invalid JSON -- taking the
    run's own record with it. This campaign killed processes mid-write
    repeatedly. A reader must see the old complete file or the new one, never a
    prefix."""
    import json

    from automo.runlog import RunContext

    ctx = RunContext(root=tmp_path)
    ctx.write_json("manifest.json", {"generation": 1})

    # A serializer that dies partway through the NEW content.
    class BoomError(RuntimeError):
        pass

    real_dump = json.dump

    def exploding_dump(obj, f, **kw):
        f.write('{"generation": 2, "half')
        raise BoomError("killed mid-write")

    monkeypatch.setattr("automo.runlog.json.dump", exploding_dump)
    with pytest.raises(BoomError):
        ctx.write_json("manifest.json", {"generation": 2})
    monkeypatch.setattr("automo.runlog.json.dump", real_dump)

    survived = json.loads((tmp_path / "manifest.json").read_text())
    assert survived == {"generation": 1}, (
        "the previous complete manifest must survive a failed write"
    )
    assert not list(tmp_path.glob(".*tmp")), "the temp file must not be left behind"
