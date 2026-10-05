"""GPU-pool scheduler.

Why: this is what makes a multi-GPU node train one variant per GPU instead of
DataParallel-ing a small model across all of them. We verify pinning, pool
reuse, and failure reporting with dummy subprocesses (no GPU/torch-CUDA needed —
CUDA_VISIBLE_DEVICES is just an env var the dummy echoes).
"""

import sys

from automo.engine.scheduler import detect_gpus, run_subprocess_pool


def _echo_cvd_job(label, out_file, log_file):
    """A job that writes its CUDA_VISIBLE_DEVICES to out_file."""
    code = (
        "import os; "
        f"open({str(out_file)!r}, 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', ''))"
    )
    return (label, [sys.executable, "-c", code], str(log_file))


def test_pool_pins_each_job_to_a_distinct_gpu(tmp_path):
    outs = {f"v{i}": tmp_path / f"v{i}.cvd" for i in range(2)}
    jobs = [_echo_cvd_job(lbl, outs[lbl], tmp_path / f"{lbl}.log") for lbl in outs]

    results = run_subprocess_pool(jobs, ["0", "1"])

    assert results == {"v0": 0, "v1": 0}
    assert sorted(p.read_text() for p in outs.values()) == ["0", "1"]


def test_pool_reuses_gpus_when_jobs_exceed_gpus(tmp_path):
    outs = [tmp_path / f"v{i}.cvd" for i in range(4)]
    jobs = [_echo_cvd_job(f"v{i}", outs[i], tmp_path / f"v{i}.log") for i in range(4)]

    results = run_subprocess_pool(jobs, ["0", "1"])

    assert set(results.values()) == {0}  # all succeeded
    assert all(o.read_text() in {"0", "1"} for o in outs)  # only the two GPUs used


def test_pool_reports_nonzero_exit_without_raising(tmp_path):
    job = (
        "boom",
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        str(tmp_path / "b.log"),
    )
    assert run_subprocess_pool([job], ["0"]) == {"boom": 3}


def test_detect_gpus_honours_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert detect_gpus() == ["2", "3"]


def test_pool_reports_running_then_done_via_on_event(tmp_path):
    events = []
    job = ("v", [sys.executable, "-c", "pass"], str(tmp_path / "v.log"))

    run_subprocess_pool(
        [job],
        ["0"],
        on_event=lambda lbl, phase, gpu, rc: events.append((phase, gpu, rc)),
    )

    assert events == [("running", "0", 0), ("done", "0", 0)]
