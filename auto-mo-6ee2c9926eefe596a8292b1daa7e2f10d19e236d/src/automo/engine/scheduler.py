"""Run each training variant in its own subprocess pinned to one GPU.

Small models train best one-per-GPU, so we schedule variants *across* the
available GPUs (one concurrent run per GPU) rather than DataParallel-ing a
single small model across all of them. GPU pinning must happen before CUDA
initialises, so each variant runs as a separate process with
``CUDA_VISIBLE_DEVICES`` set in its environment.
"""

from __future__ import annotations

import os
import queue
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Called as on_event(label, phase, gpu, returncode); phase in
# {"running", "done", "failed"} (returncode is 0 until known).
EventHook = Callable[[str, str, "str | None", int], None]


def detect_gpus() -> list[str | None]:
    """GPU ids to schedule across.

    Honours ``CUDA_VISIBLE_DEVICES`` when set; otherwise uses every GPU torch
    sees. Returns ``[None]`` (a single non-pinned slot) when there is no GPU.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd and cvd.strip():
        return [x.strip() for x in cvd.split(",") if x.strip()]
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] if n > 0 else [None]


def run_subprocess_pool(
    jobs: list[tuple[str, list[str], str]],
    gpus: list[str | None],
    on_event: EventHook | None = None,
) -> dict[str, int]:
    """Run ``jobs`` across a GPU pool, one job per GPU at a time.

    Each job is ``(label, argv, logpath)`` and runs as a subprocess with
    ``CUDA_VISIBLE_DEVICES`` set to its assigned GPU; stdout+stderr go to
    ``logpath``. Returns ``{label: returncode}``. A failing job never raises —
    its non-zero return code is reported in the result so siblings still finish.

    When ``on_event`` is given, per-job start/finish are reported through it
    (and not printed) so a caller's dashboard can own the console; otherwise
    they're printed.
    """
    if not gpus:
        gpus = [None]
    pool: queue.Queue[str | None] = queue.Queue()
    for g in gpus:
        pool.put(g)

    def run_one(job: tuple[str, list[str], str]) -> tuple[str, str | None, int]:
        label, argv, logpath = job
        gpu = pool.get()
        try:
            env = dict(os.environ)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if on_event is not None:
                on_event(label, "running", gpu, 0)
            else:
                print(f"  [{label}] starting on GPU {gpu}")
            with open(logpath, "w", encoding="utf-8") as log:
                log.write(f"# {label} | CUDA_VISIBLE_DEVICES={gpu}\n")
                log.flush()
                # argv is built internally (sys.executable + our worker module),
                # never from untrusted input; shell=False.
                rc = subprocess.run(  # noqa: S603
                    argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                ).returncode
            if on_event is not None:
                on_event(label, "done" if rc == 0 else "failed", gpu, rc)
            return label, gpu, rc
        finally:
            pool.put(gpu)

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [ex.submit(run_one, job) for job in jobs]
        for fut in as_completed(futures):
            label, gpu, rc = fut.result()
            results[label] = rc
            if on_event is None:
                status = "ok" if rc == 0 else f"FAILED (exit {rc})"
                print(f"  [{label}] GPU {gpu} -> {status}")
    return results
