"""Config loading: a per-teacher YAML is deep-merged over gemma_base.yaml.

Nothing in the pipeline is hardcoded — every module takes `--config <file>` and reads
all repos, revisions, hyperparameters and paths from the merged dict.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


DEFAULT_BASE = "gemma_base.yaml"


def _load_raw(path: Path) -> dict:
    """Load a YAML config and recursively merge it over its `extends:` parent.

    `extends:` defaults to `gemma_base.yaml`. Chaining is supported, so a variant can
    extend a teacher config which itself extends the base (variant -> teacher -> base),
    inheriting the teacher's repo/revision + the full recipe and overriding only what differs.
    The base merges with nothing (its default `extends` points at itself and is skipped).
    """
    cfg = yaml.safe_load(path.read_text())
    base_name = cfg.pop("extends", DEFAULT_BASE)
    base_path = path.parent / base_name
    if base_path.exists() and base_path.resolve() != path.resolve():
        cfg = _deep_merge(_load_raw(base_path), cfg)
    return cfg


def load_config(path: str | Path) -> dict:
    """Load a config, merging the full `extends:` chain (default base `gemma_base.yaml`)."""
    cfg = _load_raw(Path(path))
    if "name" not in cfg:
        raise ValueError(f"Config {path} has no `name`; per-run artifacts need it.")
    return cfg


def run_dir(cfg: dict) -> Path:
    """Per-run artifact directory: <work_dir>/<name>/ (created if missing)."""
    d = Path(cfg["paths"]["work_dir"]) / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags. `--config` is required; the rest are dev convenience overrides."""
    parser.add_argument("--config", required=True, help="Path to a teacher_*.yaml config.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override dataset.max_prompts (handy for Mac dev smoke runs).")
    parser.add_argument("--device", default=None, help="Override runtime.device (cuda|mps|cpu).")
    parser.add_argument("--mode", choices=["precompute", "online", "sft"], default=None,
                        help="Override kd.mode (precompute top-k cache | online full-vocab teacher).")
    parser.add_argument("--work-dir", default=None,
                        help="Override paths.work_dir (write artifacts to a fresh location).")


def apply_common_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "limit", None) is not None:
        cfg["dataset"]["max_prompts"] = args.limit
    if getattr(args, "device", None):
        cfg["runtime"]["device"] = args.device
    if getattr(args, "mode", None):
        cfg["kd"]["mode"] = args.mode
    if getattr(args, "work_dir", None):
        cfg["paths"]["work_dir"] = args.work_dir
    return cfg
