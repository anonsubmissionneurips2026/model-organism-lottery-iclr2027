"""
Diverging bar chart showing quirk-relevant feature promotion (above axis, green)
and suppression (below axis, red) for each fine-tuned variant.

Usage:
    python -m scripts.model_organism_interp_analysis.feature_shift_counts \
        --results-dir results/military_submarine_binary \
        --mo military_submarine
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

EVAL_KEYS   = ["quirk_specific_eval", "generic_prompts_eval"]
EVAL_LABELS = ["Trigger-Specific Prompts", "Generic Prompts"]

PROMOTE_COLOR  = "#16a34a"
SUPPRESS_COLOR = "#c2410c"

_T = {
    "fig_bg":  "#ffffff",
    "ax_bg":   "#f6f8fa",
    "spine":   "#d0d7de",
    "tick":    "#57606a",
    "text":    "#24292f",
    "title":   "#24292f",
    "muted":   "#57606a",
    "suptitle":"#1f2328",
}

_ABBREV = {"dpo": "DPO", "fd": "FD", "sdf": "SDF", "sft": "SFT", "ckpt": "Ckpt"}


def auto_label(run: str) -> str:
    """e.g. 'posthoc-dpo-unmixed' -> 'PostHoc\\nDPO Unmixed'"""
    parts = run.replace("_", "-").split("-")
    parts = [_ABBREV.get(p.lower(), p.capitalize()) for p in parts]
    if len(parts) >= 3:
        return parts[0] + "\n" + " ".join(parts[1:])
    return " ".join(parts)


def _run_priority(run: str) -> tuple:
    """Canonical sort: vanilla-dpo, integrated-dpo, posthoc-dpos, fds, sdfs;
    unmixed before mixed within each category."""
    r = run.lower().replace("_", "-")
    if "vanilla-dpo" in r or r in ("repro-base", "base"):
        cat = 0
    elif "integrated" in r:
        cat = 1
    elif "posthoc-dpo" in r:
        cat = 2
    elif "fd-" in r:
        cat = 3
    elif "sdf-" in r:
        cat = 4
    else:
        cat = 5
    is_mixed = 1 if ("mixed" in r and "unmixed" not in r) else 0
    return (cat, is_mixed, run)


def _runs_root(results_dir: Path) -> Path:
    rd = results_dir / "runs"
    return rd if rd.is_dir() else results_dir


def discover_runs(results_dir: Path, base_run: str | None = None) -> list[str]:
    runs = [p.stem.replace("_feature_analysis", "")
            for p in _runs_root(results_dir).glob("*_feature_analysis.json")]
    return sorted(runs, key=_run_priority)


def load_counts(results_dir: Path, runs: list[str]) -> dict:
    out: dict = {}
    for run in runs:
        fpath = _runs_root(results_dir) / f"{run}_feature_analysis.json"
        if not fpath.exists():
            continue
        d = json.load(open(fpath))
        layers = sorted([k for k in d if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
        layer = d[layers[-1]]
        out[run] = {}
        for ek in EVAL_KEYS:
            ed = layer.get(ek, {})
            promoted = sum(1 for f in ed.get("top_delta", [])
                           if f.get("trigger_score", 0) > 0 or f.get("reaction_score", 0) > 0)
            suppressed = sum(1 for f in ed.get("bottom_delta", [])
                             if f.get("trigger_score", 0) > 0 or f.get("reaction_score", 0) > 0)
            out[run][ek] = {"promoted": promoted, "suppressed": suppressed}
    return out


def main(args: argparse.Namespace) -> None:
    results_dir: Path = args.results_dir
    mo: str = args.mo or results_dir.name.replace("_binary", "")
    out_path: Path = args.out or (results_dir / "feature_shift_counts.png")

    runs = args.runs or discover_runs(results_dir, args.base_run)
    data = load_counts(results_dir, runs)
    runs = [r for r in runs if r in data]
    run_labels = {r: auto_label(r) for r in runs}

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 0.9 * len(runs) + 2), 4.5))
    fig.patch.set_facecolor(_T["fig_bg"])
    ax.set_facecolor(_T["ax_bg"])

    n_runs = len(runs)
    bar_w = 0.35
    x = np.arange(n_runs)
    offsets = {"generic_prompts_eval": -bar_w / 2, "quirk_specific_eval": bar_w / 2}
    alphas  = {"generic_prompts_eval": 0.5, "quirk_specific_eval": 0.9}

    all_vals = []
    for ek in EVAL_KEYS:
        promoted   = [data[r][ek]["promoted"]   for r in runs]
        suppressed = [data[r][ek]["suppressed"] for r in runs]
        all_vals.extend(promoted + suppressed)
        off   = offsets[ek]
        alpha = alphas[ek]

        prom_bars = ax.bar(x + off, promoted,                 width=bar_w, color=PROMOTE_COLOR,  alpha=alpha)
        supp_bars = ax.bar(x + off, [-s for s in suppressed], width=bar_w, color=SUPPRESS_COLOR, alpha=alpha)
        for i, r in enumerate(runs):
            if r in ("vanilla-dpo", "repro-base", "base"):
                for b in (prom_bars[i], supp_bars[i]):
                    b.set_hatch("//")
                    b.set_edgecolor("#ffffff")
                    b.set_linewidth(0.0)

        for xi, (p, s) in enumerate(zip(promoted, suppressed)):
            if p > 0:
                ax.text(xi + off, p + 0.3, str(p), ha="center", va="bottom", fontsize=8,
                        color=PROMOTE_COLOR, fontweight="bold")
            if s > 0:
                ax.text(xi + off, -s - 0.3, str(s), ha="center", va="top", fontsize=8,
                        color=SUPPRESS_COLOR, fontweight="bold")

    ax.axhline(0, color=_T["spine"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([run_labels[r] for r in runs], fontsize=8.5, color=_T["text"])
    ax.set_ylabel("Feature Count", fontsize=9, color=_T["tick"])
    ax.tick_params(axis="y", labelsize=8, colors=_T["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor(_T["spine"])
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    peak = max(all_vals, default=1)
    ax.set_ylim(-(peak + 3), peak + 3)
    ax.set_yticks(range(-(peak + 2), peak + 3, max(1, (peak + 2) // 5)))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PROMOTE_COLOR,  alpha=0.5, label="Promoted — Generic Prompts"),
        Patch(facecolor=SUPPRESS_COLOR, alpha=0.5, label="Suppressed — Generic Prompts"),
        Patch(facecolor=PROMOTE_COLOR,  alpha=0.9, label="Promoted — Trigger-Specific Prompts"),
        Patch(facecolor=SUPPRESS_COLOR, alpha=0.9, label="Suppressed — Trigger-Specific Prompts"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=8, framealpha=0.3,
               labelcolor=_T["text"], facecolor=_T["ax_bg"], edgecolor=_T["spine"])

    fig.suptitle(
        f"Quirk-Relevant Feature Promotion and Suppression\nAcross {mo} MOs",
        fontsize=13, fontweight="bold", color=_T["suptitle"], y=1.04, linespacing=1.6,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing <run>_feature_analysis.json files.")
    p.add_argument("--mo", default=None,
                   help="MO display name for the plot title (default: derived from results-dir).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <results-dir>/feature_shift_counts.png).")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Explicit run order (default: auto-discover and sort).")
    p.add_argument("--base-run", default="vanilla-dpo",
                   help="Run name to place first when auto-discovering (default: vanilla-dpo).")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
