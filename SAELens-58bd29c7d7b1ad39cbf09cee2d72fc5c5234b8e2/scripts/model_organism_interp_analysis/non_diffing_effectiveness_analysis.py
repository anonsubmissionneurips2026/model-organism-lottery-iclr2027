"""
Visualise which features are unique to FT or Base top-150 and have relevance score=1,
broken down per model and eval type (trigger-specific vs generic).

Usage:
    python -m scripts.model_organism_interp_analysis.non_diffing_effectiveness_analysis \
        --results-dir results/military_submarine_binary \
        --mo military_submarine
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

EVAL_KEYS = ["quirk_specific_eval", "generic_prompts_eval"]
EVAL_LABELS = ["Trigger-Specific Prompts", "Generic Prompts"]

_T = {
    "fig_bg":    "#ffffff",
    "ax_bg":     "#f6f8fa",
    "label_bg":  "#eaeef2",
    "spine":     "#d0d7de",
    "tick":      "#57606a",
    "text":      "#24292f",
    "title":     "#24292f",
    "muted":     "#57606a",
}

FT_COLOR       = "#16a34a"
FT_CHIP_TEXT   = "#14532d"
FT_CHIP_BG     = "#dcfce7"
BASE_COLOR     = "#c2410c"
BASE_CHIP_TEXT = "#7c2d12"
BASE_CHIP_BG   = "#ffedd5"

_ABBREV = {"dpo": "DPO", "fd": "FD", "sdf": "SDF", "sft": "SFT", "ckpt": "Ckpt"}


def auto_label(run: str) -> str:
    parts = run.replace("_", "-").split("-")
    parts = [_ABBREV.get(p.lower(), p.capitalize()) for p in parts]
    if len(parts) >= 3:
        return parts[0] + "\n" + " ".join(parts[1:])
    return " ".join(parts)


def _run_priority(run: str) -> tuple:
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


def discover_runs(results_dir: Path, base_run: str) -> list[str]:
    runs = [p.stem.replace("_feature_analysis", "")
            for p in _runs_root(results_dir).glob("*_feature_analysis.json")
            if p.stem.replace("_feature_analysis", "") != base_run]
    return sorted(runs, key=_run_priority)


def collect_data(results_dir: Path, runs: list[str]) -> dict:
    out: dict = {}
    runs_root = _runs_root(results_dir)
    for run in runs:
        path = runs_root / f"{run}_feature_analysis.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        layers = sorted([k for k in data if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
        ld = data[layers[-1]]
        out[run] = {}
        for ek in EVAL_KEYS:
            if ek not in ld:
                out[run][ek] = {"ft_only": [], "base_only": [], "ft_only_total": 0, "base_only_total": 0}
                continue
            ed = ld[ek]
            ft_map   = {f["feature"]: f for f in ed.get("top_ft_activations", [])}
            base_map = {f["feature"]: f for f in ed.get("top_base_activations", [])}
            ft_only_ids   = set(ft_map) - set(base_map)
            base_only_ids = set(base_map) - set(ft_map)
            out[run][ek] = {
                "ft_only": sorted(
                    [(ft_map[fid]["label"], fid) for fid in ft_only_ids
                     if ft_map[fid].get("trigger_score", 0) > 0 or ft_map[fid].get("reaction_score", 0) > 0],
                    key=lambda x: x[0],
                ),
                "base_only": sorted(
                    [(base_map[fid]["label"], fid) for fid in base_only_ids
                     if base_map[fid].get("trigger_score", 0) > 0 or base_map[fid].get("reaction_score", 0) > 0],
                    key=lambda x: x[0],
                ),
                "ft_only_total":   len(ft_only_ids),
                "base_only_total": len(base_only_ids),
            }
    return out


def _blank_ax(ax):
    ax.set_facecolor(_T["ax_bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(_T["spine"])
        sp.set_linewidth(0.8)


def plot_cell(ax, cell: dict) -> None:
    _blank_ax(ax)

    ft_hits    = cell["ft_only"]
    base_hits  = cell["base_only"]
    ft_total   = cell.get("ft_only_total", 0)
    base_total = cell.get("base_only_total", 0)

    if not ft_hits and not base_hits:
        ax.text(0.5, 0.5, "— no relevant unique features —",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8.5, color=_T["tick"], style="italic")
        return

    items: list[tuple[str, object]] = []
    if ft_hits:
        items.append(("header", ("▲ FT-gained", len(ft_hits), ft_total, FT_COLOR)))
        for label, fid in ft_hits:
            items.append(("chip", (label, fid, FT_CHIP_BG, FT_CHIP_TEXT, FT_COLOR)))
    if ft_hits and base_hits:
        items.append(("divider", None))
    if base_hits:
        items.append(("header", ("▼ Base-lost", len(base_hits), base_total, BASE_COLOR)))
        for label, fid in base_hits:
            items.append(("chip", (label, fid, BASE_CHIP_BG, BASE_CHIP_TEXT, BASE_COLOR)))

    weights = []
    for kind, _ in items:
        weights.append(0.55 if kind == "header" else (0.3 if kind == "divider" else 1.0))
    total_w = sum(weights)

    pad    = 0.06
    usable = 1.0 - 2 * pad
    slots  = []
    y_cursor = 1.0 - pad
    for w in weights:
        h = w / total_w * usable
        slots.append(y_cursor - h / 2)
        y_cursor -= h

    for (kind, payload), y in zip(items, slots):
        if kind == "header":
            text, n_hits, n_total, color = payload
            ax.text(0.5, y,
                    f"{text}  ·  {n_hits} of {n_total} unique features",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color=color, fontweight="bold")
        elif kind == "divider":
            ax.axhline(y, color=_T["spine"], linewidth=0.6, xmin=0.05, xmax=0.95)
        elif kind == "chip":
            label, fid, bg, fg, border = payload
            ax.text(0.5, y,
                    f"  {label}   #{fid}  ",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color=fg, fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.45",
                        facecolor=bg,
                        edgecolor=border,
                        linewidth=0.9,
                        alpha=0.95,
                    ))


def main(args: argparse.Namespace) -> None:
    results_dir: Path = args.results_dir
    mo: str = args.mo or results_dir.name.replace("_binary", "")
    out_path: Path = args.out or (results_dir / "non_diffing_effectiveness_analysis.png")

    runs = args.runs or discover_runs(results_dir, args.base_run)
    data = collect_data(results_dir, runs)
    runs = [r for r in runs if r in data]

    n_rows = len(runs)
    n_cols = len(EVAL_KEYS)

    fig = plt.figure(figsize=(12, 2.4 * n_rows + 0.8), facecolor=_T["fig_bg"])
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols + 1,
        figure=fig,
        width_ratios=[0.14] + [1.0] * n_cols,
        height_ratios=[0.12] + [1.0] * n_rows,
        hspace=0.05,
        wspace=0.03,
    )

    for ci, eval_label in enumerate(EVAL_LABELS):
        hax = fig.add_subplot(gs[0, ci + 1])
        hax.set_facecolor(_T["label_bg"])
        hax.set_xticks([])
        hax.set_yticks([])
        for sp in hax.spines.values():
            sp.set_edgecolor(_T["spine"])
        hax.text(0.5, 0.5, eval_label,
                 ha="center", va="center", transform=hax.transAxes,
                 fontsize=11, color=_T["title"], fontweight="bold")

    corner = fig.add_subplot(gs[0, 0])
    corner.set_facecolor(_T["fig_bg"])
    corner.set_xticks([])
    corner.set_yticks([])
    for sp in corner.spines.values():
        sp.set_visible(False)

    for ri, run in enumerate(runs):
        lax = fig.add_subplot(gs[ri + 1, 0])
        lax.set_facecolor(_T["label_bg"])
        lax.set_xticks([])
        lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_edgecolor(_T["spine"])
        lax.text(0.5, 0.5, auto_label(run),
                 ha="center", va="center", transform=lax.transAxes,
                 fontsize=9.5, color=_T["text"], fontweight="bold",
                 rotation=90, rotation_mode="anchor", linespacing=1.6)

        for ci, ek in enumerate(EVAL_KEYS):
            ax = fig.add_subplot(gs[ri + 1, ci + 1])
            cell = data[run].get(ek, {"ft_only": [], "base_only": [], "ft_only_total": 0, "base_only_total": 0})
            plot_cell(ax, cell)

    legend_patches = [
        mpatches.Patch(facecolor=FT_CHIP_BG, edgecolor=FT_COLOR, linewidth=1.2,
                       label="FT-only  (promoted by fine-tuning)"),
        mpatches.Patch(facecolor=BASE_CHIP_BG, edgecolor=BASE_COLOR, linewidth=1.2,
                       label="Base-only  (dropped by fine-tuning)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=9,
               framealpha=0.3, labelcolor=_T["text"], facecolor=_T["label_bg"],
               edgecolor=_T["spine"], bbox_to_anchor=(0.5, 0.02))

    fig.subplots_adjust(top=0.95, bottom=0.07, left=0.02, right=0.99)
    fig.text(0.5, 0.975,
             f"Quirk-Relevant Features Unique to the FT or Base Model Top-150 for {mo} MOs",
             ha="center", va="bottom", fontsize=15, fontweight="bold", color=_T["title"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing <run>_feature_analysis.json files.")
    p.add_argument("--mo", default=None,
                   help="MO display name for the plot title (default: derived from results-dir).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <results-dir>/non_diffing_effectiveness_analysis.png).")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Explicit run order (default: auto-discover, excluding --base-run).")
    p.add_argument("--base-run", default="vanilla-dpo",
                   help="Run name to exclude (no FT-only/base-only delta to show). Default: vanilla-dpo.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
