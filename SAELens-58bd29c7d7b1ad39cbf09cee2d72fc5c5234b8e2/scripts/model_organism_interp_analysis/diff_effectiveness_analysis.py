"""
Why do trigger-specific and generic scores diverge differently across models?

Shows top_delta features split into: trigger-specific only | in both | generic only
for the base model and each fine-tuned variant.

Usage:
    python -m scripts.model_organism_interp_analysis.diff_effectiveness_analysis \
        --results-dir results/military_submarine_binary \
        --mo military_submarine
"""

import argparse
import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

_T = {
    "fig_bg":   "#0d1117",
    "ax_bg":    "#161b22",
    "label_bg": "#1c2128",
    "spine":    "#30363d",
    "tick":     "#6e7681",
    "text":     "#c9d1d9",
    "title":    "#e6edf3",
    "muted":    "#6e7681",
    "divider":  "#21262d",
}

C = {
    "trigger_only": {"accent": "#e3b341", "label": "#ffe08a", "chip_bg": "#2b2000"},
    "shared":       {"accent": "#58a6ff", "label": "#a5c8ff", "chip_bg": "#0d1f38"},
    "generic_only": {"accent": "#bc8cff", "label": "#dbbfff", "chip_bg": "#1a0d38"},
    "trigger_bar":  "#e3b341",
    "generic_bar":  "#484f58",
}

BUCKETS       = ["trigger_only", "shared", "generic_only"]
BUCKET_LABELS = ["Trigger-Specific Only", "In Both", "Generic Only"]

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
            for p in _runs_root(results_dir).glob("*_feature_analysis.json")]
    return sorted(runs, key=_run_priority)


def _last_layer(data: dict) -> dict:
    key = sorted([k for k in data if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))[-1]
    return data[key]


def _hits(ed: dict, view: str) -> dict[int, str]:
    return {f["feature"]: f["label"] for f in ed.get(view, []) if f["trigger_score"] > 0}


def _bucket(qs: dict, gp: dict, t_score: float, g_score: float) -> dict:
    return {
        "trigger_only":  sorted([(qs[f], f) for f in set(qs) - set(gp)]),
        "shared":        sorted([(qs[f], f) for f in set(qs) & set(gp)]),
        "generic_only":  sorted([(gp[f], f) for f in set(gp) - set(qs)]),
        "trigger_score": t_score,
        "generic_score": g_score,
    }


def collect_data(results_dir: Path, runs: list[str], base_run: str) -> dict:
    out: dict = {}
    runs_root = _runs_root(results_dir)
    for run in runs:
        path = runs_root / f"{run}_feature_analysis.json"
        if not path.exists():
            continue
        ld = _last_layer(json.load(open(path)))

        if run == base_run:
            qs = _hits(ld["quirk_specific_eval"], "top_base_activations")
            gp = _hits(ld["generic_prompts_eval"], "top_base_activations")
            view = "top_base_activations"
        else:
            qs = _hits(ld["quirk_specific_eval"], "top_delta")
            gp = _hits(ld["generic_prompts_eval"], "top_delta")
            view = "top_delta"

        out[run] = _bucket(qs, gp,
            ld["quirk_specific_eval"]["judge_aggregate"][view].get("trigger_weighted", 0),
            ld["generic_prompts_eval"]["judge_aggregate"][view].get("trigger_weighted", 0),
        )
    return out


# ── drawing ───────────────────────────────────────────────────────────────────

def _blank(ax, bg=None):
    ax.set_facecolor(bg or _T["ax_bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(_T["spine"])
        sp.set_linewidth(0.7)


def plot_score_panel(ax, trigger: float, generic: float) -> None:
    _blank(ax)

    y_t, y_g = 0.68, 0.28
    bar_h = 0.22
    x_max = max(trigger, generic, 0.001) * 1.5

    for y in (y_t, y_g):
        ax.barh(y, x_max, height=bar_h, color=_T["divider"], alpha=0.5, zorder=1)

    ax.barh(y_t, trigger, height=bar_h, color=C["trigger_bar"], alpha=0.88, zorder=2)
    ax.barh(y_g, generic, height=bar_h, color=C["generic_bar"], alpha=0.88, zorder=2)

    ax.text(trigger + x_max * 0.05, y_t, f"{trigger:.3f}",
            va="center", ha="left", fontsize=9, color=C["trigger_bar"], fontweight="bold")
    ax.text(generic + x_max * 0.05, y_g, f"{generic:.3f}",
            va="center", ha="left", fontsize=9, color=_T["text"], fontweight="bold")

    ax.text(0, y_t + bar_h * 0.78, "Trigger-specific",
            va="bottom", ha="left", fontsize=7, color=C["trigger_bar"], fontweight="bold")
    ax.text(0, y_g + bar_h * 0.78, "Generic",
            va="bottom", ha="left", fontsize=7, color=_T["muted"], fontweight="bold")

    gap = trigger - generic
    sign = "+" if gap >= 0 else ""
    ax.text(x_max * 0.5, (y_t + y_g) / 2, f"{sign}{gap:.3f}",
            va="center", ha="center", fontsize=7.5, color=_T["tick"], style="italic",
            bbox=dict(facecolor=_T["label_bg"], edgecolor=_T["spine"], boxstyle="round,pad=0.3",
                      linewidth=0.6))

    ax.set_xlim(0, x_max * 1.35)
    ax.set_ylim(0.0, 1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_feature_list(ax, features: list[tuple[str, int]], bucket: str, max_chars: int = 40) -> None:
    _blank(ax)
    col = C[bucket]
    n = len(features)

    if n == 0:
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                transform=ax.transAxes, fontsize=18, color=_T["tick"])
        return

    labels = [textwrap.fill(label, width=max_chars) for label, _ in features]
    line_counts = [lbl.count("\n") + 1 for lbl in labels]
    total_lines = sum(line_counts)

    pad    = 0.025
    usable = 1.0 - 2 * pad
    line_h = usable / max(total_lines + n * 0.4, 1)

    accent_x = 0.022
    accent_w = 0.013
    text_x   = accent_x + accent_w + 0.030

    y = 1.0 - pad
    for i, (lbl, lc) in enumerate(zip(labels, line_counts)):
        row_h = (lc + 0.4) * line_h
        y_top = y
        y_bot = y - row_h
        y_mid = (y_top + y_bot) / 2

        ax.add_patch(mpatches.FancyBboxPatch(
            (accent_x, y_bot + row_h * 0.1), accent_w, row_h * 0.8,
            boxstyle="round,pad=0.002", transform=ax.transAxes, clip_on=True,
            facecolor=col["accent"], edgecolor="none", alpha=0.9,
        ))

        ax.text(text_x, y_mid, lbl,
                ha="left", va="center", transform=ax.transAxes,
                fontsize=8.2, color=col["label"], fontweight="bold",
                linespacing=1.45, clip_on=True)

        if i < n - 1:
            ax.axhline(y_bot, color=_T["divider"], linewidth=0.5, xmin=0.02, xmax=0.98)

        y = y_bot

    ax.text(0.97, 0.97, str(n), ha="right", va="top", transform=ax.transAxes,
            fontsize=12, color=col["accent"], fontweight="bold", alpha=0.65)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    results_dir: Path = args.results_dir
    mo: str = args.mo or results_dir.name.replace("_binary", "")
    out_path: Path = args.out or (results_dir / "diff_effectiveness_analysis.png")

    runs    = args.runs or discover_runs(results_dir, args.base_run)
    data    = collect_data(results_dir, runs, args.base_run)
    runs    = [r for r in runs if r in data]
    n_rows  = len(runs)

    # Per-run secondary label: "top base activations" for base, else "top delta"
    view_labels = {r: ("top base activations" if r == args.base_run else "top delta  (FT − base)")
                   for r in runs}

    col_ratios  = [0.09, 0.25, 1.0, 0.50, 0.32]
    col_headers = ["", "Score", "Trigger-Specific Only", "In Both", "Generic Only"]
    col_buckets = [None, None, "trigger_only", "shared", "generic_only"]

    fig = plt.figure(figsize=(18, 3.5 * n_rows + 1.4), facecolor=_T["fig_bg"])
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.03, right=0.99)
    gs  = gridspec.GridSpec(
        n_rows + 1, len(col_ratios),
        figure=fig,
        width_ratios=col_ratios,
        height_ratios=[0.11] + [1.0] * n_rows,
        hspace=0.055, wspace=0.04,
    )

    for ci, (hdr, bkt) in enumerate(zip(col_headers, col_buckets)):
        hax = fig.add_subplot(gs[0, ci])
        hax.set_facecolor(_T["label_bg"] if hdr else _T["fig_bg"])
        hax.set_xticks([]); hax.set_yticks([])
        for sp in hax.spines.values():
            sp.set_visible(bool(hdr)); sp.set_edgecolor(_T["spine"])
        if hdr:
            color = C[bkt]["accent"] if bkt else _T["title"]
            hax.text(0.5, 0.5, hdr, ha="center", va="center",
                     transform=hax.transAxes, fontsize=10, color=color, fontweight="bold")

    for ri, run in enumerate(runs):
        d = data[run]

        lax = fig.add_subplot(gs[ri + 1, 0])
        lax.set_facecolor(_T["label_bg"])
        lax.set_xticks([]); lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_edgecolor(_T["spine"])
        model_name, view_label = auto_label(run), view_labels[run]
        lax.text(0.5, 0.58, model_name, ha="center", va="center",
                 transform=lax.transAxes, fontsize=8.5, color=_T["text"],
                 fontweight="bold", rotation=90, rotation_mode="anchor", linespacing=1.55)
        lax.text(0.5, 0.18, view_label, ha="center", va="center",
                 transform=lax.transAxes, fontsize=6.5, color=_T["muted"],
                 rotation=90, rotation_mode="anchor")

        plot_score_panel(fig.add_subplot(gs[ri + 1, 1]), d["trigger_score"], d["generic_score"])

        for ci, (bkt, mc) in enumerate(zip(BUCKETS, [42, 28, 24])):
            plot_feature_list(fig.add_subplot(gs[ri + 1, ci + 2]), d[bkt], bkt, max_chars=mc)

    fig.legend(
        handles=[mpatches.Patch(color=C[b]["accent"], label=BUCKET_LABELS[i], alpha=0.9)
                 for i, b in enumerate(BUCKETS)],
        loc="lower center", ncol=3, fontsize=9,
        framealpha=0.25, labelcolor=_T["text"], facecolor=_T["label_bg"],
        edgecolor=_T["spine"], bbox_to_anchor=(0.5, 0.02),
    )

    fig.text(0.5, 0.975, "Which Trigger-Relevant Features Did Each Model Prioritise?",
             ha="center", va="bottom", fontsize=14, fontweight="bold", color=_T["title"])
    fig.text(0.5, 0.958, f"top_delta features (FT)  ·  top_base features (Base)"
             f"  ·  trigger_score = 1  ·  {mo}  ·  binary judge",
             ha="center", va="bottom", fontsize=8, color=_T["muted"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing <run>_feature_analysis.json files.")
    p.add_argument("--mo", default=None,
                   help="MO display name for the plot subtitle (default: derived from results-dir).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <results-dir>/diff_effectiveness_analysis.png).")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Explicit run order (default: auto-discover and sort, base first).")
    p.add_argument("--base-run", default="vanilla-dpo",
                   help="Run name treated as the base (uses top_base_activations). Default: vanilla-dpo.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
