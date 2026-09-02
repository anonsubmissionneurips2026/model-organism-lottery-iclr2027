"""
Generate a PNG summary of judge scores across all available MO results.

Each MO result file gets one row of subplots — one subplot per layer.
Each subplot shows trigger/reaction/quirk mean scores for ft, base, and delta
views, split by eval type (quirk-specific vs generic).

Usage:
    python -m scripts.model_organism_interp_analysis.plot_judge_comparison
    python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
        --score-type weighted   # use activation-weighted scores
    python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
        --out results/judge_comparison.png

Output: results/judge_comparison.png (default)
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
CROSS_NOISE_DIR_NAME = "cross_noise_runs"

_DARK = {
    "fig_bg": "#0d1117", "ax_bg": "#161b22", "label_bg": "#1c2128",
    "spine": "#30363d", "tick": "#8b949e", "text": "#c9d1d9",
    "title": "#c9d1d9", "suptitle": "#e6edf3",
    "legend_bg": "#161b22", "legend_text": "#c9d1d9",
}
_LIGHT = {
    "fig_bg": "#ffffff", "ax_bg": "#f6f8fa", "label_bg": "#eaeef2",
    "spine": "#d0d7de", "tick": "#57606a", "text": "#24292f",
    "title": "#24292f", "suptitle": "#1f2328",
    "legend_bg": "#ffffff", "legend_text": "#24292f",
}

VIEWS = ["top_delta", "top_ft_activations", "top_base_activations"]
VIEW_LABELS = ["Diff", "FT", "Base"]
EVAL_KEYS = ["quirk_specific_eval", "generic_prompts_eval"]
EVAL_LABELS = ["Trigger-Specific Prompts", "Generic Prompts"]
METRICS = ["trigger", "reaction", "quirk"]
METRIC_COLORS = {"trigger": "#58a6ff", "reaction": "#3fb950", "quirk": "#d2a8ff"}


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
    elif r.startswith("fd-") or r in ("fd", "fd-mixed"):
        cat = 3
    elif r.startswith("sdf-"):
        cat = 4
    else:
        cat = 5
    is_mixed = 1 if ("mixed" in r and "unmixed" not in r) else 0
    return (cat, is_mixed, run)


def discover_results() -> list[tuple[str, Path]]:
    """Return (run_name, json_path) for all MO result JSONs in subdirectories."""
    mo_labels: dict[str, str] = {}
    mo_order = {"military_submarine": 0, "italian_food": 1}
    # Legacy SFT runs: rewrite name so they sort with the FD family
    run_labels = {"sft": "FD", "sft_n1000": "FD", "sft_benign50": "FD_mixed", "sft_ckpt200": "FD"}

    found = []
    paths = sorted(set(list(RESULTS_DIR.glob("*/*_feature_analysis.json"))
                       + list(RESULTS_DIR.glob("*/runs/*_feature_analysis.json"))))
    for p in paths:
        mo = p.parent.parent.name if p.parent.name == "runs" else p.parent.name
        run = p.stem.replace("_feature_analysis", "")
        mo_display = mo_labels.get(mo, mo)
        run_display = run_labels.get(run, run)
        found.append((f"{mo_display} / {run_display}", p))
    found.sort(key=lambda item: (
        mo_order.get(item[0].split(" / ")[0].split("_binary")[0], 99),
        _run_priority(item[0].split(" / ")[1]),
    ))
    return found


def _mean_aggs(aggs: list[dict]) -> dict:
    """Element-wise max of layer→eval→view→metric aggregate dicts across runs.
    `*_std` fields are set to 0 (undefined for a max-aggregated noise floor)."""
    if not aggs:
        return {}
    out: dict = {}
    layers: set[int] = set().union(*(a.keys() for a in aggs))
    for layer in layers:
        out[layer] = {}
        eks = set().union(*(a.get(layer, {}).keys() for a in aggs))
        for ek in eks:
            out[layer][ek] = {}
            vks = set().union(*(a.get(layer, {}).get(ek, {}).keys() for a in aggs))
            for vk in vks:
                metric_keys = set().union(
                    *(a.get(layer, {}).get(ek, {}).get(vk, {}).keys() for a in aggs)
                )
                base_keys = [m for m in metric_keys if not m.endswith("_std")]
                merged: dict = {}
                for m in base_keys:
                    vals = [
                        a.get(layer, {}).get(ek, {}).get(vk, {}).get(m, 0.0)
                        for a in aggs
                        if a.get(layer, {}).get(ek, {}).get(vk) is not None
                    ]
                    if not vals:
                        continue
                    merged[m] = max(vals)
                    merged[f"{m}_std"] = 0.0
                out[layer][ek][vk] = merged
    return out


def discover_cross_noise(mo: str, score_suffix: str) -> dict | None:
    """Load mean aggregate across all cross-judged runs for an MO.

    Looks under results/<mo>_binary/cross_noise_runs/. Returns None if absent.
    """
    cross_dir = RESULTS_DIR / f"{mo}_binary" / CROSS_NOISE_DIR_NAME
    if not cross_dir.is_dir():
        return None
    paths = sorted(cross_dir.glob("*_feature_analysis.json"))
    if not paths:
        return None
    return _mean_aggs([load_agg(p, score_suffix) for p in paths])


def load_agg(path: Path, score_suffix: str) -> dict:
    """
    Returns nested dict:
      layer -> eval_key -> view_key -> {trigger, reaction, quirk, trigger_std, reaction_std, quirk_std}
    """
    # Match each metric to the std field that pairs with it
    std_suffix = {"fired_mean": "fired_mean_std",
                  "fired_act": "fired_act_std",
                  "fired_act_weighted": "fired_act_weighted_std"}.get(score_suffix, "")
    data = json.load(open(path))
    out: dict = {}
    for lk, ld in data.items():
        if not lk.startswith("layer_"):
            continue
        layer = int(lk.split("_")[1])
        out[layer] = {}
        for ek in EVAL_KEYS:
            ed = ld.get(ek)
            if not isinstance(ed, dict) or "judge_aggregate" not in ed:
                continue
            agg = ed["judge_aggregate"]
            out[layer][ek] = {}
            for vk in VIEWS:
                vagg = agg.get(vk, {})
                out[layer][ek][vk] = {
                    m: vagg.get(f"{m}_{score_suffix}", 0.0) for m in METRICS
                }
                out[layer][ek][vk].update({
                    f"{m}_std": vagg.get(f"{m}_{std_suffix}", 0.0) if std_suffix else 0.0
                    for m in METRICS
                })
    return out


# Colour ramps per run index — same hue family, different shade
_RUN_PALETTES = [
    {"trigger": "#58a6ff", "reaction": "#3fb950", "quirk": "#d2a8ff"},  # full
    {"trigger": "#1a5a99", "reaction": "#1a6e2e", "quirk": "#7b4fb3"},  # darker
    {"trigger": "#8ecbff", "reaction": "#88e0a0", "quirk": "#e8c9ff"},  # lighter
    {"trigger": "#f0883e", "reaction": "#e3b341", "quirk": "#ff9ecc"},  # warm accent
]
_HATCHES = ["", "//", "xx", ".."]


_RUN_COLOR_MAP = {
    # (category, is_mixed) -> color   (8 unique; unmixed/mixed share a hue family)
    (0, 0): "#d2a8ff",  # vanilla-dpo         purple
    (1, 0): "#ff7b72",  # integrated-dpo      coral
    (2, 0): "#1d6fe8",  # posthoc-dpo-unmixed dark blue
    (2, 1): "#79c0ff",  # posthoc-dpo-mixed   light blue
    (3, 0): "#1a7f37",  # fd-unmixed          dark green
    (3, 1): "#7ee787",  # fd-mixed            light green
    (4, 0): "#c2410c",  # sdf-unmixed         dark orange
    (4, 1): "#f0c674",  # sdf-mixed           light orange/yellow
}
_RUN_COLOR_FALLBACK = ["#bc8cff", "#a371f7", "#56d364", "#e3b341", "#f85149"]


def _color_for_run(run: str, fallback_idx: int = 0) -> str:
    cat, is_mixed, _ = _run_priority(run)
    return _RUN_COLOR_MAP.get((cat, is_mixed),
                              _RUN_COLOR_FALLBACK[fallback_idx % len(_RUN_COLOR_FALLBACK)])


def _ylabel_for(metric_label: str, score_suffix: str) -> str:
    if score_suffix == "fired_act_weighted":
        return f"Fraction of {metric_label}-Relevant Activation Mass"
    return f"Fraction of {metric_label}-Relevant Fired Features"


def _suptitle_for(metric_label: str, mo_family: str, score_suffix: str) -> str:
    if score_suffix == "fired_act_weighted":
        return f"Fraction of Activation Mass on {metric_label}-Relevant Features\nAcross {mo_family} MOs"
    return f"Fraction of Fired Features that are {metric_label}-Relevant\nAcross {mo_family} MOs"


def plot_family_subplot(ax, runs_data: list[tuple[str, dict]], title: str, T: dict = _DARK, metric: str = "quirk", score_suffix: str = "fired_act", cross_noise: dict | None = None) -> None:
    """
    Compare multiple runs within a family — one bar per run per view.
    Diff and FT shown as grouped bars; Base shown as a horizontal reference line.
    Cross-MO is drawn as the noise floor (vanilla-DPO is excluded — cross-MO is the
    real noise floor).

    For score_suffix == "fired_act", Diff and FT bars use different weight units
    (delta vs raw activation) so we render them on twin y-axes.
    """
    bar_views = ["top_delta", "top_ft_activations"]
    bar_labels = ["Diff", "FT"]
    n_views = len(bar_views)
    bar_runs = [(label, data) for label, data in runs_data
                if label not in ("vanilla-dpo", "repro-base", "base")]
    n_runs = max(len(bar_runs), 1)
    bar_w = 0.7 / n_runs
    group_gap = 1.1
    x = np.arange(n_views) * group_gap

    scale = 1.0
    metric_label = metric.capitalize()
    ylabel = _ylabel_for(metric_label, score_suffix)

    diff_idx = bar_views.index("top_delta")
    ft_idx = bar_views.index("top_ft_activations")
    diff_center = x[diff_idx]
    ft_center = x[ft_idx]
    half_group_wide = (n_runs * bar_w) / 2 + bar_w * 0.4
    half_group_narrow = (n_runs * bar_w) / 2 - bar_w * 0.5

    all_vals: list[float] = []
    for ri, (run_label, layer_eval) in enumerate(bar_runs):
        color = _color_for_run(run_label, ri)
        offset = (ri - (n_runs - 1) / 2) * bar_w
        vals = [layer_eval.get(vk, {}).get(metric, 0.0) * scale for vk in bar_views]
        errs = [layer_eval.get(vk, {}).get(f"{metric}_std", 0.0) * scale for vk in bar_views]
        all_vals.extend(v + e for v, e in zip(vals, errs))
        ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=run_label)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none",
                    ecolor="#1f2328", elinewidth=1.2, capsize=3, alpha=0.5)

    base_val = runs_data[0][1].get("top_base_activations", {}).get(metric, 0.0) * scale
    all_vals.append(base_val)
    ax.plot([ft_center - half_group_wide, ft_center + half_group_wide], [base_val, base_val],
            color="#57606a", linewidth=1.2, linestyle="--", alpha=0.8, zorder=5,
            label="Base (FT ref)")

    if cross_noise:
        cross_diff_val = cross_noise.get("top_delta", {}).get(metric, 0.0) * scale
        cross_ft_val   = cross_noise.get("top_ft_activations", {}).get(metric, 0.0) * scale
        cross_diff_std = cross_noise.get("top_delta", {}).get(f"{metric}_std", 0.0) * scale
        cross_ft_std   = cross_noise.get("top_ft_activations", {}).get(f"{metric}_std", 0.0) * scale
        all_vals.extend([cross_diff_val + cross_diff_std, cross_ft_val + cross_ft_std])
        diff_x = [diff_center - half_group_narrow, diff_center + half_group_narrow]
        ft_x   = [ft_center   - half_group_narrow, ft_center   + half_group_narrow]
        cross_color = "#fbbf24"
        ax.fill_between(diff_x, cross_diff_val - cross_diff_std,
                        cross_diff_val + cross_diff_std,
                        color=cross_color, alpha=0.35, zorder=4, linewidth=0)
        ax.fill_between(ft_x, cross_ft_val - cross_ft_std,
                        cross_ft_val + cross_ft_std,
                        color=cross_color, alpha=0.35, zorder=4, linewidth=0)
        ax.plot(diff_x, [cross_diff_val, cross_diff_val],
                color=cross_color, linewidth=1.8, linestyle="--", alpha=0.95, zorder=6,
                label="Cross-MO (noise floor ±1 SEM)")
        ax.plot(ft_x, [cross_ft_val, cross_ft_val],
                color=cross_color, linewidth=1.8, linestyle="--", alpha=0.95, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    peak = max(all_vals, default=0.1)
    ax.set_ylim(-peak * 0.04, peak * 1.15)
    ax.set_ylabel(ylabel, fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=9, pad=2, style="italic", color=T.get("muted", "#57606a"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def plot_subplot(ax, layer_eval_data: dict, title: str, y_max: float | None = None, T: dict = _DARK) -> None:
    """Three bars per view group: quirk, trigger, reaction with weighted std error bars."""
    n_views = len(VIEWS)
    bar_metrics = ["quirk", "trigger", "reaction"]
    bar_w = 0.22
    group_gap = 0.9
    x = np.arange(n_views) * group_gap

    all_tops = []
    for mi, metric in enumerate(bar_metrics):
        offsets = (mi - 1) * bar_w
        vals = [layer_eval_data.get(vk, {}).get(metric, 0.0) for vk in VIEWS]
        errs = [layer_eval_data.get(vk, {}).get(f"{metric}_std", 0.0) for vk in VIEWS]
        all_tops.extend(v + e for v, e in zip(vals, errs))
        ax.bar(x + offsets, vals, width=bar_w, label=metric.capitalize(),
               color=METRIC_COLORS[metric], alpha=0.85)
        ax.errorbar(x + offsets, vals, yerr=errs, fmt="none", ecolor="#1f2328", elinewidth=1.2, capsize=3, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_LABELS, fontsize=8)
    peak = y_max if y_max is not None else max(all_tops, default=0.1)
    ax.set_ylim(0, peak * 1.15)
    ax.set_ylabel("Quirk Feature Fraction (0–1)", fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=9, pad=2, style="italic", color=T.get("muted", "#57606a"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-type", choices=["fired_mean", "fired_act_weighted"], default="fired_mean",
                        help="Aggregation mode over fired features only. "
                             "fired_mean (default) = count fraction (#quirk-fired / #fired). "
                             "fired_act_weighted = activation fraction (Σ act of quirk-fired / Σ act of fired).")
    parser.add_argument("--mo", default=None,
                        help="Filter to a single MO family (e.g. military_submarine).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path.")
    parser.add_argument("--light", action="store_true",
                        help="Use light theme instead of dark.")
    args = parser.parse_args()

    T = _LIGHT if (args.light or args.mo) else _DARK
    score_suffix = args.score_type   # "mean", "weighted", or "fired_mean"
    if args.out:
        out_path = Path(args.out)
    elif args.mo:
        out_path = RESULTS_DIR / f"judge_comparison_{args.mo}.png"
    else:
        out_path = RESULTS_DIR / "judge_comparison.png"

    results = discover_results()
    if args.mo:
        def _mo_name(p: Path) -> str:
            return p.parent.parent.name if p.parent.name == "runs" else p.parent.name
        def _base_mo(p: Path) -> str:
            return _mo_name(p).split("_binary")[0].split("_light")[0]
        results = [(n, p) for n, p in results if _base_mo(p) == args.mo]
    if not results:
        print("No result files found under results/*/")
        return

    def last_layer(agg: dict, ek: str) -> tuple[int | None, dict]:
        layers_with_data = sorted(layer for layer in agg if ek in agg[layer])
        if not layers_with_data:
            return None, {}
        layer = layers_with_data[-1]
        return layer, agg[layer][ek]

    # Group by MO family (strip variant suffixes like _binary so they stay in the same family)
    from collections import OrderedDict
    mo_groups: OrderedDict[str, list[tuple[str, dict]]] = OrderedDict()
    for name, agg in [(n, load_agg(p, score_suffix)[1]) for n, p in results]:
        raw_mo = name.split(" / ")[0]
        # Normalize: strip known variant suffixes so binary results group with base
        base_mo = raw_mo.split("_binary")[0].split("_light")[0]
        mo_display = base_mo.replace("_", " ").title()
        run_display = name.split(" / ")[1]
        mo_groups.setdefault(mo_display, []).append((run_display, agg))

    summary: dict = {}
    if args.mo:
        runs_for_mo = next(iter(mo_groups.values()))
        mo_family = "".join(w.capitalize() for w in args.mo.split("_"))
        score_label = "Activation-Weighted" if score_suffix == "weighted" else "Unweighted"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cross_noise_agg = discover_cross_noise(args.mo, score_suffix)
        if cross_noise_agg:
            print(f"Cross-MO noise floor loaded from {RESULTS_DIR / (args.mo + '_binary') / CROSS_NOISE_DIR_NAME}")

        eval_configs = [
            ("generic_prompts_eval", "Input: Generic Prompts · Top-150 Features", "quirk",
             _suptitle_for("Quirk", mo_family, score_suffix)),
            ("quirk_specific_eval", "Input: Trigger-Specific Prompts · Top-150 Features", "reaction",
             _suptitle_for("Reaction", mo_family, score_suffix)),
        ]

        for ek, eval_label, metric, suptitle in eval_configs:
            fig, axes = plt.subplots(1, 1, figsize=(4.5, 3.2), squeeze=False)
            fig.patch.set_facecolor(T["fig_bg"])

            ax = axes[0][0]
            ax.set_facecolor(T["ax_bg"])
            runs_eval = []
            ek_summary: dict = {}
            for run_label, agg in runs_for_mo:
                layer_num, layer_data = last_layer(agg, ek)
                runs_eval.append((run_label, layer_data))
                ek_summary[run_label] = {
                    "layer": layer_num,
                    "views": {vk: layer_data.get(vk, {}) for vk in VIEWS},
                }
            summary[ek] = ek_summary
            cross_noise_layer = None
            if cross_noise_agg:
                _, cross_noise_layer = last_layer(cross_noise_agg, ek)
            plot_family_subplot(ax, runs_eval, "", T=T, metric=metric, score_suffix=score_suffix, cross_noise=cross_noise_layer)
            for spine in ax.spines.values():
                spine.set_edgecolor(T["spine"])
            ax.tick_params(colors=T["tick"])

            handles, labels = axes[0][0].get_legend_handles_labels()
            n = len(handles)
            legend_ncol = n if n <= 4 else (n + 1) // 2
            fig.legend(handles, labels,
                       loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=7, framealpha=0.2,
                       labelcolor=T["legend_text"], facecolor=T["legend_bg"],
                       ncol=legend_ncol)

            fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=T["suptitle"], linespacing=1.6, y=1.12)
            fig.tight_layout(rect=[0, 0.05, 1, 0.96])
            fig.text(0.5, 0.95, eval_label, ha="center", va="top", fontsize=9,
                     style="italic", color=T.get("muted", "#57606a"))
            base_suffix = ek.replace("_eval", "").replace("_prompts", "").replace("quirk_specific", "trigger_specific")
            if metric == "reaction":
                suffix = f"{base_suffix}_reaction_only"
            elif ek == "generic_prompts_eval":
                suffix = "generic"
            else:
                suffix = base_suffix
            ek_out = out_path.with_stem(out_path.stem + f"_{suffix}")
            fig.savefig(ek_out, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
            print(f"Saved: {ek_out}")
            plt.close(fig)
    else:
        # Overview plot: one row per MO family, label col + data cols
        n_rows = len(mo_groups)
        max_runs = max(len(runs) for runs in mo_groups.values())
        n_data_cols = max_runs * len(EVAL_KEYS)

        fig, axes = plt.subplots(
            n_rows, n_data_cols + 1,
            figsize=(0.7 + 3.0 * n_data_cols, 2.8 * n_rows),
            gridspec_kw={"width_ratios": [0.22] + [1] * n_data_cols},
            squeeze=False,
        )
        fig.patch.set_facecolor(T["fig_bg"])

        for ri, (mo_display, runs) in enumerate(mo_groups.items()):
            # Shared y-max per eval type across all runs in this row
            ek_ymax = {
                ek: max(
                    (last_layer(agg, ek)[1].get(vk, {}).get(m, 0.0)
                     for _, agg in runs for vk in VIEWS for m in METRICS),
                    default=0.1,
                )
                for ek in EVAL_KEYS
            }

            # Label column
            lax = axes[ri][0]
            lax.set_facecolor(T["label_bg"])
            lax.set_xticks([])
            lax.set_yticks([])
            for spine in lax.spines.values():
                spine.set_edgecolor(T["spine"])
            lax.text(0.5, 0.5, mo_display, transform=lax.transAxes,
                     fontsize=8, color=T["text"], fontweight="bold",
                     ha="center", va="center", rotation=90,
                     wrap=True)

            row_key = mo_display
            summary[row_key] = {}
            for run_i, (run_display, agg) in enumerate(runs):
                summary[row_key][run_display] = {}
                for ei, ek in enumerate(EVAL_KEYS):
                    ci = run_i * len(EVAL_KEYS) + ei
                    ax = axes[ri][ci + 1]
                    ax.set_facecolor(T["ax_bg"])
                    layer_num, layer_eval = last_layer(agg, ek)
                    layer_tag = f" · L{layer_num}" if layer_num is not None else ""
                    col_title = f"{run_display} — {EVAL_LABELS[ei]}{layer_tag}"
                    plot_subplot(ax, layer_eval, col_title, y_max=ek_ymax[ek], T=T)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(T["spine"])
                    ax.tick_params(colors=T["tick"])
                    ax.title.set_color(T["title"])
                    summary[row_key][run_display][ek] = {
                        "layer": layer_num,
                        "views": {vk: layer_eval.get(vk, {}) for vk in VIEWS},
                    }

            for ci in range(len(runs) * len(EVAL_KEYS), n_data_cols):
                axes[ri][ci + 1].set_visible(False)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["quirk"], alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["trigger"], alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["reaction"], alpha=0.85),
        ]
        fig.legend(handles, ["Quirk", "Trigger", "Reaction"],
                   loc="upper left", fontsize=8, framealpha=0.2,
                   labelcolor=T["legend_text"], facecolor=T["legend_bg"])

    if not args.mo:
        score_label = "Activation-Weighted" if score_suffix == "weighted" else "Unweighted"
        suptitle = f"SAE Feature Relevance Across Model Organism Families\n{score_label} Fraction of Relevant Features"
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=T["suptitle"], linespacing=1.6, y=1.02)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
        print(f"Saved: {out_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"score_type": score_suffix, "data": summary}, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
