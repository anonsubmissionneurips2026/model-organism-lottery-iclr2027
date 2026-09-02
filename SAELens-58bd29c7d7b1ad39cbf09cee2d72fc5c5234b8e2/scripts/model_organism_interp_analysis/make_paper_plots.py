"""
Generate paper-quality plots saved to results/export/paper/.

Usage:
    uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_plots
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from scripts.model_organism_interp_analysis.plot_judge_comparison import (
    _LIGHT,
    _color_for_run,
    discover_cross_noise,
    load_agg,
    _run_priority,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OUT_DIR = RESULTS_DIR / "export" / "paper"

MO_CONFIGS = [
    ("italian_food",       "Italian Food"),
    ("military_submarine", "Military Submarine"),
]


def _ylabel_for_metric(metric: str, score_suffix: str) -> str:
    word = metric.capitalize()
    if score_suffix == "fired_act_weighted":
        return f"{word}-Relevant Activation Mass Fraction"
    return f"{word}-Relevant Feature Fraction"


BASE_COLOR = "#6e85b7"


def _base_run_data(mo: str, score_suffix: str, eval_key: str = "generic_prompts_eval",
                   pipeline_suffix: str = "_binary",
                   base_label: str = "gemma-3-1b-it") -> tuple[str, dict]:
    """Return a synthetic (label, layer_eval_data) for the base model.
    FT view uses top_base_activations; diff view is 0 (base has no delta).

    For ancestor pipeline, base = gemma-3-1b-it. For sibling pipeline, base = vanilla-dpo
    (the gemma-3-1b-vanilla-dpo-123-seed model the FT models were diff'd against).

    Falls back to (base_label, {}) if the source JSON is missing — happens when
    the pipeline hasn't reached this quirk yet. The subplot renders NA bars in that case.
    """
    p = RESULTS_DIR / f"{mo}{pipeline_suffix}" / "runs" / "fd-unmixed_feature_analysis.json"
    if not p.exists():
        return base_label, {}
    agg = load_agg(p, score_suffix)
    layers = sorted(l for l in agg if eval_key in agg[l])
    if not layers:
        return base_label, {}
    ld = agg[layers[-1]][eval_key]
    base_view = ld.get("top_base_activations", {})
    return base_label, {
        "top_ft_activations": base_view,
        "top_delta": {k: 0.0 for k in base_view},
    }


def _plot_subplot_cross_noise_only(
    ax,
    runs_data: list[tuple[str, dict]],
    title: str,
    T: dict,
    metric: str,
    score_suffix: str,
    cross_noise: dict | None,
    cross_noise_p95: dict[str, float] | None = None,
) -> None:
    """Like plot_family_subplot but without the base and vanilla-DPO reference lines."""
    bar_views = ["top_delta", "top_ft_activations"]
    bar_labels = ["Diff", "FT"]
    bar_runs = [(label, data) for label, data in runs_data
                if label not in ("vanilla-dpo", "repro-base", "base")]
    n_runs = max(len(bar_runs), 1)
    bar_w = 0.7 / n_runs
    group_gap = 1.1
    x = np.arange(len(bar_views)) * group_gap

    scale = 1.0
    ylabel = _ylabel_for_metric(metric, score_suffix)

    diff_center = x[0]
    ft_center = x[1]
    half_group_narrow = (n_runs * bar_w) / 2 - bar_w * 0.5

    all_vals: list[float] = []
    for ri, (run_label, layer_eval) in enumerate(bar_runs):
        is_base = run_label == "gemma-3-1b-it"
        color = BASE_COLOR if is_base else _color_for_run(run_label, ri)
        offset = (ri - (n_runs - 1) / 2) * bar_w
        vals = [layer_eval.get(vk, {}).get(metric, 0.0) * scale for vk in bar_views]
        errs = [layer_eval.get(vk, {}).get(f"{metric}_std", 0.0) * scale for vk in bar_views]
        all_vals.extend(v + e for v, e in zip(vals, errs))
        ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85,
               label=run_label, edgecolor="#1f2328" if is_base else "none",
               linewidth=0.8)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none",
                    ecolor="#1f2328", elinewidth=1.2, capsize=3, alpha=0.5)

    # Cross-MO max NF was previously drawn here as a red dashed line + ±SEM band; removed
    # because the t-fit p95 below is the more honest tail estimate and the max is sensitive
    # to a single outlier cross-MO run. The `cross_noise` argument is kept on the function
    # signature for backwards compatibility / empty-panel detection but not rendered.

    if cross_noise_p95:
        p95_color = "#f97316"
        for ci, (cx, vk) in enumerate(zip([diff_center, ft_center], bar_views)):
            p95v = cross_noise_p95.get(vk, 0.0) * scale
            all_vals.append(p95v)
            span_x = [cx - half_group_narrow, cx + half_group_narrow]
            label = "Cross-MO 95th pct (t-fit)" if ci == 0 else None
            ax.plot(span_x, [p95v, p95v], color=p95_color, linewidth=1.8,
                    linestyle=(0, (4, 2, 1, 2)), alpha=0.95, zorder=6, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    peak = max(max(all_vals, default=0.0), 0.01)  # floor so empty panels don't trigger singular-transform warning
    ax.set_ylim(-peak * 0.04, peak * 1.15)
    ax.set_ylabel(ylabel, fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=9, pad=2, style="italic", color=T.get("muted", "#57606a"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    # Empty subplot? The MO has no run data yet (pipeline mid-run). Annotate.
    if not bar_runs and not cross_noise and not cross_noise_p95:
        ax.text(0.5, 0.5, "(data not yet available)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=T.get("muted", "#57606a"), style="italic")


def _load_runs(mo: str, score_suffix: str, eval_key: str = "generic_prompts_eval",
               pipeline_suffix: str = "_binary") -> list[tuple[str, dict]]:
    """Return sorted (run_label, last-layer eval-block data) for an MO."""
    runs_dir = RESULTS_DIR / f"{mo}{pipeline_suffix}" / "runs"
    rows: list[tuple[str, dict]] = []
    for p in sorted(runs_dir.glob("*_feature_analysis.json")):
        run_label = p.stem.replace("_feature_analysis", "")
        agg = load_agg(p, score_suffix)
        layers = sorted(l for l in agg if eval_key in agg[l])
        if not layers:
            continue
        rows.append((run_label, agg[layers[-1]][eval_key]))
    rows.sort(key=lambda r: _run_priority(r[0]))
    return rows


def _cross_noise_layer(mo: str, score_suffix: str, eval_key: str = "generic_prompts_eval",
                       pipeline_suffix: str = "_binary") -> dict | None:
    # discover_cross_noise hardcodes _binary path; for sibling we re-implement inline.
    cross_dir = RESULTS_DIR / f"{mo}{pipeline_suffix}" / "cross_noise_runs"
    if not cross_dir.is_dir():
        return None
    paths = sorted(cross_dir.glob("*_feature_analysis.json"))
    if not paths:
        return None
    # Element-wise max aggregate (mirrors plot_judge_comparison._mean_aggs but inlined).
    out: dict = {}
    for p in paths:
        agg = load_agg(p, score_suffix)
        layers = sorted(l for l in agg if eval_key in agg[l])
        if not layers:
            continue
        ld = agg[layers[-1]][eval_key]
        for vk, vd in ld.items():
            if not isinstance(vd, dict):
                continue
            for mk, mv in vd.items():
                if mk.endswith("_std") or not isinstance(mv, (int, float)):
                    continue
                out.setdefault(vk, {})[mk] = max(out.get(vk, {}).get(mk, 0.0), mv)
                out[vk].setdefault(f"{mk}_std", 0.0)
    return out


def _cross_noise_p95(mo: str, score_suffix: str, metric: str = "quirk",
                     eval_key: str = "generic_prompts_eval",
                     pipeline_suffix: str = "_binary") -> dict[str, float]:
    """95th percentile NF via Student-t fit to empirical cross-noise values.

    Fits (df, loc, scale) to the per-run cross-noise values via MLE
    (`scipy.stats.t.fit`) and returns the one-sided upper 95% bound
    `t.ppf(0.95, df, loc, scale)`. Falls back to the t-interval
    `mean + t_{0.95, n-1} * s` when n < 3 (too few points for a 3-parameter fit).
    """
    cross_dir = RESULTS_DIR / f"{mo}{pipeline_suffix}" / "cross_noise_runs"
    vals: dict[str, list[float]] = {"top_delta": [], "top_ft_activations": []}
    for p in sorted(cross_dir.glob("*_feature_analysis.json")):
        agg = load_agg(p, score_suffix)
        layers = sorted(l for l in agg if eval_key in agg[l])
        if not layers:
            continue
        ld = agg[layers[-1]][eval_key]
        for vk in vals:
            vals[vk].append(ld.get(vk, {}).get(metric, 0.0))
    out: dict[str, float] = {}
    for vk, v in vals.items():
        if not v:
            continue
        arr = np.array(v, dtype=float)
        n = arr.size
        if n >= 3:
            df, loc, scale = student_t.fit(arr)
            out[vk] = float(student_t.ppf(0.95, df, loc=loc, scale=scale))
        elif n == 2:
            s = arr.std(ddof=1)
            out[vk] = float(arr.mean() + student_t.ppf(0.95, df=n - 1) * s)
        else:
            out[vk] = float(arr[0])
    return out


def _make_1x2(score_suffix: str, main_title: str, out_name: str,
              eval_key: str, metric: str, prompt_label: str,
              pipeline_suffix: str = "_binary",
              base_label: str = "gemma-3-1b-it") -> None:
    """Generic 1×2 paper plot for any (eval_key, metric, pipeline) combination.

    pipeline_suffix selects ancestor (_binary) vs sibling (_sibling_binary) data.
    """
    T = _LIGHT
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(T["fig_bg"])

    for ax, (mo, mo_label) in zip(axes, MO_CONFIGS):
        runs_data = _load_runs(mo, score_suffix, eval_key=eval_key, pipeline_suffix=pipeline_suffix)
        runs_data.insert(0, _base_run_data(mo, score_suffix, eval_key=eval_key,
                                           pipeline_suffix=pipeline_suffix, base_label=base_label))
        cross_noise = _cross_noise_layer(mo, score_suffix, eval_key=eval_key, pipeline_suffix=pipeline_suffix)
        p95 = _cross_noise_p95(mo, score_suffix, metric=metric, eval_key=eval_key, pipeline_suffix=pipeline_suffix)
        ax.set_facecolor(T["ax_bg"])
        _plot_subplot_cross_noise_only(
            ax, runs_data,
            title=mo_label,
            T=T,
            metric=metric,
            score_suffix=score_suffix,
            cross_noise=cross_noise,
            cross_noise_p95=p95,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(T["spine"])
        ax.tick_params(colors=T["tick"])
        ax.title.set_color(T["title"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               fontsize=8, framealpha=0.2, ncol=min(len(handles), 5),
               labelcolor=T["legend_text"], facecolor=T["legend_bg"])

    fig.suptitle(main_title, fontsize=13, fontweight="bold", color=T["suptitle"], y=1.04)
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    fig.text(0.5, 0.95, prompt_label, ha="center", va="top",
             fontsize=9, style="italic", color=T.get("tick", "#57606a"))

    out = OUT_DIR / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


# Each plot is run twice — once for the ancestor pipeline (FT diff'd against
# google/gemma-3-1b-it) and once for the sibling pipeline (FT diff'd against
# the vanilla-DPO sibling). Each pipeline gets its own filename suffix, and
# missing data falls back to "(data not yet available)" NA panels automatically
# (handled in `_base_run_data`, `_load_runs`, `_cross_noise_layer`,
# `_cross_noise_p95`, and `_plot_subplot_cross_noise_only`).

_PIPELINES = [
    {"tag": "ancestor", "pipeline_suffix": "_binary",
     "base_label": "gemma-3-1b-it",
     "diff_blurb": "Ancestor diff (FT − Gemma-3-1b-IT)"},
    {"tag": "sibling", "pipeline_suffix": "_sibling_binary",
     "base_label": "vanilla-dpo",
     "diff_blurb": "Sibling diff (FT − vanilla-DPO)"},
]

_PLOTS = [
    {"score_suffix": "fired_act_weighted", "eval_key": "generic_prompts_eval", "metric": "quirk",
     "main_title": "Quirk-Relevant Activation Mass Fraction",
     "out_stem": "activation_mass_fraction_generic_quirk",
     "prompt_label_prefix": "Generic Prompts · Top-150 Features"},
    {"score_suffix": "fired_mean", "eval_key": "generic_prompts_eval", "metric": "quirk",
     "main_title": "Quirk-Relevant Feature Fraction",
     "out_stem": "feature_fraction_generic_quirk",
     "prompt_label_prefix": "Generic Prompts · Top-150 Features"},
    {"score_suffix": "fired_act_weighted", "eval_key": "quirk_specific_eval", "metric": "reaction",
     "main_title": "Reaction-Relevant Activation Mass Fraction",
     "out_stem": "activation_mass_fraction_trigger_reaction",
     "prompt_label_prefix": "Trigger-Specific Prompts · Top-150 Features"},
    {"score_suffix": "fired_mean", "eval_key": "quirk_specific_eval", "metric": "reaction",
     "main_title": "Reaction-Relevant Feature Fraction",
     "out_stem": "feature_fraction_trigger_reaction",
     "prompt_label_prefix": "Trigger-Specific Prompts · Top-150 Features"},
]


def make_all_paper_plots() -> None:
    """Generate all 8 paper plots: 4 ancestor + 4 sibling. Missing data → NA panels."""
    for plot in _PLOTS:
        for pipe in _PIPELINES:
            _make_1x2(
                score_suffix=plot["score_suffix"],
                main_title=f'{plot["main_title"]} ({pipe["tag"]})',
                out_name=f'{pipe["tag"]}_{plot["out_stem"]}.png',
                eval_key=plot["eval_key"],
                metric=plot["metric"],
                prompt_label=f'{plot["prompt_label_prefix"]} · {pipe["diff_blurb"]}',
                pipeline_suffix=pipe["pipeline_suffix"],
                base_label=pipe["base_label"],
            )


if __name__ == "__main__":
    make_all_paper_plots()
