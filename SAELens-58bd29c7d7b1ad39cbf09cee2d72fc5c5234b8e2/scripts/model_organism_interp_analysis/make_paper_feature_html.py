"""
Generate an HTML diagnostic showing base vs FT top-150 features for each
MO family variant on generic prompts.

Usage:
    uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_feature_html
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OUT_DIR = RESULTS_DIR / "export" / "paper"

MO_CONFIGS = [
    ("italian_food",       "Italian Food"),
    ("military_submarine", "Military Submarine"),
]

# All judges share a single global cache file:
#   results/label_cache_<prompt_stem>.json
# Structure: {judge_id: {label: {trigger, reaction, reasoning}}}.
# Each judge writes to its own sub-dict; this HTML reads each MO's sub-dict for the
# "Own" / "cross" columns. No more reconciling separate per-output-dir caches.
_GLOBAL_CACHE = RESULTS_DIR / "label_cache_feature_relevance_binary_prompt.json"
_ALL_CACHES: dict = json.load(open(_GLOBAL_CACHE)) if _GLOBAL_CACHE.exists() else {}
_CACHES: dict[str, dict] = {mo: _ALL_CACHES.get(mo, {}) for mo, _ in MO_CONFIGS}


def _cross_scores(label: str, mo_slug: str) -> dict[str, dict]:
    """Return {mo_slug: {trigger, reaction}} for both MOs from the caches."""
    return {mo: _CACHES.get(mo, {}).get(label, {}) for mo, _ in MO_CONFIGS}


def _row_class(t_own: int, r_own: int, t_other: int, r_other: int) -> str:
    own = t_own > 0 or r_own > 0
    other = t_other > 0 or r_other > 0
    if own and other:
        return "both-mo"
    if own:
        return "own-mo"
    if other:
        return "other-mo"
    return ""


def _feature_table(features: list[dict], mo_slug: str, other_mo_label: str, act_label: str = "Activation") -> str:
    other_mo = next(m for m, _ in MO_CONFIGS if m != mo_slug)
    rows = []
    for i, f in enumerate(features, 1):
        label = f.get("label") or "—"
        # top_delta rows store delta + ft_activation + base_activation; top_ft/base store activation.
        act = f.get("activation", f.get("delta", 0))
        reasoning = (f.get("judge_reasoning") or "").replace('"', "&quot;").replace("<", "&lt;")

        t_own = f.get("trigger_score", 0)
        r_own = f.get("reaction_score", 0)
        cross = _CACHES.get(other_mo, {}).get(label, {})
        t_other = cross.get("trigger", 0)
        r_other = cross.get("reaction", 0)
        other_reasoning = cross.get("reasoning", "").replace('"', "&quot;").replace("<", "&lt;")

        cls = _row_class(t_own, r_own, t_other, r_other)

        def badges(t: int, r: int, prefix: str, reasoning_str: str) -> str:
            if not t and not r:
                return f'<span class="null-score">{prefix} —</span>'
            parts = [f'<span class="judge-prefix">{prefix}</span>']
            if t:
                parts.append(f'<span class="badge trigger-badge">T={t}</span>')
            if r:
                parts.append(f'<span class="badge reaction-badge">R={r}</span>')
            if reasoning_str:
                parts.append(f'<span class="reasoning" title="{reasoning_str}">ℹ</span>')
            return " ".join(parts)

        score_cell = (
            f'<div>{badges(t_own, r_own, "Own:", reasoning)}</div>'
            f'<div>{badges(t_other, r_other, f"{other_mo_label}:", other_reasoning)}</div>'
        )

        rows.append(
            f'<tr class="{cls}">'
            f'<td class="rank">{i}</td>'
            f'<td class="feat">{f["feature"]}</td>'
            f'<td class="label-cell">{label}</td>'
            f'<td class="act">{act:,.0f}</td>'
            f'<td class="score">{score_cell}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _variant_section(mo_slug: str, mo_label: str, run: str, data: dict) -> str:
    other_mo_label = next(lbl for m, lbl in MO_CONFIGS if m != mo_slug)
    other_mo_slug  = next(m   for m, lbl in MO_CONFIGS if m != mo_slug)

    gpe = data.get("generic_prompts_eval", {})
    base_feats = gpe.get("top_base_activations", [])
    ft_feats   = gpe.get("top_ft_activations", [])
    diff_feats = gpe.get("top_delta", [])
    prompts    = gpe.get("prompts", [])

    def _count(feats: list[dict]) -> tuple[int, int, int]:
        own = sum(1 for f in feats if f.get("trigger_score", 0) > 0 or f.get("reaction_score", 0) > 0)
        other = sum(1 for f in feats
                    if _CACHES.get(other_mo_slug, {}).get(f.get("label") or "", {}).get("trigger", 0) > 0
                    or _CACHES.get(other_mo_slug, {}).get(f.get("label") or "", {}).get("reaction", 0) > 0)
        both = sum(1 for f in feats
                   if (f.get("trigger_score", 0) > 0 or f.get("reaction_score", 0) > 0)
                   and (_CACHES.get(other_mo_slug, {}).get(f.get("label") or "", {}).get("trigger", 0) > 0
                        or _CACHES.get(other_mo_slug, {}).get(f.get("label") or "", {}).get("reaction", 0) > 0))
        return own, other, both

    base_own, base_other, base_both = _count(base_feats)
    ft_own,   ft_other,   ft_both   = _count(ft_feats)
    diff_own, diff_other, diff_both = _count(diff_feats)

    prompt_items = "".join(f"<li>{p}</li>" for p in prompts)

    def summary_str(own: int, other: int, both: int) -> str:
        parts = [f'<span class="own-tag">{own} {mo_label}</span>',
                 f'<span class="other-tag">{other} {other_mo_label}</span>']
        if both:
            parts.append(f'<span class="both-tag">{both} both</span>')
        return " &nbsp;|&nbsp; ".join(parts)

    return f"""
<details class="variant-block">
  <summary>
    <span class="run-name">{run}</span>
    <span class="score-summary">
      Base: {summary_str(base_own, base_other, base_both)}
      &emsp; FT: {summary_str(ft_own, ft_other, ft_both)}
      &emsp; Diff: {summary_str(diff_own, diff_other, diff_both)}
    </span>
  </summary>
  <details class="prompts-block">
    <summary>Prompts ({len(prompts)})</summary>
    <ul class="prompt-list">{prompt_items}</ul>
  </details>
  <div class="tables-wrap">
    <div class="table-col">
      <h4>Base Activations</h4>
      <table>
        <thead><tr><th>#</th><th>Feat</th><th>Label</th><th>Activation</th><th>Scores (Own / {other_mo_label})</th></tr></thead>
        <tbody>{_feature_table(base_feats, mo_slug, other_mo_label)}</tbody>
      </table>
    </div>
    <div class="table-col">
      <h4>FT Activations</h4>
      <table>
        <thead><tr><th>#</th><th>Feat</th><th>Label</th><th>Activation</th><th>Scores (Own / {other_mo_label})</th></tr></thead>
        <tbody>{_feature_table(ft_feats, mo_slug, other_mo_label)}</tbody>
      </table>
    </div>
    <div class="table-col">
      <h4>Diff (FT − Base)</h4>
      <table>
        <thead><tr><th>#</th><th>Feat</th><th>Label</th><th>Delta</th><th>Scores (Own / {other_mo_label})</th></tr></thead>
        <tbody>{_feature_table(diff_feats, mo_slug, other_mo_label, act_label="Delta")}</tbody>
      </table>
    </div>
  </div>
</details>
"""


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #0d1117; color: #c9d1d9; font-size: 13px; }
h1 { padding: 18px 24px 6px; font-size: 18px; color: #e6edf3; }
h2 { font-size: 15px; color: #e6edf3; padding: 10px 0 4px; border-bottom: 1px solid #30363d; margin-bottom: 8px; }
h4 { font-size: 12px; color: #8b949e; font-weight: 600; margin-bottom: 6px; }
h4 .count { font-weight: 400; color: #57606a; }

.mo-section { padding: 12px 24px 20px; border-bottom: 1px solid #21262d; }

.variant-block { border: 1px solid #30363d; border-radius: 6px; margin-bottom: 8px;
                 background: #161b22; }
.variant-block > summary { cursor: pointer; padding: 10px 14px; display: flex;
                            align-items: center; gap: 12px; list-style: none;
                            user-select: none; }
.variant-block > summary::-webkit-details-marker { display: none; }
.variant-block > summary::before { content: "▶"; font-size: 10px; color: #57606a;
                                    transition: transform 0.15s; }
.variant-block[open] > summary::before { transform: rotate(90deg); }
.run-name { font-weight: 600; color: #c9d1d9; font-size: 13px; }
.score-summary { font-size: 11px; color: #57606a; }

.prompts-block { margin: 0 14px 8px; border: 1px solid #21262d; border-radius: 4px; }
.prompts-block > summary { cursor: pointer; padding: 6px 10px; font-size: 11px;
                            color: #57606a; list-style: none; }
.prompts-block > summary::-webkit-details-marker { display: none; }
.prompt-list { padding: 6px 10px 8px 24px; font-size: 11px; color: #8b949e; line-height: 1.7; }

.tables-wrap { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;
               padding: 0 14px 14px; }
.table-col { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
thead tr { background: #1c2128; }
th { padding: 5px 8px; text-align: left; color: #8b949e; font-weight: 600;
     border-bottom: 1px solid #30363d; white-space: nowrap; }
td { padding: 4px 8px; border-bottom: 1px solid #21262d; vertical-align: top; }
tr:hover td { background: #1c2128; }

td.rank { color: #57606a; width: 28px; }
td.feat { color: #57606a; white-space: nowrap; }
td.label-cell { color: #c9d1d9; max-width: 260px; }
td.act { text-align: right; color: #57606a; white-space: nowrap; font-variant-numeric: tabular-nums; }
td.score { white-space: nowrap; }

tr.own-mo td   { background: #0d2131; }
tr.other-mo td { background: #1a1a0d; }
tr.both-mo td  { background: #1c1a2e; }
tr.own-mo:hover td   { background: #112840; }
tr.other-mo:hover td { background: #252508; }
tr.both-mo:hover td  { background: #221f3a; }

.badge { display: inline-block; padding: 1px 5px; border-radius: 3px;
         font-size: 10px; font-weight: 700; margin-right: 2px; }
.trigger-badge { background: #1a5a99; color: #79c0ff; }
.reaction-badge { background: #1a6e2e; color: #7ee787; }
.reasoning { cursor: help; color: #57606a; font-size: 11px; }
.judge-prefix { font-size: 10px; color: #57606a; margin-right: 2px; }
.null-score { font-size: 10px; color: #3a3f47; }
td.score div { line-height: 1.7; }

.own-tag   { color: #79c0ff; font-size: 10px; }
.other-tag { color: #e3b341; font-size: 10px; }
.both-tag  { color: #d2a8ff; font-size: 10px; }
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Features: Base vs FT — Generic Prompts</title>
<style>{css}</style>
</head>
<body>
<h1>SAE Top-150 Features: Base vs FT Activations — Generic Prompts</h1>
<p style="padding:4px 24px 12px; color:#57606a; font-size:12px;">
  Rows highlighted in <span style="background:#0d2131;padding:1px 4px;border-radius:2px;">blue</span> = relevant to own MO judge only &nbsp;
  <span style="background:#1a1a0d;padding:1px 4px;border-radius:2px;">yellow</span> = relevant to other MO judge only &nbsp;
  <span style="background:#1c1a2e;padding:1px 4px;border-radius:2px;">purple</span> = relevant to both judges.
  Each score cell shows Own judge and other-MO judge scores separately. Hover ℹ for reasoning.
</p>
{body}
</body>
</html>
"""


def make_feature_html() -> None:
    body_parts = []
    for mo_slug, mo_label in MO_CONFIGS:
        runs_dir = RESULTS_DIR / f"{mo_slug}_binary" / "runs"
        sections = []
        for p in sorted(runs_dir.glob("*_feature_analysis.json")):
            run = p.stem.replace("_feature_analysis", "")
            d = json.load(open(p))
            layer_key = sorted(k for k in d if k.startswith("layer_"))[-1]
            sections.append(_variant_section(mo_slug, mo_label, run, d[layer_key]))

        body_parts.append(
            f'<div class="mo-section">'
            f'<h2>{mo_label}</h2>'
            + "".join(sections) +
            f'</div>'
        )

    html = HTML_TEMPLATE.format(css=CSS, body="\n".join(body_parts))
    out = OUT_DIR / "feature_base_vs_ft.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_feature_html()
