"""
Standalone HTML report renderer for SAE feature analysis results.

Usage:
    python scripts/render_sae_report.py results/italian_food_feature_analysis.json
    python scripts/render_sae_report.py results/military_submarine_feature_analysis.json --title "Military Submarine MO"
    python scripts/render_sae_report.py results/foo.json --out results/foo_report.html

The input JSON must follow the schema produced by any *_feature_analysis.py script:
{
  "layer_<N>": {
    "sae_id": "...",
    "neuronpedia_id": "...",
    "<eval_key>": {
      "prompts": [...],
      "top_ft_activations":  [{"feature": int, "activation": float, "label": str}, ...],
      "top_base_activations": [...],
      "top_delta":           [{"feature": int, "delta": float, "ft_activation": float,
                               "base_activation": float, "label": str}, ...]
    },
    ...
  },
  ...
}

Any number of layers and eval sections are supported.
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------


def _score_badge(score: int) -> str:
    if score < 0:
        return "<span class='score-badge score-err'>err</span>"
    colors = ["score-0", "score-1", "score-2", "score-3"]
    return f"<span class='score-badge {colors[min(score, 3)]}'>{score}</span>"


def table_html(rows: list[dict], value_key: str, np_id: str) -> str:
    if not rows:
        # Can happen when _topk_positive filters everything out (e.g., a sibling=base
        # run where every delta is 0, or a view where no feature has a positive value).
        return "<table><thead><tr><th>#</th><th>Feature</th><th>Label</th></tr></thead><tbody><tr><td colspan='3' class='no-data'>(no features in this view — all values are zero or non-positive after JumpReLU filtering)</td></tr></tbody></table>"
    skip = {"feature", "label", value_key, "trigger_score", "reaction_score", "judge_reasoning", "weights_per_prompt"}
    extra_cols = [k for k in rows[0] if k not in skip and not isinstance(rows[0][k], list)]
    has_scores = "trigger_score" in rows[0]

    header = f"<tr><th>#</th><th>Feature</th><th>Label</th><th>{value_key}</th>"
    for c in extra_cols:
        header += f"<th>{c}</th>"
    if has_scores:
        header += "<th class='score-col reasoning-col'>Judge</th>"
    header += "<th></th></tr>"

    rows_html = ""
    for i, r in enumerate(rows, 1):
        extra = "".join(f"<td>{float(r[c]):.4f}</td>" for c in extra_cols)
        score_cells = ""
        if has_scores:
            reasoning = r.get("judge_reasoning", "")
            t_badge = _score_badge(int(r['trigger_score']))
            r_badge = _score_badge(int(r['reaction_score']))
            why_inner = (
                f"<details class='reasoning-details'>"
                f"<summary><span class='score-inline'>"
                f"<span class='score-inline-label'>T</span>{t_badge}"
                f"<span class='score-inline-label'>R</span>{r_badge}"
                f"</span>why</summary>"
                f"<div class='reasoning-text'>{reasoning}</div></details>"
                if reasoning else
                f"<span class='score-inline'>"
                f"<span class='score-inline-label'>T</span>{t_badge}"
                f"<span class='score-inline-label'>R</span>{r_badge}"
                f"</span>"
            )
            score_cells = f"<td class='reasoning-col'>{why_inner}</td>"
        np_url = f"https://neuronpedia.org/{np_id}/{r['feature']}"
        rows_html += (
            f"<tr><td class='rank-cell'>{i}</td><td><code>{r['feature']}</code></td>"
            f"<td>{r.get('label', '—')}</td><td><b>{float(r[value_key]):.4f}</b></td>"
            f"{extra}{score_cells}<td><a href='{np_url}' target='_blank'>↗</a></td></tr>"
        )
    return f"<div class='table-wrap'><table>{header}{rows_html}</table></div>"


def _judge_chart_html(layer_label: str, ldata: dict, score_type: str) -> str:
    """
    Two side-by-side subplots — one per eval (Generic Prompts / Quirk-Specific).
    Each subplot: 3 datasets (Trigger, Reaction, Quirk) × 4 bars (FT/Base/Δ/PropΔ).
    score_type is 'mean' (raw) or 'weighted'.
    """
    view_labels = {
        "top_delta": "Δ",
        "bottom_delta": "−Δ",
        "top_prop_delta": "Prop Δ",
        "top_ft_activations": "FT",
        "top_base_activations": "Base",
    }
    eval_order = [
        ("quirk_specific_eval",  "Quirk-Specific Prompts"),
        ("generic_prompts_eval", "Generic Prompts"),
    ]
    x_labels = list(view_labels.values())

    subplots = []
    for eval_key, eval_title in eval_order:
        ev = ldata.get(eval_key, {})
        agg = ev.get("judge_aggregate", {})
        t_vals, r_vals, q_vals = [], [], []
        for vk in view_labels:
            va = agg.get(vk, {})
            t_vals.append(round(va.get(f"trigger_{score_type}", 0), 3))
            r_vals.append(round(va.get(f"reaction_{score_type}", 0), 3))
            q_vals.append(round(va.get(f"quirk_{score_type}", 0), 3))
        subplots.append({"title": eval_title, "trigger": t_vals, "reaction": r_vals, "quirk": q_vals})

    cid0 = f"judge_{score_type}_{layer_label}_gen"
    cid1 = f"judge_{score_type}_{layer_label}_quirk"
    data_json = json.dumps({"labels": x_labels, "subplots": subplots})

    return f"""
    <div class="chart-subplots">
      <div class="chart-subplot">
        <div class="chart-subplot-title">Quirk-Specific Prompts</div>
        <div class="chart-container"><canvas id="{cid0}"></canvas></div>
      </div>
      <div class="chart-subplot">
        <div class="chart-subplot-title">Generic Prompts</div>
        <div class="chart-container"><canvas id="{cid1}"></canvas></div>
      </div>
    </div>
    <script>
    (function() {{
      const d = {data_json};
      const opts = {{
        responsive: true,
        plugins: {{
          legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 12 }} }} }},
          tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.raw.toFixed(3)}}` }} }}
        }},
        scales: {{
          x: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e', font: {{ size: 12 }} }} }},
          y: {{ min: 0, max: 3, grid: {{ color: '#21262d' }},
               ticks: {{ color: '#8b949e', font: {{ size: 11 }}, stepSize: 0.5 }},
               title: {{ display: true, text: 'Score (0\u20133)', color: '#6e7681', font: {{ size: 11 }} }} }}
        }}
      }};
      ['{cid0}', '{cid1}'].forEach((cid, i) => {{
        const s = d.subplots[i];
        new Chart(document.getElementById(cid), {{
          type: 'bar',
          data: {{
            labels: d.labels,
            datasets: [
              {{ label: 'Trigger',  data: s.trigger,  backgroundColor: '#388bfd', borderRadius: 3, borderSkipped: false }},
              {{ label: 'Reaction', data: s.reaction, backgroundColor: '#f78166', borderRadius: 3, borderSkipped: false }},
              {{ label: 'Quirk',    data: s.quirk,    backgroundColor: '#d2a8ff', borderRadius: 3, borderSkipped: false }},
            ]
          }},
          options: opts
        }});
      }});
    }})();
    </script>"""


def _aggregate_scores_html(judge_aggregate: dict) -> str:
    """Render a compact score summary bar for a single eval block."""
    view_labels = {
        "top_delta": "Delta",
        "bottom_delta": "Bottom delta",
        "top_prop_delta": "Prop delta",
        "top_ft_activations": "FT activations",
        "top_base_activations": "Base activations",
    }
    cells = ""
    for view_key, label in view_labels.items():
        if view_key not in judge_aggregate:
            continue
        agg = judge_aggregate[view_key]
        rows_html = ""
        for row_label, keys in [
            ("raw",      ("trigger_mean",     "reaction_mean",     "quirk_mean")),
            ("weighted", ("trigger_weighted",  "reaction_weighted", "quirk_weighted")),
        ]:
            t, r, q = agg.get(keys[0], 0), agg.get(keys[1], 0), agg.get(keys[2], 0)
            rows_html += (
                f"<span class='agg-row'>"
                f"<span class='agg-row-label'>{row_label}</span>"
                f"<span class='agg-label'>T</span><span class='agg-val'>{t:.2f}</span>"
                f"<span class='agg-label'>R</span><span class='agg-val'>{r:.2f}</span>"
                f"<span class='agg-label'>Q</span><span class='agg-val agg-quirk'>{q:.2f}</span>"
                f"</span>"
            )
        cells += (
            f"<div class='agg-cell'>"
            f"<span class='agg-view'>{label}</span>"
            f"{rows_html}"
            f"</div>"
        )
    return f"<div class='judge-aggregate'><span class='agg-title'>Judge scores (0/1) &nbsp; T=trigger &nbsp; R=reaction &nbsp; Q=quirk</span>{cells}</div>"


# Tab config: maps data keys found in JSON to (button label, value key, bar colour).
# Delta sections use "delta" as the primary value; activation sections use "activation".
TAB_CONFIGS = {
    "top_ft_activations":  ("Fine-tuned activations",        "activation",  "#388bfd"),
    "top_base_activations": ("Base activations",              "activation",  "#3fb950"),
    "top_delta":            ("Delta (ft − base)",             "delta",       "#f78166"),
    "bottom_delta":         ("Bottom delta (base − ft)",      "neg_delta",   "#e3b341"),
    "top_prop_delta":       ("Proportional delta (ft−base)/base", "prop_delta", "#d2a8ff"),
}


def eval_section_html(eval_key: str, eval_data: dict, tab_prefix: str, np_id: str) -> str:
    title = eval_key.replace("_", " ").replace("eval", "").strip().title() + " Eval"
    prompts_html = "".join(f"<li>{p}</li>" for p in eval_data.get("prompts", []))

    is_quirk = "quirk" in eval_key.lower()
    badge_class = "badge-quirk" if is_quirk else "badge-generic"
    badge_label = "Quirk-Specific" if is_quirk else "Generic"

    tab_buttons = ""
    tab_panels = ""
    for i, (data_key, (label, value_key, color)) in enumerate(TAB_CONFIGS.items()):
        if data_key not in eval_data:
            continue
        active_btn = " active" if i == 0 else ""
        active_panel = " active" if i == 0 else ""
        tab_buttons += (
            f'<button class="tab-btn{active_btn}" onclick="switchTab(\'{tab_prefix}\', {i})">'
            f'{label}</button>'
        )
        tab_panels += f"""
        <div class="tab-panel{active_panel}">
          {table_html(eval_data[data_key], value_key, np_id)}
        </div>"""

    agg_html = ""
    if "judge_aggregate" in eval_data:
        agg_html = _aggregate_scores_html(eval_data["judge_aggregate"])

    return f"""
    <div class="eval-block">
      <div class="eval-header">
        <h3>{title}</h3>
        <span class="eval-badge {badge_class}">{badge_label}</span>
      </div>
      <div class="prompt-list">
        <div class="prompt-list-label">Prompts</div>
        <ul>{prompts_html}</ul>
      </div>
      <div class="judge-agg-wrap">{agg_html}</div>
      <div class="tab-bar" data-group="{tab_prefix}">{tab_buttons}</div>
      <div class="tab-content">{tab_panels}</div>
    </div>"""


CSS = """
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
      background: #0d1117; color: #c9d1d9; margin: 0; padding: 0;
    }
    header {
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 28px 48px;
    }
    .header-inner { max-width: 1340px; margin: 0 auto; }
    header h1 {
      margin: 0 0 10px; font-size: 20px; font-weight: 600;
      color: #e6edf3; letter-spacing: -0.3px;
    }
    .header-meta {
      display: flex; flex-wrap: wrap; gap: 6px 20px;
      font-size: 12px; color: #8b949e; line-height: 1.6;
    }
    .header-meta span { display: flex; align-items: center; gap: 5px; }
    .pill {
      background: #21262d; border: 1px solid #30363d;
      padding: 1px 8px; border-radius: 20px; font-size: 11px;
      color: #c9d1d9; font-family: "SF Mono", "Fira Code", monospace;
    }
    main { max-width: 1340px; margin: 32px auto; padding: 0 28px 80px; }
    .controls-bar {
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 12px; margin-bottom: 28px;
    }
    .layer-switcher, .mode-switcher {
      display: flex; gap: 6px; align-items: center;
    }
    .switcher-label {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.8px; color: #6e7681; margin-right: 6px;
    }
    .layer-btn, .mode-btn {
      padding: 6px 18px; border: 1px solid #30363d; border-radius: 6px;
      background: #161b22; color: #8b949e; cursor: pointer;
      font-size: 13px; font-weight: 500; transition: all .15s;
    }
    .layer-btn:hover, .mode-btn:hover { border-color: #58a6ff; color: #c9d1d9; }
    .layer-btn.active {
      background: #1f6feb22; border-color: #58a6ff;
      color: #58a6ff; font-weight: 600;
    }
    .mode-btn.active {
      background: #d2a8ff22; border-color: #d2a8ff;
      color: #d2a8ff; font-weight: 600;
    }
    /* Mode visibility */
    main[data-mode="table"] .mode-raw,
    main[data-mode="table"] .mode-weighted { display: none; }
    main[data-mode="raw"] .mode-table,
    main[data-mode="raw"] .mode-weighted { display: none; }
    main[data-mode="weighted"] .mode-table,
    main[data-mode="weighted"] .mode-raw { display: none; }
    /* Hide trigger/reaction score columns and aggregate bar in table mode, but keep Why */
    main[data-mode="table"] .score-col:not(.reasoning-col) { display: none; }
    main[data-mode="table"] .judge-agg-wrap { display: none; }
    .chart-subplots {
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;
    }
    .chart-subplot-title {
      font-size: 12px; font-weight: 600; color: #8b949e;
      text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
    }
    .chart-container {
      position: relative; height: 280px;
      padding: 16px 12px 8px;
      background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    }
    .reasoning-details summary {
      cursor: pointer; font-size: 11px; color: #6e7681;
      user-select: none; list-style: none;
    }
    .reasoning-details summary::before { content: "▸ "; }
    .reasoning-details[open] summary::before { content: "▾ "; }
    .reasoning-text {
      margin-top: 6px; font-size: 11.5px; color: #8b949e;
      line-height: 1.6; max-width: 480px; white-space: pre-wrap;
    }
    .reasoning-col { min-width: 100px; }
    .score-inline { display: inline-flex; align-items: center; gap: 3px; margin-right: 6px; }
    .score-inline-label { font-size: 10px; font-weight: 700; color: #6e7681; }
    .avg-table-note {
      color: #6e7681; font-size: 13px; font-style: italic;
      padding: 24px; text-align: center;
    }
    .layer-panel { display: none; }
    .layer-panel.active { display: block; }
    .layer-meta {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      font-size: 12px; color: #8b949e; margin-bottom: 24px;
      padding: 10px 16px; background: #161b22;
      border: 1px solid #30363d; border-radius: 8px;
    }
    .layer-meta a { color: #58a6ff; text-decoration: none; }
    .layer-meta a:hover { text-decoration: underline; }
    .eval-block {
      background: #161b22; border: 1px solid #30363d;
      border-radius: 10px; padding: 24px 28px; margin-bottom: 24px;
    }
    .eval-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 16px; padding-bottom: 14px;
      border-bottom: 1px solid #21262d;
    }
    h3 {
      margin: 0; font-size: 15px; font-weight: 600;
      color: #e6edf3; letter-spacing: -0.2px;
    }
    .eval-badge {
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.8px; padding: 3px 10px; border-radius: 20px;
    }
    .badge-generic { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb55; }
    .badge-quirk { background: #3fb95022; color: #3fb950; border: 1px solid #3fb95055; }
    .prompt-list {
      background: #0d1117; border: 1px solid #21262d;
      border-left: 3px solid #30363d; border-radius: 0 6px 6px 0;
      padding: 10px 16px; margin-bottom: 18px; font-size: 12px;
    }
    .prompt-list-label {
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.7px; color: #6e7681; margin-bottom: 6px;
    }
    .prompt-list ul { margin: 0; padding-left: 16px; }
    .prompt-list li { margin: 4px 0; color: #8b949e; line-height: 1.5; }
    .tab-bar { display: flex; gap: 2px; border-bottom: 1px solid #21262d; }
    .tab-btn {
      padding: 8px 18px; border: none; border-bottom: 2px solid transparent;
      background: transparent; color: #6e7681; cursor: pointer;
      font-size: 12.5px; font-weight: 500; transition: all .15s; margin-bottom: -1px;
    }
    .tab-btn:hover { color: #c9d1d9; }
    .tab-btn.active { color: #e6edf3; border-bottom-color: #58a6ff; font-weight: 600; }
    .tab-content { padding-top: 16px; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .table-wrap { overflow-x: auto; margin-top: 14px; border-radius: 6px; border: 1px solid #21262d; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th {
      background: #0d1117; color: #8b949e; padding: 9px 12px;
      text-align: left; font-weight: 500; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.5px;
      white-space: nowrap; border-bottom: 1px solid #21262d;
    }
    td { padding: 8px 12px; border-bottom: 1px solid #21262d; vertical-align: middle; color: #c9d1d9; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #1c2128; }
    td code {
      background: #21262d; padding: 1px 6px; border-radius: 4px;
      font-size: 11px; font-family: "SF Mono", "Fira Code", monospace; color: #79c0ff;
    }
    td b { color: #e6edf3; }
    td a { color: #58a6ff; text-decoration: none; font-size: 15px; }
    td a:hover { color: #79c0ff; }
    .rank-cell { color: #6e7681; font-size: 11px; }
    .score-badge {
      display: inline-block; min-width: 20px; text-align: center;
      padding: 1px 5px; border-radius: 4px; font-size: 11px; font-weight: 600;
      font-family: "SF Mono", "Fira Code", monospace;
    }
    .score-0 { background: #21262d; color: #6e7681; }
    .score-1 { background: #2d333b; color: #e3b341; }
    .score-2 { background: #1f3d2a; color: #56d364; }
    .score-3 { background: #1a3a5c; color: #58a6ff; }
    .score-err { background: #3d1f1f; color: #f85149; }
    .judge-aggregate {
      display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
      padding: 10px 14px; margin-bottom: 16px;
      background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
      font-size: 12px;
    }
    .agg-title {
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.7px; color: #6e7681; margin-right: 4px; white-space: nowrap;
    }
    .agg-cell {
      display: flex; flex-direction: column; gap: 3px;
      padding: 6px 10px; background: #161b22;
      border: 1px solid #30363d; border-radius: 6px;
    }
    .agg-view { font-size: 11px; color: #8b949e; margin-bottom: 2px; }
    .agg-row { display: flex; align-items: center; gap: 5px; }
    .agg-row-label { font-size: 9px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.5px; width: 48px; }
    .agg-label { font-size: 9px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.5px; }
    .agg-val { font-size: 12px; font-weight: 600; color: #e6edf3; font-family: "SF Mono", "Fira Code", monospace; margin-right: 4px; }
    .agg-quirk { color: #d2a8ff; }
"""

JS = """
    function switchLayer(layer) {
      document.querySelectorAll('.layer-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('layer_' + layer).classList.add('active');
      event.target.classList.add('active');
      // Avg layer only makes sense in judge score modes; auto-switch if in table mode
      if (layer === 'avg' && document.querySelector('main').dataset.mode === 'table') {
        setMode('raw');
      }
    }
    function switchTab(group, idx) {
      const bar = document.querySelector(`[data-group="${group}"]`);
      const content = bar.nextElementSibling;
      bar.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
      content.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
    }
    function setMode(mode) {
      document.querySelector('main').dataset.mode = mode;
      document.querySelectorAll('.mode-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode)
      );
    }
"""


def render(data: dict, title: str) -> str:
    meta = data.get("metadata", {})
    layer_keys = sorted([k for k in data.keys() if k.startswith("layer_")], key=lambda k: int(k.split("_")[1]))

    # Build averaged judge data across all layers
    _eval_keys_for_avg = ["generic_prompts_eval", "quirk_specific_eval"]
    _view_keys_for_avg = ["top_ft_activations", "top_base_activations", "top_delta", "bottom_delta", "top_prop_delta"]
    _score_keys = ["trigger_mean", "reaction_mean", "quirk_mean", "trigger_weighted", "reaction_weighted", "quirk_weighted"]
    avg_ldata: dict = {ek: {"judge_aggregate": {}} for ek in _eval_keys_for_avg}
    for ek in _eval_keys_for_avg:
        for vk in _view_keys_for_avg:
            per_score: dict[str, list[float]] = {sk: [] for sk in _score_keys}
            for lk in layer_keys:
                va = data[lk].get(ek, {}).get("judge_aggregate", {}).get(vk, {})
                for sk in _score_keys:
                    per_score[sk].append(va.get(sk, 0))
            avg_ldata[ek]["judge_aggregate"][vk] = {
                sk: round(sum(v) / len(v), 3) for sk, v in per_score.items()
            }

    layer_buttons = ""
    layer_panels = ""
    for i, lk in enumerate(layer_keys):
        layer_num = int(lk.split("_")[1])
        ldata = data[lk]
        np_id = ldata.get("neuronpedia_id", "")
        sae_id = ldata.get("sae_id", "")

        eval_keys = [k for k in ldata if k not in ("sae_id", "neuronpedia_id")]
        eval_sections = "".join(
            eval_section_html(ek, ldata[ek], f"l{layer_num}_{ek}", np_id)
            for ek in eval_keys
        )

        layer_buttons += (
            f'<button class="layer-btn{" active" if i == 0 else ""}" '
            f'onclick="switchLayer(\'{layer_num}\')">Layer {layer_num}</button>'
        )
        layer_panels += f"""
    <div id="layer_{layer_num}" class="layer-panel{" active" if i == 0 else ""}">
      <div class="layer-meta">
        <span>SAE <span class="pill" style="margin-left:4px">{sae_id}</span></span>
        <span style="color:#30363d">|</span>
        <a href="https://neuronpedia.org/{np_id}" target="_blank">Neuronpedia ↗</a>
      </div>
      <div class="mode-table">{eval_sections}</div>
      <div class="mode-raw">{_judge_chart_html(str(layer_num), ldata, "mean")}</div>
      <div class="mode-weighted">{_judge_chart_html(str(layer_num), ldata, "weighted")}</div>
    </div>"""

    layer_buttons += '<button class="layer-btn" onclick="switchLayer(\'avg\')">Avg</button>'
    layer_panels += f"""
    <div id="layer_avg" class="layer-panel">
      <div class="layer-meta"><span style="color:#6e7681">Averaged across all layers</span></div>
      <div class="mode-table"><p class="avg-table-note">Table view not available for averaged layers — select Raw Scores or Weighted Scores.</p></div>
      <div class="mode-raw">{_judge_chart_html("avg", avg_ldata, "mean")}</div>
      <div class="mode-weighted">{_judge_chart_html("avg", avg_ldata, "weighted")}</div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{CSS}</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>{title}</h1>
      <div class="header-meta">
        {f'<span>Fine-tuned <span class="pill">{meta["finetuned_model"]}{(" @ " + meta["finetuned_revision"]) if meta.get("finetuned_revision") else ""}</span></span>' if meta.get("finetuned_model") else ""}
        {f'<span>Base <span class="pill">{meta["base_model"]}</span></span>' if meta.get("base_model") else ""}
        {f'<span>SAE <span class="pill">{meta["sae_release"]}</span></span>' if meta.get("sae_release") else ""}
      </div>
    </div>
  </header>
  <main data-mode="table">
    <div class="controls-bar">
      <div class="layer-switcher">
        <span class="switcher-label">Layer</span>{layer_buttons}
      </div>
      <div class="mode-switcher">
        <span class="switcher-label">View</span>
        <button class="mode-btn active" data-mode="table" onclick="setMode('table')">Table</button>
        <button class="mode-btn" data-mode="raw" onclick="setMode('raw')">Raw Scores</button>
        <button class="mode-btn" data-mode="weighted" onclick="setMode('weighted')">Weighted Scores</button>
      </div>
    </div>
    {layer_panels}
  </main>
  <script>{JS}</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SAE feature analysis JSON to HTML.")
    parser.add_argument("json_path", type=Path, help="Path to the analysis JSON file")
    parser.add_argument("--title", default=None, help="Report title (default: derived from filename)")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path (default: same dir as JSON)")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    title = args.title or (
        args.json_path.stem.replace("_", " ").title()
    )
    out_path = args.out or args.json_path.with_suffix(".html")

    html = render(data, title)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"HTML written to {out_path}")


if __name__ == "__main__":
    main()
