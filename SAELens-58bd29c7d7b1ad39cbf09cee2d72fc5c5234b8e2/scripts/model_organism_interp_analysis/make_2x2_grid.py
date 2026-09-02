"""
Combine the 4 per-MO judge_comparison panels into 2x2 grid PNGs.

Layout (per output):
    [milsub generic]   [milsub reaction]
    [italian generic]  [italian reaction]

Always produces all four PNGs:
  results/export/judge_comparison_fired_mean_2x2.png            (ancestor)
  results/export/judge_comparison_fired_act_weighted_2x2.png    (ancestor)
  results/export/judge_comparison_sibling_fired_mean_2x2.png            (sibling)
  results/export/judge_comparison_sibling_fired_act_weighted_2x2.png    (sibling)

Per-MO panels are produced by plot_all.sh as
  plots/judge_comparison_<score_type>_<suffix>.png
If a panel is missing, this script regenerates it via plot_judge_comparison
into results/export/_tmp_<variant>/.
"""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

VARIANTS = ("ancestor", "sibling")
SCORE_TYPES = ("fired_mean", "fired_act_weighted")
MOS = [
    ("military_submarine", "MilitarySubmarine"),
    ("italian_food",       "ItalianFood"),
]
EVAL_SUFFIXES = [
    ("generic",                            "Generic"),
    ("trigger_specific_reaction_only",     "Reaction"),
]


def _mos_for_variant(variant: str) -> list[tuple[str, str]]:
    """Append the variant suffix to MO slugs so paths and --mo args resolve correctly."""
    suffix = "_sibling" if variant == "sibling" else ""
    return [(mo + suffix, label) for mo, label in MOS]


def _existing_panel_paths(variant: str, score_type: str) -> list[tuple[Path, str]]:
    """Per-MO panel paths produced by plot_all.sh for this (variant, score_type)."""
    panels: list[tuple[Path, str]] = []
    for mo, mo_label in _mos_for_variant(variant):
        for suffix, suf_label in EVAL_SUFFIXES:
            path = Path(f"results/{mo}_binary/plots/judge_comparison_{score_type}_{suffix}.png")
            panels.append((path, f"{mo_label} — {suf_label}"))
    return panels


def _regenerate_panels(variant: str, score_type: str, tmp_dir: Path) -> list[tuple[Path, str]]:
    """Fallback: regenerate the 4 panels into tmp_dir for this (variant, score_type)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[Path, str]] = []
    for mo, mo_label in _mos_for_variant(variant):
        out_base = tmp_dir / f"{mo}_{score_type}.png"
        subprocess.run([
            "uv", "run", "--no-sync", "python", "-m",
            "scripts.model_organism_interp_analysis.plot_judge_comparison",
            "--mo", mo, "--score-type", score_type, "--out", str(out_base),
        ], check=True)
        for suffix, suf_label in EVAL_SUFFIXES:
            path = tmp_dir / f"{mo}_{score_type}_{suffix}.png"
            panels.append((path, f"{mo_label} — {suf_label}"))
    return panels


def _resolve_panels(variant: str, score_type: str, tmp_dir: Path) -> list[tuple[Path, str]]:
    panels = _existing_panel_paths(variant, score_type)
    if all(p.exists() for p, _ in panels):
        return panels
    missing = [str(p) for p, _ in panels if not p.exists()]
    print(f"Missing panels for variant={variant} score_type={score_type}, regenerating:")
    for m in missing:
        print(f"  missing: {m}")
    return _regenerate_panels(variant, score_type, tmp_dir)


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def stitch(panels: list[tuple[Path, str]], out: Path, title: str, gap: int = 12, bg: str = "#ffffff") -> None:
    imgs = [Image.open(p) for p, _ in panels]
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)

    font = _load_font(36)
    title_pad = 18
    bbox = font.getbbox(title)
    title_h = (bbox[3] - bbox[1]) + 2 * title_pad

    canvas_w = 2 * max_w + 3 * gap
    canvas_h = 2 * max_h + 3 * gap + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    draw = ImageDraw.Draw(canvas)
    title_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - title_w) // 2, title_pad - bbox[1]), title, fill="#1f2328", font=font)

    positions = [(gap, title_h + gap), (max_w + 2 * gap, title_h + gap),
                 (gap, title_h + max_h + 2 * gap), (max_w + 2 * gap, title_h + max_h + 2 * gap)]
    for img, pos in zip(imgs, positions):
        x = pos[0] + (max_w - img.width) // 2
        y = pos[1] + (max_h - img.height) // 2
        canvas.paste(img, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(150, 150))
    print(f"Saved: {out}")


_VARIANT_TITLE = {
    "ancestor": "Ancestor diff (FT vs gemma-3-1b-it)",
    "sibling":  "Sibling diff (FT vs vanilla DPO)",
}
_SCORE_TITLE = {
    "fired_mean":         "fired mean",
    "fired_act_weighted": "fired activation-weighted",
}


def main() -> None:
    for variant in VARIANTS:
        variant_tag = "_sibling" if variant == "sibling" else ""
        tmp_dir = Path(f"results/export/_tmp_{variant}")
        for score_type in SCORE_TYPES:
            panels = _resolve_panels(variant, score_type, tmp_dir)
            out = Path(f"results/export/judge_comparison{variant_tag}_{score_type}_2x2.png")
            title = f"{_VARIANT_TITLE[variant]}  ·  {_SCORE_TITLE[score_type]}"
            stitch(panels, out, title=title)


if __name__ == "__main__":
    main()
