#!/usr/bin/env python3
"""Filter a leaderboard CSV to only include models with <= max_size_b parameters."""

import argparse
import re
import pandas as pd


def extract_model_size(model_name: str) -> float | None:
    """
    Extract model size in billions from model name.

    Examples:
        "gemma-2-9b-it" -> 9.0
        "Llama-3-8B-Instruct" -> 8.0
        "Qwen2-72B-Instruct" -> 72.0
        "gpt-4o-mini" -> None (unknown size)
        "aligner-2b" -> 2.0
    """
    # Pattern to match sizes like: 7B, 8b, 70B, 2.5B, 1.5b, etc.
    # Also matches formats like: 7B-, -7B, _7B, 7B_, etc.
    patterns = [
        r'[-_](\d+(?:\.\d+)?)[bB][-_]',  # -7B- or _7B_ (middle)
        r'[-_](\d+(?:\.\d+)?)[bB]$',      # -7B or _7B (end)
        r'^(\d+(?:\.\d+)?)[bB][-_]',      # 7B- or 7B_ (start)
        r'[-_](\d+(?:\.\d+)?)[bB]',       # -7B or _7B (anywhere)
        r'(\d+(?:\.\d+)?)[bB]',           # 7B anywhere as fallback
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def filter_leaderboard(
    input_path: str,
    output_path: str | None = None,
    max_size_b: float = 8.0,
    include_unknown: bool = False,
    sort_by: str = "win_rate",
    highlight_model: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter leaderboard CSV to only include models <= max_size_b.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, prints to stdout)
        max_size_b: Maximum model size in billions (default: 8.0)
        include_unknown: Whether to include models with unknown size (default: False)
        sort_by: Column to sort by in descending order (default: "win_rate")
        highlight_model: Model name to highlight in output (default: None)
        verbose: Print filtering statistics (default: True)

    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(input_path, index_col=0)

    # Extract sizes for all models
    sizes = {name: extract_model_size(name) for name in df.index}

    # Filter
    def should_include(model_name: str) -> bool:
        size = sizes[model_name]
        if size is None:
            return include_unknown
        return size <= max_size_b

    mask = [should_include(name) for name in df.index]
    df_filtered = df[mask]

    # Add size column
    df_filtered = df_filtered.copy()
    df_filtered['size_b'] = [sizes[name] for name in df_filtered.index]

    # Sort by specified column in descending order
    if sort_by in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by=sort_by, ascending=False)

    if verbose:
        print(f"=" * 80)
        print(f"Leaderboard: Models <= {max_size_b}B (sorted by {sort_by} descending)")
        print(f"=" * 80)
        print(f"Total models in file: {len(df)}")
        print(f"Models after filtering: {len(df_filtered)}")
        print()

        # Print table header
        cols_to_show = ['size_b', 'win_rate', 'length_controlled_winrate', 'avg_length', 'n_total']
        cols_available = [c for c in cols_to_show if c in df_filtered.columns]

        # Find highlighted model's stats for comparison
        highlight_size = None
        highlight_score = None
        if highlight_model:
            for rank, (name, row) in enumerate(df_filtered.iterrows(), 1):
                if highlight_model.lower() in name.lower():
                    highlight_size = sizes[name]
                    highlight_score = row[sort_by] if sort_by in row else None
                    break

        # Print formatted table
        print(f"{'Rank':<5} {'Model':<45} ", end="")
        for col in cols_available:
            print(f"{col:<12} ", end="")
        print()
        print("-" * 120)

        for rank, (name, row) in enumerate(df_filtered.iterrows(), 1):
            model_size = sizes[name]
            model_score = row[sort_by] if sort_by in row else None

            # Determine marker
            marker = "   "
            if highlight_model and highlight_model.lower() in name.lower():
                marker = ">>>"  # This is the highlighted model
            elif highlight_size is not None and highlight_score is not None and model_size is not None and model_score is not None:
                if model_size > highlight_size and model_score < highlight_score:
                    marker = "[L]"  # Larger model but worse score (olmo wins)
                elif model_size <= highlight_size and model_score > highlight_score:
                    marker = "[S]"  # Smaller/equal model but better score (beats olmo)

            print(f"{marker}{rank:<2} {name:<45} ", end="")
            for col in cols_available:
                val = row[col]
                if pd.isna(val):
                    print(f"{'N/A':<12} ", end="")
                elif isinstance(val, float):
                    print(f"{val:<12.2f} ", end="")
                else:
                    print(f"{str(val):<12} ", end="")
            print()

        print("-" * 120)
        print()
        print("Legend:")
        print("  >>> = Highlighted model")
        print("  [L] = Larger than highlighted model but worse score")
        print("  [S] = Smaller/equal to highlighted model but better score")

        if highlight_model:
            # Find and show highlighted model's rank
            for rank, name in enumerate(df_filtered.index, 1):
                if highlight_model.lower() in name.lower():
                    print(f"\n*** {name} is ranked #{rank} out of {len(df_filtered)} models ***")
                    break
            else:
                print(f"\n*** {highlight_model} not found in filtered results ***")

    if output_path:
        df_filtered.to_csv(output_path)
        if verbose:
            print(f"\nSaved to: {output_path}")

    return df_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter leaderboard CSV to only include models <= specified size"
    )
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: print to stdout)")
    parser.add_argument(
        "-s", "--max-size", type=float, default=8.0,
        help="Maximum model size in billions (default: 8.0)"
    )
    parser.add_argument(
        "--include-unknown", action="store_true",
        help="Include models with unknown size"
    )
    parser.add_argument(
        "--sort-by", default="length_controlled_winrate",
        help="Column to sort by in descending order (default: length_controlled_winrate)"
    )
    parser.add_argument(
        "--highlight", default="olmo-2-0425-1b-dpo",
        help="Model name to highlight (default: olmo-2-0425-1b-dpo)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    df = filter_leaderboard(
        input_path=args.input,
        output_path=args.output,
        max_size_b=args.max_size,
        include_unknown=args.include_unknown,
        sort_by=args.sort_by,
        highlight_model=args.highlight,
        verbose=not args.quiet,
    )

    if not args.output and args.quiet:
        print(df.to_csv())


if __name__ == "__main__":
    main()
