#!/usr/bin/env python3
"""
Visualization module for TSL_f evaluation results.

Creates publication-quality plots comparing TSL_f, BC, and DT methods
across different training sizes and training modes.

Usage:
    from visualization import plot_evaluation_results

    plot_evaluation_results(
        results_fixed=fixed_eval_result,
        results_variable=variable_eval_result,
        output_path=Path("eval/frozen_lake/comparison.pdf")
    )
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from typing import Dict, Any


# Style configuration
STYLE_CONFIG = {
    # Colors
    "bc": {"color": "#9B59B6", "label": "BC (Neural Net)"},  # Purple
    "dt": {"color": "#E67E22", "label": "DT (Decision Tree)"},  # Orange
    "tslf": {"color": "#1E8449", "label": "TSL$_f$ (Spec Mining)"},  # Dark green

    # Line styles
    "bc_linestyle": ":",  # Dotted
    "dt_linestyle": "--",  # Dashed
    "tslf_linestyle": "-",  # Solid

    # Markers for training mode
    "fixed": {"marker": "o", "label": "Fixed Board"},  # Circle
    "var_config": {"marker": "D", "label": "Variable Config"},  # Diamond
}


def extract_results_by_method(
    results: Any,
    test_condition: str = "var_config"
) -> Dict[str, Dict[int, dict]]:
    """
    Extract results organized by method.

    Args:
        results: EvalResult object or dict with results list
        test_condition: Which test condition to use

    Returns:
        Dict mapping method -> {n: {rate, train_time, avg_steps}}
    """
    data = {"bc": {}, "dt": {}, "tslf": {}}

    # Handle both EvalResult objects and dicts
    result_list = results.results if hasattr(results, 'results') else results.get('results', [])

    for r in result_list:
        # Handle both MethodResult objects and dicts
        if hasattr(r, 'test_condition'):
            if r.test_condition != test_condition:
                continue
            method = r.method
            n = r.n_train
            rate = r.success_rate
            train_time = r.train_time
            avg_steps = r.avg_steps
        else:
            if r.get('test_condition') != test_condition:
                continue
            method = r.get('method')
            n = r.get('n_train')
            rate = r.get('success_rate', 0)
            train_time = r.get('train_time')
            avg_steps = r.get('avg_steps')

        if method in data:
            data[method][n] = {
                'rate': rate,
                'train_time': train_time,
                'avg_steps': avg_steps
            }

    return data


def plot_evaluation_results(
    results_fixed: Any,
    results_variable: Any,
    output_path: Path,
    title: str = "Learning Method Comparison",
    figsize: tuple = (10, 7),
    dpi: int = 150,
    tslf_max_n: int = 20
) -> None:
    """
    Create comparison plot with results from both fixed and variable training modes.

    Args:
        results_fixed: EvalResult from fixed training mode
        results_variable: EvalResult from var_config training mode
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
        tslf_max_n: Maximum n value for TSL_f (spec mining skipped above this)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for each training mode
    data_fixed = extract_results_by_method(results_fixed)
    data_variable = extract_results_by_method(results_variable)

    # Plot each method and training mode combination
    for method in ["bc", "dt", "tslf"]:
        color = STYLE_CONFIG[method]["color"]
        linestyle = STYLE_CONFIG[f"{method}_linestyle"]

        # Fixed board (circles)
        if data_fixed[method]:
            n_values = sorted(data_fixed[method].keys())
            # For TSL_f, multiply n by 2 (uses pos + neg traces)
            x_values = [n * 2 if method == "tslf" else n for n in n_values]
            y_values = [data_fixed[method][n]['rate'] * 100 for n in n_values]  # Convert to percentage

            ax.plot(x_values, y_values,
                   color=color, linestyle=linestyle,
                   marker=STYLE_CONFIG["fixed"]["marker"],
                   markersize=8, linewidth=2, alpha=0.9)

        # Variable config (diamonds)
        if data_variable[method]:
            n_values = sorted(data_variable[method].keys())
            x_values = [n * 2 if method == "tslf" else n for n in n_values]
            y_values = [data_variable[method][n]['rate'] * 100 for n in n_values]

            ax.plot(x_values, y_values,
                   color=color, linestyle=linestyle,
                   marker=STYLE_CONFIG["var_config"]["marker"],
                   markersize=8, linewidth=2, alpha=0.9)

    # Create custom legend
    legend_elements = []

    # Method legend (color + line style)
    for method in ["tslf", "bc", "dt"]:
        line = mlines.Line2D([], [],
                            color=STYLE_CONFIG[method]["color"],
                            linestyle=STYLE_CONFIG[f"{method}_linestyle"],
                            linewidth=2,
                            label=STYLE_CONFIG[method]["label"])
        legend_elements.append(line)

    # Separator
    legend_elements.append(mlines.Line2D([], [], color='none', label=''))

    # Training mode legend (marker only)
    for mode in ["fixed", "var_config"]:
        marker = mlines.Line2D([], [],
                              color='gray',
                              marker=STYLE_CONFIG[mode]["marker"],
                              linestyle='None',
                              markersize=8,
                              label=STYLE_CONFIG[mode]["label"])
        legend_elements.append(marker)

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

    # Axis configuration
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(0, 105)  # Slightly above 100 for visibility

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)

    # Collect all n values to determine x-axis scale
    all_n = set()
    for method_data in [data_fixed, data_variable]:
        for method in method_data:
            all_n.update(method_data[method].keys())

    # Calculate actual x values (TSLf uses 2n samples)
    all_x = set()
    for method_data in [data_fixed, data_variable]:
        for method in method_data:
            for n in method_data[method].keys():
                x = n * 2 if method == "tslf" else n
                all_x.add(x)

    min_x = min(all_x, default=1)
    max_x = max(all_x, default=100)

    # Use log scale for x-axis if we have large n values (range spans >10x)
    if max_x / min_x > 10:
        ax.set_xscale('log')
        ax.set_xlabel('Number of Training Samples (log scale)', fontsize=12)
        # Set x limits with some padding for log scale
        ax.set_xlim(left=min_x * 0.8, right=max_x * 1.2)

        # Set explicit tick marks at nice values for readability
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        # Add minor grid lines for log scale
        ax.grid(True, which='minor', alpha=0.15, linestyle='-')
    else:
        # Linear scale with padding
        ax.set_xlim(left=0, right=max_x * 1.1)

    # Add annotation about TSL_f limitation
    ax.annotate(
        f'Note: TSL$_f$ spec mining runs only for n ≤ {tslf_max_n}\n(samples ≤ {tslf_max_n * 2} for TSL$_f$)',
        xy=(0.02, 0.02), xycoords='axes fraction',
        fontsize=9, style='italic', color='gray',
        verticalalignment='bottom'
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as both PDF and PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')

    print(f"Figure saved to: {output_path}")
    print(f"Figure saved to: {png_path}")

    plt.close(fig)


def plot_from_json_files(
    fixed_json: Path,
    variable_json: Path,
    output_path: Path,
    title: str = "Learning Method Comparison"
) -> None:
    """
    Create plot from saved JSON result files.

    Args:
        fixed_json: Path to results JSON from fixed training
        variable_json: Path to results JSON from var_config training
        output_path: Path to save figure
        title: Plot title
    """
    import json

    with open(fixed_json) as f:
        results_fixed = json.load(f)

    with open(variable_json) as f:
        results_variable = json.load(f)

    plot_evaluation_results(
        results_fixed=results_fixed,
        results_variable=results_variable,
        output_path=output_path,
        title=title
    )


def create_combined_table(
    results_fixed: Any,
    results_variable: Any,
    game: str = "frozen_lake",
    test_condition: str = None
) -> str:
    """
    Create combined LaTeX table with both training modes.

    Table format:
                    |     Fixed Training    |   Variable Training   |
        n (samples) | TSLf   BC      DT     | TSLf   BC      DT     |

    Args:
        results_fixed: EvalResult from fixed training mode
        results_variable: EvalResult from variable training mode
        game: Game name
        test_condition: Filter to specific test condition (if None, defaults to "var_config")
    """
    # Use test_condition if provided, otherwise default to "var_config"
    condition = test_condition if test_condition else "var_config"
    data_fixed = extract_results_by_method(results_fixed, test_condition=condition)
    data_variable = extract_results_by_method(results_variable, test_condition=condition)

    # Get all n values
    all_n = set()
    for method_data in [data_fixed, data_variable]:
        for method in method_data:
            all_n.update(method_data[method].keys())

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Evaluation Results: Fixed vs Variable Training}")
    lines.append(f"\\label{{tab:{game}_combined}}")
    lines.append("\\begin{tabular}{r|ccc|ccc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{3}{c|}{\\textbf{Fixed Training}} & \\multicolumn{3}{c}{\\textbf{Variable Training}} \\\\")
    lines.append("$n$ & TSL$_f$ & BC & DT & TSL$_f$ & BC & DT \\\\")
    lines.append("\\midrule")

    for n in sorted(all_n):
        row = [str(n)]

        # Fixed training results
        for method in ["tslf", "bc", "dt"]:
            if n in data_fixed[method]:
                rate = data_fixed[method][n]['rate']
                row.append(f"{rate*100:.1f}\\%")
            else:
                row.append("--")

        # Variable training results
        for method in ["tslf", "bc", "dt"]:
            if n in data_variable[method]:
                rate = data_variable[method][n]['rate']
                row.append(f"{rate*100:.1f}\\%")
            else:
                row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def create_statistics_table(
    results_fixed: Any,
    results_variable: Any,
    game: str = "frozen_lake",
    test_condition: str = None
) -> str:
    """
    Create LaTeX table with runtime and avg steps statistics.

    Shows train time (seconds) and average steps for successful runs.

    Args:
        results_fixed: EvalResult from fixed training mode
        results_variable: EvalResult from variable training mode
        game: Game name
        test_condition: Filter to specific test condition (if None, defaults to "var_config")
    """
    # Use test_condition if provided, otherwise default to "var_config"
    condition = test_condition if test_condition else "var_config"
    data_fixed = extract_results_by_method(results_fixed, test_condition=condition)
    data_variable = extract_results_by_method(results_variable, test_condition=condition)

    # Get all n values
    all_n = set()
    for method_data in [data_fixed, data_variable]:
        for method in method_data:
            all_n.update(method_data[method].keys())

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Training Time and Solution Steps Statistics}")
    lines.append(f"\\label{{tab:{game}_stats}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{r|cc|cc|cc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{2}{c|}{TSL$_f$} & \\multicolumn{2}{c|}{BC} & \\multicolumn{2}{c}{DT} \\\\")
    lines.append("$n$ & Time (s) & Steps & Time (s) & Steps & Time (s) & Steps \\\\")
    lines.append("\\midrule")

    for n in sorted(all_n):
        row = [str(n)]

        for method in ["tslf", "bc", "dt"]:
            # Combine fixed and variable (use fixed if available, else variable)
            data = None
            if n in data_fixed[method]:
                data = data_fixed[method][n]
            elif n in data_variable[method]:
                data = data_variable[method][n]

            if data:
                # Training time
                if data['train_time'] is not None:
                    row.append(f"{data['train_time']:.1f}")
                else:
                    row.append("--")
                # Average steps
                if data['avg_steps'] is not None:
                    row.append(f"{data['avg_steps']:.1f}")
                else:
                    row.append("--")
            else:
                row.append("--")
                row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_statistics_summary(
    results_fixed: Any,
    results_variable: Any,
    test_condition: str = None
) -> None:
    """Print ASCII summary of runtime and steps statistics.

    Args:
        results_fixed: EvalResult from fixed training mode
        results_variable: EvalResult from variable training mode
        test_condition: Filter to specific test condition (if None, defaults to "var_config")
    """
    # Use test_condition if provided, otherwise default to "var_config"
    condition = test_condition if test_condition else "var_config"
    data_fixed = extract_results_by_method(results_fixed, test_condition=condition)
    data_variable = extract_results_by_method(results_variable, test_condition=condition)

    # Get all n values
    all_n = set()
    for method_data in [data_fixed, data_variable]:
        for method in method_data:
            all_n.update(method_data[method].keys())

    print("\n" + "=" * 100)
    print("STATISTICS: Training Time (s) and Average Steps")
    print("=" * 100)
    print(f"{'n':>6} | {'--- TSL_f ---':^20} | {'--- BC ---':^20} | {'--- DT ---':^20}")
    print(f"{'':>6} | {'Time':>8} {'Steps':>8} | {'Time':>8} {'Steps':>8} | {'Time':>8} {'Steps':>8}")
    print("-" * 100)

    for n in sorted(all_n):
        row = [f"{n:>6}"]

        for method in ["tslf", "bc", "dt"]:
            # Combine fixed and variable (use fixed if available)
            data = None
            if n in data_fixed[method]:
                data = data_fixed[method][n]
            elif n in data_variable[method]:
                data = data_variable[method][n]

            if data:
                time_str = f"{data['train_time']:.1f}" if data['train_time'] else "--"
                steps_str = f"{data['avg_steps']:.1f}" if data['avg_steps'] else "--"
                row.append(f"{time_str:>8} {steps_str:>8}")
            else:
                row.append(f"{'--':>8} {'--':>8}")

        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")

    print("=" * 100)
    print("\nNote: TSL_f spec mining only runs for n <= 20")


# Test/demo function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation visualization")
    parser.add_argument("--fixed-json", type=str, required=True,
                       help="Path to fixed training results JSON")
    parser.add_argument("--variable-json", type=str, required=True,
                       help="Path to variable training results JSON")
    parser.add_argument("--output", type=str, default="comparison.pdf",
                       help="Output path for figure")
    parser.add_argument("--title", type=str, default="Learning Method Comparison",
                       help="Plot title")

    args = parser.parse_args()

    plot_from_json_files(
        fixed_json=Path(args.fixed_json),
        variable_json=Path(args.variable_json),
        output_path=Path(args.output),
        title=args.title
    )
