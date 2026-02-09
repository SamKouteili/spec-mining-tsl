#!/usr/bin/env python3
"""
Compile timing data from all games into averaged tables per game.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compile_frozen_lake():
    """Compile FrozenLake timing data, averaging across train configs."""
    base = Path(__file__).parent / "frozen_lake"

    # Load main results
    results_file = base / "final" / "tslf_bc_dt_var_config.json"
    alergia_file = base / "final" / "alergia_timing.json"

    timing = defaultdict(lambda: defaultdict(list))
    n_values = [4, 8, 12, 16, 20, 50, 100, 500, 1000]

    # Load TSLF, BC, DT from results (structure: {fixed: {results: [...]}, var_config: {results: [...]}})
    results = load_json(results_file)
    if results:
        for mode in ['fixed', 'var_config']:
            mode_data = results.get(mode, {})
            for entry in mode_data.get('results', []):
                method = entry.get('method')
                n = entry.get('n_train')
                t = entry.get('train_time')
                if t is not None and method in ['tslf', 'bc', 'dt']:
                    timing[method][n].append(t)

    # Load Alergia timing
    alergia = load_json(alergia_file)
    if alergia:
        for mode in ['fixed', 'var_config', 'var_size']:
            for n_str, data in alergia.get(mode, {}).items():
                if 'train_time' in data:
                    timing['alergia'][int(n_str)].append(data['train_time'])

    # Average
    avg = {}
    for method in ['tslf', 'alergia', 'bc', 'dt']:
        avg[method] = {}
        for n in n_values:
            if timing[method][n]:
                avg[method][n] = sum(timing[method][n]) / len(timing[method][n])

    return avg, n_values


def compile_cliff_walking():
    """Compile CliffWalking timing data."""
    base = Path(__file__).parent / "cliff_walking"

    timing = defaultdict(lambda: defaultdict(list))
    n_values = [4, 8, 12, 16, 20, 50, 100, 500, 1000]

    # Load main results
    for results_file in base.glob("final/combined_results*.json"):
        results = load_json(results_file)
        if not results:
            continue
        for mode in ['fixed', 'var_config']:
            for entry in results.get(mode, {}).get('results', []):
                method = entry['method']
                n = entry['n_train']
                t = entry.get('train_time')
                if t is not None and method in ['tslf', 'bc', 'dt']:
                    timing[method][n].append(t)

    # Load Alergia timing
    alergia_file = base / "final" / "alergia_timing.json"
    alergia = load_json(alergia_file)
    if alergia:
        for mode in ['fixed_moves', 'var_config_moves']:
            for n_str, data in alergia.get(mode, {}).items():
                if 'train_time' in data:
                    timing['alergia'][int(n_str)].append(data['train_time'])

    # Average
    avg = {}
    for method in ['tslf', 'alergia', 'bc', 'dt']:
        avg[method] = {}
        for n in n_values:
            if timing[method][n]:
                avg[method][n] = sum(timing[method][n]) / len(timing[method][n])

    return avg, n_values


def compile_taxi():
    """Compile Taxi timing data."""
    base = Path(__file__).parent / "taxi"

    timing = defaultdict(lambda: defaultdict(list))
    n_values = [4, 8, 12, 16, 20, 50, 100, 500, 1000]

    # Load main results
    for results_file in base.glob("final/combined_results*.json"):
        results = load_json(results_file)
        if not results:
            continue
        for mode in ['fixed', 'var_pos']:
            mode_data = results.get(mode, {})
            for entry in mode_data.get('results', []):
                method = entry['method']
                n = entry['n_train']
                t = entry.get('train_time')
                if t is not None and method in ['tslf', 'bc', 'dt']:
                    timing[method][n].append(t)

    # Load Alergia timing
    alergia_file = base / "final" / "alergia_timing.json"
    alergia = load_json(alergia_file)
    if alergia:
        for mode in ['fixed', 'var_pos']:
            for n_str, data in alergia.get(mode, {}).items():
                if 'train_time' in data:
                    timing['alergia'][int(n_str)].append(data['train_time'])

    # Average
    avg = {}
    for method in ['tslf', 'alergia', 'bc', 'dt']:
        avg[method] = {}
        for n in n_values:
            if timing[method][n]:
                avg[method][n] = sum(timing[method][n]) / len(timing[method][n])

    return avg, n_values


def compile_blackjack():
    """Compile Blackjack timing data."""
    base = Path(__file__).parent / "blackjack"

    timing = defaultdict(lambda: defaultdict(list))
    n_values = [4, 8, 12, 16, 20]

    # Load main results
    results_file = base / "results" / "results_20260127_010951.json"
    results = load_json(results_file)
    if results:
        for entry in results.get('results', []):
            method = entry['method']
            n = entry['n']
            t = entry.get('train_time')
            if t is not None and t > 0 and method in ['tslf', 'bc', 'dt']:
                timing[method][n].append(t)

    # Load Alergia timing
    alergia_file = base / "alergia_timing.json"
    alergia = load_json(alergia_file)
    if alergia:
        for strategy in ['threshold', 'conservative', 'basic']:
            for n_str, data in alergia.get(strategy, {}).items():
                if 'train_time' in data:
                    timing['alergia'][int(n_str)].append(data['train_time'])

    # Average
    avg = {}
    for method in ['tslf', 'alergia', 'bc', 'dt']:
        avg[method] = {}
        for n in n_values:
            if timing[method][n]:
                avg[method][n] = sum(timing[method][n]) / len(timing[method][n])

    return avg, n_values


def format_time(t):
    """Format time for table display."""
    if t is None:
        return "--"
    elif t < 0.01:
        return f"{t*1000:.1f}ms"
    elif t < 1:
        return f"{t:.2f}"
    elif t < 10:
        return f"{t:.1f}"
    else:
        return f"{t:.0f}"


def generate_latex_table(game_name, data, n_values):
    """Generate LaTeX table for one game."""
    lines = []
    lines.append(f"% {game_name}")
    lines.append("\\begin{tabular}{l" + "r" * len(n_values) + "}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & " + " & ".join(f"\\textbf{{{n}}}" for n in n_values) + " \\\\")
    lines.append("\\midrule")

    method_names = {
        'tslf': 'TSL$_f$',
        'alergia': 'Alergia',
        'bc': 'BehavClone',
        'dt': 'CART'
    }

    for method in ['tslf', 'alergia', 'bc', 'dt']:
        row = [method_names[method]]
        for n in n_values:
            t = data.get(method, {}).get(n)
            row.append(format_time(t))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def main():
    print("Compiling timing data...\n")

    # Compile all games
    games = {
        'FrozenLake': compile_frozen_lake(),
        'CliffWalking': compile_cliff_walking(),
        'Taxi': compile_taxi(),
        'Blackjack': compile_blackjack(),
    }

    # Print summary
    for game_name, (data, n_values) in games.items():
        print(f"\n{'='*60}")
        print(f"{game_name}")
        print('='*60)
        print(f"{'Method':<12} | " + " | ".join(f"{n:>7}" for n in n_values))
        print("-"*60)
        for method in ['tslf', 'alergia', 'bc', 'dt']:
            row = [f"{method:<12}"]
            for n in n_values:
                t = data.get(method, {}).get(n)
                row.append(f"{format_time(t):>7}")
            print(" | ".join(row))

    # Generate LaTeX
    print("\n\n" + "="*60)
    print("LATEX OUTPUT")
    print("="*60)

    for game_name, (data, n_values) in games.items():
        print(f"\n% === {game_name} ===")
        print(generate_latex_table(game_name, data, n_values))

    # Save consolidated data
    output = {game: data for game, (data, _) in games.items()}
    output_file = Path(__file__).parent / "consolidated_timing.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nConsolidated data saved to: {output_file}")


if __name__ == "__main__":
    main()
