#!/usr/bin/env python3
"""
Plot combined results for FrozenLake comparing SpecMining(TSLf), BehavClon(NN), DecTree(wLR), and Q-Learning.

Visual encoding:
- Line style: dotted = test on var_config, solid = test on var_size
- Markers: circle (o) = train on fixed, diamond (D) = train on var
- Colors: SpecMining(TSLf) (blue), BehavClon(NN) (orange), DecTree(wLR) (green), Q-Learning (red)
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def load_tslf_bc_dt_results(json_path):
    """Load TSL_f, BC, DT results from combined_results JSON."""
    with open(json_path) as f:
        data = json.load(f)

    results = {}

    for train_mode in ['fixed', 'var_config', 'var_size']:
        if train_mode not in data:
            continue

        mode_data = data[train_mode]
        if 'results' not in mode_data:
            continue

        for entry in mode_data['results']:
            method = entry['method']
            n_train = entry['n_train']
            test_condition = entry['test_condition']
            success_rate = entry['success_rate']

            key = (method, train_mode, test_condition)
            if key not in results:
                results[key] = {}
            results[key][n_train] = success_rate

    return results


def load_single_results_file(json_path):
    """Load results from a single results_*.json file."""
    with open(json_path) as f:
        data = json.load(f)

    results = {}
    train_mode = data.get('train_mode', 'unknown')

    for entry in data.get('results', []):
        method = entry['method']
        n_train = entry['n_train']
        test_condition = entry['test_condition']
        success_rate = entry['success_rate']

        key = (method, train_mode, test_condition)
        if key not in results:
            results[key] = {}
        results[key][n_train] = success_rate

    return results


def load_all_var_size_results(base_dir):
    """Load all var_size results from multiple files."""
    results = {}

    # Load from var_size directory
    var_size_dir = base_dir / "var_size"
    if var_size_dir.exists():
        # var_size training results
        for f in var_size_dir.glob("results_*.json"):
            file_results = load_single_results_file(f)
            for key, values in file_results.items():
                if key not in results:
                    results[key] = {}
                results[key].update(values)

        # fixed training -> var_size test
        fixed_dir = var_size_dir / "fixed"
        if fixed_dir.exists():
            for f in fixed_dir.glob("results_*.json"):
                file_results = load_single_results_file(f)
                for key, values in file_results.items():
                    if key not in results:
                        results[key] = {}
                    results[key].update(values)

    return results


def load_qlearning_results(json_path):
    """Load Q-learning results from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    results = {}
    for exp_name, exp_data in data.items():
        for n_episodes, metrics in exp_data.items():
            n = int(n_episodes)
            acc = metrics['accuracy']
            if exp_name not in results:
                results[exp_name] = {}
            results[exp_name][n] = acc

    return results


def load_bc_dt_high_samples(json_path):
    """Load BC/DT high sample results."""
    if not json_path or not Path(json_path).exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    return data


def plot_combined_results(
    tslf_bc_dt_var_config_path,
    qlearning_path,
    bc_dt_high_samples_path=None,
    var_size_base_dir=None,
    output_path=None
):
    """
    Create combined plot with all methods.

    Visual encoding:
    - Line style: dotted = test on var_config, solid = test on var_size
    - Markers: circle (o) = train on fixed, diamond (D) = train on var
    - Colors: SpecMining (blue), BehavClon (orange), DecTree (green), Q-Learning (red)
    """

    # Load data
    var_config_data = load_tslf_bc_dt_results(tslf_bc_dt_var_config_path)
    qlearning_data = load_qlearning_results(qlearning_path)
    bc_dt_high = load_bc_dt_high_samples(bc_dt_high_samples_path)

    # Load var_size results
    var_size_data = {}
    if var_size_base_dir:
        var_size_data = load_all_var_size_results(var_size_base_dir)

    # Method name mapping
    method_names = {
        'tslf': 'SpecMining(TSLf)',
        'bc': 'BehavClon(NN)',
        'dt': 'DecTree(wLR)',
        'qlearning': 'Q-Learning'
    }

    # Colors for methods
    colors = {
        'tslf': '#2196F3',     # Blue
        'bc': '#FF9800',       # Orange
        'dt': '#4CAF50',       # Green
        'qlearning': '#F44336' # Red
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Helper to extend TSL_f 100% line
    def extend_tslf_to_max(n_vals, rates, max_x=50000):
        """If TSL_f reaches 100%, extend the line to max_x."""
        for i, r in enumerate(rates):
            if r >= 1.0:
                if n_vals[-1] < max_x:
                    return list(n_vals) + [max_x], list(rates) + [1.0]
                break
        return n_vals, rates

    # Helper to format label
    def make_label(method, train, test):
        return f"{method_names[method]}: {train}→{test}"

    # Store handles and labels for custom legend
    handles_by_method = {'tslf': [], 'bc': [], 'dt': [], 'qlearning': []}
    labels_by_method = {'tslf': [], 'bc': [], 'dt': [], 'qlearning': []}

    # ===========================================
    # PLOT LEGEND EXPLANATION:
    # - Dotted line (':'): test on var_config
    # - Solid line ('-'): test on var_size
    # - Circle marker ('o'): train on fixed
    # - Diamond marker ('D'): train on var
    # ===========================================

    # === TSL_f (SpecMining) ===
    # Fixed train -> var_config test (dotted, circle)
    if ('tslf', 'fixed', 'var_config') in var_config_data:
        data = var_config_data[('tslf', 'fixed', 'var_config')]
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        n_vals, rates = extend_tslf_to_max(n_vals, rates)
        h, = ax.plot(n_vals, rates, color=colors['tslf'], linestyle=':', marker='o',
                markersize=8, linewidth=2.5)
        handles_by_method['tslf'].append(h)
        labels_by_method['tslf'].append(make_label('tslf', 'fixed', 'var_config'))

    # Var train -> var_config test (dotted, diamond)
    if ('tslf', 'var_config', 'var_config') in var_config_data:
        data = var_config_data[('tslf', 'var_config', 'var_config')]
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        n_vals, rates = extend_tslf_to_max(n_vals, rates)
        h, = ax.plot(n_vals, rates, color=colors['tslf'], linestyle=':', marker='D',
                markersize=8, linewidth=2.5)
        handles_by_method['tslf'].append(h)
        labels_by_method['tslf'].append(make_label('tslf', 'var', 'var_config'))

    # Fixed train -> var_size test (solid, circle)
    if ('tslf', 'fixed', 'var_size') in var_size_data:
        data = var_size_data[('tslf', 'fixed', 'var_size')]
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        n_vals, rates = extend_tslf_to_max(n_vals, rates)
        h, = ax.plot(n_vals, rates, color=colors['tslf'], linestyle='-', marker='o',
                markersize=8, linewidth=2.5)
        handles_by_method['tslf'].append(h)
        labels_by_method['tslf'].append(make_label('tslf', 'fixed', 'var_size'))

    # Var train -> var_size test (solid, diamond)
    if ('tslf', 'var_size', 'var_size') in var_size_data:
        data = var_size_data[('tslf', 'var_size', 'var_size')]
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        n_vals, rates = extend_tslf_to_max(n_vals, rates)
        h, = ax.plot(n_vals, rates, color=colors['tslf'], linestyle='-', marker='D',
                markersize=8, linewidth=2.5)
        handles_by_method['tslf'].append(h)
        labels_by_method['tslf'].append(make_label('tslf', 'var', 'var_size'))

    # === BC (BehavClon) ===
    # Fixed train -> var_config test (dotted, circle)
    if ('bc', 'fixed', 'var_config') in var_config_data:
        data = dict(var_config_data[('bc', 'fixed', 'var_config')])
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        # Add high sample data if available
        if bc_dt_high and "fixed_to_var_config" in bc_dt_high:
            high_data = bc_dt_high["fixed_to_var_config"]["bc"]
            for n, v in high_data.items():
                n = int(n)
                if n not in data:
                    data[n] = v["accuracy"]
            n_vals = sorted(data.keys())
            rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['bc'], linestyle=':', marker='o',
                markersize=6, linewidth=1.5)
        handles_by_method['bc'].append(h)
        labels_by_method['bc'].append(make_label('bc', 'fixed', 'var_config'))

    # Var train -> var_config test (dotted, diamond)
    if ('bc', 'var_config', 'var_config') in var_config_data:
        data = dict(var_config_data[('bc', 'var_config', 'var_config')])
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        if bc_dt_high and "var_config_to_var_config" in bc_dt_high:
            high_data = bc_dt_high["var_config_to_var_config"]["bc"]
            for n, v in high_data.items():
                n = int(n)
                if n not in data:
                    data[n] = v["accuracy"]
            n_vals = sorted(data.keys())
            rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['bc'], linestyle=':', marker='D',
                markersize=6, linewidth=1.5)
        handles_by_method['bc'].append(h)
        labels_by_method['bc'].append(make_label('bc', 'var', 'var_config'))

    # Fixed train -> var_size test (solid, circle)
    bc_fixed_var_size = {}
    if ('bc', 'fixed', 'var_size') in var_size_data:
        bc_fixed_var_size.update(var_size_data[('bc', 'fixed', 'var_size')])
    if bc_dt_high and "fixed_to_var_size" in bc_dt_high:
        high_data = bc_dt_high["fixed_to_var_size"]["bc"]
        for n, v in high_data.items():
            bc_fixed_var_size[int(n)] = v["accuracy"]
    if bc_fixed_var_size:
        n_vals = sorted(bc_fixed_var_size.keys())
        rates = [bc_fixed_var_size[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['bc'], linestyle='-', marker='o',
                markersize=6, linewidth=1.5)
        handles_by_method['bc'].append(h)
        labels_by_method['bc'].append(make_label('bc', 'fixed', 'var_size'))

    # Var train -> var_size test (solid, diamond)
    bc_var_var_size = {}
    if ('bc', 'var_size', 'var_size') in var_size_data:
        bc_var_var_size.update(var_size_data[('bc', 'var_size', 'var_size')])
    if bc_dt_high and "var_size_to_var_size" in bc_dt_high:
        high_data = bc_dt_high["var_size_to_var_size"]["bc"]
        for n, v in high_data.items():
            bc_var_var_size[int(n)] = v["accuracy"]
    if bc_var_var_size:
        n_vals = sorted(bc_var_var_size.keys())
        rates = [bc_var_var_size[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['bc'], linestyle='-', marker='D',
                markersize=6, linewidth=1.5)
        handles_by_method['bc'].append(h)
        labels_by_method['bc'].append(make_label('bc', 'var', 'var_size'))

    # === DT (DecTree) ===
    # Fixed train -> var_config test (dotted, circle)
    if ('dt', 'fixed', 'var_config') in var_config_data:
        data = dict(var_config_data[('dt', 'fixed', 'var_config')])
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        if bc_dt_high and "fixed_to_var_config" in bc_dt_high:
            high_data = bc_dt_high["fixed_to_var_config"]["dt"]
            for n, v in high_data.items():
                n = int(n)
                if n not in data:
                    data[n] = v["accuracy"]
            n_vals = sorted(data.keys())
            rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['dt'], linestyle=':', marker='o',
                markersize=6, linewidth=1.5)
        handles_by_method['dt'].append(h)
        labels_by_method['dt'].append(make_label('dt', 'fixed', 'var_config'))

    # Var train -> var_config test (dotted, diamond)
    if ('dt', 'var_config', 'var_config') in var_config_data:
        data = dict(var_config_data[('dt', 'var_config', 'var_config')])
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        if bc_dt_high and "var_config_to_var_config" in bc_dt_high:
            high_data = bc_dt_high["var_config_to_var_config"]["dt"]
            for n, v in high_data.items():
                n = int(n)
                if n not in data:
                    data[n] = v["accuracy"]
            n_vals = sorted(data.keys())
            rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['dt'], linestyle=':', marker='D',
                markersize=6, linewidth=1.5)
        handles_by_method['dt'].append(h)
        labels_by_method['dt'].append(make_label('dt', 'var', 'var_config'))

    # Fixed train -> var_size test (solid, circle)
    dt_fixed_var_size = {}
    if ('dt', 'fixed', 'var_size') in var_size_data:
        dt_fixed_var_size.update(var_size_data[('dt', 'fixed', 'var_size')])
    if bc_dt_high and "fixed_to_var_size" in bc_dt_high:
        high_data = bc_dt_high["fixed_to_var_size"]["dt"]
        for n, v in high_data.items():
            dt_fixed_var_size[int(n)] = v["accuracy"]
    if dt_fixed_var_size:
        n_vals = sorted(dt_fixed_var_size.keys())
        rates = [dt_fixed_var_size[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['dt'], linestyle='-', marker='o',
                markersize=6, linewidth=1.5)
        handles_by_method['dt'].append(h)
        labels_by_method['dt'].append(make_label('dt', 'fixed', 'var_size'))

    # Var train -> var_size test (solid, diamond)
    dt_var_var_size = {}
    if ('dt', 'var_size', 'var_size') in var_size_data:
        dt_var_var_size.update(var_size_data[('dt', 'var_size', 'var_size')])
    if bc_dt_high and "var_size_to_var_size" in bc_dt_high:
        high_data = bc_dt_high["var_size_to_var_size"]["dt"]
        for n, v in high_data.items():
            dt_var_var_size[int(n)] = v["accuracy"]
    if dt_var_var_size:
        n_vals = sorted(dt_var_var_size.keys())
        rates = [dt_var_var_size[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['dt'], linestyle='-', marker='D',
                markersize=6, linewidth=1.5)
        handles_by_method['dt'].append(h)
        labels_by_method['dt'].append(make_label('dt', 'var', 'var_size'))

    # === Q-Learning ===
    # Fixed train -> var_config test (dotted, circle)
    if 'Fixed → Var Config' in qlearning_data:
        data = qlearning_data['Fixed → Var Config']
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['qlearning'], linestyle=':', marker='o',
                markersize=6, linewidth=1.5)
        handles_by_method['qlearning'].append(h)
        labels_by_method['qlearning'].append(make_label('qlearning', 'fixed', 'var_config'))

    # Var train -> var_config test (dotted, diamond)
    if 'Var Config → Var Config' in qlearning_data:
        data = qlearning_data['Var Config → Var Config']
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['qlearning'], linestyle=':', marker='D',
                markersize=6, linewidth=1.5)
        handles_by_method['qlearning'].append(h)
        labels_by_method['qlearning'].append(make_label('qlearning', 'var', 'var_config'))

    # Fixed train -> var_size test (solid, circle)
    if 'Fixed → Var Size' in qlearning_data:
        data = qlearning_data['Fixed → Var Size']
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['qlearning'], linestyle='-', marker='o',
                markersize=6, linewidth=1.5)
        handles_by_method['qlearning'].append(h)
        labels_by_method['qlearning'].append(make_label('qlearning', 'fixed', 'var_size'))

    # Var train -> var_size test (solid, diamond)
    if 'Var Size → Var Size' in qlearning_data:
        data = qlearning_data['Var Size → Var Size']
        n_vals = sorted(data.keys())
        rates = [data[n] for n in n_vals]
        h, = ax.plot(n_vals, rates, color=colors['qlearning'], linestyle='-', marker='D',
                markersize=6, linewidth=1.5)
        handles_by_method['qlearning'].append(h)
        labels_by_method['qlearning'].append(make_label('qlearning', 'var', 'var_size'))

    # Formatting
    ax.set_xlabel('Training Samples (demonstrations for passive methods, episodes for Q-learning)',
                  fontsize=11)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title('FrozenLake: Spec Mining vs Baselines', fontsize=14)

    ax.set_xscale('log')
    ax.set_xlim(1, 60000)
    ax.set_ylim(0, 1.05)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.3)

    ax.grid(True, alpha=0.3)

    # Create 4-column legend (one column per method)
    # Build legend with all entries organized by method
    all_handles = []
    all_labels = []

    # Determine max rows needed
    max_rows = max(len(handles_by_method[m]) for m in ['tslf', 'bc', 'dt', 'qlearning'])

    # Pad shorter columns with empty entries
    for method in ['tslf', 'bc', 'dt', 'qlearning']:
        while len(handles_by_method[method]) < max_rows:
            # Create invisible handle
            h, = ax.plot([], [], color='none', linestyle='none')
            handles_by_method[method].append(h)
            labels_by_method[method].append('')

    # Interleave for 4-column layout (row by row)
    for row in range(max_rows):
        for method in ['tslf', 'bc', 'dt', 'qlearning']:
            all_handles.append(handles_by_method[method][row])
            all_labels.append(labels_by_method[method][row])

    ax.legend(all_handles, all_labels, loc='lower left', fontsize=7, ncol=4,
              columnspacing=1.0, handletextpad=0.5)

    # Add annotations explaining visual encoding (with colors)
    ax.annotate('Passive Learning\n(SpecMining, BehavClon, DecTree)', xy=(15, 0.65),
                fontsize=9, ha='center', style='italic', alpha=0.7)
    ax.annotate('Active Learning\n(Q-Learning)', xy=(15000, 0.50),
                fontsize=9, ha='center', style='italic', alpha=0.7)

    # Add legend key for line styles, markers, AND colors
    legend_text = ('Line: dotted=test var_config, solid=test var_size\n'
                   'Marker: o=train fixed, ◇=train var\n'
                   'Color: Blue=SpecMining, Orange=BehavClon, Green=DecTree, Red=Q-Learn')
    ax.text(0.98, 0.98, legend_text,
            transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # File paths
    base_dir = Path("frozen_lake/full_eval")
    var_config_results = base_dir / "var_config/combined_results_20260122_010849.json"
    qlearning_results = Path("frozen_lake/qlearning/results_20260122_115215.json")

    # Check for high sample BC/DT results
    bc_dt_high_path = Path("frozen_lake/baselines")
    bc_dt_files = sorted(bc_dt_high_path.glob("bc_dt_high_samples_*.json")) if bc_dt_high_path.exists() else []
    bc_dt_high_samples = str(bc_dt_files[-1]) if bc_dt_files else None

    if bc_dt_high_samples:
        print(f"Using high sample BC/DT data: {bc_dt_high_samples}")
    else:
        print("No high sample BC/DT data found")

    # Plot combined results
    plot_combined_results(
        tslf_bc_dt_var_config_path=var_config_results,
        qlearning_path=qlearning_results,
        bc_dt_high_samples_path=bc_dt_high_samples,
        var_size_base_dir=base_dir,
        output_path="frozen_lake/combined_all_methods_plot.png"
    )
