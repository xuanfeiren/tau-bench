#!/usr/bin/env python3
"""
Plot Test Scores vs Steps

This script:
1. Fetches data from wandb project "tau-bench-10-tasks-10-evals" and local files
2. For each run, collects (test_score, step) pairs
3. Calculates cumulative best score at each step
4. Forward-fills: for each unique step across all runs of an algorithm, fills missing data
   with the last known best score
5. Plots test score (y) vs steps (x) with mean and min/max shaded area

Step definitions:
- DSPy GEPA: step = iteration
- OpenEvolve: step = checkpoint / 4 (due to parallel_evaluations=4)
- PS algorithms: step = _step from wandb
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb package is required. Install with: pip install wandb")
    exit(1)

# Algorithm configurations
ALGORITHMS = {
    'DSPy_GEPA_most_frequent': {
        'display_name': 'DSPy GEPA (most freq)',
        'color': '#ff7f0e',
        'linestyle': '--',
        'linewidth': 2.0,
        'local_dir_pattern': 'dspy_results/gepa_Nov25_{run_num}/eval_results/eval_results_most_frequent.json',
        'num_runs': 6
    },
    'DSPy_GEPA_sample_by_freq': {
        'display_name': 'DSPy GEPA (sample freq)',
        'color': '#ff7f0e',
        'linestyle': ':',
        'linewidth': 2.0,
        'local_dir_pattern': 'dspy_results/gepa_Nov25_{run_num}/eval_results/eval_results_sample_by_freq.json',
        'num_runs': 6
    },
    'OpenEvolve': {
        'display_name': 'OpenEvolve',
        'color': '#2ca02c',
        'linestyle': '--',
        'linewidth': 2.0,
        'local_dir_pattern': 'results/openevolve_tau/data_run{run_num}.json',
        'num_runs': 6
    },
    'PS-mean_exploration-Nov7': {
        'display_name': 'Vanilla PS',
        'color': '#b0b0b0',
        'linestyle': '-',
        'linewidth': 3.0
    },
    'epsnet_0-PS_with_summarizer': {
        'display_name': 'PS+Summarizer',
        'color': '#d62728',
        'linestyle': '-',
        'linewidth': 3.0
    },
    'epsnet_0.1_PS': {
        'display_name': r'PS+$\varepsilon$-Net',
        'color': '#1f77b4',
        'linestyle': '-',
        'linewidth': 3.0
    },
    'epsnet_0.1-PS_with_summarizer': {
        'display_name': r'PS+$\varepsilon$-Net + Summarizer',
        'color': '#9467bd',
        'linestyle': '-',
        'linewidth': 3.0
    }
}

PROJECT_NAME = "tau-bench-10-tasks-10-evals"
SCORE_METRIC = "Test/test_score_empirical_mean"


def load_local_data():
    """
    Load local evaluation data from JSON files (GEPA and OpenEvolve).

    Returns:
        DataFrame with columns: algorithm, run_id, step, score
    """
    print("\nLoading local data (GEPA, OpenEvolve, etc.)...")

    all_data = []

    for algorithm, config in ALGORITHMS.items():
        if 'local_dir_pattern' not in config:
            continue

        num_runs = config.get('num_runs', 1)
        pattern = config['local_dir_pattern']
        runs_loaded = 0

        print(f"  {algorithm}: checking {num_runs} potential runs...")

        for run_num in range(1, num_runs + 1):
            filepath = pattern.format(run_num=run_num)

            if not os.path.exists(filepath):
                continue

            print(f"    Loading run {run_num}: {filepath}")

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"      Warning: Invalid JSON in {filepath}: {e}")
                continue

            run_id = f'{algorithm}_run{run_num}'

            if algorithm == 'OpenEvolve':
                # OpenEvolve format: step = checkpoint / 4
                for entry in data:
                    checkpoint = entry.get('checkpoint', 0)
                    # Skip 'final' checkpoint or handle it
                    if checkpoint == 'final':
                        continue
                    step = int(checkpoint) // 4  # parallel_evaluations = 4
                    all_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'step': step,
                        'score': float(entry['test_score'])
                    })
            else:
                # GEPA format: step = iteration
                for entry in data:
                    all_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'step': entry.get('iteration', 0),
                        'score': float(entry['score'])
                    })

            runs_loaded += 1
            print(f"      Loaded {len(data)} data points")

        print(f"    Total runs loaded for {algorithm}: {runs_loaded}")

    if not all_data:
        print("  No local data found")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"Total local data points: {len(df)}")

    return df


def fetch_wandb_data():
    """
    Fetch data from wandb for the PS algorithms.

    Returns:
        DataFrame with columns: algorithm, run_id, step, score
    """
    print(f"Connecting to wandb project: {PROJECT_NAME}")
    api = wandb.Api()

    try:
        runs = api.runs(f"xuanfeiren-university-of-wisconsin-madison/{PROJECT_NAME}")
    except Exception as e:
        print(f"Error connecting to wandb: {e}")
        print("Make sure you're logged in with: wandb login")
        exit(1)

    all_data = []
    wandb_algorithms = {alg: config for alg, config in ALGORITHMS.items() if 'local_dir_pattern' not in config}
    run_counts = {alg: 0 for alg in wandb_algorithms.keys()}

    print("\nFetching runs...")
    for run in runs:
        run_name = run.name

        if run_name not in wandb_algorithms:
            continue

        algorithm = run_name
        run_counts[algorithm] += 1

        print(f"  Processing: {run_name} (Run ID: {run.id})")

        try:
            history = run.history()
        except Exception as e:
            print(f"    Warning: Could not fetch history for {run_name}: {e}")
            continue

        if SCORE_METRIC not in history.columns:
            print(f"    Warning: Metric '{SCORE_METRIC}' not found in run {run_name}")
            continue

        # Extract data points using _step as the step
        for idx, row in history.iterrows():
            score = row.get(SCORE_METRIC)
            step = row.get('_step', idx)

            if pd.notna(score):
                all_data.append({
                    'algorithm': algorithm,
                    'run_id': run.id,
                    'step': int(step),
                    'score': float(score)
                })

    print("\n=== Run Counts by Algorithm ===")
    for algorithm, count in run_counts.items():
        print(f"{algorithm}: {count} runs")

    if not all_data:
        print("\nNo wandb data found for the specified algorithms.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"\nTotal wandb data points fetched: {len(df)}")

    print("\n=== Step Ranges Per Algorithm (wandb) ===")
    for algorithm in wandb_algorithms.keys():
        alg_data = df[df['algorithm'] == algorithm]
        if not alg_data.empty:
            min_step = alg_data['step'].min()
            max_step = alg_data['step'].max()
            print(f"{algorithm}: {min_step} - {max_step} steps")

    return df


def calculate_best_score_so_far(df):
    """
    Calculate cumulative maximum (best score so far) for each run.
    Uses step as the progression axis.

    Args:
        df: DataFrame with columns: algorithm, run_id, step, score

    Returns:
        DataFrame with added column: best_score_so_far
    """
    print("\nCalculating best score so far for each run...")

    results = []

    for (algorithm, run_id), group in df.groupby(['algorithm', 'run_id']):
        # Sort by step to ensure correct order
        group = group.sort_values('step').copy()

        # Calculate cumulative maximum based on step order
        group['best_score_so_far'] = group['score'].cummax()

        results.append(group)

    result_df = pd.concat(results, ignore_index=True)
    print(f"Calculated cumulative max for {len(result_df)} data points")

    return result_df


def forward_fill_by_steps(df):
    """
    Forward-fill best scores based on step counts.
    For each algorithm, creates a common step grid and forward-fills.

    Args:
        df: DataFrame with columns: algorithm, run_id, step, best_score_so_far

    Returns:
        DataFrame with forward-filled data at all step points per algorithm
    """
    print("\nForward-filling by step counts...")

    filled_data = []

    # Process each algorithm separately
    for algorithm in df['algorithm'].unique():
        alg_df = df[df['algorithm'] == algorithm]

        # Get all unique steps for THIS algorithm only
        algorithm_steps = sorted(alg_df['step'].unique())
        max_step_for_alg = max(algorithm_steps)

        print(f"\n{algorithm}:")
        print(f"  Unique steps: {len(algorithm_steps)}")
        print(f"  Step range: {min(algorithm_steps)} to {max_step_for_alg}")

        # Process each run within this algorithm
        for run_id in alg_df['run_id'].unique():
            run_data_df = alg_df[alg_df['run_id'] == run_id].sort_values('step')

            existing_steps = set(run_data_df['step'].values)
            min_step = run_data_df['step'].min()
            max_step = run_data_df['step'].max()

            run_data = []
            last_known_score = None

            for step in algorithm_steps:
                if step in existing_steps:
                    row = run_data_df[run_data_df['step'] == step].iloc[0]
                    last_known_score = row['best_score_so_far']
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'step': step,
                        'best_score_so_far': last_known_score
                    })
                elif step >= min_step and step <= max_step and last_known_score is not None:
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'step': step,
                        'best_score_so_far': last_known_score
                    })
                elif step > max_step and last_known_score is not None:
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'step': step,
                        'best_score_so_far': last_known_score
                    })

            if run_data:
                filled_data.append(pd.DataFrame(run_data))

    result_df = pd.concat(filled_data, ignore_index=True)
    print(f"\nTotal data points after forward-fill: {len(result_df)}")

    print("\n=== Per-Algorithm Step Counts After Forward-Fill ===")
    for algorithm in result_df['algorithm'].unique():
        alg_data = result_df[result_df['algorithm'] == algorithm]
        unique_steps = len(alg_data['step'].unique())
        max_step = alg_data['step'].max()
        print(f"{algorithm}: {unique_steps} unique steps, max={int(max_step)}")

    return result_df


def aggregate_across_runs(df):
    """
    Calculate mean, min, max across multiple runs for each algorithm and step.

    Args:
        df: DataFrame with columns: algorithm, run_id, step, best_score_so_far

    Returns:
        DataFrame with aggregated statistics
    """
    print("\nAggregating statistics across runs...")

    aggregated = df.groupby(['algorithm', 'step'])['best_score_so_far'].agg([
        ('mean_score', 'mean'),
        ('min_score', 'min'),
        ('max_score', 'max'),
        ('n_runs', 'count'),
        ('std_score', 'std')
    ]).reset_index()

    print(f"Aggregated into {len(aggregated)} data points")

    print("\n=== Step Counts Per Algorithm ===")
    for algorithm in ALGORITHMS.keys():
        alg_data = aggregated[aggregated['algorithm'] == algorithm]
        if not alg_data.empty:
            print(f"{algorithm}: {len(alg_data)} unique steps")

    return aggregated


def create_plot(stats_df, output_file='scores_vs_steps.pdf'):
    """
    Create publication-quality plot showing best scores vs steps.
    X-axis range: 0 to the minimum max-step across all algorithms.

    Args:
        stats_df: DataFrame with aggregated statistics
        output_file: Output filename (PDF)
    """
    print("\nCreating plot...")

    # Find the minimum max-step across all algorithms
    max_steps_per_alg = {}
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm]
        if not alg_data.empty:
            max_steps_per_alg[algorithm] = alg_data['step'].max()

    if not max_steps_per_alg:
        print("Error: No data to plot")
        return

    # X-axis limit is the minimum of all max steps
    x_limit = max(max_steps_per_alg.values())
    x_limit = 100
    print(f"X-axis limit (min of max steps): {x_limit}")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each algorithm
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('step')

        if alg_data.empty:
            print(f"  Warning: No data for {algorithm}")
            continue

        # Filter data to be within x_limit
        alg_data = alg_data[alg_data['step'] <= x_limit]

        if alg_data.empty:
            print(f"  Warning: No data for {algorithm} within x_limit")
            continue

        config = ALGORITHMS[algorithm]

        x = alg_data['step'].values
        y_mean = alg_data['mean_score'].values
        y_std = alg_data['std_score'].fillna(0).values
        n_runs = alg_data['n_runs'].values

        # Calculate standard error: std / sqrt(n)
        y_stderr = y_std / np.sqrt(n_runs)
        y_lower = y_mean - y_stderr
        y_upper = y_mean + y_stderr

        # Downsample for plotting if too many points
        if len(x) > 200:
            step_size = len(x) // 200
            indices = list(range(0, len(x), step_size))
            if (len(x) - 1) not in indices:
                indices.append(len(x) - 1)

            x_plot = x[indices]
            y_mean_plot = y_mean[indices]
            y_lower_plot = y_lower[indices]
            y_upper_plot = y_upper[indices]
        else:
            x_plot = x
            y_mean_plot = y_mean
            y_lower_plot = y_lower
            y_upper_plot = y_upper

        # Plot mean line
        ax.plot(x_plot, y_mean_plot,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                marker='o',
                markersize=4,
                markevery=max(1, len(x_plot) // 20),
                label=config['display_name'],
                zorder=3)

        # Add shaded area for standard error range
        ax.fill_between(x, y_lower, y_upper,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)

        print(f"  Plotted {algorithm}: {len(x)} step points (displayed {len(x_plot)})")

    # Customize plot
    ax.set_xlabel('Steps', fontsize=28, fontweight='bold')
    ax.set_ylabel('Test Score', fontsize=28, fontweight='bold')
    ax.set_title(r'$\tau$-bench Test Score Comparison (mean ± SE)', fontsize=30, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='lower right', frameon=True, fancybox=True, shadow=True)

    # Set x-axis limit
    ax.set_xlim(left=0, right=x_limit)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=22)

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()

    # Save as PDF and PNG
    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')

    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')

    print(f"\nPlot saved as:")
    print(f"  - {pdf_file}")
    print(f"  - {png_file}")

    plt.show()


def print_summary_statistics(stats_df):
    """
    Print summary statistics for the data.

    Args:
        stats_df: Aggregated statistics DataFrame
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm]

        if alg_data.empty:
            continue

        # Get final scores (at max step)
        final_data = alg_data[alg_data['step'] == alg_data['step'].max()]
        if not final_data.empty:
            final_step = final_data['step'].iloc[0]
            final_mean = final_data['mean_score'].iloc[0]
            final_std = final_data['std_score'].iloc[0] if pd.notna(final_data['std_score'].iloc[0]) else 0
            final_min = final_data['min_score'].iloc[0]
            final_max = final_data['max_score'].iloc[0]
        else:
            final_step = final_mean = final_std = final_min = final_max = 0

        print(f"\n{ALGORITHMS[algorithm]['display_name']} ({algorithm}):")
        print(f"  Step points: {len(alg_data)}")
        print(f"  Max step: {int(final_step)}")
        print(f"  Final best score (mean): {final_mean:.4f}")
        print(f"  Final best score (std):  {final_std:.4f}")
        print(f"  Final best score (min):  {final_min:.4f}")
        print(f"  Final best score (max):  {final_max:.4f}")
        print(f"  Score range: {alg_data['mean_score'].min():.4f} - {alg_data['mean_score'].max():.4f}")


def main():
    """Main execution function."""
    print("="*70)
    print("TEST SCORE VS STEPS PLOTTING SCRIPT")
    print("="*70)

    # Step 1: Fetch data from wandb
    wandb_df = fetch_wandb_data()

    # Step 1b: Load local data (GEPA, OpenEvolve)
    local_df = load_local_data()

    # Merge data sources
    dfs_to_concat = [df for df in [wandb_df, local_df] if not df.empty]
    if not dfs_to_concat:
        print("\nError: No data found from any source.")
        exit(1)
    raw_df = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"\nTotal combined data points: {len(raw_df)}")

    # Step 2: Calculate best score so far (cumulative max) for each run
    df_with_cummax = calculate_best_score_so_far(raw_df)

    # Step 3: Forward-fill by step counts
    df_filled = forward_fill_by_steps(df_with_cummax)

    # Step 4: Aggregate across runs
    stats_df = aggregate_across_runs(df_filled)

    # Step 5: Create the plot
    create_plot(stats_df, output_file='scores_vs_steps.pdf')

    # Step 6: Print summary statistics
    print_summary_statistics(stats_df)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("Plot saved:")
    print("  - scores_vs_steps.pdf")
    print("  - scores_vs_steps.png")
    print("="*70)


if __name__ == "__main__":
    main()
