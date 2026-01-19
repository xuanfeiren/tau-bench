#!/usr/bin/env python3
"""
Plot PS Ablation Study - Test Scores vs Total Samples AND Steps

This script:
1. Fetches data from wandb project "tau-bench-10-tasks-10-evals"
2. Only plots the 4 PS-related algorithms for ablation study
3. Generates two plots:
   - ps_ablation_samples.pdf: Test score vs Total Samples
   - ps_ablation_steps.pdf: Test score vs Steps
4. Uses the min of all max values as x-axis range for clean comparison
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

# Only PS-related algorithms for ablation study
ALGORITHMS = {
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
SAMPLES_METRIC = "Update/total_samples"


def fetch_wandb_data():
    """
    Fetch data from wandb for the PS algorithms.

    Returns:
        DataFrame with columns: algorithm, run_id, step, score, num_samples
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
    run_counts = {alg: 0 for alg in ALGORITHMS.keys()}

    print("\nFetching runs...")
    for run in runs:
        run_name = run.name

        if run_name not in ALGORITHMS:
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

        if SAMPLES_METRIC not in history.columns:
            print(f"    Warning: Metric '{SAMPLES_METRIC}' not found in run {run_name}")
            continue

        for idx, row in history.iterrows():
            score = row.get(SCORE_METRIC)
            num_samples = row.get(SAMPLES_METRIC)
            step = row.get('_step', idx)

            if pd.notna(score) and pd.notna(num_samples):
                all_data.append({
                    'algorithm': algorithm,
                    'run_id': run.id,
                    'step': int(step),
                    'score': float(score),
                    'num_samples': int(num_samples)
                })

    print("\n=== Run Counts by Algorithm ===")
    for algorithm, count in run_counts.items():
        print(f"{algorithm}: {count} runs")

    if not all_data:
        print("\nNo wandb data found for the specified algorithms.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"\nTotal wandb data points fetched: {len(df)}")

    print("\n=== Sample Ranges Per Algorithm ===")
    for algorithm in ALGORITHMS.keys():
        alg_data = df[df['algorithm'] == algorithm]
        if not alg_data.empty:
            min_samples = alg_data['num_samples'].min()
            max_samples = alg_data['num_samples'].max()
            min_step = alg_data['step'].min()
            max_step = alg_data['step'].max()
            print(f"{algorithm}: samples {min_samples}-{max_samples}, steps {min_step}-{max_step}")

    return df


# ============== SAMPLES-BASED PROCESSING ==============

def calculate_best_score_by_samples(df):
    """Calculate cumulative maximum (best score so far) for each run by samples."""
    print("\nCalculating best score so far by samples...")

    results = []
    for (algorithm, run_id), group in df.groupby(['algorithm', 'run_id']):
        group = group.sort_values('num_samples').copy()
        group['best_score_so_far'] = group['score'].cummax()
        results.append(group)

    result_df = pd.concat(results, ignore_index=True)
    print(f"Calculated cumulative max for {len(result_df)} data points")
    return result_df


def forward_fill_by_samples(df):
    """Forward-fill best scores based on sample counts."""
    print("\nForward-filling by sample counts...")

    filled_data = []

    for algorithm in df['algorithm'].unique():
        alg_df = df[df['algorithm'] == algorithm]
        algorithm_sample_counts = sorted(alg_df['num_samples'].unique())

        for run_id in alg_df['run_id'].unique():
            run_data_df = alg_df[alg_df['run_id'] == run_id].sort_values('num_samples')
            existing_samples = set(run_data_df['num_samples'].values)
            min_samples = run_data_df['num_samples'].min()
            max_samples = run_data_df['num_samples'].max()

            run_data = []
            last_known_score = None

            for sample_count in algorithm_sample_counts:
                if sample_count in existing_samples:
                    row = run_data_df[run_data_df['num_samples'] == sample_count].iloc[0]
                    last_known_score = row['best_score_so_far']
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'num_samples': sample_count,
                        'best_score_so_far': last_known_score
                    })
                elif sample_count >= min_samples and last_known_score is not None:
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'num_samples': sample_count,
                        'best_score_so_far': last_known_score
                    })

            if run_data:
                filled_data.append(pd.DataFrame(run_data))

    result_df = pd.concat(filled_data, ignore_index=True)
    print(f"Total data points after forward-fill: {len(result_df)}")
    return result_df


def aggregate_by_samples(df):
    """Aggregate statistics by samples."""
    print("\nAggregating statistics by samples...")

    aggregated = df.groupby(['algorithm', 'num_samples'])['best_score_so_far'].agg([
        ('mean_score', 'mean'),
        ('min_score', 'min'),
        ('max_score', 'max'),
        ('n_runs', 'count'),
        ('std_score', 'std')
    ]).reset_index()

    print(f"Aggregated into {len(aggregated)} data points")
    return aggregated


# ============== STEPS-BASED PROCESSING ==============

def calculate_best_score_by_steps(df):
    """Calculate cumulative maximum (best score so far) for each run by steps."""
    print("\nCalculating best score so far by steps...")

    results = []
    for (algorithm, run_id), group in df.groupby(['algorithm', 'run_id']):
        group = group.sort_values('step').copy()
        group['best_score_so_far'] = group['score'].cummax()
        results.append(group)

    result_df = pd.concat(results, ignore_index=True)
    print(f"Calculated cumulative max for {len(result_df)} data points")
    return result_df


def forward_fill_by_steps(df):
    """Forward-fill best scores based on step counts."""
    print("\nForward-filling by step counts...")

    filled_data = []

    for algorithm in df['algorithm'].unique():
        alg_df = df[df['algorithm'] == algorithm]
        algorithm_steps = sorted(alg_df['step'].unique())

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
                elif step >= min_step and last_known_score is not None:
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'step': step,
                        'best_score_so_far': last_known_score
                    })

            if run_data:
                filled_data.append(pd.DataFrame(run_data))

    result_df = pd.concat(filled_data, ignore_index=True)
    print(f"Total data points after forward-fill: {len(result_df)}")
    return result_df


def aggregate_by_steps(df):
    """Aggregate statistics by steps."""
    print("\nAggregating statistics by steps...")

    aggregated = df.groupby(['algorithm', 'step'])['best_score_so_far'].agg([
        ('mean_score', 'mean'),
        ('min_score', 'min'),
        ('max_score', 'max'),
        ('n_runs', 'count'),
        ('std_score', 'std')
    ]).reset_index()

    print(f"Aggregated into {len(aggregated)} data points")
    return aggregated


# ============== PLOTTING ==============

def create_plot_samples(stats_df, output_file='ps_ablation_samples.pdf'):
    """Create plot showing best scores vs total samples."""
    print("\nCreating samples plot...")

    # Find the minimum max-samples across all algorithms
    max_samples_per_alg = {}
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm]
        if not alg_data.empty:
            max_samples_per_alg[algorithm] = alg_data['num_samples'].max()

    if not max_samples_per_alg:
        print("Error: No data to plot")
        return

    x_limit = min(max_samples_per_alg.values())
    print(f"X-axis limit (min of max samples): {x_limit}")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))

    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('num_samples')

        if alg_data.empty:
            continue

        alg_data = alg_data[alg_data['num_samples'] <= x_limit]
        if alg_data.empty:
            continue

        config = ALGORITHMS[algorithm]

        x = alg_data['num_samples'].values
        y_mean = alg_data['mean_score'].values
        y_std = alg_data['std_score'].fillna(0).values
        n_runs = alg_data['n_runs'].values

        y_stderr = y_std / np.sqrt(n_runs)
        y_lower = y_mean - y_stderr
        y_upper = y_mean + y_stderr

        if len(x) > 200:
            step_size = len(x) // 200
            indices = list(range(0, len(x), step_size))
            if (len(x) - 1) not in indices:
                indices.append(len(x) - 1)
            x_plot = x[indices]
            y_mean_plot = y_mean[indices]
        else:
            x_plot = x
            y_mean_plot = y_mean

        ax.plot(x_plot, y_mean_plot,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                marker='o',
                markersize=4,
                markevery=max(1, len(x_plot) // 20),
                label=config['display_name'],
                zorder=3)

        ax.fill_between(x, y_lower, y_upper,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)

        print(f"  Plotted {algorithm}: {len(x)} points")

    ax.set_xlabel('Total Samples', fontsize=28, fontweight='bold')
    ax.set_ylabel('Test Score', fontsize=28, fontweight='bold')
    ax.set_title(r'PS Ablation Study on $\tau$-bench (mean ± SE)', fontsize=30, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(left=0, right=x_limit)
    ax.tick_params(axis='both', which='major', labelsize=22)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()

    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')

    print(f"  Saved: {pdf_file}, {png_file}")
    plt.close()


def create_plot_steps(stats_df, output_file='ps_ablation_steps.pdf'):
    """Create plot showing best scores vs steps."""
    print("\nCreating steps plot...")

    # Find the minimum max-steps across all algorithms
    max_steps_per_alg = {}
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm]
        if not alg_data.empty:
            max_steps_per_alg[algorithm] = alg_data['step'].max()

    if not max_steps_per_alg:
        print("Error: No data to plot")
        return

    x_limit = min(max_steps_per_alg.values())
    print(f"X-axis limit (min of max steps): {x_limit}")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))

    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('step')

        if alg_data.empty:
            continue

        alg_data = alg_data[alg_data['step'] <= x_limit]
        if alg_data.empty:
            continue

        config = ALGORITHMS[algorithm]

        x = alg_data['step'].values
        y_mean = alg_data['mean_score'].values
        y_std = alg_data['std_score'].fillna(0).values
        n_runs = alg_data['n_runs'].values

        y_stderr = y_std / np.sqrt(n_runs)
        y_lower = y_mean - y_stderr
        y_upper = y_mean + y_stderr

        if len(x) > 200:
            step_size = len(x) // 200
            indices = list(range(0, len(x), step_size))
            if (len(x) - 1) not in indices:
                indices.append(len(x) - 1)
            x_plot = x[indices]
            y_mean_plot = y_mean[indices]
        else:
            x_plot = x
            y_mean_plot = y_mean

        ax.plot(x_plot, y_mean_plot,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                marker='o',
                markersize=4,
                markevery=max(1, len(x_plot) // 20),
                label=config['display_name'],
                zorder=3)

        ax.fill_between(x, y_lower, y_upper,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)

        print(f"  Plotted {algorithm}: {len(x)} points")

    ax.set_xlabel('Steps', fontsize=28, fontweight='bold')
    ax.set_ylabel('Test Score', fontsize=28, fontweight='bold')
    ax.set_title(r'PS Ablation Study on $\tau$-bench (mean ± SE)', fontsize=30, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(left=0, right=x_limit)
    ax.tick_params(axis='both', which='major', labelsize=22)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()

    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')

    print(f"  Saved: {pdf_file}, {png_file}")
    plt.close()


def print_summary_statistics(stats_samples_df, stats_steps_df):
    """Print summary statistics for the data."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS - PS ABLATION")
    print("="*70)

    for algorithm in ALGORITHMS.keys():
        alg_samples = stats_samples_df[stats_samples_df['algorithm'] == algorithm]
        alg_steps = stats_steps_df[stats_steps_df['algorithm'] == algorithm]

        if alg_samples.empty:
            continue

        final_samples = alg_samples[alg_samples['num_samples'] == alg_samples['num_samples'].max()]
        final_steps = alg_steps[alg_steps['step'] == alg_steps['step'].max()]

        print(f"\n{ALGORITHMS[algorithm]['display_name']} ({algorithm}):")

        if not final_samples.empty:
            print(f"  Max samples: {int(final_samples['num_samples'].iloc[0])}")
            print(f"  Final score (by samples): {final_samples['mean_score'].iloc[0]:.4f}")

        if not final_steps.empty:
            print(f"  Max steps: {int(final_steps['step'].iloc[0])}")
            print(f"  Final score (by steps): {final_steps['mean_score'].iloc[0]:.4f}")


def main():
    """Main execution function."""
    print("="*70)
    print("PS ABLATION STUDY - TEST SCORE VS SAMPLES AND STEPS")
    print("="*70)

    # Fetch data from wandb
    raw_df = fetch_wandb_data()

    if raw_df.empty:
        print("\nError: No data found.")
        exit(1)

    print(f"\nTotal data points: {len(raw_df)}")

    # ===== Process and plot by SAMPLES =====
    print("\n" + "="*50)
    print("PROCESSING BY SAMPLES")
    print("="*50)

    df_samples_cummax = calculate_best_score_by_samples(raw_df)
    df_samples_filled = forward_fill_by_samples(df_samples_cummax)
    stats_samples_df = aggregate_by_samples(df_samples_filled)
    create_plot_samples(stats_samples_df, output_file='ps_ablation_samples.pdf')

    # ===== Process and plot by STEPS =====
    print("\n" + "="*50)
    print("PROCESSING BY STEPS")
    print("="*50)

    df_steps_cummax = calculate_best_score_by_steps(raw_df)
    df_steps_filled = forward_fill_by_steps(df_steps_cummax)
    stats_steps_df = aggregate_by_steps(df_steps_filled)
    create_plot_steps(stats_steps_df, output_file='ps_ablation_steps.pdf')

    # Print summary statistics
    print_summary_statistics(stats_samples_df, stats_steps_df)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("Plots saved:")
    print("  - ps_ablation_samples.pdf / .png")
    print("  - ps_ablation_steps.pdf / .png")
    print("="*70)


if __name__ == "__main__":
    main()
