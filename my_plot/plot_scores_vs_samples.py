#!/usr/bin/env python3
"""
Plot Test Scores vs Total Samples

This script:
1. Fetches data from wandb project "tau-bench-10-tasks-10-evals"
2. For each run, collects (test_score, total_samples) pairs at each step
3. Calculates cumulative best score at each sample count
4. Forward-fills: for each unique sample count across all runs, fills missing data
   with the last known best score
5. Plots test score (y) vs total samples (x) with mean and min/max shaded area
"""

import json
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
    'PS-mean_exploration-Nov7': {
        'display_name': 'Vanilla PS',
        'color': '#E74C3C',  # Red
        'linestyle': '-',
        'linewidth': 2.5
    },
    # 'pareto-PS': {
    #     'display_name': 'Pareto-Exploration PS',
    #     'color': '#3498DB',  # Blue
    #     'linestyle': '-',
    #     'linewidth': 2.5
    # },
    'epsnet_0.1-PS_with_summarizer': {
        'display_name': 'EpsNet 0.1 PS with Summarizer',
        'color': '#9B59B6',  # Purple
        'linestyle': '-',
        'linewidth': 2.5
    },
    'epsnet_0.1-PS': {
        'display_name': 'EpsNet 0.1 PS',
        'color': '#FF6B35',  # Orange
        'linestyle': '-',
        'linewidth': 2.5
    },
    'DSPy_GEPA_most_frequent': {
        'display_name': 'DSPy_GEPA most_frequent',
        'color': '#2ECC71',  # Green
        'linestyle': '-',
        'linewidth': 2.5,
        'local_file': 'dspy_results/gepa_Nov25/eval_results/eval_results_most_frequent.json'
    },
    'DSPy_GEPA_sample_by_freq': {
        'display_name': 'DSPy_GEPA sample_by_freq',
        'color': '#3498DB',  # Blue
        'linestyle': '-',
        'linewidth': 2.5,
        'local_file': 'dspy_results/gepa_Nov25/eval_results/eval_results_sample_by_freq.json'
    },
    # 'epsnet_0.1-PS_with_Detailed_summarizer-Nov21': {
    #     'display_name': 'EpsNet 0.1 PS Detailed Summarizer',
    #     'color': '#F39C12',  # Amber/Gold
    #     'linestyle': '-',
    #     'linewidth': 2.5
    # },
    # 'epsnet_0-PS_with_summarizer': {
    #     'display_name': 'PS + Summarizer',
    #     'color': '#E91E63',  # Pink
    #     'linestyle': '-',
    #     'linewidth': 2.5
    # }
}

PROJECT_NAME = "tau-bench-10-tasks-10-evals"
SCORE_METRIC = "Test/test_score_empirical_mean"
SAMPLES_METRIC = "Update/total_samples"


def load_local_gepa_data():
    """
    Load GEPA evaluation data from local JSON files.
    
    Reads algorithms that have 'local_file' key in their config.
    
    Returns:
        DataFrame with columns: algorithm, run_id, step, score, num_samples
    """
    print("\nLoading local GEPA data...")
    
    all_data = []
    
    for algorithm, config in ALGORITHMS.items():
        if 'local_file' not in config:
            continue
        
        filepath = config['local_file']
        print(f"  Loading: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"    Warning: File not found: {filepath}")
            continue
        except json.JSONDecodeError as e:
            print(f"    Warning: Invalid JSON in {filepath}: {e}")
            continue
        
        # Extract data points
        for entry in data:
            all_data.append({
                'algorithm': algorithm,
                'run_id': f'{algorithm}_run0',  # Single run per file
                'step': entry.get('iteration', 0),
                'score': float(entry['score']),
                'num_samples': int(entry['total_samples'])
            })
        
        print(f"    Loaded {len(data)} data points")
    
    if not all_data:
        print("  No local GEPA data found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    print(f"Total local GEPA data points: {len(df)}")
    
    return df


def fetch_wandb_data():
    """
    Fetch data from wandb for the specified algorithms.
    
    Returns:
        DataFrame with columns: algorithm, run_id, step, score, num_samples
    """
    print(f"Connecting to wandb project: {PROJECT_NAME}")
    api = wandb.Api()
    
    # Get all runs from the project
    try:
        runs = api.runs(f"xuanfeiren-university-of-wisconsin-madison/{PROJECT_NAME}")
    except Exception as e:
        print(f"Error connecting to wandb: {e}")
        print("Make sure you're logged in with: wandb login")
        exit(1)
    
    all_data = []
    # Only count wandb algorithms (those without local_file)
    wandb_algorithms = {alg: config for alg, config in ALGORITHMS.items() if 'local_file' not in config}
    run_counts = {alg: 0 for alg in wandb_algorithms.keys()}
    
    print("\nFetching runs...")
    for run in runs:
        run_name = run.name
        
        # Check if this run matches any of our target wandb algorithms
        if run_name not in wandb_algorithms:
            continue
        
        algorithm = run_name
        run_counts[algorithm] += 1
        
        print(f"  Processing: {run_name} (Run ID: {run.id})")
        
        # Get run history
        try:
            history = run.history()
        except Exception as e:
            print(f"    Warning: Could not fetch history for {run_name}: {e}")
            continue
        
        # Check if the metrics exist
        if SCORE_METRIC not in history.columns:
            print(f"    Warning: Metric '{SCORE_METRIC}' not found in run {run_name}")
            continue
        
        if SAMPLES_METRIC not in history.columns:
            print(f"    Warning: Metric '{SAMPLES_METRIC}' not found in run {run_name}")
            continue
        
        # Extract data points
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
    
    # Print summary
    print("\n=== Run Counts by Algorithm ===")
    for algorithm, count in run_counts.items():
        print(f"{algorithm}: {count} runs")
    
    if not all_data:
        print("\nNo wandb data found for the specified algorithms.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal wandb data points fetched: {len(df)}")
    
    # Diagnostic: Show sample ranges per algorithm
    print("\n=== Sample Ranges Per Algorithm (wandb) ===")
    for algorithm in wandb_algorithms.keys():
        alg_data = df[df['algorithm'] == algorithm]
        if not alg_data.empty:
            min_samples = alg_data['num_samples'].min()
            max_samples = alg_data['num_samples'].max()
            print(f"{algorithm}: {min_samples} - {max_samples} samples")
    
    return df


def calculate_best_score_so_far(df):
    """
    Calculate cumulative maximum (best score so far) for each run.
    Uses num_samples as the progression axis.
    
    Args:
        df: DataFrame with columns: algorithm, run_id, step, score, num_samples
        
    Returns:
        DataFrame with added column: best_score_so_far
    """
    print("\nCalculating best score so far for each run...")
    
    results = []
    
    for (algorithm, run_id), group in df.groupby(['algorithm', 'run_id']):
        # Sort by num_samples to ensure correct order (not step!)
        group = group.sort_values('num_samples').copy()
        
        # Calculate cumulative maximum based on sample order
        group['best_score_so_far'] = group['score'].cummax()
        
        results.append(group)
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"Calculated cumulative max for {len(result_df)} data points")
    
    return result_df


def forward_fill_by_samples(df):
    """
    Forward-fill best scores based on sample counts.
    For each algorithm, only fills up to that algorithm's maximum sample count.
    Different algorithms will end at different x-coordinates.
    
    Args:
        df: DataFrame with columns: algorithm, run_id, num_samples, best_score_so_far
        
    Returns:
        DataFrame with forward-filled data at all sample points per algorithm
    """
    print("\nForward-filling by sample counts...")
    
    filled_data = []
    
    # Process each algorithm separately
    for algorithm in df['algorithm'].unique():
        alg_df = df[df['algorithm'] == algorithm]
        
        # Get all unique sample counts for THIS algorithm only
        algorithm_sample_counts = sorted(alg_df['num_samples'].unique())
        max_samples_for_alg = max(algorithm_sample_counts)
        
        print(f"\n{algorithm}:")
        print(f"  Unique sample counts: {len(algorithm_sample_counts)}")
        print(f"  Sample range: {min(algorithm_sample_counts)} to {max_samples_for_alg}")
        
        # Process each run within this algorithm
        for run_id in alg_df['run_id'].unique():
            run_data_df = alg_df[alg_df['run_id'] == run_id].sort_values('num_samples')
            
            existing_samples = set(run_data_df['num_samples'].values)
            min_samples = run_data_df['num_samples'].min()
            max_samples = run_data_df['num_samples'].max()
            
            # Create data for all sample counts in THIS algorithm
            run_data = []
            last_known_score = None
            
            for sample_count in algorithm_sample_counts:
                if sample_count in existing_samples:
                    # Use actual data
                    row = run_data_df[run_data_df['num_samples'] == sample_count].iloc[0]
                    last_known_score = row['best_score_so_far']
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'num_samples': sample_count,
                        'best_score_so_far': last_known_score
                    })
                elif sample_count >= min_samples and sample_count <= max_samples and last_known_score is not None:
                    # Forward-fill with last known score
                    # Only fill between this run's start and end
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'num_samples': sample_count,
                        'best_score_so_far': last_known_score
                    })
                elif sample_count > max_samples and last_known_score is not None:
                    # For sample counts beyond this run's end, use the final score
                    run_data.append({
                        'algorithm': algorithm,
                        'run_id': run_id,
                        'num_samples': sample_count,
                        'best_score_so_far': last_known_score
                    })
            
            if run_data:
                filled_data.append(pd.DataFrame(run_data))
    
    result_df = pd.concat(filled_data, ignore_index=True)
    print(f"\nTotal data points after forward-fill: {len(result_df)}")
    
    # Show per-algorithm statistics
    print("\n=== Per-Algorithm Sample Counts After Forward-Fill ===")
    for algorithm in result_df['algorithm'].unique():
        alg_data = result_df[result_df['algorithm'] == algorithm]
        unique_samples = len(alg_data['num_samples'].unique())
        max_samples = alg_data['num_samples'].max()
        print(f"{algorithm}: {unique_samples} unique sample points, max={int(max_samples)}")
    
    return result_df


def aggregate_across_runs(df):
    """
    Calculate mean, min, max across multiple runs for each algorithm and sample count.
    
    Args:
        df: DataFrame with columns: algorithm, run_id, num_samples, best_score_so_far
        
    Returns:
        DataFrame with aggregated statistics
    """
    print("\nAggregating statistics across runs...")
    
    aggregated = df.groupby(['algorithm', 'num_samples'])['best_score_so_far'].agg([
        ('mean_score', 'mean'),
        ('min_score', 'min'),
        ('max_score', 'max'),
        ('n_runs', 'count'),
        ('std_score', 'std')
    ]).reset_index()
    
    print(f"Aggregated into {len(aggregated)} data points")
    
    # Show sample counts per algorithm
    print("\n=== Sample Counts Per Algorithm ===")
    for algorithm in ALGORITHMS.keys():
        alg_data = aggregated[aggregated['algorithm'] == algorithm]
        if not alg_data.empty:
            print(f"{algorithm}: {len(alg_data)} unique sample counts")
    
    return aggregated


def create_plot(stats_df, output_file='scores_vs_samples.pdf'):
    """
    Create publication-quality plot showing best scores vs total samples.
    
    Args:
        stats_df: DataFrame with aggregated statistics
        output_file: Output filename (PDF)
    """
    print("\nCreating plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each algorithm
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('num_samples')
        
        if alg_data.empty:
            print(f"  Warning: No data for {algorithm}")
            continue
        
        config = ALGORITHMS[algorithm]
        
        x = alg_data['num_samples'].values
        y_mean = alg_data['mean_score'].values
        y_min = alg_data['min_score'].values
        y_max = alg_data['max_score'].values
        
        # Downsample for plotting if too many points
        if len(x) > 200:
            # Keep every Nth point to reduce clutter
            step = len(x) // 200
            indices = list(range(0, len(x), step))
            if (len(x) - 1) not in indices:
                indices.append(len(x) - 1)  # Always include last point
            
            x_plot = x[indices]
            y_mean_plot = y_mean[indices]
            y_min_plot = y_min[indices]
            y_max_plot = y_max[indices]
        else:
            x_plot = x
            y_mean_plot = y_mean
            y_min_plot = y_min
            y_max_plot = y_max
        
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
        
        # Add shaded area for min/max range (use all points for smooth fill)
        ax.fill_between(x, y_min, y_max,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)
        
        print(f"  Plotted {algorithm}: {len(x)} sample points (displayed {len(x_plot)})")
    
    # Customize plot
    ax.set_xlabel('Total Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Test Score So Far', fontsize=14, fontweight='bold')
    ax.set_title('Test Score vs Total Samples', fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
        
        # Get final scores (at max samples)
        final_data = alg_data[alg_data['num_samples'] == alg_data['num_samples'].max()]
        if not final_data.empty:
            final_samples = final_data['num_samples'].iloc[0]
            final_mean = final_data['mean_score'].iloc[0]
            final_std = final_data['std_score'].iloc[0] if pd.notna(final_data['std_score'].iloc[0]) else 0
            final_min = final_data['min_score'].iloc[0]
            final_max = final_data['max_score'].iloc[0]
        else:
            final_samples = final_mean = final_std = final_min = final_max = 0
        
        print(f"\n{ALGORITHMS[algorithm]['display_name']} ({algorithm}):")
        print(f"  Sample points: {len(alg_data)}")
        print(f"  Max samples: {int(final_samples)}")
        print(f"  Final best score (mean): {final_mean:.4f}")
        print(f"  Final best score (std):  {final_std:.4f}")
        print(f"  Final best score (min):  {final_min:.4f}")
        print(f"  Final best score (max):  {final_max:.4f}")
        print(f"  Score range: {alg_data['mean_score'].min():.4f} - {alg_data['mean_score'].max():.4f}")


def main():
    """Main execution function."""
    print("="*70)
    print("TEST SCORE VS TOTAL SAMPLES PLOTTING SCRIPT")
    print("="*70)
    
    # Step 1: Fetch data from wandb
    wandb_df = fetch_wandb_data()
    
    # Step 1b: Load local GEPA data
    local_df = load_local_gepa_data()
    
    # Merge data sources
    dfs_to_concat = [df for df in [wandb_df, local_df] if not df.empty]
    if not dfs_to_concat:
        print("\nError: No data found from any source.")
        exit(1)
    raw_df = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"\nTotal combined data points: {len(raw_df)}")
    
    # Step 2: Calculate best score so far (cumulative max) for each run
    df_with_cummax = calculate_best_score_so_far(raw_df)
    
    # Step 3: Forward-fill by sample counts
    df_filled = forward_fill_by_samples(df_with_cummax)
    
    # Step 4: Aggregate across runs
    stats_df = aggregate_across_runs(df_filled)
    
    # Step 5: Create the plot
    create_plot(stats_df, output_file='scores_vs_samples.pdf')
    
    # Step 6: Print summary statistics
    print_summary_statistics(stats_df)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("Plot saved:")
    print("  - scores_vs_samples.pdf")
    print("  - scores_vs_samples.png")
    print("="*70)


if __name__ == "__main__":
    main()

