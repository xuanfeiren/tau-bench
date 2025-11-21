#!/usr/bin/env python3
"""
Plot Best Scores So Far for Multiple Algorithms

This script:
1. Fetches data from wandb project "tau-bench-10-tasks-10-evals"
2. For 4 algorithms: PS-mean_exploration-Nov7, pareto-PS, 
   epsnet_0.01-PS_with_summarizer, epsnet_0.1-PS_with_summarizer
3. Calculates cumulative maximum (best score so far) for each run
4. Handles missing values using forward-fill approach
5. Plots mean with min/max shaded area across multiple runs
"""

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
    'pareto-PS': {
        'display_name': 'Pareto-Exploration PS',
        'color': '#3498DB',  # Blue
        'linestyle': '-',
        'linewidth': 2.5
    },
    'epsnet_0.01-PS_with_summarizer': {
        'display_name': 'EpsNet 0.01 PS with Summarizer',
        'color': '#2ECC71',  # Green
        'linestyle': '-',
        'linewidth': 2.5
    },
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
    'epsnet_0.1-PS-DetailedSummarizer': {
        'display_name': 'EpsNet 0.1 PS Detailed Summarizer',
        'color': '#F39C12',  # Amber/Gold
        'linestyle': '-',
        'linewidth': 2.5
    },
    'epsnet_0-PS_with_summarizer': {
        'display_name': 'EpsNet 0 PS with Summarizer',
        'color': '#E91E63',  # Pink
        'linestyle': '-',
        'linewidth': 2.5
    }
}

PROJECT_NAME = "tau-bench-10-tasks-10-evals"
METRIC_NAME = "Test/test_score_empirical_mean"
SAMPLES_METRIC = "Update/total_samples"
MEMORY_METRIC = "Update/long_term_memory_size"


def fetch_wandb_data():
    """
    Fetch data from wandb for the specified algorithms.
    
    Returns:
        DataFrame with columns: algorithm, run_id, step, score, num_samples, memory_size
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
    run_counts = {alg: 0 for alg in ALGORITHMS.keys()}
    
    print("\nFetching runs...")
    for run in runs:
        run_name = run.name
        
        # Check if this run matches any of our target algorithms
        if run_name not in ALGORITHMS:
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
        
        # Check if the metric exists
        if METRIC_NAME not in history.columns:
            print(f"    Warning: Metric '{METRIC_NAME}' not found in run {run_name}")
            print(f"    Available columns: {list(history.columns)}")
            continue
        
        # Extract data points
        for idx, row in history.iterrows():
            score = row.get(METRIC_NAME)
            step = row.get('_step', idx)
            num_samples = row.get(SAMPLES_METRIC)
            memory_size = row.get(MEMORY_METRIC)
            
            if pd.notna(score):
                all_data.append({
                    'algorithm': algorithm,
                    'run_id': run.id,
                    'step': int(step),
                    'score': float(score),
                    'num_samples': float(num_samples) if pd.notna(num_samples) else None,
                    'memory_size': float(memory_size) if pd.notna(memory_size) else None
                })
    
    # Print summary
    print("\n=== Run Counts by Algorithm ===")
    for algorithm, count in run_counts.items():
        print(f"{algorithm}: {count} runs")
    
    if not all_data:
        print("\nError: No data found for the specified algorithms and metric.")
        exit(1)
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal data points fetched: {len(df)}")
    
    # DIAGNOSTIC: Show data points per run
    print("\n=== Data Points Per Run (Diagnostic) ===")
    for algorithm in ALGORITHMS.keys():
        alg_data = df[df['algorithm'] == algorithm]
        if not alg_data.empty:
            print(f"\n{algorithm}:")
            for run_id in alg_data['run_id'].unique():
                run_data = alg_data[alg_data['run_id'] == run_id]
                steps = sorted(run_data['step'].unique())
                print(f"  Run {run_id[:8]}...: {len(steps)} data points, steps: {steps[:20]}{'...' if len(steps) > 20 else ''}")
    
    return df


def calculate_best_score_so_far(df):
    """
    Calculate cumulative maximum (best score so far) for each run.
    
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
        
        # Calculate cumulative maximum
        group['best_score_so_far'] = group['score'].cummax()
        
        results.append(group)
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"Calculated cumulative max for {len(result_df)} data points")
    
    return result_df


def forward_fill_missing_steps(df):
    """
    Forward-fill best scores and metrics for runs that end earlier than others.
    Only fills at steps where at least one run has actual data.
    
    Args:
        df: DataFrame with columns: algorithm, run_id, step, best_score_so_far, num_samples, memory_size
        
    Returns:
        DataFrame with forward-filled data only at existing step values
    """
    print("\nForward-filling missing steps...")
    
    # Get all unique steps that actually exist in the data
    all_actual_steps = sorted(df['step'].unique())
    print(f"Actual steps in data: {all_actual_steps}")
    
    filled_data = []
    
    for (algorithm, run_id), group in df.groupby(['algorithm', 'run_id']):
        group = group.sort_values('step')
        
        existing_steps = set(group['step'].values)
        last_step = group['step'].max()
        last_row = group[group['step'] == last_step].iloc[0]
        last_best_score = last_row['best_score_so_far']
        last_num_samples = last_row.get('num_samples')
        last_memory_size = last_row.get('memory_size')
        
        # Create data for all actual steps
        run_data = []
        needs_forward_fill = last_step < all_actual_steps[-1]
        if needs_forward_fill:
            print(f"  {algorithm} (Run {run_id[:8]}...): forward-filling from step {last_step} to {all_actual_steps[-1]}")
        
        for step in all_actual_steps:
            if step in existing_steps:
                # Use actual data
                row = group[group['step'] == step].iloc[0]
                run_data.append({
                    'algorithm': algorithm,
                    'run_id': run_id,
                    'step': step,
                    'best_score_so_far': row['best_score_so_far'],
                    'num_samples': row.get('num_samples'),
                    'memory_size': row.get('memory_size')
                })
            elif step > last_step:
                # Forward-fill with last known values for steps after run ended
                run_data.append({
                    'algorithm': algorithm,
                    'run_id': run_id,
                    'step': step,
                    'best_score_so_far': last_best_score,
                    'num_samples': last_num_samples,
                    'memory_size': last_memory_size
                })
            # Don't add data for steps before this run started
        
        if run_data:
            filled_data.append(pd.DataFrame(run_data))
    
    result_df = pd.concat(filled_data, ignore_index=True)
    print(f"Total data points after forward-fill: {len(result_df)}")
    print(f"Unique steps after forward-fill: {sorted(result_df['step'].unique())}")
    
    return result_df


def aggregate_across_runs(df):
    """
    Calculate mean, min, max across multiple runs for each algorithm and step.
    
    Args:
        df: DataFrame with columns: algorithm, run_id, step, best_score_so_far, num_samples, memory_size
        
    Returns:
        DataFrame with aggregated statistics for scores, samples, and memory
    """
    print("\nAggregating statistics across runs...")
    
    # Aggregate best scores
    score_agg = df.groupby(['algorithm', 'step'])['best_score_so_far'].agg([
        ('mean_score', 'mean'),
        ('min_score', 'min'),
        ('max_score', 'max'),
        ('n_runs', 'count'),
        ('std_score', 'std')
    ]).reset_index()
    
    # Aggregate num_samples
    samples_agg = df.groupby(['algorithm', 'step'])['num_samples'].agg([
        ('mean_samples', 'mean'),
        ('min_samples', 'min'),
        ('max_samples', 'max')
    ]).reset_index()
    
    # Aggregate memory_size
    memory_agg = df.groupby(['algorithm', 'step'])['memory_size'].agg([
        ('mean_memory', 'mean'),
        ('min_memory', 'min'),
        ('max_memory', 'max')
    ]).reset_index()
    
    # Merge all aggregations
    aggregated = score_agg.merge(samples_agg, on=['algorithm', 'step'], how='left')
    aggregated = aggregated.merge(memory_agg, on=['algorithm', 'step'], how='left')
    
    print(f"Aggregated into {len(aggregated)} data points")
    
    return aggregated


def create_plot(stats_df, output_file='best_scores.pdf'):
    """
    Create publication-quality plot showing best scores over time.
    
    Args:
        stats_df: DataFrame with aggregated statistics
        output_file: Output filename (PDF)
    """
    print("\nCreating plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each algorithm
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('step')
        
        if alg_data.empty:
            print(f"  Warning: No data for {algorithm}")
            continue
        
        config = ALGORITHMS[algorithm]
        
        x = alg_data['step'].values
        y_mean = alg_data['mean_score'].values
        y_min = alg_data['min_score'].values
        y_max = alg_data['max_score'].values
        
        # Plot mean line
        ax.plot(x, y_mean,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                marker='o',
                markersize=5,
                markevery=max(1, len(x) // 20),  # Show markers every ~5% of points
                label=config['display_name'],
                zorder=3)
        
        # Add shaded area for min/max range
        ax.fill_between(x, y_min, y_max,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)
        
        print(f"  Plotted {algorithm}: {len(x)} steps")
    
    # Customize plot
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Score So Far', fontsize=14, fontweight='bold')
    ax.set_title('Cumulative Best Scores Across Algorithms', fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
    
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


def create_samples_plot(stats_df, output_file='num_samples.pdf'):
    """
    Create plot showing number of samples over time.
    
    Args:
        stats_df: DataFrame with aggregated statistics
        output_file: Output filename (PDF)
    """
    print("\nCreating num_samples plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each algorithm
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('step')
        
        if alg_data.empty or alg_data['mean_samples'].isna().all():
            print(f"  Warning: No samples data for {algorithm}")
            continue
        
        config = ALGORITHMS[algorithm]
        
        x = alg_data['step'].values
        y_mean = alg_data['mean_samples'].values
        y_min = alg_data['min_samples'].values
        y_max = alg_data['max_samples'].values
        
        # Plot mean line
        ax.plot(x, y_mean,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                marker='o',
                markersize=5,
                markevery=max(1, len(x) // 20),
                label=config['display_name'],
                zorder=3)
        
        # Add shaded area for min/max range
        ax.fill_between(x, y_min, y_max,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)
        
        print(f"  Plotted {algorithm}: {len(x)} steps")
    
    # Customize plot
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Total Samples Over Steps', fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Samples plot saved as:")
    print(f"  - {pdf_file}")
    print(f"  - {png_file}")
    
    plt.close()


def create_memory_plot(stats_df, output_file='total_proposals.pdf'):
    """
    Create plot showing total proposals (memory size) over time.
    
    Args:
        stats_df: DataFrame with aggregated statistics
        output_file: Output filename (PDF)
    """
    print("\nCreating total_proposals plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each algorithm
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('step')
        
        if alg_data.empty or alg_data['mean_memory'].isna().all():
            print(f"  Warning: No memory data for {algorithm}")
            continue
        
        config = ALGORITHMS[algorithm]
        
        x = alg_data['step'].values
        y_mean = alg_data['mean_memory'].values
        y_min = alg_data['min_memory'].values
        y_max = alg_data['max_memory'].values
        
        # Plot mean line
        ax.plot(x, y_mean,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                marker='o',
                markersize=5,
                markevery=max(1, len(x) // 20),
                label=config['display_name'],
                zorder=3)
        
        # Add shaded area for min/max range
        ax.fill_between(x, y_min, y_max,
                        color=config['color'],
                        alpha=0.2,
                        zorder=1)
        
        print(f"  Plotted {algorithm}: {len(x)} steps")
    
    # Customize plot
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Proposals (Memory Size)', fontsize=14, fontweight='bold')
    ax.set_title('Total Proposals Over Steps', fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Memory plot saved as:")
    print(f"  - {pdf_file}")
    print(f"  - {png_file}")
    
    plt.close()


def print_summary_statistics(df, stats_df):
    """
    Print summary statistics for the data.
    
    Args:
        df: Raw DataFrame with individual runs
        stats_df: Aggregated statistics DataFrame
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm]
        
        if alg_data.empty:
            continue
        
        # Get run information
        alg_runs = df[df['algorithm'] == algorithm]
        n_runs = alg_runs['run_id'].nunique()
        
        # Get final scores
        final_step_data = alg_data[alg_data['step'] == alg_data['step'].max()]
        if not final_step_data.empty:
            final_mean = final_step_data['mean_score'].iloc[0]
            final_std = final_step_data.get('std_score', pd.Series([0])).iloc[0]
            final_min = final_step_data['min_score'].iloc[0]
            final_max = final_step_data['max_score'].iloc[0]
        else:
            final_mean = final_std = final_min = final_max = 0
        
        print(f"\n{ALGORITHMS[algorithm]['display_name']} ({algorithm}):")
        print(f"  Number of runs: {n_runs}")
        print(f"  Final best score (mean): {final_mean:.4f}")
        print(f"  Final best score (std):  {final_std:.4f}")
        print(f"  Final best score (min):  {final_min:.4f}")
        print(f"  Final best score (max):  {final_max:.4f}")
        print(f"  Score range: {alg_data['mean_score'].min():.4f} - {alg_data['mean_score'].max():.4f}")


def main():
    """Main execution function."""
    print("="*70)
    print("BEST SCORES SO FAR PLOTTING SCRIPT")
    print("="*70)
    
    # Step 1: Fetch data from wandb
    raw_df = fetch_wandb_data()
    
    # Step 2: Calculate best score so far (cumulative max) for each run
    df_with_cummax = calculate_best_score_so_far(raw_df)
    
    # Step 3: Forward-fill missing steps
    df_filled = forward_fill_missing_steps(df_with_cummax)
    
    # Step 4: Aggregate across runs
    stats_df = aggregate_across_runs(df_filled)
    
    # Step 5: Create all plots
    create_plot(stats_df, output_file='best_scores.pdf')
    create_samples_plot(stats_df, output_file='num_samples.pdf')
    create_memory_plot(stats_df, output_file='total_proposals.pdf')
    
    # Step 6: Print summary statistics
    print_summary_statistics(df_filled, stats_df)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("All plots saved:")
    print("  - best_scores.pdf / best_scores.png")
    print("  - num_samples.pdf / num_samples.png")
    print("  - total_proposals.pdf / total_proposals.png")
    print("="*70)


if __name__ == "__main__":
    main()