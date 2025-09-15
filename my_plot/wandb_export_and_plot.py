#!/usr/bin/env python3
"""
Wandb Data Export and Plotting Script

This script:
1. Exports individual run data from wandb API
2. Counts the number of runs per algorithm
3. Creates publication-quality plots showing mean scores
4. Calculates proper statistics from multiple runs when available

Usage:
    python wandb_export_and_plot.py --project your-project-name
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Will use basic interpolation.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Only plotting from existing data will work.")

# Algorithm configurations with distinct colors and line styles
ALGORITHMS = {
    # Baseline algorithms - solid lines with cooler colors
    'BasicSearchAlgorithm': {
        'display_name': 'Best of n',
        'color': '#9B59B6',  # Purple
        'linestyle': '-',
        'linewidth': 2.5
    },
    # 'IslandSearchAlgorithm': {
    #     'display_name': 'IslandSearchAlgorithm', 
    #     'color': '#6C5CE7',  # Light purple
    #     'linestyle': '--',
    #     'linewidth': 2.5
    # },
    'MinibatchwithValidation': {
        'display_name': 'MinibatchwithValidation',
        'color': '#74B9FF',  # Light blue
        'linestyle': '-',
        'linewidth': 2.5
    },
    
    # # Exploration algorithms - warmer colors with varied line styles
    # 'ExploreOnlyLLM_v1.0': {
    #     'display_name': 'ExploreOnlyLLM',
    #     'color': '#00B894',  # Teal green
    #     'linestyle': '-',
    #     'linewidth': 2.5
    # },
    # 'ExplorewithLLM_v1.0': {
    #     'display_name': 'ExplorewithLLM',
    #     'color': '#E17055',  # Coral
    #     'linestyle': '--',
    #     'linewidth': 2.5
    # },
    # 'ExploreAlgorithm_v1.0': {
    #     'display_name': 'ExploreAlgorithm v1.0',
    #     'color': '#00CED1',  # Dark turquoise
    #     'linestyle': '-',
    #     'linewidth': 2.5
    # },
    # 'ExploreAlgorithm_v1.2': {
    #     'display_name': 'ExploreAlgorithm v1.2',
    #     'color': '#FF69B4',  # Hot pink
    #     'linestyle': ':',
    #     'linewidth': 2.5
    # },
    
    # LLM-based search algorithms - distinctive colors and styles
    'llm-search-new-version': {
        'display_name': 'LLM Search',
        'color': '#FF6B35',  # Bright orange-red
        'linestyle': '-',
        'linewidth': 3.0  # Slightly thicker for emphasis
    },
    # 'llm-search-no-regressor': {
    #     'display_name': 'LLM Search (no regressor)',
    #     'color': '#2E86AB',  # Deep blue
    #     'linestyle': '--',
    #     'linewidth': 3.0
    # },
    'llm-search-multigen': {
        'display_name': 'LLM Search (Multi-Gen)',
        'color': '#FF69B4',  # Hot pink
        'linestyle': '-',
        'linewidth': 3.0
    },
}
   
def export_wandb_data(project_name: str = "tau-bench-retail-compare-search-algs", 
                     output_file: str = "wandb_individual_runs.csv") -> pd.DataFrame:
    """
    Export individual run data from wandb instead of aggregated data.
    
    Args:
        project_name: Wandb project name
        output_file: Output CSV file path
        
    Returns:
        DataFrame with individual run data
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb package is required for data export")
    
    print(f"Connecting to wandb project: {project_name}")
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(f"xuanfeiren-university-of-wisconsin-madison/{project_name}")  # Using the correct entity
    
    all_data = []
    run_counts = {}
    
    for run in runs:
        # Filter runs by algorithm name - strict exact matching only
        run_name = run.name
        algorithm = None
        
        for alg_key in ALGORITHMS.keys():
            # Only match if run name exactly equals algorithm name
            if run_name == alg_key:
                algorithm = alg_key
                break
        
        if algorithm is None:
            continue
            
        print(f"Processing run: {run_name} (Algorithm: {algorithm})")
        
        # Count runs per algorithm
        if algorithm not in run_counts:
            run_counts[algorithm] = set()
        run_counts[algorithm].add(run.id)
        
        # Get run history (metrics over time)
        history = run.history()
        
        if 'Test score' not in history.columns:
            print(f"  Warning: No 'Test score' found in run {run_name}")
            continue
            
        # Extract relevant data and calculate highest_score_so_far for this run
        run_data_points = []
        for idx, row in history.iterrows():
            if pd.notna(row.get('Test score')):
                run_data_points.append({
                    'run_id': run.id,
                    'run_name': run_name,
                    'algorithm': algorithm,
                    'step': row.get('_step', idx),
                    'total_samples': row.get('Total samples', row.get('_step', idx) * 4),  # Estimate if not available
                    'test_score': row['Test score']
                })
        
        # Sort by total_samples to ensure correct chronological order for cumulative max
        run_data_points.sort(key=lambda x: x['total_samples'])
        
        # Calculate highest_score_so_far for this run
        highest_so_far = float('-inf')
        for data_point in run_data_points:
            current_score = data_point['test_score']
            highest_so_far = max(highest_so_far, current_score)
            data_point['highest_score_so_far'] = highest_so_far
            all_data.append(data_point)
    
    # Print run counts
    print("\n=== Run Counts by Algorithm ===")
    for algorithm in ALGORITHMS.keys():
        count = len(run_counts.get(algorithm, set()))
        print(f"{algorithm}: {count} runs")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("Warning: No data found. Check project name and algorithm names.")
        return df
    
    # Save raw data
    df.to_csv(output_file, index=False)
    print(f"Raw individual run data saved to: {output_file}")
    
    return df

def calculate_statistics_by_intervals(df: pd.DataFrame, interval_size: int = 100, use_cummax: bool = False) -> pd.DataFrame:
    """
    Calculate mean, standard error, and confidence intervals for each algorithm by binning data into intervals.
    
    Args:
        df: DataFrame with individual run data
        interval_size: Size of each interval (default 100)
        use_cummax: If True, calculate cumulative maximum scores instead of mean scores
        
    Returns:
        DataFrame with calculated statistics for each interval
    """
    stats_data = []
    
    # Define intervals from 0 to max total_samples
    max_samples = df['total_samples'].max()
    intervals = list(range(0, int(max_samples) + interval_size, interval_size))
    
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm].sort_values(['run_id', 'total_samples'])
        
        # Note: highest_score_so_far is already calculated in export_wandb_data
        # No need to recalculate cumulative maximum here
        
        for i in range(len(intervals) - 1):
            interval_start = intervals[i]
            interval_end = intervals[i + 1]
            interval_center = (interval_start + interval_end) / 2
            
            # Get all data points within this interval for this algorithm
            interval_data = alg_data[
                (alg_data['total_samples'] >= interval_start) & 
                (alg_data['total_samples'] < interval_end)
            ]
            
            if len(interval_data) == 0:
                continue
            
            if use_cummax:
                # For cumulative maximum, we need to take the maximum highest_score_so_far 
                # for each run within this interval, then average across runs
                run_max_scores = []
                for run_id in interval_data['run_id'].unique():
                    run_interval_data = interval_data[interval_data['run_id'] == run_id]
                    # Take the maximum highest_score_so_far for this run in this interval
                    max_score_for_run = run_interval_data['highest_score_so_far'].max()
                    run_max_scores.append(max_score_for_run)
                scores = np.array(run_max_scores)
            else:
                scores = interval_data['test_score'].values
            n_points = len(scores)
            
            if n_points > 1:
                mean_score = np.mean(scores)
                std_error = np.std(scores, ddof=1) / np.sqrt(n_points)  # Standard error
                
                # 95% confidence interval (assuming t-distribution for small samples)
                try:
                    from scipy import stats as scipy_stats
                    t_value = scipy_stats.t.ppf(0.975, df=n_points-1)  # 97.5th percentile for 95% CI
                    ci_lower = mean_score - t_value * std_error
                    ci_upper = mean_score + t_value * std_error
                except ImportError:
                    # Fallback to normal approximation if scipy not available
                    ci_lower = mean_score - 1.96 * std_error
                    ci_upper = mean_score + 1.96 * std_error
            else:
                mean_score = scores[0]
                std_error = 0
                ci_lower = mean_score
                ci_upper = mean_score
            
            stats_data.append({
                'algorithm': algorithm,
                'interval_start': interval_start,
                'interval_end': interval_end,
                'interval_center': interval_center,
                'mean_score': mean_score,
                'std_error': std_error,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_points': n_points,
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            })
    
    return pd.DataFrame(stats_data)

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean, standard error, and confidence intervals for each algorithm and total_samples.
    
    Args:
        df: DataFrame with individual run data
        
    Returns:
        DataFrame with calculated statistics
    """
    # Group by algorithm and total_samples
    grouped = df.groupby(['algorithm', 'total_samples'])['test_score']
    
    stats_data = []
    
    for (algorithm, total_samples), group in grouped:
        scores = group.values
        n_runs = len(scores)
        
        if n_runs > 1:
            mean_score = np.mean(scores)
            std_error = np.std(scores, ddof=1) / np.sqrt(n_runs)  # Standard error
            
            # 95% confidence interval (assuming t-distribution for small samples)
            from scipy import stats as scipy_stats
            t_value = scipy_stats.t.ppf(0.975, df=n_runs-1)  # 97.5th percentile for 95% CI
            ci_lower = mean_score - t_value * std_error
            ci_upper = mean_score + t_value * std_error
        else:
            mean_score = scores[0]
            std_error = 0
            ci_lower = mean_score
            ci_upper = mean_score
        
        stats_data.append({
            'algorithm': algorithm,
            'total_samples': total_samples,
            'mean_score': mean_score,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_runs': n_runs,
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        })
    
    return pd.DataFrame(stats_data)

def create_publication_plot_with_intervals(stats_df: pd.DataFrame, output_file: str = "test_scores_with_stderr.pdf"):
    """
    Create a publication-quality plot with shaded areas showing standard error.
    """
    plt.style.use('default')  # Clean style
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set fixed y-axis range
    y_min_plot = 0.35
    y_max_plot = 0.55
    
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('interval_center')
        
        if alg_data.empty:
            continue
            
        config = ALGORITHMS[algorithm]
        
        x = alg_data['interval_center'].values
        y = alg_data['mean_score'].values
        yerr = alg_data['std_error'].values
        n_points = alg_data['n_points'].values
        
        # Calculate upper and lower bounds for shaded area
        y_upper = y + yerr
        y_lower = y - yerr
        
        # Plot main line
        ax.plot(x, y,
               color=config['color'], 
               linestyle=config['linestyle'],
               linewidth=config.get('linewidth', 2.5), 
               marker='o', 
               markersize=6,
               label=config['display_name'],
               zorder=3)
        
        # Add shaded area for standard error (for all points)
        # For single points, std_error will be 0, so shadow will be just the line
        ax.fill_between(x, y_lower, y_upper,
                       color=config['color'],
                       alpha=0.1,  # Semi-transparent
                       zorder=1)
    
    # Customize plot
    ax.set_xlabel('Total samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test score', fontsize=14, fontweight='bold')
    ax.set_title('Test score with Standard Error', fontsize=16, fontweight='bold')
    
    # Set axis limits based on data
    ax.set_xlim(0, 2050)
    ax.set_ylim(y_min_plot, y_max_plot)
    
    print(f"Y-axis range set to: {y_min_plot:.1f} to {y_max_plot:.1f}")
    
    # Customize ticks
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(['0', '500', '1k', '1.5k', '2k'])
    
    # Set nice y-axis ticks
    y_tick_step = max(0.05, (y_max_plot - y_min_plot) / 6)  # Aim for ~6 ticks
    y_tick_step = np.ceil(y_tick_step * 20) / 20  # Round to nearest 0.05
    y_ticks = np.arange(np.ceil(y_min_plot / y_tick_step) * y_tick_step, 
                       y_max_plot + y_tick_step/2, 
                       y_tick_step)
    ax.set_yticks(y_ticks)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as: {pdf_file} and {png_file}")
    plt.show()

def create_max_score_plot_with_intervals(stats_df: pd.DataFrame, output_file: str = "max_scores_with_stderr.pdf"):
    """
    Create a publication-quality plot showing cumulative maximum scores with shaded areas showing standard error.
    """
    plt.style.use('default')  # Clean style
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set fixed y-axis range
    y_min_plot = 0.35
    y_max_plot = 0.55  # Increased to accommodate 0.5 scores
    
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('interval_center')
        
        if alg_data.empty:
            continue
            
        config = ALGORITHMS[algorithm]
        
        x = alg_data['interval_center'].values
        y = alg_data['mean_score'].values
        yerr = alg_data['std_error'].values
        n_points = alg_data['n_points'].values
        
        # Calculate upper and lower bounds for shaded area
        y_upper = y + yerr
        y_lower = y - yerr
        
        # Plot main line
        ax.plot(x, y,
               color=config['color'], 
               linestyle=config['linestyle'],
               linewidth=config.get('linewidth', 2.5), 
               marker='o', 
               markersize=6,
               label=config['display_name'],
               zorder=3)
        
        # Add shaded area for standard error (for all points)
        # For single points, std_error will be 0, so shadow will be just the line
        ax.fill_between(x, y_lower, y_upper,
                       color=config['color'],
                       alpha=0.1,  # Semi-transparent
                       zorder=1)
    
    # Customize plot
    ax.set_xlabel('Total samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Highest score so far', fontsize=14, fontweight='bold')
    ax.set_title('Highest score so far with Standard Error', fontsize=16, fontweight='bold')
    
    # Set axis limits based on data
    ax.set_xlim(0, 2050)
    ax.set_ylim(y_min_plot, y_max_plot)
    
    print(f"Y-axis range set to: {y_min_plot:.1f} to {y_max_plot:.1f}")
    
    # Customize ticks
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(['0', '500', '1k', '1.5k', '2k'])
    
    # Set nice y-axis ticks
    y_tick_step = max(0.05, (y_max_plot - y_min_plot) / 6)  # Aim for ~6 ticks
    y_tick_step = np.ceil(y_tick_step * 20) / 20  # Round to nearest 0.05
    y_ticks = np.arange(np.ceil(y_min_plot / y_tick_step) * y_tick_step, 
                       y_max_plot + y_tick_step/2, 
                       y_tick_step)
    ax.set_yticks(y_ticks)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Max score plot saved as: {pdf_file} and {png_file}")
    plt.show()

def create_publication_plot(stats_df: pd.DataFrame, output_file: str = "test_scores_plot.pdf"):
    """
    Create a publication-quality plot without standard error shading (legacy function).
    """
    plt.style.use('default')  # Clean style
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set fixed y-axis range
    y_min_plot = 0.3
    y_max_plot = 0.6
    
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('total_samples')
        
        if alg_data.empty:
            continue
            
        config = ALGORITHMS[algorithm]
        
        x = alg_data['total_samples'].values
        y = alg_data['mean_score'].values
        
        # Plot main line
        ax.plot(x, y, 
               color=config['color'], 
               linestyle=config['linestyle'],
               linewidth=config.get('linewidth', 2.5), 
               marker='o', 
               markersize=6,
               label=config['display_name'],
               zorder=3)
    
    # Customize plot
    ax.set_xlabel('Total samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test score', fontsize=14, fontweight='bold')
    ax.set_title('Test score', fontsize=16, fontweight='bold')
    
    # Set axis limits based on data
    ax.set_xlim(0, 2050)
    ax.set_ylim(y_min_plot, y_max_plot)
    
    print(f"Y-axis range set to: {y_min_plot:.1f} to {y_max_plot:.1f}")
    
    # Customize ticks
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(['0', '500', '1k', '1.5k', '2k'])
    
    # Set nice y-axis ticks
    y_tick_step = max(0.05, (y_max_plot - y_min_plot) / 6)  # Aim for ~6 ticks
    y_tick_step = np.ceil(y_tick_step * 20) / 20  # Round to nearest 0.05
    y_ticks = np.arange(np.ceil(y_min_plot / y_tick_step) * y_tick_step, 
                       y_max_plot + y_tick_step/2, 
                       y_tick_step)
    ax.set_yticks(y_ticks)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PDF and PNG
    pdf_file = output_file
    png_file = output_file.replace('.pdf', '.png')
    
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as: {pdf_file} and {png_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Export wandb data and create plots showing run counts and mean scores')
    parser.add_argument('--project', default='tau-bench-retail-compare-search-algs', 
                       help='Wandb project name')
    parser.add_argument('--output', default='test_scores_with_stderr.pdf',
                       help='Output plot filename')
    parser.add_argument('--interval-size', type=int, default=100,
                       help='Size of intervals for binning data (default: 100)')
    parser.add_argument('--use-intervals', action='store_true', default=True,
                       help='Use interval-based statistics (default: True)')
    
    args = parser.parse_args()
    
    stats_df = None
    
    # Export data from wandb (this is now mandatory)
    print("=== Exporting data from wandb ===")
    try:
        raw_df = export_wandb_data(args.project)
        if not raw_df.empty:
            if args.use_intervals:
                print(f"Using interval-based statistics with interval size: {args.interval_size}")
                stats_df = calculate_statistics_by_intervals(raw_df, args.interval_size)
                stats_df.to_csv('wandb_statistics.csv', index=False)
                print("Interval-based statistics saved to: wandb_statistics.csv")
            else:
                print("Using point-based statistics")
                stats_df = calculate_statistics(raw_df)
                stats_df.to_csv('wandb_statistics.csv', index=False)
                print("Point-based statistics saved to: wandb_statistics.csv")
        else:
            print("No data exported from wandb")
            return
    except Exception as e:
        print(f"Error exporting from wandb: {e}")
        return
    
    # Create plots
    print("\n=== Creating plots ===")
    
    if not stats_df.empty:
        if args.use_intervals:
            # Create regular test score plot
            create_publication_plot_with_intervals(stats_df, args.output)
            
            # Create cumulative maximum score plot
            print("\n=== Creating cumulative maximum score plot ===")
            max_stats_df = calculate_statistics_by_intervals(raw_df, args.interval_size, use_cummax=True)
            max_output_file = args.output.replace('.pdf', '_max.pdf')
            create_max_score_plot_with_intervals(max_stats_df, max_output_file)
            
            # Save max statistics
            max_stats_df.to_csv('wandb_max_statistics.csv', index=False)
            print("Max score statistics saved to: wandb_max_statistics.csv")
        else:
            create_publication_plot(stats_df, args.output)
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        for algorithm in ALGORITHMS.keys():
            alg_data = stats_df[stats_df['algorithm'] == algorithm]
            if not alg_data.empty:
                if args.use_intervals:
                    total_intervals = len(alg_data)
                    total_points = alg_data['n_points'].sum()
                    max_points_per_interval = alg_data['n_points'].max()
                    intervals_with_multiple_points = len(alg_data[alg_data['n_points'] > 1])
                    
                    print(f"\n{algorithm}:")
                    print(f"  Total intervals: {total_intervals}")
                    print(f"  Total data points: {total_points}")
                    print(f"  Max points per interval: {max_points_per_interval}")
                    print(f"  Intervals with multiple points: {intervals_with_multiple_points}")
                    print(f"  Score range: {alg_data['mean_score'].min():.3f} - {alg_data['mean_score'].max():.3f}")
                    if intervals_with_multiple_points > 0:
                        avg_stderr = alg_data[alg_data['n_points'] > 1]['std_error'].mean()
                        print(f"  Average standard error: {avg_stderr:.4f}")
                        print(f"  Has variation in data: Yes")
                    else:
                        print(f"  Has variation in data: No (single points per interval)")
                else:
                    original_points = len(alg_data)
                    max_runs = alg_data['n_runs'].max()
                    points_with_multiple_runs = len(alg_data[alg_data['n_runs'] > 1])
                    
                    print(f"\n{algorithm}:")
                    print(f"  Data points: {original_points}")
                    print(f"  Max runs per data point: {max_runs}")
                    print(f"  Data points with multiple runs: {points_with_multiple_runs}")
                    print(f"  Score range: {alg_data['mean_score'].min():.3f} - {alg_data['mean_score'].max():.3f}")
                    if points_with_multiple_runs > 0:
                        print(f"  Has variation in data: Yes")
                    else:
                        print(f"  Has variation in data: No (single runs only)")
    else:
        print("No data available for plotting")

if __name__ == "__main__":
    main() 