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

# Algorithm configurations
ALGORITHMS = {
    'BasicSearchAlgorithm': {
        'display_name': 'Best of n',
        'color': '#9B59B6',  # Purple
        'linestyle': '-'
    },
    'IslandSearchAlgorithm': {
        'display_name': 'IslandSearchAlgorithm', 
        'color': '#6C5CE7',  # Dark purple
        'linestyle': '-'
    },
    'MinibatchwithValidation': {
        'display_name': 'MinibatchwithValidation',
        'color': '#74B9FF',  # Light blue (dashed in original)
        'linestyle': '-'
    },
    'ExploreOnlyLLM_v1.0': {
        'display_name': 'ExploreOnlyLLM',
        'color': '#00B894',  # Teal
        'linestyle': '-'
    },
    'ExplorewithLLM_v1.0': {
        'display_name': 'ExplorewithLLM',
        'color': '#E17055',  # Orange/brown
        'linestyle': '-'
    },
    'ExploreAlgorithm_v1.0': {
        'display_name': 'ExploreAlgorithm',
        'color': '#00CED1',  # Cyan
        'linestyle': '-'
    }
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
            
        # Extract relevant data
        for idx, row in history.iterrows():
            if pd.notna(row.get('Test score')):
                all_data.append({
                    'run_id': run.id,
                    'run_name': run_name,
                    'algorithm': algorithm,
                    'step': row.get('_step', idx),
                    'total_samples': row.get('Total samples', row.get('_step', idx) * 4),  # Estimate if not available
                    'test_score': row['Test score']
                })
    
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

def subsample_data_points(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, 
                         max_points: int = 15, strategy: str = 'random') -> tuple:
    """
    Randomly subsample data points to reduce clutter and make the plot cleaner.
    
    Args:
        x: Original x values
        y: Original y values  
        yerr: Original standard errors
        max_points: Maximum number of points to keep
        strategy: 'random' for random sampling, 'uniform' for evenly spaced
        
    Returns:
        Tuple of (x_sampled, y_sampled, yerr_sampled)
    """
    if len(x) <= max_points:
        # Don't need to subsample
        return x, y, yerr
    
    if strategy == 'random':
        # Random sampling
        indices = np.random.choice(len(x), size=max_points, replace=False)
        indices = np.sort(indices)  # Keep chronological order
    else:
        # Uniform sampling (evenly spaced)
        indices = np.linspace(0, len(x)-1, max_points, dtype=int)
    
    return x[indices], y[indices], yerr[indices]

def create_publication_plot(stats_df: pd.DataFrame, output_file: str = "test_scores_plot.pdf", 
                           max_points: int = 15, sampling_strategy: str = 'random'):
    """
    Create a publication-quality plot without standard error shading.
    """
    plt.style.use('default')  # Clean style
    fig, ax = plt.subplots(figsize=(12, 8))
    
    np.random.seed(42)  # For reproducible results
    
    # Set fixed y-axis range
    y_min_plot = 0.3
    y_max_plot = 0.55
    
    for algorithm in ALGORITHMS.keys():
        alg_data = stats_df[stats_df['algorithm'] == algorithm].sort_values('total_samples')
        
        if alg_data.empty:
            continue
            
        config = ALGORITHMS[algorithm]
        
        x = alg_data['total_samples'].values
        y = alg_data['mean_score'].values
        yerr = alg_data['std_error'].values
        n_runs = alg_data['n_runs'].values
        
        # Subsample data points for cleaner look
        x_sampled, y_sampled, yerr_sampled = subsample_data_points(x, y, yerr, max_points, sampling_strategy)
        
        # Plot main line with subsampled points
        ax.plot(x_sampled, y_sampled, 
               color=config['color'], 
               linestyle=config['linestyle'],
               linewidth=2.5, 
               marker='o', 
               markersize=6,
               label=config['display_name'],
               zorder=3)
        
        # No standard error shading since we don't calculate standard errors
    
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
    parser.add_argument('--output', default='test_scores_plot.pdf',
                       help='Output plot filename')
    parser.add_argument('--max-points', type=int, default=10,
                       help='Maximum number of points per algorithm (for subsampling)')
    parser.add_argument('--sampling', choices=['random', 'uniform'], default='uniform',
                       help='Sampling strategy: random or uniform')
    
    args = parser.parse_args()
    
    stats_df = None
    
    # Export data from wandb (this is now mandatory)
    print("=== Exporting data from wandb ===")
    try:
        raw_df = export_wandb_data(args.project)
        if not raw_df.empty:
            stats_df = calculate_statistics(raw_df)
            stats_df.to_csv('wandb_statistics.csv', index=False)
            print("Statistics saved to: wandb_statistics.csv")
        else:
            print("No data exported from wandb")
            return
    except Exception as e:
        print(f"Error exporting from wandb: {e}")
        return
    
    # Create plot
    print("\n=== Creating plot ===")
    
    if not stats_df.empty:
        create_publication_plot(stats_df, args.output, args.max_points, args.sampling)
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        for algorithm in ALGORITHMS.keys():
            alg_data = stats_df[stats_df['algorithm'] == algorithm]
            if not alg_data.empty:
                original_points = len(alg_data)
                sampled_points = min(original_points, args.max_points)
                max_runs = alg_data['n_runs'].max()
                points_with_multiple_runs = len(alg_data[alg_data['n_runs'] > 1])
                
                print(f"\n{algorithm}:")
                print(f"  Original data points: {original_points}")
                print(f"  Sampled data points: {sampled_points}")
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