#!/usr/bin/env python3
"""
Script to plot correlation detection data from correlation_detection.csv
Visualizes the relationship between scores before and after optimization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def load_data(filename='correlation_detection.csv'):
    """Load the correlation detection data."""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} data points from {filename}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_statistics(df):
    """Calculate and display basic statistics and correlation."""
    print("\n=== Data Statistics ===")
    print(f"Before optimization - Mean: {df['score_before_opto'].mean():.3f}, Std: {df['score_before_opto'].std():.3f}")
    print(f"After optimization - Mean: {df['score_after_opto'].mean():.3f}, Std: {df['score_after_opto'].std():.3f}")
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(df['score_before_opto'], df['score_after_opto'])
    print(f"Pearson correlation: {correlation:.3f} (p-value: {p_value:.3f})")
    
    # Calculate improvement statistics
    improvement = df['score_after_opto'] - df['score_before_opto']
    print(f"Average improvement: {improvement.mean():.3f}")
    print(f"Improvement std: {improvement.std():.3f}")
    improved_count = (improvement > 0).sum()
    print(f"Cases with improvement: {improved_count}/{len(df)} ({100*improved_count/len(df):.1f}%)")
    
    return correlation, p_value, improvement

def create_plots(df, correlation, p_value, improvement):
    """Create comprehensive plots of the data."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Scatter plot with correlation
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(df['score_before_opto'], df['score_after_opto'], 
                alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal reference line (y = x)
    min_score = min(df['score_before_opto'].min(), df['score_after_opto'].min())
    max_score = max(df['score_before_opto'].max(), df['score_after_opto'].max())
    plt.plot([min_score, max_score], [min_score, max_score], 
             'r--', alpha=0.8, linewidth=2, label='No change line (y=x)')
    
    # Add best fit line
    z = np.polyfit(df['score_before_opto'], df['score_after_opto'], 1)
    p = np.poly1d(z)
    plt.plot(df['score_before_opto'], p(df['score_before_opto']), 
             'g-', alpha=0.8, linewidth=2, label=f'Best fit line')
    
    plt.xlabel('Score Before Optimization')
    plt.ylabel('Score After Optimization')
    plt.title(f'Correlation: {correlation:.3f} (p={p_value:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Improvement histogram
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(improvement, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    plt.axvline(improvement.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {improvement.mean():.3f}')
    plt.xlabel('Score Improvement (After - Before)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Score Improvements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Before vs After box plots
    ax3 = plt.subplot(2, 3, 3)
    data_to_plot = [df['score_before_opto'], df['score_after_opto']]
    box_plot = plt.boxplot(data_to_plot, labels=['Before', 'After'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightgreen')
    plt.ylabel('Score')
    plt.title('Score Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Individual improvements (line plot)
    ax4 = plt.subplot(2, 3, 4)
    x_pos = range(len(df))
    for i in x_pos:
        color = 'green' if improvement.iloc[i] > 0 else 'red'
        alpha = 0.6
        plt.plot([i, i], [df['score_before_opto'].iloc[i], df['score_after_opto'].iloc[i]], 
                color=color, alpha=alpha, linewidth=1)
    
    plt.scatter(x_pos, df['score_before_opto'], color='blue', alpha=0.7, 
                label='Before', s=40)
    plt.scatter(x_pos, df['score_after_opto'], color='orange', alpha=0.7, 
                label='After', s=40)
    plt.xlabel('Data Point Index')
    plt.ylabel('Score')
    plt.title('Individual Score Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative improvement
    ax5 = plt.subplot(2, 3, 5)
    sorted_improvement = np.sort(improvement)
    cumulative_pct = np.arange(1, len(sorted_improvement) + 1) / len(sorted_improvement) * 100
    plt.plot(sorted_improvement, cumulative_pct, linewidth=2, color='purple')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    plt.xlabel('Score Improvement')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Cumulative Distribution of Improvements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Correlation heatmap-style visualization
    ax6 = plt.subplot(2, 3, 6)
    # Create a 2D histogram
    plt.hist2d(df['score_before_opto'], df['score_after_opto'], bins=10, cmap='Blues')
    plt.colorbar(label='Count')
    plt.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, linewidth=2)
    plt.xlabel('Score Before Optimization')
    plt.ylabel('Score After Optimization')
    plt.title('2D Histogram of Score Pairs')
    
    plt.tight_layout()
    return fig

def save_plots(fig):
    """Save the plots to files."""
    # Save as PNG
    fig.savefig('correlation_detection_plots.png', dpi=300, bbox_inches='tight')
    print("Saved plots as 'correlation_detection_plots.png'")
    
    # Save as PDF
    fig.savefig('correlation_detection_plots.pdf', bbox_inches='tight')
    print("Saved plots as 'correlation_detection_plots.pdf'")

def main():
    """Main function to run the analysis and create plots."""
    print("Correlation Detection Data Analysis")
    print("=" * 40)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate statistics
    correlation, p_value, improvement = calculate_statistics(df)
    
    # Create plots
    fig = create_plots(df, correlation, p_value, improvement)
    
    # Save plots
    save_plots(fig)
    
    # Show plots
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 