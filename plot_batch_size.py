import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_batch_size_data():
    """Plot ExploreAlgorithm with different batch sizes (1, 2, 3) showing mean and min/max ranges"""
    df = pd.read_csv('data_to_plot/batch_size.csv')
    
    plt.figure(figsize=(12, 8))
    
    # Define the batch sizes and their corresponding columns
    # ExploreAlgorithm-v0.2 = batch size 1 (default)
    # ExploreAlgorithm-v0.2-batch-size-2 = batch size 2
    # ExploreAlgorithm-v0.2-batch-size-3 = batch size 3
    batch_configs = {
        'Batch Size 1': {
            'mean': 'ExploreAlgorithm-v0.2 - Test score',
            'min': 'ExploreAlgorithm-v0.2 - Test score__MIN',
            'max': 'ExploreAlgorithm-v0.2 - Test score__MAX'
        },
        'Batch Size 2': {
            'mean': 'ExploreAlgorithm-v0.2-batch-size-2 - Test score',
            'min': 'ExploreAlgorithm-v0.2-batch-size-2 - Test score__MIN',
            'max': 'ExploreAlgorithm-v0.2-batch-size-2 - Test score__MAX'
        },
        'Batch Size 3': {
            'mean': 'ExploreAlgorithm-v0.2-batch-size-3 - Test score',
            'min': 'ExploreAlgorithm-v0.2-batch-size-3 - Test score__MIN',
            'max': 'ExploreAlgorithm-v0.2-batch-size-3 - Test score__MAX'
        }
    }
    
    # Colors for each batch size
    colors = ['#2E86C1', '#28B463', '#F39C12']  # Blue, Green, Orange
    markers = ['o', 's', '^']  # Circle, Square, Triangle
    
    # Get total samples (x-axis)
    total_samples = df['Total samples'].astype(int).values
    
    # Plot each batch size
    for i, (batch_name, cols) in enumerate(batch_configs.items()):
        # Get test scores, replacing empty strings with NaN
        mean_scores = df[cols['mean']].replace('', np.nan)
        min_scores = df[cols['min']].replace('', np.nan)
        max_scores = df[cols['max']].replace('', np.nan)
        
        # Convert to numeric
        mean_scores = pd.to_numeric(mean_scores, errors='coerce')
        min_scores = pd.to_numeric(min_scores, errors='coerce')
        max_scores = pd.to_numeric(max_scores, errors='coerce')
        
        # Filter out NaN values for plotting
        valid_mask = ~mean_scores.isna()
        valid_samples = total_samples[valid_mask]
        valid_mean = mean_scores[valid_mask]
        valid_min = min_scores[valid_mask]
        valid_max = max_scores[valid_mask]
        
        if len(valid_mean) > 0:  # Only plot if there's data
            # Plot mean line
            plt.plot(valid_samples, valid_mean, marker=markers[i], linewidth=2.5, 
                     markersize=8, label=batch_name, color=colors[i])
            
            # Fill between min and max
            plt.fill_between(valid_samples, valid_min, valid_max, alpha=0.2, color=colors[i])
    
    plt.xlabel('Total Samples', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('ExploreAlgorithm Performance with Different Batch Sizes\n(Mean ± Min/Max Range)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(-50, max(total_samples) + 100)
    
    plt.tight_layout()
    plt.savefig('batch_size_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def print_batch_size_statistics():
    """Print summary statistics for each batch size"""
    df = pd.read_csv('data_to_plot/batch_size.csv')
    
    batch_configs = {
        'Batch Size 1': 'ExploreAlgorithm-v0.2 - Test score',
        'Batch Size 2': 'ExploreAlgorithm-v0.2-batch-size-2 - Test score',
        'Batch Size 3': 'ExploreAlgorithm-v0.2-batch-size-3 - Test score'
    }
    
    print("\nBatch Size Comparison - Summary Statistics:")
    print("="*60)
    
    for batch_name, col_name in batch_configs.items():
        # Get test scores, replacing empty strings with NaN
        scores = df[col_name].replace('', np.nan)
        scores = pd.to_numeric(scores, errors='coerce')
        
        # Filter out NaN values
        valid_scores = scores.dropna().values
        
        if len(valid_scores) > 0:
            print(f"{batch_name}:")
            print(f"  Data points: {len(valid_scores)}")
            print(f"  Mean: {np.mean(valid_scores):.4f}")
            print(f"  Std:  {np.std(valid_scores):.4f}")
            print(f"  Min:  {np.min(valid_scores):.4f}")
            print(f"  Max:  {np.max(valid_scores):.4f}")
            print(f"  Final score: {valid_scores[-1]:.4f}")
            
            # Show sample range
            total_samples = df['Total samples'].astype(int).values
            valid_mask = ~scores.isna()
            valid_samples = total_samples[valid_mask]
            print(f"  Sample range: {min(valid_samples)} - {max(valid_samples)}")
            print()
        else:
            print(f"{batch_name}: No valid data found")
            print()

def analyze_batch_size_performance():
    """Analyze which batch size performs best"""
    df = pd.read_csv('data_to_plot/batch_size.csv')
    
    batch_configs = {
        1: 'ExploreAlgorithm-v0.2 - Test score',
        2: 'ExploreAlgorithm-v0.2-batch-size-2 - Test score',
        3: 'ExploreAlgorithm-v0.2-batch-size-3 - Test score'
    }
    
    print("Batch Size Performance Analysis:")
    print("="*40)
    
    final_scores = {}
    mean_scores = {}
    
    for batch_size, col_name in batch_configs.items():
        scores = df[col_name].replace('', np.nan)
        scores = pd.to_numeric(scores, errors='coerce')
        valid_scores = scores.dropna().values
        
        if len(valid_scores) > 0:
            final_scores[batch_size] = valid_scores[-1]
            mean_scores[batch_size] = np.mean(valid_scores)
    
    # Find best performing batch sizes
    if final_scores:
        best_final = max(final_scores, key=final_scores.get)
        best_mean = max(mean_scores, key=mean_scores.get)
        
        print(f"Best final performance: Batch Size {best_final} ({final_scores[best_final]:.4f})")
        print(f"Best average performance: Batch Size {best_mean} ({mean_scores[best_mean]:.4f})")
        print()
        
        print("Final scores comparison:")
        for batch_size in sorted(final_scores.keys()):
            print(f"  Batch Size {batch_size}: {final_scores[batch_size]:.4f}")
        
        print("\nAverage scores comparison:")
        for batch_size in sorted(mean_scores.keys()):
            print(f"  Batch Size {batch_size}: {mean_scores[batch_size]:.4f}")

if __name__ == "__main__":
    # Create the plot
    plot_batch_size_data()
    
    # Print summary statistics
    print_batch_size_statistics()
    
    # Analyze performance
    analyze_batch_size_performance()
    
    print("\nBatch size comparison plot has been saved as: batch_size_comparison.pdf") 