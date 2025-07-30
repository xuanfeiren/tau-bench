import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_feedback_data():
    """Plot 4 lines for the 4 algorithms in feedback.csv"""
    df = pd.read_csv('data_to_plot/feedback.csv')
    
    plt.figure(figsize=(12, 8))
    
    # Define the 4 algorithms and their corresponding test score columns
    algorithms = {
        'IslandSearchAlgorithm': 'IslandSearchAlgorithm - Test score',
        'ExploreOnlyLLM_v1.0': 'ExploreOnlyLLM_v1.0 - Test score',
        'ExplorewithLLM_v1.0': 'ExplorewithLLM_v1.0 - Test score',
        'ExploreAlgorithm_v1.0': 'ExploreAlgorithm_v1.0 - Test score'
    }
    
    # Colors for each algorithm
    colors = ['#9B59B6', '#2ECC71', '#E67E22', '#3498DB']  # Purple, Green, Orange, Blue
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    # Get total samples (x-axis)
    total_samples = df['Total samples'].astype(int).values
    
    # Plot each algorithm
    for i, (alg_name, col_name) in enumerate(algorithms.items()):
        # Get test scores, replacing empty strings with NaN
        scores = df[col_name].replace('', np.nan)
        scores = pd.to_numeric(scores, errors='coerce')
        
        # Filter out NaN values for plotting
        valid_mask = ~scores.isna()
        valid_samples = total_samples[valid_mask]
        valid_scores = scores[valid_mask]
        
        if len(valid_scores) > 0:  # Only plot if there's data
            plt.plot(valid_samples, valid_scores, marker=markers[i], linewidth=2, 
                     markersize=8, label=alg_name, color=colors[i])
    
    plt.xlabel('Total Samples', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('Algorithm Test Scores vs Total Samples (Feedback Data)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(-50, max(total_samples) + 50)
    
    plt.tight_layout()
    plt.savefig('figures/feedback_scores.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def print_feedback_statistics():
    """Print summary statistics for all algorithms in feedback data"""
    df = pd.read_csv('data_to_plot/feedback.csv')
    
    algorithms = {
        'IslandSearchAlgorithm': 'IslandSearchAlgorithm - Test score',
        'ExploreOnlyLLM_v1.0': 'ExploreOnlyLLM_v1.0 - Test score',
        'ExplorewithLLM_v1.0': 'ExplorewithLLM_v1.0 - Test score',
        'ExploreAlgorithm_v1.0': 'ExploreAlgorithm_v1.0 - Test score'
    }
    
    print("\nFeedback Data Summary Statistics:")
    print("="*70)
    
    for alg_name, col_name in algorithms.items():
        # Get test scores, replacing empty strings with NaN
        scores = df[col_name].replace('', np.nan)
        scores = pd.to_numeric(scores, errors='coerce')
        
        # Filter out NaN values
        valid_scores = scores.dropna().values
        
        if len(valid_scores) > 0:
            print(f"{alg_name}:")
            print(f"  Data points: {len(valid_scores)}")
            print(f"  Mean: {np.mean(valid_scores):.4f}")
            print(f"  Std:  {np.std(valid_scores):.4f}")
            print(f"  Min:  {np.min(valid_scores):.4f}")
            print(f"  Max:  {np.max(valid_scores):.4f}")
            
            # Show sample range
            total_samples = df['Total samples'].astype(int).values
            valid_mask = ~scores.isna()
            valid_samples = total_samples[valid_mask]
            print(f"  Sample range: {min(valid_samples)} - {max(valid_samples)}")
            print()
        else:
            print(f"{alg_name}: No valid data found")
            print()

def show_data_distribution():
    """Show how many data points each algorithm has"""
    df = pd.read_csv('data_to_plot/feedback.csv')
    
    algorithms = {
        'IslandSearchAlgorithm': 'IslandSearchAlgorithm - Test score',
        'ExploreOnlyLLM_v1.0': 'ExploreOnlyLLM_v1.0 - Test score',
        'ExplorewithLLM_v1.0': 'ExplorewithLLM_v1.0 - Test score',
        'ExploreAlgorithm_v1.0': 'ExploreAlgorithm_v1.0 - Test score'
    }
    
    print("Data Distribution by Algorithm:")
    print("="*50)
    
    for alg_name, col_name in algorithms.items():
        scores = df[col_name].replace('', np.nan)
        scores = pd.to_numeric(scores, errors='coerce')
        valid_count = scores.notna().sum()
        total_count = len(scores)
        
        print(f"{alg_name}: {valid_count}/{total_count} data points ({valid_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    # Show data distribution first
    show_data_distribution()
    
    # Create the plot
    plot_feedback_data()
    
    # Print summary statistics
    print_feedback_statistics()
    
    print("Feedback plot has been saved as: feedback_scores.pdf") 