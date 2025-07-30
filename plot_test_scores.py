import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_minibatch():
    """Plot 3 lines (mean, min, max) for Minibatch data"""
    df = pd.read_csv('data_to_plot/Minibatch.csv')
    
    plt.figure(figsize=(10, 6))
    
    steps = df['Step'].values
    mean_scores = df['MinibatchwithValidation - Test score'].astype(float).values
    min_scores = df['MinibatchwithValidation - Test score__MIN'].astype(float).values
    max_scores = df['MinibatchwithValidation - Test score__MAX'].astype(float).values
    
    plt.plot(steps, mean_scores, marker='o', linewidth=2, markersize=6, 
             label='Mean', color='#2ECC71')
    plt.plot(steps, min_scores, marker='s', linewidth=2, markersize=6, 
             label='Min', color='#E74C3C')
    plt.plot(steps, max_scores, marker='^', linewidth=2, markersize=6, 
             label='Max', color='#3498DB')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('MinibatchwithValidation Test Scores Over Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('minibatch_scores.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_island_search():
    """Plot single line for Island Search data"""
    df = pd.read_csv('data_to_plot/Islandsearch.csv')
    
    plt.figure(figsize=(10, 6))
    
    steps = df['Step'].values
    scores = df['IslandSearchAlgorithm - Test score'].astype(float).values
    
    plt.plot(steps, scores, marker='o', linewidth=2, markersize=6, 
             label='IslandSearchAlgorithm', color='#9B59B6')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('Island Search Algorithm Test Scores Over Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('island_search_scores.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_basic_search():
    """Plot single line for Basic Search data"""
    df = pd.read_csv('data_to_plot/Basicsearch.csv')
    
    plt.figure(figsize=(10, 6))
    
    steps = df['Step'].values
    scores = df['BasicSearchAlgorithm - Test score'].astype(float).values
    
    plt.plot(steps, scores, marker='o', linewidth=2, markersize=6, 
             label='BasicSearchAlgorithm', color='#E67E22')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('Basic Search Algorithm Test Scores Over Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('basic_search_scores.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_explore():
    """Plot mean with min/max range for 3 algorithms in Explore data"""
    df = pd.read_csv('data_to_plot/Explore.csv')
    
    plt.figure(figsize=(10, 6))
    
    algorithms = {
        'ExploreOnlyLLM': {
            'mean': 'Name: ExploreOnlyLLM_v1.0 - Test score',
            'min': 'Name: ExploreOnlyLLM_v1.0 - Test score__MIN',
            'max': 'Name: ExploreOnlyLLM_v1.0 - Test score__MAX'
        },
        'ExplorewithLLM': {
            'mean': 'Name: ExplorewithLLM_v1.0 - Test score',
            'min': 'Name: ExplorewithLLM_v1.0 - Test score__MIN',
            'max': 'Name: ExplorewithLLM_v1.0 - Test score__MAX'
        },
        'ExploreAlgorithm': {
            'mean': 'Name: ExploreAlgorithm_v1.0 - Test score',
            'min': 'Name: ExploreAlgorithm_v1.0 - Test score__MIN',
            'max': 'Name: ExploreAlgorithm_v1.0 - Test score__MAX'
        }
    }
    
    colors = ['#9B59B6', '#2ECC71', '#E67E22']  # Purple, Green, Orange
    steps = df['Step'].values
    
    for i, (alg_name, cols) in enumerate(algorithms.items()):
        mean_scores = df[cols['mean']].astype(float).values
        min_scores = df[cols['min']].astype(float).values
        max_scores = df[cols['max']].astype(float).values
        
        # Plot mean line
        plt.plot(steps, mean_scores, marker='o', linewidth=2, markersize=6, 
                 label=alg_name, color=colors[i])
        
        # Fill between min and max
        plt.fill_between(steps, min_scores, max_scores, alpha=0.2, color=colors[i])
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('Explore Algorithms Test Scores Over Steps (Mean with Min/Max Range)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('explore_scores.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def print_summary_statistics():
    """Print summary statistics for all algorithms"""
    print("\nSummary Statistics:")
    print("="*70)
    
    # Minibatch statistics
    df_minibatch = pd.read_csv('data_to_plot/Minibatch.csv')
    scores = df_minibatch['MinibatchwithValidation - Test score'].astype(float).values
    print(f"MinibatchwithValidation:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print()
    
    # Island Search statistics
    df_island = pd.read_csv('data_to_plot/Islandsearch.csv')
    scores = df_island['IslandSearchAlgorithm - Test score'].astype(float).values
    print(f"IslandSearchAlgorithm:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print()
    
    # Basic Search statistics
    df_basic = pd.read_csv('data_to_plot/Basicsearch.csv')
    scores = df_basic['BasicSearchAlgorithm - Test score'].astype(float).values
    print(f"BasicSearchAlgorithm:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print()
    
    # Explore algorithms statistics
    df_explore = pd.read_csv('data_to_plot/Explore.csv')
    algorithms = {
        'ExploreOnlyLLM_v1.0': 'Name: ExploreOnlyLLM_v1.0 - Test score',
        'ExplorewithLLM_v1.0': 'Name: ExplorewithLLM_v1.0 - Test score',
        'ExploreAlgorithm_v1.0': 'Name: ExploreAlgorithm_v1.0 - Test score'
    }
    
    for alg_name, col_name in algorithms.items():
        scores = df_explore[col_name].astype(float).values
        print(f"{alg_name}:")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std:  {np.std(scores):.4f}")
        print(f"  Min:  {np.min(scores):.4f}")
        print(f"  Max:  {np.max(scores):.4f}")
        print()

if __name__ == "__main__":
    # Create all 4 plots
    plot_minibatch()
    plot_island_search()
    plot_basic_search()
    plot_explore()
    
    # Print summary statistics
    print_summary_statistics()
    
    print("All plots have been saved as PDF files:")
    print("- minibatch_scores.pdf")
    print("- island_search_scores.pdf")
    print("- basic_search_scores.pdf")
    print("- explore_scores.pdf") 