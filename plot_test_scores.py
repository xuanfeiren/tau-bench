import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('test2.csv')

# Extract the main test score columns for each algorithm
algorithms = {
    'ExploreOnlyLLM_v1.0': 'ExploreOnlyLLM_v1.0 - Test score',
    'ExplorewithLLM_v1.0': 'ExplorewithLLM_v1.0 - Test score', 
    'ExploreAlgorithm_v1.0': 'ExploreAlgorithm_v1.0 - Test score'
}

# Create the plot
plt.figure(figsize=(10, 6))

# Colors for each algorithm
colors = ['#9B59B6', '#2ECC71', '#E67E22']  # Purple, Green, Orange

# Plot each algorithm
for i, (alg_name, col_name) in enumerate(algorithms.items()):
    steps = df['Step'].values
    scores = df[col_name].astype(float).values
    
    plt.plot(steps, scores, marker='o', linewidth=2, markersize=6, 
             label=alg_name, color=colors[i])

# Customize the plot
plt.xlabel('Step', fontsize=12)
plt.ylabel('Test Score', fontsize=12)
plt.title('Algorithm Test Scores Over Steps', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Set axis limits and ticks
plt.xlim(-0.5, 10.5)
plt.xticks(range(11))

# Add some styling
plt.tight_layout()
plt.savefig('example.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Print summary statistics
print("\nSummary Statistics:")
print("="*50)
for alg_name, col_name in algorithms.items():
    scores = df[col_name].astype(float).values
    print(f"{alg_name}:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print() 