import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df = pd.read_csv('data.csv')

# Extract the data
steps = df['Step']
batch_size_3_score = df['ExploreAlgorithm-v0.2-batch-size-3 - Test score']
batch_size_3_min = df['ExploreAlgorithm-v0.2-batch-size-3 - Test score__MIN']
batch_size_3_max = df['ExploreAlgorithm-v0.2-batch-size-3 - Test score__MAX']

batch_size_2_score = df['ExploreAlgorithm-v0.2-batch-size-2 - Test score']
batch_size_2_min = df['ExploreAlgorithm-v0.2-batch-size-2 - Test score__MIN']
batch_size_2_max = df['ExploreAlgorithm-v0.2-batch-size-2 - Test score__MAX']

# Create the plot
plt.figure(figsize=(10, 6))

# Plot solid lines for test scores
plt.plot(steps, batch_size_3_score, 'b-', linewidth=2, label='Batch Size 3', alpha=0.8)
plt.plot(steps, batch_size_2_score, 'r-', linewidth=2, label='Batch Size 2', alpha=0.8)

# Add shadow areas using fill_between for min/max ranges
plt.fill_between(steps, batch_size_3_min, batch_size_3_max, alpha=0.3, color='blue')
plt.fill_between(steps, batch_size_2_min, batch_size_2_max, alpha=0.3, color='red')

# Customize the plot
plt.xlabel('Step', fontsize=12)
plt.ylabel('Test Score', fontsize=12)
plt.title('ExploreAlgorithm v0.2 Performance Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Set x-axis to show all steps
plt.xticks(steps)

# Adjust layout and save
plt.tight_layout()
plt.savefig('figure_for_data.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as 'figure_for_data.pdf'") 