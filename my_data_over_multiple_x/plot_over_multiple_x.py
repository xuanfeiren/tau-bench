import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Define the x-axis variables to plot
x_variables = ['eval_step', 'prop_step', 'num_samples', 'num_proposals']
y_variable = 'score'

# Find all algorithm folders (directories that contain CSV files)
algorithm_folders = []
for item in script_dir.iterdir():
    if item.is_dir() and item.name != '__pycache__':
        # Check if folder contains CSV files
        csv_files = list(item.glob('*.csv'))
        if csv_files:
            algorithm_folders.append(item)

print(f"Found {len(algorithm_folders)} algorithm folders: {[f.name for f in algorithm_folders]}")

# Process data for each x-axis variable
for x_var in x_variables:
    print(f"\nProcessing {x_var}...")
    
    # Use larger figure size for better visibility
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Process each algorithm
    for folder in algorithm_folders:
        algorithm_name = folder.name
        csv_files = sorted(folder.glob('*.csv'))
        
        if not csv_files:
            continue
        
        print(f"  Processing {algorithm_name} with {len(csv_files)} runs...")
        
        # Read all CSV files for this algorithm
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if x_var in df.columns and y_variable in df.columns:
                    # Extract x and y values
                    data = df[[x_var, y_variable]].copy()
                    data = data.dropna()  # Remove any NaN values
                    all_data.append(data)
            except Exception as e:
                print(f"    Warning: Could not read {csv_file}: {e}")
                continue
        
        if not all_data:
            print(f"    Warning: No valid data found for {algorithm_name}")
            continue
        
        # If only one run, just plot it directly
        if len(all_data) == 1:
            data = all_data[0]
            ax.plot(data[x_var], data[y_variable], label=algorithm_name, linewidth=3, marker='o', markersize=4, markevery=max(1, len(data) // 20))
        else:
            # Multiple runs: need to interpolate to common x-values
            # Get the union of all x-values
            all_x_values = set()
            for data in all_data:
                all_x_values.update(data[x_var].values)
            
            # Sort x-values
            x_common = np.array(sorted(all_x_values))
            
            # Interpolate each run to common x-values
            interpolated_scores = []
            for data in all_data:
                # Sort by x_var
                data_sorted = data.sort_values(by=x_var)
                x_data = data_sorted[x_var].values
                y_data = data_sorted[y_variable].values
                
                # Interpolate to common x-values
                y_interp = np.interp(x_common, x_data, y_data)
                interpolated_scores.append(y_interp)
            
            # Convert to numpy array for easier computation
            interpolated_scores = np.array(interpolated_scores)
            
            # Compute mean and standard error
            mean_scores = np.mean(interpolated_scores, axis=0)
            std_scores = np.std(interpolated_scores, axis=0, ddof=1)  # Sample standard deviation
            se_scores = std_scores / np.sqrt(len(interpolated_scores))  # Standard error
            
            # Plot mean line with better styling
            ax.plot(x_common, mean_scores, label=algorithm_name, linewidth=3, marker='o', markersize=4, markevery=max(1, len(x_common) // 20))
            
            # Plot error shading (standard error)
            ax.fill_between(x_common, 
                           mean_scores - se_scores, 
                           mean_scores + se_scores,
                           alpha=0.2)
    
    # Set labels and title with large, bold fonts
    ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=28, fontweight='bold')
    ax.set_ylabel('Score', fontsize=28, fontweight='bold')
    ax.set_title(f'Score vs {x_var.replace("_", " ").title()}', fontsize=30, fontweight='bold')
    
    # Legend with better styling
    ax.legend(fontsize=16, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Add border (keep all spines visible with thicker lines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    # Save figure
    output_file = script_dir / f"{x_var}.pdf"
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved {output_file}")
    plt.close()

print("\nDone! Generated 4 PDF files:")
for x_var in x_variables:
    print(f"  - {x_var}.pdf")
