#!/usr/bin/env python3
"""
Embedding Distance Analysis Script

This script analyzes the L2 distances between embedding vectors stored in embeddings_array.npy.
It calculates pairwise distances and provides comprehensive statistical summaries.

Author: Generated for tau-bench project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import pandas as pd
import os
from pathlib import Path

def load_embeddings(file_path):
    """Load embeddings from numpy file."""
    try:
        embeddings = np.load(file_path)
        print(f"✓ Loaded embeddings from {file_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Data type: {embeddings.dtype}")
        print(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        return embeddings
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        return None

def calculate_l2_distances(embeddings):
    """Calculate L2 distances between all pairs of embeddings."""
    print("\n📏 Calculating L2 distances...")
    
    # Calculate pairwise L2 distances
    # pdist computes the condensed distance matrix (upper triangle only)
    distances = pdist(embeddings, metric='euclidean')
    
    # Convert to full square matrix for easier analysis
    distance_matrix = squareform(distances)
    
    print(f"  ✓ Calculated {len(distances)} pairwise distances")
    print(f"  ✓ Distance matrix shape: {distance_matrix.shape}")
    
    return distances, distance_matrix

def compute_statistics(distances):
    """Compute comprehensive statistics of the distances."""
    print("\n📊 Computing distance statistics...")
    
    stats_dict = {
        'count': len(distances),
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'median': np.median(distances),
        'q25': np.percentile(distances, 25),
        'q75': np.percentile(distances, 75),
        'q90': np.percentile(distances, 90),
        'q95': np.percentile(distances, 95),
        'q99': np.percentile(distances, 99),
        'iqr': np.percentile(distances, 75) - np.percentile(distances, 25),
        'skewness': stats.skew(distances),
        'kurtosis': stats.kurtosis(distances)
    }
    
    return stats_dict

def print_statistics(stats_dict):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("📈 L2 DISTANCE STATISTICS SUMMARY")
    print("="*60)
    
    print(f"Total number of pairwise distances: {stats_dict['count']:,}")
    print(f"Mean distance:                      {stats_dict['mean']:.4f}")
    print(f"Standard deviation:                 {stats_dict['std']:.4f}")
    print(f"Minimum distance:                   {stats_dict['min']:.4f}")
    print(f"Maximum distance:                   {stats_dict['max']:.4f}")
    print(f"Median distance:                    {stats_dict['median']:.4f}")
    
    print("\n📊 Percentiles:")
    print(f"  25th percentile (Q1):             {stats_dict['q25']:.4f}")
    print(f"  75th percentile (Q3):             {stats_dict['q75']:.4f}")
    print(f"  90th percentile:                  {stats_dict['q90']:.4f}")
    print(f"  95th percentile:                  {stats_dict['q95']:.4f}")
    print(f"  99th percentile:                  {stats_dict['q99']:.4f}")
    
    print(f"\n📏 Spread measures:")
    print(f"  Interquartile Range (IQR):        {stats_dict['iqr']:.4f}")
    print(f"  Coefficient of Variation:         {stats_dict['std']/stats_dict['mean']:.4f}")
    
    print(f"\n📐 Distribution shape:")
    print(f"  Skewness:                         {stats_dict['skewness']:.4f}")
    print(f"  Kurtosis:                         {stats_dict['kurtosis']:.4f}")
    
    # Interpretation
    print(f"\n🔍 Interpretation:")
    if stats_dict['skewness'] > 0.5:
        print("  • Distribution is right-skewed (longer tail on the right)")
    elif stats_dict['skewness'] < -0.5:
        print("  • Distribution is left-skewed (longer tail on the left)")
    else:
        print("  • Distribution is approximately symmetric")
    
    if stats_dict['kurtosis'] > 0:
        print("  • Distribution has heavier tails than normal (leptokurtic)")
    elif stats_dict['kurtosis'] < 0:
        print("  • Distribution has lighter tails than normal (platykurtic)")
    else:
        print("  • Distribution has similar tail behavior to normal")

def create_visualizations(distances, embeddings, output_dir):
    """Create visualizations of the distance distribution."""
    print(f"\n🎨 Creating visualizations in {output_dir}...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Embedding L2 Distance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of distances
    axes[0, 0].hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(distances):.3f}')
    axes[0, 0].axvline(np.median(distances), color='orange', linestyle='--', 
                       label=f'Median: {np.median(distances):.3f}')
    axes[0, 0].set_xlabel('L2 Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of L2 Distances')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    box_plot = axes[0, 1].boxplot(distances, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    axes[0, 1].set_ylabel('L2 Distance')
    axes[0, 1].set_title('Box Plot of L2 Distances')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot (normal distribution comparison)
    stats.probplot(distances, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_distances = np.sort(distances)
    cumulative_prob = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    axes[1, 1].plot(sorted_distances, cumulative_prob, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('L2 Distance')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'embedding_distance_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_file}")
    
    # Show the plot
    plt.show()
    
    # Create a heatmap of distance matrix (sample for large matrices)
    if embeddings.shape[0] <= 100:
        # For smaller matrices, show full heatmap
        distance_matrix = squareform(distances)
        plt.figure(figsize=(12, 10))
        sns.heatmap(distance_matrix, cmap='viridis', square=True)
        plt.title('Pairwise L2 Distance Matrix Heatmap')
        plt.xlabel('Embedding Index')
        plt.ylabel('Embedding Index')
        
        heatmap_file = os.path.join(output_dir, 'distance_matrix_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved distance matrix heatmap to {heatmap_file}")
        plt.show()
    else:
        # For larger matrices, show a sample
        sample_size = min(50, embeddings.shape[0])
        sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_distances = pdist(sample_embeddings, metric='euclidean')
        sample_matrix = squareform(sample_distances)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sample_matrix, cmap='viridis', square=True)
        plt.title(f'Sample Pairwise L2 Distance Matrix Heatmap ({sample_size} embeddings)')
        plt.xlabel('Sample Embedding Index')
        plt.ylabel('Sample Embedding Index')
        
        heatmap_file = os.path.join(output_dir, 'distance_matrix_sample_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved sample distance matrix heatmap to {heatmap_file}")
        plt.show()

def save_results(stats_dict, distances, output_dir):
    """Save results to files."""
    print(f"\n💾 Saving results to {output_dir}...")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats_dict])
    stats_file = os.path.join(output_dir, 'embedding_distance_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"  ✓ Saved statistics to {stats_file}")
    
    # Save all distances to numpy file
    distances_file = os.path.join(output_dir, 'pairwise_l2_distances.npy')
    np.save(distances_file, distances)
    print(f"  ✓ Saved distances array to {distances_file}")
    
    # Save detailed statistics to text file
    report_file = os.path.join(output_dir, 'embedding_distance_report.txt')
    with open(report_file, 'w') as f:
        f.write("EMBEDDING L2 DISTANCE ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total number of pairwise distances: {stats_dict['count']:,}\n")
        f.write(f"Mean distance: {stats_dict['mean']:.6f}\n")
        f.write(f"Standard deviation: {stats_dict['std']:.6f}\n")
        f.write(f"Minimum distance: {stats_dict['min']:.6f}\n")
        f.write(f"Maximum distance: {stats_dict['max']:.6f}\n")
        f.write(f"Median distance: {stats_dict['median']:.6f}\n\n")
        
        f.write("PERCENTILES:\n")
        f.write(f"25th percentile (Q1): {stats_dict['q25']:.6f}\n")
        f.write(f"75th percentile (Q3): {stats_dict['q75']:.6f}\n")
        f.write(f"90th percentile: {stats_dict['q90']:.6f}\n")
        f.write(f"95th percentile: {stats_dict['q95']:.6f}\n")
        f.write(f"99th percentile: {stats_dict['q99']:.6f}\n\n")
        
        f.write("SPREAD MEASURES:\n")
        f.write(f"Interquartile Range (IQR): {stats_dict['iqr']:.6f}\n")
        f.write(f"Coefficient of Variation: {stats_dict['std']/stats_dict['mean']:.6f}\n\n")
        
        f.write("DISTRIBUTION SHAPE:\n")
        f.write(f"Skewness: {stats_dict['skewness']:.6f}\n")
        f.write(f"Kurtosis: {stats_dict['kurtosis']:.6f}\n")
    
    print(f"  ✓ Saved detailed report to {report_file}")

def find_closest_and_farthest_pairs(embeddings, distance_matrix):
    """Find the closest and farthest pairs of embeddings."""
    print("\n🔍 Finding closest and farthest embedding pairs...")
    
    # Mask the diagonal (distance from embedding to itself = 0)
    masked_matrix = distance_matrix.copy()
    np.fill_diagonal(masked_matrix, np.inf)
    
    # Find closest pair
    min_idx = np.unravel_index(np.argmin(masked_matrix), masked_matrix.shape)
    min_distance = masked_matrix[min_idx]
    
    # Find farthest pair
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    max_distance = distance_matrix[max_idx]
    
    print(f"  🔗 Closest pair: embeddings {min_idx[0]} and {min_idx[1]}")
    print(f"     Distance: {min_distance:.6f}")
    
    print(f"  🔗 Farthest pair: embeddings {max_idx[0]} and {max_idx[1]}")
    print(f"     Distance: {max_distance:.6f}")
    
    return {
        'closest_pair': min_idx,
        'closest_distance': min_distance,
        'farthest_pair': max_idx,
        'farthest_distance': max_distance
    }

def main():
    """Main analysis function."""
    print("🚀 Starting Embedding L2 Distance Analysis")
    print("="*50)
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    embeddings_file = base_dir / 'embeddings_array.npy'
    output_dir = base_dir / 'my_processing_agents'
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Load embeddings
    embeddings = load_embeddings(embeddings_file)
    if embeddings is None:
        return
    
    # Calculate distances
    distances, distance_matrix = calculate_l2_distances(embeddings)
    
    # Compute statistics
    stats_dict = compute_statistics(distances)
    
    # Print statistics
    print_statistics(stats_dict)
    
    # Find extreme pairs
    extreme_pairs = find_closest_and_farthest_pairs(embeddings, distance_matrix)
    
    # Create visualizations
    create_visualizations(distances, embeddings, output_dir)
    
    # Save results
    save_results(stats_dict, distances, output_dir)
    
    print("\n" + "="*60)
    print("✅ Analysis completed successfully!")
    print("="*60)
    
    return {
        'statistics': stats_dict,
        'extreme_pairs': extreme_pairs,
        'distances': distances,
        'distance_matrix': distance_matrix
    }

if __name__ == "__main__":
    results = main()
