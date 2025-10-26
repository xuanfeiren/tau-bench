#!/usr/bin/env python3
"""
Analyze epsilon-cover of 768-dimensional unit sphere by embedding vectors.
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm

def sample_unit_sphere(n_samples, dim=768):
    """Sample points uniformly on the unit sphere in dim dimensions."""
    # Sample from standard normal distribution
    points = np.random.randn(n_samples, dim)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def compute_epsilon_coverage(embeddings, test_points, epsilon):
    """
    Compute what fraction of test_points are within epsilon distance 
    of at least one embedding vector.
    """
    print(f"Computing coverage for epsilon={epsilon}...")
    
    # Compute distances between test points and all embeddings
    # Use chunking to avoid memory issues with large arrays
    chunk_size = 1000
    n_covered = 0
    
    for i in tqdm(range(0, len(test_points), chunk_size), desc="Processing chunks"):
        chunk_end = min(i + chunk_size, len(test_points))
        chunk = test_points[i:chunk_end]
        
        # Compute distances from this chunk to all embeddings
        distances = cdist(chunk, embeddings, metric='euclidean')
        
        # Check if any embedding is within epsilon distance
        min_distances = np.min(distances, axis=1)
        n_covered += np.sum(min_distances <= epsilon)
    
    coverage_fraction = n_covered / len(test_points)
    return coverage_fraction, n_covered

def analyze_embedding_distribution(embeddings):
    """Analyze the distribution of embedding vectors."""
    print("Analyzing embedding distribution...")
    
    # Check norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms - mean: {np.mean(norms):.6f}, std: {np.std(norms):.6f}")
    print(f"Embedding norms - min: {np.min(norms):.6f}, max: {np.max(norms):.6f}")
    
    # Compute pairwise distances between embeddings
    print("Computing pairwise distances between embeddings...")
    pairwise_distances = cdist(embeddings, embeddings, metric='euclidean')
    
    # Remove diagonal (distance to self = 0)
    mask = np.triu(np.ones_like(pairwise_distances, dtype=bool), k=1)
    distances = pairwise_distances[mask]
    
    print(f"Pairwise distances - mean: {np.mean(distances):.4f}, std: {np.std(distances):.4f}")
    print(f"Pairwise distances - min: {np.min(distances):.4f}, max: {np.max(distances):.4f}")
    
    return distances

def main():
    print("Loading embeddings...")
    embeddings = np.load('embeddings_array.npy')
    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Normalize embeddings to unit sphere (in case they're not exactly normalized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Analyze embedding distribution
    pairwise_distances = analyze_embedding_distribution(embeddings)
    
    # Sample test points on unit sphere
    n_test_points = 100000  # Start with 100k points
    print(f"\nSampling {n_test_points} test points on unit sphere...")
    test_points = sample_unit_sphere(n_test_points, dim=768)
    
    # Test different epsilon values
    epsilons = [0.1, 0.3]
    results = {}
    
    print(f"\nTesting epsilon-cover with {len(embeddings)} embedding vectors...")
    
    for epsilon in epsilons:
        coverage_fraction, n_covered = compute_epsilon_coverage(embeddings, test_points, epsilon)
        results[epsilon] = {
            'coverage_fraction': coverage_fraction,
            'n_covered': n_covered,
            'n_total': n_test_points
        }
        
        print(f"\nEpsilon = {epsilon}:")
        print(f"  Coverage: {coverage_fraction:.6f} ({coverage_fraction*100:.4f}%)")
        print(f"  Points covered: {n_covered} out of {n_test_points}")
    
    # Theoretical analysis
    print(f"\n" + "="*60)
    print("THEORETICAL CONTEXT:")
    print(f"Number of embedding vectors: {len(embeddings)}")
    print(f"Dimension: {embeddings.shape[1]}")
    
    # Volume of epsilon-ball in high dimensions
    from scipy.special import gamma
    for epsilon in epsilons:
        # Volume of d-dimensional ball of radius epsilon
        d = 768
        vol_ball = (np.pi**(d/2) / gamma(d/2 + 1)) * (epsilon**d)
        # Surface area of unit sphere
        surf_area = 2 * (np.pi**(d/2) / gamma(d/2))
        
        print(f"\nEpsilon = {epsilon}:")
        print(f"  Volume of {epsilon}-ball: {vol_ball:.2e}")
        print(f"  Surface area of unit sphere: {surf_area:.2e}")
        print(f"  Naive coverage estimate: {min(1.0, len(embeddings) * vol_ball / surf_area):.6f}")
    
    # Save results
    print(f"\nSaving results...")
    np.save('epsilon_cover_results.npy', results)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Coverage results
    plt.subplot(2, 2, 1)
    eps_vals = list(results.keys())
    coverage_vals = [results[eps]['coverage_fraction'] for eps in eps_vals]
    plt.bar(range(len(eps_vals)), coverage_vals)
    plt.xlabel('Epsilon')
    plt.ylabel('Coverage Fraction')
    plt.title('Epsilon-Cover of Unit Sphere')
    plt.xticks(range(len(eps_vals)), [f'{eps}' for eps in eps_vals])
    for i, v in enumerate(coverage_vals):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Plot 2: Pairwise distance distribution
    plt.subplot(2, 2, 2)
    plt.hist(pairwise_distances, bins=50, alpha=0.7, density=True)
    plt.axvline(0.1, color='red', linestyle='--', label='ε=0.1')
    plt.axvline(0.3, color='orange', linestyle='--', label='ε=0.3')
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Pairwise Distances')
    plt.legend()
    
    # Plot 3: Coverage vs epsilon (more detailed)
    plt.subplot(2, 2, 3)
    detailed_epsilons = np.linspace(0.05, 0.5, 10)
    detailed_coverage = []
    
    print(f"\nComputing detailed coverage curve...")
    for eps in tqdm(detailed_epsilons, desc="Detailed analysis"):
        coverage, _ = compute_epsilon_coverage(embeddings, test_points[:10000], eps)  # Use fewer points for speed
        detailed_coverage.append(coverage)
    
    plt.plot(detailed_epsilons, detailed_coverage, 'b-o', markersize=4)
    plt.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='ε=0.1')
    plt.axvline(0.3, color='orange', linestyle='--', alpha=0.7, label='ε=0.3')
    plt.xlabel('Epsilon')
    plt.ylabel('Coverage Fraction')
    plt.title('Coverage vs Epsilon')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Embedding norms
    plt.subplot(2, 2, 4)
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    plt.hist(embedding_norms, bins=30, alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', label='Unit norm')
    plt.xlabel('L2 Norm')
    plt.ylabel('Count')
    plt.title('Distribution of Embedding Norms')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('epsilon_cover_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAnalysis complete! Results saved to epsilon_cover_results.npy")
    print(f"Visualization saved to epsilon_cover_analysis.png")

if __name__ == "__main__":
    main()
