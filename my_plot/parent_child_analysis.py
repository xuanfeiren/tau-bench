#!/usr/bin/env python3
"""
Parent-Child Score Analysis Tool

Analyzes the relationship between parent candidate scores and child candidate scores.
Supports both empirical mean scores and predicted scores from regressors.

Usage:
    python parent_child_analysis.py --empirical ../parent_children_scores.npy
    python parent_child_analysis.py --predicted ../predicted_parent_children_scores.npy
    python parent_child_analysis.py --both ../parent_children_scores.npy ../predicted_parent_children_scores.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import argparse
import os
import sys
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ParentChildAnalyzer:
    """Comprehensive analyzer for parent-child score relationships."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.results = {}
        
    def load_data(self, filepath: str) -> np.ndarray:
        """
        Load data from .npy file with error handling.
        
        Args:
            filepath: Path to the .npy file
            
        Returns:
            Loaded numpy array
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            data = np.load(filepath)
            print(f"✓ Loaded data from {filepath}")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            
            return data
            
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            sys.exit(1)
    
    def extract_parent_child_scores(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract parent and children scores from the data array.
        
        Assumes data structure where:
        - First column/element represents parent scores
        - Remaining columns/elements represent children scores
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (parent_scores, children_scores)
        """
        if data.ndim == 1:
            # If 1D array, assume alternating parent-child pattern or split in half
            if len(data) % 2 == 0:
                mid = len(data) // 2
                parent_scores = data[:mid]
                children_scores = data[mid:]
            else:
                # Assume first element is parent, rest are children
                parent_scores = np.array([data[0]] * (len(data) - 1))
                children_scores = data[1:]
        elif data.ndim == 2:
            # If 2D array, assume first column is parent, rest are children
            parent_scores = data[:, 0]
            children_scores = data[:, 1:].flatten()
            # Repeat parent scores to match children count
            n_children_per_parent = data.shape[1] - 1
            parent_scores = np.repeat(parent_scores, n_children_per_parent)
        else:
            raise ValueError(f"Unsupported data dimensionality: {data.ndim}")
        
        print(f"  Parent scores: {len(parent_scores)} values")
        print(f"  Children scores: {len(children_scores)} values")
        
        return parent_scores, children_scores
    
    def compute_statistics(self, parent_scores: np.ndarray, children_scores: np.ndarray, 
                          label: str = "") -> Dict[str, Any]:
        """
        Compute comprehensive statistics for parent-child score relationships.
        
        Args:
            parent_scores: Array of parent scores
            children_scores: Array of children scores
            label: Label for this analysis
            
        Returns:
            Dictionary containing statistical results
        """
        print(f"\n📊 Computing statistics{' for ' + label if label else ''}...")
        
        # Remove any NaN or infinite values
        mask = np.isfinite(parent_scores) & np.isfinite(children_scores)
        parent_clean = parent_scores[mask]
        children_clean = children_scores[mask]
        
        if len(parent_clean) == 0:
            print("⚠️  No valid data points found!")
            return {}
        
        stats_dict = {
            'n_samples': len(parent_clean),
            'parent_stats': {
                'mean': np.mean(parent_clean),
                'std': np.std(parent_clean),
                'min': np.min(parent_clean),
                'max': np.max(parent_clean),
                'median': np.median(parent_clean)
            },
            'children_stats': {
                'mean': np.mean(children_clean),
                'std': np.std(children_clean),
                'min': np.min(children_clean),
                'max': np.max(children_clean),
                'median': np.median(children_clean)
            }
        }
        
        # Correlation analyses
        try:
            pearson_r, pearson_p = pearsonr(parent_clean, children_clean)
            spearman_r, spearman_p = spearmanr(parent_clean, children_clean)
            kendall_tau, kendall_p = kendalltau(parent_clean, children_clean)
            
            stats_dict['correlations'] = {
                'pearson': {'r': pearson_r, 'p_value': pearson_p},
                'spearman': {'r': spearman_r, 'p_value': spearman_p},
                'kendall': {'tau': kendall_tau, 'p_value': kendall_p}
            }
        except Exception as e:
            print(f"⚠️  Error computing correlations: {e}")
            stats_dict['correlations'] = {}
        
        # Linear regression
        try:
            X = parent_clean.reshape(-1, 1)
            y = children_clean
            
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            
            stats_dict['regression'] = {
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'r2_score': r2_score(y, y_pred),
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred))
            }
        except Exception as e:
            print(f"⚠️  Error computing regression: {e}")
            stats_dict['regression'] = {}
        
        # Print summary
        self._print_statistics_summary(stats_dict, label)
        
        return stats_dict
    
    def _print_statistics_summary(self, stats_dict: Dict[str, Any], label: str = ""):
        """Print a formatted summary of statistics."""
        print(f"\n📈 Statistical Summary{' - ' + label if label else ''}:")
        print("=" * 50)
        
        print(f"Sample size: {stats_dict['n_samples']}")
        
        print(f"\nParent Scores:")
        p_stats = stats_dict['parent_stats']
        print(f"  Mean: {p_stats['mean']:.4f} ± {p_stats['std']:.4f}")
        print(f"  Range: [{p_stats['min']:.4f}, {p_stats['max']:.4f}]")
        print(f"  Median: {p_stats['median']:.4f}")
        
        print(f"\nChildren Scores:")
        c_stats = stats_dict['children_stats']
        print(f"  Mean: {c_stats['mean']:.4f} ± {c_stats['std']:.4f}")
        print(f"  Range: [{c_stats['min']:.4f}, {c_stats['max']:.4f}]")
        print(f"  Median: {c_stats['median']:.4f}")
        
        if 'correlations' in stats_dict and stats_dict['correlations']:
            print(f"\nCorrelations:")
            corr = stats_dict['correlations']
            if 'pearson' in corr:
                print(f"  Pearson r: {corr['pearson']['r']:.4f} (p={corr['pearson']['p_value']:.4f})")
            if 'spearman' in corr:
                print(f"  Spearman ρ: {corr['spearman']['r']:.4f} (p={corr['spearman']['p_value']:.4f})")
            if 'kendall' in corr:
                print(f"  Kendall τ: {corr['kendall']['tau']:.4f} (p={corr['kendall']['p_value']:.4f})")
        
        if 'regression' in stats_dict and stats_dict['regression']:
            print(f"\nLinear Regression:")
            reg = stats_dict['regression']
            print(f"  Equation: y = {reg['slope']:.4f}x + {reg['intercept']:.4f}")
            print(f"  R² Score: {reg['r2_score']:.4f}")
            print(f"  RMSE: {reg['rmse']:.4f}")
    
    def create_comprehensive_plot(self, parent_scores: np.ndarray, children_scores: np.ndarray, 
                                 title: str = "Parent vs Children Scores", 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive analysis plot with statistics embedded.
        
        Args:
            parent_scores: Array of parent scores
            children_scores: Array of children scores
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Clean data
        mask = np.isfinite(parent_scores) & np.isfinite(children_scores)
        parent_clean = parent_scores[mask]
        children_clean = children_scores[mask]
        
        # Create figure with better layout
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main scatter plot (larger, spans 2x2)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        ax_main.scatter(parent_clean, children_clean, alpha=0.7, s=60, color='steelblue', edgecolors='white', linewidth=0.5)
        
        # Add y=x reference line (parent = children)
        min_val = min(min(parent_clean), min(children_clean))
        max_val = max(max(parent_clean), max(children_clean))
        ax_main.plot([min_val, max_val], [min_val, max_val], 'gray', linestyle='--', alpha=0.7, linewidth=2, label='Parent = Children')
        
        # Add regression line and statistics
        if len(parent_clean) > 1:
            z = np.polyfit(parent_clean, children_clean, 1)
            p = np.poly1d(z)
            ax_main.plot(parent_clean, p(parent_clean), "red", alpha=0.8, linewidth=2.5, label='Regression Line')
            
            # Calculate comprehensive statistics
            pearson_r, pearson_p = pearsonr(parent_clean, children_clean)
            spearman_r, spearman_p = spearmanr(parent_clean, children_clean)
            
            # Create statistics text box
            stats_text = f'''Statistics (n={len(parent_clean)}):
Pearson r = {pearson_r:.3f} (p={pearson_p:.3f})
Spearman ρ = {spearman_r:.3f} (p={spearman_p:.3f})
R² = {pearson_r**2:.3f}
Slope = {z[0]:.3f}
Intercept = {z[1]:.3f}'''
            
            ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                        verticalalignment='top', fontsize=10, family='monospace')
        
        ax_main.set_xlabel('Parent Scores', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Children Scores', fontsize=12, fontweight='bold')
        ax_main.set_title('Parent-Child Score Relationship', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='lower right', fontsize=10)
        
        # Combined distribution plot (top right)
        ax_dist = fig.add_subplot(gs[0, 2])
        ax_dist.hist(parent_clean, bins=20, alpha=0.6, color='skyblue', label=f'Parent (μ={np.mean(parent_clean):.2f})', density=True)
        ax_dist.hist(children_clean, bins=20, alpha=0.6, color='lightcoral', label=f'Children (μ={np.mean(children_clean):.2f})', density=True)
        ax_dist.set_xlabel('Score Values')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title('Score Distributions')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Residual plot (middle right)
        if len(parent_clean) > 1:
            ax_resid = fig.add_subplot(gs[1, 2])
            residuals = children_clean - p(parent_clean)
            ax_resid.scatter(parent_clean, residuals, alpha=0.6, s=40, color='green')
            ax_resid.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Add residual statistics
            resid_std = np.std(residuals)
            ax_resid.axhline(y=2*resid_std, color='orange', linestyle=':', alpha=0.6)
            ax_resid.axhline(y=-2*resid_std, color='orange', linestyle=':', alpha=0.6)
            
            ax_resid.set_xlabel('Parent Scores')
            ax_resid.set_ylabel('Residuals')
            ax_resid.set_title(f'Residuals (σ={resid_std:.3f})')
            ax_resid.grid(True, alpha=0.3)
        
        # Summary statistics table (bottom)
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        # Create summary table
        summary_data = [
            ['Metric', 'Parent Scores', 'Children Scores'],
            ['Count', f'{len(parent_clean)}', f'{len(children_clean)}'],
            ['Mean ± Std', f'{np.mean(parent_clean):.3f} ± {np.std(parent_clean):.3f}', 
             f'{np.mean(children_clean):.3f} ± {np.std(children_clean):.3f}'],
            ['Range', f'[{np.min(parent_clean):.3f}, {np.max(parent_clean):.3f}]', 
             f'[{np.min(children_clean):.3f}, {np.max(children_clean):.3f}]'],
            ['Median', f'{np.median(parent_clean):.3f}', f'{np.median(children_clean):.3f}']
        ]
        
        table = ax_table.table(cellText=summary_data, cellLoc='center', loc='center',
                              colWidths=[0.2, 0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comprehensive plot saved to {save_path}")
        
        return fig
    
    def compare_empirical_vs_predicted(self, empirical_data: np.ndarray, 
                                     predicted_data: np.ndarray) -> Dict[str, Any]:
        """
        Compare empirical and predicted scores (console output only).
        
        Args:
            empirical_data: Empirical scores data
            predicted_data: Predicted scores data
            
        Returns:
            Dictionary containing comparison results
        """
        print("\n🔄 Comparing Empirical vs Predicted Scores...")
        
        # Extract scores
        emp_parent, emp_children = self.extract_parent_child_scores(empirical_data)
        pred_parent, pred_children = self.extract_parent_child_scores(predicted_data)
        
        # Ensure same length for comparison
        min_len = min(len(emp_parent), len(pred_parent))
        emp_parent = emp_parent[:min_len]
        pred_parent = pred_parent[:min_len]
        
        min_len_children = min(len(emp_children), len(pred_children))
        emp_children = emp_children[:min_len_children]
        pred_children = pred_children[:min_len_children]
        
        # Compute comparison metrics
        comparison_results = {
            'parent_comparison': {
                'correlation': pearsonr(emp_parent, pred_parent)[0] if len(emp_parent) > 1 else 0,
                'mse': mean_squared_error(emp_parent, pred_parent),
                'mae': mean_absolute_error(emp_parent, pred_parent),
                'r2': r2_score(emp_parent, pred_parent)
            },
            'children_comparison': {
                'correlation': pearsonr(emp_children, pred_children)[0] if len(emp_children) > 1 else 0,
                'mse': mean_squared_error(emp_children, pred_children),
                'mae': mean_absolute_error(emp_children, pred_children),
                'r2': r2_score(emp_children, pred_children)
            }
        }
        
        # Print detailed comparison summary
        print("\n📊 Detailed Comparison Summary:")
        print("=" * 60)
        print(f"{'Metric':<20} {'Parent Scores':<18} {'Children Scores':<18}")
        print("-" * 60)
        print(f"{'Sample Size':<20} {len(emp_parent):<18} {len(emp_children):<18}")
        print(f"{'Correlation (r)':<20} {comparison_results['parent_comparison']['correlation']:<18.4f} {comparison_results['children_comparison']['correlation']:<18.4f}")
        print(f"{'R² Score':<20} {comparison_results['parent_comparison']['r2']:<18.4f} {comparison_results['children_comparison']['r2']:<18.4f}")
        print(f"{'MAE':<20} {comparison_results['parent_comparison']['mae']:<18.4f} {comparison_results['children_comparison']['mae']:<18.4f}")
        print(f"{'RMSE':<20} {np.sqrt(comparison_results['parent_comparison']['mse']):<18.4f} {np.sqrt(comparison_results['children_comparison']['mse']):<18.4f}")
        
        # Calculate and display error statistics
        parent_errors = emp_parent - pred_parent
        children_errors = emp_children - pred_children
        
        print(f"{'Error Mean':<20} {np.mean(parent_errors):<18.4f} {np.mean(children_errors):<18.4f}")
        print(f"{'Error Std':<20} {np.std(parent_errors):<18.4f} {np.std(children_errors):<18.4f}")
        print("=" * 60)
        
        return comparison_results
    
    def analyze_single_dataset(self, filepath: str, label: str = "") -> Dict[str, Any]:
        """
        Analyze a single dataset (empirical or predicted).
        
        Args:
            filepath: Path to the .npy file
            label: Label for this analysis
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n🔍 Analyzing {label if label else 'dataset'}: {filepath}")
        
        # Load data
        data = self.load_data(filepath)
        
        # Extract parent and children scores
        parent_scores, children_scores = self.extract_parent_child_scores(data)
        
        # Compute statistics
        stats = self.compute_statistics(parent_scores, children_scores, label)
        
        # Create comprehensive plot
        plot_title = f"Parent vs Children Scores{' - ' + label if label else ''}"
        save_path = f"{label.lower().replace(' ', '_')}_analysis.png" if label else "analysis.png"
        
        fig = self.create_comprehensive_plot(parent_scores, children_scores, plot_title, save_path)
        plt.show()
        
        return {
            'data': data,
            'parent_scores': parent_scores,
            'children_scores': children_scores,
            'statistics': stats,
            'figure': fig
        }


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Analyze parent-child score relationships')
    
    # Default file paths (relative to current working directory)
    default_empirical = 'parent_children_scores.npy'
    default_predicted = 'predicted_parent_children_scores.npy'
    
    parser.add_argument('--empirical', type=str, 
                      help='Path to empirical scores .npy file')
    parser.add_argument('--predicted', type=str, 
                      help='Path to predicted scores .npy file')
    parser.add_argument('--both', nargs=2, metavar=('EMPIRICAL', 'PREDICTED'),
                      help='Paths to both empirical and predicted scores .npy files')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ParentChildAnalyzer(figsize=tuple(args.figsize))
    
    print("🚀 Parent-Child Score Analysis Tool")
    print("=" * 50)
    
    # Determine what analysis to perform
    empirical_path = None
    predicted_path = None
    
    if args.both:
        # Explicit both files provided
        empirical_path, predicted_path = args.both
    elif args.empirical:
        # Only empirical provided
        empirical_path = args.empirical
    elif args.predicted:
        # Only predicted provided
        predicted_path = args.predicted
    else:
        # No arguments provided - check for default files
        print("🔍 No arguments provided, checking for default files...")
        
        empirical_exists = os.path.exists(default_empirical)
        predicted_exists = os.path.exists(default_predicted)
        
        if empirical_exists and predicted_exists:
            print(f"✓ Found both default files - analyzing both datasets")
            empirical_path = default_empirical
            predicted_path = default_predicted
        elif empirical_exists:
            print(f"✓ Found empirical file only: {default_empirical}")
            empirical_path = default_empirical
        elif predicted_exists:
            print(f"✓ Found predicted file only: {default_predicted}")
            predicted_path = default_predicted
        else:
            print("❌ No default files found. Please provide file paths using:")
            print("  --empirical <path>     for empirical scores only")
            print("  --predicted <path>     for predicted scores only") 
            print("  --both <emp> <pred>    for both datasets")
            sys.exit(1)
    
    # Perform analysis based on available data
    if empirical_path and predicted_path:
        # Analyze both and compare
        print(f"\n📊 Analyzing both datasets:")
        print(f"  Empirical: {empirical_path}")
        print(f"  Predicted: {predicted_path}")
        
        # Analyze empirical data
        emp_results = analyzer.analyze_single_dataset(empirical_path, "Empirical")
        
        # Analyze predicted data
        pred_results = analyzer.analyze_single_dataset(predicted_path, "Predicted")
        
        # Compare both datasets
        comparison = analyzer.compare_empirical_vs_predicted(
            emp_results['data'], pred_results['data']
        )
        
    elif empirical_path:
        # Analyze empirical data only
        print(f"\n📊 Analyzing empirical dataset: {empirical_path}")
        results = analyzer.analyze_single_dataset(empirical_path, "Empirical")
        
    elif predicted_path:
        # Analyze predicted data only
        print(f"\n📊 Analyzing predicted dataset: {predicted_path}")
        results = analyzer.analyze_single_dataset(predicted_path, "Predicted")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()