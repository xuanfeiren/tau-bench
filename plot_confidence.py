import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_confidence_intervals():
    """Plot ExploreAlgorithm test scores with confidence intervals (LCB and UCB)"""
    df = pd.read_csv('data_to_plot/confidence.csv')
    
    plt.figure(figsize=(12, 8))
    
    # Extract data
    steps = df['Step'].values
    test_scores = df['ExploreAlgorithm_v1.0 - Test score'].astype(float).values
    test_min = df['ExploreAlgorithm_v1.0 - Test score__MIN'].astype(float).values
    test_max = df['ExploreAlgorithm_v1.0 - Test score__MAX'].astype(float).values
    
    # Lower and Upper Confidence Bounds
    lcb = df['ExploreAlgorithm_v1.0 - LCB'].replace('', np.nan)
    ucb = df['ExploreAlgorithm_v1.0 - UCB'].replace('', np.nan)
    lcb = pd.to_numeric(lcb, errors='coerce').values
    ucb = pd.to_numeric(ucb, errors='coerce').values
    
    # Plot test scores
    plt.plot(steps, test_scores, 'o-', linewidth=2.5, markersize=8, 
             label='Test Score', color='#2E86C1', zorder=3)
    
    # Plot confidence interval (LCB to UCB) where data exists
    valid_confidence = ~(np.isnan(lcb) | np.isnan(ucb))
    if np.any(valid_confidence):
        valid_steps = steps[valid_confidence]
        valid_lcb = lcb[valid_confidence]
        valid_ucb = ucb[valid_confidence]
        
        plt.fill_between(valid_steps, valid_lcb, valid_ucb, 
                        alpha=0.3, color='#2E86C1', 
                        label='Confidence Interval (LCB-UCB)', zorder=1)
        
        # Plot LCB and UCB lines
        plt.plot(valid_steps, valid_lcb, '--', linewidth=1.5, 
                color='#E74C3C', label='Lower Confidence Bound (LCB)', zorder=2)
        plt.plot(valid_steps, valid_ucb, '--', linewidth=1.5, 
                color='#27AE60', label='Upper Confidence Bound (UCB)', zorder=2)
    
    # Add min/max range as error bars (optional - can be removed if too cluttered)
    plt.errorbar(steps, test_scores, 
                yerr=[test_scores - test_min, test_max - test_scores],
                fmt='none', capsize=4, capthick=1, alpha=0.6, 
                color='#34495E', label='Min/Max Range', zorder=0)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('ExploreAlgorithm v1.0 Test Scores with Confidence Intervals', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(-0.5, max(steps) + 0.5)
    
    plt.tight_layout()
    plt.savefig('confidence_intervals.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_confidence_simple():
    """Alternative simpler plot focusing just on confidence intervals"""
    df = pd.read_csv('data_to_plot/confidence.csv')
    
    plt.figure(figsize=(12, 6))
    
    # Extract data
    steps = df['Step'].values
    test_scores = df['ExploreAlgorithm_v1.0 - Test score'].astype(float).values
    
    # Lower and Upper Confidence Bounds
    lcb = df['ExploreAlgorithm_v1.0 - LCB'].replace('', np.nan)
    ucb = df['ExploreAlgorithm_v1.0 - UCB'].replace('', np.nan)
    lcb = pd.to_numeric(lcb, errors='coerce').values
    ucb = pd.to_numeric(ucb, errors='coerce').values
    
    # Plot test scores
    plt.plot(steps, test_scores, 'o-', linewidth=3, markersize=8, 
             label='Test Score', color='#2E86C1')
    
    # Plot confidence interval where data exists
    valid_confidence = ~(np.isnan(lcb) | np.isnan(ucb))
    if np.any(valid_confidence):
        valid_steps = steps[valid_confidence]
        valid_lcb = lcb[valid_confidence]
        valid_ucb = ucb[valid_confidence]
        
        plt.fill_between(valid_steps, valid_lcb, valid_ucb, 
                        alpha=0.3, color='#2E86C1', 
                        label='95% Confidence Interval')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('ExploreAlgorithm v1.0 Performance with 95% Confidence Interval', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.xlim(-0.5, max(steps) + 0.5)
    plt.tight_layout()
    plt.savefig('confidence_intervals_simple.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def print_confidence_statistics():
    """Print summary statistics for confidence interval data"""
    df = pd.read_csv('data_to_plot/confidence.csv')
    
    test_scores = df['ExploreAlgorithm_v1.0 - Test score'].astype(float).values
    lcb = df['ExploreAlgorithm_v1.0 - LCB'].replace('', np.nan)
    ucb = df['ExploreAlgorithm_v1.0 - UCB'].replace('', np.nan)
    lcb = pd.to_numeric(lcb, errors='coerce').values
    ucb = pd.to_numeric(ucb, errors='coerce').values
    
    print("\nConfidence Interval Analysis:")
    print("="*50)
    
    print("Test Score Statistics:")
    print(f"  Mean: {np.mean(test_scores):.4f}")
    print(f"  Std:  {np.std(test_scores):.4f}")
    print(f"  Min:  {np.min(test_scores):.4f}")
    print(f"  Max:  {np.max(test_scores):.4f}")
    print()
    
    # Analyze confidence intervals
    valid_confidence = ~(np.isnan(lcb) | np.isnan(ucb))
    if np.any(valid_confidence):
        valid_lcb = lcb[valid_confidence]
        valid_ucb = ucb[valid_confidence]
        valid_scores = test_scores[valid_confidence]
        
        confidence_width = valid_ucb - valid_lcb
        
        print("Confidence Interval Statistics:")
        print(f"  Data points with CI: {len(valid_lcb)}/{len(test_scores)}")
        print(f"  Average CI width: {np.mean(confidence_width):.4f}")
        print(f"  Min CI width: {np.min(confidence_width):.4f}")
        print(f"  Max CI width: {np.max(confidence_width):.4f}")
        print()
        
        print("Lower Confidence Bound:")
        print(f"  Mean: {np.mean(valid_lcb):.4f}")
        print(f"  Range: {np.min(valid_lcb):.4f} - {np.max(valid_lcb):.4f}")
        print()
        
        print("Upper Confidence Bound:")
        print(f"  Mean: {np.mean(valid_ucb):.4f}")
        print(f"  Range: {np.min(valid_ucb):.4f} - {np.max(valid_ucb):.4f}")
        print()
        
        # Check if test scores are within confidence intervals
        within_ci = (valid_scores >= valid_lcb) & (valid_scores <= valid_ucb)
        print(f"Test scores within CI: {np.sum(within_ci)}/{len(within_ci)} ({np.mean(within_ci)*100:.1f}%)")

def show_confidence_data():
    """Display the confidence interval data in a readable format"""
    df = pd.read_csv('data_to_plot/confidence.csv')
    
    print("Confidence Interval Data:")
    print("="*80)
    print(f"{'Step':<4} {'Test Score':<12} {'LCB':<12} {'UCB':<12} {'CI Width':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        step = int(row['Step'])
        score = row['ExploreAlgorithm_v1.0 - Test score']
        lcb = row['ExploreAlgorithm_v1.0 - LCB']
        ucb = row['ExploreAlgorithm_v1.0 - UCB']
        
        if pd.notna(lcb) and pd.notna(ucb) and lcb != '' and ucb != '':
            lcb_val = float(lcb)
            ucb_val = float(ucb)
            width = ucb_val - lcb_val
            print(f"{step:<4} {score:<12.4f} {lcb_val:<12.4f} {ucb_val:<12.4f} {width:<12.4f}")
        else:
            print(f"{step:<4} {score:<12.4f} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

if __name__ == "__main__":
    # Show raw data
    show_confidence_data()
    
    # Create both plots
    plot_confidence_intervals()
    plot_confidence_simple()
    
    # Print statistics
    print_confidence_statistics()
    
    print("\nConfidence interval plots have been saved as:")
    print("- confidence_intervals.pdf (detailed version)")
    print("- confidence_intervals_simple.pdf (clean version)") 