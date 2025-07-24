import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

def calculate_proportion_with_error(data, key):
    """Calculate proportion and 95% confidence interval using Wilson score interval"""
    n = len(data)
    if n == 0:
        return 0, 0, 0
    
    # Count successes
    successes = sum(1 for item in data if item.get(key, False))
    p = successes / n
    
    # Wilson score interval for 95% confidence
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    # Return proportion and error (half the confidence interval width)
    error = (upper_bound - lower_bound) / 2
    return p, error, n

def load_json_files(file_patterns):
    """Load multiple JSON files based on patterns"""
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    datasets = {}
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get filename without extension for label
            label = Path(file_path).stem
            datasets[label] = data
            print(f"Loaded {len(data)} records from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return datasets

def calculate_mismatch(data):
    """Calculate fraction where setting and option guesses are mismatched"""
    mismatches = 0
    total = 0
    
    for item in data:
        setting_correct = item.get('did_guess_setting_correctly', False)
        option_correct = item.get('did_guess_correct_option', False)
        
        # Mismatch occurs when one is correct and the other is not
        if setting_correct != option_correct:
            mismatches += 1
        total += 1
    
    return mismatches / total if total > 0 else 0

def create_grouped_bar_plot(data_dict, metric_key, title, ylabel, filename=None, 
                           groups=None, labels=None, colors=None, group_labels=None):
    """Create a grouped bar plot with error bars"""
    file_labels = list(data_dict.keys()) if labels is None else labels
    proportions = []
    errors = []
    sample_sizes = []
    
    for key, data in data_dict.items():
        prop, error, n = calculate_proportion_with_error(data, metric_key)
        proportions.append(prop)
        errors.append(error)
        sample_sizes.append(n)
    
    # Default grouping (all in one group)
    if groups is None:
        groups = [0] * len(file_labels)
    
    # Default colors
    default_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange', 
                     'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if colors is None:
        bar_colors = [default_colors[i % len(default_colors)] for i in range(len(file_labels))]
    else:
        bar_colors = [default_colors[c % len(default_colors)] for c in colors]
    
    # Group the data
    unique_groups = sorted(set(groups))
    group_data = {g: [] for g in unique_groups}
    
    for i, group in enumerate(groups):
        group_data[group].append({
            'label': file_labels[i],
            'proportion': proportions[i],
            'error': errors[i],
            'sample_size': sample_sizes[i],
            'color': bar_colors[i]
        })
    
    # Calculate positions
    fig, ax = plt.subplots(figsize=(10, 6))  # Reduced width from 15 to 10
    
    bar_width = 0.8
    group_gap = 1.5  # Gap between groups
    within_group_gap = 0.0  # No gap within groups
    
    positions = []
    group_centers = []
    current_pos = 0
    
    for group_idx, group in enumerate(unique_groups):
        group_size = len(group_data[group])
        group_start = current_pos
        
        for i in range(group_size):
            positions.append(current_pos)
            current_pos += bar_width + within_group_gap
        
        # Calculate group center for group label
        group_end = current_pos - within_group_gap
        group_centers.append((group_start + group_end - bar_width) / 2)
        
        # Add gap for next group
        current_pos += group_gap
    
    # Plot bars
    all_bars = []
    pos_idx = 0
    
    for group in unique_groups:
        for item in group_data[group]:
            bar = ax.bar(positions[pos_idx], item['proportion'], 
                        yerr=item['error'], capsize=5, alpha=0.7,
                        color=item['color'], width=bar_width)
            all_bars.append((bar, item))
            pos_idx += 1
    
    # Add annotations with proportion ± error and sample size
    for bar, item in all_bars:
        height = bar[0].get_height()
        # Format proportion ± error
        prop_text = f"{item['proportion']:.3f} ± {item['error']:.3f}"
        sample_text = f"n={item['sample_size']}"
        
        # Display proportion ± error above the bar
        ax.text(bar[0].get_x() + bar[0].get_width()/2., 
               height + item['error'] + 0.01,
               prop_text, ha='center', va='bottom', fontsize=8)
        
        # Display sample size below the proportion
        ax.text(bar[0].get_x() + bar[0].get_width()/2., 
               height + item['error'] + 0.04,
               sample_text, ha='center', va='bottom', fontsize=8)
    
    # Set labels
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.2)  # Increased to accommodate both text lines
    
    # Individual labels
    ax.set_xticks(positions)
    ax.set_xticklabels([item['label'] for group in unique_groups for item in group_data[group]], 
                      rotation=45, ha='right')
    
    # Add group labels if provided with larger separation
    if group_labels is not None and len(group_labels) == len(unique_groups):
        # Add group labels below individual labels with larger separation
        for i, (center, group_label) in enumerate(zip(group_centers, group_labels)):
            ax.text(center, -0.25, group_label, ha='center', va='top', 
                   fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())
    
    # Add vertical lines between groups
    if len(unique_groups) > 1:
        for i in range(len(unique_groups) - 1):
            # Find the boundary between groups
            boundary = (group_centers[i] + group_centers[i + 1]) / 2
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        fig.savefig(filename, dpi=500, bbox_inches='tight')
        print(f"Figure saved as {filename}")
    
    return fig

def create_bar_plot(data_dict, metric_key, title, ylabel, filename=None):
    """Create a simple bar plot with error bars (backward compatibility)"""
    return create_grouped_bar_plot(data_dict, metric_key, title, ylabel, filename)

def analyze_json_files(file_patterns, groups=None, labels=None, colors=None, group_labels=None):
    """Main analysis function with grouping support"""
    # Load all datasets
    datasets = load_json_files(file_patterns)
    
    if not datasets:
        print("No data files found!")
        return
    
    # Create plots for each metric with grouping
    fig1 = create_grouped_bar_plot(datasets, 'did_guess_setting_correctly', 
                          'Fraction of Times Corretly Chose the Answer',
                          'Proportion Correct', 'setting_correct_analysis.png',
                          groups=groups, labels=labels, colors=colors, group_labels=group_labels)
    
    fig2 = create_grouped_bar_plot(datasets, 'did_guess_correct_option',
                          'Fraction of Times Correctly Identified the Setting', 
                          'Proportion Correct', 'option_correct_analysis.png',
                          groups=groups, labels=labels, colors=colors, group_labels=group_labels)
    
def analyze_json_files(file_patterns, groups=None, labels=None, colors=None, group_labels=None):
    """Main analysis function with grouping support"""
    # Load all datasets
    datasets = load_json_files(file_patterns)
    
    if not datasets:
        print("No data files found!")
        return
    
    # Create plots for each metric with grouping
    fig1 = create_grouped_bar_plot(datasets, 'did_guess_setting_correctly', 
                          'Fraction of Times "did_guess_setting_correctly" is True',
                          'Proportion Correct', 'setting_correct_analysis.png',
                          groups=groups, labels=labels, colors=colors, group_labels=group_labels)
    
    fig2 = create_grouped_bar_plot(datasets, 'did_guess_correct_option',
                          'Fraction of Times "did_guess_correct_option" is True', 
                          'Proportion Correct', 'option_correct_analysis.png',
                          groups=groups, labels=labels, colors=colors, group_labels=group_labels)
    
    # Calculate mismatch rates
    dataset_keys = list(datasets.keys())
    file_labels = dataset_keys if labels is None else labels
    mismatch_rates = []
    mismatch_errors = []
    sample_sizes = []
    
    for key, data in datasets.items():
        # Calculate mismatch as a binary outcome for each item
        mismatch_data = []
        for item in data:
            setting_correct = item.get('did_guess_setting_correctly', False)
            option_correct = item.get('did_guess_correct_option', False)
            mismatch_data.append({'mismatch': setting_correct != option_correct})
        
        prop, error, n = calculate_proportion_with_error(mismatch_data, 'mismatch')
        mismatch_rates.append(prop)
        mismatch_errors.append(error)
        sample_sizes.append(n)
    
    # Plot mismatch rates with grouping
    # Default grouping (all in one group)
    if groups is None:
        plot_groups = [0] * len(file_labels)
    else:
        plot_groups = groups
    
    # Default colors
    default_colors = ['orange', 'red', 'purple', 'brown', 'pink', 'gray', 
                     'olive', 'cyan', 'magenta', 'yellow', 'blue', 'green']
    if colors is None:
        bar_colors = [default_colors[i % len(default_colors)] for i in range(len(file_labels))]
    else:
        bar_colors = [default_colors[c % len(default_colors)] for c in colors]
    
    # Group the mismatch data
    unique_groups = sorted(set(plot_groups))
    group_data = {g: [] for g in unique_groups}
    
    for i, group in enumerate(plot_groups):
        group_data[group].append({
            'label': file_labels[i],
            'rate': mismatch_rates[i],
            'error': mismatch_errors[i],
            'sample_size': sample_sizes[i],
            'color': bar_colors[i]
        })
    
    # Create the mismatch plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))  # Reduced width from 15 to 10
    
    bar_width = 0.8
    group_gap = 1.5
    within_group_gap = 0.0  # No gap within groups
    
    positions = []
    group_centers = []
    current_pos = 0
    
    for group_idx, group in enumerate(unique_groups):
        group_size = len(group_data[group])
        group_start = current_pos
        
        for i in range(group_size):
            positions.append(current_pos)
            current_pos += bar_width + within_group_gap
        
        group_end = current_pos - within_group_gap
        group_centers.append((group_start + group_end - bar_width) / 2)
        current_pos += group_gap
    
    # Plot mismatch bars
    all_bars = []
    pos_idx = 0
    
    for group in unique_groups:
        for item in group_data[group]:
            bar = ax3.bar(positions[pos_idx], item['rate'], 
                         yerr=item['error'], capsize=5, alpha=0.7,
                         color=item['color'], width=bar_width)
            all_bars.append((bar, item))
            pos_idx += 1
    
    # Add annotations with rate ± error and sample size
    for bar, item in all_bars:
        height = bar[0].get_height()
        # Format rate ± error
        rate_text = f"{item['rate']:.3f} ± {item['error']:.3f}"
        sample_text = f"n={item['sample_size']}"
        
        # Display rate ± error above the bar
        ax3.text(bar[0].get_x() + bar[0].get_width()/2., 
                height + item['error'] + 0.01,
                rate_text, ha='center', va='bottom', fontsize=8)
        
        # Display sample size below the rate
        ax3.text(bar[0].get_x() + bar[0].get_width()/2., 
                height + item['error'] + 0.04,
                sample_text, ha='center', va='bottom', fontsize=8)
    
    ax3.set_ylabel('Mismatch Rate')
    ax3.set_title('Fraction of Setting and Answers are Mismatched')
    ax3.set_ylim(0, 1.2)  # Increased to accommodate both text lines
    
    # Individual labels
    ax3.set_xticks(positions)
    ax3.set_xticklabels([item['label'] for group in unique_groups for item in group_data[group]], 
                       rotation=45, ha='right')
    
    # Add group labels if provided with larger separation
    if group_labels is not None and len(group_labels) == len(unique_groups):
        for i, (center, group_label) in enumerate(zip(group_centers, group_labels)):
            ax3.text(center, -0.25, group_label, ha='center', va='top', 
                    fontsize=12, fontweight='bold', transform=ax3.get_xaxis_transform())
    
    # Add vertical lines between groups
    if len(unique_groups) > 1:
        for i in range(len(unique_groups) - 1):
            boundary = (group_centers[i] + group_centers[i + 1]) / 2
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save the mismatch figure
    fig3.savefig('mismatch_analysis.png', dpi=500, bbox_inches='tight')
    print("Figure saved as mismatch_analysis.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    dataset_keys = list(datasets.keys())
    display_labels = dataset_keys if labels is None else labels
    
    for i, (key, data) in enumerate(datasets.items()):
        setting_prop, setting_err, n = calculate_proportion_with_error(data, 'did_guess_setting_correctly')
        option_prop, option_err, _ = calculate_proportion_with_error(data, 'did_guess_correct_option')
        mismatch_rate = mismatch_rates[i]
        mismatch_error = mismatch_errors[i]
        
        display_label = display_labels[i]
        print(f"\n{display_label} (n={n}):")
        print(f"  Setting Correct: {setting_prop:.3f} ± {setting_err:.3f}")
        print(f"  Option Correct:  {option_prop:.3f} ± {option_err:.3f}")
        print(f"  Mismatch Rate:   {mismatch_rate:.3f} ± {mismatch_error:.3f}")
    
    plt.show()
    return fig1, fig2, fig3

# Example usage:
if __name__ == "__main__":
    # Specify your JSON file patterns here
    file_patterns = [
        "ultrafeedback_RMdataset_shuffled_5example_dataset_output.json", 
        "ultrafeedback_olmoSFT1B_eval_filtered_5example_dataset_output.json", 
        "ultrafeedback_olmoSFT7B_eval_filtered_5example_dataset_output.json",
        "ultrafeedback_RMdataset_shuffled_10example_dataset_output.json",          
        "ultrafeedback_olmoSFT1B_eval_filtered_10example_dataset_output.json", 
        "ultrafeedback_olmoSFT7B_eval_filtered_10example_dataset_output.json", 
    ]
    groups = [0, 0, 0, 1, 1, 1]
    labels = ["RM Dataset", "Olmo SFT 1B", "Olmo SFT 7B", "RM Dataset", "Olmo SFT 1B", "Olmo SFT 7B"]
    colors = [0, 1, 2, 0, 1, 2]
    group_labels = ["5 Examples", "10 Examples"]
    
    print("JSON Analysis Script with Grouping")
    print("="*50)
    print("This script will analyze JSON files with grouped bar charts")
    print()
    
    # Example 1: With all grouping options
    print("Running analysis with grouping...")
    analyze_json_files(file_patterns, groups=groups, labels=labels, 
                      colors=colors, group_labels=group_labels)
    
    # Example 2: Simple usage (backward compatible)
    # analyze_json_files(file_patterns)
    
    # Example 3: Only with groups
    # analyze_json_files(file_patterns, groups=groups)
    
    # Example 4: Custom labels only
    # analyze_json_files(file_patterns, labels=labels)