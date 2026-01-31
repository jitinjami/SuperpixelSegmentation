import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

plots_dir = "../files/plots"
os.makedirs(plots_dir, exist_ok=True)

def load_results(base_dir="../files/Results"):
    data = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found!")
        return pd.DataFrame()

    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path) or dataset.startswith('.'):
            continue
            
        for method in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method)
            if not os.path.isdir(method_path) or method.startswith('.'):
                continue
            
            for num_pixels in os.listdir(method_path):
                num_pixels_path = os.path.join(method_path, num_pixels)
                summary_path = os.path.join(num_pixels_path, "summary_bootstrap.json")
                
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            stats = json.load(f)
                        
                        entry = {
                            'dataset': dataset,
                            'method': method,
                            'pixels_per_superpixel': int(num_pixels)
                        }
                        
                        data_source = stats.get('aggregated_stats', {})
                        
                        for metric in ['boundary_recall', 'undersegmentation_error', 'achievable_segmentation_accuracy', 'class_homogeneity']:
                            if metric in data_source and data_source[metric]:
                                entry[metric] = data_source[metric]['mean']
                                
                        data.append(entry)
                    except Exception as e:
                        print(f"Error reading {summary_path}: {e}")

    return pd.DataFrame(data)

def rank_methods(group):
    group['rank_BR'] = group['boundary_recall'].rank(ascending=False)
    group['rank_ASA'] = group['achievable_segmentation_accuracy'].rank(ascending=False)
    group['rank_CH'] = group['class_homogeneity'].rank(ascending=False)
    group['rank_UE'] = group['undersegmentation_error'].rank(ascending=True) 
    
    group['avg_rank'] = group[['rank_BR', 'rank_ASA', 'rank_CH', 'rank_UE']].mean(axis=1)
    return group

method_colors = {
    'edtrs': '#9B59B6',    # Purple
    'slic': '#E74C3C',      # Red
    'watershed': '#3498DB',     # Blue  
    'felzenszwalb': '#2ECC71',  # Green
    'quickshift': '#F39C12'        # Orange
}

method_styles = {
    'edtrs': {'marker': 'v', 'linestyle': '-', 'markersize': 8},      
    'slic': {'marker': 'o', 'linestyle': '-', 'markersize': 8},
    'watershed': {'marker': 's', 'linestyle': '--', 'markersize': 8},
    'felzenszwalb': {'marker': '', 'linestyle': '-.', 'markersize': 8},
    'quickshift': {'marker': '', 'linestyle': ':', 'markersize': 8}
}

target_xticks = [275, 550, 825, 1100]

def plot_rank_stability():
    print("Generating Rank Stability Plot...")
    df = load_results()
    
    if df.empty:
        print("No data found!")
        return

    all_ranked_df = df.groupby(['dataset', 'pixels_per_superpixel'], group_keys=False).apply(rank_methods)

    datasets = ['azh', 'wsnet', 'cwdb', 'cicatrix']
    metrics_rank_map = {
        'rank_BR': 'Boundary Recall',
        'rank_CH': 'Class Homogeneity',
        'rank_UE': 'Undersegmentation Error',
        'rank_ASA': 'Achievable Segmentation Accuracy',
    }
    metrics_cols = list(metrics_rank_map.keys())

    width_inch = 7 
    height_inch = width_inch * 0.8 
    
    fig, axes = plt.subplots(4, 4, figsize=(width_inch, height_inch), sharex=True, sharey=True)
    
    legend_handles = []
    legend_labels = []

    for row_idx, dataset in enumerate(datasets):
        ds_data = all_ranked_df[all_ranked_df['dataset'] == dataset]
        
        for col_idx, (rank_col, metric_name) in enumerate(metrics_rank_map.items()):
            ax = axes[row_idx, col_idx]
            
            methods_in_ds = ds_data['method'].unique()
            desired_order = ['edtrs', 'slic', 'watershed', 'felzenszwalb', 'quickshift']
            
            for method in desired_order:
                if method not in methods_in_ds:
                    continue
                    
                method_data = ds_data[ds_data['method'] == method].sort_values('pixels_per_superpixel')
                
                if not method_data.empty:
                    line, = ax.plot(
                        method_data['pixels_per_superpixel'], 
                        method_data[rank_col],
                        label=method.upper(),
                        color=method_colors.get(method, 'gray'),
                        marker=method_styles.get(method, {}).get('marker', 'o'),
                        linestyle=method_styles.get(method, {}).get('linestyle', '-'),
                        linewidth=1, 
                        markersize=3   
                    )
                    
                    if row_idx == 0 and col_idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(method.upper())

            ax.invert_yaxis()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if col_idx == 0:
                ax.set_ylabel(f"{dataset.upper()}\nRank", fontsize=7, fontweight='bold')
            
            if row_idx == 0:
                ax.set_title(metric_name, fontsize=7, fontweight='bold')
                
            ax.set_xticks(target_xticks)
            
            if row_idx == 3:
                ax.set_xlabel("Pixels/Superpixel", fontsize=7, fontweight='bold')
            
            max_rank = len(desired_order) # approx 5
            ax.set_yticks(range(1, max_rank + 1))
            
            ax.set_ylim(max_rank + 0.75, 0.25)
            ax.set_xlim(225, 1150)

    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(legend_labels), fontsize=7, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.subplots_adjust(wspace=0.1)
    
    output_path = os.path.join(plots_dir, "rank_stability_grid.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_rank_stability()
