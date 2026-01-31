import os
import pandas as pd
import numpy as np
import json
import random

def calculate_bootstraps(base_dir="../files/Results"):
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    datasets = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')])
    
    metrics = ['boundary_recall', 'undersegmentation_error', 'achievable_segmentation_accuracy', 'class_homogeneity']

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        dataset_path = os.path.join(base_dir, dataset)
        
        # 1. Discover all images for this dataset to define the master list
        # We'll look for the first valid results.csv we can find to establish the universe of files
        master_filenames = set()
        
        # Walk to find a results.csv
        for root, dirs, files in os.walk(dataset_path):
            if 'results.csv' in files:
                try:
                    df = pd.read_csv(os.path.join(root, 'results.csv'))
                    if 'filename' in df.columns:
                        master_filenames.update(df['filename'].tolist())
                        # We only need one good list, assuming all methods process the same dataset. 
                        # To be safe, let's keep it cumulative or just stop at the first big one?
                        # Usually datasets are static. Let's trust the first non-empty one.
                        if len(master_filenames) > 0:
                            break
                except Exception as e:
                    print(f"Warning reading {os.path.join(root, 'results.csv')}: {e}")
        
        master_filenames = sorted(list(master_filenames))
        num_images = len(master_filenames)
        if num_images == 0:
            print(f"No results found for dataset {dataset}, skipping.")
            continue
            
        print(f"Found {num_images} unique images in results.")

        # 2. Generate 100 consistent bootstrap masks (80% subsampling)
        # We store the *list of filenames* for each bootstrap iteration
        n_bootstraps = 100
        sample_size = int(0.8 * num_images)
        bootstrap_sets = []
        
        # Seed based on dataset name for reproducibility across runs
        # Use a deterministic seed derived from dataset name
        seed_base = sum(ord(c) for c in dataset) 
        
        for i in range(n_bootstraps):
            random.seed(seed_base + i) # Ensure consistency for "dataset + bootstrap_i"
            subset = random.sample(master_filenames, sample_size)
            bootstrap_sets.append(set(subset)) # Use set for O(1) lookup

        # 3. Process all experiments in this dataset
        for method in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method)
            if not os.path.isdir(method_path) or method.startswith('.'):
                continue
                
            for num_pixels in os.listdir(method_path):
                num_pixels_path = os.path.join(method_path, num_pixels)
                results_csv = os.path.join(num_pixels_path, "results.csv")
                
                if not os.path.exists(results_csv):
                    continue
                    
                try:
                    df = pd.read_csv(results_csv)
                    if df.empty or 'filename' not in df.columns:
                        continue
                    
                    # Prepare storage for this experiment
                    bootstrap_summary = {m: [] for m in metrics}
                    
                    # Calculate mean for each bootstrap
                    for i, allowed_files in enumerate(bootstrap_sets):
                        # Filter dataframe (Faster lookup)
                        filtered_df = df[df['filename'].apply(lambda x: x in allowed_files)]
                        
                        if filtered_df.empty:
                            print(f"Warning: Empty bootstrap {i} for {method}/{num_pixels}")
                            for m in metrics:
                                bootstrap_summary[m].append(None) 
                            continue

                        for metric in metrics:
                            if metric in filtered_df.columns:
                                mean_val = filtered_df[metric].mean()
                                bootstrap_summary[metric].append(mean_val)
                            else:
                                bootstrap_summary[metric].append(None)
                    
                    # Calculate Aggregated Stats
                    final_output = {
                        "individual_bootstraps": bootstrap_summary,
                        "aggregated_stats": {}
                    }
                    
                    for metric in metrics:
                        values = [v for v in bootstrap_summary[metric] if v is not None]
                        if values:
                            final_output["aggregated_stats"][metric] = {
                                "mean": np.mean(values),
                                "std": np.std(values)
                            }
                        else:
                            final_output["aggregated_stats"][metric] = {"mean": None, "std": None}

                    # Save summary_bootstrap.json
                    output_path = os.path.join(num_pixels_path, "summary_bootstrap.json")
                    
                    # Serialize
                    # Handle NaNs/Nones for JSON
                    class NpEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.integer):
                                return int(obj)
                            if isinstance(obj, np.floating):
                                return float(obj)
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return super(NpEncoder, self).default(obj)

                    with open(output_path, 'w') as f:
                        json.dump(final_output, f, indent=4, cls=NpEncoder)
                        
                    print(f"Generated bootstrap summary for {dataset}/{method}/{num_pixels}")
                    
                except Exception as e:
                    print(f"Error processing {results_csv}: {e}")

if __name__ == "__main__":
    calculate_bootstraps()
