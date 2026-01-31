import os
import argparse
import numpy as np
import json
import random

from src.utils import generate_superpixels, evaluate_superpixels, get_mask_dict, setup_logger

def main():
    parser = argparse.ArgumentParser(description='Superpixel Segmentation')
    parser.add_argument('--dataset', '-d', type=str, choices=['cicatrix', 'wsnet', 'cwdb', 'azh', 'medetec'], required=True)
    parser.add_argument('--method', '-m', type=str, choices=['edtrs', 'slic', 'felzenszwalb', 'quickshift', 'watershed'], required=True)
    parser.add_argument('--num_pixel_per_spixel', '-n', type=int, choices=[275, 550, 825, 1100], default=550)
    parser.add_argument('--file_directory', '-fd', type=str, default=".")
    parser.add_argument('--save_files', '-sf', action='store_true')
    args = parser.parse_args()
    
    dataset = args.dataset
    num_pixel_per_spixel = args.num_pixel_per_spixel
    method = args.method
    save_files = args.save_files
    file_directory = args.file_directory

    random.seed(0)
    np.random.seed(0)

    log_dir = f'{file_directory}/Logs/'
    logger = setup_logger(log_dir=log_dir, log_file=f'{dataset}_{method}_{num_pixel_per_spixel}.log')

    dirs = {}
    dirs['dataset_dir'] = f'{file_directory}/Data/{dataset}/images'
    dirs['ground_truth_dir'] = f'{file_directory}/Data/{dataset}/rgb_masks'
    dirs['config_results_dir'] = f'{file_directory}/Results/{dataset}/{method}/{num_pixel_per_spixel}'

    os.makedirs(dirs['config_results_dir'], exist_ok=True) # Create results directory

    logger.info(f"Starting superpixel comparison experiments for {dataset}")
    logger.info(f"Target no. of pixels per superpixel: {num_pixel_per_spixel}")

    quickshift_params = {}
    master_params_path = os.path.join('config', 'master_quickshift_params.json')
    if method == 'quickshift':
        if os.path.exists(master_params_path):
            with open(master_params_path, 'r') as f:
                master_params = json.load(f)
                num_str = str(num_pixel_per_spixel)
                if dataset in master_params and num_str in master_params[dataset]:
                    quickshift_params = master_params[dataset][num_str]
                    logger.info(f"Loaded Quickshift params for {dataset}/{num_str}: {quickshift_params}")
                else:
                    logger.warning(f"No custom params found for {dataset}/{num_str}, using defaults.")
        else:
            logger.warning(f"Master params file not found at {master_params_path}")

    mask_colors = get_mask_dict(dataset)

    generate_superpixels(num_pixel_per_spixel, logger, dirs, method, save_files, **quickshift_params)
        
    final_stats = evaluate_superpixels(logger, dirs, dataset, mask_colors)

    summary_path = os.path.join(dirs['config_results_dir'], 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_stats, f, indent=2)

if __name__ == '__main__':
    main()