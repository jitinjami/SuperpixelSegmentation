import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import joblib
from joblib import Parallel, delayed

from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray

from src.edtrs.edtrs import edtrs
from src.performance_measures import PerformanceMeasures

def setup_logger(log_dir='logs', log_file='superpixel_job.log'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(log_dir, f"{timestamp}_{log_file}")

    logging.basicConfig(
        filename=full_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('SuperpixelLogger')

def get_mask_dict(dataset):
    if dataset == "cwdb":
        mask_dict = {'class 1': (0, 0, 0), 
                     'class 2': (163, 35, 142), 
                     'class 3': (237, 28, 36), 
                     'class 4': (253, 231, 36), 
                     'class 5': (255, 242, 0), 
                     'class 6': (255, 255, 255)} # Background
    elif dataset == "wsnet" or dataset == "azh":
        mask_dict = {'class 1': (0, 0, 0), 
                     'class 2': (255, 255, 255)} # Wound
    elif dataset == "cicatrix":
        mask_dict = {'class 1': (255, 255, 255),  #Background - White
                      'class 2': (12, 80 ,155), #Flat Wound Border
                      'class 3': (228, 183 ,229), #Punched Out Wound Border
                      'class 4': (238, 83 ,0), #Granulation
                      'class 5': (0,157,99), #Slough
                      'class 6': (0, 0, 0)} #Necrosis
        
    return mask_dict

def get_mask_colors(logger, dirs, dataset):
    logger.info("Getting Classes in RGB to build mask_dict")
    all_unique_colors = set()

    for mask_filename in os.listdir(dirs['ground_truth_dir']):
        if ".DS_Store" in mask_filename:
            continue
        mask_filepath = os.path.join(dirs['ground_truth_dir'], mask_filename)
        mask = cv2.imread(mask_filepath)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        pixels = mask_rgb.reshape(-1, 3)
        if dataset == "wsnet" or dataset == "azh":
            pixels = np.where(pixels > 128, 255, 0)
        
        unique_colors_in_image = set(tuple(int(x) for x in pixel) for pixel in pixels) 

        all_unique_colors.update(unique_colors_in_image)

    unique_colors_list = sorted(list(all_unique_colors))
    
    mask_colors = {}
    for i, color in enumerate(unique_colors_list, 1):
        mask_colors[f"class {i}"] = color

    return mask_colors

def save_labels(dirs, filename, labels):
    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(dirs['spixels_path'], base_name)

    np.save(save_path, labels)

def _process_single_image_generation(dirs, filename, num_pixel_per_spixel, method, save_files, kwargs):
    """Helper to process a single image for generation."""
    
    save_path = os.path.join(dirs['spixels_path'], os.path.splitext(filename)[0]+'.npy')
    if os.path.isfile(save_path):
        return f"Skipped {filename} (Already exists)"

    image_path = os.path.join(dirs['dataset_dir'], filename)
    img_color = cv2.imread(image_path)
    
    if img_color is None:
        return f"[SKIP] Failed to load image: {image_path}"

    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    image_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) 

    if img_color is None or image_gray is None:
         return f"[SKIP] Failed to load/convert image: {image_path}"

    num_pixels_in_image = (img_rgb.shape[0] * img_rgb.shape[1])
    calculated_num_superpixels = int(num_pixels_in_image / num_pixel_per_spixel)

    if calculated_num_superpixels <= 1:
        return f"Skipped {filename} (Image too small)"

    try:
        class DummyLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
        
        local_logger = DummyLogger()

        if method == 'edtrs':
            labels, _ = edtrs(local_logger, img_rgb, image_gray, calculated_num_superpixels)
        elif method == 'slic':
            labels = slic(img_rgb, n_segments=calculated_num_superpixels, start_label=1)
        elif method == 'felzenszwalb':
            labels = felzenszwalb(img_rgb, scale=1, min_size=int(num_pixel_per_spixel/5))
        elif method == 'quickshift':
            qs_kernel_size = kwargs.get('kernel_size', 5)
            qs_max_dist = kwargs.get('max_dist', 10)
            labels = quickshift(img_rgb, kernel_size=qs_kernel_size, max_dist=qs_max_dist, ratio=1.0)
        elif method == 'watershed':
            labels = watershed(sobel(rgb2gray(img_rgb)), markers=calculated_num_superpixels)

        if save_files:
            save_labels(dirs, filename, labels)
            
        return f"Processed {filename}"
        
    except Exception as e:
        return f"Failed {filename}: {str(e)}"

def generate_superpixels(num_pixel_per_spixel, logger, dirs, method, save_files, **kwargs):
    """Run for specific number of segments"""
    logger.info(f"\n{'='*50}")
    logger.info(f"RUNNING FOR {num_pixel_per_spixel} PIXELS PER SUPERPIXELS (PARALLEL)")
    logger.info(f"{'='*50}")

    dirs['spixels_path'] = os.path.join(dirs['config_results_dir'], 'spixels')
    os.makedirs(dirs['spixels_path'], exist_ok=True)
    
    filenames = sorted(os.listdir(dirs['dataset_dir']))

    n_jobs = int(os.environ.get('N_JOBS', -1))
    effective_jobs = n_jobs if n_jobs != -1 else joblib.cpu_count()
    
    logger.info(f"Starting parallel processing for {len(filenames)} images using {effective_jobs} threads...")
    
    results = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(_process_single_image_generation)(
            dirs, filename, num_pixel_per_spixel, method, save_files, kwargs
        ) for filename in filenames
    )
    
    for res in results:
        if "Skipped" in res:
            logger.info(res)
        elif "Failed" in res or "SKIP" in res:
             logger.warning(res)
            
    logger.info("Parallel generation completed.")

def visualize_superpixels(logger, dirs, dataset):
    logger.info(f"\n{'='*50}")
    logger.info("VISULALIZING FOR SUPERPIXELS")
    logger.info(f"{'='*50}")

    dirs['viz_path'] = os.path.join(dirs['config_results_dir'], 'viz')
    os.makedirs(dirs['viz_path'], exist_ok=True)

    if len(os.listdir(dirs['viz_path'])) > 0:
        logger.info("Superpixels visualizations already exist for this configuration")
        return None
    
    for i, superpixel_file in enumerate(sorted(os.listdir(dirs['viz_path']))):
        save_path = os.path.join(dirs['viz_path'], superpixel_file)
        logger.info(f"Processing {superpixel_file} ({i+1}/{len(os.listdir(dirs['spixels_path']))})")
        
        label = np.load(os.path.join(dirs['spixels_path'], superpixel_file))

        if dataset == 'cicatrix':
            img_filename = os.path.splitext(superpixel_file)[0] + '.JPG'
            mask_filename = os.path.splitext(superpixel_file)[0] + '.png'
        elif dataset == "wsnet":
            img_filename = os.path.splitext(superpixel_file)[0] + '.jpg'
            mask_filename = os.path.splitext(superpixel_file)[0] + '.jpg'
        elif dataset == 'cwdb' or dataset == "azh":
            img_filename = os.path.splitext(superpixel_file)[0] + '.png'
            mask_filename = os.path.splitext(superpixel_file)[0] + '.png'
    
        image_path = os.path.join(dirs['dataset_dir'], img_filename)
        img_color = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        vis_img = mark_boundaries(img_rgb, label)
        plt.imsave(save_path, vis_img, format='jpg', dpi=150)

def _process_single_image_evaluation(dirs, superpixel_file, dataset, mask_colors, performance):
    """Helper to evaluate a single image."""
    try:
        label = np.load(os.path.join(dirs['spixels_path'], superpixel_file))

        if dataset == 'cicatrix':
            img_filename = os.path.splitext(superpixel_file)[0] + '.JPG'
            mask_filename = os.path.splitext(superpixel_file)[0] + '.png'
        elif dataset == "wsnet":
            img_filename = os.path.splitext(superpixel_file)[0] + '.jpg'
            mask_filename = os.path.splitext(superpixel_file)[0] + '.jpg'
        elif dataset == 'cwdb' or dataset == "azh":
            img_filename = os.path.splitext(superpixel_file)[0] + '.png'
            mask_filename = os.path.splitext(superpixel_file)[0] + '.png'
    
        image_path = os.path.join(dirs['dataset_dir'], img_filename)
        img_color = cv2.imread(image_path)
        
        if img_color is None:
             return {'filename': superpixel_file, 'error': f"Failed to load image {image_path}"}
             
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        ground_truth_path = os.path.join(dirs['ground_truth_dir'], mask_filename)
        ground_truth = cv2.imread(ground_truth_path)
        
        if ground_truth is None:
             return {'filename': superpixel_file, 'error': f"Failed to load mask {ground_truth_path}"}

        if dataset == "wsnet" or dataset == "azh":
            ground_truth = np.where(ground_truth > 128, 255, 0)
            
        if dataset not in ["wsnet","azh"]:
            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

        if img_rgb.shape[0] != ground_truth.shape[0]:
            ground_truth = cv2.resize(ground_truth, (img_rgb.shape[0], img_rgb.shape[1]), interpolation=cv2.INTER_NEAREST)

        individual_eval_results = performance.evaluation(img_rgb, label, ground_truth, mask_colors)
        
        timing_str = individual_eval_results.pop('_timing_str', "")
        
        result_entry = {'filename': superpixel_file}
        result_entry.update(individual_eval_results)
        
        if timing_str:
            result_entry['log_message'] = f"Processed {superpixel_file} - {timing_str}"
            
        return result_entry

    except Exception as e:
        return {'filename': superpixel_file, 'error': str(e)}

def evaluate_superpixels(logger, dirs, dataset, mask_colors):
    """Run for specific number of segments"""
    logger.info(f"\n{'='*50}")
    logger.info("Evaluating FOR SUPERPIXELS (PARALLEL)")
    logger.info(f"{'='*50}")
    
    border_metrics = False

    if dataset == "cicatrix":
        border_metrics = True

    performance = PerformanceMeasures(tolerance_radius=3, border_metrics = border_metrics)

    eval_results = {}
    
    files = sorted(os.listdir(dirs['spixels_path']))
    n_jobs = int(os.environ.get('N_JOBS', -1))
    effective_jobs = n_jobs if n_jobs != -1 else joblib.cpu_count()
    
    logger.info(f"Starting parallel evaluation for {len(files)} files using {effective_jobs} threads...")
    
    results_generator = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(_process_single_image_evaluation)(
            dirs, f, dataset, mask_colors, performance
        ) for f in files
    )

    valid_results = []
    
    for res in results_generator:
        if 'error' in res:
            logger.warning(f"Error evaluating {res['filename']}: {res['error']}")
            continue
            
        if 'log_message' in res:
            logger.info(res['log_message'])
            del res['log_message'] # Clean up before saving to CSV
            
        valid_results.append(res)
        
        for key, value in res.items():
            if key == 'filename': continue
            if key not in eval_results:
                eval_results[key] = []
            eval_results[key].append(value)
    
    if valid_results:
        csv_path = os.path.join(dirs['config_results_dir'], 'results.csv')
        try:
            fieldnames = list(valid_results[0].keys())
            if 'filename' in fieldnames:
                fieldnames.remove('filename')
                fieldnames = ['filename'] + fieldnames
                
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_results)
            logger.info(f"Saved individual image metrics to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save results.csv: {e}")

    final_eval_stats = {key: {'mean': np.mean(values), 'std': np.std(values)} 
              for key, values in eval_results.items()}
    if final_eval_stats:
        logger.info(f"Final Eval Stats: {final_eval_stats}")
    
    return final_eval_stats
