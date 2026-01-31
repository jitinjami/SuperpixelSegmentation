import os
import cv2
import numpy as np
from skimage.segmentation import quickshift
from src.utils import save_labels

def generate_superpixels_quickshift(kernel_size, max_dist, logger, dirs, save_files):
    """Run Quickshift with specific parameters"""
    logger.info(f"\n{'='*50}")
    logger.info(f"RUNNING QUICKSHIFT: kernel_size={kernel_size}, max_dist={max_dist}")
    logger.info(f"{'='*50}")

    dirs['spixels_path'] = os.path.join(dirs['config_results_dir'], 'spixels')
    os.makedirs(dirs['spixels_path'], exist_ok=True)
    
    filenames = sorted(os.listdir(dirs['dataset_dir']))

    for i, filename in enumerate(filenames):
        try: 
            save_path = os.path.join(dirs['spixels_path'], os.path.splitext(filename)[0]+'.npy')
            if os.path.isfile(save_path):
                logger.info("Superpixels already exist for this image")
                continue

            logger.info(f"Processing {filename} ({i+1}/{len(filenames)})")

            image_path = os.path.join(dirs['dataset_dir'], filename)
            img_color = cv2.imread(image_path)
            
            if img_color is None:
                logger.warning(f"[SKIP] Failed to load image: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            labels = quickshift(img_rgb, kernel_size=kernel_size, max_dist=max_dist, ratio=1.0)
            
            num_superpixels = len(np.unique(labels))
            logger.info(f"Generated {num_superpixels} superpixels")

            if save_files:
                save_labels(dirs, filename, labels)
        except Exception as e:
            logger.error(f"Processing failed for {filename}: {str(e)}")
