import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from src.utils import setup_logger

def get_class_colors(dataset):
    """
    Define class colors for each dataset.
    Returns the mask dictionary with RGB tuples for each class.
    These will be displayed with their original colors in the overlay.
    """
    if dataset == "cwdb":
        mask_dict = {
            'class 1': (0, 0, 0), 
            'class 2': (163, 35, 142), 
            'class 3': (237, 28, 36), 
            'class 4': (253, 231, 36), 
            'class 5': (255, 242, 0), 
            'class 6': (255, 255, 255)  # Background
        }
    elif dataset == "wsnet" or dataset == "azh":
        mask_dict = {
            'class 1': (0, 0, 0), 
            'class 2': (255, 255, 255)  # Wound
        }
    elif dataset == "cicatrix":
        mask_dict = {
            'class 1': (255, 255, 255),  # Background - White
            'class 2': (12, 80, 155),     # Flat Wound Border
            'class 3': (228, 183, 229),   # Punched Out Wound Border
            'class 4': (238, 83, 0),      # Granulation
            'class 5': (0, 157, 99),      # Slough
            'class 6': (0, 0, 0)          # Necrosis
        }
    else:
        mask_dict = {
            'class 1': (0, 0, 0),
            'class 2': (255, 255, 255)
        }
    
    return mask_dict


def visualize_separate_images(logger, base_dirs, dataset, methods, num_pixels,
                               mask_alpha=0.4,
                               spixel_boundary_color=(255, 255, 0), 
                               gt_boundary_color=(0, 255, 0)):
    """
    Create separate visualizations for each image and method.
    
    Saves:
    1. original_gt_overlay.jpg - Original + GT boundaries + colored mask overlay
    2. original_gt_boundaries.jpg - Original + GT boundaries only
    3. {method}_overlay_boundaries.jpg - Original + GT overlay + GT boundaries + superpixel boundaries (for each method)
    
    Args:
        logger: Logger object
        base_dirs: Dictionary with 'dataset_dir' and 'ground_truth_dir'
        dataset: Dataset name
        methods: List of superpixel methods
        num_pixels: Number of superpixels
        mask_alpha: Transparency for mask overlay
        spixel_boundary_color: RGB color for superpixel boundaries
        gt_boundary_color: RGB color for ground truth boundaries
    """
    logger.info(f"\n{'='*50}")
    logger.info("CREATING SEPARATE VISUALIZATIONS")
    logger.info(f"{'='*50}")
    
    class_colors = get_class_colors(dataset)
    
    output_dir = os.path.join(base_dirs['output_dir'], f'separate_{num_pixels}')
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset == 'cicatrix':
        img_ext, mask_ext = '.JPG', '.png'
    elif dataset == "wsnet":
        img_ext, mask_ext = '.jpg', '.jpg'
    elif dataset in ['cwdb', 'azh']:
        img_ext, mask_ext = '.png', '.png'
    else:
        img_ext, mask_ext = '.png', '.png'
    
    image_files = sorted([f for f in os.listdir(base_dirs['dataset_dir']) if f.endswith(img_ext)])
    
    for i, img_file in enumerate(image_files):
        logger.info(f"Processing {img_file} ({i+1}/{len(image_files)})")
        
        try:
            base_name = os.path.splitext(img_file)[0]
            
            img_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            image_path = os.path.join(base_dirs['dataset_dir'], img_file)
            img_color = cv2.imread(image_path)
            if img_color is None:
                logger.error(f"Could not load image: {image_path}")
                continue
            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            
            mask_file = base_name + mask_ext
            mask_path = os.path.join(base_dirs['ground_truth_dir'], mask_file)
            mask = cv2.imread(mask_path)
            
            gt_boundaries = None
            mask_rgb = None
            if mask is not None:
                if len(mask.shape) == 2:
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                else:
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                
                h, w = mask_rgb.shape[:2]
                pixels = mask_rgb.reshape(-1, 3)
                unique_colors, labels = np.unique(pixels, axis=0, return_inverse=True)
                label_map = labels.reshape(h, w)
                
                gt_boundaries = find_boundaries(label_map, mode='thick')
            else:
                logger.warning(f"Could not load mask: {mask_path}, skipping this image")
                continue

            img_with_overlay = img_rgb.astype(np.float32).copy()
            for class_name, class_color in class_colors.items():
                class_mask = np.all(mask_rgb == class_color, axis=2)
                if np.any(class_mask):
                    overlay = np.zeros_like(img_with_overlay)
                    overlay[class_mask] = class_color
                    mask_region = class_mask[:, :, np.newaxis]
                    img_with_overlay = np.where(mask_region,
                                          (1 - mask_alpha) * img_with_overlay + mask_alpha * overlay,
                                          img_with_overlay)
            img_with_overlay = np.clip(img_with_overlay, 0, 255).astype(np.uint8)
            
            try:
                vis_overlay = img_with_overlay.copy()
                vis_overlay[gt_boundaries] = gt_boundary_color
                
                save_path = os.path.join(img_output_dir, 'original_gt_overlay.jpg')
                plt.imsave(save_path, vis_overlay, format='jpg', dpi=150)
                logger.info(f"  Saved: original_gt_overlay.jpg")
            except Exception as e:
                logger.error(f"Error creating GT overlay for {img_file}: {str(e)}")
            
            try:
                vis_gt_only = img_rgb.copy()
                vis_gt_only[gt_boundaries] = gt_boundary_color
                
                save_path = os.path.join(img_output_dir, 'original_gt_boundaries.jpg')
                plt.imsave(save_path, vis_gt_only, format='jpg', dpi=150)
                logger.info(f"  Saved: original_gt_boundaries.jpg")
            except Exception as e:
                logger.error(f"Error creating GT boundaries only for {img_file}: {str(e)}")
            
            for method in methods:
                try:
                    spixel_file = base_name + '.npy'
                    spixel_path = os.path.join(
                        base_dirs['file_directory'], 
                        f'Results/{dataset}/{method}/{num_pixels}/spixels',
                        spixel_file
                    )
                    
                    if not os.path.exists(spixel_path):
                        logger.warning(f"  Superpixel file not found for {method}: {spixel_path}")
                        continue
                    
                    label = np.load(spixel_path)
                    
                    vis_method = img_with_overlay.copy()
                    
                    vis_method[gt_boundaries] = gt_boundary_color
                    
                    spixel_boundaries = find_boundaries(label, mode='thick')
                    vis_method[spixel_boundaries] = spixel_boundary_color
                    
                    save_path = os.path.join(img_output_dir, f'{method}_overlay_boundaries.jpg')
                    plt.imsave(save_path, vis_method, format='jpg', dpi=150)
                    logger.info(f"  Saved: {method}_overlay_boundaries.jpg")
                    
                except Exception as e:
                    logger.error(f"  Error processing method {method} for {img_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing image {img_file}: {str(e)}")
            continue
    
    logger.info(f"Completed separate visualizations for {len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description='Create separate visualization images')
    parser.add_argument('--dataset', '-d', type=str, 
                       choices=['cicatrix', 'wsnet', 'cwdb', 'azh'], 
                       default='cwdb')
    parser.add_argument('--num_pixel', '-n', type=int, 
                       choices=[275, 550, 825, 1100], 
                       default=0)
    parser.add_argument('--file_directory', '-fd', type=str, default=".")
    parser.add_argument('--mask_alpha', '-ma', type=float, default=0.2,
                       help='Transparency for mask overlay (0.0-1.0)')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    num_pixels = args.num_pixel
    file_directory = args.file_directory
    mask_alpha = args.mask_alpha
    
    methods = ['edtrs', 'slic', 'felzenszwalb', 'watershed', 'quickshift']
    
    log_dir = f'{file_directory}/Logs/'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir=log_dir, 
                         log_file=f'visualization_separate_{dataset}_{num_pixels}.log')
    
    base_dirs = {
        'dataset_dir': f'{file_directory}/Data/{dataset}/images',
        'ground_truth_dir': f'{file_directory}/Data/{dataset}/rgb_masks',
        'file_directory': file_directory,
        'output_dir': f'{file_directory}/Visualizations/{dataset}'
    }
    
    for key, path in base_dirs.items():
        if key != 'output_dir' and not os.path.exists(path):
            logger.error(f"Directory not found: {path}")
            return
    
    logger.info(f"Starting separate visualizations for {dataset}")
    logger.info(f"Number of superpixels: {num_pixels}")
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Mask overlay alpha: {mask_alpha}")
    
    visualize_separate_images(logger, base_dirs, dataset, methods, num_pixels, mask_alpha)
    
    logger.info("Visualization complete!")
    logger.info(f"Output directory: {base_dirs['output_dir']}/separate_{num_pixels}")


if __name__ == '__main__':
    main()