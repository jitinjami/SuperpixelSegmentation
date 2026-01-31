import numpy as np
import cv2
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from skimage import measure
from skimage.segmentation import find_boundaries

class PerformanceMeasures:
    """
    Comprehensive superpixel evaluation metrics following the methodology from:
    "Superpixels: An Evaluation of the State-of-the-Art" by Stutz et al.
    
    Implements the standard metrics used in superpixel literature:
    - Boundary Recall (Rec)
    - Undersegmentation Error (UE) 
    - Explained Variation (EV)
    - Compactness (CO)
    - Achievable Segmentation Accuracy (ASA)
    - Additional medical imaging specific metrics
    """
    
    def __init__(self, tolerance_radius=2, border_metrics = False):
        """
        Initialize the metrics calculator.
        
        Args:
            tolerance_radius: Radius for boundary matching tolerance (typically 0.0025 * image_diagonal)
        """
        self.tolerance_radius = tolerance_radius
        self.border_metrics = border_metrics
    
    def _rgb_to_labels(self, rgb_image, tissue_colors):
        """Convert RGB ground truth to label format."""
        labels = np.zeros(rgb_image.shape[:2], dtype=np.int32)
        border_only_labels = np.zeros(rgb_image.shape[:2], dtype=np.int32)

        self.border_gt_class = None
        
        for i, (tissue_name, color) in enumerate(tissue_colors.items()):
            color_array = np.array(color)
            mask = np.all(rgb_image == color_array, axis=2)
            labels[mask] = i

            if self.border_metrics:
                if tissue_name == "class 2" or tissue_name == "class 3":
                    border_only_labels[mask] = 1
            
        return labels, border_only_labels

    def _get_contingency_matrix(self, superpixel_labels, ground_truth_labels):
        sp_flat = superpixel_labels.ravel()
        gt_flat = ground_truth_labels.ravel()
        
        n_sp = sp_flat.max() + 1
        n_gt = gt_flat.max() + 1
        
        contingency, _, _ = np.histogram2d(
            sp_flat, gt_flat, 
            bins=(n_sp, n_gt), 
            range=[[0, n_sp], [0, n_gt]]
        )
        return contingency

    def boundary_recall(self, superpixel_labels, ground_truth):
        """
        Compute Boundary Recall (Rec) - measures boundary adherence.
        Uses dilation-based tolerance (more practical than distance transform).
        
        Rec = |SP_boundaries ∩ Dilated_GT_boundaries| / |GT_boundaries|
        
        Args:
            superpixel_labels: 2D array of superpixel labels
            ground_truth: 2D array of ground truth segmentation
            
        Returns:
            float: Boundary recall score (higher is better)
        """
        sp_boundaries = find_boundaries(superpixel_labels, mode='thick')
        gt_boundaries = find_boundaries(ground_truth, mode='thick')
            
        kernel = np.ones((2*self.tolerance_radius+1, 2*self.tolerance_radius+1), np.uint8)
        dilated_gt_boundaries = cv2.dilate(gt_boundaries.astype(np.uint8), kernel, iterations=1)
        
        true_positive_edges = np.sum(sp_boundaries & dilated_gt_boundaries.astype(bool))
        total_gt_edges = np.sum(gt_boundaries)
        
        if total_gt_edges == 0:
            return 1.0
            
        return true_positive_edges / total_gt_edges
    
    def calculate_asa_fast(self, contingency, superpixel_labels_size):
        max_overlaps = np.max(contingency, axis=1)
    
        return np.sum(max_overlaps) / superpixel_labels_size

    def calculate_ue_fast(self, contingency, superpixel_labels_size):
    
        sp_sizes = np.sum(contingency, axis=1, keepdims=True) # Column vector
        differences = sp_sizes - contingency
    
        error_matrix = np.minimum(contingency, differences)
    
        return np.sum(error_matrix) / superpixel_labels_size
    
    def calculate_ch_fast(self, contingency):

        sp_sizes = np.sum(contingency, axis=1)
        existing_sp_mask = sp_sizes > 0
        
        active_contingency = contingency[existing_sp_mask]

        unique_class_counts = np.count_nonzero(active_contingency, axis=1)

        pure_superpixels = np.sum(unique_class_counts == 1)
        total_superpixels = active_contingency.shape[0]

        return pure_superpixels / total_superpixels if total_superpixels > 0 else 0.0
   
    def evaluation(self, image, superpixel_labels, ground_truth, tissue_colors=None):
        """
        Perform comprehensive evaluation of superpixel segmentation.
        
        Args:
            image: Original image
            superpixel_labels: 2D array of superpixel labels
            ground_truth: Ground truth segmentation (2D labels or 3D RGB)
            tissue_colors: Optional dict for medical imaging evaluation
            
        Returns:
            dict: Complete evaluation metrics
        """
        metrics = {}
        
        if len(ground_truth.shape) == 3 and tissue_colors:
            gt_labels, _ = self._rgb_to_labels(ground_truth, tissue_colors)
        else:
            gt_labels = ground_truth
        
        if len(superpixel_labels.shape) == 3:
            superpixel_labels = np.squeeze(superpixel_labels, axis = 0)

        contingency = self._get_contingency_matrix(superpixel_labels, gt_labels)
        
        t0 = time.time()
        metrics['boundary_recall'] = self.boundary_recall(superpixel_labels, gt_labels)
        t_br = time.time() - t0

        t0 = time.time()
        metrics['class_homogeneity'] = self.calculate_ch_fast(contingency)
        t_ch = time.time() - t0

        t0 = time.time()
        metrics['undersegmentation_error'] = self.calculate_ue_fast(contingency, superpixel_labels.size)
        t_ue = time.time() - t0

        t0 = time.time()
        metrics['achievable_segmentation_accuracy'] = self.calculate_asa_fast(contingency, superpixel_labels.size)
        t_asa = time.time() - t0

        metrics['num_superpixels'] = len(np.unique(superpixel_labels))
        metrics['mean_num_pixels_per_superpixels'] = np.mean(np.unique(superpixel_labels, return_counts=True)[1])
        
        timing_str = (f"BR: {t_br:.4f}s, CH: {t_ch:.4f}s, UE: {t_ue:.4f}s, ASA: {t_asa:.4f}s")
        metrics['_timing_str'] = timing_str
        
        return metrics
    
    
