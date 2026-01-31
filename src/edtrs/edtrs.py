import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


from src.edtrs.superpixel_generator import Superpixel
from src.edtrs.edge_detection_processor import EdgeDetectionProcessor
from src.edtrs.texture_feature_processor import TextureFeatureProcessor

def edtrs(logger, img_color, image_gray, num_pixels):
    # Enhance
    img_color_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_color_lab[:, :, 0] = clahe.apply(img_color_lab[:, :, 0])
    img_color = cv2.cvtColor(img_color_lab, cv2.COLOR_LAB2BGR)

    # Edge Detection
    edge_processor = EdgeDetectionProcessor(img_color, logger)
    edge_processor.process_edges()
    edge_gray = cv2.cvtColor(edge_processor.final_edge_image, cv2.COLOR_RGB2GRAY).astype(float) / 255.0

    # Texture Features
    texture_processor = TextureFeatureProcessor(image_gray, patch_size=12, L=11, logger=logger)
    texture_processor.compute_texture_features()
    texture_processor.feature_matrix = gaussian_filter(texture_processor.feature_matrix, sigma=(1, 1, 0), order=0)
    reduced_features = texture_processor.apply_pca()

    h, w = img_color.shape[:2]
    texture_resized = np.zeros((h, w, 2))
    for c in range(2):
        texture_resized[:, :, c] = cv2.resize(reduced_features[:, :, c], (w, h), interpolation=cv2.INTER_CUBIC)

    # Generate superpixels
    superpixel_generator = Superpixel(edge_gray, texture_resized, img_color.shape, num_pixels, image_gray, logger)
    edtrs_labels = superpixel_generator.generate_superpixels(max_iterations=15, tolerance=1e-4)

    return edtrs_labels, superpixel_generator