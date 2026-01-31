import numpy as np
import cv2
import scipy.ndimage

class EdgeDetectionProcessor:
    def __init__(self, image, logger=None):
        """
        Initialize with an image.
        """
        self.logger = logger

        self.image = image
        self.edge_intensity = None
        self.sobel_edges = None
        self.canny_edges = None
        self.combined_edges = None
        self.entropy_map = None
        self.K_grid = None
        self.final_edge_image = None
        

    def compute_sigma(self, region):
        """Compute adaptive sigma with variance calculation."""
        try:
            variance = np.percentile(region, 75) - np.percentile(region, 25)
            sigma = 1.0 / (10 * variance + 1e-6)
            return np.clip(sigma, 0.1, 2.0)
        except Exception as e:
            self.logger.error(f"Error in compute_sigma: {e}")
            return 1.0

    def compute_2D_entropy(self, edge_intensity, window_size=9):
        """
        Compute 2D entropy using a sliding window.
        """
        try:
            h, w, c = edge_intensity.shape
            entropy_map = np.zeros((h, w, 3))
            half_window = window_size // 2
            padded = np.pad(edge_intensity, ((half_window, half_window), (half_window, half_window), (0, 0)), mode='reflect')

            for ch in range(c):
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i + window_size, j:j + window_size, ch]
                        hist, _ = np.histogram(window, bins=32, density=True)
                        entropy_map[i, j, ch] = -np.sum(hist * np.log2(hist + 1e-10))

            entropy_map = (entropy_map - np.min(entropy_map)) / (np.max(entropy_map) - np.min(entropy_map))
            return entropy_map
        except Exception as e:
            self.logger.error(f"Error in compute_2D_entropy: {e}")
            return np.zeros_like(edge_intensity)

    def compute_K_grid(self):
        """Compute K* grid per channel."""
        try:
            mean_entropy = np.mean(self.entropy_map, axis=(0, 1))
            self.K_grid = np.arctan((self.entropy_map - mean_entropy) / 10) + 2
            return self.K_grid
        except Exception as e:
            self.logger.error(f"Error in compute_K_grid: {e}")
            return None

    def apply_sobel_operator(self):
        """
        Apply Sobel operator to each RGB channel separately.
        """
        try:
            self.sobel_edges = np.zeros_like(self.edge_intensity, dtype=np.float32)

            for c in range(3):
                channel = self.edge_intensity[:, :, c]
                smoothed = cv2.GaussianBlur(channel, (7, 7), 2.0)
                sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobel_x**2 + sobel_y**2)
                sobel = (sobel - np.min(sobel)) / (np.max(sobel))
                self.sobel_edges[:, :, c] = sobel
            
            return self.sobel_edges
        except Exception as e:
            self.logger.error(f"Error in apply_sobel_operator: {e}")
            return None

    def apply_canny_operator(self):
        """
        Apply Canny edge detection.
        """
        try:
            self.canny_edges = np.zeros_like(self.edge_intensity, dtype=np.float32)

            for c in range(3):
                channel = np.uint8(self.edge_intensity[:, :, c] * 255)
                edges = cv2.Canny(channel, threshold1=100, threshold2=200)
                self.canny_edges[:, :, c] = edges / 255.0
            
            return self.canny_edges
        except Exception as e:
            self.logger.error(f"Error in apply_canny_operator: {e}")
            return None

    def compute_virtual_pixel_block(self):
        """
        Compute edge intensity using a Gaussian filter.
        """
        try:
            height, width, _ = self.image.shape
            self.edge_intensity = np.zeros((height, width, 3))
            
            for c in range(3):
                channel = self.image[:, :, c].astype(float)
                local_std = np.std(scipy.ndimage.uniform_filter(channel, size=11), axis=(0, 1))
                sigma = np.clip(1.0 / (10 * local_std + 1e-6), 0.1, 2.0)
                self.edge_intensity[:, :, c] = scipy.ndimage.gaussian_filter(channel, sigma)
            
            self.edge_intensity = (self.edge_intensity - np.min(self.edge_intensity)) / (np.max(self.edge_intensity) - np.min(self.edge_intensity) + 1e-10)
            return self.edge_intensity
        except Exception as e:
            self.logger.error(f"Error in compute_virtual_pixel_block: {e}")
            return None

    def generate_final_edge_image(self):
        """
        Generate final RGB edge image.
        """
        try:
            adjusted_edges = self.combined_edges * self.K_grid
            adjusted_edges[adjusted_edges < 0.2] = 0
            self.final_edge_image = (adjusted_edges * 255).astype(np.uint8)
            return self.final_edge_image
        except Exception as e:
            self.logger.error(f"Error in generate_final_edge_image: {e}")
            return None

    def process_edges(self):
        """
        Run the entire edge detection pipeline.
        """
        try:
            self.compute_virtual_pixel_block()
            self.apply_sobel_operator()
            self.apply_canny_operator()
            self.combined_edges = 0.7 * self.sobel_edges + 0.3 * self.canny_edges
            self.combined_edges = np.clip(self.combined_edges, 0, 1)

            combined_edges_2d = np.mean(self.combined_edges, axis=2)
            combined_edges_2d = (combined_edges_2d > 0.1).astype(np.float32)
            combined_edges_2d = scipy.ndimage.binary_dilation(combined_edges_2d, structure=np.ones((3, 3))).astype(np.float32)
            combined_edges_2d = scipy.ndimage.binary_erosion(combined_edges_2d, structure=np.ones((2, 2))).astype(np.float32)

            self.combined_edges = np.expand_dims(combined_edges_2d, axis=2)
            self.entropy_map = self.compute_2D_entropy(self.combined_edges)
            self.compute_K_grid()
            self.generate_final_edge_image()
        except Exception as e:
            self.logger.error(f"Error in process_edges: {e}")