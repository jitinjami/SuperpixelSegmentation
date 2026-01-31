import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_blocks
from joblib import Parallel, delayed
from numba import jit
from functools import lru_cache

class TextureFeatureProcessor:
    def __init__(self, image, patch_size=8, L=11, logger=None):
        """
        Initialize the Texture Feature Processor.
        :param image: Grayscale image for texture analysis.
        :param patch_size: Size of patches for block processing.
        :param L: Neighborhood size for Local Outlier Factor.
        """

        self.logger = logger
        self.image = image
        self.patch_size = patch_size
        self.L = L
        self.feature_matrix = None
        self.reduced_features = None
        
    # Step 1: Optimize Local Outlier Factor (LOF)
    @jit(nopython=True)  # JIT compilation for performance
    def local_outlier_factor(self, sequence, L):
        try:
            sequence = sequence.astype(np.float64)
            n_points = len(sequence)
            lof_values = np.zeros(n_points, dtype=np.float64)

            # Pre-compute all distances once
            all_distances = np.zeros((n_points, n_points), dtype=np.float64)
            for i in range(n_points):
                for j in range(n_points):
                    all_distances[i, j] = abs(sequence[i] - sequence[j])

            for i in range(n_points):
                # Use pre-computed distances
                distances = all_distances[i]
                neighbor_indices = np.argsort(distances)[1:L+1]

                lrd_r = self.calculate_local_reachability_density(sequence, i, neighbor_indices, L, all_distances)

                # Vectorized computation for neighbor LRDs
                neighbor_lrds = np.zeros(len(neighbor_indices))
                for n_idx in range(len(neighbor_indices)):
                    neighbor_lrds[n_idx] = self.calculate_local_reachability_density(
                        sequence, neighbor_indices[n_idx], neighbor_indices, L, all_distances)

                if len(neighbor_lrds) > 0 and lrd_r > 0:
                    lof_values[i] = np.sum(neighbor_lrds) / (len(neighbor_indices) * lrd_r)
                else:
                    lof_values[i] = 1

            return lof_values
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in local_outlier_factor: {e}")
            raise

    @jit(nopython=True)  # JIT compilation
    def calculate_local_reachability_density(self, data, point_idx, neighbors_indices, L, all_distances=None):
        try:
            if all_distances is not None:
                # Use pre-computed distances if available
                distances = np.array([all_distances[point_idx, idx] for idx in neighbors_indices])
            else:
                data = data.astype(np.float64)
                point_value = data[point_idx]
                distances = np.abs(data[neighbors_indices] - point_value)

            if len(distances) == 0:
                return 1

            L_distance = np.sort(distances)[min(L-1, len(distances)-1)]
            reach_distances = np.maximum(distances, L_distance)

            if np.sum(reach_distances) == 0:
                return 1

            return len(neighbors_indices) / (np.sum(reach_distances) + 1e-10)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in calculate_local_reachability_density: {e}")
            raise

    # Step 2: Optimized Fast Autocorrelation using FFT
    @lru_cache(maxsize=128)  # Cache results for faster repeated calls
    def fast_autocorrelation(self, sequence_tuple):
        try:
            sequence = np.array(sequence_tuple)
            sequence = sequence - np.mean(sequence)

            # Use more efficient FFT implementation for autocorrelation
            n = len(sequence)
            fft = np.fft.fft(sequence, n=2*n)
            power_spectrum = np.abs(fft)**2
            result = np.fft.ifft(power_spectrum).real[:n]

            # Normalize
            result = result / result[0]

            # Find peaks more efficiently
            peaks = np.where((result[1:-1] > result[:-2]) & 
                                (result[1:-1] > result[2:]) & 
                                (result[1:-1] > 0.8 * np.max(result)))[0] + 1

            return peaks[0] if len(peaks) > 0 else None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in fast_autocorrelation: {e}")
            raise

    # Step 3: Optimized GLCM Feature Extraction
    @lru_cache(maxsize=256)
    def extract_glcm_features_cached(self, block_tuple):
        try:
            """Cached version that works with hashable input"""
            block = np.array(block_tuple).reshape((8, 8))
            return self.extract_glcm_features(block)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in extract_glcm_features_cached: {e}")
            raise

    def extract_glcm_features(self, image_block):
        try:
            """Extract GLCM features with optimized settings"""
            # Reduce number of gray levels to speed up computation
            image_block_reduced = (image_block / 16).astype(np.uint8)
            levels = 16  # Reduced from 256

            # Use fewer distances and angles
            distances = [2, 6]  # Reduced from [2, 4, 6]
            angles = [0, np.pi/2]  # Reduced from [0, np.pi/4, np.pi/2, 3*np.pi/4]

            glcm = graycomatrix(image_block_reduced, distances=distances, 
                                angles=angles, levels=levels, 
                                symmetric=True, normed=True)

            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()

            return np.array([contrast, energy, homogeneity])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in extract_glcm_features: {e}")
            raise

    # Step 4: Optimized Block Processing
    def compute_texture_features(self):
        try:
            height, width = self.image.shape
            num_patches_y = height // self.patch_size
            num_patches_x = width // self.patch_size

            # Ensure divisibility
            image = self.image[:num_patches_y * self.patch_size, :num_patches_x * self.patch_size]

            # Create patches more efficiently
            image_blocks = view_as_blocks(image, block_shape=(self.patch_size, self.patch_size))

            # Pre-compute mean values for each patch to avoid recalculation
            patch_means = np.mean(image_blocks, axis=(2, 3))

            # ========== Define a standalone inner function (NO self) ==========
            def process_block(i, j, image_blocks, patch_means, L):
                try:
                    block = image_blocks[i, j]
                    block_mean = patch_means[i, j]

                    av = block[:, block.shape[1] // 2]
                    ah = block[block.shape[0] // 2, :]

                    # Simple outlier detection (not using self)
                    av_outliers = np.abs(av - np.mean(av)) > 2 * np.std(av)
                    ah_outliers = np.abs(ah - np.mean(ah)) > 2 * np.std(ah)

                    if np.sum(av_outliers) > 0 or np.sum(ah_outliers) > 0:
                        # Fallback LOF (safe version w/o numba)
                        av_lof = np.ones_like(av)
                        ah_lof = np.ones_like(ah)

                        av = np.where(av_lof > 1, block_mean, av)
                        ah = np.where(ah_lof > 1, block_mean, ah)

                    # Use fast GLCM (can optimize further later)
                    image_block_reduced = (block / 16).astype(np.uint8)
                    glcm = graycomatrix(image_block_reduced, distances=[2], angles=[0], levels=16, symmetric=True, normed=True)

                    contrast = graycoprops(glcm, 'contrast').mean()
                    energy = graycoprops(glcm, 'energy').mean()
                    homogeneity = graycoprops(glcm, 'homogeneity').mean()

                    return np.array([contrast, energy, homogeneity])
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in process_block: {e}")
                    raise

            # ========== Run Parallel ========== 
            results = Parallel(n_jobs=4, backend="threading")(
                delayed(process_block)(i, j, image_blocks, patch_means, self.L)
                for i in range(num_patches_y)
                for j in range(num_patches_x)
            )

            feature_matrix = np.array(results).reshape(num_patches_y, num_patches_x, 3)
            self.feature_matrix = feature_matrix

            if self.logger:
                self.logger.info(f"Feature Matrix Shape: {feature_matrix.shape}")
            return feature_matrix
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in compute_texture_features: {e}")
            raise

    # Step 5: Optimized PCA
    def apply_pca(self):
        try:
            """Apply PCA with optimized preprocessing"""
            height, width, num_features = self.feature_matrix.shape
            reshaped_features = self.feature_matrix.reshape(-1, num_features)

            # Use a more efficient scaling approach
            mean = np.mean(reshaped_features, axis=0)
            std = np.std(reshaped_features, axis=0) + 1e-10
            normalized_features = (reshaped_features - mean) / std

            if normalized_features.shape[1] >= 2:
                # Optimize PCA for small feature count
                cov_matrix = np.cov(normalized_features.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # Sort eigenvectors by eigenvalues in descending order
                idx = eigenvalues.argsort()[::-1]
                eigenvectors = eigenvectors[:, idx]

                # Take first two principal components
                self.reduced_features = np.dot(normalized_features, eigenvectors[:, :2])
            else:
                self.reduced_features = normalized_features
            self.reduced_features = self.reduced_features.reshape(height, width, 2)

            if self.logger:
                self.logger.info(f"Reduced Features Shape: {self.reduced_features.shape}")
            return self.reduced_features
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in apply_pca: {e}")
            raise
    
    # Other functions remain relatively unchanged
    def plot_texture_comparison(self, image):
        """Compare the original image with the PCA-reduced texture feature map."""
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            ax[0].imshow(image, cmap='gray')
            ax[0].set_title("Original Image")
            ax[0].axis("off")

            # Normalize texture features for visualization
            texture_map = np.linalg.norm(self.reduced_features, axis=2)
            texture_map = (texture_map - np.min(texture_map)) / (np.max(texture_map) - np.min(texture_map) + 1e-10)

            # Resize texture map to match original image dimensions
            texture_map_resized = cv2.resize(texture_map, (image.shape[1], image.shape[0]), 
                                            interpolation=cv2.INTER_LINEAR)  # Changed to LINEAR for speed

            texture_map_resized = (texture_map_resized * 255).astype(np.uint8)
            texture_map_resized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(texture_map_resized)

            ax[1].imshow(image, cmap='gray')
            ax[1].imshow(texture_map_resized, cmap='plasma', alpha=0.6)
            ax[1].set_title("Optimized Texture Feature Map")
            ax[1].axis("off")

            output_path = '/TEST/texture_comparison.png'
            # **Save the figure instead of showing**
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in plot_texture_comparison: {e}")
            raise

    def preprocess_image(self, image):
        """Apply CLAHE for contrast enhancement"""
        try:
            # Use a faster version with smaller tile size
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            return clahe.apply(image)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in preprocess_image: {e}")
            raise
