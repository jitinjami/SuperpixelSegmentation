import numpy as np
import cv2


class Superpixel:
    def __init__(self, edge_features, texture_features, image_shape, num_superpixels=500, gray_values=None, logger=None):
        try:
            self.edge_features = edge_features / np.max(edge_features)
            self.texture_features = cv2.normalize(texture_features, None, 0, 1, cv2.NORM_MINMAX)
            self.height, self.width, _ = image_shape
            self.K = num_superpixels
            self.S = int(np.sqrt((self.height * self.width) / self.K))
            self.gray_values = gray_values
            self.logger = logger 

            self.labels = -1 * np.ones((self.height, self.width), dtype=np.int32)
            self.distances = np.full((self.height, self.width), np.inf)
            self.centers = np.zeros((self.K, 7))  # (y, x, R, G, B, ft1, ft2)

            if self.gray_values is None:
                raise ValueError("Gray values of the image must be provided")
        except Exception as e:
            self.logger.error(f"Error initializing Superpixel: {e}")
            raise

    def initialize_centers(self):
        try:
            texture_height, texture_width = self.texture_features.shape[:2]
            scale_y = texture_height / self.height
            scale_x = texture_width / self.width
            h_step = self.height // int(np.sqrt(self.K))
            w_step = self.width // int(np.sqrt(self.K))

            center_index = 0
            for i in range(h_step // 2, self.height, h_step):
                for j in range(w_step // 2, self.width, w_step):
                    if center_index >= self.K:
                        break

                    y_range = np.clip(np.arange(i - 1, i + 2), 0, self.height - 1)
                    x_range = np.clip(np.arange(j - 1, j + 2), 0, self.width - 1)
                    yv, xv = np.meshgrid(y_range, x_range, indexing='ij')

                    min_idx = np.argmin(self.edge_features[yv, xv])
                    center_y, center_x = yv.ravel()[min_idx], xv.ravel()[min_idx]

                    if len(self.gray_values.shape) == 2:
                        self.gray_rgb = np.stack([self.gray_values] * 3, axis=-1)
                    else:
                        self.gray_rgb = self.gray_values

                    tex_y = min(int(center_y * scale_y), texture_height - 1)
                    tex_x = min(int(center_x * scale_x), texture_width - 1)

                    self.centers[center_index] = [
                        center_y, center_x,
                        self.gray_rgb[center_y, center_x, 0],
                        self.gray_rgb[center_y, center_x, 1],
                        self.gray_rgb[center_y, center_x, 2],
                        self.texture_features[tex_y, tex_x, 0],
                        self.texture_features[tex_y, tex_x, 1]
                    ]
                    center_index += 1
            self.logger.info('initialize_centers completed successfully')
        except Exception as e:
            self.logger.error(f"Error in initialize_centers: {e}")
            raise

    def calculate_edge_dependent_coefficient(self):
        try:
            max_edge = np.max(self.edge_features)
            threshold = 0.7 * max_edge
            self.edge_coefficients = np.ones_like(self.edge_features) * 0.3
            self.edge_coefficients[self.edge_features >= threshold] = 2.0
            self.logger.info('calculate_edge_dependent_coefficient completed successfully')
        except Exception as e:
            self.logger.error(f"Error in calculate_edge_dependent_coefficient: {e}")
            raise

    def update_clusters(self):
        try:
            texture_height, texture_width = self.texture_features.shape[:2]
            scale_y = texture_height / self.height
            scale_x = texture_width / self.width

            if len(self.gray_values.shape) == 2:
                self.gray_rgb = np.stack([self.gray_values] * 3, axis=-1)
            else:
                self.gray_rgb = self.gray_values

            for k in range(self.K):
                center_y, center_x, center_R, center_G, center_B, center_ft1, center_ft2 = self.centers[k]

                y_min, y_max = max(0, int(center_y) - 2 * self.S), min(self.height, int(center_y) + 2 * self.S)
                x_min, x_max = max(0, int(center_x) - 2 * self.S), min(self.width, int(center_x) + 2 * self.S)

                y_range = np.arange(y_min, y_max)
                x_range = np.arange(x_min, x_max)
                yv, xv = np.meshgrid(y_range, x_range, indexing='ij')

                dl = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2) / (2 * self.S)

                gray_pixel_value = np.mean(self.gray_rgb[yv, xv], axis=-1) / 255.0
                center_gray_value = np.mean([center_R, center_G, center_B]) / 255.0
                dg = np.abs(center_gray_value - gray_pixel_value)

                tex_y = np.clip((yv * scale_y).astype(int), 0, texture_height - 1)
                tex_x = np.clip((xv * scale_x).astype(int), 0, texture_width - 1)
                ft1 = self.texture_features[tex_y, tex_x, 0]
                ft2 = self.texture_features[tex_y, tex_x, 1]
                dt = np.sqrt((center_ft1 - ft1) ** 2 + (center_ft2 - ft2) ** 2) / np.sqrt(2)

                edge_values = self.edge_features[yv, xv]
                edge_max = np.max(self.edge_features)
                n = 0.85
                threshold = n * edge_max
                alpha = np.where(edge_values < threshold, np.exp(edge_values), np.exp(-edge_values))

                D = 2.0 * alpha * dl + 1.5 * dg + 0.5 * dt

                mask = D < self.distances[yv, xv]
                self.labels[yv[mask], xv[mask]] = k
                self.distances[yv[mask], xv[mask]] = D[mask]

            self.logger.info('update_clusters completed successfully')
        except Exception as e:
            self.logger.error(f"Error in update_clusters: {e}")
            raise

    def update_centers(self):
        try:
            for k in range(self.K):
                mask = (self.labels == k)
                if np.sum(mask) == 0:
                    continue

                y_indices, x_indices = np.where(mask)
                y_mean, x_mean = np.mean(y_indices).astype(int), np.mean(x_indices).astype(int)

                tex_y = min(int(y_mean * self.texture_features.shape[0] / self.height), self.texture_features.shape[0] - 1)
                tex_x = min(int(x_mean * self.texture_features.shape[1] / self.width), self.texture_features.shape[1] - 1)

                R = np.mean(self.gray_rgb[y_indices, x_indices, 0])
                G = np.mean(self.gray_rgb[y_indices, x_indices, 1])
                B = np.mean(self.gray_rgb[y_indices, x_indices, 2])

                self.centers[k] = [
                    y_mean, x_mean, R, G, B,
                    self.texture_features[tex_y, tex_x, 0],
                    self.texture_features[tex_y, tex_x, 1]
                ]
            self.logger.info('update_centers completed successfully')
        except Exception as e:
            self.logger.error(f"Error in update_centers: {e}")
            raise

    def generate_superpixels(self, max_iterations=10, tolerance=1e-3):
        try:
            self.initialize_centers()
            self.calculate_edge_dependent_coefficient()

            for iteration in range(max_iterations):
                self.logger.info(f"Iteration: {iteration}")
                self.update_clusters()
                old_centers = self.centers.copy()
                self.update_centers()

                movement = np.linalg.norm(self.centers[:, :2] - old_centers[:, :2], axis=1)
                if np.mean(movement) < tolerance:
                    break

            self.logger.info('generate_superpixels completed successfully')
            return self.labels
        except Exception as e:
            self.logger.error(f"Error in generate_superpixels: {e}")
            raise

    def visualize_superpixels(self, original_image=None):
        try:
            mean_color_img = np.zeros_like(original_image)

            for k in range(self.K):
                mask = (self.labels == k)
                if np.sum(mask) > 0:
                    for c in range(3):
                        mean_color = np.mean(original_image[mask, c])
                        mean_color_img[mask, c] = mean_color

            vis_img = mean_color_img.copy()

            contours = []
            for k in range(self.K):
                mask = (self.labels == k).astype(np.uint8)
                found_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(found_contours)

            boundary_thickness = 1
            cv2.drawContours(vis_img, contours, -1, (255, 0, 0), thickness=boundary_thickness)

            result = cv2.addWeighted(original_image, 0.6, vis_img, 0.4, 0)

            self.logger.info('visualize_superpixels completed successfully')
            return result
        except Exception as e:
            self.logger.error(f"Error in visualize_superpixels: {e}")
            raise
