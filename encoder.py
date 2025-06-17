import cv2
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
from skimage.metrics import structural_similarity as ssim_metric
from scipy.spatial import KDTree

from gcp_utils.bigquery_db import BigQueryDB
from fragment import Fragment
from time import time
from decorators import timing


class Encoder:
    def __init__(self):
        self.ssim_threshold = 0.87
        self.model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))
        self.kernel_size = 160
        self.step_size = 160
        self.db = BigQueryDB()

    def fill_db(self, dir_path: str):
        checkpoints = [20000, 50000, 100000, 150000, 200000]
        check_index = 0
        for file_path in os.listdir(dir_path):
            if not file_path.endswith('.png') and not file_path.endswith('.jpeg') and not file_path.endswith('.jpg'):
                continue

            print(f'Processing image {file_path}')
            img = cv2.imread(os.path.join(dir_path, file_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fragments = self.extract_fragments(img, self.kernel_size, self.step_size)

            print(f'Fragments count: {len(fragments)}')

            prep_fragments = self.prepare_fragments(fragments)

            start_time = time()

            for fragment in prep_fragments:
                self.db.add_fragment(fragment)

            print(f'Fragments adding time: {time() - start_time}')

            if len(self.db.storage) >= checkpoints[check_index] and self.db_type == 'file':
                self.db.save_results(path_to_save=f"fragments_base/frag_count_{checkpoints[check_index]}.npy")
                check_index += 1

    def add_fragments_from_img(self, img: np.array):
        fragments = self.extract_fragments(img, self.kernel_size, self.step_size)
        prep_fragments = self.prepare_fragments(fragments)

        new_fragments = []
        if not self.db.is_empty():
            for fragment in prep_fragments:
                similar_fragment_id = self.db.find_similar_fragment_id(fragment.features)
                similar_fragment_img = self.db.get_fragment_by_id(similar_fragment_id).image
                similarity = self.get_ssim(fragment.image, similar_fragment_img)

                if similarity < self.ssim_threshold:
                    new_fragments.append(fragment)
        else:
            new_fragments = prep_fragments

        print(f'All fragments from image: {len(fragments)}')
        print(f'New unique fragments count: {len(new_fragments)}')

        start_time = time()
        status = self.db.add_fragments(new_fragments)
        print(f"Image fragments adding to bq time: {time() - start_time}")

        return status, len(new_fragments)

    @timing("Encoding time")
    def encode(self, img: np.array) -> bytes:
        fragments = self.extract_fragments(img, self.kernel_size, self.step_size)
        prepared_fragments = self.prepare_fragments(fragments)

        print(f'Fragments count: {len(fragments)}')

        encoded_fragment_info = []  # List of tuples (fragment_id, x, y)
        fragment_matches = []  # List of tuples (fragment, fragment_id, similarity_score)
        reused_fragments = 0

        for fragment in prepared_fragments:
            matched_fragment_id = self.db.find_similar_fragment_id(fragment.features)
            matched_fragment_image = self.db.get_fragment_by_id(matched_fragment_id).image
            similarity_score = self.get_ssim(fragment.image, matched_fragment_image)

            if similarity_score > self.ssim_threshold:
                reused_fragments += 1
                fragment_matches.append((fragment, matched_fragment_id, similarity_score))
            else:
                new_fragment_id = self.db.add_fragment(fragment, flag=True)
                fragment_matches.append((fragment, new_fragment_id, 1))

        # Filter and sort similarity data
        fragment_matches.sort(key=lambda x: x[2], reverse=True)

        # Create a mask to track occupied areas
        mask = np.zeros(img.shape[:2], dtype=bool)

        # Add fragments to encoded_fragment_info
        for fragment, fragment_id, _ in fragment_matches:
            x, y = fragment.x, fragment.y
            h, w = self.kernel_size, self.kernel_size

            # Clip fragment coordinates to fit image boundaries
            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(img.shape[1], x + w), min(img.shape[0], y + h)

            # Check if the fragment overlaps with the occupied area
            if np.any(mask[y_start:y_end, x_start:x_end]):
                continue

            # Update the mask
            mask[y_start:y_end, x_start:x_end] = True

            # Add the fragment to the encoded data
            encoded_fragment_info.extend([fragment_id, x, y])

        # Convert encoded_fragment_info to bytes and return
        encoded_bytes = np.array(encoded_fragment_info, dtype=np.uint32).tobytes()

        # Update fragment tree
        self.db.build_tree()

        print(f'Encoded data size: {len(encoded_bytes)}')
        print(f'Reused fragments count: {reused_fragments}')
        print(f'Reuse ratio: {reused_fragments / len(fragment_matches) * 100:.2f}%')

        return encoded_bytes

    @timing("Decoding time")
    def decode(self, compressed_data: bytes, image_shape: tuple, restore_image: bool = False) -> np.array:
        # Decompress the encoded fragment info
        decoded_array = np.frombuffer(compressed_data, dtype=np.uint32)

        # Parse fragment info into (fragment_id, x, y) tuples
        decoded_fragments = [
            tuple(decoded_array[i:i + 3])
            for i in range(0, len(decoded_array), 3)
        ]

        fragments = []
        for fragment_id, x, y in decoded_fragments:
            fragment = self.db.get_fragment_by_id(fragment_id)
            fragment.x, fragment.y = x, y
            fragments.append(fragment)

        # Reconstruct the original image from fragments
        reconstructed_image = self.reconstruct_image(fragments, image_shape, restore_image)
        return reconstructed_image

    @tf.function
    def resize_and_predict(self, images: tf.Tensor) -> tf.Tensor:
        with tf.device('/GPU:0'):
            resized_images = tf.image.resize(images, [224, 224])
            return self.model(resized_images)

    @timing("Feature extraction from fragments")
    def prepare_fragments(self, fragments: list) -> list:
        batch_size = 64

        # Transform fragments into a tensor
        images_np = np.array([fragment.image for fragment in fragments])
        images_tensor = tf.convert_to_tensor(images_np, dtype=tf.float32)

        # Prepare fragments in batches
        num_fragments = len(fragments)
        prepared_fragments = []

        for start in range(0, num_fragments, batch_size):
            end = min(start + batch_size, num_fragments)
            batch_images = images_tensor[start:end]

            # Get features from CNN model
            batch_features = self.resize_and_predict(batch_images)
            batch_features = batch_features.numpy().reshape(batch_features.shape[0], -1)

            for i, features in enumerate(batch_features):
                original_fragment = fragments[start + i]
                prepared_fragments.append(
                    Fragment(
                        img=original_fragment.image,
                        features=features,
                        x=original_fragment.x,
                        y=original_fragment.y
                    )
                )

        return prepared_fragments

    @staticmethod
    def extract_fragments(image: np.ndarray, kernel_size: int, step_size: int) -> List[Fragment]:
        height, width = image.shape[:2]

        # Check for valid parameters
        if not (2 <= kernel_size <= min(height, width)):
            raise ValueError(f"Invalid kernel size: {kernel_size}. Must be between 2 and min(height, width).")
        if not (1 <= step_size <= kernel_size):
            raise ValueError(f"Invalid step size: {step_size}. Must be between 1 and kernel_size.")

        # Create a view of the image with the desired fragment shape and strides
        fragment_h, fragment_w, channels = kernel_size, kernel_size, 3
        num_fragments_y = (height - fragment_h) // step_size + 1
        num_fragments_x = (width - fragment_w) // step_size + 1

        fragment_shape = (num_fragments_y, num_fragments_x, fragment_h, fragment_w, channels)
        strides = (
            image.strides[0] * step_size,
            image.strides[1] * step_size,
            image.strides[0],
            image.strides[1],
            image.strides[2]
        )
        fragments_view = as_strided(image, shape=fragment_shape, strides=strides)

        fragments = [
            Fragment(img=fragments_view[y, x], x=x * step_size, y=y * step_size)
            for y in range(num_fragments_y)
            for x in range(num_fragments_x)
        ]

        return fragments

    def reconstruct_image(self, fragments: List[Fragment], image_shape: tuple,
                          should_restore_image: bool) -> np.ndarray:
        """Reconstructs an image from fragments."""
        height, width = self._adjust_image_dimensions(image_shape)
        reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)

        # Prepare fragment dictionary and blend initial fragments into the image
        fragment_dict = self._populate_fragment_dict(fragments, reconstructed_image, mask)

        # Calculate coordinates of empty fragments
        empty_fragments = self._calculate_empty_fragments(mask, height, width)

        if not empty_fragments:
            return reconstructed_image

        if should_restore_image:
            fragments_coords, fragments_features, fragments_images = self._prepare_kdtree_data(fragment_dict)
            tree = KDTree(fragments_coords)

            for y, x in empty_fragments:
                self._process_empty_fragment(
                    y, x, height, width, fragments_coords, fragments_features, fragments_images, tree,
                    reconstructed_image
                )

            self.blend_fragments(empty_fragments, reconstructed_image, image_shape)

        return reconstructed_image

    @staticmethod
    def get_ssim_for_fragments(img1: np.ndarray, img2: np.ndarray) -> float:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        return ssim_metric(img1, img2, channel_axis=2, win_size=7, data_range=1.0)

    # Helper Functions
    def _adjust_image_dimensions(self, image_shape: tuple) -> tuple:
        height = image_shape[0] - (image_shape[0] % self.kernel_size)
        width = image_shape[1] - (image_shape[1] % self.kernel_size)
        return height, width

    def _populate_fragment_dict(self, fragments: List[Fragment], reconstructed_image: np.ndarray,
                                mask: np.ndarray) -> dict:
        fragment_dict = {}
        for fragment in fragments:
            x, y = int(fragment.x), int(fragment.y)
            frag_height, frag_width, _ = fragment.image.shape
            reconstructed_image[y:y + frag_height, x:x + frag_width] = fragment.image
            mask[y:y + frag_height, x:x + frag_width] = True
            fragment_dict[(y, x)] = (fragment.features, fragment.image)
        return fragment_dict

    def _calculate_empty_fragments(self, mask: np.ndarray, height: int, width: int) -> list:
        return [
            (y, x)
            for y in range(0, height - self.kernel_size + 1, self.kernel_size)
            for x in range(0, width - self.kernel_size + 1, self.kernel_size)
            if not mask[y:y + self.kernel_size, x:x + self.kernel_size].any()
        ]

    def _prepare_kdtree_data(self, fragment_dict: dict) -> tuple:
        fragments_coords = np.array(list(fragment_dict.keys()))
        fragments_features = np.array([v[0] for v in fragment_dict.values()])
        fragments_images = np.array([v[1] for v in fragment_dict.values()])
        return fragments_coords, fragments_features, fragments_images

    def _process_empty_fragment(
            self, y: int, x: int, height: int, width: int, fragments_coords: np.ndarray,
            fragments_features: np.ndarray, fragments_images: np.ndarray, tree: KDTree, reconstructed_image: np.ndarray
    ):
        NUM_NEIGHBORS_EDGE = 5
        NUM_NEIGHBORS_INTERNAL = 8

        is_edge = (x == 0 or y == 0 or x + self.kernel_size >= width or y + self.kernel_size >= height)
        num_neighbors = NUM_NEIGHBORS_EDGE if is_edge else NUM_NEIGHBORS_INTERNAL

        # Find nearest neighbors
        dists, idxs = tree.query((y, x), k=num_neighbors)
        neighbor_features = fragments_features[idxs]
        neighbor_images = fragments_images[idxs]

        # Predict features for the missing fragment
        predicted_feature = np.mean(neighbor_features, axis=0)
        candidates = self.db.find_k_similar_fragments(predicted_feature, k=10)
        best_candidate, best_score = self._find_best_candidate(candidates, neighbor_images)

        # Fill the fragment
        reconstructed_image[y:y + self.kernel_size, x:x + self.kernel_size] = best_candidate.image
        best_candidate.x, best_candidate.y = x, y
        return best_candidate

    def _find_best_candidate(self, candidates: list, neighbor_images: np.ndarray) -> tuple:
        best_candidate = candidates[0]
        best_score = 0
        for candidate in candidates:
            scores = [
                self.get_ssim_for_fragments(candidate.image, neighbor_image)
                for neighbor_image in neighbor_images
            ]
            if scores:
                avg_ssim = np.mean(scores)
                if avg_ssim > best_score:
                    best_score = avg_ssim
                    best_candidate = candidate
        return best_candidate, best_score

    def blend_fragments(self, fragments: List[Tuple[int, int]], image: np.ndarray, img_shape: tuple) -> np.ndarray:
        height, width = img_shape
        size = self.kernel_size
        edge_width = max(1, self.kernel_size // 10)

        for y, x in fragments:
            y_start = max(0, y - edge_width)
            y_end = min(height, y + size + edge_width)
            x_start = max(0, x - edge_width)
            x_end = min(width, x + size + edge_width)

            fragment = image[y_start:y_end, x_start:x_end].copy().astype(np.float32)

            # Блюримо весь фрагмент
            blurred = cv2.GaussianBlur(fragment, (21, 21), 5.0)

            # === Створюємо градієнтну альфа-маску ===
            h, w = fragment.shape[:2]
            alpha = np.ones((h, w), dtype=np.float32)

            ramp = np.linspace(0, 1, edge_width)

            # верх
            alpha[:edge_width, :] *= ramp[::-1, None]
            alpha[edge_width:edge_width*2, :] *= ramp[:, None]
            # низ
            alpha[-edge_width:, :] *= ramp[:][:, None]
            alpha[-edge_width*2:-edge_width:, :] *= ramp[::-1][:, None]
            # ліво
            alpha[:, :edge_width] *= ramp[None, ::-1]
            alpha[:, edge_width:edge_width*2] *= ramp[None, :]
            # право
            alpha[:, -edge_width:] *= ramp[:][None, :]
            alpha[:, -edge_width*2:-edge_width] *= ramp[::-1][None, :]


            # Додаємо розмірність для каналу (RGB)
            alpha = alpha[..., None]

            # === Плавне альфа-змішування ===
            result = fragment * alpha + blurred * (1 - alpha)
            result = np.clip(result, 0, 255).astype(np.uint8)

            # Записуємо назад
            image[y_start:y_end, x_start:x_end] = result

        return image

    def set_ssim_threshold(self, threshold: float):
        self.ssim_threshold = threshold

    def get_ssim(self, original_img: np.ndarray, decoded_img: np.ndarray) -> float:
        """
        Computes SSIM between two images. Resizes only if images are larger than 1024x1024.
        Ensures they match in size before comparison.
        """
        target_size = (1024, 1024)

        def resize_if_larger(img: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
            h, w = img.shape[:2]
            max_h, max_w = max_size
            if h > max_h or w > max_w:
                return cv2.resize(img, (max_w, max_h))
            return img

        # Застосовуємо масштабування лише якщо зображення великі
        original_img = resize_if_larger(original_img, target_size)
        decoded_img = resize_if_larger(decoded_img, target_size)

        # Вирівнюємо розмір, якщо все ще не збігається
        if original_img.shape != decoded_img.shape:
            print(f"[INFO] Resizing original to match decoded: {decoded_img.shape}")
            original_img = cv2.resize(original_img, (decoded_img.shape[1], decoded_img.shape[0]))

        return ssim_metric(original_img, decoded_img, win_size=7, channel_axis=2)

    def is_overlapping(self, fragment1, fragment2):
        x1, y1, w1, h1 = fragment1[1], fragment1[2], self.kernel_size, self.kernel_size
        x2, y2, w2, h2 = fragment2[1], fragment2[2], self.kernel_size, self.kernel_size

        if x1 >= x2 + w2 or x2 >= x1 + w1 or y1 >= y2 + h2 or y2 >= y1 + h1:
            return False

        return True
