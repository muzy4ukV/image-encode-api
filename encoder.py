import cv2
import os
from typing import List

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
        self.ssim_threshold = 0.8
        self.model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))
        self.kernel_size = 16
        self.step_size = 8
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

            if len(self.db.fragments) >= checkpoints[check_index] and self.db_type == 'file':
                self.db.save_results(path_to_save=f"fragments_base/frag_count_{checkpoints[check_index]}.npy")
                check_index += 1

    def add_fragments_from_img(self, img: np.array):
        fragments = self.extract_fragments(img, self.kernel_size, self.step_size)
        prep_fragments = self.prepare_fragments(fragments)

        new_fragments = []
        if not self.db.is_empty():
            for fragment in prep_fragments:
                similar_fragment_id = self.db.find_similar_fragment_id(fragment.features)
                similar_fragment_img = self.db.get_fragment_by_id(similar_fragment_id)['image']
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
            if self.db.is_empty():
                new_fragment_id = self.db.add_fragment(fragment, flag=True)
                fragment_matches.append((fragment, new_fragment_id, 1))
                continue

            matched_fragment_id = self.db.find_similar_fragment_id(fragment.features)
            matched_fragment_image = self.db.get_fragment_by_id(matched_fragment_id)['image']
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
        encoded_bytes = np.array(encoded_fragment_info, dtype=np.uint64).tobytes()

        #Update fragment tree
        self.db.build_tree()

        print(f'Encoded data size: {len(encoded_bytes)}')
        print(f'Reused fragments count: {reused_fragments}')
        print(f'Reuse ratio: {reused_fragments / len(fragment_matches) * 100:.2f}%')

        return encoded_bytes

    @timing("Decoding time")
    def decode(self, compressed_data: bytes, image_shape: tuple, restore_image: bool = False) -> np.array:
        # Decompress the encoded fragment info
        decoded_array = np.frombuffer(compressed_data, dtype=np.uint64)

        # Parse fragment info into (fragment_id, x, y) tuples
        decoded_fragments = [
            tuple(decoded_array[i:i + 3])
            for i in range(0, len(decoded_array), 3)
        ]

        fragments = []
        for fragment_id, x, y in decoded_fragments:
            fragment = self.db.get_fragment_by_id(fragment_id)
            fragment_image = cv2.resize(fragment['image'], (self.kernel_size, self.kernel_size))
            fragments.append(Fragment(img=fragment_image, feature=fragment['feature'], x=x, y=y))

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
                        feature=features,
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

    def reconstruct_image(self, fragments: List[Fragment], image_shape: tuple, restore_image: bool) -> np.array:
        height = image_shape[0] - (image_shape[0] % self.kernel_size)
        width = image_shape[1] - (image_shape[1] % self.kernel_size)

        reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)

        # Підготовка словника фрагментів
        fragment_dict = {}  # {(y, x): feature}

        for fragment in fragments:
            x, y = int(fragment.x), int(fragment.y)
            h, w, _ = fragment.image.shape
            fragment_dict[(y, x)] = (fragment.features, fragment.image)

            reconstructed_image[y:y + h, x:x + w] = fragment.image
            mask[y:y + h, x:x + w] = True

        # Координати порожніх фрагментів (верхній лівий кут)
        empty_fragments = [
            (y, x)
            for y in range(0, height - self.kernel_size + 1, self.kernel_size)
            for x in range(0, width - self.kernel_size + 1, self.kernel_size)
            if not mask[y:y + self.kernel_size, x:x + self.kernel_size].any()
        ]

        if not empty_fragments:
            return reconstructed_image

        if restore_image:
            # KD-дерево по координатах заповнених фрагментів
            fragments_coords = np.array(list(fragment_dict.keys()))
            fragments_features = np.array([v[0] for v in fragment_dict.values()])
            fragments_images = np.array([v[1] for v in fragment_dict.values()])

            tree = KDTree(fragments_coords)

            for i, (y, x) in enumerate(empty_fragments):
                # Перевірка чи фрагмент на краю
                is_edge = (
                        x == 0 or y == 0 or
                        x + self.kernel_size >= width or
                        y + self.kernel_size >= height
                )
                num_neighbors = 5 if is_edge else 8

                # Пошук найближчих сусідів
                dists, idxs = tree.query((y, x), k=num_neighbors)
                neighbor_coords = fragments_coords[idxs]
                nearest_features = fragments_features[idxs]
                neighbor_fr_images = fragments_images[idxs]

                print(f"Empty at ({y}, {x}), neighbors at {neighbor_coords.tolist()}")

                predicted_feature = np.mean(nearest_features, axis=0)
                candidates = self.db.find_k_similar_fragments(predicted_feature, k=10)
                best_candidate = candidates[0]
                best_score = 0

                for candidate in candidates:
                    scores = [
                        self.get_ssim_for_fragments(candidate['image'], neighbor_fragment)
                        for neighbor_fragment in neighbor_fr_images
                    ]
                    if scores:
                        avg_ssim = np.mean(scores)
                        if avg_ssim > best_score:
                            best_score = avg_ssim
                            best_candidate = candidate

                reconstructed_image[y:y + self.kernel_size, x:x + self.kernel_size] = best_candidate['image']
        #reconstructed_image = self.blend_fragments(fragments, reconstructed_image, image_shape)

        return reconstructed_image

    @staticmethod
    def get_ssim_for_fragments(img1: np.ndarray, img2: np.ndarray) -> float:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        return ssim_metric(img1, img2, channel_axis=2, win_size=3, data_range=1.0)


    def blend_fragments(self, fragments: List[Fragment], image: np.ndarray, img_shape: tuple) -> np.ndarray:
        height, width = img_shape
        kernel_size = max(self.kernel_size // 2, 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for fragment in fragments:
            x, y = int(fragment.x), int(fragment.y)
            h, w, _ = fragment.image.shape

            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(width, x + w), min(height, y + h)

            region_img = image[y_start:y_end, x_start:x_end]
            region_frag = fragment.image[y_start - y:y_end - y, x_start - x:x_end - x]

            edges = cv2.Canny(region_img, 100, 200)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            if np.mean(edges) > 0:
                blended = cv2.addWeighted(region_img, 0.5, region_frag, 0.5, 0)
                image[y_start:y_end, x_start:x_end] = blended
            else:
                image[y_start:y_end, x_start:x_end] = region_frag

        return image

    def set_ssim_threshold(self, threshold: float):
        self.ssim_threshold = threshold

    def get_ssim(self, original_img: np.array, decoded_img: np.array) -> float:
        """
        Computes the Structural Similarity Index (SSIM) between two images.
        """
        return ssim_metric(original_img, decoded_img, multichannel=True, win_size=7, channel_axis=2)

    def is_overlapping(self, fragment1, fragment2):
        x1, y1, w1, h1 = fragment1[1], fragment1[2], self.kernel_size, self.kernel_size
        x2, y2, w2, h2 = fragment2[1], fragment2[2], self.kernel_size, self.kernel_size

        if x1 >= x2 + w2 or x2 >= x1 + w1 or y1 >= y2 + h2 or y2 >= y1 + h1:
            return False

        return True
