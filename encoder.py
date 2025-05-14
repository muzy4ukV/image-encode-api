import cv2
import os
import lzma
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
        self.db = BigQueryDB()
        self.similarity_threshold = 0.9
        self.model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))
        self.kernel_size = 16
        self.step_size = 8

    def fill_db(self, dir_path: str):
        checkpoints = [20000, 50000, 100000, 150000, 200000]
        check_index = 0
        for file_path in os.listdir(dir_path):
            if not file_path.endswith('.png') and not file_path.endswith('.jpeg') and not file_path.endswith('.jpg'):
                continue
            
            print(f'Processing image {file_path}')
            img = cv2.imread(os.path.join(dir_path, file_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fragments = self.split_image_into_fragments(img, self.kernel_size, self.step_size)
             
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
        fragments = self.split_image_into_fragments(img, self.kernel_size, self.step_size)
        prep_fragments = self.prepare_fragments(fragments)
        print(f'Fragments count: {len(fragments)}')
        start_time = time()
        self.db.add_fragments(prep_fragments)
        print(f"Image fragments adding to bq time: {time() - start_time}")

        return len(fragments)

    @timing("Encoding time")
    def encode(self, img: np.array) -> bytes:
        fragments = self.split_image_into_fragments(img, self.kernel_size, self.step_size)
        prep_fragments = self.prepare_fragments(fragments)
        print(f'Fragments count: {len(fragments)}')

        encoded_data = [] # List of tuples (fragment_id, x, y)
        similarity_data = [] # List of tuples (fragment, fragment_id, similarity)
        fragments_count = 0
        for fragment in prep_fragments:
            if self.db.is_empty():
                new_fragment_id = self.db.add_fragment(fragment)
                similarity_data.append((fragment, new_fragment_id, 1))
                continue

            similar_fragment_id = self.db.find_similar_fragment_id(fragment.feature)
            similar_fragment_img = self.db.get_image_by_id(similar_fragment_id)
            similarity = self.get_ssim(fragment.img, similar_fragment_img)

            if similarity > self.similarity_threshold:
                fragments_count+=1
                similarity_data.append((fragment, similar_fragment_id, similarity))
            else:
                new_fragment_id = self.db.add_fragment(fragment)
                similarity_data.append((fragment, new_fragment_id, 1))

        # Filter and sort similarity data
        similarity_data.sort(key=lambda x: x[2], reverse=True)

        # Create a mask to track occupied areas
        mask = np.zeros(img.shape[:2], dtype=bool)

        # Add fragments to encoded_data
        for fragment, fragment_id, _ in similarity_data:
            x, y = fragment.x, fragment.y
            h, w = self.kernel_size, self.kernel_size

            # Clip fragment coordinates to fit image boundaries
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(img.shape[1], x + w)
            y_max = min(img.shape[0], y + h)

            # Check if the fragment overlaps with the occupied area
            if np.any(mask[y_min:y_max, x_min:x_max]):
                continue

            # Update the mask
            mask[y_min:y_max, x_min:x_max] = True

            # Add the fragment to the encoded data
            encoded_data.extend([fragment_id, x, y])

        # Convert encoded_data to bytes and return
        encoded_bytes = np.array(encoded_data, dtype=np.uint64).tobytes()
        compressed_data = lzma.compress(encoded_bytes)
        print(f'Encoded data size: {len(compressed_data)}')
        print(f'Count of fragments using: {fragments_count}')
        print(f'Percent of fragments using: {fragments_count/len(similarity_data)*100:.2f}%')

        return compressed_data
    
    @timing("Decoding time")
    def decode(self, compressed_indexes: bytes, img_size: tuple) -> np.array:
        decoded_data = lzma.decompress(compressed_indexes)
        decoded_array = np.frombuffer(decoded_data, dtype=np.uint64)
        decoded_info = [tuple(decoded_array[i:i+3]) for i in range(0, len(decoded_array), 3)]
        decoded_fragments = []
  
        for fragment in decoded_info:
            fragment_img = self.db.get_image_by_id(fragment[0])
            fragment_img = cv2.resize(fragment_img, (self.kernel_size, self.kernel_size))
            decoded_fragments.append(Fragment(img=fragment_img, x=fragment[1], y=fragment[2]))

        reconstructed_image = self.reconstruct_image(decoded_fragments, img_size)
        return reconstructed_image


    @tf.function
    def resize_and_predict(self, images):
        with tf.device('/GPU:0'):
            resized_images = tf.image.resize(images, [224, 224])
            return self.model(resized_images)

    @timing("Extraction features from fragments time")
    def prepare_fragments(self, fragments):
        batch_size = 64

        # Transform fragments into a tensor
        fragment_images = np.array([fragment.img for fragment in fragments])
        fragment_images = tf.convert_to_tensor(fragment_images, dtype=tf.float32)

        # Prepare fragments in batches
        num_batches = (len(fragments) + batch_size - 1) // batch_size
        prep_fragments = []
        for i in range(num_batches):
            batch_images = fragment_images[i*batch_size:(i+1)*batch_size]
            features = self.resize_and_predict(batch_images)
            features = features.numpy().reshape(features.shape[0], -1)

            # Добавляем обработанные фрагменты
            for j, feature in enumerate(features):
                fragment_index = i*batch_size + j
                if fragment_index < len(fragments):
                    fragment = fragments[fragment_index]
                    prep_fragments.append(Fragment(img=fragment.img, feature=feature, x=fragment.x, y=fragment.y))

        return prep_fragments
    

    def is_overlapping(self, fragment1, fragment2):
        """
        Checks if two fragments overlap.

        Args:
            fragment1 (tuple): First fragment in the form (fragment, x, y).
            fragment2 (tuple): Second fragment in the form (fragment, x, y).

        Returns:
            bool: True if the fragments overlap, False otherwise.
        """
        x1, y1, w1, h1 = fragment1[1], fragment1[2], self.kernel_size, self.kernel_size
        x2, y2, w2, h2 = fragment2[1], fragment2[2], self.kernel_size, self.kernel_size

        if x1 >= x2 + w2 or x2 >= x1 + w1 or y1 >= y2 + h2 or y2 >= y1 + h1:
            return False

        return True
           

    @staticmethod
    def split_image_into_fragments(image, kernel_size, step_size) -> List[Fragment]:
        """
        Splits an image into fragments.

        Args:
            image (numpy.array): The input image.
            kernel_size (int): Size of the square fragment.
            step_size (int): Step size for moving the window.

        Returns:
            list: A list of tuples (fragment, x, y), where fragment is a fragment of the image,
                (x, y) is the coordinate of the top-left corner of the fragment.
        """
        fragments = []
        height, width = image.shape[:2]

        # Check for valid parameters
        if kernel_size < 2 or kernel_size > min(height, width):
            raise ValueError("Invalid kernel size")
        if step_size < 1 or step_size > kernel_size:
            raise ValueError("Invalid step size")

        # Create a view of the image with the desired fragment shape and strides
        fragment_shape = (kernel_size, kernel_size, 3)
        strides = (image.strides[0]*step_size, image.strides[1]*step_size, image.strides[0], image.strides[1], image.strides[2])
        fragments_view = as_strided(image, shape=((height-kernel_size)//step_size+1, (width-kernel_size)//step_size+1, *fragment_shape), strides=strides)

        # Create a list of fragments
        for y in range(fragments_view.shape[0]):
            for x in range(fragments_view.shape[1]):
                fragment = fragments_view[y, x]
                fragments.append(Fragment(img=fragment, x=x*step_size, y=y*step_size))

        return fragments  


    def reconstruct_image(self, fragments: List[Fragment], img_size: tuple) -> np.array:
        """
        Reconstructs the image from non-overlapping fragments.

        Args:
            fragments (list): List of tuples (fragment, x, y) where fragment is a fragment of the image,
                            and (x, y) is the coordinate of the top-left corner of the fragment.
            img_size (tuple): Size of the original image (height, width).

        Returns:
            numpy.array: Reconstructed image.
        """
        # Create an empty image
        reconstructed_image = np.zeros((*img_size, 3), dtype=np.uint8)

        # Create a mask to mark occupied pixels
        mask = np.zeros(img_size, dtype=bool)
        for fragment in fragments:
            # Convert coordinates to integers
            x = int(fragment.x)
            y = int(fragment.y)
            h, w, _ = fragment.img.shape
            # Assign the fragment to the corresponding region in the reconstructed image
            reconstructed_image[y:y+h, x:x+w] = fragment.img
            mask[y:y+h, x:x+w] = True

        # Get the coordinates of non-black pixels
        fragment_indices = np.argwhere(mask)

        # Get the coordinates of empty pixels outside the filled fragments
        empty_indices = np.argwhere(~mask)

        # Create a KDTree from non-black pixel indices
        non_black_tree = KDTree(fragment_indices)

        # Define the number of nearest neighbors to consider
        num_neighbors = 16

        # For each empty pixel, find the nearest non-empty pixels and assign their color
        for y, x in empty_indices:
            # Find the nearest non-empty pixels using KDTree
            dist, nearest_indices = non_black_tree.query([(y, x)], k=num_neighbors)  # Find the nearest non-black pixels
            nearest_yx = fragment_indices[nearest_indices[0]]  # Get the coordinates of the nearest non-black pixels
            # Calculate the mean color of the nearest non-black pixels
            mean_color = np.mean(reconstructed_image[nearest_yx[:, 0], nearest_yx[:, 1]], axis=0)
            # Assign the mean color to the empty pixels
            reconstructed_image[y, x] = mean_color

        #reconstructed_image = self.blend_fragments(fragments, reconstructed_image, img_size)

        return reconstructed_image
    

    def blend_fragments(self, fragments: List[Fragment], reconstructed_image: np.array, img_size: tuple) -> np.array:
        for fragment in fragments:
            # Convert coordinates to integers
            x = int(fragment.x)
            y = int(fragment.y)
            h, w, _ = fragment.img.shape

            # Clip fragment coordinates to fit image boundaries
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(img_size[1], x + w)
            y_max = min(img_size[0], y + h)

            # Get overlapping region
            overlapping_region = reconstructed_image[y_min:y_max, x_min:x_max]
            # Get corresponding region in the fragment
            fragment_region = fragment.img[y_min - y:y_max - y, x_min - x:x_max - x]

            # Detect edges between the overlapping region and the fragment region
            edges = cv2.Canny(overlapping_region, 100, 200)
            
            # Apply morphological closing to get a wider border
            size = max(self.kernel_size // 2, 5)            
            kernel = np.ones((size, size), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Check if there is a significant color change along the border
            if np.mean(edges) > 0:
                # Perform blending only on the border
                blended_region = cv2.addWeighted(overlapping_region, 0.5, fragment_region, 0.5, 0)
                reconstructed_image[y_min:y_max, x_min:x_max] = blended_region
            else:
                # No significant color change detected, just copy the fragment
                reconstructed_image[y_min:y_max, x_min:x_max] = fragment_region

        return reconstructed_image


    def set_similarity_threshold(self, threshold: float):
        self.similarity_threshold = threshold

    def get_ssim(self, original_img: np.array, decoded_img: np.array) -> float:
        """
        Computes the Structural Similarity Index (SSIM) between two images.
        """
        similarity = ssim_metric(original_img, decoded_img, multichannel=True, win_size=min(self.db.target_size, 7), channel_axis=2)
        return similarity
    
