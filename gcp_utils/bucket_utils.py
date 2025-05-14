import numpy as np
from google.cloud import storage
from .credentials import CREDENTIALS
import cv2
import tempfile
import os

class GCSBucketUtils:

    def __init__(self):
        self.bucket_name = os.environ.get('GCP_BUCKET_NAME')
        self.client = storage.Client(credentials=CREDENTIALS, project=CREDENTIALS.project_id)
        self.bucket = self.client.get_bucket(self.bucket_name)

    def add_fragment(self, array: np.array, label: int):
        """
        Converts a numpy array to a PNG image, saves it with a unique name, and uploads it to the GCS bucket.
        :param label:
        :param array: Numpy array to be converted to a PNG image.
        """

        # Create a temporary file to save the PNG
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            # Convert the numpy array to a PNG file
            cv2.imwrite(temp_file.name, array)

            # Upload the file to the GCS bucket
            blob = self.bucket.blob(f"{label}.png")
            blob.upload_from_filename(temp_file.name)

    def get_fragment(self, fragment_id):
        blob = self.bucket.blob(f"{fragment_id}.png")
        content = blob.download_as_bytes()
        return content  # content — це bytes
