import numpy as np
from google.cloud import storage
import cv2
import tempfile
import uuid
import os

class GCSBucketUtils:

    def __init__(self):
        self.bucket_name = os.environ.get('GCP_BUCKET_NAME')
        self.service_account_path = os.environ.get('GOOGLE_CREDENTIALS_PATH')
        self.client = storage.Client.from_service_account_json(self.service_account_path)
        self.bucket = self.client.get_bucket(self.bucket_name)

    def add_fragment(self, array: np.array) -> str:
        """
        Converts a numpy array to a PNG image, saves it with a unique name, and uploads it to the GCS bucket.

        :param array: Numpy array to be converted to a PNG image.
        :return: Generated unique file name used for saving in the GCS bucket.
        """

        # Generate a unique file name with .png extension
        file_uuid = uuid.uuid4()
        unique_file_name = f"{file_uuid}.png"

        # Create a temporary file to save the PNG
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            # Convert the numpy array to a PNG file
            cv2.imwrite(temp_file.name, array)

            # Upload the file to the GCS bucket
            blob = self.bucket.blob(unique_file_name)
            blob.upload_from_filename(temp_file.name)

        # Return the unique file name
        return str(file_uuid)
