import numpy as np
from google.cloud import storage
import cv2
import tempfile
import uuid

class GCSBucketUtils:
    DEFAULT_SERVICE_ACCOUNT_PATH = 'service-key.json'
    DEFAULT_BUCKET_NAME = 'fragments-base'
    def __init__(self, bucket_name=DEFAULT_BUCKET_NAME, service_account_path=DEFAULT_SERVICE_ACCOUNT_PATH):
        self.bucket_name = bucket_name
        self.service_account_path = service_account_path
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
