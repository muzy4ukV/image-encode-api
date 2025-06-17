from google.cloud import storage
from .credentials import CREDENTIALS
import os
from datetime import timedelta, datetime, UTC
import io
import numpy as np
from .fragments_storage import FragmentsStorage


class GCSBucketUtils:

    def __init__(self):
        self.bucket_name = os.environ.get('GCP_BUCKET_NAME')
        self.client = storage.Client(credentials=CREDENTIALS, project=CREDENTIALS.project_id)
        self.bucket = self.client.bucket(self.bucket_name)

    def add_fragments_to_gcs(self, fragments: FragmentsStorage):
        """
        Зберігає словник фрагментів у вигляді .npy та завантажує його до GCS з іменем, що містить дату й кількість.

        :param fragments: Словник {int: Fragment}, який потрібно зберегти
        """
        # Генерація назви файлу
        fragment_count = fragments.get_fragments_count()
        date_str = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        object_name = f"fragments_{date_str}_{fragment_count}.npy"

        # Серіалізація у пам'ять
        buffer = io.BytesIO()
        np.save(buffer, fragments, allow_pickle=True)  # type: ignore[arg-type]
        buffer.seek(0)

        # Завантаження до GCS
        blob = self.bucket.blob(object_name)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        print(f"Fragments snapshot uploaded to GCS as {object_name}")
        return object_name

    def get_signed_url(self, object_name: str, expiration_minutes: int = 15) -> str:
        """
        Generates a signed URL for a GCS object for direct client download.
        Checks if the object exists before generating the link.
        """
        blob = self.bucket.blob(object_name)

        # 🔍 Перевірка, чи файл існує в бакеті
        if not blob.exists():
            raise FileNotFoundError(f"Object '{object_name}' does not exist in bucket '{self.bucket_name}'.")

        # ✅ Генерація signed URL
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
            response_disposition=f'attachment; filename="{object_name}"'
        )
        return url