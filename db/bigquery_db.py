import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account

from .base import BaseDB
from .bucket_utils import GCSBucketUtils

class BigQueryDB(BaseDB):
    def __init__(self):
        super().__init__()
        # Вказуємо шлях до вашого сервісного ключа
        self.service_account_path = 'service-key.json'
        # Завантажуємо облікові дані
        credentials = service_account.Credentials.from_service_account_file(self.service_account_path)
        # Створюємо клієнта BigQuery
        self.client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        self.target_size = 16
        self.tree = None
        self.target_table = self.client.get_table(self.client.dataset('fragments').table('fragments_features'))
        self.project_id = 'image-encoding'
        self.dataset_id = 'fragments'
        self.table_id = 'fragments_features'
        self.bucket = GCSBucketUtils()

    def is_empty(self):
        return False

    def build_tree(self):
        query = """
            SELECT name, COUNT(*) as total
            FROM `your_project_id.your_dataset_id.your_table_id`
            GROUP BY name
            ORDER BY total DESC
        """
        query_job = self.client.query(query)  # Запуск запиту

        # Обробка результатів
        for row in query_job:
            print(f"name: {row['name']}, total: {row['total']}")

    def add_fragment(self, fragment: np.array) -> str | None:
        img_id = self.bucket.add_fragment(fragment)

        fragment_doc = {
            "id": img_id,
            "feature": fragment.feature.tobytes()
        }

        # Вставка даних в таблицю BigQuery
        try:
            errors = self.client.insert_rows_json(self.target_table, [fragment_doc])  # Вставляємо дані
            if errors:
                print(f"Error occurred while inserting row: {errors}")
                return None
            else:
                return fragment_doc['id']
        except Exception as e:
            print(f"Error occurred while adding fragment to BigQuery: {e}")
            return None

    def health(self):
        # Тепер ви можете виконувати запити до BigQuery. Наприклад, вивести список датасетів:
        datasets = self.client.list_datasets()

        print("Datasets in project:", self.client.project)
        for dataset in datasets:
            print(dataset.dataset_id)