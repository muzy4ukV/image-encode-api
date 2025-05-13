import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import pandas as pd

from typing import Optional

from fragment import Fragment
from .base import BaseDB
from .bucket_utils import GCSBucketUtils

class BigQueryDB(BaseDB):
    def __init__(self):
        super().__init__()
        # Завантажуємо облікові дані
        credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_CREDENTIALS_PATH'])
        # Створюємо клієнта BigQuery
        self.client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        self.target_size = 16
        self.tree = None
        self.project_id = credentials.project_id
        self.dataset_id = os.environ['GCP_DATASET_ID']
        self.table_id = os.environ['GCP_TABLE_ID']
        self.target_table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{self.table_id}")
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

    def add_fragments(self, fragments: list[Fragment]) -> Optional[str]:
        rows = []
        for fragment in fragments:
            img_uuid = self.bucket.add_fragment(fragment.img)
            rows.append({
                "id": img_uuid,
                "features": fragment.feature.tobytes()
            })

        fragments_df = pd.DataFrame(rows)

        try:
            job = self.client.load_table_from_dataframe(fragments_df, self.target_table)
            job.result()
            return "Data loaded successfully"

        except Exception as e:
            print(f"Error occurred while adding fragments to BigQuery: {e}")
            return None

    def health(self):
        # Тепер ви можете виконувати запити до BigQuery. Наприклад, вивести список датасетів:
        datasets = self.client.list_datasets()

        print("Datasets in project:", self.client.project)
        for dataset in datasets:
            print(dataset.dataset_id)

    def get_db_size(self):
        return self.target_table.num_rows  # Кількість рядків (може бути кешованою)